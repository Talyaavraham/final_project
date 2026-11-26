"""
Does zooming in on the people help a detector see a knife / a gun?

On UCF-Crime (320x240) security.pt saw a knife in 0 of 46 labelled events
(live/results/ucf_2026-11-24). A knife there is a few pixels, below what a
YOLO head resolves on the whole frame. The alert rules only care about an
object in somebody's hand, so the detector can look where the people are:
every person box is cut out, enlarged, and the detector runs on that crop.

This script measures that idea before anything is built into the live
engine. For frames sampled inside the labelled knife / weapon events and for
frames of videos labelled as having none, it runs every detector twice:

    full    the whole frame (what the live engine does today)
    zoom    the crops of the (up to 3) biggest people, enlarged

and reports, per detector and variant: in how many events the object was seen
at confidence 0.15 / 0.25 / 0.5, and in what share of the no-event frames
it was "seen" anyway (false alarms).

    python eval/zoom_bench.py eval/ground_truth_uca.json eval/data/ucf          # run (resumes) + report
    python eval/zoom_bench.py eval/ground_truth_uca.json eval/data/ucf --report # report only

Raw confidences: eval/results/zoom_raw.json (--raw); the run of 2026-11-25 is
kept as eval/results/zoom_raw_2026-11-25.json, its report as zoom_2026-11-25.txt.

Result (2026-11-25): it does not help. Knife: at most 3 of 46 events, for any
of four detectors, full or zoom. Gun: security.pt sees 5 of 81 events on the
whole frame and 2 on the crops, with more false alarms. At 320x240 the object
is not there to be seen; the idea is not built into the live engine.
"""
import argparse
import glob
import json
import os
import sys
import time

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
BACKEND = os.path.dirname(HERE)
SCENARIO_CLASS = {"knife": "knife", "weapon": "gun"}      # ground-truth scenario -> detector class
LEVELS = (0.15, 0.25, 0.5)
MIN_CONF = 0.1          # kept in the raw file
MAX_PEOPLE = 3
PAD = 0.25              # the person box grows by this share on every side (a raised arm leaves the box)
SLACK = 3.0             # s around a window that count as neither event nor "outside"


def windows(spec, scenario):
    """-> list of [s, e] for the scenario, or None when the video has no query for it."""
    qs = [q for q in spec["queries"] if q.get("scenario") == scenario]
    return [w for q in qs for w in q["expected"]] if qs else None


def plan(spec, duration, step, per_event, per_video):
    """
    The frames to look at in one video: [(t, {scenario: label})], label =
    ["event", i] inside the i-th window, "near" outside the windows of a video
    that has some, "clean" in a video labelled as having none, None = no label.
    """
    wins = {sc: windows(spec, sc) for sc in SCENARIO_CLASS}
    if all(w is None for w in wins.values()):
        return []
    times = set()
    for w in wins.values():
        for s, e in w or []:
            n = min(per_event, max(1, int((e - s) / step) + 1))
            times |= {round(float(t), 2) for t in np.linspace(s, min(e, duration - 0.2), n)}
    times |= {round(float(t), 2) for t in np.linspace(1.0, max(1.0, duration - 1.0), per_video)}

    def label(t, w):
        if w is None:
            return None
        for i, (s, e) in enumerate(w):
            if s <= t <= e:
                return ["event", i]
        if any(s - SLACK <= t <= e + SLACK for s, e in w):
            return "slack"
        return "near" if w else "clean"

    return [(t, {sc: label(t, w) for sc, w in wins.items()}) for t in sorted(times) if 0 <= t < duration]


def person_crops(frame, boxes):
    h, w = frame.shape[:2]
    out = []
    for x1, y1, x2, y2 in sorted(boxes, key=lambda b: -(b[2] - b[0]) * (b[3] - b[1]))[:MAX_PEOPLE]:
        px, py = PAD * (x2 - x1), PAD * (y2 - y1)
        a, b = int(max(0, x1 - px)), int(max(0, y1 - py))
        c, d = int(min(w, x2 + px)), int(min(h, y2 + py))
        if c - a >= 16 and d - b >= 16:
            out.append(frame[b:d, a:c])
    return out


def best(results, names, keep):
    """best confidence per class name over a list of ultralytics results."""
    out = {}
    for r in results:
        for c, sc in zip(r.boxes.cls.tolist(), r.boxes.conf.tolist()):
            n = keep.get(str(names[int(c)]).lower())
            if n:
                out[n] = max(out.get(n, 0.0), round(sc, 3))
    return out


def load_models(spec):
    from ultralytics import YOLO
    models = {}
    for item in spec.split(","):
        name, _, path = item.partition("=")
        path = path or name
        try:
            m = YOLO(os.path.join(BACKEND, path) if os.path.isfile(os.path.join(BACKEND, path)) else path)
            if "world" in path:
                m.set_classes(["knife", "gun"])
            models[os.path.splitext(name)[0]] = m
        except Exception as e:                                   # a model that cannot load is left out, loudly
            print(f"!! {path} skipped: {type(e).__name__}: {e}", flush=True)
    return models


KEEP = {"knife": "knife", "gun": "gun", "pistol": "gun", "handgun": "gun"}


def run(args, gt, raw):
    from ultralytics import YOLO
    people = YOLO(os.path.join(BACKEND, "yolov8n.pt"))
    models = load_models(args.models)
    paths = {os.path.basename(f): f for f in glob.glob(os.path.join(args.folder, "**", "*.mp4"), recursive=True)}
    todo = [v for v in gt["videos"] if v not in raw and v in paths]
    print(f"{len(todo)} videos to look at with {list(models)} ({len(raw)} done)", flush=True)
    t0 = time.time()
    for k, v in enumerate(todo, 1):
        cap = cv2.VideoCapture(paths[v])
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        duration = (cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0) / fps
        frames = []
        for t, labels in plan(gt["videos"][v], duration, args.step, args.per_event, args.per_video):
            cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
            ok, frame = cap.read()
            if not ok:
                continue
            p = people.predict(frame, classes=[0], conf=0.25, imgsz=640, verbose=False)[0]
            crops = person_crops(frame, p.boxes.xyxy.tolist())
            det = {}
            for name, m in models.items():
                full = best(m.predict(frame, conf=MIN_CONF, imgsz=640, verbose=False), m.names, KEEP)
                zoom = best(m.predict(crops, conf=MIN_CONF, imgsz=args.zoom_imgsz, verbose=False), m.names, KEEP) if crops else {}
                det[name] = {"full": full, "zoom": zoom}
            frames.append({"t": t, "labels": labels, "people": len(crops), "det": det})
        cap.release()
        raw[v] = frames
        json.dump(raw, open(args.raw, "w"))
        print(f"[{k}/{len(todo)}] {v}: {len(frames)} frames ({(time.time() - t0) / 60:.1f} min)", flush=True)


def report(raw):
    models = sorted({m for fr in raw.values() for f in fr for m in f["det"]})
    variants = {"full": lambda d: d["full"], "zoom": lambda d: d["zoom"],
                "full+zoom": lambda d: {c: max(d["full"].get(c, 0), d["zoom"].get(c, 0)) for c in ("knife", "gun")}}
    lines = []
    for sc, cls in SCENARIO_CLASS.items():
        events, clean, near, with_people = {}, [], [], 0
        for v, fr in raw.items():
            for f in fr:
                lab = f["labels"].get(sc)
                if isinstance(lab, list):
                    events.setdefault((v, lab[1]), []).append(f)
                elif lab == "clean":
                    clean.append(f)
                elif lab == "near":
                    near.append(f)
        with_people = sum(1 for fs in events.values() if any(f["people"] for f in fs))
        lines.append(f"\n== {sc} (class '{cls}'): {len(events)} events ({with_people} with a person found), "
                     f"{len(clean)} frames of videos labelled without it, {len(near)} frames outside the windows ==")
        lines.append(f"{'detector':16s} {'variant':10s} " + " ".join(f"{'events@' + str(l):>12s}" for l in LEVELS)
                     + "   " + " ".join(f"{'clean@' + str(l):>11s}" for l in LEVELS)
                     + "   " + " ".join(f"{'outside@' + str(l):>13s}" for l in LEVELS))
        for m in models:
            for vn, get in variants.items():
                conf = lambda f: get(f["det"][m]).get(cls, 0.0) if m in f["det"] else 0.0
                ev = [max(conf(f) for f in fs) for fs in events.values()]
                row = f"{m:16s} {vn:10s} "
                row += " ".join(f"{sum(c >= l for c in ev):5d} ({100 * sum(c >= l for c in ev) / max(1, len(ev)):3.0f}%)" for l in LEVELS)
                row += "   " + " ".join(f"{100 * sum(conf(f) >= l for f in clean) / max(1, len(clean)):10.1f}%" for l in LEVELS)
                row += "   " + " ".join(f"{100 * sum(conf(f) >= l for f in near) / max(1, len(near)):12.1f}%" for l in LEVELS)
                lines.append(row)
    lines.append("\nevents@X: labelled events in which the class was detected at confidence >= X in at least one sampled frame."
                 "\nclean@X: share of the frames of videos labelled as NOT containing it where it was 'detected' anyway (false alarms)."
                 "\noutside@X: the same share for frames outside the windows of videos that do contain it (the object may really be visible there).")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("gt")
    ap.add_argument("folder")
    ap.add_argument("--models", default="security.pt,security_v2.pt,coco_s=yolov8s.pt,world=yolov8s-worldv2.pt",
                    help="name=file, comma separated")
    ap.add_argument("--raw", default=os.path.join(HERE, "results", "zoom_raw.json"))
    ap.add_argument("--step", type=float, default=1.5, help="s between the frames sampled inside an event")
    ap.add_argument("--per-event", type=int, default=8)
    ap.add_argument("--per-video", type=int, default=8, help="frames spread over the whole video (no-event frames)")
    ap.add_argument("--zoom-imgsz", type=int, default=480)
    ap.add_argument("--report", action="store_true", help="only print the report of the raw file")
    args = ap.parse_args()
    gt = json.load(open(args.gt, encoding="utf-8"))
    raw = json.load(open(args.raw)) if os.path.exists(args.raw) else {}
    if not args.report:
        run(args, gt, raw)
    print(report(raw))


if __name__ == "__main__":
    sys.exit(main())
