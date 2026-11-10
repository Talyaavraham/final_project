"""
How much does a detector alone find of the benchmark events? Runs a YOLO
model (default security.pt) on one frame per second of every ground-truth
video, keeps the best confidence per class per second, and scores
"seconds above a threshold, grouped into windows" against the expected
windows with hit scoring — for a grid of thresholds, per scenario.

    python eval/detector_bench.py eval/ground_truth_uca.json eval/data/ucf
    python eval/detector_bench.py ... --classes gun=weapon,knife=knife,fire=fire,smoke=fire   # smoke counts as fire

The per-second confidences are saved next to the ground truth
(<gt>_<model>_raw.json), so a new threshold grid needs no re-run (--raw).
"""
import argparse
import json
import os
import sys
import time
from collections import defaultdict

import cv2

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from evaluate import prf, score  # noqa: E402


def scan(model, path, imgsz, every=1.0):
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    step = max(1, int(round(fps * every)))
    out, i = {}, 0
    while True:
        ok = cap.grab()
        if not ok:
            break
        if i % step == 0:
            ok, frame = cap.retrieve()
            if ok:
                res = model.predict(frame, conf=0.1, imgsz=imgsz, verbose=False)[0]
                best = {}
                for b in res.boxes or []:
                    name = str(model.names[int(b.cls[0])]).lower()
                    best[name] = max(best.get(name, 0.0), float(b.conf[0]))
                if best:
                    out[round(i / fps, 1)] = best
        i += 1
    cap.release()
    return out


def windows(times, gap=2.0):
    out = []
    for t in sorted(times):
        if out and t <= out[-1][1] + gap:
            out[-1][1] = t
        else:
            out.append([t, t])
    return [{"t": a, "dur": max(1.0, b - a + 1.0)} for a, b in out]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("gt")
    ap.add_argument("folder")
    ap.add_argument("--model", default="security.pt")
    ap.add_argument("--imgsz", type=int, default=800)
    ap.add_argument("--classes", default="gun=weapon,knife=knife,fire=fire",
                    help="detector class=scenario")
    ap.add_argument("--thr", default="0.2,0.3,0.4,0.5,0.6")
    ap.add_argument("--raw", help="reuse saved per-second confidences instead of running the model")
    args = ap.parse_args()
    cls2sc = dict(kv.split("=") for kv in args.classes.split(","))
    gt = json.load(open(args.gt, encoding="utf-8"))
    raw_path = args.raw or os.path.splitext(args.gt)[0] + f"_{os.path.splitext(os.path.basename(args.model))[0]}_raw.json"
    raw = json.load(open(raw_path)) if os.path.exists(raw_path) else {}
    if not args.raw:
        from ultralytics import YOLO
        model = YOLO(args.model)
        todo = [v for v in gt["videos"] if v not in raw and os.path.exists(os.path.join(args.folder, v))]
        print(f"{len(todo)} videos to scan with {args.model} ({len(raw)} already in {raw_path})", flush=True)
        t0 = time.time()
        for n, v in enumerate(todo, 1):
            t1 = time.time()
            raw[v] = {str(k): b for k, b in scan(model, os.path.join(args.folder, v), args.imgsz).items()}
            json.dump(raw, open(raw_path, "w"))
            print(f"[{n}/{len(todo)}] {v}: {len(raw[v])} s with detections ({time.time()-t1:.0f}s, {(time.time()-t0)/60:.1f} min)", flush=True)
    scenarios = sorted(set(cls2sc.values()))
    print(f"{'thr':>4} | " + " ".join(f"{s:>22}" for s in scenarios))
    for thr in (float(x) for x in args.thr.split(",")):
        agg = defaultdict(lambda: [0, 0, 0])
        for video, spec in gt["videos"].items():
            secs = raw.get(video)
            if secs is None:
                continue
            for q in spec["queries"]:
                sc = q.get("scenario")
                if sc not in scenarios:
                    continue
                hits = [float(t) for t, best in secs.items()
                        if any(cls2sc.get(c) == sc and conf >= thr for c, conf in best.items())]
                tp, fp, fn = score(windows(hits), q["expected"], 0.3, "hit")
                agg[sc][0] += tp; agg[sc][1] += fp; agg[sc][2] += fn
        cells = []
        for s in scenarios:
            p, r, f = prf(*agg[s])
            cells.append(f"P{p:.2f} R{r:.2f} F1 {f:.2f} ({agg[s][0]}/{agg[s][0]+agg[s][1]}/{agg[s][0]+agg[s][2]})")
        print(f"{thr:>4.1f} | " + " ".join(f"{c:>22}" for c in cells))
    print("cells: precision / recall / F1 (windows found / windows returned / windows expected), hit scoring")


if __name__ == "__main__":
    main()
