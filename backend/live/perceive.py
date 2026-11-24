"""
Step 1 of measuring the live alerts: run the live perception (pose skeleton +
object detectors, exactly what the server runs) over recorded videos and
cache what it saw, frame by frame. This is the slow part (the models); the
alert rules are then evaluated on the cache in seconds (eval_alerts.py), as
often as you like, for any rule.

    python live/perceive.py --videos live/data/videos                     # every video in a folder
    python live/perceive.py --gt eval/ground_truth_uca.json --videos eval/data/ucf   # the videos of a ground truth
    python live/perceive.py ... --fps 3                                   # frames per second analysed (default 3)

Output: live/data/perception/<video>.jsonl
    line 1: {"video", "fps", "duration", "w", "h", "models"}
    then  : {"t": seconds, "persons": [{id, box, kpts}], "objects": [{cls, conf, box}]}
Objects are kept down to conf 0.15, so eval_alerts.py can try other thresholds.
A video already cached with the same fps is skipped (--force to redo).

--fps is the rate the live system would actually process: on a laptop CPU
about 1-3 frames per second, on a GPU 10+. The timing of the rules depends
on it, so measure at the rate you will run.
"""
import argparse
import glob
import json
import os
import sys
import time

import cv2

HERE = os.path.dirname(os.path.abspath(__file__))
BACKEND = os.path.dirname(HERE)
sys.path.insert(0, BACKEND)
CWD = os.getcwd()                       # paths on the command line are relative to here
os.chdir(BACKEND)                       # model files (security.pt) are found from here
import alert_rules                      # noqa: E402
from live_engine import LiveEngine      # noqa: E402

OUT = os.path.join(HERE, "data", "perception")
KEEP_CONF = 0.15
VIDEO_EXT = (".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v")
# classes worth keeping for security alerts (anything a rule may ask for)
KEEP = {"knife", "gun", "fire", "smoke", "bag", "cell phone", "bottle", "scissors", "baseball bat",
        "cup", "laptop", "umbrella", "book", "dog", "cat"}


def videos_from(args):
    paths = []
    if args.gt:
        names = set()
        for g in args.gt:
            with open(g, encoding="utf-8") as f:
                names |= set(json.load(f)["videos"])
        for n in sorted(names):
            p = os.path.join(args.videos, n)
            if os.path.isfile(p):
                paths.append(p)
        missing = len(names) - len(paths)
        if missing:
            print(f"{missing} videos of the ground truth are not in {args.videos} (skipped)")
    else:
        paths = sorted(p for p in glob.glob(os.path.join(args.videos, "*")) if p.lower().endswith(VIDEO_EXT))
    return paths


def perceive_video(engine, path, out_path, fps, progress=None):
    """progress(fraction) is called now and then (for the server's status)."""
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        print(f"  cannot open {path}")
        return False
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration = n / src_fps if n else 0.0
    engine.start_stream()                    # new track ids for every video
    tmp = out_path + ".part"
    t0, next_t, i, kept = time.time(), 0.0, 0, 0
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(json.dumps({"video": os.path.basename(path), "fps": fps, "duration": round(duration, 2),
                            "w": w, "h": h, "models": engine.info()["detectors"] + [engine.info()["pose"]]}) + "\n")
        while True:
            ok = cap.grab()
            if not ok:
                break
            t = i / src_fps
            i += 1
            if t + 1e-6 < next_t:
                continue
            ok, frame = cap.retrieve()
            if not ok:
                break
            next_t += 1.0 / fps
            with engine.lock:
                persons, objects = engine.perceive(frame, wanted=KEEP, conf=KEEP_CONF)
            f.write(json.dumps({"t": round(t, 3), "persons": persons, "objects": objects}) + "\n")
            kept += 1
            if progress and n and kept % 5 == 0:
                progress(min(0.99, i / n))
    cap.release()
    os.replace(tmp, out_path)
    if progress:
        progress(1.0)
    print(f"  {kept} frames in {time.time() - t0:.0f}s")
    return True


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--videos", required=True, help="folder of videos")
    ap.add_argument("--gt", nargs="*", help="only the videos of these ground-truth files")
    ap.add_argument("--fps", type=float, default=3.0)
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    here = lambda p: os.path.normpath(os.path.join(CWD, p))
    args.videos, args.out = here(args.videos), here(args.out)
    args.gt = [here(g) for g in args.gt or []]
    os.makedirs(args.out, exist_ok=True)
    paths = videos_from(args)
    print(f"{len(paths)} videos, {args.fps} fps -> {args.out}")
    engine = LiveEngine()
    for k, p in enumerate(paths, 1):
        out_path = os.path.join(args.out, os.path.basename(p) + ".jsonl")
        if os.path.isfile(out_path) and not args.force:
            with open(out_path, encoding="utf-8") as f:
                head = json.loads(f.readline() or "{}")
            if head.get("fps") == args.fps:
                print(f"[{k}/{len(paths)}] {os.path.basename(p)}: cached")
                continue
        print(f"[{k}/{len(paths)}] {os.path.basename(p)}")
        perceive_video(engine, p, out_path, args.fps)


if __name__ == "__main__":
    main()
