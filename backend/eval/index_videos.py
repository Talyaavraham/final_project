"""
Index a folder of videos (sub-folders included), then optionally score a
ground truth.

Through the running backend (POST /index, then evaluate.py):

    python eval/index_videos.py eval/data/ucf --api http://localhost:8000
    python eval/index_videos.py eval/data/ucf --gt eval/ground_truth_uca.json

Without a server, straight into an index file (then offline_eval.py). The
models come from the environment, as for the backend (YOLO_MODEL,
EXTRA_MODELS, IMGSZ):

    python eval/index_videos.py C:/path/Anomaly-Videos-Part-1 --db eval/heldout.db \
        --gt eval/ground_truth_heldout.json --only-gt

Already-indexed videos are skipped (server: same content hash; --db: same name).
"""
import argparse
import glob
import json
import os
import subprocess
import sys
import time

import httpx

HERE = os.path.dirname(os.path.abspath(__file__))


def index_one(api, path, force=False, timeout=3600):
    name = os.path.basename(path)
    with open(path, "rb") as f:
        r = httpx.post(f"{api}/index", files={"file": (name, f, "video/mp4")},
                       params={"force": "true"} if force else None, timeout=600)
    r.raise_for_status()
    info = r.json()
    if info.get("cached"):
        return info["video"], "cached", 0.0
    t0 = time.time()
    while time.time() - t0 < timeout:
        s = httpx.get(f"{api}/status", params={"video": info["video"]}, timeout=30).json()
        if s.get("status") == "ready":
            return info["video"], s.get("model", "?"), time.time() - t0
        if str(s.get("status", "")).startswith("error"):
            return info["video"], "ERROR " + s["status"], time.time() - t0
        time.sleep(2)
    return info["video"], "TIMEOUT", time.time() - t0


def direct_indexer(db_path):
    """-> index_one(path, force) that runs the indexer in this process, on `db_path`."""
    sys.path.insert(0, os.path.join(HERE, ".."))
    import db
    import indexer
    conn = db.connect(db_path)

    def index_one_direct(path, force=False):
        name = os.path.basename(path)
        if not force and db.get_video(conn, name):
            return name, "cached", 0.0
        t0 = time.time()
        s = indexer.index_video(path, name, db_path=db_path)
        ok = s.get("status") == "ready"
        return name, f"{s.get('model', '?')}, {s.get('segments')} segments" if ok else "ERROR " + str(s.get("status")), time.time() - t0
    return index_one_direct


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("folder")
    ap.add_argument("--api", default="http://localhost:8000")
    ap.add_argument("--db", help="index into this file in-process instead of through a server")
    ap.add_argument("--force", action="store_true", help="re-index even if cached")
    ap.add_argument("--gt", help="ground truth json -> score it afterwards (evaluate.py, or offline_eval.py with --db)")
    ap.add_argument("--only-gt", action="store_true", help="index only the videos named in --gt")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--videos", help="comma-separated file names: only these (e.g. the ones that failed)")
    args = ap.parse_args()
    files = sorted(glob.glob(os.path.join(args.folder, "**", "*.mp4"), recursive=True))
    if args.videos:
        wanted = {v.strip() for v in args.videos.split(",")}
        files = [f for f in files if os.path.basename(f) in wanted]
    if args.only_gt:
        wanted = set(json.load(open(args.gt, encoding="utf-8"))["videos"])
        files = [f for f in files if os.path.basename(f) in wanted]
    if args.limit:
        files = files[:args.limit]
    print(f"{len(files)} videos -> {args.db or args.api}", flush=True)
    direct = direct_indexer(args.db) if args.db else None
    t_all = time.time()
    for i, p in enumerate(files, 1):
        try:
            name, model, secs = direct(p, args.force) if direct else index_one(args.api, p, args.force)
            print(f"[{i}/{len(files)}] {name}: {model} ({secs:.0f}s)", flush=True)
        except Exception as e:  # noqa: BLE001
            print(f"[{i}/{len(files)}] {os.path.basename(p)}: FAILED {e}", flush=True)
    print(f"indexing done in {(time.time()-t_all)/60:.1f} min", flush=True)
    if args.gt and args.db:
        subprocess.call([sys.executable, os.path.join(HERE, "offline_eval.py"), args.db, args.gt])
    elif args.gt:
        subprocess.call([sys.executable, os.path.join(HERE, "..", "evaluate.py"), args.gt, "--api", args.api, "--match", "hit"])


if __name__ == "__main__":
    main()
