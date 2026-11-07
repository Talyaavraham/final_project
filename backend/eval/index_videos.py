"""
Index a folder of videos through the running backend (POST /index), waiting
for each to finish, then optionally run evaluate.py on a ground truth.

    python eval/index_videos.py eval/data/ucf --api http://localhost:8000
    python eval/index_videos.py eval/data/ucf --gt eval/ground_truth_uca.json

Already-indexed videos (same content hash) are skipped by the server.
"""
import argparse
import glob
import os
import subprocess
import sys
import time

import httpx


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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("folder")
    ap.add_argument("--api", default="http://localhost:8000")
    ap.add_argument("--force", action="store_true", help="re-index even if cached")
    ap.add_argument("--gt", help="ground truth json -> run evaluate.py afterwards")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--videos", help="comma-separated file names: only these (e.g. the ones that failed)")
    args = ap.parse_args()
    files = sorted(glob.glob(os.path.join(args.folder, "*.mp4")))
    if args.videos:
        wanted = {v.strip() for v in args.videos.split(",")}
        files = [f for f in files if os.path.basename(f) in wanted]
    if args.limit:
        files = files[:args.limit]
    print(f"{len(files)} videos -> {args.api}", flush=True)
    t_all = time.time()
    for i, p in enumerate(files, 1):
        try:
            name, model, secs = index_one(args.api, p, args.force)
            print(f"[{i}/{len(files)}] {name}: {model} ({secs:.0f}s)", flush=True)
        except Exception as e:  # noqa: BLE001
            print(f"[{i}/{len(files)}] {os.path.basename(p)}: FAILED {e}", flush=True)
    print(f"indexing done in {(time.time()-t_all)/60:.1f} min", flush=True)
    if args.gt:
        here = os.path.dirname(os.path.abspath(__file__))
        subprocess.call([sys.executable, os.path.join(here, "..", "evaluate.py"), args.gt, "--api", args.api, "--match", "hit"])


if __name__ == "__main__":
    main()
