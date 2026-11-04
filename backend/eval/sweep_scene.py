"""
Offline sweep of the scene-search thresholds on an indexed benchmark DB.
No server, nothing re-indexed: the whole-frame CLIP vectors are in the DB.

    python eval/sweep_scene.py eval/benchmark.db eval/ground_truth_uca.json
    python eval/sweep_scene.py ... --prefix "security camera footage of "

Prints precision / recall / F1 overall and per scenario for a grid of
(min_score, z) — the MIN_SCENE / SCENE_Z knobs of search.scene_windows.
"""
import argparse
import json
import os
import sqlite3
import sys
from collections import defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import db                      # noqa: E402
import embeddings              # noqa: E402
import search                  # noqa: E402
from evaluate import prf, score  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("db")
    ap.add_argument("gt")
    ap.add_argument("--prefix", default="", help="text put before every query, e.g. 'security camera footage of '")
    ap.add_argument("--iou", type=float, default=0.3)
    ap.add_argument("--min", default="0,0.2,0.24,0.26,0.28,0.3")
    ap.add_argument("--z", default="1,1.5,2,2.5")
    args = ap.parse_args()

    conn = sqlite3.connect(args.db)
    gt = json.load(open(args.gt, encoding="utf-8"))
    indexed = {r[0] for r in conn.execute("SELECT video FROM videos")}
    texts, data, used = {}, [], set()
    for video, spec in gt["videos"].items():
        if video not in indexed:
            continue
        used.add(video)
        frames = [(t, embeddings.from_blob(b)) for t, b in db.load_frames(conn, video)]
        for q in spec["queries"]:
            key = args.prefix + q["text"]
            if key not in texts:
                texts[key] = embeddings.embed_text(key)
            data.append((q.get("scenario") or "all", frames, texts[key], q["expected"]))
    print(f"{len(data)} queries on {len(used)} indexed videos (of {len(gt['videos'])}), prefix={args.prefix!r}")
    scenarios = sorted({d[0] for d in data})
    print(f"{'min':>5} {'z':>4} | {'ALL P/R/F1':>16} | " + " ".join(f"{s:>12}" for s in scenarios))
    for mn in (float(x) for x in args.min.split(",")):
        for z in (float(x) for x in args.z.split(",")):
            agg = defaultdict(lambda: [0, 0, 0])
            for sc, frames, tv, exp in data:
                tp, fp, fn = score(search.scene_windows(frames, tv, min_score=mn, z=z), exp, args.iou, "hit")
                for k in (sc, "ALL"):
                    agg[k][0] += tp; agg[k][1] += fp; agg[k][2] += fn
            p, r, f = prf(*agg["ALL"])
            cells = " ".join(f"{prf(*agg[s])[2]:>5.2f}({agg[s][0]}/{agg[s][0]+agg[s][1]})" if s in agg else f"{'-':>12}" for s in scenarios)
            print(f"{mn:>5.2f} {z:>4.1f} | {p:.2f}/{r:.2f}/{f:.2f} ({agg['ALL'][0]}/{agg['ALL'][0]+agg['ALL'][1]}/{agg['ALL'][0]+agg['ALL'][2]}) | {cells}")
    print("cells: F1 (windows found/segments returned); ALL: P/R/F1 (found/returned/expected). hit scoring, see evaluate.score_hits")


if __name__ == "__main__":
    main()
