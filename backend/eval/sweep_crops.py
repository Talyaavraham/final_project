"""
Offline sweep of the crop-similarity threshold (MIN_SEMANTIC) for queries
that name an object class plus free text, e.g. "a person holding a knife"
-> person tracks ranked by CLIP similarity of their best crop. Runs on an
indexed benchmark DB, no server.

    python eval/sweep_crops.py eval/benchmark.db eval/ground_truth_uca.json

Prints precision / recall / F1 (hit scoring) overall and per scenario for a
grid of thresholds, so the floor can be set where "every person" stops
counting as "a person with a knife".
"""
import argparse
import json
import os
import sqlite3
import sys
from collections import defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import db                                   # noqa: E402
import embeddings                           # noqa: E402
from evaluate import prf, score             # noqa: E402
from query_parser import parse_rules        # noqa: E402
from search import filter_segments, semantic_rank  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("db")
    ap.add_argument("gt")
    ap.add_argument("--min", default="0.20,0.22,0.24,0.26,0.28,0.30,0.32")
    ap.add_argument("--models", default="", help="only videos indexed with these models (comma list)")
    args = ap.parse_args()

    conn = sqlite3.connect(args.db)
    gt = json.load(open(args.gt, encoding="utf-8"))
    meta = {r[0]: r for r in conn.execute("SELECT video, model, width, height FROM videos")}
    models = {m for m in args.models.split(",") if m}
    texts, data, used = {}, [], set()
    for video, spec in gt["videos"].items():
        if video not in meta or (models and meta[video][1] not in models):
            continue
        used.add(video)
        _, _, W, H = meta[video]
        classes = list(db.list_classes(conn, video))
        segs = db.load_segments(conn, video)
        for s in segs:
            s["embedding_vec"] = embeddings.from_blob(s.pop("embedding"))
        for q in spec["queries"]:
            p = parse_rules(q["text"], classes)
            if not p["objects"] or not p.get("semantic"):
                continue                         # scene queries: see sweep_scene.py
            key = p["semantic"]
            if key not in texts:
                texts[key] = embeddings.embed_text(key)
            cand = filter_segments(segs, p["objects"], p["colors"], p["actions"], None, True, W or 16, H or 9)
            data.append((q.get("scenario") or "all", cand, texts[key], q["expected"]))
    print(f"{len(data)} object+text queries on {len(used)} videos (of {len(gt['videos'])})")
    scenarios = sorted({d[0] for d in data})
    print(f"{'min':>5} | {'ALL P/R/F1':>26} | " + " ".join(f"{s:>14}" for s in scenarios))
    for mn in (float(x) for x in args.min.split(",")):
        agg = defaultdict(lambda: [0, 0, 0])
        for sc, cand, tv, exp in data:
            res = semantic_rank(cand, tv, min_score=mn)
            tp, fp, fn = score(res, exp, 0.3, "hit")
            for k in (sc, "ALL"):
                agg[k][0] += tp; agg[k][1] += fp; agg[k][2] += fn
        p, r, f = prf(*agg["ALL"])
        cells = " ".join(f"{prf(*agg[s])[2]:>5.2f}({agg[s][0]}/{agg[s][0]+agg[s][1]})" for s in scenarios)
        print(f"{mn:>5.2f} | {p:.2f}/{r:.2f}/{f:.2f} ({agg['ALL'][0]}/{agg['ALL'][0]+agg['ALL'][1]}/{agg['ALL'][0]+agg['ALL'][2]}) | {cells}")
    print("cells: F1 (windows found/segments returned); hit scoring, see evaluate.score_hits")


if __name__ == "__main__":
    main()
