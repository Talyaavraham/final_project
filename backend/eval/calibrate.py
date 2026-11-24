"""
Pick the search thresholds for videos nobody has seen: one setting for every
benchmark at once, instead of one tuned per benchmark.

    python eval/calibrate.py eval/benchmark.db:eval/ground_truth_uca_part4.json \
        eval/benchmark_v2.db:eval/ground_truth_uca.json \
        --check eval/heldout.db:eval/ground_truth_heldout.json

Every (db, ground truth) pair goes through the same code as offline_eval.py
(parse_rules + search.run_query) for a grid of MIN_SCENE x SCENE_Z x
MIN_SEMANTIC_UNKNOWN. The chosen setting maximises the MEAN F1 over the
benchmarks, so a big benchmark does not decide alone. Two checks that it
carries over:
  * leave one benchmark out: tuned on the others, scored on it
  * --check: a set never used for tuning (the held-out videos), scored once
"""
import argparse
import hashlib
import itertools
import json
import os
import re
import sqlite3
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
import db                                   # noqa: E402
import embeddings                           # noqa: E402
import search                               # noqa: E402
from evaluate import prf, score             # noqa: E402
from query_parser import parse_rules        # noqa: E402

KNOBS = ("MIN_SCENE", "SCENE_Z", "MIN_SEMANTIC_UNKNOWN")


def load(pair):
    """'db:gt' -> (name, [(video meta, segments, frames, [(parsed query, expected), ...]), ...])"""
    parts = re.split(r":(?![\\/])", pair)          # not the colon of a drive letter (C:\...)
    if len(parts) != 2:
        sys.exit(f"expected DB:GT, got {pair!r}")
    path, gt_path = parts
    conn = sqlite3.connect(path)
    gt = json.load(open(gt_path, encoding="utf-8"))
    dim = embeddings.embed_texts(["x"]).shape[1]
    videos = []
    for video, spec in gt["videos"].items():
        meta = db.get_video(conn, video)
        if meta is None:
            continue
        meta["video"] = video
        segments = db.load_segments(conn, video)
        for s in segments:
            s["embedding_vec"] = embeddings.from_blob(s.pop("embedding"))
            if s["embedding_vec"] is not None:
                s["embedding_vec"] = s["embedding_vec"].reshape(-1, dim)
        frames = [(t, embeddings.from_blob(b)) for t, b in db.load_frames(conn, video)]
        classes = sorted({s["cls"] for s in segments})
        queries = [(parse_rules(q["text"], classes or None), q["expected"]) for q in spec["queries"]]
        videos.append((meta, segments, frames, queries))
    name = os.path.splitext(os.path.basename(gt_path))[0].replace("ground_truth_", "")
    n_q = sum(len(v[3]) for v in videos)
    print(f"{name}: {len(videos)} indexed videos of {len(gt['videos'])}, {n_q} queries  ({path})", flush=True)
    return name, videos


def half(video):
    return int(hashlib.md5(video.encode()).hexdigest(), 16) % 2


def evaluate(videos, setting, only_half=None):
    for k, v in setting.items():
        setattr(search, k, v)
    tp = fp = fn = 0
    for meta, segments, frames, queries in videos:
        if only_half is not None and half(meta["video"]) != only_half:
            continue
        for p, expected in queries:
            res, _ = search.run_query(segments, frames, p["objects"], p["colors"], p["actions"],
                                      p["timeRange"], p["semantic"], p["unknown"],
                                      embed=embeddings.embed_texts, W=meta["width"] or 16, H=meta["height"] or 9)
            a, b, c = score(res, expected, 0.3, "hit")
            tp, fp, fn = tp + a, fp + b, fn + c
    return tp, fp, fn


def fmt(c):
    p, r, f = prf(*c)
    return f"P={p:.2f} R={r:.2f} F1={f:.3f} ({c[0]}/{c[0] + c[2]} windows, {c[0] + c[1]} segments)"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("sets", nargs="+", metavar="DB:GT", help="benchmarks to tune on")
    ap.add_argument("--check", action="append", default=[], metavar="DB:GT", help="scored once, never tuned on")
    ap.add_argument("--min-scene", default="0.1,0.2,0.3,0.4,0.5,0.6")
    ap.add_argument("--scene-z", default="1.0,1.5,2.0,2.5,3.0")
    ap.add_argument("--min-unknown", default="0.25,0.27,0.29")
    ap.add_argument("--adapter", default="0", help="text adapter .npz, 0 = plain CLIP (default)")
    ap.add_argument("--out", help="write the full grid as json")
    args = ap.parse_args()
    embeddings.set_adapter(args.adapter)
    print("text adapter:", embeddings.ADAPTER_PATH if embeddings.adapter() is not None else "none (plain CLIP)")

    default = {k: getattr(search, k) for k in KNOBS}
    sets = dict(load(p) for p in args.sets)
    checks = dict(load(p) for p in args.check)
    grid = [dict(zip(KNOBS, v)) for v in itertools.product(
        *[[float(x) for x in s.split(",")] for s in (args.min_scene, args.scene_z, args.min_unknown)])]
    if default not in grid:
        grid.append(default)

    t0 = time.time()
    table = []
    for i, setting in enumerate(grid):
        counts = {}
        for n, v in sets.items():
            h = [evaluate(v, setting, 0), evaluate(v, setting, 1)]
            counts[n] = tuple(a + b for a, b in zip(*h))
            counts[n, 0], counts[n, 1] = h
        table.append((setting, counts))
        if i == 0:
            print(f"  one setting takes {time.time() - t0:.0f}s, {len(grid)} settings", flush=True)

    def mean_f1(counts, names):
        return sum(prf(*counts[n])[2] for n in names) / len(names)

    names = list(sets)
    table.sort(key=lambda row: -mean_f1(row[1], names))
    print(f"\ntop settings by mean F1 over {', '.join(names)}:")
    for setting, counts in table[:10]:
        print(f"  {setting}  mean F1 {mean_f1(counts, names):.3f}  "
              + "  ".join(f"{n} {prf(*counts[n])[2]:.3f}" for n in names))
    best, best_counts = table[0]
    def_counts = next(c for s, c in table if s == default)

    print(f"\ncurrent defaults {default}:")
    for n in names:
        print(f"  {n:12s} {fmt(def_counts[n])}")
    print(f"chosen {best}:")
    for n in names:
        print(f"  {n:12s} {fmt(best_counts[n])}")

    if len(names) > 1:
        print("\nleave one benchmark out (tuned on the others, scored on it):")
        for out in names:
            rest = [n for n in names if n != out]
            s, c = max(table, key=lambda row: mean_f1(row[1], rest))
            print(f"  {out:12s} tuned {s} -> F1 {prf(*c[out])[2]:.3f}  "
                  f"(defaults {prf(*def_counts[out])[2]:.3f}, best possible {max(prf(*c2[out])[2] for _, c2 in table):.3f})")

    print("\ntwo halves of the videos (tuned on one half, scored on the other):")
    for n in names:
        for h in (0, 1):
            s, c = max(table, key=lambda row: mean_f1({m: row[1][m, h] for m in names}, names))
            o = 1 - h
            print(f"  {n:12s} tuned on half {h} {s} -> half {o} F1 {prf(*c[n, o])[2]:.3f}  "
                  f"(defaults {prf(*def_counts[n, o])[2]:.3f})")

    for n, v in checks.items():
        print(f"\ncheck {n} (never tuned on):")
        print(f"  defaults {fmt(evaluate(v, default))}")
        print(f"  chosen   {fmt(evaluate(v, best))}")

    print("\nto use it: environment variables " + " ".join(f"{k}={v}" for k, v in best.items())
          + "  (or change the defaults in search.py)")
    if args.out:
        json.dump([{"setting": s, **{n: list(c[n]) for n in names}} for s, c in table], open(args.out, "w"), indent=1)


if __name__ == "__main__":
    main()
