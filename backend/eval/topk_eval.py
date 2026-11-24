"""
Ranking instead of alarms: is the event among the first k moments the search
would show? No threshold involved, so it measures how well CLIP's scores
rank the seconds of a video, not where the cut-off sits.

    python eval/topk_eval.py eval/benchmark.db eval/ground_truth_uca_part4.json
    python eval/topk_eval.py ... --adapter clip_adapter.npz

For every query with an expected window: score every second the way
search.scene_windows does (match_prob against the "nothing special" prompts),
take the peaks in order, at least --spacing seconds apart (so the top 5 are
5 different moments, not 5 neighbouring seconds of one), and check whether a
peak falls inside a window (+-1 s). Chance level: the same number of peaks
picked at random seconds, same spacing, averaged over many draws. A window
that covers half the video is easy to hit by chance, so compare to it.
"""
import argparse
import json
import os
import random
import sqlite3
import sys
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import db                                   # noqa: E402
import embeddings                           # noqa: E402
import search                               # noqa: E402

KS = (1, 3, 5, 10)


def peaks(ts, order, spacing, k):
    """the first k seconds of `order` that are >= spacing apart from those already taken"""
    out = []
    for i in order:
        if all(abs(ts[i] - ts[j]) >= spacing for j in out):
            out.append(i)
            if len(out) == k:
                break
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("db")
    ap.add_argument("gt")
    ap.add_argument("--spacing", type=float, default=5.0, help="s between two moments shown")
    ap.add_argument("--tol", type=float, default=1.0, help="s around a window that still counts")
    ap.add_argument("--draws", type=int, default=200, help="random draws for the chance level")
    ap.add_argument("--adapter", default="0", help="text adapter .npz, 0 = plain CLIP (default)")
    ap.add_argument("--mode", choices=("scene", "crop"), default="scene",
                    help="scene: whole-frame vectors (search.scene_windows); crop: the CLIP vectors of the "
                         "tracked objects, a second scored by the best track on screen (search.crop_score)")
    ap.add_argument("--split", choices=("Train", "Val", "Test"), help="only the videos of this UCA split")
    args = ap.parse_args()
    embeddings.set_adapter(args.adapter)
    print("text adapter:", embeddings.ADAPTER_PATH if embeddings.adapter() is not None else "none (plain CLIP)")
    only = None
    if args.split:
        p = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "uca_src", f"UCFCrime_{args.split}.json")
        only = {n + ".mp4" for n in json.load(open(p, encoding="utf-8"))}

    conn = sqlite3.connect(args.db)
    gt = json.load(open(args.gt, encoding="utf-8"))
    rnd = random.Random(0)
    kmax = max(KS)
    hits = defaultdict(lambda: np.zeros(len(KS)))
    chance = defaultdict(lambda: np.zeros(len(KS)))
    n_q, first_rank, cover = defaultdict(int), defaultdict(list), defaultdict(list)
    vec_cache = {}
    for video, spec in gt["videos"].items():
        if only is not None and video not in only:
            continue
        frames = db.load_frames(conn, video)
        qs = [q for q in spec["queries"] if q["expected"]]
        if not frames or not qs:
            continue
        t_all = np.array([t for t, _ in frames], dtype=np.float32)
        v_all = np.stack([embeddings.from_blob(b) for _, b in frames])
        ts, inv = np.unique(t_all, return_inverse=True)
        if args.mode == "crop":
            dim = v_all.shape[1]
            segs = [(s["t"], s["t"] + s["dur"], embeddings.from_blob(s["embedding"]).reshape(-1, dim))
                    for s in db.load_segments(conn, video) if s.get("embedding")]
        for q in qs:
            if q["text"] not in vec_cache:
                pos, neg = search.scene_prompts(q["text"])
                vec_cache[q["text"]] = (embeddings.embed_texts(pos), embeddings.embed_texts(neg))
            if args.mode == "crop":
                tv = embeddings.embed_texts([q["text"]])[0]
                sims = np.full(len(ts), -1.0, dtype=np.float32)
                for a, b, vec in segs:
                    on = (ts >= a) & (ts <= b)
                    sims[on] = np.maximum(sims[on], search.crop_score(vec, tv))
            else:
                probs = search.match_prob(v_all, *vec_cache[q["text"]])
                sims = np.zeros(len(ts), dtype=np.float32)
                np.maximum.at(sims, inv, probs)
            inside = np.zeros(len(ts), dtype=bool)
            for lo, hi in q["expected"]:
                inside |= (ts >= lo - args.tol) & (ts <= hi + args.tol)
            sc = q.get("scenario") or "all"
            top = peaks(ts, np.argsort(-sims), args.spacing, kmax)
            got = [inside[i] for i in top]
            rank = next((r + 1 for r, g in enumerate(got) if g), None)
            ch = np.zeros(len(KS))
            for _ in range(args.draws):
                order = list(range(len(ts)))
                rnd.shuffle(order)
                g = [inside[i] for i in peaks(ts, order, args.spacing, kmax)]
                ch += [any(g[:k]) for k in KS]
            for key in (sc, "OVERALL"):
                n_q[key] += 1
                hits[key] += [any(got[:k]) for k in KS]
                chance[key] += ch / args.draws
                first_rank[key].append(rank)
                cover[key].append(inside.mean())

    print(f"mode {args.mode}: moments >= {args.spacing:g} s apart, a hit = inside an expected window +-{args.tol:g} s; "
          f"chance = random moments, same spacing ({args.draws} draws)")
    head = "  ".join(f"top{k:<2} (chance)" for k in KS)
    print(f"{'':10s} {'queries':>7}  {head}  window share of video")
    for key in sorted(n_q, key=lambda k: (k == "OVERALL", k)):
        n = n_q[key]
        cells = "  ".join(f"{hits[key][i] / n:.2f}  ({chance[key][i] / n:.2f})" for i in range(len(KS)))
        print(f"{key:10s} {n:7d}  {cells}  {np.mean(cover[key]):.2f}")
    ranks = [r for r in first_rank["OVERALL"] if r]
    print(f"first correct moment: median rank {np.median(ranks):.0f} when within the top {kmax} "
          f"({len(ranks)} of {n_q['OVERALL']} queries)")


if __name__ == "__main__":
    main()
