"""
The benchmark without a server: every ground-truth query goes through the
rule parser and search.run_query on an indexed DB — the same code /search
runs — and is scored like evaluate.py. Seconds instead of minutes, so a
change to the search logic can be measured right away.

    python eval/offline_eval.py eval/benchmark.db eval/ground_truth_uca.json
    python eval/offline_eval.py ... --extra-raw eval/ground_truth_uca_security_raw.json
    python eval/offline_eval.py ... --set MIN_SCENE=0.3 --set SCENE_Z=2
    python eval/offline_eval.py ... --track-views 1      # ablation: best crop only

--extra-raw adds the tracks an extra detector WOULD have put in the index,
from the per-second confidences saved by eval/detector_bench.py (1 fps, so
slightly fewer than a real 3 fps index): the way to measure the detector and
CLIP together without re-indexing for hours. Those tracks have no boxes, so
"a person with a gun" cannot check that the gun is near the person.

Prints P / R / F1 overall, per scenario, and for the two halves of the videos
(split by name hash): tune on one half, check on the other.
"""
import argparse
import hashlib
import json
import os
import sqlite3
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import db                                   # noqa: E402
import embeddings                           # noqa: E402
import search                               # noqa: E402
from evaluate import prf, score             # noqa: E402
from query_parser import parse_rules        # noqa: E402

EXTRA_CONF, SINGLE_CONF, GAP = 0.2, 0.5, 3.0     # as indexer.EXTRA_CONF / search.make_segment / MAX_GAP


def raw_segments(seconds):
    """{"12.0": {"fire": 0.4}, ...} -> pseudo segments, one per run of detections of a class."""
    by_cls = {}
    for t, best in seconds.items():
        for cls, conf in best.items():
            if conf >= EXTRA_CONF:
                by_cls.setdefault(cls, []).append((float(t), conf))
    out = []
    for cls, dets in by_cls.items():
        dets.sort()
        runs = [[dets[0]]]
        for d in dets[1:]:
            if d[0] - runs[-1][-1][0] > GAP:
                runs.append([])
            runs[-1].append(d)
        for run in runs:
            if len(run) == 1 and run[0][1] < SINGLE_CONF:
                continue
            best = max(run, key=lambda d: d[1])
            out.append({"t": run[0][0], "dur": max(1.0, run[-1][0] - run[0][0]), "cls": cls,
                        "conf": sum(c for _, c in run) / len(run), "color": "", "speed": 0.0,
                        "entering": False, "exiting": False, "best_t": best[0],
                        "box": {"x": 0.0, "y": 0.0, "w": 100.0, "h": 100.0},
                        "path": [[t, 0.0, 0.0, 100.0, 100.0] for t, _ in run],
                        "track_id": -2, "entity_id": -2, "embedding_vec": None})
    return out


def half(video):
    return int(hashlib.md5(video.encode()).hexdigest(), 16) % 2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("db")
    ap.add_argument("gt")
    ap.add_argument("--extra-raw", help="per-second confidences of an extra detector (eval/detector_bench.py)")
    ap.add_argument("--set", action="append", default=[], metavar="NAME=VALUE",
                    help="override a constant of search.py, e.g. MIN_SCENE=0.3")
    ap.add_argument("--track-views", type=int, default=0, metavar="N",
                    help="use only the first N CLIP views of every track (1 = the best crop alone), to see what the context views add")
    ap.add_argument("--report", help="write the per-query rows to this json")
    args = ap.parse_args()
    for kv in args.set:
        k, v = kv.split("=")
        assert hasattr(search, k), f"search.py has no {k}"
        setattr(search, k, float(v))

    conn = sqlite3.connect(args.db)
    gt = json.load(open(args.gt, encoding="utf-8"))
    raw = json.load(open(args.extra_raw)) if args.extra_raw else {}
    dim = embeddings.embed_texts(["x"]).shape[1]
    rows, skipped = [], 0
    for video, spec in gt["videos"].items():
        meta = db.get_video(conn, video)
        if meta is None:
            skipped += 1
            continue
        segments = db.load_segments(conn, video)
        for s in segments:
            s["embedding_vec"] = embeddings.from_blob(s.pop("embedding"))
            if args.track_views and s["embedding_vec"] is not None:
                s["embedding_vec"] = s["embedding_vec"].reshape(-1, dim)[:args.track_views]
        segments += raw_segments(raw.get(video, {}))
        frames = [(t, embeddings.from_blob(b)) for t, b in db.load_frames(conn, video)]
        classes = sorted({s["cls"] for s in segments})
        for q in spec["queries"]:
            p = parse_rules(q["text"], classes or None)
            res, _ = search.run_query(segments, frames, p["objects"], p["colors"], p["actions"],
                                      p["timeRange"], p["semantic"], p["unknown"],
                                      embed=embeddings.embed_texts, W=meta["width"] or 16, H=meta["height"] or 9)
            tp, fp, fn = score(res, q["expected"], 0.3, "hit")
            rows.append({"video": video, "query": q["text"], "scenario": q.get("scenario") or "all",
                         "returned": len(res), "expected": len(q["expected"]), "tp": tp, "fp": fp, "fn": fn})

    def line(name, sel):
        tp, fp, fn = (sum(r[k] for r in sel) for k in ("tp", "fp", "fn"))
        p, r, f = prf(tp, fp, fn)
        return f"{name:10s} P={p:.2f} R={r:.2f} F1={f:.3f}  (found {tp} of {tp + fn} windows, {tp + fp} segments returned)"

    for sc in sorted({r["scenario"] for r in rows}):
        print(line(sc, [r for r in rows if r["scenario"] == sc]))
    print(line("OVERALL", rows) + (f"  [{skipped} videos not in the index]" if skipped else ""))
    for h in (0, 1):
        print(line(f"half {h}", [r for r in rows if half(r["video"]) == h]))
    if args.report:
        json.dump(rows, open(args.report, "w"), indent=1)


if __name__ == "__main__":
    main()
