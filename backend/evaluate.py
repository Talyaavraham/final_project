"""
Precision / recall of the search engine against a hand-labelled ground truth.

    python evaluate.py eval/ground_truth.json [--api http://localhost:8000]

ground_truth.json:
{
  "videos": {
    "street.mp4": {
      "queries": [
        {"text": "a person with a dog",
         "expected": [[12.0, 18.5], [40.0, 47.0]]},     # time windows (s)
        {"text": "red car", "expected": []}
      ]
    }
  }
}

A returned match (t, t+dur) counts as a true positive when it overlaps an
expected window (IoU >= --iou, default 0.3). Each expected window is matched
at most once. Prints per-query and overall precision, recall and F1, and
writes eval/report.json.
"""

import argparse
import json
import os
import sys
import time

import httpx


def interval_iou(a, b):
    lo, hi = max(a[0], b[0]), min(a[1], b[1])
    inter = max(0.0, hi - lo)
    union = (a[1] - a[0]) + (b[1] - b[0]) - inter
    return inter / union if union > 0 else 0.0


def score(matches, expected, iou_thr):
    got = [(m["t"], m["t"] + m["dur"]) for m in matches]
    used = set()
    tp = 0
    for g in got:
        best, best_i = 0.0, None
        for i, e in enumerate(expected):
            if i in used:
                continue
            v = interval_iou(g, e)
            if v > best:
                best, best_i = v, i
        if best_i is not None and best >= iou_thr:
            used.add(best_i)
            tp += 1
    fp = len(got) - tp
    fn = len(expected) - tp
    return tp, fp, fn


def prf(tp, fp, fn):
    p = tp / (tp + fp) if tp + fp else (1.0 if fn == 0 else 0.0)
    r = tp / (tp + fn) if tp + fn else 1.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return p, r, f


def wait_ready(api, video, timeout=3600):
    t0 = time.time()
    while time.time() - t0 < timeout:
        st = httpx.get(f"{api}/status", params={"video": video}).json()
        if st["status"] == "ready":
            return
        if st["status"].startswith("error"):
            raise SystemExit(f"{video}: {st['status']}")
        if st["status"] == "unknown":
            raise SystemExit(f"{video}: not indexed — upload it first (POST /index)")
        time.sleep(2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ground_truth")
    ap.add_argument("--api", default="http://localhost:8000")
    ap.add_argument("--iou", type=float, default=0.3)
    ap.add_argument("--near", action="store_true", default=True)
    ap.add_argument("--no-near", dest="near", action="store_false")
    args = ap.parse_args()

    gt = json.load(open(args.ground_truth, encoding="utf-8"))
    info = httpx.get(f"{args.api}/api").json()
    print(f"server: model={info['model']} clip={info['clip']} llm_parser={info['llm_parser']}")

    rows, TP, FP, FN = [], 0, 0, 0
    for video, spec in gt["videos"].items():
        wait_ready(args.api, video)
        for q in spec["queries"]:
            parsed = httpx.post(f"{args.api}/parse",
                                json={"text": q["text"], "video": video}, timeout=60).json()
            res = httpx.post(f"{args.api}/search", json={
                "video": video, "near": args.near, "limit": 1000,
                "query": {**parsed, "text": q["text"]}}, timeout=120).json()
            tp, fp, fn = score(res["results"], q["expected"], args.iou)
            p, r, f = prf(tp, fp, fn)
            TP, FP, FN = TP + tp, FP + fp, FN + fn
            rows.append({"video": video, "query": q["text"], "scenario": q.get("scenario"),
                         "parsed": parsed,
                         "returned": len(res["results"]), "expected": len(q["expected"]),
                         "tp": tp, "fp": fp, "fn": fn,
                         "precision": round(p, 3), "recall": round(r, 3), "f1": round(f, 3)})
            print(f"{video:24s} {q['text'][:38]:40s} P={p:.2f} R={r:.2f} F1={f:.2f} "
                  f"(got {len(res['results'])}, want {len(q['expected'])}, via {parsed['source']})")

    # per scenario (ground truths that tag their queries, e.g. the UCA benchmark)
    by_scenario = {}
    for row in rows:
        if row["scenario"]:
            s = by_scenario.setdefault(row["scenario"], {"tp": 0, "fp": 0, "fn": 0, "queries": 0})
            s["tp"] += row["tp"]; s["fp"] += row["fp"]; s["fn"] += row["fn"]; s["queries"] += 1
    if by_scenario:
        print("\nBY SCENARIO")
        for name, s in sorted(by_scenario.items()):
            p, r, f = prf(s["tp"], s["fp"], s["fn"])
            s.update(precision=round(p, 3), recall=round(r, 3), f1=round(f, 3))
            print(f"  {name:10s} P={p:.2f} R={r:.2f} F1={f:.2f}  (tp={s['tp']} fp={s['fp']} fn={s['fn']}, {s['queries']} queries)")

    P, R, F = prf(TP, FP, FN)
    print(f"\nOVERALL  precision={P:.3f}  recall={R:.3f}  f1={F:.3f}  "
          f"(tp={TP} fp={FP} fn={FN}, {len(rows)} queries)")
    out = os.path.splitext(args.ground_truth)[0] + "_report.json"
    json.dump({"server": info, "iou": args.iou, "overall": {"precision": P, "recall": R, "f1": F},
               "by_scenario": by_scenario, "queries": rows}, open(out, "w", encoding="utf-8"),
              indent=2, ensure_ascii=False)
    print("report:", out)


if __name__ == "__main__":
    sys.exit(main())
