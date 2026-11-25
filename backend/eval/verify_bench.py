"""
Would a second look remove the false alarms?  The search returns candidates
with fair recall and low precision (0.12 on the benchmark). Here a small
vision-language model, used as it is (no training), looks at three frames of
every returned segment and answers the scenario's yes/no question; segments
below a P(yes) threshold are dropped and the benchmark is scored again.

    python eval/verify_bench.py eval/benchmark_v2.db eval/ground_truth_uca.json --videos eval/data/ucf
    python eval/verify_bench.py eval/heldout.db eval/ground_truth_heldout.json \
        --videos C:/path/Anomaly-Videos-Part-1 --thresholds fire=0.5,weapon=0.5

Without --thresholds: a sweep per scenario, how well P(yes) separates right
from wrong segments (AUC, 0.5 = chance), and a cross-check: thresholds picked
on one half of the videos, applied to the other half. With --thresholds:
only those, for a set nothing was tuned on.

P(yes) of every segment is cached (eval/data/verify_<db>.json), so only the
first run needs the model and the videos.
"""
import argparse
import glob
import json
import os
import sqlite3
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from offline_eval import answers, half      # noqa: E402  (also puts backend/ on the path)
from evaluate import prf, score             # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL = "HuggingFaceTB/SmolVLM2-500M-Video-Instruct"
QUESTION = {
    "fire":    "Is there fire or flames in these images?",
    "weapon":  "Is a person holding a gun in these images?",
    "knife":   "Is a person holding a knife in these images?",
    "snatch":  "Is someone grabbing a bag from another person in these images?",
    "package": "Is a person carrying a package or a box in these images?",
}
LOOKS = (0.2, 0.5, 0.8)             # where in the segment the frames are taken
GRID = [round(0.1 * i, 1) for i in range(10)]      # 0.0 = keep everything (today's search)


class Verifier:
    """P(yes) of a yes/no question about a few images: the model's first answer token, yes against no."""

    def __init__(self, name=MODEL):
        import torch
        from transformers import AutoModelForImageTextToText, AutoProcessor
        self.torch = torch
        try:
            self.proc = AutoProcessor.from_pretrained(name)
        except ImportError:         # SmolVLM's own processor wants num2words, which only its video path uses
            from transformers import Idefics3Processor
            self.proc = Idefics3Processor.from_pretrained(name)
        self.proc.image_processor.do_image_splitting = False    # one 512 px view per frame: the source is 320x240
        self.model = AutoModelForImageTextToText.from_pretrained(name, torch_dtype=torch.float32).eval()
        tok = self.proc.tokenizer
        first = lambda words: sorted({tok.encode(w, add_special_tokens=False)[0] for w in words})  # noqa: E731
        self.yes, self.no = first(["Yes", " Yes", "yes", " yes"]), first(["No", " No", "no", " no"])

    def p_yes(self, images, question):
        content = [{"type": "image"} for _ in images] + [{"type": "text", "text": question + " Answer yes or no."}]
        prompt = self.proc.apply_chat_template([{"role": "user", "content": content}], add_generation_prompt=True)
        inputs = self.proc(text=prompt, images=images, return_tensors="pt")
        with self.torch.no_grad():
            logits = self.model(**inputs).logits[0, -1]
        yes, no = self.torch.logsumexp(logits[self.yes], 0), self.torch.logsumexp(logits[self.no], 0)
        return float(self.torch.sigmoid(yes - no))


def frames_at(path, times):
    import cv2
    from PIL import Image
    cap, out = cv2.VideoCapture(path), []
    for t in times:
        cap.set(cv2.CAP_PROP_POS_MSEC, max(0.0, t) * 1000)
        ok, frame = cap.read()
        if ok:
            out.append(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
    cap.release()
    return out


def key(video, scenario, s):
    return f"{video}|{scenario}|{s['t']:.2f}|{s['dur']:.2f}"


def right(seg, expected):
    return score([seg], expected, 0.3, "hit")[1] == 0


def counts(rows, thresholds):
    """rows -> {scenario: [tp, fp, fn]} after dropping segments with P(yes) below the scenario's threshold."""
    out = {}
    for r in rows:
        kept = [s for s in r["segs"] if s["p"] >= thresholds.get(r["scenario"], 0.0)]
        c = out.setdefault(r["scenario"], [0, 0, 0])
        for i, v in enumerate(score(kept, r["expected"], 0.3, "hit")):
            c[i] += v
    return out


def total(by_scenario):
    return [sum(c[i] for c in by_scenario.values()) for i in range(3)]


def fmt(name, c):
    p, r, f = prf(*c)
    return f"{name:10s} P={p:.2f} R={r:.2f} F1={f:.3f}  (found {c[0]} of {c[0] + c[2]} windows, {c[0] + c[1]} segments returned)"


def auc(pos, neg):
    """Chance that a right segment gets a higher P(yes) than a wrong one."""
    if not pos or not neg:
        return float("nan")
    return sum((a > b) + 0.5 * (a == b) for a in pos for b in neg) / (len(pos) * len(neg))


def best_thresholds(rows):
    """Per scenario, the grid threshold with the best F1 (the lowest one on a tie)."""
    out = {}
    for sc in sorted({r["scenario"] for r in rows}):
        sel = [r for r in rows if r["scenario"] == sc]
        out[sc] = max(GRID, key=lambda t: (prf(*counts(sel, {sc: t})[sc])[2], -t))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("db")
    ap.add_argument("gt")
    ap.add_argument("--videos", help="folder with the videos (searched recursively); not needed once everything is cached")
    ap.add_argument("--thresholds", help="fixed P(yes) thresholds, e.g. fire=0.5,weapon=0.5 (no sweep)")
    ap.add_argument("--cache", help="default: eval/data/verify_<db name>.json")
    ap.add_argument("--model", default=MODEL)
    args = ap.parse_args()
    cache_path = args.cache or os.path.join(HERE, "data", f"verify_{os.path.splitext(os.path.basename(args.db))[0]}.json")
    cache = json.load(open(cache_path)) if os.path.exists(cache_path) else {}
    paths = {os.path.basename(p): p for p in glob.glob(os.path.join(args.videos, "**", "*.mp4"), recursive=True)} \
        if args.videos else {}

    gt = json.load(open(args.gt, encoding="utf-8"))
    rows, verifier, asked, t0 = [], None, 0, time.time()
    for video, q, res in answers(sqlite3.connect(args.db), gt):
        if q is None:
            continue
        sc = q.get("scenario") or "all"
        segs = []
        for s in res:
            k = key(video, sc, s)
            if k not in cache:
                if video not in paths:
                    sys.exit(f"{video}: not scored yet and not under --videos")
                verifier = verifier or Verifier(args.model)
                images = frames_at(paths[video], [s["t"] + s["dur"] * f for f in LOOKS])
                cache[k] = verifier.p_yes(images, QUESTION[sc]) if images else 1.0     # unreadable: keep the segment
                asked += 1
                if asked % 25 == 0:
                    json.dump(cache, open(cache_path, "w"))
                    print(f"  {asked} segments, {(time.time() - t0) / asked:.1f} s each", flush=True)
            segs.append({"t": s["t"], "dur": s["dur"], "p": cache[k]})
        rows.append({"video": video, "scenario": sc, "expected": q["expected"], "segs": segs})
    if asked:
        json.dump(cache, open(cache_path, "w"))
        print(f"{asked} segments scored in {(time.time() - t0) / 60:.1f} min -> {cache_path}")

    base = counts(rows, {})
    if args.thresholds:
        thr = {k: float(v) for k, v in (kv.split("=") for kv in args.thresholds.split(","))}
        after = counts(rows, thr)
        for sc in sorted(base):
            print(fmt(sc, base[sc]), "\n  ->", fmt(f"p>={thr.get(sc, 0.0)}", after[sc]))
        print(fmt("OVERALL", total(base)), "\n  ->", fmt("verified", total(after)))
        return

    for sc in sorted(base):
        sel = [r for r in rows if r["scenario"] == sc]
        ps = [(s["p"], right(s, r["expected"])) for r in sel for s in r["segs"]]
        print(f"\n{sc}: \"{QUESTION[sc]}\"  AUC {auc([p for p, ok in ps if ok], [p for p, ok in ps if not ok]):.2f} "
              f"({sum(ok for _, ok in ps)} right, {sum(not ok for _, ok in ps)} wrong segments)")
        for t in GRID:
            print(" ", fmt(f"p>={t}", counts(sel, {sc: t})[sc]))
    best = best_thresholds(rows)
    print("\nbest thresholds on all videos:", ",".join(f"{k}={v}" for k, v in best.items()))
    print(fmt("OVERALL", total(base)), "\n  ->", fmt("verified", total(counts(rows, best))), " (optimistic: picked on these videos)")
    for h in (0, 1):
        fit, test = [r for r in rows if half(r["video"]) == h], [r for r in rows if half(r["video"]) != h]
        thr = best_thresholds(fit)
        print(f"thresholds from half {h} on half {1 - h}:", ",".join(f"{k}={v}" for k, v in thr.items()))
        print(" ", fmt("before", total(counts(test, {}))), "\n ", fmt("after", total(counts(test, thr))))


if __name__ == "__main__":
    main()
