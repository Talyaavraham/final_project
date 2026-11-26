"""
The TRAINING videos of the learned model (live/learned.py): UCF-Crime videos that UCA describes and
that are NOT among the 107 test videos (eval/ground_truth_uca.json), so nothing learned has been seen
in the test.

    python live/train_set.py --scenarios weapon,fire --negative-minutes 120 \
        --out eval/ground_truth_train.json

Positives: every video with a UCA sentence of a scenario (eval/uca_ground_truth.py TAGS), or of a
category that always has it (Shooting -> weapon, Arson / Explosion -> fire). Every scenario is asked
of every chosen video: UCA describes every event of a video, so no sentence = not there.
Negatives: Normal videos, in a fixed pseudo-random order, up to --negative-minutes.
Writes the ground truth in the usual format, and prints the names for the download selection.
"""
import argparse
import hashlib
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
BACKEND = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(BACKEND, "eval"))
import uca_ground_truth as u          # noqa: E402

TEST = os.path.join(BACKEND, "eval", "ground_truth_uca.json")
# the UCA annotations (eval/data/ is not in git): the same source as eval/colab_benchmark.ipynb
UCA_BASE = "https://raw.githubusercontent.com/Xuange923/Surveillance-Video-Understanding/main/UCF%20Annotation/json/"


def load_uca():
    """eval/data/uca_all.json, downloaded and merged from the three UCA splits the first time."""
    if not os.path.isfile(u.ALL):
        import urllib.request
        uca = {}
        for split in ("Train", "Val", "Test"):
            print(f"downloading UCA {split} ...", flush=True)
            with urllib.request.urlopen(UCA_BASE + f"UCFCrime_{split}.json", timeout=120) as r:
                uca.update(json.load(r))
        os.makedirs(os.path.dirname(u.ALL), exist_ok=True)
        with open(u.ALL, "w", encoding="utf-8") as f:
            json.dump(uca, f)
    with open(u.ALL, encoding="utf-8") as f:
        return json.load(f)


def build(uca, test_names, scenarios, negative_minutes, max_minutes=None):
    pos, normal = [], []
    for n in sorted(uca):
        if n in test_names:
            continue
        windows, _ = u.tagged_windows(uca[n])
        always = u.CATEGORY_SCENARIO.get(u.category(n))
        if always in scenarios or any(sc in windows for sc in scenarios):
            pos.append(n)
        elif u.category(n).startswith("Normal"):
            normal.append(n)
    if max_minutes:                                    # cap the positives (shortest first: cheaper to download)
        pos.sort(key=lambda n: float(uca[n]["duration"]))
        kept, left = [], max_minutes * 60.0
        for n in pos:
            if float(uca[n]["duration"]) <= left:
                kept.append(n)
                left -= float(uca[n]["duration"])
        pos = kept
    neg, left = [], negative_minutes * 60.0
    for n in sorted(normal, key=lambda n: hashlib.md5(n.encode()).hexdigest()):
        if float(uca[n]["duration"]) <= left:
            neg.append(n)
            left -= float(uca[n]["duration"])
    videos = {}
    for n in sorted(pos + neg):
        windows, _ = u.tagged_windows(uca[n])
        videos[n + ".mp4"] = {"category": u.category(n), "queries": [
            {"text": u.QUERY[sc], "scenario": sc, "expected": u.merge(windows.get(sc, []))} for sc in scenarios]}
    return videos, pos, neg


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scenarios", default="weapon,fire")
    ap.add_argument("--negative-minutes", type=float, default=120)
    ap.add_argument("--max-minutes", type=float, default=None, help="cap on the minutes of positive videos")
    ap.add_argument("--out", default=os.path.join(BACKEND, "eval", "ground_truth_train.json"))
    args = ap.parse_args()
    scenarios = [s.strip() for s in args.scenarios.split(",") if s.strip()]
    uca = load_uca()
    with open(TEST, encoding="utf-8") as f:
        test_names = {os.path.splitext(v)[0] for v in json.load(f)["videos"]}
    videos, pos, neg = build(uca, test_names, scenarios, args.negative_minutes, args.max_minutes)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump({"_doc": "training videos of live/learned.py, built by live/train_set.py; none is in "
                           "eval/ground_truth_uca.json", "videos": videos}, f, ensure_ascii=False, indent=1)
    mins = lambda names: sum(float(uca[n]["duration"]) for n in names) / 60
    for sc in scenarios:
        ev = sum(len(q["expected"]) for v in videos.values() for q in v["queries"] if q["scenario"] == sc)
        print(f"{sc}: {ev} events")
    print(f"{len(pos)} videos with an event ({mins(pos):.0f} min) + {len(neg)} normal ({mins(neg):.0f} min) -> {args.out}")


if __name__ == "__main__":
    main()
