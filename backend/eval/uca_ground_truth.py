"""
Turn the UCA sentences of the selected UCF-Crime videos into a ground truth
for evaluate.py: one query per security scenario, expected = the time windows
of the sentences that describe it (overlapping windows merged).

    python eval/uca_ground_truth.py            -> eval/ground_truth_uca.json

A sentence counts for a scenario only when it says the thing is on screen
(TAGS): "poured gasoline on the door" is not fire yet, "the fire truck
arrived" is not fire either. Normal videos get every query with
expected = [] so false alarms count; Arson / Shooting videos always get
their query, even when no sentence matched, so a miss there is a real miss.

A held-out set — UCF-Crime videos on disk that are NOT in the benchmark, so
no threshold was ever tuned on them:

    python eval/uca_ground_truth.py --heldout C:/path/Anomaly-Videos-Part-1 \
        --scenarios fire,weapon --out eval/ground_truth_heldout.json

Its negatives are videos of the same folder with no sentence about the chosen
scenarios (UCA describes every event of a video, so no sentence = not there),
picked in a fixed pseudo-random order up to --negative-minutes of video.
"""
import argparse
import glob
import hashlib
import json
import os
import re
from collections import Counter, defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
SEL = os.path.join(HERE, "data", "ucf_selection.json")
ALL = os.path.join(HERE, "data", "uca_all.json")
OUT = os.path.join(HERE, "ground_truth_uca.json")

# what an operator would type for each scenario (English; the parser handles it)
QUERY = {
    "knife":   "a person holding a knife",
    "weapon":  "a person with a gun",
    "fire":    "fire",
    "snatch":  "someone grabbing a bag",
    "package": "a person carrying a package",
}

OBJ = r"(bags?|purses?|handbags?|backpacks?|phones?|wallets?|necklaces?|chains?|belongings)"
GRAB = r"(snatch\w*|grab\w*|yank\w*|fighting for|fought for|ran (away|off) with|took away|stole|steal\w*)"
TAGS = {
    "fire":    re.compile(r"\b(fires?|flames?|burn(s|ing|ed)?|blaz\w*|ignit\w*|explo\w*)\b"),
    "knife":   re.compile(r"\b(knife|knives|machete|blades?|stab\w*|cleaver)\b"),
    "weapon":  re.compile(r"\b(guns?|pistols?|rifles?|shotguns?|handguns?|firearms?|shoot\w*|shots?|weapons?)\b"),
    "snatch":  re.compile(rf"\b{GRAB}\b.*\b{OBJ}\b|\b{OBJ}\b.*\b{GRAB}\b"),
    "package": re.compile(r"\b(packages?|parcels?|deliver\w*|couriers?)\b"),
}
# contain a tag word but do not put the thing on screen
NOT_VISIBLE = re.compile(r"\b(fire ?trucks?|fire ?fighters?|firem[ae]n|fire ?engines?|fire ?extinguishers?|"
                         r"fire ?hydrants?|smoke ?detectors?|gun ?shops?|box ?office|shooting ?range)\b")
# a video of this category is always asked its scenario
CATEGORY_SCENARIO = {"Arson": "fire", "Shooting": "weapon", "Explosion": "fire"}


OPEN_FIRE = re.compile(r"\b(open(s|ed|ing)?|return(s|ed|ing)?) fire\b")      # shooting, nothing burns


def tag(sentence):
    s = OPEN_FIRE.sub("shoot", NOT_VISIBLE.sub(" ", sentence.lower()))
    return [sc for sc, rx in TAGS.items() if rx.search(s)]


def category(name):
    return re.sub(r"_?\d+_x264$", "", name)


def merge(windows, gap=1.0):
    out = []
    for lo, hi in sorted(windows):
        if out and lo <= out[-1][1] + gap:
            out[-1][1] = max(out[-1][1], hi)
        else:
            out.append([lo, hi])
    return out


def tagged_windows(entry):
    """UCA entry -> {scenario: [[t0, t1], ...]} (not merged) and the tagged (scenario, t, sentence) rows."""
    windows, rows = defaultdict(list), []
    for t, sent in zip(entry["timestamps"], entry["sentences"]):
        for sc in tag(sent):
            windows[sc].append(list(t))
            rows.append((sc, t, sent))
    return windows, rows


def heldout_names(folder, uca, benchmark, scenarios, negative_minutes):
    """-> (positives, negatives): videos under `folder` that UCA describes and the benchmark does not use."""
    on_disk = {os.path.splitext(os.path.basename(p))[0]
               for p in glob.glob(os.path.join(folder, "**", "*.mp4"), recursive=True)}
    names = sorted(n for n in on_disk if n in uca and n not in benchmark)
    pos, rest = [], []
    for n in names:
        windows, _ = tagged_windows(uca[n])
        asked = CATEGORY_SCENARIO.get(category(n)) in scenarios
        (pos if asked or any(sc in windows for sc in scenarios) else rest).append(n)
    neg, left = [], negative_minutes * 60.0
    for n in sorted(rest, key=lambda n: hashlib.md5(n.encode()).hexdigest()):
        if uca[n]["duration"] <= left:
            neg.append(n)
            left -= uca[n]["duration"]
    return pos, sorted(neg)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--heldout", metavar="DIR", help="folder of UCF-Crime videos: build a set from those NOT in the benchmark")
    ap.add_argument("--scenarios", default=",".join(QUERY), help="held-out: only these scenarios")
    ap.add_argument("--negative-minutes", type=float, default=20.0, help="held-out: minutes of videos without the scenarios")
    ap.add_argument("--out", default=OUT)
    args = ap.parse_args()
    sel = json.load(open(SEL))
    uca = json.load(open(ALL, encoding="utf-8"))
    scenarios = [sc for sc in QUERY if sc in args.scenarios.split(",")]
    negatives = set()
    if args.heldout:
        names, neg = heldout_names(args.heldout, uca, sel, scenarios, args.negative_minutes)
        negatives = set(neg)
        names = sorted(names + neg)
    else:
        names = list(sel)

    videos, tagged, samples = {}, Counter(), defaultdict(list)
    for name in names:
        cat = category(name)
        windows, rows = tagged_windows(uca[name])
        for sc, t, sent in rows:
            if sc in scenarios:
                tagged[sc] += 1
                if len(samples[sc]) < 5:
                    samples[sc].append(f"    {name} {t}: {sent.strip()[:90]}")
        if cat.startswith("Normal") or name in negatives:
            asked = scenarios
        else:
            asked = sorted((set(windows) | {CATEGORY_SCENARIO.get(cat)}) & set(scenarios))
        queries = [{"text": QUERY[sc], "scenario": sc, "expected": merge(windows.get(sc, []))}
                   for sc in asked]
        videos[name + ".mp4"] = {"category": cat, "queries": queries}
    json.dump({"videos": videos}, open(args.out, "w"), indent=1)
    n_q = sum(len(v["queries"]) for v in videos.values())
    n_w = sum(len(q["expected"]) for v in videos.values() for q in v["queries"])
    minutes = sum(uca[n]["duration"] for n in names) / 60
    print(f"{len(videos)} videos ({minutes:.0f} min, {len(negatives)} negatives), {n_q} queries, "
          f"{n_w} expected windows -> {args.out}")
    for sc in scenarios:
        print(f"  {sc}: {tagged[sc]} sentences")
        print("\n".join(samples[sc]))


if __name__ == "__main__":
    main()
