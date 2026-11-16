"""
Turn the UCA sentences of the selected UCF-Crime videos into a ground truth
for evaluate.py: one query per security scenario, expected = the time windows
of the sentences that describe it (overlapping windows merged).

    python eval/uca_ground_truth.py            -> eval/ground_truth_uca.json

Theft / vandalism scenarios (UCF-Crime "Anomaly-Videos-Part-4": Shoplifting,
Stealing, Vandalism) are opt-in, so the default output above does not change:

    python eval/uca_ground_truth.py --sel eval/data/ucf_selection_part4.json \
        --scenarios conceal,vandal,vehicle --out eval/ground_truth_uca_part4.json

A sentence counts for a scenario only when it says the thing is on screen
(TAGS): "poured gasoline on the door" is not fire yet, "the fire truck
arrived" is not fire either. Normal videos get every query with
expected = [] so false alarms count; Arson / Shooting videos always get
their query, even when no sentence matched, so a miss there is a real miss.
"""
import argparse
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
    "conceal": "a person putting an item into their bag or pocket",
    "vandal":  "someone smashing or damaging property",
    "vehicle": "someone breaking into a car or taking a motorcycle",
}
DEFAULT_SCENARIOS = ["knife", "weapon", "fire", "snatch", "package"]
HIDE = (r"(bags?|pockets?|purses?|backpacks?|schoolbags?|handbags?|clothes|clothing|jackets?|coats?|"
        r"pants|trousers|waist|sleeves?|arms|bosom|chest|shirt)")
TWO_WHEEL = r"(motorcycles?|motorbikes?|bicycles?|bikes?|scooters?)"

OBJ = r"(bags?|purses?|handbags?|backpacks?|phones?|wallets?|necklaces?|chains?|belongings)"
GRAB = r"(snatch\w*|grab\w*|yank\w*|fighting for|fought for|ran (away|off) with|took away|stole|steal\w*)"
TAGS = {
    "fire":    re.compile(r"\b(fires?|flames?|burn(s|ing|ed)?|blaz\w*|ignit\w*|explo\w*)\b"),
    "knife":   re.compile(r"\b(knife|knives|machete|blades?|stab\w*|cleaver)\b"),
    "weapon":  re.compile(r"\b(guns?|pistols?|rifles?|shotguns?|handguns?|firearms?|shoot\w*|shots?|weapons?)\b"),
    "snatch":  re.compile(rf"\b{GRAB}\b.*\b{OBJ}\b|\b{OBJ}\b.*\b{GRAB}\b"),
    "package": re.compile(r"\b(packages?|parcels?|deliver\w*|couriers?)\b"),
    # "put the phone into his pocket", "stuffed the clothes into the bag"
    "conceal": re.compile(rf"\b(put|puts|putting|stuff\w*|hid|hide|hides|hiding|slipp?\w*|tuck\w*|stash\w*|"
                          rf"conceal\w*)\b[^.,;]*\b(in|into|inside|under)\b[^.,;]*\b{HIDE}\b"),
    "vandal":  re.compile(r"\b(smash\w*|vandali\w*|graffiti|spray[- ]?paint\w*|scratch\w*|shatter\w*|destroy\w*|"
                          r"damag\w*|kick(ed|ing|s)? (the|a|at)? ?(door|car|window|glass|sign|trash)\w*|"
                          r"(broke|break|breaking) the|overturn\w*|knock(ed)? over|"
                          r"threw [^.]*\b(at|into) the (window|car|door|glass))"),
    "vehicle": re.compile(rf"\b(pr(y|ied|ying)|broke into|break into|breaking into)\b|"
                          rf"\b(push\w*|rode|ride|drove|drives|drive)\b[^.]*\b{TWO_WHEEL}\b[^.]*\b(away|off|left)\b"),
}
# contain a tag word but do not put the thing on screen
NOT_VISIBLE = re.compile(r"\b(fire ?trucks?|fire ?fighters?|firem[ae]n|fire ?engines?|fire ?extinguishers?|"
                         r"fire ?hydrants?|smoke ?detectors?|gun ?shops?|box ?office|shooting ?range)\b")
# a video of this category is always asked its scenario
CATEGORY_SCENARIO = {"Arson": "fire", "Shooting": "weapon", "Explosion": "fire",
                     "Shoplifting": "conceal", "Vandalism": "vandal"}


def tag(sentence, scenarios=None):
    s = NOT_VISIBLE.sub(" ", sentence.lower())
    return [sc for sc, rx in TAGS.items() if (scenarios is None or sc in scenarios) and rx.search(s)]


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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sel", default=SEL, help="video selection json (names -> {bytes})")
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--scenarios", default=",".join(DEFAULT_SCENARIOS),
                    help=f"comma separated, from: {','.join(QUERY)}")
    args = ap.parse_args()
    active = [s.strip() for s in args.scenarios.split(",") if s.strip()]
    unknown = [s for s in active if s not in QUERY]
    if unknown:
        ap.error(f"unknown scenarios: {unknown}")
    sel = json.load(open(args.sel))
    uca = json.load(open(ALL, encoding="utf-8"))
    videos, tagged, samples = {}, Counter(), defaultdict(list)
    for name in sel:
        if name not in uca:
            print(f"  skip {name}: no UCA annotation")
            continue
        cat = category(name)
        windows = defaultdict(list)
        for t, sent in zip(uca[name]["timestamps"], uca[name]["sentences"]):
            for sc in tag(sent, active):
                windows[sc].append(list(t))
                tagged[sc] += 1
                if len(samples[sc]) < 5:
                    samples[sc].append(f"    {name} {t}: {sent.strip()[:90]}")
        forced = {CATEGORY_SCENARIO[cat]} & set(active) if cat in CATEGORY_SCENARIO else set()
        if cat.startswith("Normal"):
            scenarios = list(active)
        else:
            scenarios = sorted(set(windows) | forced)
        queries = [{"text": QUERY[sc], "scenario": sc, "expected": merge(windows.get(sc, []))}
                   for sc in scenarios]
        videos[name + ".mp4"] = {"category": cat, "queries": queries}
    json.dump({"videos": videos}, open(args.out, "w"), indent=1)
    n_q = sum(len(v["queries"]) for v in videos.values())
    n_w = sum(len(q["expected"]) for v in videos.values() for q in v["queries"])
    print(f"{len(videos)} videos, {n_q} queries, {n_w} expected windows -> {args.out}")
    for sc in active:
        print(f"  {sc}: {tagged[sc]} sentences")
        print("\n".join(samples[sc]))


if __name__ == "__main__":
    main()
