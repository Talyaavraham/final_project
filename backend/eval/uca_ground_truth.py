"""
Turn the UCA sentences of the selected UCF-Crime videos into a ground truth
for evaluate.py: one query per security scenario, expected = the time windows
of the sentences that describe it (overlapping windows merged).

    python eval/uca_ground_truth.py            -> eval/ground_truth_uca.json

Normal videos get every query with expected = [] so false alarms count.
"""
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
SEL = os.path.join(HERE, "data", "ucf_selection.json")
OUT = os.path.join(HERE, "ground_truth_uca.json")

# what an operator would type for each scenario (English; the parser handles it)
QUERY = {
    "knife":   "a person holding a knife",
    "weapon":  "a person with a gun",
    "fire":    "fire",
    "snatch":  "someone grabbing a bag",
    "package": "a person carrying a package",
}


def merge(windows, gap=1.0):
    out = []
    for lo, hi in sorted(windows):
        if out and lo <= out[-1][1] + gap:
            out[-1][1] = max(out[-1][1], hi)
        else:
            out.append([lo, hi])
    return out


def main():
    sel = json.load(open(SEL))
    videos = {}
    for name, info in sel.items():
        windows = {}
        for ev in info["events"]:
            windows.setdefault(ev["scenario"], []).append(ev["t"])
        queries = []
        scenarios = info["scenarios"] if "normal" not in info["scenarios"] else list(QUERY)
        for sc in scenarios:
            queries.append({"text": QUERY[sc], "scenario": sc,
                            "expected": merge(windows.get(sc, []))})
        videos[name + ".mp4"] = {"category": name.rstrip("_x264").rstrip("0123456789"), "queries": queries}
    json.dump({"videos": videos}, open(OUT, "w"), indent=1)
    n_q = sum(len(v["queries"]) for v in videos.values())
    n_w = sum(len(q["expected"]) for v in videos.values() for q in v["queries"])
    print(f"{len(videos)} videos, {n_q} queries, {n_w} expected windows -> {OUT}")


if __name__ == "__main__":
    main()
