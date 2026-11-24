"""
Where does an alert scenario break: in the detector, or in the rules?

For every labelled event of a scenario (eval/ground_truth_uca.json, your
labels.json), on the cached perception (live/perceive.py) it checks:

    person      a person was seen during the event
    seen@X      the scenario's object (scenarios.json "target") was detected
                during the event at confidence >= X (the cache keeps >= 0.15)
    max conf    median of the best confidence per event
    in hand     a wrist was on that object at least once during the event
    outside     detections of the object per hour OUTSIDE every event
                (what turns into false alarms)

and gives a verdict per scenario:
    detector blind          the object is almost never seen, at any confidence:
                            retrain the detector, rules cannot help
    threshold too high      seen at 0.15 far more often than at 0.25: try --conf
    detector OK             it is seen: the gap is in the rules / the labels

    python live/diagnose.py --gt eval/ground_truth_uca.json --perception <folder>
"""
import argparse
import json
import os
import statistics
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)
import alert_rules as ar                  # noqa: E402
from eval_alerts import load_gt, load_perception, PERCEPTION, SCENARIOS, RESULTS, SLACK_BEFORE, SLACK_AFTER  # noqa: E402

LEVELS = (0.15, 0.25, 0.5)


def wrist_on(person, box):
    ph = person["box"][3] - person["box"][1]
    pad = max(0.3 * max(box[2] - box[0], box[3] - box[1]), 0.06 * ph)
    for w in (ar.L_WRIST, ar.R_WRIST):
        k = person["kpts"][w]
        if k[2] >= ar.KPT_CONF and ar._near((k[0], k[1]), box, pad):
            return True
    return False


def event_stats(frames, s, e, target):
    win = [f for f in frames if s - SLACK_BEFORE <= f["t"] <= e + SLACK_AFTER]
    best, person, hand = 0.0, False, False
    for f in win:
        person |= bool(f["persons"])
        for o in f["objects"]:
            if o["cls"] in target:
                best = max(best, o["conf"])
                hand |= any(wrist_on(p, o["box"]) for p in f["persons"])
    return {"frames": len(win), "person": person, "best": best, "hand": hand}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gt", nargs="+", required=True)
    ap.add_argument("--perception", default=PERCEPTION)
    ap.add_argument("--name", default="diagnose")
    args = ap.parse_args()
    gt = load_gt(args.gt)
    with open(SCENARIOS, encoding="utf-8") as f:
        cfg = {k: v for k, v in json.load(f).items() if not k.startswith("_")}
    cache = {}
    for v in gt:
        p = os.path.join(args.perception, v + ".jsonl")
        if os.path.isfile(p):
            cache[v] = load_perception(p)
    if not cache:
        sys.exit("no perception cache for these videos (run live/perceive.py)")

    head = (f"{'scenario':12s} {'target':12s} {'events':>6s} {'person':>7s} "
            + " ".join(f"{'seen@' + str(l):>9s}" for l in LEVELS)
            + f" {'max conf':>8s} {'in hand':>8s} {'outside/h':>9s}  verdict")
    lines, report = [head], {}
    for sc, c in cfg.items():
        target = set(c.get("target") or [])
        vids = [v for v in cache if sc in gt[v]]
        if not target or not vids:
            continue
        stats = []
        outside, hours = 0, 0.0
        for v in vids:
            head_, frames = cache[v]
            wins = gt[v][sc]
            for s, e in wins:
                stats.append(event_stats(frames, s, e, target))
            hours += (head_.get("duration") or (frames[-1]["t"] if frames else 0)) / 3600
            for f in frames:
                if any(s - SLACK_BEFORE <= f["t"] <= e + SLACK_AFTER for s, e in wins):
                    continue
                outside += sum(1 for o in f["objects"] if o["cls"] in target and o["conf"] >= 0.25)
        n = len(stats)
        if not n:
            continue
        pct = lambda k: 100 * k / n
        seen = {l: sum(1 for x in stats if x["best"] >= l) for l in LEVELS}
        best = [x["best"] for x in stats if x["best"] > 0]
        med = statistics.median(best) if best else 0.0
        hand = sum(1 for x in stats if x["hand"])
        person = sum(1 for x in stats if x["person"])
        oph = outside / hours if hours else 0.0
        if seen[0.15] < 0.1 * n:
            verdict = "detector blind: retrain it, rules cannot help"
        elif seen[0.15] > 1.5 * seen[0.25]:
            verdict = "threshold too high: try a lower --conf"
        else:
            verdict = "detector sees it: the gap is in the rules / labels"
        report[sc] = {"target": sorted(target), "events": n, "person": person, "seen": seen,
                      "median_max_conf": med, "in_hand": hand, "outside_per_hour_at_0.25": oph, "verdict": verdict}
        lines.append(f"{sc:12s} {','.join(sorted(target)):12s} {n:6d} {pct(person):6.0f}% "
                     + " ".join(f"{pct(seen[l]):8.0f}%" for l in LEVELS)
                     + f" {med:8.2f} {pct(hand):7.0f}% {oph:9.1f}  {verdict}")
    lines.append("\nperson / seen / in hand: % of the labelled events. outside/h: detections (conf >= 0.25) "
                 "outside every event, per hour of those videos.")
    print("\n".join(lines))
    os.makedirs(RESULTS, exist_ok=True)
    with open(os.path.join(RESULTS, args.name + ".txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    with open(os.path.join(RESULTS, args.name + ".json"), "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
