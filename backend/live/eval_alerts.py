"""
Step 2 of measuring the live alerts: replay the cached perception
(perceive.py) through the alert rules and compare with the ground truth.

    python live/eval_alerts.py --gt eval/ground_truth_uca.json
    python live/eval_alerts.py --gt live/data/labels.json            # your own labelled videos
    python live/eval_alerts.py --gt a.json b.json --conf 0.35         # another detector threshold
    python live/eval_alerts.py ... --scenarios knife,fire

For every scenario of the ground truth, every method of live/scenarios.json
(e.g. "object only" vs "raises knife") is run on the same frames, with the
same Evaluator the live server uses (alert_rules.py), and scored:

    recall          events caught: an alert between SLACK_BEFORE s before the
                    event's start and SLACK_AFTER s after its end
    false / hour    alerts outside every event of that scenario, per hour of
                    video of the videos that have this scenario (incl. videos
                    where it never happens: their expected list is empty)
    delay           median seconds from the event's start to its first alert
                    (negative = the alert came a little before the labelled start)

Only videos that have a perception cache are counted (the rest is listed).
Writes live/results/<name>.json + .txt.
"""
import argparse
import datetime
import json
import os
import statistics
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
BACKEND = os.path.dirname(HERE)
sys.path.insert(0, BACKEND)
import alert_rules            # noqa: E402

PERCEPTION = os.path.join(HERE, "data", "perception")
SCENARIOS = os.path.join(HERE, "scenarios.json")
RESULTS = os.path.join(HERE, "results")
SLACK_BEFORE, SLACK_AFTER = 1.0, 2.0
DEFAULT_CONF = 0.25


def load_gt(paths):
    """{video: {scenario: [[start, end], ...]}}; later files add to earlier ones."""
    gt = {}
    for p in paths:
        with open(p, encoding="utf-8") as f:
            for video, v in json.load(f)["videos"].items():
                for q in v.get("queries", []):
                    gt.setdefault(video, {}).setdefault(q["scenario"], []).extend(
                        [list(map(float, w)) for w in q.get("expected", [])])
    return gt


def load_perception(path):
    with open(path, encoding="utf-8") as f:
        head = json.loads(f.readline())
        frames = [json.loads(l) for l in f if l.strip()]
    return head, frames


def make_rule(name, spec):
    r = alert_rules.make_rule(name, spec["kind"], spec.get("object", ""), spec.get("min_people", 1),
                              spec.get("duration"), spec.get("cooldown"), hands=spec.get("hands", 2))
    r["conf"] = float(spec.get("conf", DEFAULT_CONF))
    return r


def run_rule(rule, frames, conf=None):
    """alert times of one rule over one video's frames."""
    c = rule["conf"] if conf is None else conf
    ev, times = alert_rules.Evaluator(), []
    for fr in frames:
        objs = [o for o in fr["objects"] if o["conf"] >= c]
        alerts, _ = ev.step([rule], fr["persons"], objs, fr["t"])
        times += [fr["t"]] * len(alerts)
    return times


def score(windows, alerts):
    """-> (hits, delays, false): per event caught or not, delays of caught ones, false alerts."""
    hits, delays = 0, []
    for s, e in windows:
        inside = [t for t in alerts if s - SLACK_BEFORE <= t <= e + SLACK_AFTER]
        if inside:
            hits += 1
            delays.append(min(inside) - s)
    false = sum(1 for t in alerts if not any(s - SLACK_BEFORE <= t <= e + SLACK_AFTER for s, e in windows))
    return hits, delays, false


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gt", nargs="+", required=True)
    ap.add_argument("--perception", default=PERCEPTION)
    ap.add_argument("--methods", default=SCENARIOS, help="scenario -> methods (json)")
    ap.add_argument("--scenarios", default="", help="only these, comma separated")
    ap.add_argument("--conf", type=float, default=None, help="detector confidence for every method")
    ap.add_argument("--name", default="", help="name of the result files")
    args = ap.parse_args()

    gt = load_gt(args.gt)
    with open(args.methods, encoding="utf-8") as f:
        cfg = {k: v for k, v in json.load(f).items() if not k.startswith("_")}
    only = {s for s in args.scenarios.split(",") if s}

    cached, missing = {}, []
    for video in gt:
        p = os.path.join(args.perception, video + ".jsonl")
        if os.path.isfile(p):
            cached[video] = load_perception(p)
        else:
            missing.append(video)
    if missing:
        print(f"{len(missing)} of {len(gt)} videos have no perception cache (run live/perceive.py): "
              + ", ".join(missing[:8]) + (" …" if len(missing) > 8 else ""))
    if not cached:
        sys.exit("nothing to evaluate")

    report, lines = {"date": datetime.date.today().isoformat(), "gt": args.gt, "conf": args.conf,
                     "slack": [SLACK_BEFORE, SLACK_AFTER], "scenarios": {}}, []
    scen_names = sorted({s for v in gt.values() for s in v} & set(cfg))
    unknown = sorted({s for v in gt.values() for s in v} - set(cfg))
    if unknown:
        print("scenarios without methods in scenarios.json (skipped): " + ", ".join(unknown))
    head = f"{'scenario':12s} {'method':16s} {'videos':>6s} {'hours':>6s} {'events':>6s} {'caught':>6s} {'recall':>7s} {'alerts':>6s} {'false':>6s} {'false/h':>8s} {'delay':>7s}"
    lines.append(head)
    for sc in scen_names:
        if only and sc not in only:
            continue
        vids = [v for v in cached if sc in gt[v]]
        if not vids:
            continue
        hours = sum(cached[v][0].get("duration") or (cached[v][1][-1]["t"] if cached[v][1] else 0) for v in vids) / 3600
        report["scenarios"][sc] = {}
        for mname, spec in cfg[sc]["methods"].items():
            rule = make_rule(mname, spec)
            tot_ev = tot_hit = tot_alerts = tot_false = 0
            delays, per_video = [], {}
            for v in vids:
                alerts = run_rule(rule, cached[v][1], args.conf)
                hits, d, false = score(gt[v][sc], alerts)
                tot_ev += len(gt[v][sc]); tot_hit += hits; tot_alerts += len(alerts); tot_false += false
                delays += d
                per_video[v] = {"events": len(gt[v][sc]), "caught": hits, "alerts": [round(t, 1) for t in alerts], "false": false}
            rec = tot_hit / tot_ev if tot_ev else None
            fph = tot_false / hours if hours else None
            dly = statistics.median(delays) if delays else None
            report["scenarios"][sc][mname] = {"rule": spec, "videos": len(vids), "hours": round(hours, 3),
                                              "events": tot_ev, "caught": tot_hit, "recall": rec,
                                              "alerts": tot_alerts, "false": tot_false, "false_per_hour": fph,
                                              "median_delay": dly, "per_video": per_video}
            rec_s = f"{rec * 100:.1f}%" if rec is not None else "—"
            fph_s = f"{fph:.1f}" if fph is not None else "—"
            dly_s = f"{dly:.1f}s" if dly is not None else "—"
            lines.append(f"{sc:12s} {mname:16s} {len(vids):6d} {hours:6.2f} {tot_ev:6d} {tot_hit:6d} "
                         f"{rec_s:>7s} {tot_alerts:6d} {tot_false:6d} {fph_s:>8s} {dly_s:>7s}")
    lines.append(f"\nrecall: alert from {SLACK_BEFORE}s before an event to {SLACK_AFTER}s after it. "
                 f"false/h: alerts outside every event, per hour of those videos. delay: median, from the event's start.")
    print("\n".join(lines))
    os.makedirs(RESULTS, exist_ok=True)
    name = args.name or f"eval_{report['date']}"
    with open(os.path.join(RESULTS, name + ".json"), "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=1)
    with open(os.path.join(RESULTS, name + ".txt"), "w", encoding="utf-8") as f:
        f.write(f"python live/eval_alerts.py --gt {' '.join(args.gt)}" + (f" --conf {args.conf}" if args.conf else "") + "\n\n")
        f.write("\n".join(lines) + "\n")
    print(f"\nsaved {os.path.join(RESULTS, name)}.json / .txt")


if __name__ == "__main__":
    main()
