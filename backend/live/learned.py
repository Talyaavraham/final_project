"""
Approach (c) of the research question: a model LEARNED from pose-and-object sequences, next to
(a) "object seen" and (b) the hand-written rules of alert_rules.py.

Input: the same per-frame cache the rules are measured on (live/perceive.py): skeletons + boxes.
For every frame, a window of the last WINDOW seconds is turned into numbers (features below);
a gradient-boosted classifier learns P(event | window) from the frames inside / outside the labelled
events of the TRAINING videos (UCA sentences of UCF-Crime videos that are not in the test set:
live/train_set.py). An alert fires when P >= threshold, at most once per cooldown, like a rule.

    # train on the training videos
    python live/learned.py train --gt eval/ground_truth_train.json --perception DIR_TRAIN \
        --scenario weapon --out live/models/learned_weapon.joblib
    # test on the 107 benchmark videos: one "method" per threshold, same table / json as eval_alerts.py
    python live/learned.py eval --gt eval/ground_truth_uca.json --perception DIR_TEST \
        --model live/models/learned_weapon.joblib --name learned_weapon

The result json has the eval_alerts.py layout, so live/compare.py compares the learned model with
"gun seen" / "holds gun" on the same videos.

Features of one frame (target = the scenario's object classes, e.g. gun):
  obj_conf     best confidence of a target object      n_obj      number of target objects
  n_person     people in view (max 5)                  touch_conf best target conf with a wrist on it
  hand_dist    closest wrist - target box, in person heights (2 = none)
  raise_any    highest wrist above its shoulder, in person heights   raise_obj  the same for the wrist
               nearest the object                      obj_size   sqrt(area) of the best target box / frame
  person_speed fastest person centre move since the previous frame, in person heights
  obj_speed    move of the best target box since the previous frame, in frame diagonals
and of the window: mean, max, min and last value of each, plus the fraction of frames with a target.
"""
import argparse
import json
import math
import os
import statistics
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)
import alert_rules as ar                                                         # noqa: E402
from eval_alerts import (load_gt, load_perception, score, bootstrap, video_hours,  # noqa: E402
                         RESULTS, SLACK_BEFORE, SLACK_AFTER)

WINDOW = 2.0                      # seconds of history per decision
TARGETS = {"weapon": {"gun"}, "fire": {"fire", "smoke"}, "knife": {"knife"}, "snatch": {"bag"}}
FRAME_FEATURES = ["obj_conf", "n_obj", "n_person", "touch_conf", "hand_dist", "raise_any", "raise_obj",
                  "obj_size", "person_speed", "obj_speed"]
THRESHOLDS = (0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95)
COOLDOWN = ar.DEFAULT_COOLDOWN


def _kp(p, i):
    k = p["kpts"][i] if p.get("kpts") and len(p["kpts"]) > i else None
    return (k[0], k[1]) if k and k[2] >= ar.KPT_CONF else None


def _height(p):
    return max(1.0, p["box"][3] - p["box"][1])


def _centre(b):
    return ((b[0] + b[2]) / 2, (b[1] + b[3]) / 2)


def _dist_to_box(pt, b):
    dx = max(b[0] - pt[0], 0, pt[0] - b[2])
    dy = max(b[1] - pt[1], 0, pt[1] - b[3])
    return math.hypot(dx, dy)


def frame_features(fr, prev, target, w, h):
    objs = [o for o in fr["objects"] if o["cls"] in target]
    persons = fr["persons"]
    best = max(objs, key=lambda o: o["conf"], default=None)
    f = dict.fromkeys(FRAME_FEATURES, 0.0)
    f["hand_dist"] = 2.0
    f["obj_conf"] = best["conf"] if best else 0.0
    f["n_obj"] = float(min(len(objs), 5))
    f["n_person"] = float(min(len(persons), 5))
    diag = math.hypot(w or 1, h or 1)
    if best:
        b = best["box"]
        f["obj_size"] = math.sqrt(max(0.0, (b[2] - b[0]) * (b[3] - b[1]))) / diag
    nearest = None                                    # (distance, person, wrist)
    for p in persons:
        ph = _height(p)
        for wi in (ar.L_WRIST, ar.R_WRIST):
            pt = _kp(p, wi)
            if pt is None:
                continue
            sh = _kp(p, ar.WRIST_SHOULDER[wi])
            if sh:
                f["raise_any"] = max(f["raise_any"], (sh[1] - pt[1]) / ph)
            for o in objs:
                d = _dist_to_box(pt, o["box"]) / ph
                if nearest is None or d < nearest[0]:
                    nearest = (d, p, wi, sh)
                if d < 0.15:
                    f["touch_conf"] = max(f["touch_conf"], o["conf"])
    if nearest:
        d, p, wi, sh = nearest
        f["hand_dist"] = min(2.0, d)
        if sh:
            f["raise_obj"] = (sh[1] - _kp(p, wi)[1]) / _height(p)
    if prev is not None:
        before = {p.get("id"): p for p in prev["persons"] if p.get("id") is not None}
        for p in persons:
            q = before.get(p.get("id"))
            if q:
                (x1, y1), (x2, y2) = _centre(p["box"]), _centre(q["box"])
                f["person_speed"] = max(f["person_speed"], math.hypot(x1 - x2, y1 - y2) / _height(p))
        pb = max((o for o in prev["objects"] if o["cls"] in target), key=lambda o: o["conf"], default=None)
        if best and pb:
            (x1, y1), (x2, y2) = _centre(best["box"]), _centre(pb["box"])
            f["obj_speed"] = math.hypot(x1 - x2, y1 - y2) / diag
    return [f[k] for k in FRAME_FEATURES]


def feature_names():
    names = [f"{k}_{a}" for k in FRAME_FEATURES for a in ("mean", "max", "min", "last")]
    return names + ["obj_frac"]


def video_features(head, frames, target, window=WINDOW):
    """-> (times, rows): one row of window features per frame."""
    w, h = head.get("w") or 0, head.get("h") or 0
    per, times, rows, prev = [], [], [], None
    for fr in frames:
        per.append((fr["t"], frame_features(fr, prev, target, w, h)))
        prev = fr
    lo = 0
    for i, (t, _) in enumerate(per):
        while per[lo][0] < t - window:
            lo += 1
        win = [v for _, v in per[lo:i + 1]]
        row = []
        for j in range(len(FRAME_FEATURES)):
            col = [v[j] for v in win]
            row += [sum(col) / len(col), max(col), min(col), col[-1]]
        row.append(sum(1 for v in win if v[0] > 0) / len(win))
        times.append(t)
        rows.append(row)
    return times, rows


def labels(times, windows):
    return [1 if any(s <= t <= e for s, e in windows) else 0 for t in times]


def dataset(gt, perception, scenario):
    X, y, groups = [], [], []
    target = TARGETS[scenario]
    for v, scen in gt.items():
        p = os.path.join(perception, v + ".jsonl")
        if scenario not in scen or not os.path.isfile(p):
            continue
        head, frames = load_perception(p)
        times, rows = video_features(head, frames, target)
        X += rows
        y += labels(times, scen[scenario])
        groups += [v] * len(rows)
    return X, y, groups


def train(args):
    from sklearn.ensemble import HistGradientBoostingClassifier
    import joblib
    gt = load_gt(args.gt)
    X, y, groups = dataset(gt, args.perception, args.scenario)
    if not X or sum(y) == 0:
        sys.exit(f"no training frames with a '{args.scenario}' event under {args.perception}")
    print(f"{len(set(groups))} videos, {len(X)} frames, {sum(y)} inside an event ({100 * sum(y) / len(y):.1f}%)")
    clf = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, max_leaf_nodes=31,
                                         l2_regularization=1.0, class_weight="balanced", random_state=0)
    clf.fit(X, y)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    joblib.dump({"model": clf, "scenario": args.scenario, "target": sorted(TARGETS[args.scenario]),
                 "window": WINDOW, "features": feature_names(), "train_gt": args.gt,
                 "train_videos": len(set(groups)), "train_frames": len(X), "train_pos": int(sum(y))}, args.out)
    print("saved", args.out)


def alerts_from(times, probs, thr, cooldown=COOLDOWN):
    out, last = [], -1e9
    for t, p in zip(times, probs):
        if p >= thr and t - last >= cooldown:
            out.append(t)
            last = t
    return out


def evaluate(args):
    import joblib
    pack = joblib.load(args.model)
    clf, sc = pack["model"], pack["scenario"]
    gt = load_gt(args.gt)
    vids = [v for v in gt if sc in gt[v] and os.path.isfile(os.path.join(args.perception, v + ".jsonl"))]
    if not vids:
        sys.exit("no test videos with a perception cache")
    probs, cached = {}, {}
    for v in vids:
        head, frames = load_perception(os.path.join(args.perception, v + ".jsonl"))
        times, rows = video_features(head, frames, set(pack["target"]), pack["window"])
        probs[v] = (times, clf.predict_proba(rows)[:, 1].tolist() if rows else [])
        cached[v] = video_hours(head, frames)
    hours = sum(cached.values())
    report = {"model": args.model, "gt": args.gt, "scenarios": {sc: {}},
              "slack": [SLACK_BEFORE, SLACK_AFTER], "train": {k: pack[k] for k in pack if k.startswith("train")}}
    head = (f"{'scenario':12s} {'method':16s} {'videos':>6s} {'hours':>6s} {'events':>6s} {'caught':>6s} {'recall':>7s} "
            f"{'95% CI':>13s} {'alerts':>6s} {'false':>6s} {'false/h':>8s} {'95% CI':>13s} {'delay':>7s}")
    lines = [head]
    for thr in THRESHOLDS:
        per_video, delays = {}, []
        tot = {"ev": 0, "hit": 0, "al": 0, "fa": 0}
        for v in vids:
            times, pr = probs[v]
            al = alerts_from(times, pr, thr)
            hits, d, false = score(gt[v][sc], al)
            per_video[v] = {"events": len(gt[v][sc]), "caught": hits, "alerts": [round(t, 1) for t in al],
                            "false": false, "hours": round(cached[v], 5)}
            tot["ev"] += len(gt[v][sc]); tot["hit"] += hits; tot["al"] += len(al); tot["fa"] += false
            delays += d
        rec = tot["hit"] / tot["ev"] if tot["ev"] else None
        fph = tot["fa"] / hours if hours else None
        dly = statistics.median(delays) if delays else None
        ci = bootstrap(per_video)
        name = f"learned p>={thr}"
        report["scenarios"][sc][name] = {"threshold": thr, "videos": len(vids), "hours": round(hours, 3),
                                         "events": tot["ev"], "caught": tot["hit"], "recall": rec,
                                         "alerts": tot["al"], "false": tot["fa"], "false_per_hour": fph,
                                         "median_delay": dly, "ci95": ci, "per_video": per_video}
        rci = f"{ci[0] * 100:.0f}-{ci[1] * 100:.0f}%" if ci and ci[0] is not None else "—"
        fci = f"{ci[2]:.1f}-{ci[3]:.1f}" if ci and ci[2] is not None else "—"
        lines.append(f"{sc:12s} {name:16s} {len(vids):6d} {hours:6.2f} {tot['ev']:6d} {tot['hit']:6d} "
                     f"{(f'{rec * 100:.1f}%' if rec is not None else '—'):>7s} {rci:>13s} {tot['al']:6d} {tot['fa']:6d} "
                     f"{(f'{fph:.1f}' if fph is not None else '—'):>8s} {fci:>13s} "
                     f"{(f'{dly:.1f}s' if dly is not None else '—'):>7s}")
    imp = getattr(clf, "feature_importances_", None)
    print("\n".join(lines))
    os.makedirs(RESULTS, exist_ok=True)
    with open(os.path.join(RESULTS, args.name + ".json"), "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=1)
    with open(os.path.join(RESULTS, args.name + ".txt"), "w", encoding="utf-8") as f:
        f.write(f"python live/learned.py eval --model {args.model}  (trained on {pack.get('train_videos')} videos, "
                f"{pack.get('train_pos')} event frames of {pack.get('train_frames')})\n\n" + "\n".join(lines) + "\n")
    print(f"\nsaved {os.path.join(RESULTS, args.name)}.json / .txt")
    if imp is not None:
        top = sorted(zip(feature_names(), imp), key=lambda x: -x[1])[:8]
        print("most used features:", ", ".join(f"{n} {v:.2f}" for n, v in top))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("train")
    t.add_argument("--gt", nargs="+", required=True)
    t.add_argument("--perception", required=True)
    t.add_argument("--scenario", required=True, choices=sorted(TARGETS))
    t.add_argument("--out", required=True)
    e = sub.add_parser("eval")
    e.add_argument("--gt", nargs="+", required=True)
    e.add_argument("--perception", required=True)
    e.add_argument("--model", required=True)
    e.add_argument("--name", required=True)
    args = ap.parse_args()
    train(args) if args.cmd == "train" else evaluate(args)


if __name__ == "__main__":
    main()
