"""
Is method A really better than method B, or is it noise?

Two methods (each from a result json of live/eval_alerts.py, possibly at different detector thresholds)
are compared on the SAME videos with a paired bootstrap: whole videos are resampled with replacement and
the difference of recall (and of false alarms per hour) is recomputed every time.

    python live/compare.py live/results/sweep_sec3_conf0.55.json weapon "holds gun" \\
                           live/results/sweep_sec3_conf0.45.json weapon "object only"

Prints both methods, the difference A - B with its 95% interval, and how often A beat B.
If the interval of the recall difference contains 0, the data cannot tell them apart.
"""
import json
import random
import sys


def load(path, scenario, method):
    with open(path, encoding="utf-8") as f:
        m = json.load(f)["scenarios"][scenario][method]
    return m["per_video"]


def totals(sample):
    ev = sum(v["events"] for v in sample)
    h = sum(v["hours"] for v in sample)
    return (sum(v["caught"] for v in sample) / ev if ev else 0.0,
            sum(v["false"] for v in sample) / h if h else 0.0)


def paired(a, b, n=5000, seed=0):
    vids = sorted(set(a) & set(b))
    if any("hours" not in a[v] or "hours" not in b[v] for v in vids):
        sys.exit("these result files are older than the bootstrap (no per-video hours): run eval_alerts.py again")
    rnd, d_rec, d_fph = random.Random(seed), [], []
    for _ in range(n):
        s = [vids[rnd.randrange(len(vids))] for _ in vids]
        ra, fa = totals([a[v] for v in s])
        rb, fb = totals([b[v] for v in s])
        d_rec.append(ra - rb)
        d_fph.append(fa - fb)
    q = lambda xs, p: sorted(xs)[min(len(xs) - 1, int(p * len(xs)))]
    full_a, full_b = totals([a[v] for v in vids]), totals([b[v] for v in vids])
    return {"videos": len(vids), "a": full_a, "b": full_b,
            "d_recall": (full_a[0] - full_b[0], q(d_rec, 0.025), q(d_rec, 0.975)),
            "d_fph": (full_a[1] - full_b[1], q(d_fph, 0.025), q(d_fph, 0.975)),
            "a_better": sum(1 for x in d_rec if x > 0) / n}


def main():
    if len(sys.argv) != 7:
        sys.exit(__doc__)
    ja, sa, ma, jb, sb, mb = sys.argv[1:]
    r = paired(load(ja, sa, ma), load(jb, sb, mb))
    print(f"A = {ma} ({ja})\nB = {mb} ({jb})\non {r['videos']} videos\n")
    print(f"A: recall {r['a'][0] * 100:.1f}%  false/h {r['a'][1]:.1f}")
    print(f"B: recall {r['b'][0] * 100:.1f}%  false/h {r['b'][1]:.1f}")
    d, lo, hi = r["d_recall"]
    print(f"\nrecall A - B : {d * 100:+.1f} points, 95% CI {lo * 100:+.1f} .. {hi * 100:+.1f}")
    d, lo, hi = r["d_fph"]
    print(f"false/h A - B: {d:+.1f}, 95% CI {lo:+.1f} .. {hi:+.1f}")
    print(f"A caught more than B in {r['a_better'] * 100:.0f}% of the resamples")
    lo = r["d_recall"][1]
    print("-> " + ("A catches more, beyond noise" if lo > 0 else
                   "B catches more, beyond noise" if r["d_recall"][2] < 0 else
                   "cannot tell them apart on these videos (the interval contains 0)"))


if __name__ == "__main__":
    main()
