"""
One perception cache out of two runs of live/perceive.py over the SAME videos at the same fps, taking
each class from the run whose detector is better at it. This is what the live system runs now
(security.pt for fire / smoke, security_v3.pt for guns / knives, live_engine._default_extra), so the
rules and the learned model can be measured on it without analysing the videos a third time.

    python live/merge_perception.py --base DIR_V3 --take fire,smoke --from DIR_V1 --out DIR_COMBO

Every frame keeps the people and the objects of --base, except the --take classes, which come from
--from (frames are matched by time). Videos missing from --from are skipped.
"""
import argparse
import glob
import json
import os


def merge(base_path, from_path, take, out_path):
    with open(base_path, encoding="utf-8") as f:
        head = json.loads(f.readline())
        base = [json.loads(l) for l in f if l.strip()]
    with open(from_path, encoding="utf-8") as f:
        head2 = json.loads(f.readline())
        other = {round(fr["t"], 2): fr for fr in (json.loads(l) for l in f if l.strip())}
    if head.get("fps") != head2.get("fps"):
        raise ValueError(f"{base_path}: fps {head.get('fps')} vs {head2.get('fps')}")
    head["models"] = {"base": head.get("models"), "from": head2.get("models"), "take": sorted(take)}
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(json.dumps(head) + "\n")
        for fr in base:
            o = other.get(round(fr["t"], 2), {"objects": []})
            fr["objects"] = [x for x in fr["objects"] if x["cls"] not in take] + \
                            [x for x in o["objects"] if x["cls"] in take]
            f.write(json.dumps(fr) + "\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base", required=True)
    ap.add_argument("--from", dest="src", required=True)
    ap.add_argument("--take", required=True, help="classes taken from --from, comma separated")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    take = {c.strip() for c in args.take.split(",") if c.strip()}
    os.makedirs(args.out, exist_ok=True)
    n = 0
    for p in sorted(glob.glob(os.path.join(args.base, "*.jsonl"))):
        q = os.path.join(args.src, os.path.basename(p))
        if os.path.isfile(q):
            merge(p, q, take, os.path.join(args.out, os.path.basename(p)))
            n += 1
    print(f"{n} videos -> {args.out}")


if __name__ == "__main__":
    main()
