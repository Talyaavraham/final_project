"""
Several detectors as one, for eval/detector_bench.py: merges their per-second
raw files, taking from each file only the classes named for it (the same
"model=cls+cls" idea as LIVE_EXTRA / EXTRA_MODELS). A class named for two
files gets the higher confidence of the two.

    python eval/merge_raw.py OUT.json v1_raw.json=fire+smoke v3_raw.json=gun+knife
    python eval/detector_bench.py GT.json FOLDER --raw OUT.json --classes gun=weapon,knife=knife,fire=fire,smoke=fire

Only videos present in every file are kept.
"""
import json
import sys


def merge(parts):
    """parts: [(raw dict {video: {second: {class: conf}}}, set of classes)] -> one raw dict."""
    videos = set.intersection(*(set(raw) for raw, _ in parts))
    out = {}
    for v in sorted(videos):
        secs = {}
        for raw, classes in parts:
            for t, best in raw[v].items():
                for c, conf in best.items():
                    if c in classes:
                        cell = secs.setdefault(t, {})
                        cell[c] = max(cell.get(c, 0.0), conf)
        out[v] = secs
    return out


def main(argv):
    if len(argv) < 2 or any("=" not in a for a in argv[1:]):
        sys.exit(__doc__)
    parts = []
    for a in argv[1:]:
        path, _, classes = a.rpartition("=")
        parts.append((json.load(open(path)), set(classes.split("+"))))
    out = merge(parts)
    json.dump(out, open(argv[0], "w"))
    print(f"{len(out)} videos -> {argv[0]}")


if __name__ == "__main__":
    main(sys.argv[1:])
