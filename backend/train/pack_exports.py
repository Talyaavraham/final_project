"""
Merge public Roboflow "YOLOv8" exports of the same objects into one compact,
leak-free export (a zip in the Roboflow layout), ready for
security_dataset.py --own / the Colab notebook.

    python train/pack_exports.py "C:/.../knife.v2i.yolov8" "C:/.../Knife detection.v1i.yolov8" \
        --out train/security/knives.zip

Why not use the exports as they are:
  * Roboflow writes every training image 3 times (its augmentations). One copy
    is kept: the trainer augments on its own, and 3 near-identical copies only
    make the class look bigger than it is.
  * Public exports are often re-uploads of each other (of the 2,078 images of
    the first knife set, 1,948 are in the second). The same source image
    (same file stem before ".rf.<hash>") is kept once, from the first export
    given, a clean copy (valid / test) before an augmented one (train).
  * An image that is in the validation split of ANY export goes to validation,
    so no validation image has a twin in training. "test" splits are used for
    training: the test that counts is our own benchmark.
  * Polygon labels are turned into boxes.
"""
import argparse
import os
import re
import sys
import zipfile
from collections import Counter, defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from night_dataset import read_names  # noqa: E402

IMG_EXT = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
SPLITS = ("valid", "test", "train")            # clean copies first


def source_key(filename):
    """'abc_jpg.rf.0123abcd.jpg' -> 'abc_jpg': the image before Roboflow's augmentation."""
    stem = os.path.splitext(os.path.basename(filename))[0]
    return re.sub(r"[^\w.-]", "_", re.sub(r"\.rf\.[0-9a-f]+$", "", stem)).lower()


def box_lines(label_path, names, ids):
    """YOLO label file -> lines 'id cx cy w h' with merged class ids; polygons become boxes."""
    lines = []
    if not os.path.exists(label_path):
        return lines
    with open(label_path, encoding="utf-8") as f:
        for row in f:
            p = row.split()
            if len(p) < 5:
                continue
            name = names.get(int(float(p[0])), "").strip().lower()
            if name not in ids:
                continue
            v = [float(x) for x in p[1:]]
            if len(v) > 4:                       # polygon: x1 y1 x2 y2 ...
                xs, ys = v[0::2], v[1::2]
                v = [(min(xs) + max(xs)) / 2, (min(ys) + max(ys)) / 2, max(xs) - min(xs), max(ys) - min(ys)]
            if v[2] <= 0 or v[3] <= 0:
                continue
            lines.append("%d %.6f %.6f %.6f %.6f" % (ids[name], *v))
    return lines


def plan(exports):
    """
    exports: [dir, ...] -> (names, items, stats); items = [(split, key, image path, label lines)],
    split is "train" or "valid", every source image once.
    """
    names, found, stats = [], defaultdict(list), Counter()
    per_export = []
    for d in exports:
        nm = read_names(os.path.join(d, "data.yaml"))
        per_export.append(nm)
        for n in nm.values():
            if n.strip().lower() not in names:
                names.append(n.strip().lower())
    ids = {n: i for i, n in enumerate(names)}
    for e, d in enumerate(exports):
        for split in SPLITS:
            img_dir = os.path.join(d, split, "images")
            if not os.path.isdir(img_dir):
                continue
            for fn in sorted(os.listdir(img_dir)):
                if fn.lower().endswith(IMG_EXT):
                    stats[f"export {e + 1} {split}"] += 1
                    found[source_key(fn)].append((e, SPLITS.index(split), fn, d, split))
    items = []
    for key in sorted(found):
        copies = sorted(found[key])
        split = "valid" if any(c[4] == "valid" for c in copies) else "train"
        e, _, fn, d, src_split = copies[0]
        lbl = os.path.join(d, src_split, "labels", os.path.splitext(fn)[0] + ".txt")
        items.append((split, key, os.path.join(d, src_split, "images", fn), box_lines(lbl, per_export[e], ids)))
        stats[f"kept {split}"] += 1
        stats["copies dropped"] += len(copies) - 1
    return names, items, stats


def pack(exports, out_zip):
    names, items, stats = plan(exports)
    os.makedirs(os.path.dirname(os.path.abspath(out_zip)), exist_ok=True)
    boxes = Counter()
    with zipfile.ZipFile(out_zip, "w", zipfile.ZIP_STORED) as z:      # jpg does not compress
        z.writestr("data.yaml", "train: ../train/images\nval: ../valid/images\n\nnc: %d\nnames: %r\n" % (len(names), names))
        for split, key, img, lines in items:
            z.write(img, f"{split}/images/{key}{os.path.splitext(img)[1].lower()}")
            z.writestr(f"{split}/labels/{key}.txt", "\n".join(lines) + ("\n" if lines else ""))
            for line in lines:
                boxes[names[int(line.split()[0])]] += 1
    return names, stats, boxes


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("exports", nargs="+", help="Roboflow YOLOv8 export folders; the first one wins a duplicate")
    ap.add_argument("--out", required=True, help="zip to write")
    args = ap.parse_args()
    names, stats, boxes = pack(args.exports, args.out)
    for k in sorted(stats):
        print(f"  {k}: {stats[k]}")
    print(f"classes {names}, boxes {dict(boxes)} -> {args.out} ({os.path.getsize(args.out) / 1e6:.0f} MB)")


if __name__ == "__main__":
    main()
