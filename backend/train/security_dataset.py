"""
Build the training set for the security detector (security.pt):

  * weapons: "Real-time gun detection in CCTV: an open problem" (real CCTV,
    1080p, Pascal VOC xml) -> classes gun (Handgun + Short_rifle), knife.
    eval/data/weapons/weapons_images_2fps.zip, downloaded from Hugging Face
    (jsalazar/US-Real-time-gun-detection-in-CCTV-An-open-problem-dataset).
  * fire + smoke: D-Fire (YOLO labels inside parquet files, Hugging Face
    mirror badsaarow/d-fire) -> classes fire, smoke.
  * your own labelled frames (Roboflow "YOLOv8" export), optional.

Output: one YOLO dataset with classes  gun, knife, fire, smoke  (+ person
is NOT trained here: the COCO model keeps that job; the security model is
run next to it, per video, like game.pt / night.pt).

    python train/security_dataset.py                       # everything
    python train/security_dataset.py --limit 300           # quick test
    python train/security_dataset.py --own DIR             # add own export

Negatives (images with no box) are kept, capped by --neg-ratio.
"""
import argparse
import io
import os
import random
import re
import shutil
import sys
import zipfile

HERE = os.path.dirname(os.path.abspath(__file__))
BACKEND = os.path.dirname(HERE)
WEAPONS_ZIP = os.path.join(BACKEND, "eval", "data", "weapons", "weapons_images_2fps.zip")
DFIRE_DIR = os.path.join(HERE, "security", "dfire", "data")
OUT_DEFAULT = os.path.join(HERE, "security", "dataset")

CLASSES = ["gun", "knife", "fire", "smoke"]
WEAPON_MAP = {"handgun": "gun", "short_rifle": "gun", "rifle": "gun", "pistol": "gun",
              "gun": "gun", "knife": "knife"}
DFIRE_MAP = {0: "smoke", 1: "fire"}          # D-Fire convention (verified on samples)

sys.path.insert(0, HERE)
from night_dataset import ALIASES, find_splits, read_names, convert_label  # noqa: E402

ALIASES.update({"gun": "gun", "pistol": "gun", "handgun": "gun", "rifle": "gun", "weapon": "gun",
                "fire": "fire", "flame": "fire", "smoke": "smoke"})


class Writer:
    def __init__(self, out, val_ratio=0.1, seed=0):
        self.out, self.val_ratio = out, val_ratio
        self.rnd = random.Random(seed)
        self.n = {"train": 0, "val": 0}
        self.boxes = {c: 0 for c in CLASSES}
        for split in ("train", "val"):
            os.makedirs(os.path.join(out, "images", split), exist_ok=True)
            os.makedirs(os.path.join(out, "labels", split), exist_ok=True)

    def add(self, name, image_bytes, lines, split=None, ext=".jpg"):
        split = split or ("val" if self.rnd.random() < self.val_ratio else "train")
        with open(os.path.join(self.out, "images", split, name + ext), "wb") as f:
            f.write(image_bytes)
        with open(os.path.join(self.out, "labels", split, name + ".txt"), "w") as f:
            f.write("\n".join(lines) + ("\n" if lines else ""))
        for line in lines:
            self.boxes[CLASSES[int(line.split()[0])]] += 1
        self.n[split] += 1

    def yaml(self):
        with open(os.path.join(self.out, "data.yaml"), "w", encoding="utf-8") as f:
            f.write(f"path: {os.path.abspath(self.out)}\ntrain: images/train\nval: images/val\nnames:\n")
            for i, c in enumerate(CLASSES):
                f.write(f"  {i}: {c}\n")


def voc_to_yolo(xml_text):
    """Pascal VOC xml -> YOLO lines with our class ids (unknown names dropped)."""
    w = int(re.search(r"<width>(\d+)</width>", xml_text).group(1))
    h = int(re.search(r"<height>(\d+)</height>", xml_text).group(1))
    lines = []
    for m in re.finditer(r"<object>(.*?)</object>", xml_text, re.S):
        o = m.group(1)
        name = re.search(r"<name>(.*?)</name>", o).group(1).strip().lower()
        cls = WEAPON_MAP.get(name)
        if cls is None:
            continue
        x1, y1, x2, y2 = (float(re.search(rf"<{k}>([\d.]+)</{k}>", o).group(1))
                          for k in ("xmin", "ymin", "xmax", "ymax"))
        if x2 <= x1 or y2 <= y1:
            continue
        lines.append(f"{CLASSES.index(cls)} {(x1+x2)/2/w:.6f} {(y1+y2)/2/h:.6f} {(x2-x1)/w:.6f} {(y2-y1)/h:.6f}")
    return lines


def add_weapons(writer, limit=0, neg_ratio=0.5, rnd=None):
    rnd = rnd or random.Random(1)
    z = zipfile.ZipFile(WEAPONS_ZIP)
    xmls = sorted(n for n in z.namelist() if n.endswith(".xml"))
    pos, neg = [], []
    for n in xmls:
        lines = voc_to_yolo(z.read(n).decode("utf-8", "replace"))
        (pos if lines else neg).append((n, lines))
    rnd.shuffle(neg)
    neg = neg[:int(len(pos) * neg_ratio)]
    items = pos + neg
    rnd.shuffle(items)
    if limit:
        items = items[:limit]
    for n, lines in items:
        jpg = n[:-4] + ".jpg"
        if jpg not in z.NameToInfo:
            continue
        writer.add("weap_" + os.path.basename(n)[:-4], z.read(jpg), lines)
    print(f"weapons: {len(pos)} with boxes + {len(neg)} negatives -> {len(items)} images")


def add_dfire(writer, limit=0, neg_ratio=0.3, rnd=None):
    import glob
    import pyarrow.parquet as pq
    rnd = rnd or random.Random(2)
    files = sorted(glob.glob(os.path.join(DFIRE_DIR, "*.parquet")))
    if not files:
        print("D-Fire: no parquet files found, skipped")
        return
    n_pos = n_neg = 0
    for f in files:
        split = "val" if os.path.basename(f).startswith("test") else "train"
        t = pq.read_table(f, columns=["image", "label", "filename"])
        for batch in t.to_batches(512):
            for r in batch.to_pylist():
                lines = []
                for line in (r["label"] or "").strip().splitlines():
                    p = line.split()
                    if len(p) < 5:
                        continue
                    cls = DFIRE_MAP.get(int(float(p[0])))
                    if cls:
                        lines.append(" ".join([str(CLASSES.index(cls))] + p[1:5]))
                if not lines:
                    # keep a share of the empty images as hard negatives; the
                    # val split (D-Fire's test set) keeps fewer so epochs stay short
                    if rnd.random() > (neg_ratio if split == "train" else neg_ratio / 3):
                        continue
                    n_neg += 1
                else:
                    n_pos += 1
                stem = "dfire_" + os.path.splitext(r["filename"])[0]
                ext = os.path.splitext(r["image"].get("path") or r["filename"])[1] or ".jpg"
                writer.add(stem, r["image"]["bytes"], lines, split=split, ext=ext)
                if limit and n_pos + n_neg >= limit:
                    print(f"D-Fire: {n_pos} with boxes + {n_neg} negatives (limit)")
                    return
    print(f"D-Fire: {n_pos} with boxes + {n_neg} negatives")


def add_own(writer, src, repeat=4):
    from night_dataset import class_map
    cmap = class_map(read_names(os.path.join(src, "data.yaml")), CLASSES)
    if not cmap:
        print(f"{src}: none of its classes are in {CLASSES}, skipped")
        return
    splits = find_splits(src)
    n = 0
    for split, items in splits.items():
        for img, lbl in items:
            lines = convert_label(lbl, cmap)
            with open(img, "rb") as f:
                data = f.read()
            for r in range(repeat if split == "train" else 1):
                writer.add(f"own_{os.path.splitext(os.path.basename(img))[0]}_r{r}", data, lines,
                           split=split, ext=os.path.splitext(img)[1])
                n += 1
    print(f"own ({src}): {n} images written")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=OUT_DEFAULT)
    ap.add_argument("--own", action="append", default=[], metavar="DIR")
    ap.add_argument("--no-weapons", action="store_true")
    ap.add_argument("--no-fire", action="store_true")
    ap.add_argument("--limit", type=int, default=0, help="images per source (quick test)")
    args = ap.parse_args()
    if os.path.isdir(args.out):
        shutil.rmtree(args.out)
    w = Writer(args.out)
    if not args.no_weapons:
        add_weapons(w, args.limit)
    if not args.no_fire:
        add_dfire(w, args.limit)
    for d in args.own:
        add_own(w, d)
    w.yaml()
    print(f"images: train {w.n['train']}, val {w.n['val']} | boxes: {w.boxes} -> {args.out}")


if __name__ == "__main__":
    main()
