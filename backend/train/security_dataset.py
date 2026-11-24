"""
Build the training set for the security detector (security.pt):

  * weapons: "Real-time gun detection in CCTV: an open problem" (real CCTV,
    1080p, Pascal VOC xml) -> classes gun (Handgun + Short_rifle), knife.
    eval/data/weapons/weapons_images_2fps.zip, downloaded from Hugging Face
    (jsalazar/US-Real-time-gun-detection-in-CCTV-An-open-problem-dataset).
  * fire + smoke: D-Fire (YOLO labels inside parquet files, Hugging Face
    mirror badsaarow/d-fire) -> classes fire, smoke.
  * more exports in the Roboflow "YOLOv8" layout, optional (--own): your own
    labelled frames (few, so repeated 4x), or a big public set packed with
    train/pack_exports.py (used once), e.g. knives: the CCTV set above has
    only 210 knife boxes.

Output: one YOLO dataset with classes  gun, knife, fire, smoke  (+ person
is NOT trained here: the COCO model keeps that job; the security model is
run next to it, per video, like game.pt / night.pt).

    python train/security_dataset.py                       # everything
    python train/security_dataset.py --limit 300           # quick test
    python train/security_dataset.py --own DIR             # add own export
    python train/security_dataset.py --own A --own B       # several

With --own, data_base.yaml validates on the weapons + D-Fire images alone:
the same validation images as a build without --own, so two trainings can be
compared (public knife photos are easier than CCTV and would flatter it).

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
                "knife": "knife", "knives": "knife", "blade": "knife",
                "fire": "fire", "flame": "fire", "smoke": "smoke"})


class Writer:
    def __init__(self, out, val_ratio=0.1, seed=0):
        self.out, self.val_ratio = out, val_ratio
        self.rnd = random.Random(seed)
        self.n = {"train": 0, "val": 0}
        self.files = {"train": [], "val": []}        # image paths, in the order written
        self.boxes = {c: 0 for c in CLASSES}
        for split in ("train", "val"):
            os.makedirs(os.path.join(out, "images", split), exist_ok=True)
            os.makedirs(os.path.join(out, "labels", split), exist_ok=True)

    def add(self, name, image_bytes, lines, split=None, ext=".jpg"):
        split = split or ("val" if self.rnd.random() < self.val_ratio else "train")
        path = os.path.join(self.out, "images", split, name + ext)
        with open(path, "wb") as f:
            f.write(image_bytes)
        self.files[split].append(os.path.abspath(path))
        with open(os.path.join(self.out, "labels", split, name + ".txt"), "w") as f:
            f.write("\n".join(lines) + ("\n" if lines else ""))
        for line in lines:
            self.boxes[CLASSES[int(line.split()[0])]] += 1
        self.n[split] += 1

    def yaml(self, name="data.yaml", val=None):
        """data.yaml of the whole set; with `val` (image paths), a yaml that validates on those images only."""
        val_entry = "images/val"
        if val is not None:
            val_entry = os.path.splitext(name)[0] + "_val.txt"
            with open(os.path.join(self.out, val_entry), "w", encoding="utf-8") as f:
                f.write("\n".join(val) + "\n")
        with open(os.path.join(self.out, name), "w", encoding="utf-8") as f:
            f.write(f"path: {os.path.abspath(self.out)}\ntrain: images/train\nval: {val_entry}\nnames:\n")
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


# ---------------------------------------------------------------- WeaponDetection (Hugging Face)
# Subh775/WeaponDetection (CC BY 4.0, from Roboflow "weapon-detection" by yolov7test): ~9,600 images,
# 29 classes, COCO boxes [x, y, w, h]. Its many knife boxes are what security.pt lacked.
WEAPONDET_REPO = "Subh775/WeaponDetection"
WD_KNIFE = {"knife", "knife_deploy", "knife_weapon"}
WD_GUN = {"guns", "pistol", "pistols", "handgun", "heavy gun", "heavy_gun", "heavyweapon",
          "rifle", "shotgun", "long guns", "long_guns", "larga"}
# boxes that may hold a gun or a knife without saying which: the image is dropped rather than
# kept with an unlabelled weapon in it (that would teach the model that a knife is background)
WD_AMBIGUOUS = {"weapon", "weapons", "guns perspective", "guns_perspective", "al", "stabbing"}


def weapondet_lines(bboxes, cats, names, w, h):
    """COCO boxes + category ids -> YOLO lines (gun / knife), or None when the image has an ambiguous weapon box.
    Classes that are not objects of ours (person, hand, blood, aggressor, ...) are ignored."""
    lines = []
    for (x, y, bw, bh), c in zip(bboxes, cats):
        n = str(names[int(c)]).strip().lower()
        if n in WD_AMBIGUOUS:
            return None
        cls = "knife" if n in WD_KNIFE else "gun" if n in WD_GUN else None
        if cls is None or bw <= 1 or bh <= 1 or not w or not h:
            continue
        cx, cy = min(max((x + bw / 2) / w, 0), 1), min(max((y + bh / 2) / h, 0), 1)
        lines.append(f"{CLASSES.index(cls)} {cx:.6f} {cy:.6f} {min(bw / w, 1):.6f} {min(bh / h, 1):.6f}")
    return lines


def cctv_look(img, rnd):
    """A photo made to look like a security camera frame: low resolution, blur, heavy JPEG, sometimes grey.
    Boxes are relative, so they stay valid. Returns JPEG bytes."""
    from PIL import Image, ImageFilter
    img = img.convert("RGB")
    w, h = img.size
    f = rnd.uniform(0.2, 0.45)                                     # UCF-Crime is 320x240
    small = img.resize((max(16, int(w * f)), max(16, int(h * f))), Image.BILINEAR)
    if rnd.random() < 0.5:
        small = small.filter(ImageFilter.GaussianBlur(rnd.uniform(0.3, 1.0)))
    img = small.resize((w, h), Image.BILINEAR)
    if rnd.random() < 0.3:
        img = img.convert("L").convert("RGB")                      # night / IR cameras
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=rnd.randint(20, 45))
    return buf.getvalue()


def add_weapondet(writer, dataset, limit=0, cctv=True, rnd=None):
    """dataset: the DatasetDict of datasets.load_dataset(WEAPONDET_REPO). Its validation split goes to val;
    train + test to train, each image twice when cctv=True (as is + cctv_look)."""
    rnd = rnd or random.Random(3)
    stats = {"images": 0, "dropped": 0, "no_box": 0, "cctv": 0}
    for split_name, ds in dataset.items():
        split = "val" if split_name.startswith("val") else "train"
        objs = ds.features["objects"]
        feat = getattr(objs, "feature", objs)
        if isinstance(feat, (list, tuple)):
            feat = feat[0]
        cat_key = next(k for k in feat if k != "bbox" and hasattr(getattr(feat[k], "feature", feat[k]), "names")
                       or k in ("category", "categories", "label", "labels", "class"))
        cf = getattr(feat[cat_key], "feature", feat[cat_key])
        names = cf.names
        for i, r in enumerate(ds):
            o = r["objects"]
            if isinstance(o, list):            # a list of {bbox, category} instead of a dict of lists
                o = {k: [x[k] for x in o] for k in ("bbox", cat_key)}
            img = r["image"]
            w, h = r.get("width") or img.size[0], r.get("height") or img.size[1]
            lines = weapondet_lines(o["bbox"], o[cat_key], names, w, h)
            if lines is None:
                stats["dropped"] += 1
                continue
            if not lines:                      # people / blood only: a few as negatives
                if rnd.random() > 0.2:
                    stats["no_box"] += 1
                    continue
            buf = io.BytesIO()
            img.convert("RGB").save(buf, "JPEG", quality=92)
            stem = f"wd_{split_name}_{i}"
            writer.add(stem, buf.getvalue(), lines, split=split)
            stats["images"] += 1
            if cctv and split == "train" and lines:
                writer.add(stem + "_cctv", cctv_look(img, rnd), lines, split=split)
                stats["cctv"] += 1
            if limit and stats["images"] >= limit:
                break
    print(f"WeaponDetection: {stats}")
    return stats


BIG_EXPORT = 500      # training images; below this an export is "a few own frames" and is repeated


def add_own(writer, src, repeat=None):
    """repeat: copies of every training image. None = 4 for a small export, 1 for a big public one."""
    from night_dataset import class_map
    cmap = class_map(read_names(os.path.join(src, "data.yaml")), CLASSES)
    if not cmap:
        print(f"{src}: none of its classes are in {CLASSES}, skipped")
        return
    splits = find_splits(src)
    if repeat is None:
        repeat = 4 if len(splits["train"]) < BIG_EXPORT else 1
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
    ap.add_argument("--own-repeat", type=int, default=None,
                    help=f"copies of every --own training image (default: 4 for an export under {BIG_EXPORT} training images, else 1)")
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
    base_val = list(w.files["val"])
    for d in args.own:
        add_own(w, d, args.own_repeat)
    w.yaml()
    if args.own:
        w.yaml("data_base.yaml", base_val)
    print(f"images: train {w.n['train']}, val {w.n['val']} | boxes: {w.boxes} -> {args.out}")


if __name__ == "__main__":
    main()
