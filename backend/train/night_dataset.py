"""
Build a low-light (night) training set for the detector.

Stock YOLOv8 misses people in dark surveillance footage (see train/README.md,
"night"). Fine-tuning on night images fixes most of it. Two sources:

  * ExDark (Exclusively Dark Image Dataset): 7,363 low-light photos with
    boxes for 12 classes, downloaded automatically from Hugging Face in YOLO
    format (dronefreak/ExDark, Roboflow export, images 640x640).
  * your own frames: `frames` pulls stills out of a video, you label them in
    Roboflow / CVAT / LabelImg and export as "YOLOv8" — a folder with
    data.yaml + train/images + train/labels (valid/ and test/ optional).

Class names are mapped to the COCO names the rest of the system already
understands (People -> person, Motorbike -> motorcycle, ...), so the parser,
RENAME/SKIP tables and the UI need no change.

Usage (from backend/):
  python train/night_dataset.py frames videos/dark.mp4 --every 2
      -> train/night/frames/dark_000042.0.jpg ...   (label these)
  python train/night_dataset.py build --own train/night/own_export
      -> train/night/dataset/{images,labels}/{train,val} + data.yaml
  python train/night_dataset.py build --no-exdark --own X   (own frames only)
  python train/night_dataset.py build --classes person,car --limit 200 (quick)

Then train (Colab: train/colab_train_night.ipynb) and save best.pt as
backend/night.pt; YOLO_MODEL=yolov8n.pt,game.pt,night.pt picks it per video.
"""
import argparse
import os
import random
import shutil
import sys

HF_REPO = "dronefreak/ExDark"

# final class list, COCO names, in this order
CLASSES = ["person", "car", "bus", "bicycle", "motorcycle", "dog", "cat",
           "boat", "bottle", "chair", "cup", "dining table"]

# names found in ExDark / Roboflow / hand-made exports -> COCO name
ALIASES = {
    "people": "person", "person": "person", "pedestrian": "person", "man": "person",
    "woman": "person", "human": "person",
    "car": "car", "vehicle": "car", "taxi": "car", "van": "car",
    "bus": "bus",
    "bicycle": "bicycle", "bike": "bicycle",
    "motorbike": "motorcycle", "motorcycle": "motorcycle", "scooter": "motorcycle",
    "dog": "dog", "cat": "cat", "boat": "boat", "bottle": "bottle",
    "chair": "chair", "cup": "cup",
    "table": "dining table", "dining table": "dining table",
}

IMG_EXT = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


# --------------------------------------------------------------- frames
def extract_frames(videos, out_dir, every=2.0, max_frames=400):
    """Stills every `every` seconds from each video, named <stem>_<t>.jpg."""
    import cv2
    os.makedirs(out_dir, exist_ok=True)
    written = 0
    for path in videos:
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            print("cannot open", path, file=sys.stderr)
            continue
        fps = cap.get(cv2.CAP_PROP_FPS) or 25
        step = max(1, int(round(fps * every)))
        stem = os.path.splitext(os.path.basename(path))[0]
        i = n = 0
        while n < max_frames:
            if not cap.grab():
                break
            if i % step == 0:
                ok, frame = cap.retrieve()
                if not ok:
                    break
                name = f"{stem}_{i / fps:08.1f}.jpg"
                cv2.imwrite(os.path.join(out_dir, name), frame,
                            [cv2.IMWRITE_JPEG_QUALITY, 95])
                n += 1
            i += 1
        cap.release()
        print(f"{path}: {n} frames")
        written += n
    return written


# --------------------------------------------------------------- sources
def read_names(yaml_path):
    """`names` of a data.yaml: list or {id: name} -> {id: name}. No PyYAML needed."""
    try:
        import yaml
        with open(yaml_path, encoding="utf-8") as f:
            names = yaml.safe_load(f).get("names")
    except ImportError:
        names = _names_without_yaml(yaml_path)
    if isinstance(names, dict):
        return {int(k): str(v) for k, v in names.items()}
    return {i: str(n) for i, n in enumerate(names or [])}


def _names_without_yaml(path):
    import ast
    names = None
    with open(path, encoding="utf-8") as f:
        lines = f.read().splitlines()
    for i, line in enumerate(lines):
        if line.startswith("names:"):
            rest = line[len("names:"):].strip()
            if rest:                                  # names: ['a', 'b']
                names = ast.literal_eval(rest)
            else:                                     # names:\n  0: a
                names = {}
                for sub in lines[i + 1:]:
                    if not sub.startswith((" ", "\t")) or ":" not in sub:
                        break
                    k, v = sub.split(":", 1)
                    names[int(k.strip())] = v.strip().strip("'\"")
    return names


def find_splits(src):
    """
    Splits of a YOLO export. Supports Roboflow (train/images) and Ultralytics
    (images/train) layouts. Returns {our_split: [(img, label_or_None), ...]}.
    """
    out = {"train": [], "val": []}
    found = False
    for name, ours in (("train", "train"), ("valid", "val"), ("val", "val"),
                       ("test", None)):
        for img_dir, lbl_dir in ((os.path.join(src, name, "images"), os.path.join(src, name, "labels")),
                                 (os.path.join(src, "images", name), os.path.join(src, "labels", name))):
            if not os.path.isdir(img_dir):
                continue
            found = True
            if ours is None:          # test split: not used for training
                continue
            for fn in sorted(os.listdir(img_dir)):
                if fn.lower().endswith(IMG_EXT):
                    lbl = os.path.join(lbl_dir, os.path.splitext(fn)[0] + ".txt")
                    out[ours].append((os.path.join(img_dir, fn),
                                      lbl if os.path.exists(lbl) else None))
    if not found:                      # flat images/ + labels/, no split
        img_dir, lbl_dir = os.path.join(src, "images"), os.path.join(src, "labels")
        if os.path.isdir(img_dir):
            for fn in sorted(os.listdir(img_dir)):
                if fn.lower().endswith(IMG_EXT):
                    lbl = os.path.join(lbl_dir, os.path.splitext(fn)[0] + ".txt")
                    out["train"].append((os.path.join(img_dir, fn),
                                         lbl if os.path.exists(lbl) else None))
    return out


def class_map(names, keep):
    """{source id: our id} for every source class whose alias is in `keep`."""
    m = {}
    for sid, n in names.items():
        ours = ALIASES.get(n.strip().lower())
        if ours in keep:
            m[sid] = keep.index(ours)
    return m


def convert_label(src_path, cmap):
    """Rewrite a YOLO label file with our ids; boxes of dropped classes removed."""
    lines = []
    if src_path and os.path.exists(src_path):
        with open(src_path, encoding="utf-8") as f:
            for line in f:
                parts = line.split()
                if len(parts) < 5:
                    continue
                sid = int(float(parts[0]))
                if sid in cmap:
                    lines.append(" ".join([str(cmap[sid])] + parts[1:5]))
    return lines


def download_exdark(cache_dir, attempts=6):
    """
    ~14,700 small files. Without a token Hugging Face rate-limits the
    download after a few hundred files (HTTP 429), so: resume + wait + retry.
    A free token (HF_TOKEN env var, or `huggingface-cli login`) lifts the limit.
    """
    import time
    from huggingface_hub import snapshot_download
    from huggingface_hub.errors import HfHubHTTPError
    wait = 60
    for i in range(attempts):
        try:
            path = snapshot_download(HF_REPO, repo_type="dataset", local_dir=cache_dir,
                                     max_workers=4)
            # when the API itself is rate-limited the call returns the partial
            # local copy without raising -> check it is really complete
            n = exdark_train_images(path)
            if n >= EXDARK_MIN_TRAIN:
                return os.path.join(path, "data")
            msg = f"only {n} of ~5874 training images so far"
        except HfHubHTTPError as e:
            if "429" not in str(e):
                raise
            msg = "rate-limited by Hugging Face"
        except (ConnectionError, OSError) as e:       # dropped connection mid-way
            msg = f"connection error ({type(e).__name__})"
        if i == attempts - 1:
            raise SystemExit(f"ExDark download incomplete ({msg}). Set HF_TOKEN "
                             "(free account) and run again — it resumes where it stopped.")
        print(f"{msg}, waiting {wait}s and resuming (set HF_TOKEN to avoid this)…",
              flush=True)
        time.sleep(wait)
        wait = min(wait * 2, 600)


EXDARK_MIN_TRAIN = 5800


def exdark_train_images(path):
    d = os.path.join(path, "data", "train", "images")
    return sum(1 for f in os.listdir(d) if f.lower().endswith(IMG_EXT)) if os.path.isdir(d) else 0


# --------------------------------------------------------------- build
def build(out_dir, sources, keep=CLASSES, val_ratio=0.1, limit=0, repeat=None,
          seed=0):
    """
    sources: [(tag, dir), ...] each a YOLO export with data.yaml.
    repeat: {tag: n} -> copy that source's training images n times (small
            hand-labelled sets next to 6k ExDark images).
    Returns {class: box count}.
    """
    rnd = random.Random(seed)
    repeat = repeat or {}
    if os.path.isdir(out_dir):
        shutil.rmtree(out_dir)
    for split in ("train", "val"):
        os.makedirs(os.path.join(out_dir, "images", split))
        os.makedirs(os.path.join(out_dir, "labels", split))

    counts = {c: 0 for c in keep}
    n_img = {"train": 0, "val": 0}
    for tag, src in sources:
        yaml_path = os.path.join(src, "data.yaml")
        if not os.path.exists(yaml_path):
            raise SystemExit(f"{src}: no data.yaml (export as 'YOLOv8' format)")
        cmap = class_map(read_names(yaml_path), keep)
        if not cmap:
            print(f"{tag}: none of its classes are in {keep}, skipped")
            continue
        splits = find_splits(src)
        if not splits["val"] and splits["train"]:       # make our own val split
            items = splits["train"][:]
            rnd.shuffle(items)
            k = max(1, int(len(items) * val_ratio)) if len(items) > 1 else 0
            splits = {"val": items[:k], "train": items[k:]}
        for split, items in splits.items():
            if limit:
                items = items[:limit]
            times = repeat.get(tag, 1) if split == "train" else 1
            for img, lbl in items:
                lines = convert_label(lbl, cmap)
                for line in lines:
                    counts[keep[int(line.split()[0])]] += times
                base = f"{tag}_{os.path.basename(img)}"
                for r in range(times):
                    stem, ext = os.path.splitext(base)
                    name = stem + (f"_r{r}" if r else "") + ext
                    shutil.copy2(img, os.path.join(out_dir, "images", split, name))
                    with open(os.path.join(out_dir, "labels", split, stem +
                              (f"_r{r}" if r else "") + ".txt"), "w") as f:
                        f.write("\n".join(lines) + ("\n" if lines else ""))
                    n_img[split] += 1
        print(f"{tag}: train {len(splits['train'])}, val {len(splits['val'])}"
              + (f" (x{repeat[tag]})" if tag in repeat else ""))

    with open(os.path.join(out_dir, "data.yaml"), "w", encoding="utf-8") as f:
        f.write(f"path: {os.path.abspath(out_dir)}\ntrain: images/train\nval: images/val\nnames:\n")
        for i, c in enumerate(keep):
            f.write(f"  {i}: {c}\n")
    print(f"images: train {n_img['train']}, val {n_img['val']}")
    print("boxes:", {c: n for c, n in counts.items() if n})
    return counts


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    f = sub.add_parser("frames", help="stills from videos, for labelling")
    f.add_argument("videos", nargs="+")
    f.add_argument("--out", default="train/night/frames")
    f.add_argument("--every", type=float, default=2.0, help="seconds between stills")
    f.add_argument("--max", type=int, default=400, help="stills per video")

    b = sub.add_parser("build", help="merge ExDark + own exports into one dataset")
    b.add_argument("--out", default="train/night/dataset")
    b.add_argument("--own", action="append", default=[], metavar="DIR",
                   help="YOLO export of your labelled frames (repeatable)")
    b.add_argument("--no-exdark", action="store_true")
    b.add_argument("--exdark-cache", default="train/night/exdark")
    b.add_argument("--classes", default=",".join(CLASSES),
                   help="comma-separated subset of: " + ", ".join(CLASSES))
    b.add_argument("--own-repeat", type=int, default=4,
                   help="copies of each own training image (balance vs ExDark)")
    b.add_argument("--limit", type=int, default=0, help="images per split per source (quick test)")
    b.add_argument("--val-ratio", type=float, default=0.1)
    args = ap.parse_args(argv)

    if args.cmd == "frames":
        n = extract_frames(args.videos, args.out, args.every, args.max)
        print(f"{n} frames in {args.out} — label them (Roboflow / CVAT) and export as YOLOv8")
        return

    keep = [c.strip() for c in args.classes.split(",") if c.strip()]
    bad = [c for c in keep if c not in CLASSES]
    if bad:
        raise SystemExit(f"unknown classes {bad}; choose from {CLASSES}")
    sources, repeat = [], {}
    if not args.no_exdark:
        print("downloading ExDark from Hugging Face (~280 MB, cached)…")
        sources.append(("exdark", download_exdark(args.exdark_cache)))
    for i, d in enumerate(args.own):
        tag = f"own{i}"
        sources.append((tag, d))
        repeat[tag] = args.own_repeat
    if not sources:
        raise SystemExit("nothing to build: drop --no-exdark or pass --own DIR")
    build(args.out, sources, keep, args.val_ratio, args.limit, repeat)


if __name__ == "__main__":
    main()
