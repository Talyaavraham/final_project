r"""
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

Exports made of video frames (CCTV datasets) need three more things:

    python train/pack_exports.py "C:/.../CCTV Weapon Detect.v16i.yolov8" --out train/security/guns_cctv.zip \
        --exclude-gt eval/ground_truth_uca.json eval/ground_truth_heldout.json --group "^([a-z]+\d+_x264)_" --upright

  * --exclude-gt: frames of the videos of our benchmarks are left out. Many
    public CCTV sets are cut from UCF-Crime; training on frames of a video and
    then measuring on that video says nothing.
  * --group REGEX: the frames of one video (group 1 of the regex, matched on
    the lower-case file name) go to one split. Roboflow splits frame by frame,
    so a video's frames are on both sides and the validation score flatters.
    Which videos validate is decided by a hash of the name (15% of them).
  * --upright: Roboflow's "90-degree rotation" augmentation leaves 3 of 4
    training copies on their side or upside down, which no camera films.
    The kept training copy is turned back: the rotation in which a person
    detector (yolov8n) is surest of the people is the upright one. When that
    is not clear for one copy, the next copy of the same image is tried; an
    image with no clear answer in any copy (nobody in it, a camera that looks
    straight down) is kept as it is. Measured on 300 validation / test images
    turned at random: an answer for 64%, and 93% of the answers right.
    Validation / test copies are not augmented and stay as they are.
"""
import argparse
import hashlib
import json
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


VAL_GROUPS = 15         # % of the groups (videos) that validate, with --group
UPRIGHT_MARGIN = 1.5    # the best rotation must beat the next one by this factor (measured: see train/README.md)


def excluded_videos(gt_files):
    """ground-truth files -> the lower-case video names without extension ("robbery001_x264")."""
    out = set()
    for g in gt_files or []:
        with open(g, encoding="utf-8") as f:
            out |= {os.path.splitext(v)[0].lower() for v in json.load(f)["videos"]}
    return out


def turn_lines(lines, k):
    """label lines after the image was turned k times 90 degrees counter-clockwise (np.rot90)."""
    out = []
    for line in lines:
        c, x, y, w, h = line.split()
        x, y, w, h = float(x), float(y), float(w), float(h)
        for _ in range(k % 4):
            x, y, w, h = y, 1 - x, h, w
        out.append("%s %.6f %.6f %.6f %.6f" % (c, x, y, w, h))
    return out


def upright_turns(img, people, margin=UPRIGHT_MARGIN):
    """
    How many 90-degree counter-clockwise turns put the image upright, or None
    when it cannot be told: a person detector is far surer of people who stand
    on their feet, so the rotation with the highest summed confidence wins,
    if it wins clearly.
    """
    import numpy as np
    rots = [np.ascontiguousarray(np.rot90(img, k)) for k in range(4)]
    scores = [float(r.boxes.conf.sum()) for r in people.predict(rots, classes=[0], conf=0.25, imgsz=640, verbose=False)]
    k = max(range(4), key=scores.__getitem__)
    second = sorted(scores)[-2]
    return k if scores[k] > 0 and scores[k] >= margin * second else None


def plan(exports, exclude=(), group=None):
    """
    exports: [dir, ...] -> (names, items, stats);
    items = [(split, key, image path, label lines, source split, other copies [(image path, label lines)])],
    split is "train" or "valid", every source image once.
    exclude: video names whose frames are left out; group: compiled regex, see the module text.
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
        if any(key.startswith(v + "_") or key == v for v in exclude):
            stats["left out (benchmark videos)"] += 1
            continue
        split = "valid" if any(c[4] == "valid" for c in copies) else "train"
        m = group.match(key) if group else None
        if m:                                       # every frame of a video in the same split
            split = "valid" if int(hashlib.md5(m.group(1).encode()).hexdigest(), 16) % 100 < VAL_GROUPS else "train"
            stats["frames of grouped videos"] += 1
        e, _, fn, d, src_split = copies[0]
        lbl = os.path.join(d, src_split, "labels", os.path.splitext(fn)[0] + ".txt")
        others = [(os.path.join(d2, s2, "images", f2),
                   box_lines(os.path.join(d2, s2, "labels", os.path.splitext(f2)[0] + ".txt"), per_export[e2], ids))
                  for e2, _, f2, d2, s2 in copies[1:] if (e2, s2) == (e, src_split)]
        items.append((split, key, os.path.join(d, src_split, "images", fn), box_lines(lbl, per_export[e], ids),
                      src_split, others))
        stats[f"kept {split}"] += 1
        stats["copies dropped"] += len(copies) - 1
    return names, items, stats


def pack(exports, out_zip, exclude=(), group=None, upright=None):
    """upright: a function image -> turns (0-3) or None; applied to copies that come from a "train" split."""
    names, items, stats = plan(exports, exclude, group)
    os.makedirs(os.path.dirname(os.path.abspath(out_zip)), exist_ok=True)
    boxes = Counter()
    with zipfile.ZipFile(out_zip, "w", zipfile.ZIP_STORED) as z:      # jpg does not compress
        z.writestr("data.yaml", "train: ../train/images\nval: ../valid/images\n\nnc: %d\nnames: %r\n" % (len(names), names))
        for split, key, img, lines, src_split, others in items:
            name = f"{split}/images/{key}{os.path.splitext(img)[1].lower()}"
            turned = None
            if upright and src_split == "train":
                import cv2
                import numpy as np
                for c_img, c_lines in [(img, lines)] + others:
                    pixels = cv2.imread(c_img)
                    k = upright(pixels) if pixels is not None else None
                    if k is not None:
                        turned = (c_img, c_lines, pixels, k)
                        break
                stats["orientation unclear, kept as it is" if turned is None else
                      "turned upright" if turned[3] else "already upright"] += 1
            if turned and turned[3]:
                c_img, c_lines, pixels, k = turned
                lines = turn_lines(c_lines, k)
                ok, buf = cv2.imencode(".jpg", np.ascontiguousarray(np.rot90(pixels, k)), [cv2.IMWRITE_JPEG_QUALITY, 92])
                z.writestr(os.path.splitext(name)[0] + ".jpg", buf.tobytes())
            elif turned:
                img, lines = turned[0], turned[1]
                z.write(img, name)
            else:
                z.write(img, name)
            z.writestr(f"{split}/labels/{key}.txt", "\n".join(lines) + ("\n" if lines else ""))
            for line in lines:
                boxes[names[int(line.split()[0])]] += 1
    return names, stats, boxes


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("exports", nargs="+", help="Roboflow YOLOv8 export folders; the first one wins a duplicate")
    ap.add_argument("--out", required=True, help="zip to write")
    ap.add_argument("--exclude-gt", nargs="*", default=[], help="ground-truth files: frames of their videos are left out")
    ap.add_argument("--group", help="regex on the lower-case file name; group 1 = the video the frame comes from")
    ap.add_argument("--upright", action="store_true", help="turn rotated training copies back (needs ultralytics)")
    args = ap.parse_args()
    upright = None
    if args.upright:
        from ultralytics import YOLO
        people = YOLO(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "yolov8n.pt"))
        upright = lambda img: upright_turns(img, people)
    names, stats, boxes = pack(args.exports, args.out, excluded_videos(args.exclude_gt),
                               re.compile(args.group) if args.group else None, upright)
    for k in sorted(stats):
        print(f"  {k}: {stats[k]}")
    print(f"classes {names}, boxes {dict(boxes)} -> {args.out} ({os.path.getsize(args.out) / 1e6:.0f} MB)")


if __name__ == "__main__":
    main()
