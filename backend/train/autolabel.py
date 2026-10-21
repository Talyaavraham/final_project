"""
Automatic labelling of the pixel-art game video (ex4_demo) for YOLO training.

The game draws every object in one flat colour, so instead of labelling frames
by hand we segment them by colour and turn the blobs into YOLO boxes:

  character  the only thing with a black outline (besides the HUD text)
  tree       a trunk (dark brown) plus the green canopy above it
  fruit      small red / yellow discs (colour is left to the indexer)
  sun        big yellow disc

Day/night tinting scales every pixel, so each frame is first normalised by the
sky colour. Frames too dark to label are skipped.

    python train/autolabel.py videos/ex4_demo_2025_26_winter.mp4 train/dataset --every 6 --debug

writes dataset/images/{train,val}, dataset/labels/{train,val}, dataset/data.yaml
(and dataset/debug/*.jpg with the boxes drawn when --debug is given).
"""

import argparse
import os
import random

import cv2
import numpy as np

CLASSES = ["character", "tree", "fruit", "sun"]
SKY_REF = np.array([216, 185, 120], np.float32)   # BGR of the sky in daylight

# reference BGR colours (daylight) and tolerance (euclidean distance)
COLORS = {
    "trunk":  ((14, 44, 90), 35),
    "ground": ((63, 110, 194), 60),
    "yellow": ((0, 185, 240), 50),
    "sun":    ((0, 240, 240), 40),
}


def normalise(frame):
    """Undo the day/night tint using the sky in the top-centre of the frame."""
    sky = frame[80:160, 860:1060].reshape(-1, 3).astype(np.float32).mean(0)
    k = float(sky.mean() / SKY_REF.mean())
    if k < 0.12:                      # night: nothing to see
        return None, k
    return np.clip(frame.astype(np.float32) / k, 0, 255).astype(np.uint8), k


def mask_near(img, bgr, tol):
    d = np.linalg.norm(img.astype(np.int16) - np.array(bgr, np.int16), axis=2)
    return (d < tol).astype(np.uint8)


def components(mask, min_area):
    n, _, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    return [(x, y, w, h) for x, y, w, h, a in stats[1:] if a >= min_area]


def label_frame(frame):
    """Returns (list of (class_id, x1, y1, x2, y2)), or None if too dark."""
    img, k = normalise(frame)
    if img is None:
        return None
    H, W = img.shape[:2]
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    boxes = []

    # --- fruit: red discs + yellow discs
    red = ((img[..., 2] > 150) & (img[..., 1] < 70) & (img[..., 0] < 70)).astype(np.uint8)
    yellow = mask_near(img, *COLORS["yellow"])
    for m in (red, yellow):
        for x, y, w, h in components(m, 80):
            if 12 <= w <= 45 and 12 <= h <= 45:
                boxes.append((2, x, y, x + w, y + h))

    # --- sun: big yellow disc (the halo is translucent, we box the core)
    for x, y, w, h in components(mask_near(img, *COLORS["sun"]), 1500):
        if w >= 60 and h >= 60:
            boxes.append((3, x, y, x + w, y + h))

    # --- character: black outline. HUD text (top-left) is black too.
    black = (img.max(2) < 50).astype(np.uint8)
    black[:80, :300] = 0
    black = cv2.dilate(black, np.ones((5, 5), np.uint8))   # join outline parts
    for x, y, w, h in components(black, 300):
        if 25 <= w <= 110 and 45 <= h <= 130:
            boxes.append((0, x + 2, y + 2, x + w - 2, y + h - 2))

    # --- trees: trunk + the canopy above it
    green = ((hsv[..., 0] >= 35) & (hsv[..., 0] <= 85) &
             (hsv[..., 1] > 100) & (hsv[..., 2] > 60)).astype(np.uint8)
    green = cv2.dilate(green, np.ones((9, 9), np.uint8))   # merge leaf squares
    n, lab, stats, _ = cv2.connectedComponentsWithStats(green, connectivity=8)
    trunk = mask_near(img, *COLORS["trunk"])
    trunk = cv2.morphologyEx(trunk, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    nt, tlab, tstats, _ = cv2.connectedComponentsWithStats(trunk, connectivity=8)
    trunks = []
    for i in range(1, nt):
        x, y, w, h, a = tstats[i]
        if a < 400 or w > 80:
            continue
        # dark ground squares can touch the trunk: cut the component where a
        # row gets wider than a trunk
        rows = (tlab[y:y + h, x:x + w] == i).sum(1)
        wide = np.where(rows > 60)[0]
        if len(wide):
            h = int(wide[0])
        if h >= 60:
            trunks.append([x, y, x + w, y + h])
    # a leaf square can split one trunk in two: merge pieces that overlap in x
    trunks.sort()
    merged = []
    for t in trunks:
        if merged and t[0] <= merged[-1][2] and merged[-1][0] <= t[2]:
            m = merged[-1]
            merged[-1] = [min(m[0], t[0]), min(m[1], t[1]), max(m[2], t[2]), max(m[3], t[3])]
        else:
            merged.append(t)
    ground = mask_near(img, *COLORS["ground"])
    for x1, y1, x2, y2 in merged:
        cx = (x1 + x2) // 2
        # the trunk ends where the ground surface starts (dark ground squares
        # under the trunk have the trunk's colour)
        strip = ground[y1:, max(0, cx - 50):cx + 50].sum(1)
        top = np.where(strip > 20)[0]
        if len(top):
            y2 = min(y2, y1 + int(top[0]) + 12)
        # canopy = green components touching a window around the trunk top
        win = lab[max(0, y1 - 40):y2, max(0, cx - 60):cx + 60]
        ids = {int(i) for i in np.unique(win) if i}
        for i in ids:
            gx, gy, gw, gh = stats[i][:4]
            # overlapping canopies: keep the part near this trunk
            x1 = min(x1, max(gx, cx - 170))
            x2 = max(x2, min(gx + gw, cx + 170))
            y1 = min(y1, gy)
        if y2 - y1 >= 80:
            boxes.append((1, x1, y1, x2, y2))

    return [(c, max(0, x1), max(0, y1), min(W, x2), min(H, y2)) for c, x1, y1, x2, y2 in boxes]


def draw(frame, boxes):
    out = frame.copy()
    palette = [(255, 255, 255), (0, 200, 0), (0, 0, 255), (0, 220, 220)]
    for c, x1, y1, x2, y2 in boxes:
        cv2.rectangle(out, (x1, y1), (x2, y2), palette[c], 2)
        cv2.putText(out, CLASSES[c], (x1, max(12, y1 - 4)), 0, 0.5, palette[c], 1)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("out")
    ap.add_argument("--every", type=int, default=6, help="use every Nth frame")
    ap.add_argument("--val", type=float, default=0.2)
    ap.add_argument("--debug", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    for split in ("train", "val"):
        os.makedirs(os.path.join(a.out, "images", split), exist_ok=True)
        os.makedirs(os.path.join(a.out, "labels", split), exist_ok=True)
    if a.debug:
        os.makedirs(os.path.join(a.out, "debug"), exist_ok=True)

    rng = random.Random(a.seed)
    cap = cv2.VideoCapture(a.video)
    i = kept = skipped = 0
    counts = [0] * len(CLASSES)
    while True:
        ok = cap.grab()
        if not ok:
            break
        if i % a.every == 0:
            _, frame = cap.retrieve()
            boxes = label_frame(frame)
            if boxes is None or not boxes:
                skipped += 1
            else:
                split = "val" if rng.random() < a.val else "train"
                name = f"f{i:05d}"
                H, W = frame.shape[:2]
                cv2.imwrite(os.path.join(a.out, "images", split, name + ".jpg"), frame,
                            [cv2.IMWRITE_JPEG_QUALITY, 92])
                with open(os.path.join(a.out, "labels", split, name + ".txt"), "w") as f:
                    for c, x1, y1, x2, y2 in boxes:
                        counts[c] += 1
                        f.write(f"{c} {(x1 + x2) / 2 / W:.6f} {(y1 + y2) / 2 / H:.6f} "
                                f"{(x2 - x1) / W:.6f} {(y2 - y1) / H:.6f}\n")
                if a.debug and kept % 10 == 0:
                    cv2.imwrite(os.path.join(a.out, "debug", name + ".jpg"), draw(frame, boxes))
                kept += 1
        i += 1

    with open(os.path.join(a.out, "data.yaml"), "w") as f:
        f.write(f"path: {os.path.abspath(a.out)}\ntrain: images/train\nval: images/val\n"
                f"names:\n" + "".join(f"  {i}: {n}\n" for i, n in enumerate(CLASSES)))
    print(f"frames labelled: {kept}, skipped (dark/empty): {skipped}")
    print("boxes per class:", dict(zip(CLASSES, counts)))


if __name__ == "__main__":
    main()
