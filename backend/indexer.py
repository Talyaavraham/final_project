"""
Video analysis + indexing.

  video file -> YOLO detection + ByteTrack tracking (SAMPLE_FPS frames/s)
             -> colour per detection
             -> detections written to SQLite in batches (nothing is lost if
                the server dies after an hour of indexing)
             -> per-track segments (speed, entering/exiting, best box)
             -> optional CLIP embedding of each track's best crop
             -> tracks that look alike are merged into one entity

Models (env YOLO_MODEL, one name or a comma-separated list — with a list the
model that finds the most in 20 probe frames indexes the video):
  yolov8n.pt / yolov8s.pt / yolov8m.pt   closed set — the 80 COCO classes
  yolov8s-worldv2.pt / yolov8m-worldv2.pt  open vocabulary — any text label,
                                          we index the OPEN_VOCAB list below
  train/runs/<name>/weights/best.pt      a model fine-tuned on your own classes
                                          (see train/README.md); class names
                                          come from the weights file
"""

import os
import time
from collections import Counter

import cv2
import numpy as np

import db as database
import embeddings
from search import merge_entities, track_segments

# one model, or a comma-separated list: each video is indexed by the model
# that finds the most in a quick probe (e.g. "yolov8n.pt,game.pt")
MODEL_NAME = os.environ.get("YOLO_MODEL", "yolov8n.pt")
MODEL_NAMES = [m.strip() for m in MODEL_NAME.split(",") if m.strip()]
PROBE_FRAMES = 20           # frames sampled per model when choosing a model
CONF_THRESHOLD = float(os.environ.get("CONF_THRESHOLD", "0.3"))
SAMPLE_FPS = float(os.environ.get("SAMPLE_FPS", "3"))
IMGSZ = int(os.environ.get("IMGSZ", "640"))   # detector input size; 800+ for small objects
FLUSH_EVERY = 2000          # detections per DB write
# ByteTrack: no camera-motion compensation step, so it is faster than the
# default BoT-SORT and does not need texture to work on
TRACKER = os.environ.get("TRACKER", "bytetrack.yaml")
MAX_CROP = 224              # px, crops kept for CLIP

# class names that we rename (the rest keep the model's name, lower-cased).
# COCO: the three bag classes. Open Images (yolov8s-oiv7.pt): people classes.
RENAME = {"backpack": "bag", "handbag": "bag", "suitcase": "bag",
          "man": "person", "woman": "person", "boy": "person", "girl": "person",
          "human": "person", "land vehicle": "car", "taxi": "car", "van": "car"}
# body parts (Open Images) would add a second box on every person
SKIP = {"human face", "human body", "human head", "human hair", "human arm",
        "human hand", "human leg", "human foot", "human eye", "human mouth",
        "human nose", "human ear", "human beard", "clothing", "footwear"}

# classes indexed by a YOLO-World model — COCO plus things people ask about
# in street / security footage. Override with env OPEN_VOCAB="tree,apple,..."
OPEN_VOCAB = [
    "person", "bicycle", "car", "motorcycle", "bus", "truck", "train", "boat",
    "dog", "cat", "bird", "horse", "bag", "umbrella", "stroller", "shopping cart",
    "scooter", "skateboard", "wheelchair", "helmet", "hat", "cell phone",
    "bottle", "ball", "bench", "traffic light", "stop sign", "fire hydrant",
    "chair", "table", "laptop", "box", "ladder", "weapon",
]
if os.environ.get("OPEN_VOCAB"):
    OPEN_VOCAB = [w.strip() for w in os.environ["OPEN_VOCAB"].split(",") if w.strip()]


# ------------------------------------------------------------ colour name
HUE_NAMES = [  # (upper bound of OpenCV hue 0-180, name)
    (7, "red"), (20, "orange"), (34, "yellow"), (85, "green"),
    (130, "blue"), (150, "purple"), (166, "pink"), (180, "red"),
]


def dominant_color(crop_bgr) -> str:
    """Every pixel votes for a colour name; the most common name wins."""
    if crop_bgr is None or crop_bgr.size == 0:
        return "unknown"
    h0, w0 = crop_bgr.shape[:2]
    crop = crop_bgr[int(h0 * .15):int(h0 * .85), int(w0 * .2):int(w0 * .8)]
    if crop.size == 0:
        crop = crop_bgr
    scale = 48 / max(crop.shape[:2])
    if scale < 1:
        crop = cv2.resize(crop, None, fx=scale, fy=scale,
                          interpolation=cv2.INTER_AREA)
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV).reshape(-1, 3).astype(int)
    h, s, v = hsv[:, 0], hsv[:, 1], hsv[:, 2]

    names = np.empty(len(h), dtype=object)
    chroma = (s >= 50) & (v >= 50)
    for hi, name in reversed(HUE_NAMES):
        names[chroma & (h <= hi)] = name
    names[chroma & (names == "orange") & (v < 150)] = "brown"
    names[v < 50] = "black"
    gray = ~chroma & (v >= 50)
    names[gray & (v >= 170) & (s < 40)] = "white"
    names[gray & ~((v >= 170) & (s < 40))] = "gray"
    return Counter(names.tolist()).most_common(1)[0][0]


# --------------------------------------------------------------- detector
_models: dict = {}


def load_model(name: str = None):
    """YOLO model by file name, loaded once per process."""
    name = name or MODEL_NAMES[0]
    if name not in _models:
        from ultralytics import YOLO
        model = YOLO(name)
        if "world" in name.lower():
            model.set_classes(OPEN_VOCAB)
        _models[name] = model
    return _models[name]


def class_name(model, cls_id: int) -> str:
    raw = str(model.names[cls_id]).lower()
    return RENAME.get(raw, raw)


def probe_score(model, frames, conf: float = 0.5) -> float:
    """Sum of confidences of what the model finds in the frames (>= conf)."""
    score = 0.0
    for frame in frames:
        res = model.predict(frame, verbose=False, conf=conf, imgsz=IMGSZ)[0]
        for box in res.boxes or []:
            if class_name(model, int(box.cls[0])) not in SKIP:
                score += float(box.conf[0])
    return score


def sample_frames(path: str, n: int = PROBE_FRAMES):
    cap = cv2.VideoCapture(path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    frames = []
    for i in range(n):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(total * (i + 0.5) / n))
        ok, frame = cap.read()
        if ok:
            frames.append(frame)
    cap.release()
    return frames


def choose_model(path: str, names=None, status: dict = None):
    """
    The model that finds the most in PROBE_FRAMES frames of the video.
    One model configured -> no probe. Ties go to the first name in the list.
    Returns (name, model).
    """
    names = names or MODEL_NAMES
    if len(names) == 1:
        return names[0], load_model(names[0])
    frames = sample_frames(path)
    best = None
    for name in names:
        if status is not None:
            status["stage"] = f"trying {name}"
        model = load_model(name)
        score = probe_score(model, frames)
        if best is None or score > best[0]:
            best = (score, name, model)
    return best[1], best[2]


def _shrink(img, max_side=MAX_CROP):
    s = max_side / max(img.shape[:2])
    if s < 1:
        img = cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    return img


# ----------------------------------------------------------------- index
def index_video(path: str, name: str, sha1: str = "", status: dict = None,
                db_path: str = None, model=None, sample_fps: float = SAMPLE_FPS):
    """
    Index one video. `status` (a dict) is updated in place with
    {"status": "indexing"|"ready"|"error: ...", "progress": 0..1, "stage": ...}
    `model` lets tests inject a fake detector.
    """
    status = status if status is not None else {}
    status.update(status="indexing", progress=0.0, stage="loading model")
    conn = database.connect(db_path)
    try:
        model_name = MODEL_NAME
        if model is None:
            model_name, model = choose_model(path, status=status)
            status["stage"] = "detecting"
        status["model"] = model_name
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            raise RuntimeError("cannot open video (unsupported codec?)")
        fps = cap.get(cv2.CAP_PROP_FPS) or 25
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
        step = max(1, int(round(fps / sample_fps)))

        database.delete_video(conn, name)
        conn.commit()

        rows, W, H = [], 0, 0
        best_crop = {}          # (track_id, cls) -> (conf, jpeg bytes)
        untracked_id = -1
        frame_i = 0
        status["stage"] = "detecting"
        while True:
            if not cap.grab():          # grab() skips decoding unused frames
                break
            if frame_i % step == 0:
                ok, frame = cap.retrieve()
                if not ok:
                    break
                H, W = frame.shape[:2]
                t = frame_i / fps
                res = model.track(frame, persist=True, verbose=False,
                                  conf=CONF_THRESHOLD, tracker=TRACKER,
                                  imgsz=IMGSZ)[0]
                for box in res.boxes or []:
                    cls = class_name(model, int(box.cls[0]))
                    if cls in SKIP:
                        continue
                    x1, y1, x2, y2 = (int(round(float(c))) for c in box.xyxy[0])
                    x1, y1 = max(0, x1), max(0, y1)
                    x2, y2 = min(W, x2), min(H, y2)
                    if x2 <= x1 or y2 <= y1:
                        continue
                    if cls == "person":     # shirt = upper body of the box
                        bh = y2 - y1
                        color_crop = frame[y1 + int(bh * .15):y1 + int(bh * .55), x1:x2]
                    else:
                        color_crop = frame[y1:y2, x1:x2]
                    if box.id is not None:
                        tid = int(box.id[0])
                    else:
                        tid = untracked_id
                        untracked_id -= 1
                    conf = float(box.conf[0])
                    rows.append((name, t, tid, cls, conf,
                                 100 * x1 / W, 100 * y1 / H,
                                 100 * (x2 - x1) / W, 100 * (y2 - y1) / H,
                                 dominant_color(color_crop)))
                    key = (tid, cls)
                    if embeddings.available() and conf > best_crop.get(key, (0,))[0]:
                        ok_enc, buf = cv2.imencode(
                            ".jpg", _shrink(frame[y1:y2, x1:x2]),
                            [cv2.IMWRITE_JPEG_QUALITY, 85])
                        if ok_enc:
                            best_crop[key] = (conf, buf.tobytes())
                if len(rows) >= FLUSH_EVERY:
                    database.insert_detections(conn, rows)
                    conn.commit()
                    rows = []
                if total:
                    status["progress"] = min(0.9, 0.9 * frame_i / total)
            frame_i += 1
        cap.release()
        database.insert_detections(conn, rows)
        conn.commit()
        duration = frame_i / fps

        # ---- segments --------------------------------------------------
        status.update(stage="building segments", progress=0.9)
        dets = database.load_detections(conn, name)
        segments = track_segments(dets, duration, W or 16, H or 9)

        # ---- embeddings + entity merge --------------------------------
        if best_crop and segments:
            status.update(stage="computing embeddings", progress=0.93)
            keys = list(best_crop)
            crops = [cv2.imdecode(np.frombuffer(best_crop[k][1], np.uint8),
                                  cv2.IMREAD_COLOR) for k in keys]
            vecs = embeddings.embed_images(crops)
            if vecs is not None:
                by_key = dict(zip(keys, vecs))
                for s in segments:
                    v = by_key.get((s["track_id"], s["cls"]))
                    s["embedding_vec"] = v
                    s["embedding"] = embeddings.to_blob(v)
                merge_entities(segments)

        database.replace_segments(conn, name, segments)
        database.put_video(conn, name, duration, W, H, sha1, model_name, time.time())
        conn.commit()
        status.update(status="ready", progress=1.0, stage="done",
                      segments=len(segments))
    except Exception as e:  # noqa: BLE001
        status.update(status=f"error: {e}", progress=0, stage="failed")
    finally:
        conn.close()
    return status
