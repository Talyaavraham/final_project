"""
Video analysis + indexing.

  video file -> YOLO detection + ByteTrack tracking (SAMPLE_FPS frames/s)
             -> colour per detection
             -> detections written to SQLite in batches (nothing is lost if
                the server dies after an hour of indexing)
             -> per-track segments (speed, entering/exiting, best box)
             -> optional second pass, CLIP (embed_video): vectors of whole
                frames and of several views of every track
             -> tracks that look alike are merged into one entity

    python indexer.py reembed [--db index.db] [--videos videos]
        redo only the CLIP pass of every indexed video (after changing
        CLIP_MODEL / TRACK_VIEWS / SCENE_EVERY); the detections are kept

Models (env YOLO_MODEL, one name or a comma-separated list — with a list the
model that finds the most in 20 probe frames indexes the video):
  yolov8n.pt / yolov8s.pt / yolov8m.pt   closed set — the 80 COCO classes
  yolov8s-worldv2.pt / yolov8m-worldv2.pt  open vocabulary — any text label,
                                          we index the OPEN_VOCAB list below
  train/runs/<name>/weights/best.pt      a model fine-tuned on your own classes
                                          (see train/README.md); class names
                                          come from the weights file
"""
import math
import os
import time
import traceback
from collections import Counter, defaultdict

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
PROBE_CONF = 0.7            # only detections the model is sure of count in the probe
PROBE_MARGIN = 2.0          # a specialised model replaces the default (first) one only if it finds
PROBE_MIN = 0.25 * PROBE_FRAMES   # twice as much, at least this much in total...
PROBE_FOCUS = 0.3           # ...and 30% of what it finds at conf 0.3 survives PROBE_CONF ("at home")
# models that run on every frame IN ADDITION to the chosen one, e.g.
# "security.pt" (gun / knife / fire / smoke) next to the COCO model
EXTRA_MODELS = [m.strip() for m in os.environ.get("EXTRA_MODELS", "").split(",") if m.strip()]
NO_COLOR = {"fire", "smoke"}  # "white fire" is not a thing: no colour attribute for these
EXTRA_ID_BASE = 100000
EXTRA_CONF = float(os.environ.get("EXTRA_CONF", "0.2"))  # extra models (gun, knife, fire) are rare-event detectors: keep weaker hits, tracks still need 2+      # track ids of extra models are offset so they never collide
CONF_THRESHOLD = float(os.environ.get("CONF_THRESHOLD", "0.3"))
SAMPLE_FPS = float(os.environ.get("SAMPLE_FPS", "3"))
IMGSZ = int(os.environ.get("IMGSZ", "640"))   # detector input size; 800+ for small objects
FLUSH_EVERY = 2000          # detections per DB write
# ByteTrack: no camera-motion compensation step, so it is faster than the
# default BoT-SORT and does not need texture to work on
TRACKER = os.environ.get("TRACKER", "bytetrack.yaml")
MAX_CROP = 224              # px, CLIP's input size: views are shrunk to this
SCENE_EVERY = float(os.environ.get("SCENE_EVERY", "1.0"))   # s between whole-frame CLIP vectors (0 = off)
# views of a track kept for CLIP next to its best crop: square crops with some
# surroundings, spread over the track's life. "holding a knife" is true for a
# few seconds of a track and shows next to the person, not in the tight box
TRACK_VIEWS = int(os.environ.get("TRACK_VIEWS", "4"))
VIEW_PAD = 1.3              # side of a context view / longer side of the box
WIDE_FRAME = 1.4            # frames wider than this (x the short side) get several square views
EMBED_BATCH = 64            # images per CLIP call

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


def probe_score(model, frames, conf: float = PROBE_CONF, low: float = 0.3):
    """
    (sure, loose): sum of confidences of what the model finds in the frames,
    counting detections >= conf and >= low respectively. A model out of its
    domain finds plenty at low confidence and little it is sure of (game.pt
    "sees" fruit and suns in CCTV footage at 0.3-0.6), so sure/loose says
    whether the model is at home in this video.
    """
    sure = loose = 0.0
    for frame in frames:
        res = model.predict(frame, verbose=False, conf=low, imgsz=IMGSZ)[0]
        for box in res.boxes or []:
            if class_name(model, int(box.cls[0])) in SKIP:
                continue
            c = float(box.conf[0])
            loose += c
            if c >= conf:
                sure += c
    return sure, loose


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
    The first name is the default. Another model takes over only when it finds
    clearly more (PROBE_MARGIN x) that it is sure of in PROBE_FRAMES frames,
    enough of it (PROBE_MIN), and is at home in the footage (PROBE_FOCUS) —
    see probe_score. One model configured -> no probe. Returns (name, model).
    """
    names = names or MODEL_NAMES
    if len(names) == 1:
        return names[0], load_model(names[0])
    frames = sample_frames(path)
    models, scores = {}, {}
    for name in names:
        if status is not None:
            status["stage"] = f"trying {name}"
        models[name] = load_model(name)
        scores[name] = probe_score(models[name], frames)
    best = names[0]
    for name in names[1:]:
        sure, loose = scores[name]
        if (sure >= PROBE_MIN and sure > PROBE_MARGIN * scores[names[0]][0]
                and sure >= PROBE_FOCUS * loose and sure > scores[best][0]):
            best = name
    return best, models[best]


def reset_tracker(model):
    """
    track(persist=True) keeps the tracker inside the cached model, so without
    this the tracks of the previous video are still "lost, waiting to come
    back" during the first seconds of the next one, and ids keep counting up.
    """
    for tracker in getattr(getattr(model, "predictor", None), "trackers", None) or []:
        tracker.reset()


def _shrink(img, max_side=MAX_CROP):
    s = max_side / max(img.shape[:2])
    if s < 1:
        img = cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    return img


# ------------------------------------------------------------------ CLIP
def frame_views(frame):
    """
    The views of a frame given to CLIP. CLIP looks at the centre square: of a
    4:3 frame that is 75%, and on UCF-Crime (4:3) the centre square alone
    scored as well as squares covering the whole frame, at half the cost. Of a
    16:9 frame it is 56% only, and whatever happens at the sides (a door, a
    parked car) is not in the vector at all, so a wide frame is covered by
    several squares (not measured: the benchmark has no wide videos).
    """
    H, W = frame.shape[:2]
    side, long = min(H, W), max(H, W)
    if long <= WIDE_FRAME * side:
        return [frame]
    n = math.ceil(long / side)
    starts = [round(i * (long - side) / (n - 1)) for i in range(n)]
    return [frame[:, a:a + side] if W >= H else frame[a:a + side] for a in starts]


def context_crop(frame, x1, y1, x2, y2, pad=VIEW_PAD):
    """The square around a box, `pad` times its longer side, clipped to the frame."""
    H, W = frame.shape[:2]
    side = pad * max(x2 - x1, y2 - y1)
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    a1, b1 = int(max(0, cx - side / 2)), int(max(0, cy - side / 2))
    a2, b2 = int(min(W, cx + side / 2)), int(min(H, cy + side / 2))
    return frame[b1:max(b2, b1 + 1), a1:max(a2, a1 + 1)]


def view_points(path, k=None):
    """Up to k points of a track's path [[t, x, y, w, h], ...], spread evenly over it."""
    k = TRACK_VIEWS if k is None else k
    if k <= 0 or not path:
        return []
    n = min(k, len(path))
    idx = sorted({round(i * (len(path) - 1) / max(1, n - 1)) for i in range(n)})
    return [path[i] for i in idx]


def embed_video(path: str, name: str, segments: list, conn, status: dict = None,
                scene_every: float = None) -> bool:
    """
    The CLIP pass: reads the video once more and stores
      * whole-frame vectors every `scene_every` s (frame_views) in `frames`
      * per segment a (k, D) matrix in s["embedding_vec"] / s["embedding"]:
        row 0 its best crop (the box the UI shows), then the context views
    It runs after the segments are known, so the views are chosen per segment
    with the whole track in hand. False when CLIP is not available.
    """
    scene_every = SCENE_EVERY if scene_every is None else scene_every
    if not embeddings.available():
        return False
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        return False
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    jobs = defaultdict(list)            # frame index -> [("scene", t) | ("seg", i, row, box)]
    for i, s in enumerate(segments):
        if not s["path"]:
            continue
        best = min(s["path"], key=lambda p: abs(p[0] - s["best_t"]))
        jobs[int(round(best[0] * fps))].append(("seg", i, 0, best[1:]))
        for row, p in enumerate(view_points(s["path"]), 1):
            jobs[int(round(p[0] * fps))].append(("seg", i, row, p[1:]))
    end = max([s["t"] + s["dur"] for s in segments] + [0.0])
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or int(end * fps) + 1
    if scene_every > 0:
        for k in range(int(total / fps / scene_every) + 1):
            if int(round(k * scene_every * fps)) < total:
                jobs[int(round(k * scene_every * fps))].append(("scene", k * scene_every))

    database.delete_frames(conn, name)
    rows = defaultdict(dict)            # segment index -> {row: vector}
    pending = []                        # (job, image)
    ok_all = True

    def flush():
        nonlocal ok_all
        if not pending:
            return
        vecs = embeddings.embed_images([img for _, img in pending])
        if vecs is None:
            ok_all = False
        else:
            database.insert_frames(conn, name, [(job[1], embeddings.to_blob(v))
                                                for (job, _), v in zip(pending, vecs) if job[0] == "scene"])
            conn.commit()
            for (job, _), v in zip(pending, vecs):
                if job[0] == "seg":
                    rows[job[1]][job[2]] = v
        pending.clear()

    last = max(jobs) if jobs else -1
    i = 0
    while i <= last and ok_all:
        if not cap.grab():
            break
        if i in jobs:
            ok, frame = cap.retrieve()
            if ok:
                H, W = frame.shape[:2]
                for job in jobs[i]:
                    if job[0] == "scene":
                        # short side down to CLIP's input size, not the long one
                        pending += [(job, _shrink(v, MAX_CROP * max(v.shape[:2]) / min(v.shape[:2])))
                                    for v in frame_views(frame)]
                        continue
                    x, y, w, h = job[3]
                    x1, y1 = int(x * W / 100), int(y * H / 100)
                    x2, y2 = max(x1 + 1, int((x + w) * W / 100)), max(y1 + 1, int((y + h) * H / 100))
                    crop = frame[y1:y2, x1:x2] if job[2] == 0 else context_crop(frame, x1, y1, x2, y2)
                    if crop.size:
                        pending.append((job, _shrink(crop)))
                if len(pending) >= EMBED_BATCH:
                    flush()
                if status is not None:
                    status["progress"] = 0.9 + 0.09 * i / max(1, last)
        i += 1
    cap.release()
    flush()
    for i, s in enumerate(segments):
        if rows[i]:
            s["embedding_vec"] = np.stack([rows[i][r] for r in sorted(rows[i])])
            s["embedding"] = embeddings.to_blob(s["embedding_vec"])
    return ok_all


def reembed(db_path: str = None, video_dir: str = "videos"):
    """Redo the CLIP pass of every indexed video; detections and segments stay as they are."""
    conn = database.connect(db_path)
    videos = [r[0] for r in conn.execute("SELECT video FROM videos ORDER BY video")]
    t0 = time.time()
    for n, name in enumerate(videos, 1):
        path = os.path.join(video_dir, name)
        if not os.path.isfile(path):
            print(f"[{n}/{len(videos)}] {name}: file not found in {video_dir}, skipped", flush=True)
            continue
        segments = database.load_segments(conn, name)
        done = embed_video(path, name, segments, conn)
        if done:
            database.update_embeddings(conn, name, [(s["seg_id"], s.get("embedding")) for s in segments])
        conn.commit()
        print(f"[{n}/{len(videos)}] {name}: {len(segments)} segments{'' if done else ' (CLIP FAILED)'} "
              f"({(time.time() - t0) / 60:.1f} min)", flush=True)
    conn.close()


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
        extras = [n for n in EXTRA_MODELS if n != model_name]
        runs = [(model, 0, CONF_THRESHOLD)] + [(load_model(n), EXTRA_ID_BASE * (k + 1), EXTRA_CONF)
                                            for k, n in enumerate(extras)]
        if extras:
            model_name = "+".join([model_name] + extras)
        for m, _, _ in runs:
            reset_tracker(m)
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
                boxes = []
                for m, id_base, conf_thr in runs:
                    res = m.track(frame, persist=True, verbose=False,
                                  conf=conf_thr, tracker=TRACKER,
                                  imgsz=IMGSZ)[0]
                    boxes += [(m, id_base, box) for box in (res.boxes or [])]
                for m, id_base, box in boxes:
                    cls = class_name(m, int(box.cls[0]))
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
                        tid = int(box.id[0]) + id_base
                    else:
                        tid = untracked_id
                        untracked_id -= 1
                    conf = float(box.conf[0])
                    rows.append((name, t, tid, cls, conf,
                                 100 * x1 / W, 100 * y1 / H,
                                 100 * (x2 - x1) / W, 100 * (y2 - y1) / H,
                                 dominant_color(color_crop) if cls not in NO_COLOR else ""))
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

        # ---- CLIP: scenes, views of every track, entity merge ---------
        if embeddings.available():
            status["stage"] = "computing embeddings"
            if embed_video(path, name, segments, conn, status):
                merge_entities(segments)

        database.replace_segments(conn, name, segments)
        database.put_video(conn, name, duration, W, H, sha1, model_name, time.time())
        conn.commit()
        status.update(status="ready", progress=1.0, stage="done",
                      segments=len(segments))
    except Exception as e:  # noqa: BLE001
        traceback.print_exc()
        status.update(status=f"error: {e}", progress=0, stage="failed")
    finally:
        conn.close()
    return status


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="maintenance of an index")
    ap.add_argument("command", choices=["reembed"])
    ap.add_argument("--db", default=None, help="index file (default: DB_PATH / index.db)")
    ap.add_argument("--videos", default=os.environ.get("VIDEO_DIR", "videos"), help="folder of the video files")
    a = ap.parse_args()
    reembed(a.db, a.videos)
