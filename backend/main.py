"""
Video Search Agent — Backend
============================
FastAPI server that:
  1. Receives a video file            -> POST /index
  2. Runs YOLOv8 detection + tracking  (samples ~3 frames/sec)
  3. Extracts attributes (color) and stores everything in SQLite
  4. Answers structured queries        -> POST /search
     (the frontend sends the parsed query: objects, colors, actions, timeRange)

Run:
    pip install -r requirements.txt
    uvicorn main:app --reload --port 8000

Then open index.html — it will connect automatically.
Set the env var YOLO_MODEL (e.g. yolov8s.pt / yolov8m.pt) for better accuracy.
"""

import os
import shutil
import sqlite3
import threading
from collections import Counter, defaultdict

import cv2
import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

# ----------------------------------------------------------------------
# Setup
# ----------------------------------------------------------------------
app = FastAPI(title="Video Search Agent API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # dev only — restrict in production
    allow_methods=["*"],
    allow_headers=["*"],
)

VIDEO_DIR = "videos"
DB_PATH = "index.db"
os.makedirs(VIDEO_DIR, exist_ok=True)

# yolov8n = nano (fast). Use yolov8s/m/l for better accuracy if you have a GPU.
MODEL_NAME = os.environ.get("YOLO_MODEL", "yolov8n.pt")
CONF_THRESHOLD = 0.3
SAMPLE_FPS = 3.0

# COCO class-id -> our vocabulary (must match the frontend VOCAB keys)
COCO_MAP = {
    0: "person", 1: "bicycle", 2: "car", 3: "motorcycle", 5: "bus",
    7: "truck", 15: "cat", 16: "dog",
    24: "bag", 26: "bag", 28: "bag",  # backpack, handbag, suitcase
}

INDEX_STATUS = {}  # video_name -> {"status": ..., "progress": 0..1}
INDEX_LOCK = threading.Lock()  # one indexing job at a time (CPU/GPU bound)


def db():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS detections (
            video     TEXT,
            t         REAL,     -- seconds from start
            track_id  INTEGER,
            cls       TEXT,
            conf      REAL,
            x REAL, y REAL, w REAL, h REAL,   -- bbox in % of frame
            color     TEXT
        )""")
    conn.execute("""
        CREATE TABLE IF NOT EXISTS videos (
            video     TEXT PRIMARY KEY,
            duration  REAL,
            width     INTEGER,
            height    INTEGER
        )""")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_q ON detections(video, cls, t)")
    return conn


# ----------------------------------------------------------------------
# Attribute extraction: dominant color name of a crop
# ----------------------------------------------------------------------
# Every pixel votes for a color name; the most common name wins.
# This is much more robust than the mean hue / mean saturation of the crop.
HUE_NAMES = [  # (upper bound of OpenCV hue 0-180, name)
    (7, "red"), (20, "orange"), (34, "yellow"), (85, "green"),
    (130, "blue"), (150, "purple"), (166, "pink"), (180, "red"),
]


def dominant_color(crop_bgr) -> str:
    if crop_bgr is None or crop_bgr.size == 0:
        return "unknown"
    # keep the central part of the box — the edges are mostly background
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
    # dark orange is what people call brown
    names[chroma & (names == "orange") & (v < 150)] = "brown"
    names[v < 50] = "black"
    gray = ~chroma & (v >= 50)
    names[gray & (v >= 170) & (s < 40)] = "white"
    names[gray & ~((v >= 170) & (s < 40))] = "gray"
    return Counter(names.tolist()).most_common(1)[0][0]


# ----------------------------------------------------------------------
# 1-2. Video Analysis + Indexing Modules
# ----------------------------------------------------------------------
def index_video(path: str, name: str, sample_fps: float = SAMPLE_FPS):
    from ultralytics import YOLO  # heavy import — only when indexing

    with INDEX_LOCK:
        try:
            # a fresh model per video, so tracker state never leaks
            # between videos (track ids restart from 1)
            model = YOLO(MODEL_NAME)
            cap = cv2.VideoCapture(path)
            if not cap.isOpened():
                raise RuntimeError("cannot open video (unsupported codec?)")
            fps = cap.get(cv2.CAP_PROP_FPS) or 25
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
            step = max(1, int(round(fps / sample_fps)))

            rows, W, H = [], 0, 0
            untracked_id = -1
            frame_i = 0
            while True:
                # grab() skips decoding of frames we do not analyse
                if not cap.grab():
                    break
                if frame_i % step == 0:
                    ok, frame = cap.retrieve()
                    if not ok:
                        break
                    H, W = frame.shape[:2]
                    t = frame_i / fps
                    res = model.track(frame, persist=True, verbose=False,
                                      conf=CONF_THRESHOLD,
                                      classes=list(COCO_MAP))[0]
                    for box in res.boxes or []:
                        cls = COCO_MAP.get(int(box.cls[0]))
                        if cls is None:
                            continue
                        x1, y1, x2, y2 = (int(round(c)) for c in box.xyxy[0])
                        x1, y1 = max(0, x1), max(0, y1)
                        x2, y2 = min(W, x2), min(H, y2)
                        if x2 <= x1 or y2 <= y1:
                            continue
                        # for a person, the shirt = upper body (15%–55% of box)
                        if cls == "person":
                            bh = y2 - y1
                            crop = frame[y1 + int(bh * .15):y1 + int(bh * .55),
                                         x1:x2]
                        else:
                            crop = frame[y1:y2, x1:x2]
                        if box.id is not None:
                            tid = int(box.id[0])
                        else:  # untracked box -> its own one-off track
                            tid = untracked_id
                            untracked_id -= 1
                        rows.append((name, t, tid, cls, float(box.conf[0]),
                                     100 * x1 / W, 100 * y1 / H,
                                     100 * (x2 - x1) / W, 100 * (y2 - y1) / H,
                                     dominant_color(crop)))
                    if total:
                        INDEX_STATUS[name]["progress"] = min(1.0, frame_i / total)
                frame_i += 1
            cap.release()

            conn = db()
            conn.execute("DELETE FROM detections WHERE video=?", (name,))
            conn.executemany(
                "INSERT INTO detections VALUES (?,?,?,?,?,?,?,?,?,?)", rows)
            conn.execute("INSERT OR REPLACE INTO videos VALUES (?,?,?,?)",
                         (name, frame_i / fps, W, H))
            conn.commit()
            conn.close()
            INDEX_STATUS[name] = {"status": "ready", "progress": 1.0}
        except Exception as e:  # noqa: BLE001
            INDEX_STATUS[name] = {"status": f"error: {e}", "progress": 0}


def safe_name(filename: str) -> str:
    name = os.path.basename(filename or "").strip()
    if not name or name.startswith("."):
        raise HTTPException(400, "invalid file name")
    return name


@app.post("/index")
def index_endpoint(file: UploadFile = File(...)):
    name = safe_name(file.filename)
    path = os.path.join(VIDEO_DIR, name)
    with open(path, "wb") as f:  # stream to disk, not into memory
        shutil.copyfileobj(file.file, f)
    INDEX_STATUS[name] = {"status": "indexing", "progress": 0.0}
    threading.Thread(target=index_video, args=(path, name),
                     daemon=True).start()
    return {"status": "indexing", "video": name}


@app.get("/status")
def status(video: str):
    st = INDEX_STATUS.get(video)
    if st is None:  # indexed in a previous server run?
        conn = db()
        known = conn.execute("SELECT 1 FROM videos WHERE video=?",
                             (video,)).fetchone()
        conn.close()
        st = {"status": "ready", "progress": 1.0} if known \
            else {"status": "unknown", "progress": 0}
    return {"video": video, **st}


# ----------------------------------------------------------------------
# 3. Search Engine
# ----------------------------------------------------------------------
# Speeds are measured in "body heights per second" (box-center movement
# divided by the box height), so they do not depend on how far the
# object is from the camera. A walking person is ~0.5-1, running ~1.5+.
RUN_SPEED = 1.2
MOVE_SPEED = 0.15   # below this => standing / parked
MAX_GAP = 3.0       # seconds without a detection that split a track
EDGE = 2.0          # % of the frame — a box this close touches the border


def track_segments(rows, duration=None, W=16, H=9):
    """Group raw detections into per-track segments with speed etc."""
    tracks = defaultdict(list)
    for t, tid, cls, conf, x, y, w, h, color in rows:
        tracks[(tid, cls)].append((t, conf, x, y, w, h, color))

    segments = []
    for (tid, cls), dets in tracks.items():
        dets.sort()
        # split a track where it disappears for a while
        parts, cur = [], [dets[0]]
        for d in dets[1:]:
            if d[0] - cur[-1][0] > MAX_GAP:
                parts.append(cur)
                cur = []
            cur.append(d)
        parts.append(cur)

        for part in parts:
            seg = make_segment(tid, cls, part, duration, W, H)
            if seg:
                segments.append(seg)
    return segments


def make_segment(tid, cls, dets, duration, W, H):
    confs = [d[1] for d in dets]
    # a single weak detection is usually noise
    if len(dets) == 1 and confs[0] < 0.5:
        return None
    t0, t1 = dets[0][0], dets[-1][0]

    # median per-step speed of the box center, in box-heights / second
    steps = []
    for a, b in zip(dets, dets[1:]):
        dt = b[0] - a[0]
        if dt <= 0:
            continue
        dx = (b[2] + b[4] / 2 - a[2] - a[4] / 2) * W / 100
        dy = (b[3] + b[5] / 2 - a[3] - a[5] / 2) * H / 100
        size = max(1e-6, (a[5] + b[5]) / 2 * H / 100)
        steps.append(np.hypot(dx, dy) / size / dt)
    speed = float(np.median(steps)) if steps else 0.0

    def touches_edge(d):
        x, y, w, h = d[2:6]
        return x < EDGE or y < EDGE or x + w > 100 - EDGE or y + h > 100 - EDGE

    entering = touches_edge(dets[0]) and t0 > 1.0 and len(dets) > 1
    exiting = (touches_edge(dets[-1]) and len(dets) > 1 and
               (duration is None or t1 < duration - 1.0))

    colors = Counter(d[6] for d in dets if d[6] != "unknown")
    best = max(dets, key=lambda d: d[1])
    return {
        "t": t0,
        "dur": max(1.0, t1 - t0),
        "conf": float(np.mean(confs)),
        "cls": cls,
        "color": colors.most_common(1)[0][0] if colors else "",
        "speed": speed,
        "entering": entering,
        "exiting": exiting,
        "box": {"x": best[2], "y": best[3], "w": best[4], "h": best[5]},
        "path": [[round(d[0], 2), round(d[2], 2), round(d[3], 2),
                  round(d[4], 2), round(d[5], 2)] for d in dets],
        "track_id": tid,
    }


def action_ok(s, actions):
    checks = {
        "running": s["speed"] >= RUN_SPEED,
        "walking": MOVE_SPEED <= s["speed"] < RUN_SPEED,
        "moving": s["speed"] >= MOVE_SPEED,
        "parked": s["speed"] < MOVE_SPEED,
        "entering": s["entering"],
        "exiting": s["exiting"],
    }
    return all(checks.get(a, True) for a in actions)


def filter_segments(segments, objects, colors, actions, time_range):
    # attributes (color / action) describe the first requested object:
    # "a man in red running with a dog" -> red + running apply to the man
    main = objects[0] if objects else None

    def attrs_ok(s):
        if main and s["cls"] != main:
            return True
        if colors and s["color"] not in colors:
            return False
        return action_ok(s, actions)

    segments = [s for s in segments if attrs_ok(s)]
    if time_range:
        lo, hi = time_range
        segments = [s for s in segments
                    if s["t"] <= hi and s["t"] + s["dur"] >= lo]

    # multi-object query ("person with a dog"): keep segments that overlap
    # in time with at least one segment of every other requested class
    if len(objects) > 1:
        by_cls = defaultdict(list)
        for s in segments:
            by_cls[s["cls"]].append(s)

        def overlaps(a, b):
            return a["t"] <= b["t"] + b["dur"] and b["t"] <= a["t"] + a["dur"]

        segments = [s for s in segments
                    if all(any(overlaps(s, o) for o in by_cls[c])
                           for c in objects if c != s["cls"])]
    return segments


@app.post("/search")
def search(payload: dict):
    """
    payload = { "video": "name.mp4",
                "query": { "objects": [...], "colors": [...],
                           "actions": [...],
                           "timeRange": [startSec, endSec] | null } }
    """
    q = payload.get("query") or {}
    video = payload.get("video")
    objects = q.get("objects") or []
    colors = q.get("colors") or []
    actions = q.get("actions") or []
    time_range = q.get("timeRange")

    conn = db()
    meta = conn.execute("SELECT duration, width, height FROM videos "
                        "WHERE video=?", (video,)).fetchone()
    if meta is None:
        conn.close()
        raise HTTPException(404, "video not indexed")
    sql = ("SELECT t, track_id, cls, conf, x, y, w, h, color "
           "FROM detections WHERE video=?")
    args = [video]
    if objects:
        sql += f" AND cls IN ({','.join('?' * len(objects))})"
        args += objects
    rows = conn.execute(sql, args).fetchall()
    conn.close()

    duration, W, H = meta
    segments = track_segments(rows, duration, W or 16, H or 9)
    segments = filter_segments(segments, objects, colors, actions, time_range)
    segments.sort(key=lambda s: s["t"])
    return [{
        "t": s["t"],
        "dur": s["dur"],
        "conf": s["conf"],
        "cls": s["cls"],
        "label": f'{s["color"]} {s["cls"]}'.strip(),
        "box": s["box"],
        "path": s["path"],
        "track": s["track_id"],
    } for s in segments[:100]]


@app.get("/")
def root():
    return {"ok": True, "service": "video-search-agent", "model": MODEL_NAME}
