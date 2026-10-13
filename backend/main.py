"""
Video Search Agent — Backend
============================
FastAPI server that:
  1. Receives a video file            -> POST /index
  2. Runs YOLOv8 detection + tracking  (samples ~2 frames/sec)
  3. Extracts attributes (color) and stores everything in SQLite
  4. Answers structured queries        -> POST /search
     (the frontend sends the parsed query: objects, colors, actions, timeRange)

Run:
    pip install -r requirements.txt
    uvicorn main:app --reload --port 8000

Then open video-search-agent.html — it will connect automatically.
"""

import os
import sqlite3
import threading
import colorsys
from collections import defaultdict

import cv2
import numpy as np
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from ultralytics import YOLO

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

# yolov8n = nano (fast). Use yolov8m/l for better accuracy if you have a GPU.
model = YOLO("yolov8n.pt")

# COCO class-id -> our vocabulary (must match the frontend VOCAB keys)
COCO_MAP = {
    0: "person", 1: "bicycle", 2: "car", 3: "motorcycle",
    5: "bus", 7: "truck", 16: "dog", 24: "bag",  # 24 = backpack
}

INDEX_STATUS = {}  # video_name -> "indexing" / "ready" / "error: ..."


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
    conn.execute("CREATE INDEX IF NOT EXISTS idx_q ON detections(video, cls, t)")
    return conn


# ----------------------------------------------------------------------
# Attribute extraction: dominant color name of a crop
# ----------------------------------------------------------------------
COLOR_NAMES = {
    "red":    [(0, 10), (170, 180)],   # hue ranges (OpenCV H is 0-180)
    "yellow": [(20, 35)],
    "green":  [(36, 85)],
    "blue":   [(86, 130)],
}


def dominant_color(crop_bgr) -> str:
    if crop_bgr.size == 0:
        return "unknown"
    hsv = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    mean_s, mean_v = s.mean(), v.mean()
    if mean_v < 50:
        return "black"
    if mean_s < 40 and mean_v > 160:
        return "white"
    if mean_s < 40:
        return "gray"
    hist = cv2.calcHist([h], [0], None, [180], [0, 180]).flatten()
    hue = int(hist.argmax())
    for name, ranges in COLOR_NAMES.items():
        for lo, hi in ranges:
            if lo <= hue <= hi:
                return name
    return "unknown"


# ----------------------------------------------------------------------
# 1-2. Video Analysis + Indexing Modules
# ----------------------------------------------------------------------
def index_video(path: str, name: str, sample_fps: float = 2.0):
    try:
        cap = cv2.VideoCapture(path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 25
        step = max(1, int(round(fps / sample_fps)))
        conn = db()
        conn.execute("DELETE FROM detections WHERE video=?", (name,))

        frame_i = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if frame_i % step == 0:
                t = frame_i / fps
                H, W = frame.shape[:2]
                # persist=True keeps track IDs consistent between calls
                results = model.track(frame, persist=True, verbose=False)[0]
                if results.boxes is not None:
                    for box in results.boxes:
                        cls_id = int(box.cls[0])
                        if cls_id not in COCO_MAP:
                            continue
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        cls = COCO_MAP[cls_id]
                        # for a person, color of the shirt = upper half of box
                        y_mid = y1 + (y2 - y1) // 2
                        crop = frame[y1:y_mid, x1:x2] if cls == "person" \
                            else frame[y1:y2, x1:x2]
                        conn.execute(
                            "INSERT INTO detections VALUES (?,?,?,?,?,?,?,?,?,?)",
                            (name, t,
                             int(box.id[0]) if box.id is not None else -1,
                             cls, float(box.conf[0]),
                             100 * x1 / W, 100 * y1 / H,
                             100 * (x2 - x1) / W, 100 * (y2 - y1) / H,
                             dominant_color(crop)))
            frame_i += 1

        conn.commit()
        conn.close()
        INDEX_STATUS[name] = "ready"
    except Exception as e:  # noqa: BLE001
        INDEX_STATUS[name] = f"error: {e}"


@app.post("/index")
async def index_endpoint(file: UploadFile = File(...)):
    path = os.path.join(VIDEO_DIR, file.filename)
    with open(path, "wb") as f:
        f.write(await file.read())
    INDEX_STATUS[file.filename] = "indexing"
    threading.Thread(target=index_video, args=(path, file.filename),
                     daemon=True).start()
    return {"status": "indexing", "video": file.filename}


@app.get("/status")
def status(video: str):
    return {"video": video, "status": INDEX_STATUS.get(video, "unknown")}


# ----------------------------------------------------------------------
# 3. Search Engine
# ----------------------------------------------------------------------
RUN_SPEED = 6.0   # % of frame-width per second => "running"
WALK_SPEED = 1.0  # below this => "parked" / standing


def track_segments(rows):
    """Group raw detections into per-track segments with speed."""
    tracks = defaultdict(list)
    for t, tid, cls, conf, x, y, w, h, color in rows:
        tracks[(tid, cls)].append((t, conf, x, y, w, h, color))

    segments = []
    for (tid, cls), dets in tracks.items():
        dets.sort()
        t0, t1 = dets[0][0], dets[-1][0]
        # average speed of box center (in % of frame per second)
        speed = 0.0
        if len(dets) > 1 and t1 > t0:
            cx0 = dets[0][2] + dets[0][4] / 2
            cx1 = dets[-1][2] + dets[-1][4] / 2
            cy0 = dets[0][3] + dets[0][5] / 2
            cy1 = dets[-1][3] + dets[-1][5] / 2
            speed = float(np.hypot(cx1 - cx0, cy1 - cy0) / (t1 - t0))
        colors = [d[6] for d in dets]
        best = max(dets, key=lambda d: d[1])
        segments.append({
            "t": t0,
            "dur": max(2.0, t1 - t0),
            "conf": float(np.mean([d[1] for d in dets])),
            "cls": cls,
            "color": max(set(colors), key=colors.count),
            "speed": speed,
            "box": {"x": best[2], "y": best[3], "w": best[4], "h": best[5]},
            "track_id": tid,
        })
    return segments


@app.post("/search")
async def search(payload: dict):
    """
    payload = { "video": "name.mp4",
                "query": { "objects": [...], "colors": [...],
                           "actions": [...], "timeRange": [min,min] | null } }
    """
    q = payload.get("query", {})
    video = payload.get("video")
    objects = q.get("objects") or []
    colors = q.get("colors") or []
    actions = q.get("actions") or []
    time_range = q.get("timeRange")

    conn = db()
    sql = ("SELECT t, track_id, cls, conf, x, y, w, h, color "
           "FROM detections WHERE video=?")
    args = [video]
    if objects:
        sql += f" AND cls IN ({','.join('?' * len(objects))})"
        args += objects
    rows = conn.execute(sql, args).fetchall()
    conn.close()

    segments = track_segments(rows)

    # attribute filters
    if colors:
        segments = [s for s in segments if s["color"] in colors]
    if time_range:
        lo, hi = time_range[0] * 60, time_range[1] * 60
        segments = [s for s in segments if lo <= s["t"] <= hi]
    if "running" in actions:
        segments = [s for s in segments if s["speed"] >= RUN_SPEED]
    elif "walking" in actions:
        segments = [s for s in segments
                    if WALK_SPEED <= s["speed"] < RUN_SPEED]
    elif "parked" in actions:
        segments = [s for s in segments if s["speed"] < WALK_SPEED]

    # multi-object query ("person with a dog"): keep segments that overlap
    # in time with at least one segment of every other requested class
    if len(objects) > 1:
        by_cls = defaultdict(list)
        for s in segments:
            by_cls[s["cls"]].append(s)

        def overlaps(a, b):
            return a["t"] <= b["t"] + b["dur"] and b["t"] <= a["t"] + a["dur"]

        kept = []
        for s in segments:
            if all(any(overlaps(s, o) for o in by_cls[c])
                   for c in objects if c != s["cls"] and by_cls[c]):
                kept.append(s)
        segments = kept

    segments.sort(key=lambda s: s["t"])
    return [{
        "t": s["t"],
        "dur": s["dur"],
        "conf": s["conf"],
        "label": f'{s["color"]} {s["cls"]}'.strip(),
        "box": s["box"],
    } for s in segments[:50]]


@app.get("/")
def root():
    return {"ok": True, "service": "video-search-agent"}
