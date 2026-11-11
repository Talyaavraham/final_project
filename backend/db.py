"""
SQLite storage for the Video Search Agent.

Tables
------
videos      one row per indexed video (duration, size, content hash, model used)
detections  raw per-frame detections (kept for debugging / re-segmentation)
segments    per-track segments computed ONCE at index time — this is what
            /search reads, so a query never re-processes raw detections.
"""

import json
import os
import sqlite3
from typing import Iterable, Optional

DB_PATH = os.environ.get("DB_PATH", "index.db")

SCHEMA = """
CREATE TABLE IF NOT EXISTS videos (
    video      TEXT PRIMARY KEY,
    duration   REAL,
    width      INTEGER,
    height     INTEGER,
    sha1       TEXT,
    model      TEXT,
    indexed_at REAL
);
CREATE TABLE IF NOT EXISTS detections (
    video TEXT, t REAL, track_id INTEGER, cls TEXT, conf REAL,
    x REAL, y REAL, w REAL, h REAL, color TEXT
);
CREATE INDEX IF NOT EXISTS idx_det ON detections(video, cls, t);
CREATE TABLE IF NOT EXISTS segments (
    video     TEXT,
    seg_id    INTEGER,
    track_id  INTEGER,
    entity_id INTEGER,
    cls       TEXT,
    t         REAL,
    dur       REAL,
    conf      REAL,
    color     TEXT,
    speed     REAL,
    entering  INTEGER,
    exiting   INTEGER,
    box       TEXT,      -- JSON {x,y,w,h} in % of frame
    path      TEXT,      -- JSON [[t,x,y,w,h], ...]
    embedding BLOB,      -- float32 CLIP vectors, (k, D) flattened: best crop, then context views (optional)
    best_t    REAL       -- time of the most confident detection (= box)
);
CREATE INDEX IF NOT EXISTS idx_seg ON segments(video, cls, t);
CREATE TABLE IF NOT EXISTS frames (
    video     TEXT,
    t         REAL,
    embedding BLOB        -- float32 CLIP vector of a square view of the frame (scene search);
                          -- a wide frame has several rows with the same t
);
CREATE INDEX IF NOT EXISTS idx_frames ON frames(video, t);
"""


def connect(path: Optional[str] = None) -> sqlite3.Connection:
    conn = sqlite3.connect(path or DB_PATH, check_same_thread=False)
    conn.executescript(SCHEMA)
    # migration for index files created before best_t existed
    cols = {r[1] for r in conn.execute("PRAGMA table_info(segments)")}
    if "best_t" not in cols:
        conn.execute("ALTER TABLE segments ADD COLUMN best_t REAL")
    return conn


# ---------------------------------------------------------------- videos
def get_video(conn, video: str):
    row = conn.execute(
        "SELECT duration, width, height, sha1, model FROM videos WHERE video=?",
        (video,)).fetchone()
    if row is None:
        return None
    return dict(zip(("duration", "width", "height", "sha1", "model"), row))


def find_by_hash(conn, sha1: str) -> Optional[str]:
    """Name of an already-indexed video with the same content, if any."""
    row = conn.execute("SELECT video FROM videos WHERE sha1=?", (sha1,)).fetchone()
    return row[0] if row else None


def put_video(conn, video, duration, width, height, sha1, model, indexed_at):
    conn.execute("INSERT OR REPLACE INTO videos VALUES (?,?,?,?,?,?,?)",
                 (video, duration, width, height, sha1, model, indexed_at))


def delete_video(conn, video: str):
    for table in ("detections", "segments", "frames", "videos"):
        conn.execute(f"DELETE FROM {table} WHERE video=?", (video,))


# ---------------------------------------------------------------- frames
def insert_frames(conn, video: str, rows: Iterable[tuple]):
    """rows: (t, embedding blob) — whole-frame CLIP vectors, one per ~second"""
    conn.executemany("INSERT INTO frames VALUES (?,?,?)",
                     [(video, t, blob) for t, blob in rows])


def delete_frames(conn, video: str):
    conn.execute("DELETE FROM frames WHERE video=?", (video,))


def load_frames(conn, video: str):
    """-> [(t, embedding blob), ...] in time order"""
    return conn.execute("SELECT t, embedding FROM frames WHERE video=? ORDER BY t",
                        (video,)).fetchall()


# ------------------------------------------------------------ detections
def insert_detections(conn, rows: Iterable[tuple]):
    """rows: (video, t, track_id, cls, conf, x, y, w, h, color)"""
    conn.executemany("INSERT INTO detections VALUES (?,?,?,?,?,?,?,?,?,?)", rows)


def load_detections(conn, video: str):
    """-> [(t, track_id, cls, conf, x, y, w, h, color), ...]"""
    return conn.execute(
        "SELECT t, track_id, cls, conf, x, y, w, h, color FROM detections "
        "WHERE video=? ORDER BY t", (video,)).fetchall()


# -------------------------------------------------------------- segments
def replace_segments(conn, video: str, segments: list[dict]):
    conn.execute("DELETE FROM segments WHERE video=?", (video,))
    conn.executemany(
        "INSERT INTO segments VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
        [(video, i, s["track_id"], s.get("entity_id", s["track_id"]),
          s["cls"], s["t"], s["dur"], s["conf"], s["color"], s["speed"],
          int(s["entering"]), int(s["exiting"]),
          json.dumps(s["box"]), json.dumps(s["path"]),
          s.get("embedding"), s.get("best_t", s["t"]))
         for i, s in enumerate(segments)])


def update_embeddings(conn, video: str, rows: Iterable[tuple]):
    """rows: (seg_id, embedding blob): replace the CLIP vectors of existing segments"""
    conn.executemany("UPDATE segments SET embedding=? WHERE video=? AND seg_id=?",
                     [(blob, video, seg_id) for seg_id, blob in rows])


def load_segments(conn, video: str, classes: Optional[list[str]] = None):
    sql = ("SELECT seg_id, track_id, entity_id, cls, t, dur, conf, color, "
           "speed, entering, exiting, box, path, embedding, best_t "
           "FROM segments WHERE video=?")
    args: list = [video]
    if classes:
        sql += f" AND cls IN ({','.join('?' * len(classes))})"
        args += classes
    out = []
    for r in conn.execute(sql + " ORDER BY t", args):
        out.append({
            "seg_id": r[0], "track_id": r[1], "entity_id": r[2], "cls": r[3],
            "t": r[4], "dur": r[5], "conf": r[6], "color": r[7],
            "speed": r[8], "entering": bool(r[9]), "exiting": bool(r[10]),
            "box": json.loads(r[11]), "path": json.loads(r[12]),
            "embedding": r[13], "best_t": r[14] if r[14] is not None else r[4],
        })
    return out


def list_classes(conn, video: str) -> dict[str, int]:
    return dict(conn.execute(
        "SELECT cls, COUNT(*) FROM segments WHERE video=? GROUP BY cls",
        (video,)).fetchall())
