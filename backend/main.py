"""
Video Search Agent — Backend
============================
FastAPI server:
  POST /index              upload a video -> index in the background
  GET  /                   the web interface (index.html)
  GET  /api                server capabilities (model, CLIP, Claude, ffmpeg)
  GET  /status?video=      indexing progress
  GET  /videos             indexed videos
  GET  /classes?video=     which classes were found (for the UI / parser)
  POST /parse              natural language -> structured query (Claude / rules)
  POST /search             structured (+ free text) query -> matches
  GET  /frame              one frame as JPEG, with the match box drawn
  GET  /clip               a clip as MP4 (optionally with tracked boxes)
  POST /export             ZIP with frames + clips + results.json/csv
  DELETE /videos/{name}    remove a video and its index

Run:
    pip install -r requirements.txt
    uvicorn main:app --reload --port 8000

Env:
    YOLO_MODEL=yolov8s.pt | yolov8m.pt | yolov8s-worldv2.pt (open vocabulary)
    YOLO_MODEL=yolov8n.pt,game.pt   several: the best one per video is chosen
    CLIP_RERANK=0          disable CLIP (semantic search + entity merge)
    ANTHROPIC_API_KEY=...  enable the Claude query parser
"""

import hashlib
import os
import shutil
import threading
import time
from typing import Optional

from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response
from pydantic import BaseModel, Field

import db as database
import embeddings
import export
import indexer
import query_parser
from search import filter_segments, scene_windows, semantic_rank

MIN_CROP_NO_OBJECT = 0.25   # crop similarity needed when the question names no object class

app = FastAPI(title="Video Search Agent API", version="2.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.environ.get("CORS_ORIGINS", "*").split(","),
    allow_methods=["*"],
    allow_headers=["*"],
)

VIDEO_DIR = os.environ.get("VIDEO_DIR", "videos")
EXPORT_DIR = os.environ.get("EXPORT_DIR", "exports")
os.makedirs(VIDEO_DIR, exist_ok=True)
os.makedirs(EXPORT_DIR, exist_ok=True)

INDEX_STATUS: dict[str, dict] = {}
INDEX_LOCK = threading.Lock()          # one indexing job at a time


# ------------------------------------------------------------------ models
class QueryModel(BaseModel):
    objects: list[str] = []
    colors: list[str] = []
    actions: list[str] = []
    timeRange: Optional[list[Optional[float]]] = None
    semantic: str = ""                 # free text for CLIP re-ranking
    text: str = ""                     # the original sentence (for logs/export)


class SearchRequest(BaseModel):
    video: str
    query: QueryModel = Field(default_factory=QueryModel)
    limit: int = Field(100, ge=1, le=1000)
    offset: int = Field(0, ge=0)
    near: bool = True                  # multi-object: require spatial proximity


class ParseRequest(BaseModel):
    text: str
    video: Optional[str] = None


class ExportRequest(BaseModel):
    video: str
    matches: list[dict]
    query: str = ""
    clips: bool = True
    annotate: bool = True
    pad: float = Field(1.0, ge=0, le=10)


# ----------------------------------------------------------------- helpers
def safe_name(filename: str) -> str:
    name = os.path.basename(filename or "").strip().replace("\\", "_")
    if not name or name.startswith(".") or "/" in name:
        raise HTTPException(400, "invalid file name")
    return name


def video_path(name: str) -> str:
    path = os.path.join(VIDEO_DIR, safe_name(name))
    if not os.path.isfile(path):
        raise HTTPException(404, "video file not found")
    return path


def require_indexed(conn, name: str) -> dict:
    meta = database.get_video(conn, name)
    if meta is None:
        raise HTTPException(404, "video not indexed")
    return meta


def _run_index(path, name, sha1):
    with INDEX_LOCK:
        indexer.index_video(path, name, sha1, status=INDEX_STATUS[name])


# --------------------------------------------------------------- endpoints
INDEX_HTML = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "index.html")


@app.get("/")
def root():
    """The web interface — one URL for everything: http://localhost:8000"""
    if os.path.isfile(INDEX_HTML):
        return FileResponse(INDEX_HTML, media_type="text/html")
    return capabilities()


@app.get("/api")
def capabilities():
    return {
        "ok": True, "service": "video-search-agent", "version": "2.0",
        "model": indexer.MODEL_NAME,
        "models": indexer.MODEL_NAMES, "extra_models": indexer.EXTRA_MODELS,
        "open_vocab": "world" in indexer.MODEL_NAME.lower(),
        "clip": embeddings.available(),
        "clip_error": embeddings.error(),
        "llm_parser": query_parser.llm_available(),
        "ffmpeg": export.ffmpeg_exe() is not None,
    }


@app.post("/index")
def index_endpoint(file: UploadFile = File(...), force: bool = False):
    name = safe_name(file.filename)
    if name in INDEX_STATUS and INDEX_STATUS[name].get("status") == "indexing":
        raise HTTPException(409, "this video is being indexed right now")

    # stream to disk and hash on the way
    path = os.path.join(VIDEO_DIR, name)
    tmp = path + ".part"
    sha = hashlib.sha1()
    with open(tmp, "wb") as f:
        while chunk := file.file.read(1 << 20):
            sha.update(chunk)
            f.write(chunk)
    sha1 = sha.hexdigest()
    os.replace(tmp, path)

    conn = database.connect()
    try:
        same = database.find_by_hash(conn, sha1)
        meta = database.get_video(conn, name)
        if not force and meta and meta["sha1"] == sha1:
            INDEX_STATUS[name] = {"status": "ready", "progress": 1.0, "stage": "cached"}
            return {"status": "ready", "video": name, "cached": True}
        if not force and same and same != name and os.path.isfile(os.path.join(VIDEO_DIR, same)):
            # identical content under another name -> reuse that index
            INDEX_STATUS[same] = {"status": "ready", "progress": 1.0, "stage": "cached"}
            os.remove(path)
            return {"status": "ready", "video": same, "cached": True}
    finally:
        conn.close()

    INDEX_STATUS[name] = {"status": "indexing", "progress": 0.0, "stage": "queued"}
    threading.Thread(target=_run_index, args=(path, name, sha1), daemon=True).start()
    return {"status": "indexing", "video": name}


@app.get("/status")
def status(video: str):
    st = INDEX_STATUS.get(video)
    if st is None:                      # indexed in a previous server run?
        conn = database.connect()
        known = database.get_video(conn, video)
        conn.close()
        st = ({"status": "ready", "progress": 1.0, "stage": "done"} if known
              else {"status": "unknown", "progress": 0, "stage": ""})
    return {"video": video, **st}


@app.get("/videos")
def videos():
    conn = database.connect()
    rows = conn.execute("SELECT video, duration, width, height, model, indexed_at "
                        "FROM videos ORDER BY indexed_at DESC").fetchall()
    conn.close()
    return [dict(zip(("video", "duration", "width", "height", "model", "indexed_at"), r))
            for r in rows]


@app.delete("/videos/{name}")
def delete_video(name: str):
    name = safe_name(name)
    conn = database.connect()
    database.delete_video(conn, name)
    conn.commit()
    conn.close()
    INDEX_STATUS.pop(name, None)
    p = os.path.join(VIDEO_DIR, name)
    if os.path.isfile(p):
        os.remove(p)
    return {"deleted": name}


@app.get("/classes")
def classes(video: str):
    conn = database.connect()
    require_indexed(conn, video)
    counts = database.list_classes(conn, video)
    conn.close()
    return {"video": video, "classes": counts}


@app.post("/parse")
def parse(req: ParseRequest):
    # the classes the detector actually found in this video, so the parser can
    # map "man" to "character" when the video was indexed by the game model
    classes = None
    if req.video:
        conn = database.connect()
        found = database.list_classes(conn, req.video)
        conn.close()
        classes = sorted(found) or None
    return query_parser.parse(req.text, classes)


@app.post("/search")
def search(req: SearchRequest):
    q = req.query
    conn = database.connect()
    meta = require_indexed(conn, req.video)
    segments = database.load_segments(conn, req.video, q.objects or None)
    conn.close()

    W, H = meta["width"] or 16, meta["height"] or 9
    for s in segments:
        s["embedding_vec"] = embeddings.from_blob(s.pop("embedding"))

    tr = None
    if q.timeRange:
        lo = q.timeRange[0] or 0
        hi = q.timeRange[1] if len(q.timeRange) > 1 else None
        tr = [lo, hi]
    segments = filter_segments(segments, q.objects, q.colors, q.actions, tr,
                               require_near=req.near, W=W, H=H)

    semantic_used = False
    if q.semantic and embeddings.available():
        vec = embeddings.embed_text(q.semantic)
        if vec is not None:
            segments = semantic_rank(segments, vec)
            semantic_used = True
            # whole-frame search: the moments of the video that look like the text
            conn = database.connect()
            frames = [(t, embeddings.from_blob(b)) for t, b in database.load_frames(conn, req.video)]
            conn.close()
            scenes = scene_windows([f for f in frames if f[1] is not None], vec)
            if tr:
                lo, hi = tr[0], float("inf") if tr[1] is None else tr[1]
                scenes = [s for s in scenes if s["t"] <= hi and s["t"] + s["dur"] >= lo]
            # no object named ("fire", "someone grabbing a bag") -> scenes lead and
            # object crops must match clearly; an object named but nothing
            # matched -> scenes are the fallback
            if not q.objects:
                segments = scenes + [s for s in segments if s.get("score", 0) >= MIN_CROP_NO_OBJECT]
            elif not segments:
                segments = scenes
    if not semantic_used:
        segments.sort(key=lambda s: s["t"])

    total = len(segments)
    page = segments[req.offset:req.offset + req.limit]
    return {
        "total": total, "offset": req.offset, "semantic": semantic_used,
        "results": [{
            "t": s["t"], "dur": s["dur"], "conf": s["conf"], "cls": s["cls"],
            "label": (f'scene: {q.semantic[:40]}' if s["cls"] == "scene"
                      else f'{s["color"]} {s["cls"]}'.strip()),
            "box": s["box"], "path": s["path"], "best_t": s.get("best_t", s["t"]),
            "track": s["track_id"], "entity": s["entity_id"],
            "speed": round(s["speed"], 3),
            "score": round(s.get("score", 0.0), 3),
        } for s in page],
    }


# ------------------------------------------------------------------ outputs
@app.get("/frame")
def frame(video: str, t: float = Query(..., ge=0),
          x: Optional[float] = None, y: Optional[float] = None,
          w: Optional[float] = None, h: Optional[float] = None,
          label: str = ""):
    path = video_path(video)
    box = [x, y, w, h] if None not in (x, y, w, h) else None
    try:
        data = export.frame_jpeg(path, t, box, label)
    except RuntimeError as e:
        raise HTTPException(400, str(e))
    return Response(data, media_type="image/jpeg", headers={
        "Content-Disposition": f'inline; filename="{os.path.splitext(video)[0]}_{t:.1f}s.jpg"'})


@app.get("/clip")
def clip(video: str, t0: float = Query(..., ge=0), t1: float = Query(..., gt=0),
         track: Optional[int] = None, annotate: bool = True):
    """Cut [t0, t1]. With `track`, the tracked box of that segment is drawn."""
    path = video_path(video)
    if t1 - t0 > 600:
        raise HTTPException(400, "clip longer than 10 minutes")
    matches = []
    if annotate and track is not None:
        conn = database.connect()
        require_indexed(conn, video)
        for s in database.load_segments(conn, video):
            if s["track_id"] == track and s["t"] <= t1 and s["t"] + s["dur"] >= t0:
                matches.append({"path": s["path"], "label": f'{s["color"]} {s["cls"]}'.strip()})
        conn.close()
    key = hashlib.sha1(f"{video}|{t0:.2f}|{t1:.2f}|{track}|{annotate}".encode()).hexdigest()[:16]
    out = os.path.join(EXPORT_DIR, f"{key}.mp4")
    if not os.path.isfile(out):
        try:
            export.cut_clip(path, t0, t1, out, matches)
        except Exception as e:  # noqa: BLE001
            raise HTTPException(500, f"clip failed: {e}")
    fname = f"{os.path.splitext(video)[0]}_{t0:.0f}-{t1:.0f}s.mp4"
    return FileResponse(out, media_type="video/mp4", filename=fname)


@app.post("/export")
def export_zip(req: ExportRequest):
    path = video_path(req.video)
    if not req.matches:
        raise HTTPException(400, "nothing to export")
    if len(req.matches) > 200:
        raise HTTPException(400, "export at most 200 matches at a time")
    try:
        data = export.build_zip(path, req.video, req.matches, req.query,
                                clips=req.clips, annotate=req.annotate, pad=req.pad)
    except Exception as e:  # noqa: BLE001
        raise HTTPException(500, f"export failed: {e}")
    fname = f"{os.path.splitext(req.video)[0]}_results_{int(time.time())}.zip"
    return Response(data, media_type="application/zip",
                    headers={"Content-Disposition": f'attachment; filename="{fname}"'})


# cleanup of old rendered clips (older than a day) at start-up
for _f in os.listdir(EXPORT_DIR):
    _p = os.path.join(EXPORT_DIR, _f)
    if os.path.isfile(_p) and time.time() - os.path.getmtime(_p) > 86400:
        try:
            os.remove(_p)
        except OSError:
            pass
