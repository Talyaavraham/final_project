"""
Video Search Agent — Backend
============================
FastAPI server:
  POST /index              upload a video -> index in the background
  GET  /                   the web interface (index.html)
  GET  /api                server capabilities (model, CLIP, language model, ffmpeg)
  GET  /status?video=      indexing progress
  GET  /videos             indexed videos
  GET  /classes?video=     which classes were found (for the UI / parser)
  POST /parse              natural language -> structured query (local rules)
  POST /search             structured (+ free text) query -> matches
  GET  /frame              one frame as JPEG, with the match box drawn
  GET  /clip               a clip as MP4 (optionally with tracked boxes)
  POST /export             ZIP with frames + clips + results.json/csv
  DELETE /videos/{name}    remove a video and its index
  POST /live/rules         live mode: a sentence -> an alert rule (our language model / rules)
  GET  /live/rules, DELETE /live/rules/{id}
  WS   /live/ws            live mode: JPEG frames in -> people, objects, alerts out
  POST /events             search a recorded video for an ACTION ("someone snatches a bag"): the
                           live perception runs over the video once (cached), then the alert rule
  GET  /events/status?video=
  GET/POST /labels, DELETE /labels/{video}, POST /labels/video
                           labelling mode: events marked in our own videos (live/data/labels.json)

Run:
    pip install -r requirements.txt
    uvicorn main:app --reload --port 8000

Env:
    YOLO_MODEL=yolov8s.pt | yolov8m.pt | yolov8s-worldv2.pt (open vocabulary)
    YOLO_MODEL=yolov8n.pt,game.pt   several: the best one per video is chosen
    CLIP_RERANK=0          disable CLIP (semantic search + entity merge)
Everything runs locally: no outside API is called.
"""

import hashlib
import os
import shutil
import threading
import time
from typing import Optional

from fastapi import FastAPI, File, HTTPException, Query, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response
from pydantic import BaseModel, Field

import db as database
import embeddings
import export
import indexer
import alert_rules
import query_parser
from live_engine import engine as live
from search import run_query, with_context

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
    object_colors: dict[str, list[str]] = {}   # colours of the other objects: {"car": ["white"]}
    actions: list[str] = []
    timeRange: Optional[list[Optional[float]]] = None
    semantic: str = ""                 # free text for CLIP re-ranking
    unknown: list[str] = []            # words no detector class covers -> stricter CLIP floor
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
        "nlu": alert_rules.nlu_info(),
        "ffmpeg": export.ffmpeg_exe() is not None,
        "live": live.info(),
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
    # only the classes asked for, unless CLIP has to look at every track
    # (no class named, or only event classes: "fire", "a gun")
    only = [o for o in q.objects if o not in query_parser.EVENT_CLASSES] and q.objects
    segments = database.load_segments(conn, req.video, only or None)
    frames = [(t, embeddings.from_blob(b)) for t, b in database.load_frames(conn, req.video)]
    conn.close()

    for s in segments:
        s["embedding_vec"] = embeddings.from_blob(s.pop("embedding"))
    tr = None
    if q.timeRange:
        tr = [q.timeRange[0] or 0, q.timeRange[1] if len(q.timeRange) > 1 else None]
    segments, semantic_used = run_query(
        segments, [f for f in frames if f[1] is not None],
        q.objects, q.colors, q.actions, tr, q.semantic, q.unknown,
        embed=embeddings.embed_texts if embeddings.available() else None,
        near=req.near, W=meta["width"] or 16, H=meta["height"] or 9, object_colors=q.object_colors)

    # "X next to Y": the results are the X's, each with the Y it was next to
    segments = with_context(segments, q.objects, W=meta["width"] or 16, H=meta["height"] or 9, require_near=req.near)
    total = len(segments)
    page = segments[req.offset:req.offset + req.limit]
    return {
        "total": total, "offset": req.offset, "semantic": semantic_used,
        "results": [{
            "t": s["t"], "dur": s["dur"], "conf": s["conf"], "cls": s["cls"],
            "label": (f'scene: {(q.semantic or q.text)[:40]}' if s["cls"] == "scene"
                      else f'{s["color"]} {s["cls"]}'.strip()),
            "box": s["box"], "path": s["path"], "best_t": s.get("best_t", s["t"]),
            "track": s["track_id"], "entity": s["entity_id"],
            "context": [{"cls": c["cls"], "label": f'{c["color"]} {c["cls"]}'.strip(), "box": c["box"],
                         "path": c["path"], "t": c["t"], "dur": c["dur"]} for c in s.get("context", [])],
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


# ---------------------------------------------------------------- live mode
class RuleRequest(BaseModel):
    text: str


@app.post("/live/rules")
def add_live_rule(req: RuleRequest):
    """A sentence -> an alert rule. {rule, source} or {rule: null, reason}."""
    if not req.text.strip():
        raise HTTPException(400, "empty text")
    out = alert_rules.parse(req.text, live.classes())
    if out.get("rule"):
        out["rule"], out["new"] = live.add_rule(out["rule"])
    return out


@app.get("/live/rules")
def list_live_rules():
    return {"rules": live.list_rules()}


@app.delete("/live/rules/{rule_id}")
def delete_live_rule(rule_id: str):
    if not live.remove_rule(rule_id):
        raise HTTPException(404, "no such rule")
    return {"ok": True}


@app.websocket("/live/ws")
async def live_ws(ws: WebSocket):
    """The browser sends one JPEG (binary message) and waits for the JSON answer."""
    await ws.accept()
    await run_in_threadpool(live.start_stream)
    try:
        while True:
            data = await ws.receive_bytes()
            try:
                out = await run_in_threadpool(live.process, data)
            except Exception as e:  # noqa: BLE001  (model download failed, ...)
                live.error = str(e)
                out = {"error": str(e)}
            await ws.send_json(out)
    except WebSocketDisconnect:
        pass


# ---------------------------------------------------- actions in recorded videos
# The same alert rules as the live camera, on an uploaded video: "מישהו חוטף
# תיק" in the search chat. The live perception (pose + detectors) runs over
# the video once at EVENT_FPS and is cached in live/data/perception (the same
# cache live/eval_alerts.py uses); then any action rule is answered in seconds.
EVENT_FPS = float(os.environ.get("EVENT_FPS", "3"))
EVENT_KINDS = {"hold", "raise_object", "hands_up", "ignite", "snatch"}
EVENT_STATUS: dict[str, dict] = {}
_offline = {"engine": None}
_offline_lock = threading.Lock()


class EventRequest(BaseModel):
    video: str
    text: str


def _perception_path(video):
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "live", "data", "perception", video + ".jsonl")


def _perceive_bg(video, path, out):
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "live"))
    from perceive import perceive_video
    from live_engine import LiveEngine
    st = EVENT_STATUS[video]
    try:
        with _offline_lock:            # one video at a time; its own engine, so the live camera is not disturbed
            if _offline["engine"] is None:
                _offline["engine"] = LiveEngine()
            os.makedirs(os.path.dirname(out), exist_ok=True)
            perceive_video(_offline["engine"], path, out, EVENT_FPS, progress=lambda p: st.update(progress=p))
        st.update(status="ready", progress=1.0)
    except Exception as e:  # noqa: BLE001
        st.update(status="error", error=str(e))


def _cached_fps(out):
    import json
    try:
        with open(out, encoding="utf-8") as f:
            return json.loads(f.readline()).get("fps")
    except (OSError, ValueError):
        return None


@app.get("/events/status")
def events_status(video: str):
    out = _perception_path(safe_name(video))
    if _cached_fps(out) == EVENT_FPS:
        return {"status": "ready", "progress": 1.0}
    return EVENT_STATUS.get(safe_name(video), {"status": "none", "progress": 0.0})


@app.post("/events")
def search_events(req: EventRequest):
    """{"event": false} when the sentence is not an action (the normal search answers it);
    {"event": true, "status": "perceiving", "progress"} while the video is analysed;
    {"event": true, "status": "ready", "rule", "results"} with the moments it happens."""
    import json
    parsed = alert_rules.parse(req.text, live.classes())
    rule = parsed.get("rule")
    if not rule or rule["kind"] not in EVENT_KINDS:
        return {"event": False}
    video = safe_name(req.video)
    path = video_path(video)
    out = _perception_path(video)
    if _cached_fps(out) != EVENT_FPS:
        st = EVENT_STATUS.get(video)
        if not st or st.get("status") in ("error", "ready"):
            EVENT_STATUS[video] = {"status": "perceiving", "progress": 0.0}
            threading.Thread(target=_perceive_bg, args=(video, path, out), daemon=True).start()
        return {"event": True, "status": EVENT_STATUS[video]["status"], "progress": EVENT_STATUS[video]["progress"],
                "error": EVENT_STATUS[video].get("error"), "rule": rule}
    with open(out, encoding="utf-8") as f:
        head = json.loads(f.readline())
        frames = [json.loads(l) for l in f if l.strip()]
    W, H = max(1, head.get("w") or 1), max(1, head.get("h") or 1)
    rule["cooldown"] = 3.0                     # a recorded video: list every separate moment
    ev, results = alert_rules.Evaluator(), []
    for fr in frames:
        objs = [o for o in fr["objects"] if o["conf"] >= live_conf()]
        for a in ev.step([rule], fr["persons"], objs, fr["t"])[0]:
            b = a.get("box")
            box = ({"x": 100 * b[0] / W, "y": 100 * b[1] / H, "w": 100 * (b[2] - b[0]) / W, "h": 100 * (b[3] - b[1]) / H}
                   if b else {"x": 0, "y": 0, "w": 100, "h": 100})
            t = fr["t"]
            results.append({"t": max(0.0, t - 2.0), "dur": 4.0, "best_t": t, "label": rule["label"],
                            "cls": "event" if b else "scene", "conf": 1.0, "box": box})
    return {"event": True, "status": "ready", "rule": rule, "results": results, "total": len(results)}


def live_conf():
    import live_engine
    return live_engine.CONF


# ------------------------------------------------------------- labelling
# Our own videos, labelled in the UI ("תיוג"): for every video, every
# scenario of live/scenarios.json gets its list of [start, end] windows, so a
# scenario that was not marked counts as "did not happen" (false alarms are
# measured too). Same format as eval/ground_truth_uca.json, so
# live/eval_alerts.py reads both.
LIVE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "live")
LABELS = os.path.join(LIVE_DIR, "data", "labels.json")
LABEL_VIDEOS = os.path.join(LIVE_DIR, "data", "videos")
LABELS_LOCK = threading.Lock()


def _scenario_names():
    import json
    with open(os.path.join(LIVE_DIR, "scenarios.json"), encoding="utf-8") as f:
        cfg = json.load(f)
    return {k: v.get("name", k) for k, v in cfg.items() if not k.startswith("_")}


def _read_labels():
    import json
    if not os.path.isfile(LABELS):
        return {"videos": {}}
    with open(LABELS, encoding="utf-8") as f:
        return json.load(f)


def _write_labels(data):
    import json
    os.makedirs(os.path.dirname(LABELS), exist_ok=True)
    tmp = LABELS + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=1)
    os.replace(tmp, LABELS)


class LabelEvent(BaseModel):
    scenario: str
    start: float = Field(..., ge=0)
    end: float = Field(..., ge=0)


class LabelRequest(BaseModel):
    video: str
    duration: float = 0
    events: list[LabelEvent] = []


@app.get("/labels")
def get_labels():
    data = _read_labels()
    have = set(os.listdir(LABEL_VIDEOS)) if os.path.isdir(LABEL_VIDEOS) else set()
    for name, v in data["videos"].items():
        v["file_copied"] = name in have
    return {"scenarios": _scenario_names(), "videos": data["videos"]}


@app.post("/labels")
def save_labels(req: LabelRequest):
    names = _scenario_names()
    bad = sorted({e.scenario for e in req.events} - set(names))
    if bad:
        raise HTTPException(400, f"unknown scenario: {', '.join(bad)}")
    if any(e.end <= e.start for e in req.events):
        raise HTTPException(400, "an event ends before it starts")
    video = safe_name(req.video)
    queries = []
    for sc, title in names.items():
        wins = sorted([round(e.start, 2), round(e.end, 2)] for e in req.events if e.scenario == sc)
        queries.append({"scenario": sc, "text": title, "expected": wins})
    with LABELS_LOCK:
        data = _read_labels()
        data["videos"][video] = {"category": "own", "duration": round(req.duration, 2),
                                 "labelled_at": time.strftime("%Y-%m-%d %H:%M"), "queries": queries}
        _write_labels(data)
    return {"ok": True, "video": video, "events": len(req.events)}


@app.delete("/labels/{video}")
def delete_labels(video: str):
    with LABELS_LOCK:
        data = _read_labels()
        if data["videos"].pop(video, None) is None:
            raise HTTPException(404, "no labels for this video")
        _write_labels(data)
    return {"ok": True}


@app.post("/labels/video")
def upload_label_video(file: UploadFile = File(...)):
    """a copy of the labelled video in live/data/videos, where live/perceive.py looks."""
    os.makedirs(LABEL_VIDEOS, exist_ok=True)
    name = safe_name(file.filename or "video.mp4")
    with open(os.path.join(LABEL_VIDEOS, name), "wb") as f:
        shutil.copyfileobj(file.file, f)
    return {"ok": True, "video": name}


# cleanup of old rendered clips (older than a day) at start-up
for _f in os.listdir(EXPORT_DIR):
    _p = os.path.join(EXPORT_DIR, _f)
    if os.path.isfile(_p) and time.time() - os.path.getmtime(_p) > 86400:
        try:
            os.remove(_p)
        except OSError:
            pass
