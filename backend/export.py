"""
Outputs: annotated frames (JPEG), clips (MP4) and a ZIP of everything.

ffmpeg comes from the `imageio-ffmpeg` wheel (bundled binary) or from PATH.
Without ffmpeg, clips are written with OpenCV (no audio, mp4v codec).
"""

import csv
import io
import json
import os
import shutil
import subprocess
import tempfile
import zipfile
from typing import Optional

import cv2
import numpy as np

from search import box_at

BOX_COLOR = (76, 86, 240)      # BGR — same red as the UI
ALT_COLOR = (240, 201, 76)


def ffmpeg_exe() -> Optional[str]:
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:  # noqa: BLE001
        return shutil.which("ffmpeg")


# ------------------------------------------------------------------ frames
def _draw_box(frame, box, label="", color=BOX_COLOR):
    H, W = frame.shape[:2]
    x, y, w, h = box
    x1, y1 = int(x / 100 * W), int(y / 100 * H)
    x2, y2 = int((x + w) / 100 * W), int((y + h) / 100 * H)
    th = max(2, W // 400)
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, th)
    if label:
        scale = max(0.5, W / 1600)
        (tw, tht), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, scale, th)
        ty = y1 - 6 if y1 - tht - 10 > 0 else y2 + tht + 6
        cv2.rectangle(frame, (x1, ty - tht - 4), (x1 + tw + 6, ty + 4), color, -1)
        cv2.putText(frame, label, (x1 + 3, ty), cv2.FONT_HERSHEY_SIMPLEX,
                    scale, (255, 255, 255), th, cv2.LINE_AA)
    return frame


def read_frame(video_path: str, t: float):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError("cannot open video")
    cap.set(cv2.CAP_PROP_POS_MSEC, max(0.0, t) * 1000)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError(f"cannot read frame at {t:.2f}s")
    return frame


def frame_jpeg(video_path: str, t: float, box=None, label: str = "",
               quality: int = 90) -> bytes:
    """One frame as JPEG bytes, with the box drawn when given ([x,y,w,h] %)."""
    frame = read_frame(video_path, t)
    if box:
        _draw_box(frame, box, label)
    ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise RuntimeError("jpeg encode failed")
    return buf.tobytes()


# ------------------------------------------------------------------- clips
def cut_clip(video_path: str, t0: float, t1: float, out_path: str,
             matches: Optional[list] = None) -> str:
    """
    Write [t0, t1] of the video to out_path (.mp4).
    matches: list of {path, label} to draw as tracked boxes -> the clip is
    re-rendered frame by frame; without it the clip is cut with audio.
    """
    t0, t1 = max(0.0, t0), max(t0 + 0.5, t1)
    exe = ffmpeg_exe()
    if matches:
        return _annotated_clip(video_path, t0, t1, out_path, matches, exe)
    if exe:
        cmd = [exe, "-y", "-loglevel", "error", "-ss", f"{t0:.3f}", "-i", video_path,
               "-t", f"{t1 - t0:.3f}", "-c:v", "libx264", "-preset", "veryfast",
               "-crf", "23", "-pix_fmt", "yuv420p", "-movflags", "+faststart",
               "-c:a", "aac", out_path]
        subprocess.run(cmd, check=True, stdin=subprocess.DEVNULL)
        return out_path
    return _annotated_clip(video_path, t0, t1, out_path, [], None)


def _annotated_clip(video_path, t0, t1, out_path, matches, exe):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError("cannot open video")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.set(cv2.CAP_PROP_POS_MSEC, t0 * 1000)

    if exe:
        proc = subprocess.Popen(
            [exe, "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "bgr24",
             "-s", f"{W}x{H}", "-r", f"{fps:.3f}", "-i", "-",
             "-c:v", "libx264", "-preset", "veryfast", "-crf", "23",
             "-pix_fmt", "yuv420p", "-movflags", "+faststart", out_path],
            stdin=subprocess.PIPE)
        write = proc.stdin.write
    else:
        writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))
        write = writer.write

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        t = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000
        if t > t1:
            break
        for i, m in enumerate(matches):
            b = box_at(m.get("path") or [], t)
            if b is None and m.get("box") and m.get("t") is not None:
                if m["t"] - 0.5 <= t <= m["t"] + m.get("dur", 1):
                    bb = m["box"]
                    b = [bb["x"], bb["y"], bb["w"], bb["h"]]
            if b is not None:
                _draw_box(frame, b, m.get("label", ""),
                          BOX_COLOR if i == 0 else ALT_COLOR)
        write(frame.tobytes() if exe else frame)
    cap.release()
    if exe:
        proc.stdin.close()
        proc.wait()
        if proc.returncode != 0:
            raise RuntimeError("ffmpeg failed")
    else:
        writer.release()
    return out_path


# --------------------------------------------------------------------- zip
def _safe(s: str) -> str:
    return "".join(c if c.isalnum() or c in "-_." else "_" for c in s)[:60]


def build_zip(video_path: str, video_name: str, matches: list, query: str = "",
              clips: bool = True, annotate: bool = True, pad: float = 1.0) -> bytes:
    """
    matches: the /search results. Produces
      frames/NN_label_mm-ss.jpg  clips/NN_label_mm-ss.mp4  results.json  results.csv
    """
    out = io.BytesIO()
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as zf, \
            tempfile.TemporaryDirectory() as tmp:
        rows = []
        for i, m in enumerate(matches):
            t = m.get("best_t", m["t"])
            label = m.get("label") or m.get("cls", "match")
            stem = f"{i + 1:02d}_{_safe(label)}_{int(t) // 60:02d}-{int(t) % 60:02d}"
            box = m.get("box")
            jpg = frame_jpeg(video_path, t,
                             [box["x"], box["y"], box["w"], box["h"]] if box and annotate else None,
                             label if annotate else "")
            zf.writestr(f"frames/{stem}.jpg", jpg)
            clip_name = ""
            if clips:
                clip_path = os.path.join(tmp, f"{stem}.mp4")
                cut_clip(video_path, m["t"] - pad, m["t"] + m.get("dur", 1) + pad,
                         clip_path, [m] if annotate else None)
                clip_name = f"clips/{stem}.mp4"
                zf.write(clip_path, clip_name)
            rows.append({
                "n": i + 1, "time": round(m["t"], 2),
                "duration": round(m.get("dur", 0), 2), "class": m.get("cls"),
                "label": label, "confidence": round(m.get("conf", 0), 3),
                "track": m.get("track"), "entity": m.get("entity"),
                "frame": f"frames/{stem}.jpg", "clip": clip_name,
            })
        zf.writestr("results.json", json.dumps(
            {"video": video_name, "query": query, "matches": rows}, indent=2,
            ensure_ascii=False))
        buf = io.StringIO()
        w = csv.DictWriter(buf, fieldnames=list(rows[0]) if rows else ["n"])
        w.writeheader()
        w.writerows(rows)
        zf.writestr("results.csv", buf.getvalue())
    return out.getvalue()
