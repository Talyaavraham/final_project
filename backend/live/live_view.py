"""
Live view, stage 1 of the live-action-alerts direction: connect to a camera,
run a pose model (skeleton per person, tracked) and an object detector (knife /
gun), and draw both on the stream in real time. No alerts yet - this stage is
for checking that everything runs fast enough and that the detections look
right, and (with --log) for recording the raw material the action detector
will be built on: per frame, every person's 17 keypoints and every object box.

    python live/live_view.py                                  # webcam 0
    python live/live_view.py --source rtsp://user:pass@ip/stream
    python live/live_view.py --source clip.mp4 --save out.mp4 --log out.jsonl
    python live/live_view.py --extra security.pt              # + gun / knife detector

Keys: q / Esc = quit.
"""
import argparse
import json
import os
import threading
import time

import cv2
import numpy as np

# COCO keypoints (YOLOv8-pose order)
L_SHOULDER, R_SHOULDER = 5, 6
L_WRIST, R_WRIST = 9, 10
WRISTS = (L_WRIST, R_WRIST)
SKELETON = [(5, 7), (7, 9), (6, 8), (8, 10), (5, 6), (5, 11), (6, 12), (11, 12),
            (11, 13), (13, 15), (12, 14), (14, 16), (0, 1), (0, 2), (1, 3), (2, 4)]
KPT_CONF = 0.3          # keypoints below this are not drawn / not used
NEAR_PAD = 0.3          # object box grown by this (x its size) for "wrist on object"

GREEN, RED, YELLOW, WHITE, BLACK = (80, 220, 80), (40, 40, 230), (0, 220, 255), (255, 255, 255), (0, 0, 0)


class FrameSource:
    """
    Reads frames on a background thread. For a live stream only the newest
    frame is kept: if the models are slower than the camera, old frames are
    dropped instead of piling up (otherwise the view drifts further behind
    reality every second). A file is read frame by frame, nothing dropped.
    """

    def __init__(self, source):
        self.live = not (isinstance(source, str) and os.path.isfile(source))
        self.cap = cv2.VideoCapture(int(source) if str(source).isdigit() else source)
        if not self.cap.isOpened():
            raise SystemExit(f"cannot open video source: {source}")
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 25.0
        self.frame, self.t, self.ended = None, 0.0, False
        self.cond = threading.Condition()
        if self.live:
            threading.Thread(target=self._grab, daemon=True).start()

    def _grab(self):
        while True:
            ok, frame = self.cap.read()
            with self.cond:
                if not ok:
                    self.ended = True
                else:
                    self.frame, self.t = frame, time.time()
                self.cond.notify_all()
            if not ok:
                return

    def read(self):
        """-> (frame, capture time) or (None, None) at the end."""
        if not self.live:
            ok, frame = self.cap.read()
            return (frame, time.time()) if ok else (None, None)
        with self.cond:
            while self.frame is None and not self.ended:
                self.cond.wait()
            frame, t, self.frame = self.frame, self.t, None
        return (frame, t) if frame is not None else (None, None)

    def release(self):
        self.cap.release()


def load_detectors(det, extra):
    from ultralytics import YOLO
    models = [YOLO(det)] if det else []
    models += [YOLO(m) for m in extra]
    return models


def detect_objects(models, frame, wanted, conf, imgsz, device):
    """Boxes of the wanted classes from every detector: [{cls, conf, box}]."""
    out = []
    for model in models:
        r = model.predict(frame, conf=conf, imgsz=imgsz, device=device, verbose=False)[0]
        for box, c, s in zip(r.boxes.xyxy.tolist(), r.boxes.cls.tolist(), r.boxes.conf.tolist()):
            name = str(model.names[int(c)]).lower()
            if name in wanted:
                out.append({"cls": name, "conf": round(s, 3), "box": [round(v, 1) for v in box]})
    return out


def people(pose_result):
    """Tracked people from a pose result: [{id, box, kpts (17 x [x, y, conf])}]."""
    r = pose_result
    if r.keypoints is None or r.boxes is None or len(r.boxes) == 0:
        return []
    ids = r.boxes.id.int().tolist() if r.boxes.id is not None else [-1] * len(r.boxes)
    xy = r.keypoints.xy.cpu().numpy()
    kc = r.keypoints.conf.cpu().numpy() if r.keypoints.conf is not None else np.ones(xy.shape[:2])
    out = []
    for i, box in enumerate(r.boxes.xyxy.tolist()):
        kpts = np.concatenate([xy[i], kc[i][:, None]], axis=1)
        out.append({"id": ids[i], "box": [round(v, 1) for v in box],
                    "kpts": [[round(float(x), 1), round(float(y), 1), round(float(c), 3)] for x, y, c in kpts]})
    return out


def near(point, box, pad=NEAR_PAD):
    x, y = point
    x1, y1, x2, y2 = box
    dx, dy = (x2 - x1) * pad, (y2 - y1) * pad
    return x1 - dx <= x <= x2 + dx and y1 - dy <= y <= y2 + dy


def draw(frame, persons, objects, stats):
    for o in objects:
        x1, y1, x2, y2 = map(int, o["box"])
        cv2.rectangle(frame, (x1, y1), (x2, y2), RED, 2)
        label(frame, f"{o['cls']} {o['conf']:.2f}", (x1, y1), RED)
    for p in persons:
        k = p["kpts"]
        for a, b in SKELETON:
            if k[a][2] >= KPT_CONF and k[b][2] >= KPT_CONF:
                cv2.line(frame, (int(k[a][0]), int(k[a][1])), (int(k[b][0]), int(k[b][1])), GREEN, 2)
        for x, y, c in k:
            if c >= KPT_CONF:
                cv2.circle(frame, (int(x), int(y)), 3, GREEN, -1)
        # wrists: bigger, yellow when on top of an object. Debug view only -
        # not an alert (a hand next to a knife on a cutting board is normal)
        for w in WRISTS:
            x, y, c = k[w]
            if c < KPT_CONF:
                continue
            on_obj = any(near((x, y), o["box"]) for o in objects)
            cv2.circle(frame, (int(x), int(y)), 7, YELLOW if on_obj else WHITE, 2)
        x1, y1 = int(p["box"][0]), int(p["box"][1])
        label(frame, f"id {p['id']}", (x1, y1), GREEN)
    text = f"{stats['fps']:.1f} fps | latency {stats['latency'] * 1000:.0f} ms | people {len(persons)} | objects {len(objects)}"
    cv2.rectangle(frame, (0, 0), (len(text) * 9 + 12, 26), BLACK, -1)
    cv2.putText(frame, text, (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, WHITE, 1, cv2.LINE_AA)
    return frame


def label(frame, text, xy, color):
    x, y = xy
    (w, h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    y = max(y, h + 6)
    cv2.rectangle(frame, (x, y - h - 6), (x + w + 4, y), color, -1)
    cv2.putText(frame, text, (x + 2, y - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, BLACK, 1, cv2.LINE_AA)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", default="0", help="camera index, RTSP/HTTP URL or a video file (default 0)")
    ap.add_argument("--pose", default="yolov8n-pose.pt", help="pose model")
    ap.add_argument("--det", default="yolov8n.pt", help="object detector (COCO has 'knife'); '' = none")
    ap.add_argument("--extra", default="", help="more detectors, comma separated, e.g. security.pt")
    ap.add_argument("--objects", default="knife,gun", help="classes to show (comma separated)")
    ap.add_argument("--conf", type=float, default=0.25, help="detector confidence")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--device", default=None, help="'cpu', '0' (GPU) ... default: auto")
    ap.add_argument("--save", default="", help="write the annotated stream to this .mp4")
    ap.add_argument("--log", default="", help="write people keypoints + objects per frame to this .jsonl")
    ap.add_argument("--no-show", action="store_true", help="no window (server / Colab)")
    ap.add_argument("--max-frames", type=int, default=0, help="stop after this many frames (0 = never)")
    args = ap.parse_args()

    from ultralytics import YOLO
    pose = YOLO(args.pose)
    detectors = load_detectors(args.det, [m.strip() for m in args.extra.split(",") if m.strip()])
    wanted = {c.strip().lower() for c in args.objects.split(",") if c.strip()}

    src = FrameSource(args.source)
    writer, log = None, open(args.log, "w", encoding="utf-8") if args.log else None
    n, fps, t_prev, t_start = 0, 0.0, time.time(), time.time()
    try:
        while True:
            frame, t_cap = src.read()
            if frame is None:
                break
            res = pose.track(frame, persist=True, tracker="bytetrack.yaml", imgsz=args.imgsz,
                             device=args.device, verbose=False)[0]
            persons = people(res)
            objects = detect_objects(detectors, frame, wanted, args.conf, args.imgsz, args.device)

            now = time.time()
            fps = 0.9 * fps + 0.1 / max(now - t_prev, 1e-6) if n else 1.0 / max(now - t_prev, 1e-6)
            t_prev = now
            # live: capture -> drawn. file: just the processing time of this frame
            draw(frame, persons, objects, {"fps": fps, "latency": now - t_cap})

            if log:
                t_video = n / src.fps if not src.live else round(now - t_start, 3)
                log.write(json.dumps({"frame": n, "t": round(t_video, 3), "persons": persons,
                                      "objects": objects}) + "\n")
            if args.save:
                if writer is None:
                    h, w = frame.shape[:2]
                    writer = cv2.VideoWriter(args.save, cv2.VideoWriter_fourcc(*"mp4v"),
                                             src.fps if not src.live else 15, (w, h))
                writer.write(frame)
            if not args.no_show:
                cv2.imshow("live view (q to quit)", frame)
                if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                    break
            n += 1
            if args.max_frames and n >= args.max_frames:
                break
    finally:
        src.release()
        if writer:
            writer.release()
        if log:
            log.close()
        cv2.destroyAllWindows()
    print(f"{n} frames, {n / max(time.time() - t_start, 1e-6):.1f} fps average")


if __name__ == "__main__":
    main()
