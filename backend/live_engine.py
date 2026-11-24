"""
Live mode: frames from the browser's camera -> people (pose skeleton, tracked)
+ objects (detectors) -> alert rules (alert_rules.py).

The browser sends one JPEG at a time over the WebSocket /live/ws and waits for
the answer before sending the next one, so the server always works on the
newest frame and nothing piles up when the models are slower than the camera.

Env:
    LIVE_POSE=yolov8n-pose.pt       pose model (yolov8s-pose.pt: better, slower)
    LIVE_DET=yolov8n.pt             object detector(s), comma separated (COCO has 'knife')
    LIVE_EXTRA=security.pt          extra detectors (gun / knife / fire / smoke); default: security.pt if present
    LIVE_CONF=0.25  LIVE_IMGSZ=640  LIVE_DEVICE=0 (GPU) / cpu
"""
import os
import threading
import time

import cv2
import numpy as np

import alert_rules

HERE = os.path.dirname(os.path.abspath(__file__))
POSE_MODEL = os.environ.get("LIVE_POSE", "yolov8n-pose.pt")
DET_MODELS = [m.strip() for m in os.environ.get("LIVE_DET", "yolov8n.pt").split(",") if m.strip()]
_default_extra = "security.pt" if os.path.isfile(os.path.join(HERE, "security.pt")) else ""
EXTRA_MODELS = [m.strip() for m in os.environ.get("LIVE_EXTRA", _default_extra).split(",") if m.strip()]
CONF = float(os.environ.get("LIVE_CONF", "0.25"))
IMGSZ = int(os.environ.get("LIVE_IMGSZ", "640"))
DEVICE = os.environ.get("LIVE_DEVICE") or None
# A detector runs only when a rule asks for one of its classes: a "raise a
# hand" rule needs the pose model alone, which is ~3x faster on a CPU than
# pose + COCO + security.pt on every frame.


def _resolve(name):
    """model files that sit next to this file (security.pt) work from any cwd."""
    local = os.path.join(HERE, name)
    return local if os.path.isfile(local) else name


class LiveEngine:
    def __init__(self):
        self.lock = threading.Lock()          # one frame through the models at a time
        self.rules_lock = threading.Lock()
        self.pose = None
        self.detectors = []
        self.error = None
        self.rules = []
        self.evaluator = alert_rules.Evaluator()

    # ------------------------------------------------------------ models
    def load(self):
        if self.pose is not None:
            return
        from ultralytics import YOLO
        self.pose = YOLO(_resolve(POSE_MODEL))
        self.detectors = [YOLO(_resolve(m)) for m in DET_MODELS + EXTRA_MODELS]

    def classes(self):
        """class names the detectors know (for the rule parser)."""
        if not self.detectors:
            return alert_rules.DEFAULT_CLASSES
        names = {"person"}
        for d in self.detectors:
            names |= {alert_rules.RENAME.get(str(n).lower(), str(n).lower()) for n in d.names.values()}
        return sorted(names)

    def info(self):
        return {"pose": POSE_MODEL, "detectors": DET_MODELS + EXTRA_MODELS,
                "loaded": self.pose is not None, "error": self.error}

    # ------------------------------------------------------------ rules
    def add_rule(self, rule):
        """-> the rule kept: an existing one when the same alert is already set."""
        key = lambda r: (r["kind"], r["object"], r["min_people"], r.get("hands", 0))
        with self.rules_lock:
            for r in self.rules:
                if key(r) == key(rule):
                    return r, False
            self.rules.append(rule)
        print(f"[live] rule: {rule['label']}  kind={rule['kind']} object={rule['object'] or '-'} "
              f"hands={rule.get('hands', 0)} duration={rule['duration']}s  by={rule['source']}  <- \"{rule['text']}\"", flush=True)
        return rule, True

    def remove_rule(self, rule_id):
        with self.rules_lock:
            n = len(self.rules)
            self.rules = [r for r in self.rules if r["id"] != rule_id]
            return len(self.rules) < n

    def list_rules(self):
        with self.rules_lock:
            return list(self.rules)

    # ------------------------------------------------------------ frames
    def start_stream(self):
        """a new camera / connection: new track ids, no timing left over."""
        with self.lock:
            if self.pose is not None:
                for tracker in getattr(getattr(self.pose, "predictor", None), "trackers", None) or []:
                    tracker.reset()
            self.evaluator.reset()

    def perceive(self, frame, wanted=None, conf=None):
        """One frame -> (persons, objects). The pose model tracks people across
        calls. `wanted`: object classes to detect (a detector that knows none
        of them is skipped); None = every class except person. Call with the
        lock held. Shared by the live server and live/perceive.py."""
        self.load()
        r = self.pose.track(frame, persist=True, tracker="bytetrack.yaml", imgsz=IMGSZ,
                            device=DEVICE, verbose=False)[0]
        persons = _people(r)
        objects = []
        for d in self.detectors:
            names = {str(n).lower() for n in d.names.values()}
            if wanted is not None and not wanted & {alert_rules.RENAME.get(n, n) for n in names}:
                continue
            res = d.predict(frame, conf=CONF if conf is None else conf, imgsz=IMGSZ, device=DEVICE, verbose=False)[0]
            for box, c, sc in zip(res.boxes.xyxy.tolist(), res.boxes.cls.tolist(), res.boxes.conf.tolist()):
                name = str(d.names[int(c)]).lower()
                name = alert_rules.RENAME.get(name, name)
                if name != "person" and (wanted is None or name in wanted):
                    objects.append({"cls": name, "conf": round(sc, 3), "box": [round(v, 1) for v in box]})
        return persons, objects

    def process(self, jpeg: bytes) -> dict:
        frame = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)
        if frame is None:
            return {"error": "bad frame"}
        t0 = time.time()
        with self.lock:
            rules = self.list_rules()
            persons, objects = self.perceive(frame, {x["object"] for x in rules if x["object"]})
            alerts, progress = self.evaluator.step(rules, persons, objects, time.monotonic())
        for a in alerts:   # visible in the server window, for debugging
            print(f"[live] ALERT: {a['label']}  (rule {a['rule']}, person {a['person']})", flush=True)
        h, w = frame.shape[:2]
        return {"w": w, "h": h, "persons": persons, "objects": objects, "alerts": alerts,
                "progress": progress, "ms": round((time.time() - t0) * 1000)}


def _people(r):
    if r.keypoints is None or r.boxes is None or len(r.boxes) == 0:
        return []
    ids = r.boxes.id.int().tolist() if r.boxes.id is not None else [-1] * len(r.boxes)
    xy = r.keypoints.xy.cpu().numpy()
    kc = r.keypoints.conf.cpu().numpy() if r.keypoints.conf is not None else np.ones(xy.shape[:2])
    out = []
    for i, box in enumerate(r.boxes.xyxy.tolist()):
        out.append({"id": ids[i], "box": [round(v, 1) for v in box],
                    "kpts": [[round(float(x), 1), round(float(y), 1), round(float(c), 3)]
                             for (x, y), c in zip(xy[i], kc[i])]})
    return out


engine = LiveEngine()
