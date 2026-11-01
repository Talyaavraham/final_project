"""
Shared fixtures: a tiny synthetic video and a fake detector, so the whole
pipeline (index -> search -> export) runs in a second without YOLO weights.
"""

import os
import sys

import cv2
import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

FPS, W, H, SECONDS = 25, 320, 180, 6


@pytest.fixture(scope="session")
def video_file(tmp_path_factory):
    """6 s clip: a red square walks left->right, a small brown square follows it,
    a blue square stays parked in the corner."""
    path = str(tmp_path_factory.mktemp("vid") / "street.mp4")
    out = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), FPS, (W, H))
    for i in range(FPS * SECONDS):
        frame = np.full((H, W, 3), 200, np.uint8)
        x = int(20 + i * 1.0)
        cv2.rectangle(frame, (x, 60), (x + 30, 140), (40, 40, 220), -1)     # person (red)
        cv2.rectangle(frame, (x + 40, 120), (x + 60, 140), (30, 60, 110), -1)  # dog (brown)
        cv2.rectangle(frame, (290, 20), (315, 50), (220, 60, 30), -1)        # car (blue)
        out.write(frame)
    out.release()
    return path


class _Box:
    def __init__(self, cls, xyxy, tid, conf):
        self.cls = [cls]
        self.xyxy = [xyxy]
        self.id = [tid] if tid is not None else None
        self.conf = [conf]


class _Result:
    def __init__(self, boxes):
        self.boxes = boxes


class FakeModel:
    """Mimics ultralytics YOLO.track() on the synthetic video above."""
    names = {0: "person", 2: "car", 16: "dog"}

    def __init__(self):
        self.calls = 0

    def track(self, frame, **kw):
        # locate the squares by colour instead of trusting the call count
        b, g, r = cv2.split(frame)
        boxes = []
        red = np.argwhere((r > 180) & (b < 80))
        if len(red):
            y0, x0 = red.min(0); y1, x1 = red.max(0)
            boxes.append(_Box(0, [x0, y0, x1 + 1, y1 + 1], 1, 0.9))
        brown = np.argwhere((r > 90) & (r < 130) & (b < 50))
        if len(brown):
            y0, x0 = brown.min(0); y1, x1 = brown.max(0)
            boxes.append(_Box(16, [x0, y0, x1 + 1, y1 + 1], 2, 0.8))
        blue = np.argwhere((b > 180) & (r < 80))
        if len(blue):
            y0, x0 = blue.min(0); y1, x1 = blue.max(0)
            boxes.append(_Box(2, [x0, y0, x1 + 1, y1 + 1], 3, 0.85))
        self.calls += 1
        return [_Result(boxes)]

    predict = track            # the model probe uses predict()


class BlindModel(FakeModel):
    """Sees nothing — the wrong model for the synthetic video."""
    names = {0: "character"}

    def track(self, frame, **kw):
        self.calls += 1
        return [_Result([])]

    predict = track


@pytest.fixture
def fake_model():
    return FakeModel()


@pytest.fixture
def no_clip(monkeypatch):
    import embeddings
    monkeypatch.setattr(embeddings, "_ENABLED", False)
