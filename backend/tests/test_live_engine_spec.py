"""live_engine.model_spec + the per-model class filter (security.pt for fire, security_v3.pt for knives and guns)."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import live_engine as le  # noqa: E402


def test_model_spec():
    assert le.model_spec("yolov8n.pt") == ("yolov8n.pt", None)
    assert le.model_spec("security_v3.pt=gun+Knife") == ("security_v3.pt", {"gun", "knife"})
    assert le.model_spec("x.pt=handbag") == ("x.pt", {"bag"})          # RENAME applies


class _Boxes:
    def __init__(self, rows):
        self.xyxy = np.array([r[0] for r in rows], float).reshape(-1, 4)
        self.cls = np.array([r[1] for r in rows], float)
        self.conf = np.array([r[2] for r in rows], float)


class _Det:
    def __init__(self, names, rows):
        self.names, self.rows, self.calls = names, rows, 0

    def predict(self, *a, **k):
        self.calls += 1
        return [type("R", (), {"boxes": _Boxes(self.rows)})()]


class _Pose:
    def track(self, *a, **k):
        return [type("R", (), {"boxes": None, "keypoints": None})()]


def test_each_detector_reports_only_its_classes(monkeypatch):
    names = {0: "gun", 1: "knife", 2: "fire", 3: "smoke"}
    old = _Det(names, [([0, 0, 5, 5], 1, 0.9), ([0, 0, 9, 9], 2, 0.8)])     # knife + fire
    new = _Det(names, [([1, 1, 5, 5], 1, 0.7), ([1, 1, 9, 9], 2, 0.6)])     # knife + fire
    e = le.LiveEngine()
    e.pose, e.detectors, e.only = _Pose(), [old, new], [{"fire", "smoke"}, {"gun", "knife"}]
    monkeypatch.setattr(le, "_people", lambda r: [])
    _, objs = e.perceive(np.zeros((10, 10, 3), np.uint8))
    assert sorted((o["cls"], o["conf"]) for o in objs) == [("fire", 0.8), ("knife", 0.7)]
    _, objs = e.perceive(np.zeros((10, 10, 3), np.uint8), wanted={"knife"})
    assert [o["cls"] for o in objs] == ["knife"] and old.calls == 1      # security.pt=fire+smoke skipped
    assert e.classes() == ["fire", "gun", "knife", "person", "smoke"]
