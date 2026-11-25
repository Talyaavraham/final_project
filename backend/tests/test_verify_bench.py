"""eval/verify_bench.py: the scoring around the vision-language model (the model itself is not loaded)."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "eval"))
import verify_bench as vb       # noqa: E402


def seg(t, p, dur=2.0):
    return {"t": t, "dur": dur, "p": p}


ROWS = [
    # the window 10-20 is found by a confident segment; two wrong segments, one of them confident too
    {"video": "a.mp4", "scenario": "fire", "expected": [[10, 20]], "segs": [seg(12, 0.9), seg(40, 0.2), seg(60, 0.7)]},
    # a video without fire: every segment is a false alarm
    {"video": "b.mp4", "scenario": "fire", "expected": [], "segs": [seg(5, 0.1), seg(30, 0.3)]},
    # the only right segment gets a low score: a threshold loses the window
    {"video": "a.mp4", "scenario": "weapon", "expected": [[50, 60]], "segs": [seg(52, 0.1), seg(80, 0.6)]},
]


def test_threshold_drops_segments_and_rescores():
    assert vb.counts(ROWS, {}) == {"fire": [1, 4, 0], "weapon": [1, 1, 0]}          # tp, fp, fn
    assert vb.counts(ROWS, {"fire": 0.5}) == {"fire": [1, 1, 0], "weapon": [1, 1, 0]}
    assert vb.counts(ROWS, {"weapon": 0.5})["weapon"] == [0, 1, 1]                  # the window is lost
    assert vb.total(vb.counts(ROWS, {"fire": 0.5})) == [2, 2, 0]


def test_best_threshold_and_auc():
    best = vb.best_thresholds(ROWS)
    assert best["fire"] == 0.8          # only the right segment is left: precision 1
    assert best["weapon"] == 0.0        # any threshold removes the right segment first
    assert vb.auc([0.9], [0.2, 0.7, 0.1, 0.3]) == 1.0
    assert vb.auc([0.1], [0.6]) == 0.0
    assert vb.right(seg(12, 0.9), [[10, 20]]) and not vb.right(seg(40, 0.2), [[10, 20]])
