"""eval/uca_ground_truth.py: which sentences count for a scenario, and how a held-out set is picked."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "eval"))
import uca_ground_truth as g  # noqa: E402


def test_tag_only_what_is_on_screen():
    assert g.tag("The house is burning") == ["fire"]
    assert g.tag("The fire truck arrived") == []
    assert g.tag("Police opened fire") == ["weapon"]          # shooting, nothing burns
    assert g.tag("He grabbed her bag and ran") == ["snatch"]


def entry(duration, *sentences):
    return {"duration": duration, "timestamps": [[0.0, 5.0]] * len(sentences), "sentences": list(sentences)}


def test_heldout_skips_the_benchmark_and_caps_the_negatives(tmp_path):
    uca = {"Arson001_x264": entry(60, "a fire started"),          # in the benchmark -> never held out
           "Arson002_x264": entry(60, "a man walked by"),         # Arson: asked "fire" even with no sentence
           "Abuse001_x264": entry(60, "a man pointed a gun"),
           "Abuse002_x264": entry(90, "a man hit a dog"),
           "Abuse003_x264": entry(90, "a woman shouted"),
           "Abuse004_x264": entry(60, "the knife fell")}          # not a chosen scenario -> a negative
    (tmp_path / "Abuse").mkdir()
    for name in list(uca) + ["Abuse099_x264"]:                    # 099: on disk, not in UCA
        (tmp_path / "Abuse" / (name + ".mp4")).write_bytes(b"")
    pos, neg = g.heldout_names(str(tmp_path), uca, {"Arson001_x264"}, ["fire", "weapon"], negative_minutes=2.5)
    assert pos == ["Abuse001_x264", "Arson002_x264"]
    assert set(neg) < {"Abuse002_x264", "Abuse003_x264", "Abuse004_x264"}
    assert sum(uca[n]["duration"] for n in neg) == 150            # 2.5 min: one of 90 s + the one of 60 s
