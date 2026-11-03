"""train/night_dataset.py: class remapping and merging of YOLO exports (no download)."""
import os
import sys

import cv2
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "train"))
import night_dataset as nd  # noqa: E402


def make_export(root, names, layout="roboflow", split="train", imgs=3):
    """A tiny YOLO export: data.yaml + images + labels (one box per image)."""
    os.makedirs(root, exist_ok=True)
    with open(os.path.join(root, "data.yaml"), "w") as f:
        f.write(f"nc: {len(names)}\nnames: {names!r}\n")
    if layout == "roboflow":
        idir, ldir = os.path.join(root, split, "images"), os.path.join(root, split, "labels")
    else:
        idir, ldir = os.path.join(root, "images", split), os.path.join(root, "labels", split)
    os.makedirs(idir), os.makedirs(ldir)
    for i in range(imgs):
        cv2.imwrite(os.path.join(idir, f"f{i}.jpg"), np.zeros((16, 16, 3), np.uint8))
        with open(os.path.join(ldir, f"f{i}.txt"), "w") as f:
            f.write(f"{i % len(names)} 0.5 0.5 0.2 0.3\n")
    return root


def test_class_map_uses_aliases_and_drops_unknown():
    names = {0: "People", 1: "Motorbike", 2: "Table", 3: "Giraffe"}
    m = nd.class_map(names, nd.CLASSES)
    assert m[0] == nd.CLASSES.index("person")
    assert m[1] == nd.CLASSES.index("motorcycle")
    assert m[2] == nd.CLASSES.index("dining table")
    assert 3 not in m
    # subset: only the kept classes get ids, numbered within the subset
    assert nd.class_map(names, ["car", "person"]) == {0: 1}


def test_build_merges_exports_and_makes_val_split(tmp_path):
    a = make_export(str(tmp_path / "a"), ["Bicycle", "People"], "roboflow", imgs=10)
    b = make_export(str(tmp_path / "b"), ["person", "car"], "ultralytics", imgs=4)
    out = str(tmp_path / "ds")
    counts = nd.build(out, [("ex", a), ("own0", b)], keep=["person", "car"],
                      val_ratio=0.2, repeat={"own0": 3})
    train = sorted(os.listdir(os.path.join(out, "images", "train")))
    val = sorted(os.listdir(os.path.join(out, "images", "val")))
    # a: 10 images, 2 to val, 8 train;  b: 4 images, 1 val, 3 train x3 copies
    assert len(val) == 3 and len(train) == 8 + 9
    assert any(n.startswith("own0_") and n.endswith("_r2.jpg") for n in train)
    # bicycle boxes dropped, People/person -> 0, car -> 1
    labels = {}
    for split in ("train", "val"):
        for fn in os.listdir(os.path.join(out, "labels", split)):
            with open(os.path.join(out, "labels", split, fn)) as f:
                labels[fn] = f.read().split()
    ids = {int(v[0]) for v in labels.values() if v}
    assert ids <= {0, 1}
    assert any(not v for fn, v in labels.items() if fn.startswith("ex_"))  # bicycle-only -> background
    assert counts["person"] > 0 and counts["car"] > 0
    with open(os.path.join(out, "data.yaml")) as f:
        yaml_txt = f.read()
    assert "0: person" in yaml_txt and "1: car" in yaml_txt and "images/val" in yaml_txt


def test_read_names_list_and_dict(tmp_path):
    p = tmp_path / "list.yaml"
    p.write_text("nc: 2\nnames: ['Car', 'People']\n")
    assert nd.read_names(str(p)) == {0: "Car", 1: "People"}
    p2 = tmp_path / "dict.yaml"
    p2.write_text("path: x\nnames:\n  0: person\n  1: car\ntrain: a\n")
    assert nd.read_names(str(p2)) == {0: "person", 1: "car"}


def test_frames_extraction(tmp_path):
    video = str(tmp_path / "v.mp4")
    w = cv2.VideoWriter(video, cv2.VideoWriter_fourcc(*"mp4v"), 10, (32, 32))
    for _ in range(50):                      # 5 s at 10 fps
        w.write(np.zeros((32, 32, 3), np.uint8))
    w.release()
    n = nd.extract_frames([video], str(tmp_path / "frames"), every=1.0)
    assert n == 5
    names = sorted(os.listdir(tmp_path / "frames"))
    assert names[0] == "v_000000.0.jpg" and names[-1] == "v_000004.0.jpg"


def test_exdark_missing_reports_short_folders(tmp_path, monkeypatch):
    monkeypatch.setattr(nd, "EXDARK_EXPECTED", {"train/images": 2, "train/labels": 2})
    d = tmp_path / "data" / "train" / "images"
    d.mkdir(parents=True)
    (d / "a.jpg").write_bytes(b"x"), (d / "b.jpg").write_bytes(b"x")
    assert nd.exdark_missing(str(tmp_path)) == ["train/labels 0/2"]
    (tmp_path / "data" / "train" / "labels").mkdir()
    for n in ("a.txt", "b.txt"):
        (tmp_path / "data" / "train" / "labels" / n).write_text("0 0.5 0.5 0.1 0.1\n")
    assert nd.exdark_missing(str(tmp_path)) == []


def test_build_refuses_export_without_yaml(tmp_path):
    with pytest.raises(SystemExit):
        nd.build(str(tmp_path / "out"), [("x", str(tmp_path))])
