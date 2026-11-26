"""train/pack_exports.py + train/security_dataset.py: merging public exports into the security training set (no download)."""
import os
import sys
import zipfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "train"))
import pack_exports as pe  # noqa: E402
import security_dataset as sd  # noqa: E402


def export(root, files, names=("knife",)):
    """files: {"train/abc.rf.01ab.jpg": "label text", ...} -> a Roboflow-layout export."""
    os.makedirs(root)
    with open(os.path.join(root, "data.yaml"), "w") as f:
        f.write(f"nc: {len(names)}\nnames: {list(names)!r}\n")
    for rel, label in files.items():
        split, fn = rel.split("/")
        for sub in ("images", "labels"):
            os.makedirs(os.path.join(root, split, sub), exist_ok=True)
        with open(os.path.join(root, split, "images", fn), "wb") as f:
            f.write(rel.encode())                       # content = where it came from
        with open(os.path.join(root, split, "labels", os.path.splitext(fn)[0] + ".txt"), "w") as f:
            f.write(label)
    return root


BOX = "0 0.5 0.5 0.2 0.3\n"


def test_pack_keeps_every_source_image_once_and_validation_wins(tmp_path):
    a = export(str(tmp_path / "a"), {
        "train/one_jpg.rf.aaaa1111.jpg": BOX, "train/one_jpg.rf.bbbb2222.jpg": BOX,    # two augmented copies
        "train/two_jpg.rf.cccc3333.jpg": BOX,
        "test/three_jpg.rf.dddd4444.jpg": BOX})
    b = export(str(tmp_path / "b"), {
        "valid/two_jpg.rf.eeee5555.jpg": BOX,                                          # a's training image
        "train/four_jpg.rf.ffff6666.jpg": "0 0.1 0.2 0.5 0.2 0.5 0.6 0.1 0.6\n"})      # polygon
    out = str(tmp_path / "packed.zip")
    names, stats, boxes = pe.pack([a, b], out)
    z = zipfile.ZipFile(out)
    files = sorted(n for n in z.namelist() if "/images/" in n)
    assert files == ["train/images/four_jpg.jpg", "train/images/one_jpg.jpg",
                     "train/images/three_jpg.jpg", "valid/images/two_jpg.jpg"]
    assert z.read("train/images/one_jpg.jpg") == b"train/one_jpg.rf.aaaa1111.jpg"     # first copy
    assert z.read("valid/images/two_jpg.jpg") == b"train/two_jpg.rf.cccc3333.jpg"     # first export's pixels, in validation
    assert z.read("train/labels/four_jpg.txt").decode().split() == ["0", "0.300000", "0.400000", "0.400000", "0.400000"]
    assert names == ["knife"] and boxes == {"knife": 4} and stats["copies dropped"] == 2


def test_packed_knives_join_the_security_set(tmp_path):
    src = export(str(tmp_path / "src"), {f"train/k{i}_jpg.rf.{i:08x}.jpg": BOX for i in range(3)}
                 | {"valid/v_jpg.rf.99999999.jpg": BOX})
    z = str(tmp_path / "knives.zip")
    pe.pack([src], z)
    zipfile.ZipFile(z).extractall(str(tmp_path / "knives"))
    w = sd.Writer(str(tmp_path / "ds"))
    w.add("weap_a", b"x", ["0 0.5 0.5 0.1 0.1"], split="val")
    base_val = list(w.files["val"])
    sd.add_own(w, str(tmp_path / "knives"))                  # 3 training images: "a few own frames" -> 4 copies
    assert w.n == {"train": 12, "val": 2} and w.boxes["knife"] == 13
    sd.add_own(w, str(tmp_path / "knives"), repeat=1)
    assert w.n["train"] == 15
    w.yaml()
    w.yaml("data_base.yaml", base_val)
    ds = str(tmp_path / "ds")
    assert "val: images/val" in open(os.path.join(ds, "data.yaml")).read()
    assert "val: data_base_val.txt" in open(os.path.join(ds, "data_base.yaml")).read()
    listed = open(os.path.join(ds, "data_base_val.txt")).read().split("\n")[:-1]
    assert [os.path.basename(p) for p in listed] == ["weap_a.jpg"]


def test_benchmark_videos_are_left_out_and_a_video_stays_in_one_split(tmp_path):
    import json
    import re
    files = {f"{s}/Robbery001_x264_mp4-{i:04d}_jpg.rf.{i:08x}.jpg": BOX for i, s in enumerate(["train", "valid", "test"])}
    files |= {f"{s}/Robbery050_x264_mp4-{i:04d}_jpg.rf.{i + 16:08x}.jpg": BOX
              for i, s in enumerate(["train", "valid", "train", "test"])}          # one video on both sides of the split
    files |= {"valid/photo_jpg.rf.000000aa.jpg": BOX}
    src = export(str(tmp_path / "cctv"), files, names=("pistol",))
    gt = str(tmp_path / "gt.json")
    json.dump({"videos": {"Robbery001_x264.mp4": {}}}, open(gt, "w"))
    assert pe.excluded_videos([gt]) == {"robbery001_x264"}
    out = str(tmp_path / "packed.zip")
    names, stats, boxes = pe.pack([src], out, pe.excluded_videos([gt]), re.compile(r"^([a-z]+\d+_x264)_"))
    imgs = [n for n in zipfile.ZipFile(out).namelist() if "/images/" in n]
    assert not any("robbery001" in n for n in imgs) and stats["left out (benchmark videos)"] == 3
    video = [n for n in imgs if "robbery050" in n]
    assert len(video) == 4 and len({n.split("/")[0] for n in video}) == 1          # all its frames together
    assert "valid/images/photo_jpg.jpg" in imgs                                    # not a video frame: Roboflow's split


def test_rotated_training_copies_are_turned_back(tmp_path):
    import cv2
    import numpy as np
    # a box in the top-left corner; one counter-clockwise quarter turn puts it in the bottom-left corner
    assert pe.turn_lines(["0 0.100000 0.200000 0.100000 0.300000"], 1) == ["0 0.200000 0.900000 0.300000 0.100000"]
    assert pe.turn_lines(["0 0.100000 0.200000 0.100000 0.300000"], 4) == ["0 0.100000 0.200000 0.100000 0.300000"]
    src = export(str(tmp_path / "rot"), {"train/a_jpg.rf.00000001.jpg": "0 0.1 0.2 0.1 0.3\n",
                                         "train/b_jpg.rf.00000002.jpg": BOX, "valid/c_jpg.rf.00000003.jpg": BOX})
    for split, fn in (("train", "a_jpg.rf.00000001.jpg"), ("train", "b_jpg.rf.00000002.jpg"), ("valid", "c_jpg.rf.00000003.jpg")):
        cv2.imwrite(os.path.join(src, split, "images", fn), np.zeros((40, 80, 3), np.uint8))    # wide images
    asked = []

    def fake_upright(img):                  # the first image asked about lies on its side, the second cannot be told
        asked.append(img.shape[:2])
        return 1 if len(asked) == 1 else None

    out = str(tmp_path / "packed.zip")
    names, stats, boxes = pe.pack([src], out, upright=fake_upright)
    z = zipfile.ZipFile(out)
    shape = lambda n: cv2.imdecode(np.frombuffer(z.read(n), np.uint8), cv2.IMREAD_COLOR).shape[:2]
    assert shape("train/images/a_jpg.jpg") == (80, 40)                              # turned
    assert z.read("train/labels/a_jpg.txt").decode().split() == ["0", "0.200000", "0.900000", "0.300000", "0.100000"]
    assert shape("train/images/b_jpg.jpg") == (40, 80)                              # unclear: kept as it is
    assert shape("valid/images/c_jpg.jpg") == (40, 80) and len(asked) == 2          # validation copies are never asked about
    assert stats["turned upright"] == 1 and stats["orientation unclear, kept as it is"] == 1


def test_weapondet_lines_maps_and_drops_ambiguous():
    names = ["weapons", "Aggressor", "Knife", "Pistol", "Person"]
    lines = sd.weapondet_lines([[10, 20, 30, 40], [0, 0, 50, 50], [5, 5, 90, 90]], [2, 3, 4], names, 100, 200)
    assert lines == ["1 0.250000 0.200000 0.300000 0.200000", "0 0.250000 0.125000 0.500000 0.250000"]
    assert sd.weapondet_lines([[0, 0, 5, 5]], [0], names, 100, 100) is None      # "weapons": which one?


def test_cctv_look_keeps_size_and_writes_jpeg():
    import io
    import random
    from PIL import Image
    out = sd.cctv_look(Image.new("RGB", (640, 480), (200, 30, 30)), random.Random(0))
    im = Image.open(io.BytesIO(out))
    assert im.size == (640, 480) and im.format == "JPEG"


def test_add_weapondet_on_a_fake_dataset(tmp_path):
    from PIL import Image
    datasets = __import__("pytest").importorskip("datasets")
    feats = datasets.Features({
        "image": datasets.Image(), "width": datasets.Value("int32"), "height": datasets.Value("int32"),
        "objects": datasets.Sequence({"bbox": datasets.Sequence(datasets.Value("float32"), length=4),
                                      "category": datasets.ClassLabel(names=["weapons", "Knife", "Pistol", "Person"])})})
    img = Image.new("RGB", (100, 100))
    rows = {"image": [img, img, img], "width": [100] * 3, "height": [100] * 3,
            "objects": [{"bbox": [[10, 10, 20, 20]], "category": [1]},
                        {"bbox": [[10, 10, 20, 20]], "category": [0]},
                        {"bbox": [[1, 1, 30, 30], [40, 40, 9, 9]], "category": [3, 2]}]}
    dd = datasets.DatasetDict({"train": datasets.Dataset.from_dict(rows, features=feats),
                               "validation": datasets.Dataset.from_dict(rows, features=feats)})
    w = sd.Writer(str(tmp_path / "ds"))
    st = sd.add_weapondet(w, dd)
    assert st["dropped"] == 2 and st["images"] == 4 and st["cctv"] == 2
    assert w.n == {"train": 4, "val": 2} and w.boxes["knife"] == 3 and w.boxes["gun"] == 3


def test_add_dangerous_maps_machete_drops_rifle(tmp_path):
    from PIL import Image
    root = tmp_path / "Dangerous Items"
    (root).mkdir()
    (root / "data.yaml").write_text("names: ['machete', 'knife', 'baseball bat', 'rifle', 'gun']\n")
    labels = {"train/a": "0 0.5 0.5 0.2 0.2\n", "train/b": "3 0.5 0.5 0.2 0.2\n4 0.1 0.1 0.1 0.1\n",
              "train/c": "2 0.5 0.5 0.2 0.2\n", "val/d": "1 0.5 0.5 0.1 0.1\n4 0.3 0.3 0.1 0.1\n"}
    for k, v in labels.items():
        split, stem = k.split("/")
        for sub in ("images", "labels"):
            (root / sub / split).mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (64, 64)).save(root / "images" / split / (stem + ".jpg"))
        (root / "labels" / split / (stem + ".txt")).write_text(v)
    w = sd.Writer(str(tmp_path / "ds"))
    st = sd.add_dangerous(w, str(root))
    assert st == {"images": 3, "dropped_rifle": 1, "cctv": 1}       # a (+cctv copy), c (bat: background), d
    assert w.n == {"train": 3, "val": 1} and w.boxes == {"gun": 1, "knife": 3, "fire": 0, "smoke": 0}   # machete x2 (cctv copy) + knife
    import pytest
    with pytest.raises(SystemExit):
        sd.check_enough(w)
