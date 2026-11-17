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
