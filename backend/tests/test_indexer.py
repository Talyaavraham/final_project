"""Model selection per video and class-name normalisation (no real YOLO)."""

import os

import indexer
from conftest import BlindModel, FakeModel, _Box, _Result


def test_class_names_are_lowercased_renamed_and_body_parts_skipped():
    class M:
        names = {0: "Man", 1: "Human face", 2: "Tree", 3: "backpack"}
    assert indexer.class_name(M, 0) == "person"
    assert indexer.class_name(M, 1) in indexer.SKIP
    assert indexer.class_name(M, 2) == "tree"
    assert indexer.class_name(M, 3) == "bag"


def test_choose_model_picks_the_one_that_sees_the_video(video_file, monkeypatch):
    sharp, blind = FakeModel(), BlindModel()
    monkeypatch.setattr(indexer, "load_model",
                        lambda name=None: {"sharp.pt": sharp, "blind.pt": blind}[name])
    status = {}
    name, model = indexer.choose_model(video_file, ["blind.pt", "sharp.pt"], status)
    assert name == "sharp.pt" and model is sharp
    assert blind.calls == indexer.PROBE_FRAMES      # every model sees the same frames
    assert status["stage"] == "trying sharp.pt"


def test_single_model_is_not_probed(video_file, monkeypatch):
    only = FakeModel()
    monkeypatch.setattr(indexer, "load_model", lambda name=None: only)
    assert indexer.choose_model(video_file, ["only.pt"])[0] == "only.pt"
    assert only.calls == 0


def test_index_records_the_chosen_model(video_file, tmp_path, monkeypatch, no_clip):
    import db as database
    sharp, blind = FakeModel(), BlindModel()
    monkeypatch.setattr(indexer, "MODEL_NAMES", ["blind.pt", "sharp.pt"])
    monkeypatch.setattr(indexer, "load_model",
                        lambda name=None: {"sharp.pt": sharp, "blind.pt": blind}[name])
    dbp = str(tmp_path / "i.db")
    status = indexer.index_video(video_file, "street.mp4", db_path=dbp)
    assert status["status"] == "ready" and status["model"] == "sharp.pt"
    conn = database.connect(dbp)
    assert database.get_video(conn, "street.mp4")["model"] == "sharp.pt"
    assert set(database.list_classes(conn, "street.mp4")) == {"person", "dog", "car"}


def test_extra_models_run_next_to_the_chosen_one(video_file, tmp_path, monkeypatch, no_clip):
    import db as database

    class GunModel(FakeModel):
        """a 'security' detector: sees one gun wherever the fake person is"""
        names = {0: "gun"}

        def track(self, frame, **kw):
            res = FakeModel.track(self, frame, **kw)[0]
            res.boxes = [b for b in res.boxes if b.cls[0] == 0]  # the person's box -> "gun" here
            return [res]

    main_model, gun = FakeModel(), GunModel()
    monkeypatch.setattr(indexer, "MODEL_NAMES", ["main.pt"])
    monkeypatch.setattr(indexer, "EXTRA_MODELS", ["security.pt"])
    monkeypatch.setattr(indexer, "load_model",
                        lambda name=None: {"main.pt": main_model, "security.pt": gun}[name])
    dbp = str(tmp_path / "x.db")
    status = indexer.index_video(video_file, "street.mp4", db_path=dbp)
    assert status["status"] == "ready" and status["model"] == "main.pt+security.pt"
    conn = database.connect(dbp)
    classes = database.list_classes(conn, "street.mp4")
    assert {"person", "dog", "car", "gun"} <= set(classes)
    # the gun's track ids live in their own range, so they never merge with a person's
    gun_ids = {r[1] for r in database.load_detections(conn, "street.mp4") if r[2] == "gun"}
    assert gun_ids and min(gun_ids) >= indexer.EXTRA_ID_BASE


class LoudModel(FakeModel):
    """Finds plenty it is not sure of — a specialised model out of its domain."""
    names = {0: "fruit"}

    def track(self, frame, **kw):
        self.calls += 1
        return [_Result([_Box(0, [0, 0, 10, 10], None, 0.5) for _ in range(6)])]

    predict = track


def test_default_model_keeps_the_video_from_a_loud_specialist(video_file, monkeypatch):
    default, loud = FakeModel(), LoudModel()
    monkeypatch.setattr(indexer, "load_model",
                        lambda name=None: {"coco.pt": default, "loud.pt": loud}[name])
    # loud.pt sums more confidence than coco.pt (6 x 0.5 vs ~2.5 per frame)
    # but nothing above PROBE_CONF, so the default stays
    assert indexer.choose_model(video_file, ["coco.pt", "loud.pt"])[0] == "coco.pt"


# ------------------------------------------------------------- CLIP pass
def fake_clip(monkeypatch):
    """CLIP stand-in: the vector of an image is its normalised mean colour."""
    import numpy as np
    import embeddings
    calls = []

    def embed_images(imgs):
        calls.append(len(imgs))
        v = np.stack([im.reshape(-1, 3).mean(0) for im in imgs]).astype(np.float32) + 1e-3
        return v / np.linalg.norm(v, axis=1, keepdims=True)
    monkeypatch.setattr(embeddings, "available", lambda: True)
    monkeypatch.setattr(embeddings, "embed_images", embed_images)
    return calls


def test_frame_views_cover_a_wide_frame_with_squares():
    import numpy as np
    wide = np.zeros((180, 320, 3), np.uint8)
    views = indexer.frame_views(wide)
    assert [v.shape[:2] for v in views] == [(180, 180), (180, 180)]
    wide[:, :5] = 255; wide[:, -5:] = 255                   # both edges are in some view
    assert views[0][:, 0].max() == 255 and views[1][:, -1].max() == 255
    assert len(indexer.frame_views(np.zeros((1080, 1920, 3), np.uint8))) == 2
    assert len(indexer.frame_views(np.zeros((400, 1920, 3), np.uint8))) == 5
    assert [v.shape[:2] for v in indexer.frame_views(np.zeros((320, 180, 3), np.uint8))] == [(180, 180)] * 2
    assert len(indexer.frame_views(np.zeros((240, 260, 3), np.uint8))) == 1     # nearly square: as it is


def test_view_points_are_spread_over_the_track():
    path = [[float(t), 0, 0, 1, 1] for t in range(10)]
    assert [p[0] for p in indexer.view_points(path, 4)] == [0, 3, 6, 9]
    assert [p[0] for p in indexer.view_points(path[:2], 4)] == [0, 1]
    assert indexer.view_points(path, 0) == [] and indexer.view_points([], 4) == []


def test_context_crop_is_a_square_around_the_box_clipped_to_the_frame():
    import numpy as np
    frame = np.zeros((100, 200, 3), np.uint8)
    assert indexer.context_crop(frame, 90, 40, 100, 60, pad=2.0).shape[:2] == (40, 40)
    assert indexer.context_crop(frame, 0, 0, 10, 20, pad=2.0).shape[:2] == (30, 25)   # corner: clipped


def test_index_stores_scene_views_and_several_views_per_track(video_file, tmp_path, monkeypatch, fake_model):
    import numpy as np
    import db as database
    import embeddings
    fake_clip(monkeypatch)
    dbp = str(tmp_path / "c.db")
    status = indexer.index_video(video_file, "street.mp4", db_path=dbp, model=fake_model)
    assert status["status"] == "ready"
    conn = database.connect(dbp)
    frames = database.load_frames(conn, "street.mp4")
    ts = [t for t, _ in frames]
    assert sorted(set(ts)) == [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]          # every SCENE_EVERY second
    assert len(ts) == 12                                             # 320x180: two square views each
    segs = {s["cls"]: s for s in database.load_segments(conn, "street.mp4")}
    person = embeddings.from_blob(segs["person"]["embedding"]).reshape(-1, 3)
    assert len(person) == 1 + indexer.TRACK_VIEWS                    # best crop + context views
    # row 0 is the tight crop: all red (BGR 40,40,220); the context views let the grey background in
    assert person[0].argmax() == 2 and person[0][2] > 0.95
    assert all(row[0] > person[0][0] for row in person[1:])

    # reembed: same index, vectors recomputed from the stored tracks
    monkeypatch.setattr(indexer, "TRACK_VIEWS", 2)
    indexer.reembed(dbp, os.path.dirname(video_file))
    conn = database.connect(dbp)
    segs = {s["cls"]: s for s in database.load_segments(conn, "street.mp4")}
    again = embeddings.from_blob(segs["person"]["embedding"]).reshape(-1, 3)
    assert len(again) == 3 and np.allclose(again[0], person[0], atol=1e-4)
    assert len(database.load_frames(conn, "street.mp4")) == 12       # replaced, not doubled
