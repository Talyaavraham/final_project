"""Model selection per video and class-name normalisation (no real YOLO)."""

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
