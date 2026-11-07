import numpy as np

from search import (filter_segments, merge_entities, near, semantic_rank,
                    track_segments)


def seg(cls, t, dur, x, y, w=10, h=20, step=0.5, tid=1, color="red", dx=0.0):
    """Build a segment through track_segments from a synthetic path."""
    rows = []
    n = int(dur / step) + 1
    for i in range(n):
        rows.append((t + i * step, tid, cls, 0.9, x + dx * i, y, w, h, color))
    return track_segments(rows, duration=100, W=16, H=9)[0]


def test_track_split_on_gap():
    rows = [(0, 1, "person", .9, 10, 10, 10, 20, "red"),
            (1, 1, "person", .9, 11, 10, 10, 20, "red"),
            (10, 1, "person", .9, 50, 10, 10, 20, "red"),   # > MAX_GAP later
            (11, 1, "person", .9, 51, 10, 10, 20, "red")]
    segs = track_segments(rows)
    assert len(segs) == 2
    assert segs[0]["t"] == 0 and segs[1]["t"] == 10


def test_single_weak_detection_dropped():
    assert track_segments([(0, 1, "dog", .4, 10, 10, 10, 10, "brown")]) == []
    assert len(track_segments([(0, 1, "dog", .6, 10, 10, 10, 10, "brown")])) == 1


def test_speed_and_actions():
    walker = seg("person", 0, 5, 10, 40, dx=2.0)      # ~0.36 heights/s
    runner = seg("person", 0, 5, 10, 40, dx=8.0)      # ~1.4 heights/s
    parked = seg("car", 0, 5, 80, 10)
    assert filter_segments([walker, runner], ["person"], [], ["walking"], None) == [walker]
    assert filter_segments([walker, runner], ["person"], [], ["running"], None) == [runner]
    assert filter_segments([parked, walker], [], [], ["parked"], None) == [parked]


def test_color_applies_to_subject_only():
    red_person = seg("person", 0, 3, 10, 40, color="red")
    blue_person = seg("person", 0, 3, 30, 40, color="blue", tid=2)
    dog = seg("dog", 0, 3, 14, 60, color="brown", tid=3)
    out = filter_segments([red_person, blue_person, dog], ["person", "dog"],
                          ["red"], [], None, require_near=False)
    assert red_person in out and dog in out and blue_person not in out


def test_time_range_open_end():
    a = seg("person", 0, 3, 10, 40)
    b = seg("person", 50, 3, 10, 40, tid=2)
    assert filter_segments([a, b], [], [], [], [40, None]) == [b]
    assert filter_segments([a, b], [], [], [], [0, 10]) == [a]


def test_near_requires_spatial_proximity():
    person = seg("person", 0, 4, 10, 40)
    close_dog = seg("dog", 0, 4, 22, 60, w=6, h=6, tid=2)       # 12% away
    far_dog = seg("dog", 0, 4, 80, 60, w=6, h=6, tid=3)         # 70% away
    assert near(person, close_dog)
    assert not near(person, far_dog)
    out = filter_segments([person, close_dog, far_dog], ["person", "dog"], [], [], None)
    assert close_dog in out and far_dog not in out and person in out
    # without the spatial check the far dog overlaps in time and is kept
    out = filter_segments([person, close_dog, far_dog], ["person", "dog"], [], [], None,
                          require_near=False)
    assert far_dog in out


def test_near_false_when_no_time_overlap():
    a = seg("person", 0, 2, 10, 40)
    b = seg("dog", 10, 2, 12, 40, tid=2)
    assert not near(a, b)


def test_merge_entities_by_embedding():
    v = np.array([1.0, 0, 0], np.float32)
    a = dict(seg("person", 0, 2, 10, 40, tid=1), embedding_vec=v)
    b = dict(seg("person", 10, 2, 10, 40, tid=2), embedding_vec=v)
    c = dict(seg("person", 20, 2, 10, 40, tid=3),
             embedding_vec=np.array([0, 1.0, 0], np.float32))
    d = dict(seg("person", 1, 2, 50, 40, tid=4), embedding_vec=v)  # overlaps a
    merge_entities([a, b, c, d])
    assert a["entity_id"] == b["entity_id"] == 1
    assert c["entity_id"] == 3
    assert d["entity_id"] == 4          # visible at the same time as a


def test_semantic_rank_orders_and_filters():
    q = np.array([1.0, 0, 0], np.float32)
    good = dict(seg("person", 0, 2, 10, 40, tid=1), embedding_vec=np.array([0.9, 0.1, 0], np.float32))
    bad = dict(seg("person", 5, 2, 10, 40, tid=2), embedding_vec=np.array([0, 1.0, 0], np.float32))
    none = dict(seg("person", 9, 2, 10, 40, tid=3))
    out = semantic_rank([bad, none, good], q)
    assert [s["track_id"] for s in out] == [1, 3]
    assert out[0]["score"] > 0.8


def test_unknown_words_raise_the_semantic_floor():
    from search import MIN_SEMANTIC, MIN_SEMANTIC_UNKNOWN, semantic_floor
    assert semantic_floor([]) == MIN_SEMANTIC
    assert semantic_floor(["knife"]) == MIN_SEMANTIC_UNKNOWN > MIN_SEMANTIC
    q = np.array([1.0, 0, 0], np.float32)
    so_so = dict(seg("person", 0, 2, 10, 40, tid=1), embedding_vec=np.array([0.23, 0.97, 0], np.float32))
    assert semantic_rank([so_so], q, min_score=semantic_floor([]))          # "a person" keeps it
    assert not semantic_rank([so_so], q, min_score=semantic_floor(["knife"]))  # "a person with a knife" does not


def test_scene_windows_groups_standout_frames():
    from search import scene_windows
    q = np.array([1.0, 0, 0], np.float32)
    far = np.array([0, 1.0, 0], np.float32)
    near = np.array([0.95, 0.31, 0], np.float32)
    # 30 s of nothing, a 3 s burst at 10-12, a lone hit at 25
    frames = [(float(t), near if t in (10, 11, 12, 25) else far) for t in range(30)]
    out = scene_windows(frames, q)
    assert [round(s["t"]) for s in sorted(out, key=lambda s: s["t"])] == [10, 25]
    burst = next(s for s in out if round(s["t"]) == 10)
    assert burst["dur"] == 3.0 and burst["cls"] == "scene" and burst["score"] > 0.9
    assert burst["box"] == {"x": 0.0, "y": 0.0, "w": 100.0, "h": 100.0}
    assert scene_windows([], q) == []


def test_frames_roundtrip(tmp_path):
    import db as database
    import embeddings
    conn = database.connect(str(tmp_path / "x.db"))
    v = np.array([0.6, 0.8, 0.0], np.float32)
    database.insert_frames(conn, "v.mp4", [(0.0, embeddings.to_blob(v)), (1.0, embeddings.to_blob(v))])
    rows = database.load_frames(conn, "v.mp4")
    assert [r[0] for r in rows] == [0.0, 1.0]
    assert np.allclose(embeddings.from_blob(rows[0][1]), v)
    database.delete_video(conn, "v.mp4")
    assert database.load_frames(conn, "v.mp4") == []
