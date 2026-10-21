"""End-to-end: upload -> index (fake detector) -> search -> frame / clip / zip."""

import io
import json
import os
import time
import zipfile

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def client(tmp_path, monkeypatch, fake_model, no_clip):
    import db
    import indexer
    import main
    monkeypatch.setattr(db, "DB_PATH", str(tmp_path / "index.db"))
    monkeypatch.setattr(main, "VIDEO_DIR", str(tmp_path / "videos"))
    monkeypatch.setattr(main, "EXPORT_DIR", str(tmp_path / "exports"))
    os.makedirs(main.VIDEO_DIR); os.makedirs(main.EXPORT_DIR)
    monkeypatch.setattr(indexer, "load_model", lambda *a, **k: fake_model)
    main.INDEX_STATUS.clear()
    return TestClient(main.app)


def upload(client, video_file, name="street.mp4", **params):
    with open(video_file, "rb") as f:
        r = client.post("/index", files={"file": (name, f, "video/mp4")}, params=params)
    assert r.status_code == 200, r.text
    return r.json()


def wait_ready(client, name, timeout=30):
    t0 = time.time()
    while time.time() - t0 < timeout:
        st = client.get("/status", params={"video": name}).json()
        if st["status"] == "ready":
            return st
        assert st["status"].startswith(("indexing", "ready")), st
        time.sleep(0.1)
    raise AssertionError("indexing did not finish")


def test_root_serves_ui_and_api_reports_capabilities(client):
    r = client.get("/")
    assert r.status_code == 200 and "text/html" in r.headers["content-type"]
    assert "Video Search Agent" in r.text
    info = client.get("/api").json()
    assert info["ok"] and "ffmpeg" in info and "clip" in info and "llm_parser" in info


def test_index_search_and_outputs(client, video_file):
    assert upload(client, video_file)["status"] == "indexing"
    st = wait_ready(client, "street.mp4")
    assert st["segments"] == 3

    classes = client.get("/classes", params={"video": "street.mp4"}).json()["classes"]
    assert set(classes) == {"person", "dog", "car"}

    # plain object query
    r = client.post("/search", json={"video": "street.mp4", "query": {"objects": ["person"]}}).json()
    assert r["total"] == 1
    m = r["results"][0]
    assert m["cls"] == "person" and m["label"] == "red person"
    assert m["dur"] >= 4 and m["path"]
    # best_t is the time of the detection the box belongs to
    assert any(abs(p[0] - m["best_t"]) < 1e-6 and abs(p[1] - m["box"]["x"]) < 0.01 for p in m["path"])

    # colour + action of the subject
    r = client.post("/search", json={"video": "street.mp4",
                                     "query": {"objects": ["person"], "colors": ["blue"]}}).json()
    assert r["total"] == 0
    r = client.post("/search", json={"video": "street.mp4",
                                     "query": {"objects": ["car"], "actions": ["parked"]}}).json()
    assert r["total"] == 1 and r["results"][0]["label"] == "blue car"

    # "person with a dog": the dog walks next to the person, the car is far
    r = client.post("/search", json={"video": "street.mp4",
                                     "query": {"objects": ["person", "dog"]}}).json()
    assert {x["cls"] for x in r["results"]} == {"person", "dog"}
    r = client.post("/search", json={"video": "street.mp4",
                                     "query": {"objects": ["person", "car"]}}).json()
    assert r["total"] == 0
    r = client.post("/search", json={"video": "street.mp4", "near": False,
                                     "query": {"objects": ["person", "car"]}}).json()
    assert r["total"] == 2

    # pagination
    r = client.post("/search", json={"video": "street.mp4", "limit": 1, "offset": 1}).json()
    assert r["total"] == 3 and len(r["results"]) == 1

    # frame with box
    r = client.get("/frame", params={"video": "street.mp4", "t": 1.0, "x": m["box"]["x"],
                                     "y": m["box"]["y"], "w": m["box"]["w"], "h": m["box"]["h"],
                                     "label": "person"})
    assert r.status_code == 200 and r.headers["content-type"] == "image/jpeg"
    assert r.content[:2] == b"\xff\xd8"

    # clip, plain and annotated
    r = client.get("/clip", params={"video": "street.mp4", "t0": 0.5, "t1": 2.0, "annotate": False})
    assert r.status_code == 200 and len(r.content) > 1000
    r = client.get("/clip", params={"video": "street.mp4", "t0": 0.5, "t1": 2.0, "track": m["track"]})
    assert r.status_code == 200 and b"ftyp" in r.content[:64]

    # zip export
    r = client.post("/export", json={"video": "street.mp4", "query": "person",
                                     "matches": [m], "pad": 0.5})
    assert r.status_code == 200 and r.headers["content-type"] == "application/zip"
    z = zipfile.ZipFile(io.BytesIO(r.content))
    names = z.namelist()
    assert any(n.startswith("frames/") and n.endswith(".jpg") for n in names)
    assert any(n.startswith("clips/") and n.endswith(".mp4") for n in names)
    res = json.loads(z.read("results.json"))
    assert res["query"] == "person" and res["matches"][0]["class"] == "person"
    assert "results.csv" in names


def test_reupload_same_content_is_cached(client, video_file):
    upload(client, video_file)
    wait_ready(client, "street.mp4")
    assert upload(client, video_file)["cached"] is True
    # same bytes under a new name -> reuse the existing index
    out = upload(client, video_file, name="copy.mp4")
    assert out == {"status": "ready", "video": "street.mp4", "cached": True}
    # force re-index
    assert upload(client, video_file, force=True)["status"] == "indexing"
    wait_ready(client, "street.mp4")


def test_status_survives_restart_and_delete(client, video_file):
    import main
    upload(client, video_file)
    wait_ready(client, "street.mp4")
    main.INDEX_STATUS.clear()
    assert client.get("/status", params={"video": "street.mp4"}).json()["status"] == "ready"
    assert client.get("/status", params={"video": "nope.mp4"}).json()["status"] == "unknown"
    assert client.delete("/videos/street.mp4").json() == {"deleted": "street.mp4"}
    assert client.get("/videos").json() == []
    assert client.post("/search", json={"video": "street.mp4"}).status_code == 404


def test_parse_endpoint_uses_rules_offline(client, monkeypatch):
    import query_parser
    monkeypatch.setattr(query_parser, "llm_available", lambda: False)
    p = client.post("/parse", json={"text": "a red car after 1:00"}).json()
    assert p["objects"] == ["car"] and p["colors"] == ["red"]
    assert p["timeRange"] == [60, None] and p["source"] == "rules"


def test_bad_names_rejected(client):
    r = client.post("/index", files={"file": ("../evil.mp4", b"xx", "video/mp4")})
    assert r.status_code in (200, 400)
    if r.status_code == 200:
        assert r.json()["video"] == "evil.mp4"
    assert client.get("/frame", params={"video": "missing.mp4", "t": 1}).status_code == 404
