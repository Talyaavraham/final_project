"""live/learned.py on synthetic videos: a gun in a hand (the event) vs a gun lying on a table (not)."""
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
BACKEND = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(BACKEND, "live"))
import learned  # noqa: E402


def person(x, wrist_y=250, pid=1):
    k = [[x, 100, 0.9] for _ in range(17)]
    k[5], k[6] = [x - 10, 150, 0.9], [x + 10, 150, 0.9]          # shoulders
    k[9], k[10] = [x - 20, wrist_y, 0.9], [x + 20, 250, 0.9]      # wrists
    return {"id": pid, "box": [x - 50, 80, x + 50, 400], "kpts": k}


def write_video(path, events, seconds=60, fps=3):
    lines = [json.dumps({"video": os.path.basename(path), "fps": fps, "duration": seconds, "w": 640, "h": 480})]
    for i in range(seconds * fps):
        t = i / fps
        inside = any(s <= t <= e for s, e in events)
        if inside:                                   # gun in the left hand, raised
            p, objs = person(200, wrist_y=140), [{"cls": "gun", "conf": 0.5, "box": [170, 125, 190, 150]}]
        else:                                        # a gun-like thing on a table far from the person
            p = person(200)
            objs = [{"cls": "gun", "conf": 0.5, "box": [560, 400, 590, 420]}] if (i // 30) % 2 else []
        lines.append(json.dumps({"t": round(t, 3), "persons": [p], "objects": objs}))
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")


def make(tmp, names_events):
    perc = tmp / "perc"
    perc.mkdir(exist_ok=True)
    gt = {"videos": {}}
    for name, ev in names_events.items():
        write_video(str(perc / (name + ".jsonl")), ev)
        gt["videos"][name] = {"queries": [{"scenario": "weapon", "expected": ev}]}
    p = tmp / f"gt_{len(list(tmp.iterdir()))}.json"
    p.write_text(json.dumps(gt))
    return str(p), str(perc)


def test_features_see_the_hand_on_the_gun():
    fr = {"persons": [person(200, wrist_y=140)], "objects": [{"cls": "gun", "conf": 0.5, "box": [170, 125, 190, 150]}]}
    f = dict(zip(learned.FRAME_FEATURES, learned.frame_features(fr, None, {"gun"}, 640, 480)))
    assert f["touch_conf"] == 0.5 and f["hand_dist"] < 0.05 and f["raise_obj"] > 0
    far = {"persons": [person(200)], "objects": [{"cls": "gun", "conf": 0.5, "box": [560, 400, 590, 420]}]}
    g = dict(zip(learned.FRAME_FEATURES, learned.frame_features(far, None, {"gun"}, 640, 480)))
    assert g["touch_conf"] == 0 and g["hand_dist"] > 0.5 and g["obj_conf"] == 0.5
    times, rows = learned.video_features({"w": 640, "h": 480}, [dict(fr, t=0.0), dict(far, t=1.0)], {"gun"})
    assert len(rows) == 2 and len(rows[0]) == len(learned.feature_names())


def test_train_then_eval_beats_object_seen(tmp_path):
    pytest = __import__("pytest")
    pytest.importorskip("sklearn"); pytest.importorskip("joblib")
    train_gt, train_perc = make(tmp_path, {f"tr{i}.mp4": [[10 + i, 14 + i], [40, 43]] for i in range(6)})
    test_dir = tmp_path / "test"; test_dir.mkdir()
    test_gt, test_perc = make(test_dir, {f"te{i}.mp4": [[20 + i, 24 + i]] for i in range(3)})
    model = str(tmp_path / "m.joblib")
    run = lambda *a: subprocess.run([sys.executable, os.path.join(BACKEND, "live", "learned.py"), *a],
                                    check=True, capture_output=True, text=True, cwd=BACKEND)
    run("train", "--gt", train_gt, "--perception", train_perc, "--scenario", "weapon", "--out", model)
    name = str(tmp_path / "res")
    out = run("eval", "--gt", test_gt, "--perception", test_perc, "--model", model, "--name", name).stdout
    rep = json.load(open(name + ".json"))["scenarios"]["weapon"]["learned p>=0.5"]
    assert rep["caught"] == 3 and rep["false"] == 0, out           # the table gun is not an event


def test_merge_takes_each_class_from_its_detector(tmp_path):
    import merge_perception as mp
    head = {"video": "v", "fps": 3, "w": 10, "h": 10}
    def cache(path, objs):
        path.write_text(json.dumps(head) + "\n" + json.dumps({"t": 0.333, "persons": [], "objects": objs}) + "\n")
    cache(tmp_path / "a.jsonl", [{"cls": "gun", "conf": 0.6, "box": [0, 0, 1, 1]}, {"cls": "fire", "conf": 0.2, "box": [0, 0, 1, 1]}])
    cache(tmp_path / "b.jsonl", [{"cls": "gun", "conf": 0.1, "box": [0, 0, 1, 1]}, {"cls": "fire", "conf": 0.9, "box": [0, 0, 1, 1]}])
    mp.merge(str(tmp_path / "a.jsonl"), str(tmp_path / "b.jsonl"), {"fire", "smoke"}, str(tmp_path / "c.jsonl"))
    fr = json.loads((tmp_path / "c.jsonl").read_text().splitlines()[1])
    assert sorted((o["cls"], o["conf"]) for o in fr["objects"]) == [("fire", 0.9), ("gun", 0.6)]


def test_train_set_never_uses_a_test_video():
    import train_set
    uca = json.load(open(os.path.join(BACKEND, "eval", "data", "uca_all.json"), encoding="utf-8"))
    test = {os.path.splitext(v)[0] for v in json.load(open(train_set.TEST, encoding="utf-8"))["videos"]}
    videos, pos, neg = train_set.build(uca, test, ["weapon", "fire"], 60, 300)
    assert videos and not {os.path.splitext(v)[0] for v in videos} & test
