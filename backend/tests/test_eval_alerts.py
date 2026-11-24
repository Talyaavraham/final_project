"""live/eval_alerts.py: scoring of alert times against labelled events."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "live"))
import eval_alerts as ev   # noqa: E402


def test_score_hits_false_and_delay():
    windows = [[10.0, 15.0], [40.0, 42.0]]
    alerts = [9.5, 11.0, 30.0, 43.5]          # 9.5: 0.5 s early (slack 1 s); 43.5: 1.5 s late (slack 2 s)
    hits, delays, false = ev.score(windows, alerts)
    assert hits == 2 and false == 1 and delays == [-0.5, 3.5]


def test_no_events_every_alert_is_false():
    assert ev.score([], [1.0, 2.0]) == (0, [], 2)


def person(wrist_y, pid=1):
    k = [[100.0, 100.0, 0.9] for _ in range(17)]
    k[5], k[6], k[9], k[10] = [90, 150, .9], [110, 150, .9], [80, wrist_y, .9], [120, 250, .9]
    return {"id": pid, "box": [50, 80, 150, 400], "kpts": k}


def test_run_rule_on_frames_and_conf_threshold():
    knife = lambda c: [{"cls": "knife", "conf": c, "box": [70, 240, 90, 260]}]
    frames = [{"t": i / 3, "persons": [person(250)], "objects": knife(0.3)} for i in range(6)]        # holds low
    frames += [{"t": 2 + i / 3, "persons": [person(120)], "objects": []} for i in range(6)]           # raises
    raise_rule = ev.make_rule("raises knife", {"kind": "raise_object", "object": "knife"})
    assert len(ev.run_rule(raise_rule, frames)) == 1
    assert ev.run_rule(raise_rule, frames, conf=0.5) == []      # knife below threshold: never held
    obj_rule = ev.make_rule("object only", {"kind": "object", "object": "knife"})
    assert len(ev.run_rule(obj_rule, frames)) == 1


def test_diagnose_event_stats():
    import diagnose
    knife = {"cls": "knife", "conf": 0.2, "box": [70, 240, 90, 260]}
    frames = [{"t": 1.0, "persons": [person(250)], "objects": [knife]},
              {"t": 5.0, "persons": [], "objects": []}]
    st = diagnose.event_stats(frames, 0.5, 2.0, {"knife"})
    assert st["person"] and st["hand"] and st["best"] == 0.2
    assert diagnose.event_stats(frames, 4.5, 6.0, {"knife"})["best"] == 0.0
