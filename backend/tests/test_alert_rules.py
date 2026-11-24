"""Live alert rules: parsing (keyword parser) and timing (Evaluator)."""
import alert_rules as ar


def rule(text):
    r = ar.parse_rules(text)
    assert r is not None, text
    return r


def test_parse_kinds():
    assert (rule("תתריעי לי אם מישהו מרים סכין")["kind"], rule("תתריעי לי אם מישהו מרים סכין")["object"]) == ("raise_object", "knife")
    assert rule("alert me if someone raises a knife")["kind"] == "raise_object"
    assert (rule("תתריע אם יש אקדח")["kind"], rule("תתריע אם יש אקדח")["object"]) == ("object", "gun")
    assert (rule("מישהו מחזיק טלפון")["kind"], rule("מישהו מחזיק טלפון")["object"]) == ("hold", "cell phone")
    assert rule("hands up")["kind"] == "hands_up"
    assert rule("מרים ידיים")["kind"] == "hands_up"
    assert rule("if there is fire")["object"] == "fire"


def test_parse_numbers():
    assert rule("יותר משני אנשים בחדר")["min_people"] == 3
    assert rule("more than 2 people")["min_people"] == 3
    assert rule("at least 2 people")["min_people"] == 2
    assert rule("someone holding a phone for 3 seconds")["duration"] == 3.0
    assert rule("תתריעי אם יש סכין במשך 2 שניות")["duration"] == 2.0


def test_parse_not_understood():
    assert ar.parse_rules("מה השעה") is None
    assert ar.parse_rules("someone falls down") is None      # not supported yet: no wrong rule
    assert ar.parse_rules("someone is talking") is None


def person(wrist_y, pid=1, right_y=250):
    k = [[100.0, 100.0, 0.9] for _ in range(17)]
    k[ar.L_SHOULDER] = [90, 150, 0.9]
    k[ar.R_SHOULDER] = [110, 150, 0.9]
    k[ar.L_WRIST] = [80, wrist_y, 0.9]
    k[ar.R_WRIST] = [120, right_y, 0.9]
    return {"id": pid, "box": [50, 80, 150, 400], "kpts": k}


def knife(y):
    return [{"cls": "knife", "conf": 0.5, "box": [70, y - 10, 90, y + 10]}]


def run(ev, r, frames, t0=0.0, dt=0.1):
    t, fired = t0, 0
    for persons, objects in frames:
        t += dt
        alerts, _ = ev.step([r], persons, objects, t)
        fired += len(alerts)
    return fired, t


def test_raise_knife_fires_even_if_knife_lost_in_motion():
    r, ev = rule("מישהו מרים סכין"), ar.Evaluator()
    low = [([person(250)], knife(250))] * 10           # holds it low: no alert
    assert run(ev, r, low)[0] == 0
    up = [([person(120)], [])] * 8                     # raises it, detector loses the knife
    assert run(ev, r, up, t0=1.0)[0] == 1


def test_raised_hand_without_object_is_not_raise_knife():
    r, ev = rule("מישהו מרים סכין"), ar.Evaluator()
    assert run(ev, r, [([person(120)], [])] * 30)[0] == 0


def test_knife_on_table_is_not_held():
    r, ev = rule("someone holding a knife"), ar.Evaluator()
    table = [{"cls": "knife", "conf": 0.6, "box": [300, 300, 340, 320]}]
    assert run(ev, r, [([person(250)], table)] * 30)[0] == 0


def test_cooldown_and_hands_up():
    r, ev = rule("hands up"), ar.Evaluator()
    both = [([person(120, right_y=120)], [])] * 40     # 4 s, cooldown 5 s -> one alert
    assert run(ev, r, both)[0] == 1


def test_short_flicker_is_bridged():
    r, ev = rule("תתריעי אם יש סכין במשך 2 שניות"), ar.Evaluator()
    frames = []
    for i in range(30):                                # knife seen 2 of every 3 frames
        frames.append(([], knife(100) if i % 3 else []))
    assert run(ev, r, frames)[0] == 1


def test_people_count():
    r, ev = rule("more than 2 people"), ar.Evaluator()
    two = [([person(250, 1), person(250, 2)], [])] * 20
    assert run(ev, r, two)[0] == 0
    three = [([person(250, 1), person(250, 2), person(250, 3)], [])] * 20
    assert run(ev, r, three, t0=2.0)[0] == 1


def test_friendly_text_in_the_users_language():
    r = rule("תתריעי לי אם מישהו מרים סכין")
    assert r["label"] == "מישהו מרים סכין"
    assert r["description"] == "אתריע לך כשמישהו מרים סכין"
    assert rule("someone holding a phone")["description"] == "I'll alert you when someone holds a phone"


def test_one_hand_or_both():
    one, both = rule("תתריעי כשמישהו מרים את היד"), rule("מישהו מרים ידיים")
    assert (one["kind"], one["hands"], both["hands"]) == ("hands_up", 1, 2)
    assert rule("hands up")["hands"] == 2
    assert rule("someone raises a hand")["hands"] == 1
    raised_left = [([person(120)], [])] * 15            # only the left wrist above the shoulder
    assert run(ar.Evaluator(), one, raised_left)[0] == 1
    assert run(ar.Evaluator(), both, raised_left)[0] == 0


def test_slow_computer_still_alerts():
    """1 frame per 1-1.5 s (a laptop CPU): a raised hand must still fire."""
    r = rule("מישהו מרים יד")
    for dt in (1.0, 1.5):
        frames = [([person(120)], [])] * 6
        assert run(ar.Evaluator(), r, frames, dt=dt)[0] >= 1, dt


def test_slow_computer_phone_hold():
    r = rule("מישהו מחזיק טלפון")
    phone = [{"cls": "cell phone", "conf": 0.5, "box": [70, 240, 90, 260]}]
    assert run(ar.Evaluator(), r, [([person(250)], phone)] * 5, dt=1.2)[0] == 1


def test_multiword_object_wins():
    assert rule("מישהו מחזיק מחשב נייד")["object"] == "laptop"
    assert rule("מישהו מחזיק נייד")["object"] == "cell phone"


def test_combined_parser_runs_locally():
    out = ar.parse("תתריעי לי אם מישהו מרים סכין")
    assert out["rule"]["kind"] == "raise_object" and out["rule"]["object"] == "knife"
    assert out["source"] in ("rules", "model", "model+rules")
    assert ar.parse("מה השעה")["rule"] is None
    assert not hasattr(ar, "parse_llm")


# ------------------------------------------------ ignite / snatch: positives and look-alikes
def walker(pid, x, wrist=None):
    """a person standing at x (box 100 wide, 320 tall); the right wrist at `wrist` if given."""
    k = [[x + 50.0, 120.0, 0.9] for _ in range(17)]
    k[ar.L_SHOULDER], k[ar.R_SHOULDER] = [x + 40, 150, .9], [x + 60, 150, .9]
    k[ar.L_WRIST] = [x + 30, 260, .9]
    k[ar.R_WRIST] = list(wrist) + [.9] if wrist else [x + 70, 260, .9]
    return {"id": pid, "box": [x, 80, x + 100, 400], "kpts": k}


def obj(cls, cx, cy, size=40, conf=0.6):
    return {"cls": cls, "conf": conf, "box": [cx - size / 2, cy - size / 2, cx + size / 2, cy + size / 2]}


def play(r, frames, dt=1 / 3):
    ev, t, fired = ar.Evaluator(), 0.0, 0
    for persons, objects in frames:
        t += dt
        fired += len(ev.step([r], persons, objects, t)[0])
    return fired


def test_parse_ignite_and_snatch():
    assert rule("מישהו מצית אש")["kind"] == "ignite"
    assert rule("someone sets fire to the bin")["kind"] == "ignite"
    assert rule("יש שריפה")["kind"] == "object"                 # fire alone: the object rule
    sn = rule("מישהו חוטף תיק")
    assert (sn["kind"], sn["object"]) == ("snatch", "bag")
    assert rule("someone runs off with a laptop")["object"] == "laptop"
    assert ar.parse_rules("מישהו גונב") is None                 # steals what? not understood


def test_ignite_fire_appears_where_someone_stood():
    r = rule("מישהו מצית אש")
    before = [([walker(1, 100)], [])] * 15                       # 5 s, a person at x=100
    fire = [([walker(1, 100)], [obj("fire", 190, 380)])] * 6     # fire appears at their feet
    assert play(r, before + fire) == 1


def test_ignite_person_left_before_the_fire():
    r = rule("מישהו מצית אש")
    frames = [([walker(1, 100)], [])] * 12 + [([], [])] * 9 + [([], [obj("fire", 190, 380)])] * 6
    assert play(r, frames) == 1                                  # was there 3 s before: within PRESENCE_BEFORE


def test_ignite_not_for_fire_that_was_already_burning():
    r = rule("מישהו מצית אש")
    frames = [([walker(1, 100)], [obj("fire", 190, 380)])] * 30   # fire from the first frame
    assert play(r, frames) == 0


def test_ignite_not_for_fire_far_from_anyone():
    r = rule("מישהו מצית אש")
    frames = [([walker(1, 100)], [])] * 15 + [([walker(1, 100)], [obj("fire", 900, 380)])] * 6
    assert play(r, frames) == 0


def test_snatch_someone_takes_anothers_bag():
    r = rule("מישהו חוטף תיק")
    own = [([walker(1, 100)], [obj("bag", 160, 330)])] * 9                         # A next to their bag, 3 s
    grab = [([walker(1, 100), walker(2, 220, wrist=(165, 330))], [obj("bag", 160, 330)])] * 2
    run = [([walker(1, 100), walker(2, 220 + 40 * i, wrist=(165 + 40 * i, 330))], [obj("bag", 160 + 40 * i, 330)])
           for i in range(1, 8)]                                                  # B carries it away
    assert play(r, own + grab + run) == 1


def test_snatch_not_when_the_owner_takes_their_own_bag():
    r = rule("מישהו חוטף תיק")
    own = [([walker(1, 100, wrist=(160, 330))], [obj("bag", 160, 330)])] * 9
    leave = [([walker(1, 100 + 40 * i, wrist=(160 + 40 * i, 330))], [obj("bag", 160 + 40 * i, 330)]) for i in range(1, 8)]
    assert play(r, own + leave) == 0


def test_snatch_a_bag_that_was_left_alone():
    r = rule("מישהו חוטף תיק")
    alone = [([], [obj("bag", 500, 330)])] * 9                                     # a bag on its own, 3 s
    take = [([walker(2, 400 + 40 * i, wrist=(500 + 40 * i, 330))], [obj("bag", 500 + 40 * i, 330)]) for i in range(8)]
    assert play(r, alone + take) == 1


def test_snatch_not_when_someone_only_touches_it():
    r = rule("מישהו חוטף תיק")
    own = [([walker(1, 100)], [obj("bag", 160, 330)])] * 9
    touch = [([walker(1, 100), walker(2, 220, wrist=(165, 330))], [obj("bag", 160, 330)])] * 12
    assert play(r, own + touch) == 0


def test_ignite_fire_only_ignores_smoke():
    before = [([walker(1, 100)], [])] * 15
    smoke = [([walker(1, 100)], [obj("smoke", 190, 380)])] * 6
    assert play(ar.make_rule("x", "ignite", "fire"), before + smoke) == 0
    assert play(ar.make_rule("x", "ignite", "fire_or_smoke"), before + smoke) == 1
    assert play(ar.make_rule("x", "object", "fire_or_smoke"), smoke) == 1
