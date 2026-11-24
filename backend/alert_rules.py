"""
Live alert rules: a sentence (Hebrew / English) -> a rule, and the rule
evaluated frame by frame on what the pose model and the detectors see.

A rule is one of seven kinds:

    object        an object appears in the frame         "alert me if there is a knife"
    hold          a person holds an object               "מישהו מחזיק טלפון"
    raise_object  a person raises an object above the    "תתריעי אם מישהו מרים סכין"
                  shoulder (the hand that held it)
    hands_up      a person raises a hand (hands=1) or     "מישהו מרים יד" / "hands up"
                  both hands (hands=2) above the shoulders
    person        at least N people in the frame         "יותר משני אנשים בחדר"
    ignite        fire / smoke appears where someone      "מישהו מצית אש"
                  was standing (not fire that was there from the start)
    snatch        someone takes an object that is not     "מישהו חוטף תיק"
                  theirs (it has another owner, or was left alone) and carries it away

"Holds" = a wrist keypoint is on (near) the object's box. Detectors flicker
on small objects like knives, so a wrist that touched the object keeps it for
a moment (TOUCH_MEMORY): a knife that disappears in motion blur while it is
swung up still counts. A condition must hold for the rule's `duration`
(short gaps up to GAP are bridged) before an alert fires, and a rule fires
at most once per `cooldown`.

parse() turns text into a rule with our own language model (nlu/, trained
on our phrase dataset) combined with a keyword parser; everything runs
locally, nothing is sent to an outside service. Evaluator.step() takes the
people / objects of one frame and returns the alerts.
"""
import json
import os
import re
import uuid
from typing import Optional

KINDS = ["object", "hold", "raise_object", "hands_up", "person", "ignite", "snatch"]
NEEDS_OBJECT = {"object", "hold", "raise_object", "snatch"}

DEFAULT_DURATION = {"object": 0.5, "hold": 1.0, "raise_object": 0.4, "hands_up": 1.0, "person": 1.0,
                    "ignite": 0.5, "snatch": 0.3}
DEFAULT_COOLDOWN = 5.0
GAP = 0.7               # s: a condition that is off for less than this is still "on"
TOUCH_MEMORY = 1.5      # s: a wrist that was on an object keeps "holding" it this long
# On a slow computer frames come further apart than GAP: both windows then
# stretch to this many frame intervals, so a raised hand seen on two frames
# 1.2 s apart is one raise, not two separate ones that each last 0 s.
GAP_FRAMES = 2.5
KPT_CONF = 0.3

# COCO keypoints (YOLOv8-pose)
NOSE, L_SHOULDER, R_SHOULDER, L_WRIST, R_WRIST = 0, 5, 6, 9, 10
WRIST_SHOULDER = {L_WRIST: L_SHOULDER, R_WRIST: R_SHOULDER}

COCO = ["person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
        "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog",
        "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
        "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite",
        "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle",
        "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
        "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch", "potted plant",
        "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone",
        "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
        "teddy bear", "hair drier", "toothbrush"]
SECURITY = ["gun", "knife", "fire", "smoke"]        # security.pt
# COCO's three bag classes are one class for us (as in indexer.RENAME)
BAG_CLASSES = {"backpack", "handbag", "suitcase"}
RENAME = {c: "bag" for c in BAG_CLASSES}
DEFAULT_CLASSES = sorted({RENAME.get(c, c) for c in COCO + SECURITY})

# words -> detector class, for the keyword parser
OBJECT_WORDS = {
    "knife": ["knife", "knives", "blade", "סכין", "סכינים", "סכינה"],
    "gun": ["gun", "guns", "pistol", "handgun", "rifle", "weapon", "firearm",
            "אקדח", "אקדחים", "רובה", "נשק", "כלי נשק"],
    "fire": ["fire", "flames", "flame", "אש", "שריפה", "שרפה", "להבה", "להבות"],
    "smoke": ["smoke", "עשן"],
    "cell phone": ["phone", "cellphone", "mobile", "smartphone", "טלפון", "פלאפון", "נייד", "סמארטפון"],
    "scissors": ["scissors", "מספריים"],
    "bottle": ["bottle", "בקבוק"],
    "cup": ["cup", "mug", "כוס", "ספל"],
    "baseball bat": ["bat", "baseball bat", "מחבט", "אלה"],
    "laptop": ["laptop", "מחשב נייד", "לפטופ"],
    "book": ["book", "ספר"],
    "bag": ["backpack", "bag", "handbag", "purse", "suitcase", "תיק", "תרמיל", "ארנק", "מזוודה", "תיק יד"],
    "umbrella": ["umbrella", "מטריה", "מטרייה"],
    "dog": ["dog", "כלב"],
    "cat": ["cat", "חתול"],
}
RAISE_WORDS = ["raise", "raises", "raising", "raised", "lift", "lifts", "lifting", "swing", "swings",
               "swinging", "wave", "waves", "waving", "brandish", "brandishes", "brandishing",
               "מרים", "מרימה", "מרימים", "הרים", "הרימה", "להרים", "מניף", "מניפה", "הניף", "להניף",
               "מנופף", "מנופפת"]
HOLD_WORDS = ["hold", "holds", "holding", "grab", "grabs", "grabbing", "carry", "carries", "carrying",
              "pick", "picks", "has", "with", "מחזיק", "מחזיקה", "מחזיקים", "אוחז", "אוחזת", "תופס",
              "תופסת", "לוקח", "לוקחת", "עם"]
HAND_WORDS = ["hands", "hand", "arms", "arm", "ידיים", "ידים", "יד", "זרוע", "זרועות"]
BOTH_HANDS = ["hands", "arms", "both", "ידיים", "ידים", "זרועות", "שתי", "שני"]   # plural -> both hands
UP_WORDS = ["up", "למעלה", "מעל"]
PEOPLE_WORDS = ["person", "people", "persons", "someone", "somebody", "anyone", "man", "woman",
                "אדם", "אנשים", "מישהו", "מישהי", "איש", "אישה", "בנאדם"]
PRESENCE_WORDS = ["enter", "enters", "entering", "appears", "appear", "there", "comes", "arrives", "shows",
                  "נכנס", "נכנסת", "נכנסים", "מופיע", "מופיעה", "מופיעים", "יש", "בחדר", "בפריים", "מגיע", "מגיעה"]
# actions the live system cannot see yet: better "not understood" than a wrong rule
UNSUPPORTED_WORDS = ["fall", "falls", "falling", "fell", "fight", "fights", "fighting", "hit", "hits",
                     "punch", "punches", "kick", "kicks", "scream", "screams", "run", "runs", "running",
                     "נופל", "נופלת", "נפל", "נפלה", "רב", "רבים", "מכה",
                     "מרביץ", "מרביצה", "בועט", "צועק", "צועקת", "רץ", "רצה", "מתקוטטים"]
# taking something that is not yours (needs an object: "steals a bag"; "someone steals" alone is not understood)
SNATCH_WORDS = ["snatch", "snatches", "snatching", "steal", "steals", "stealing", "stole", "rob", "robs",
                "runs off with", "takes away", "grabs and runs", "pickpockets",
                "חוטף", "חוטפת", "חטף", "חטפה", "לחטוף", "גונב", "גונבת", "גנב", "גנבה", "לגנוב", "סוחב",
                "סוחבת", "בורח עם", "בורחת עם", "לוקח בלי רשות"]
# starting a fire ("there is a fire" alone is the object rule)
IGNITE_WORDS = ["ignite", "ignites", "igniting", "set fire", "sets fire", "setting fire", "lights a fire",
                "lighting a fire", "starts a fire", "starting a fire", "arson", "sets on fire", "burns",
                "מצית", "מציתה", "הצית", "הציתה", "להצית", "הצתה", "מדליק אש", "מדליקה אש", "הדליק אש",
                "מבעיר", "מבעירה", "שורף", "שורפת", "שרף", "מעלה באש", "מצית אש"]
HE_NAMES = {"knife": "סכין", "gun": "נשק", "fire": "אש", "smoke": "עשן", "cell phone": "טלפון",
            "scissors": "מספריים", "bottle": "בקבוק", "cup": "כוס", "baseball bat": "מחבט",
            "laptop": "מחשב נייד", "book": "ספר", "bag": "תיק", "umbrella": "מטריה",
            "dog": "כלב", "cat": "חתול"}
EN_NAMES = {"cell phone": "phone", "baseball bat": "bat"}
HEB_NUM = {"אחד": 1, "אחת": 1, "שני": 2, "שתי": 2, "שניים": 2, "שתיים": 2, "שלושה": 3, "שלוש": 3,
           "ארבעה": 4, "ארבע": 4, "חמישה": 5, "חמש": 5, "two": 2, "three": 3, "four": 4, "five": 5}
HE_PREFIX = set("והשבלמכ")


def tokenize(text: str):
    return re.findall(r"[\w֐-׿]+(?:-[\w֐-׿]+)?", text.lower())


def _has(tokens, word):
    """word (maybe several words) in the text; Hebrew prefixes ו/ה/ש/ב/ל/מ/כ allowed."""
    parts = word.split()
    for i in range(len(tokens) - len(parts) + 1):
        ok = True
        for j, p in enumerate(parts):
            t = tokens[i + j].replace("-", "")
            if t == p:
                continue
            extra = len(t) - len(p)
            if 0 < extra <= 2 and t.endswith(p) and set(t[:extra]) <= HE_PREFIX:
                continue
            ok = False
            break
        if ok:
            return True
    return False


def _any(tokens, words):
    return any(_has(tokens, w) for w in words)


def is_hebrew(text: str) -> bool:
    return bool(re.search(r"[֐-׿]", text))


def _number_before(text, unit_regex):
    m = re.search(r"(\d+(?:\.\d+)?)\s*-?\s*(?:" + unit_regex + r")", text)
    if m:
        return float(m.group(1))
    m = re.search(r"(" + "|".join(HEB_NUM) + r")\s+(?:" + unit_regex + r")", text)
    return float(HEB_NUM[m.group(1)]) if m else None


def parse_duration(text: str) -> Optional[float]:
    d = _number_before(text.lower(), r"seconds?|secs?|sec\b|s\b|שניות|שניה|שנייה")
    if d is None and re.search(r"(?<![\w֐-׿])(שנייה|שניה)(?![\w֐-׿])", text):
        d = 1.0
    return d


def parse_people(text: str) -> Optional[int]:
    t = text.lower()
    n = _number_before(t, r"people|persons|אנשים|איש|בני אדם")
    if n is None:
        return None
    more = re.search(r"(more than|over|at least|יותר מ|מעל|לפחות)", t)
    if more and "at least" not in more.group(1) and "לפחות" not in more.group(1):
        return int(n) + 1
    return int(n)


def make_rule(text, kind, obj="", min_people=1, duration=None, cooldown=None,
              description="", source="rules", label="", hands=2):
    kind = kind if kind in KINDS else None
    if kind is None:
        return None
    if kind == "snatch" and not obj:
        obj = "bag"
    rule = {
        "id": uuid.uuid4().hex[:8], "text": text, "kind": kind,
        "object": obj if kind in NEEDS_OBJECT else "",
        "min_people": max(1, int(min_people or 1)) if kind == "person" else 1,
        "hands": (1 if int(hands or 2) == 1 else 2) if kind == "hands_up" else 0,
        "duration": float(duration) if duration is not None else DEFAULT_DURATION[kind],
        "cooldown": float(cooldown) if cooldown is not None else DEFAULT_COOLDOWN,
        "source": source,
    }
    he = is_hebrew(text)
    rule["description"] = description or describe(rule, he)
    rule["label"] = label or short_label(rule, he)
    rule["lang"] = "he" if he else "en"
    return rule


def _what(rule, hebrew):
    """the event in a few words: 'someone raises a knife' / 'מישהו מרים סכין'."""
    o, k, n = rule["object"], rule["kind"], rule["min_people"]
    o = HE_NAMES.get(o, o) if hebrew else EN_NAMES.get(o, o)
    if hebrew:
        return {
            "object": f"רואים {o}",
            "hold": f"מישהו מחזיק {o}",
            "raise_object": f"מישהו מרים {o}",
            "hands_up": "מישהו מרים יד" if rule.get("hands") == 1 else "מישהו מרים את שתי הידיים",
            "person": "מישהו בפריים" if n == 1 else f"{n} אנשים או יותר בפריים",
            "ignite": "מישהו מצית אש",
            "snatch": f"מישהו לוקח {o} שלא שלו",
        }[k]
    return {
        "object": f"a {o} is in view",
        "hold": f"someone holds a {o}",
        "raise_object": f"someone raises a {o}",
        "hands_up": "someone raises a hand" if rule.get("hands") == 1 else "someone raises both hands",
        "person": "someone is in view" if n == 1 else f"{n} or more people in view",
        "ignite": "someone starts a fire",
        "snatch": f"someone takes a {o} that is not theirs",
    }[k]


def short_label(rule, hebrew=False):
    w = _what(rule, hebrew)
    return w[0].upper() + w[1:] if not hebrew else w


def describe(rule, hebrew=False):
    """the confirmation the user reads: 'I'll alert you when ...'."""
    w = _what(rule, hebrew)
    return (f"אתריע לך כש-{w}" if w[0].isdigit() else f"אתריע לך כש{w}") if hebrew else f"I'll alert you when {w}"


def parse_rules(text: str, classes=None) -> Optional[dict]:
    """Keyword parser (Hebrew + English). None = not understood."""
    classes = set(classes or DEFAULT_CLASSES)
    tokens = tokenize(text)
    obj = ""
    # longest words first: "מחשב נייד" is a laptop, not a phone ("נייד")
    for cls, w in sorted(((c, w) for c, ws in OBJECT_WORDS.items() for w in ws), key=lambda x: -len(x[1].split())):
        if cls in classes and _has(tokens, w):
            obj = cls
            break
    if not obj:   # any other detector class named in English ("a red cup" -> cup)
        for cls in sorted(classes, key=len, reverse=True):
            if cls != "person" and _has(tokens, cls):
                obj = cls
                break
    duration = parse_duration(text)
    raising = _any(tokens, RAISE_WORDS)
    hands = _any(tokens, HAND_WORDS)
    up = _any(tokens, UP_WORDS)
    holding = _any(tokens, HOLD_WORDS)
    n_people = parse_people(text)
    low = " " + " ".join(tokens) + " "

    if _any(tokens, IGNITE_WORDS) or any(f" {w} " in low for w in IGNITE_WORDS if " " in w):
        return make_rule(text, "ignite", "", duration=duration, source="rules")
    if (obj and obj not in ("fire", "smoke")) and (_any(tokens, SNATCH_WORDS) or any(f" {w} " in low for w in SNATCH_WORDS if " " in w)):
        return make_rule(text, "snatch", obj, duration=duration, source="rules")
    if _any(tokens, UNSUPPORTED_WORDS + SNATCH_WORDS):
        return None
    if hands and (raising or up) and not obj:
        kind = "hands_up"
    elif obj and raising:
        kind = "raise_object"
    elif obj and holding and _any(tokens, PEOPLE_WORDS + ["hand", "hands", "יד", "ביד"]):
        kind = "hold"
    elif obj:
        kind = "object"
    elif n_people or (_any(tokens, PEOPLE_WORDS) and _any(tokens, PRESENCE_WORDS)):
        kind = "person"
    else:
        return None
    n_hands = 2 if _any(tokens, BOTH_HANDS) or "hands up" in text.lower() else 1
    return make_rule(text, kind, obj, n_people or 1, duration, source="rules", hands=n_hands)


# ------------------------------------------------------------------ our language model
# How the model (nlu/model.npz) and the keyword parser are combined. Measured
# on nlu/data (python nlu/evaluate.py): the model is better at the kind of
# request in free sentences and at "one hand / both hands"; the keywords are
# better at naming the object. So:
#   * both agree on the kind        -> that kind; object from the keywords when they found one
#   * keywords found nothing        -> the model, only if it is sure (MODEL_SURE)
#   * they disagree                 -> the model if it is very sure (MODEL_OVERRIDE), else the keywords
# The two thresholds were set by hand before looking at the test sets.
MODEL_SURE = 0.6
MODEL_OVERRIDE = 0.8


def rule_objects(rule):
    """the detector classes a rule needs (what the live server must detect)."""
    if rule["kind"] == "ignite":
        return set(FIRE)
    return {rule["object"]} if rule.get("object") else set()


def _nlu():
    try:
        from nlu.model import load
    except ImportError:
        return None
    return load()


def nlu_info():
    m = _nlu()
    return {"model": bool(m), "encoder": m.encoder_name if m else None,
            "trained_on": m.trained_on if m else None}


def understand(text, classes=None):
    """-> {"kind", "object", "hands", "source", "confidence"}; kind "none" = not understood."""
    kw = parse_rules(text, classes)
    k = {"kind": kw["kind"], "object": kw["object"], "hands": kw.get("hands", 0)} if kw else None
    m = _nlu()
    if m is None:
        return {**(k or {"kind": "none", "object": "", "hands": 0}), "source": "rules", "confidence": None}
    p = m.predict(text)
    mk = {"kind": p["kind"], "object": p["object"] if p["kind"] in NEEDS_OBJECT else "",
          "hands": p["hands"] if p["kind"] == "hands_up" else 0}
    conf = round(p["kind_p"], 2)
    if k is None:
        if p["kind_p"] >= MODEL_SURE:
            return {**mk, "source": "model", "confidence": conf}
        return {"kind": "none", "object": "", "hands": 0, "source": "model", "confidence": conf}
    if k["kind"] == mk["kind"] or p["kind_p"] >= MODEL_OVERRIDE:
        kind = mk["kind"]
        obj = (k["object"] or mk["object"]) if kind in NEEDS_OBJECT else ""
        hands = mk["hands"] if kind == "hands_up" else 0
        return {"kind": kind, "object": obj, "hands": hands,
                "source": "model+rules" if k["kind"] == mk["kind"] else "model", "confidence": conf}
    return {**k, "source": "rules", "confidence": conf}


def parse(text: str, classes=None) -> dict:
    """-> {"rule": rule, "source"} or {"rule": None, "reason", "source"}"""
    classes = list(classes or DEFAULT_CLASSES)
    he = is_hebrew(text)
    u = understand(text, classes)
    if u["kind"] == "none":
        return {"rule": None, "source": u["source"], "reason":
                "את זה אני עוד לא יודעת לזהות, או שלא הבנתי 🤔" if he else "I can't detect that yet, or I didn't understand 🤔"}
    if u["kind"] in NEEDS_OBJECT and u["object"] not in set(classes):
        return {"rule": None, "source": u["source"], "reason":
                f"הגלאי לא מכיר \"{u['object']}\"" if he else f"the detector does not know \"{u['object']}\""}
    rule = make_rule(text, u["kind"], u["object"], parse_people(text) or 1, parse_duration(text),
                     source=u["source"], hands=u["hands"] or 2)
    rule["confidence"] = u["confidence"]
    return {"rule": rule, "source": u["source"]}


# ------------------------------------------------------------------ evaluation
# Building blocks, each a small function of what one or more frames show:
#   on(wrist, object)        a wrist on / next to an object's box
#   held(person, object)     on(...) recently (TOUCH_MEMORY): survives a few frames the detector misses
#   raised(person, wrist)    the wrist above its shoulder
#   near(person, box)        the person's box close to a box
#   appeared(object)         an object track that was not there when the stream started
#   owner(object)            the person who stayed next to an object longest (OWN_TIME at least)
#   moved(object, since)     how far the object travelled since a moment
# The kinds of rules combine them (Evaluator.conditions).
FIRE = {"fire", "smoke"}
PRESENCE_BEFORE = 10.0   # s: someone near the spot this long before a fire is first seen -> they lit it
STREAM_SETTLE = 3.0      # s: fire already there in the first seconds of a stream is not an ignition
OWN_TIME = 1.5           # s next to an object to own it (or for an object to be "left" somewhere)
SNATCH_MOVE = 0.5        # the object travelled this x the taker's height since they touched it


def _kp(p, i):
    k = p["kpts"][i]
    return (k[0], k[1]) if k[2] >= KPT_CONF else None


def _near(pt, box, pad):
    x, y = pt
    x1, y1, x2, y2 = box
    return x1 - pad <= x <= x2 + pad and y1 - pad <= y <= y2 + pad


def _raised(p, wrist):
    """wrist above its shoulder (image y grows downwards)."""
    w = _kp(p, wrist)
    s = _kp(p, WRIST_SHOULDER[wrist]) or _kp(p, L_SHOULDER if wrist == R_WRIST else R_SHOULDER)
    return bool(w and s and w[1] < s[1])


def _center(b):
    return ((b[0] + b[2]) / 2, (b[1] + b[3]) / 2)


def _dist(a, b):
    return ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5


def _box_gap(a, b):
    """0 when two boxes overlap, else the distance between their edges."""
    dx = max(b[0] - a[2], a[0] - b[2], 0)
    dy = max(b[1] - a[3], a[1] - b[3], 0)
    return (dx * dx + dy * dy) ** 0.5


def _iou(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / u if u > 0 else 0.0


def _pid(p, i):
    return p.get("id", -1) if p.get("id", -1) != -1 else f"i{i}"


def _height(p):
    return max(1.0, p["box"][3] - p["box"][1])


class ObjectTracker:
    """Gives every object an id across frames (greedy, per class: best IoU, or
    the closest centre within 1.5 object sizes). The detectors run frame by
    frame without tracking; snatching and ignition need to know that it is
    the same bag / the same fire."""

    def __init__(self):
        self.tracks = {}       # id -> {cls, box, first, last, first_box}
        self.next = 1

    def update(self, objects, now, max_age):
        free = {tid for tid, t in self.tracks.items() if now - t["last"] <= max_age}
        for o in sorted(objects, key=lambda o: -o["conf"]):
            best, score = None, 0.0
            for tid in free:
                t = self.tracks[tid]
                if t["cls"] != o["cls"]:
                    continue
                size = max(t["box"][2] - t["box"][0], t["box"][3] - t["box"][1], 1.0)
                sc = max(_iou(t["box"], o["box"]), 0.5 * (1 - _dist(_center(t["box"]), _center(o["box"])) / (1.5 * size)))
                if sc > score:
                    best, score = tid, sc
            if best is None or score <= 0.05:
                best = self.next
                self.next += 1
                self.tracks[best] = {"cls": o["cls"], "first": now, "first_box": o["box"]}
            free.discard(best)
            self.tracks[best].update(box=o["box"], last=now)
            o["tid"] = best
        self.tracks = {tid: t for tid, t in self.tracks.items() if now - t["last"] <= max_age}


class Evaluator:
    """Keeps the timing state between frames. One per live stream."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.start = None      # time of the first frame of the stream
        self.last = None       # time of the previous frame
        self.dt = 0.0          # smoothed time between frames
        self.touch = {}        # (person, wrist, class) -> last time the wrist was on it
        self.otouch = {}       # (object id, person) -> [first, last] time a wrist was on that object
        self.near_time = {}    # (object id, person) -> seconds the person stood next to the object
        self.alone_time = {}   # object id -> seconds seen with nobody next to it
        self.history = {}      # person -> [(t, box)] over PRESENCE_BEFORE s
        self.taken_at = {}     # (object id, person) -> (box, owner, left alone): the object when that person
                               # first touched it. Ownership is decided then: whoever carries a bag
                               # stands next to it, so asking later would make the thief its owner.
        self.objects = ObjectTracker()
        self.state = {}        # (rule id, key) -> [first, last] time the condition was on
        self.fired = {}        # rule id -> last alert time

    @property
    def gap(self):
        return max(GAP, GAP_FRAMES * self.dt)

    @property
    def memory(self):
        return max(TOUCH_MEMORY, GAP_FRAMES * self.dt)

    # ------------------------------------------------ per-frame bookkeeping
    def _observe(self, persons, objects, now, dt):
        self.objects.update(objects, now, self.memory)
        for i, p in enumerate(persons):
            pid = _pid(p, i)
            h = self.history.setdefault(pid, [])
            h.append((now, p["box"]))
            while h and now - h[0][0] > PRESENCE_BEFORE + self.gap:
                h.pop(0)
            ph = _height(p)
            for w in (L_WRIST, R_WRIST):
                pt = _kp(p, w)
                if not pt:
                    continue
                for o in objects:
                    b = o["box"]
                    pad = max(0.3 * max(b[2] - b[0], b[3] - b[1]), 0.06 * ph)
                    if _near(pt, b, pad):
                        self.touch[(pid, w, o["cls"])] = now
                        st = self.otouch.setdefault((o["tid"], pid), [now, now])
                        st[1] = now
        for o in objects:
            c, nobody = _center(o["box"]), True
            for i, p in enumerate(persons):
                b, pad = p["box"], 0.25 * _height(p)
                if b[0] - pad <= c[0] <= b[2] + pad and b[1] - pad <= c[1] <= b[3] + pad:
                    key = (o["tid"], _pid(p, i))
                    self.near_time[key] = self.near_time.get(key, 0.0) + dt
                    nobody = False
            if nobody:
                self.alone_time[o["tid"]] = self.alone_time.get(o["tid"], 0.0) + dt
        self.touch = {k: t for k, t in self.touch.items() if now - t <= self.memory}
        self.otouch = {k: v for k, v in self.otouch.items() if now - v[1] <= max(self.memory, OWN_TIME * 4)}
        self.history = {k: h for k, h in self.history.items() if h and now - h[-1][0] <= PRESENCE_BEFORE + self.gap}

    def _held(self, pid, wrist, cls, now):
        t = self.touch.get((pid, wrist, cls))
        return t is not None and now - t <= self.memory

    def owner(self, tid, exclude=None):
        """the person who stood next to the object longest (OWN_TIME at least), or None."""
        best, secs = None, OWN_TIME
        for (t, pid), s in self.near_time.items():
            if t == tid and pid != exclude and s >= secs:
                best, secs = pid, s
        return best

    # ------------------------------------------------ the kinds of rules
    def conditions(self, rule, persons, objects, now):
        """keys (person id / object id / 'all') for which the rule's condition is true, with the box to show."""
        k, cls = rule["kind"], rule["object"]
        if k == "object":
            hits = [o for o in objects if o["cls"] == cls]
            return {"all": hits[0]["box"]} if hits else {}
        if k == "person":
            return {"all": None} if len(persons) >= rule["min_people"] else {}
        if k == "ignite":
            return self._ignite(objects)
        if k == "snatch":
            return self._snatch(cls, persons, objects, now)
        out = {}
        for i, p in enumerate(persons):
            pid = _pid(p, i)
            if k == "hold":
                on = any(self._held(pid, w, cls, now) for w in (L_WRIST, R_WRIST))
            elif k == "raise_object":
                on = any(self._held(pid, w, cls, now) and _raised(p, w) for w in (L_WRIST, R_WRIST))
            else:   # hands_up: one or both wrists above the shoulders
                on = _raised(p, L_WRIST) + _raised(p, R_WRIST) >= (rule.get("hands") or 2)
            if on:
                out[pid] = p["box"]
        return out

    def _ignite(self, objects):
        """fire / smoke that appeared (not there when the stream started) where
        someone was: a person's box near the fire's first box, at any moment
        from PRESENCE_BEFORE s before the fire was first seen until now."""
        out = {}
        for o in objects:
            if o["cls"] not in FIRE:
                continue
            t = self.objects.tracks.get(o["tid"])
            if not t or t["first"] - self.start < STREAM_SETTLE:
                continue
            fb = t["first_box"]
            for pid, hist in self.history.items():
                if any(t["first"] - PRESENCE_BEFORE <= ht and _box_gap(hb, fb) <= 0.5 * max(1.0, hb[3] - hb[1])
                       for ht, hb in hist):
                    out[f"o{o['tid']}"] = o["box"]
                    break
        return out

    def _snatch(self, cls, persons, objects, now):
        """someone takes an object that is not theirs and carries it away: a
        wrist of person P on the object, then the object travels SNATCH_MOVE x
        P's height from where P took it, while P still holds it. Not theirs =
        it has an owner who is not P (someone who stood next to it longest), or
        it was left alone for OWN_TIME before P came."""
        out, by_id = {}, {_pid(p, i): p for i, p in enumerate(persons)}
        for o in objects:
            if o["cls"] != cls:
                continue
            for (tid, pid), (first, last) in self.otouch.items():
                if tid != o["tid"] or now - last > self.memory or pid not in by_id or (tid, pid) not in self.taken_at:
                    continue
                start, owner, left = self.taken_at[(tid, pid)]
                if owner == pid or (owner is None and not left):
                    continue
                if _dist(_center(start), _center(o["box"])) >= SNATCH_MOVE * _height(by_id[pid]):
                    out[f"o{tid}"] = o["box"]
        return out

    # ------------------------------------------------ one frame
    def step(self, rules, persons, objects, now):
        """-> (alerts, progress): alerts that fire now, and per rule how far
        (0..1) its condition is towards `duration`."""
        if self.start is None:
            self.start = now
        d = 0.0
        if self.last is not None and now > self.last:
            d = now - self.last
            self.dt = d if not self.dt else 0.7 * self.dt + 0.3 * d
        self.last = now
        objects = [dict(o) for o in objects]          # tracker ids are added to copies
        self._observe(persons, objects, now, min(d, self.gap))
        # where each object was when each person first touched it ("moved since")
        for key, (first, last) in self.otouch.items():
            if first == now and key not in self.taken_at:
                box = next((o["box"] for o in objects if o.get("tid") == key[0]), None)
                if box:
                    self.taken_at[key] = (box, self.owner(key[0]), self.alone_time.get(key[0], 0.0) >= OWN_TIME)
        self.taken_at = {k: v for k, v in self.taken_at.items() if k in self.otouch}
        alerts, progress = [], {}
        for r in rules:
            best = 0.0
            for key, box in self.conditions(r, persons, objects, now).items():
                sk = (r["id"], key)
                st = self.state.get(sk)
                if st is None or now - st[1] > self.gap:
                    st = [now, now]
                else:
                    st[1] = now
                self.state[sk] = st
                held = st[1] - st[0]
                best = max(best, min(1.0, held / r["duration"]) if r["duration"] > 0 else 1.0)
                if held >= r["duration"] and now - self.fired.get(r["id"], -1e9) >= r["cooldown"]:
                    self.fired[r["id"]] = now
                    alerts.append({"rule": r["id"], "kind": r["kind"], "object": r["object"],
                                   "description": r["description"], "label": r["label"], "text": r["text"],
                                   "person": key if not str(key).startswith("o") and key != "all" else None,
                                   "box": box})
            progress[r["id"]] = round(best, 2)
        # forget conditions that went away long ago
        self.state = {k: v for k, v in self.state.items() if now - v[1] <= self.gap}
        return alerts, progress
