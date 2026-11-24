"""
Live alert rules: a sentence (Hebrew / English) -> a rule, and the rule
evaluated frame by frame on what the pose model and the detectors see.

A rule is one of five kinds:

    object        an object appears in the frame         "alert me if there is a knife"
    hold          a person holds an object               "מישהו מחזיק טלפון"
    raise_object  a person raises an object above the    "תתריעי אם מישהו מרים סכין"
                  shoulder (the hand that held it)
    hands_up      a person raises a hand (hands=1) or     "מישהו מרים יד" / "hands up"
                  both hands (hands=2) above the shoulders
    person        at least N people in the frame         "יותר משני אנשים בחדר"

"Holds" = a wrist keypoint is on (near) the object's box. Detectors flicker
on small objects like knives, so a wrist that touched the object keeps it for
a moment (TOUCH_MEMORY): a knife that disappears in motion blur while it is
swung up still counts. A condition must hold for the rule's `duration`
(short gaps up to GAP are bridged) before an alert fires, and a rule fires
at most once per `cooldown`.

Pure logic, no models: parse() turns text into a rule, Evaluator.step()
takes the people / objects of one frame and returns the alerts.
"""
import json
import os
import re
import uuid
from typing import Optional

KINDS = ["object", "hold", "raise_object", "hands_up", "person"]
NEEDS_OBJECT = {"object", "hold", "raise_object"}

DEFAULT_DURATION = {"object": 0.5, "hold": 1.0, "raise_object": 0.4, "hands_up": 1.0, "person": 1.0}
DEFAULT_COOLDOWN = 5.0
GAP = 0.7               # s: a condition that is off for less than this is still "on"
TOUCH_MEMORY = 1.5      # s: a wrist that was on an object keeps "holding" it this long
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
DEFAULT_CLASSES = sorted(set(COCO + SECURITY))

# words -> detector class, for the rule-based parser (Claude handles the rest)
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
    "backpack": ["backpack", "bag", "תיק", "תרמיל"],
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
                     "steal", "steals", "stealing", "נופל", "נופלת", "נפל", "נפלה", "רב", "רבים", "מכה",
                     "מרביץ", "מרביצה", "בועט", "צועק", "צועקת", "רץ", "רצה", "גונב", "גונבת", "מתקוטטים"]
HE_NAMES = {"knife": "סכין", "gun": "נשק", "fire": "אש", "smoke": "עשן", "cell phone": "טלפון",
            "scissors": "מספריים", "bottle": "בקבוק", "cup": "כוס", "baseball bat": "מחבט",
            "laptop": "מחשב נייד", "book": "ספר", "backpack": "תיק", "umbrella": "מטריה",
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
        }[k]
    return {
        "object": f"a {o} is in view",
        "hold": f"someone holds a {o}",
        "raise_object": f"someone raises a {o}",
        "hands_up": "someone raises a hand" if rule.get("hands") == 1 else "someone raises both hands",
        "person": "someone is in view" if n == 1 else f"{n} or more people in view",
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
    for cls, words in OBJECT_WORDS.items():
        if cls in classes and _any(tokens, words):
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

    if _any(tokens, UNSUPPORTED_WORDS):
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


# ------------------------------------------------------------------ Claude
CLAUDE_MODEL = os.environ.get("CLAUDE_MODEL", "claude-opus-5-5")

SCHEMA = {
    "type": "object",
    "properties": {
        "kind": {"type": "string", "enum": KINDS + ["unsupported"]},
        "object": {"type": "string", "description": "Detector class for object / hold / raise_object, else \"\""},
        "min_people": {"type": "integer", "description": "For kind person: alert at this many people or more; else 1"},
        "hands": {"type": "integer", "description": "For kind hands_up: 1 = one hand is enough, 2 = both hands; else 0"},
        "duration": {"type": ["number", "null"], "description": "Seconds the condition must last, if the user said so; else null"},
        "description": {"type": "string", "description": "Friendly confirmation in the user's language, e.g. \"I'll alert you when someone raises a knife\" / \"אתריע לך כשמישהו מרים סכין\""},
        "label": {"type": "string", "description": "The event in 2-5 words, in the user's language, e.g. \"Someone raises a knife\" / \"מישהו מרים סכין\""},
        "reason": {"type": "string", "description": "For unsupported: what cannot be detected, in the user's language; else \"\""},
    },
    "required": ["kind", "object", "min_people", "hands", "duration", "description", "label", "reason"],
    "additionalProperties": False,
}

SYSTEM = """The user sets up an alert on a live camera, in Hebrew or English.
Turn the request into ONE rule for a system that sees, in every frame, people
(a skeleton with wrists and shoulders) and objects from a detector.

Kinds:
- object: an object is in the frame ("alert if there's a knife").
- hold: a person holds an object (a wrist on the object).
- raise_object: a person raises / lifts / swings / brandishes an object
  above the shoulder ("someone raises a knife", "מישהו מרים סכין").
- hands_up: a person raises a hand above the shoulder (no object). hands = 1
  for "a hand" / "יד", 2 for "both hands" / "ידיים" / "hands up".
- person: at least min_people people in the frame ("more than 2 people" -> 3;
  "someone enters" -> 1).
- unsupported: anything else (falls, fights, faces, sounds, zones of the room,
  objects that are not in the class list). Say why in "reason".

Allowed object classes: {classes}
Map synonyms to a class: weapon / pistol / נשק / אקדח -> gun, טלפון -> cell phone,
blade / סכין -> knife, שריפה -> fire.
"""


def llm_available() -> bool:
    try:
        import anthropic  # noqa: F401
    except ImportError:
        return False
    return bool(os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN"))


def parse_llm(text: str, classes) -> Optional[dict]:
    """The rule from Claude; {"unsupported": reason} when it cannot be detected;
    None when Claude cannot be used (no key, network, ...)."""
    try:
        import anthropic
        client = anthropic.Anthropic(timeout=20.0, max_retries=1)
        response = client.messages.create(
            model=CLAUDE_MODEL, max_tokens=1024,
            system=SYSTEM.format(classes=", ".join(classes)),
            messages=[{"role": "user", "content": text}],
            output_config={"effort": "low", "format": {"type": "json_schema", "schema": SCHEMA}},
        )
        if response.stop_reason != "end_turn":
            return None
        data = json.loads(next((b.text for b in response.content if b.type == "text"), ""))
    except Exception:          # no key / network / bad JSON -> keyword parser
        return None
    kind = data.get("kind")
    if kind == "unsupported" or kind not in KINDS:
        return {"unsupported": data.get("reason") or data.get("description") or ""}
    obj = data.get("object") or ""
    if kind in NEEDS_OBJECT and obj not in set(classes):
        return {"unsupported": data.get("reason") or f"'{obj}' is not a class the detector knows"}
    return make_rule(text, kind, obj, data.get("min_people") or 1, data.get("duration"),
                     description=data.get("description") or "", source="llm",
                     label=data.get("label") or "", hands=data.get("hands") or 2)


def parse(text: str, classes=None) -> dict:
    """-> {"rule": rule} or {"rule": None, "reason": ...}"""
    classes = list(classes or DEFAULT_CLASSES)
    if llm_available():
        r = parse_llm(text, classes)
        if r is not None:
            if "unsupported" in r:
                return {"rule": None, "reason": r["unsupported"], "source": "llm"}
            return {"rule": r, "source": "llm"}
    rule = parse_rules(text, classes)
    if rule is None:
        return {"rule": None, "source": "rules", "reason":
                "לא הצלחתי להבין על מה להתריע 🤔" if is_hebrew(text) else "I couldn't tell what to alert you about 🤔"}
    return {"rule": rule, "source": "rules"}


# ------------------------------------------------------------------ evaluation
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


class Evaluator:
    """Keeps the timing state between frames. One per live stream."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.touch = {}        # (person, wrist, class) -> last time the wrist was on it
        self.state = {}        # (rule id, key) -> [first, last] time the condition was on
        self.fired = {}        # rule id -> last alert time

    def _touches(self, persons, objects, now):
        for i, p in enumerate(persons):
            pid = p.get("id", -1) if p.get("id", -1) != -1 else f"i{i}"
            ph = p["box"][3] - p["box"][1]
            for w in (L_WRIST, R_WRIST):
                pt = _kp(p, w)
                if not pt:
                    continue
                for o in objects:
                    b = o["box"]
                    pad = max(0.3 * max(b[2] - b[0], b[3] - b[1]), 0.06 * ph)
                    if _near(pt, b, pad):
                        self.touch[(pid, w, o["cls"])] = now
        self.touch = {k: t for k, t in self.touch.items() if now - t <= TOUCH_MEMORY}

    def _held(self, pid, wrist, cls, now):
        t = self.touch.get((pid, wrist, cls))
        return t is not None and now - t <= TOUCH_MEMORY

    def conditions(self, rule, persons, objects, now):
        """keys (person id / 'all') for which the rule's condition is true, with the box to show."""
        k, cls = rule["kind"], rule["object"]
        if k == "object":
            hits = [o for o in objects if o["cls"] == cls]
            return {"all": hits[0]["box"]} if hits else {}
        if k == "person":
            return {"all": None} if len(persons) >= rule["min_people"] else {}
        out = {}
        for i, p in enumerate(persons):
            pid = p.get("id", -1) if p.get("id", -1) != -1 else f"i{i}"
            if k == "hold":
                on = any(self._held(pid, w, cls, now) for w in (L_WRIST, R_WRIST))
            elif k == "raise_object":
                on = any(self._held(pid, w, cls, now) and _raised(p, w) for w in (L_WRIST, R_WRIST))
            else:   # hands_up: one or both wrists above the shoulders
                on = _raised(p, L_WRIST) + _raised(p, R_WRIST) >= (rule.get("hands") or 2)
            if on:
                out[pid] = p["box"]
        return out

    def step(self, rules, persons, objects, now):
        """-> (alerts, progress): alerts that fire now, and per rule how far
        (0..1) its condition is towards `duration`."""
        self._touches(persons, objects, now)
        alerts, progress = [], {}
        for r in rules:
            best = 0.0
            for key, box in self.conditions(r, persons, objects, now).items():
                sk = (r["id"], key)
                st = self.state.get(sk)
                if st is None or now - st[1] > GAP:
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
                                   "person": key if key != "all" else None, "box": box})
            progress[r["id"]] = round(best, 2)
        # forget conditions that went away long ago
        self.state = {k: v for k, v in self.state.items() if now - v[1] <= GAP}
        return alerts, progress
