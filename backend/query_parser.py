"""
Natural-language query -> structured query.

Rule-based vocabulary (Hebrew + English), runs locally: nothing is sent to
an outside service. (An earlier version could call the Claude API; it was
removed so that the whole system runs on our own models.)

Returns:
  {
    "objects":   ["person", "dog"],      # detector classes, subject first
    "colors":    ["red"],                # of the subject
    "actions":   ["running"],            # running/walking/moving/parked/entering/exiting
    "timeRange": [lo, hi] | None,        # seconds, hi may be None (= end)
    "semantic":  "a woman pushing a stroller" | "",   # free text for CLIP
    "unknown":   ["stroller"],           # words we could not map to a class
    "source":    "rules",
  }
"""

import json
import os
import re
from typing import Optional

# ------------------------------------------------------------- vocabulary
# detector class -> words (English + Hebrew). Classes are COCO names, except
# the three bag classes that are merged into "bag".
VOCAB_OBJECTS = {
    "person": ["person", "persons", "people", "man", "men", "woman", "women",
               "someone", "somebody", "pedestrian", "pedestrians", "guy", "girl",
               "boy", "child", "kid", "kids", "human",
               "איש", "אדם", "אישה", "אנשים", "בחור", "בחורה", "ילד", "ילדה",
               "ילדים", "גבר", "נשים", "מישהו", "הולך רגל", "הולכי רגל", "בנאדם"],
    "dog": ["dog", "dogs", "puppy", "כלב", "כלבה", "כלבים"],
    "cat": ["cat", "cats", "kitten", "חתול", "חתולה", "חתולים"],
    "car": ["car", "cars", "vehicle", "vehicles", "רכב", "מכונית", "אוטו",
            "רכבים", "מכוניות"],
    "bus": ["bus", "buses", "אוטובוס", "אוטובוסים"],
    "truck": ["truck", "trucks", "lorry", "משאית", "משאיות"],
    "bicycle": ["bicycle", "bicycles", "bike", "bikes", "cyclist", "אופניים",
                "אופנים", "רוכב אופניים"],
    "motorcycle": ["motorcycle", "motorcycles", "motorbike", "scooter",
                   "אופנוע", "קטנוע", "אופנועים"],
    "bag": ["bag", "bags", "backpack", "handbag", "suitcase", "תיק", "תיקים",
            "מזוודה", "תרמיל"],
    "train": ["train", "רכבת"],
    "boat": ["boat", "ship", "סירה", "ספינה"],
    "airplane": ["airplane", "plane", "מטוס"],
    "traffic light": ["traffic light", "רמזור"],
    "stop sign": ["stop sign", "תמרור עצור"],
    "bench": ["bench", "ספסל"],
    "bird": ["bird", "birds", "ציפור", "ציפורים"],
    "horse": ["horse", "סוס"],
    "sheep": ["sheep", "כבשה", "כבשים"],
    "cow": ["cow", "פרה", "פרות"],
    "umbrella": ["umbrella", "מטריה"],
    "sports ball": ["ball", "כדור"],
    "skateboard": ["skateboard", "סקייטבורד"],
    "bottle": ["bottle", "בקבוק"],
    "cup": ["cup", "כוס"],
    "chair": ["chair", "כיסא", "כסא"],
    "cell phone": ["phone", "cellphone", "smartphone", "טלפון", "פלאפון", "נייד"],
    "laptop": ["laptop", "מחשב נייד", "לפטופ"],
    "book": ["book", "ספר"],
    "potted plant": ["plant", "עציץ"],
    "tv": ["tv", "television", "טלוויזיה"],
    "fire hydrant": ["hydrant", "ברז כיבוי"],
    # classes of the security detector (train/security_dataset.py, EXTRA_MODELS=security.pt)
    "gun": ["gun", "guns", "pistol", "handgun", "rifle", "firearm", "weapon",
            "אקדח", "רובה", "נשק", "כלי נשק"],
    "knife": ["knife", "knives", "blade", "machete", "סכין", "סכינים", "להב"],
    "fire": ["fire", "flames", "flame", "burning", "blaze", "אש", "שריפה", "להבות", "בוער"],
    "smoke": ["smoke", "עשן"],
    "clock": ["clock", "שעון"],
    # classes of the fine-tuned game model (train/README.md)
    "character": ["character", "characters", "player", "hero", "avatar",
                  "sprite", "דמות", "דמויות", "שחקן", "גיבור"],
    "tree": ["tree", "trees", "עץ", "עצים"],
    "fruit": ["fruit", "fruits", "apple", "apples", "lemon", "lemons",
              "פרי", "פירות", "תפוח", "תפוחים", "לימון", "לימונים"],
    "sun": ["sun", "שמש"],
}

# rare-event classes of the security detector. It misses most of them on real
# CCTV (eval/results/detector_security_*.txt), so "no detection" is not "not
# there": queries about them are also answered by CLIP
EVENT_CLASSES = {"gun", "knife", "fire", "smoke"}

# when the video was indexed by a model without this class, use the stand-in
CLASS_FALLBACK = {"person": "character", "sports ball": "fruit"}
VOCAB_COLORS = {
    "red": ["red", "אדום", "אדומה", "אדומים"],
    "white": ["white", "לבן", "לבנה", "לבנים"],
    "black": ["black", "dark", "שחור", "שחורה", "שחורים"],
    "gray": ["gray", "grey", "silver", "אפור", "אפורה", "כסוף"],
    "blue": ["blue", "כחול", "כחולה", "כחולים"],
    "green": ["green", "ירוק", "ירוקה", "ירוקים"],
    "yellow": ["yellow", "צהוב", "צהובה", "צהובים"],
    "orange": ["orange", "כתום", "כתומה"],
    "purple": ["purple", "violet", "סגול", "סגולה"],
    "pink": ["pink", "ורוד", "ורודה"],
    "brown": ["brown", "חום", "חומה"],
}
VOCAB_ACTIONS = {
    "running": ["running", "runs", "run", "jogging", "sprinting", "רץ", "רצה",
                "רצים", "רצות"],
    "walking": ["walking", "walks", "walk", "הולך", "הולכת", "הולכים", "צועד"],
    "moving": ["moving", "driving", "riding", "נוסע", "נוסעת", "נוסעים", "רוכב",
               "רוכבת", "זז"],
    "parked": ["parked", "parking", "standing", "stopped", "waiting", "חונה",
               "חונים", "עומד", "עומדת", "עומדים", "מחכה"],
    "entering": ["entering", "enter", "enters", "arriving", "coming", "נכנס",
                 "נכנסת", "נכנסים", "מגיע"],
    "exiting": ["exiting", "exit", "leaving", "leaves", "יוצא", "יוצאת",
                "יוצאים", "עוזב"],
    # direction of travel in the image (search.heading)
    "going left": ["to the left", "leftward", "leftwards", "moving left", "going left", "heading left",
                   "walking left", "running left", "driving left", "שמאלה", "לכיוון שמאל", "לשמאל"],
    "going right": ["to the right", "rightward", "rightwards", "moving right", "going right", "heading right",
                    "walking right", "running right", "driving right", "ימינה", "לכיוון ימין", "לימין"],
    "going up": ["going up", "moving up", "upward", "upwards", "כלפי מעלה", "עולה", "עולים"],
    "going down": ["going down", "moving down", "downward", "downwards", "כלפי מטה", "יורד", "יורדת", "יורדים"],
    # where in the frame the object is (search.ZONES)
    "on the left": ["on the left", "at the left", "left side", "left part", "left half", "left of the frame",
                    "צד שמאל", "משמאל", "בשמאל"],
    "on the right": ["on the right", "at the right", "right side", "right part", "right half", "right of the frame",
                     "צד ימין", "מימין", "בימין"],
    "at the top": ["at the top", "top of the frame", "top part", "top half", "upper part", "upper half",
                   "בחלק העליון", "חלק עליון", "למעלה"],
    "at the bottom": ["at the bottom", "bottom of the frame", "bottom part", "bottom half", "lower part",
                      "lower half", "בחלק התחתון", "חלק תחתון", "למטה"],
    "in the center": ["in the center", "in the centre", "in the middle", "center of the frame",
                      "במרכז", "באמצע"],
}
# "for more than 30 seconds", "at least 2 minutes", "יותר מ-30 שניות", "יותר מדקה" -> minDuration
_LONGER = r"(?:for\s+)?(?:more than|longer than|over|at least|יותר מ|לפחות|מעל)"
DWELL = re.compile(_LONGER + r"[\s-]*(\d+(?:\.\d+)?)\s*(seconds?|secs?|minutes?|mins?|שניות|שנייה|שניה|דקות|דקה)",
                   re.IGNORECASE)
DWELL_MINUTE = re.compile(_LONGER + r"[\s-]*(?:(?:a|one)\s+minute|דקה)", re.IGNORECASE)
ACTIONS = list(VOCAB_ACTIONS)
COLORS = list(VOCAB_COLORS)

HE_PREFIX = re.compile(r"^[הוושבלמכ]")
STOP = {"a", "an", "the", "with", "and", "or", "in", "on", "at", "of", "to",
        "find", "show", "me", "all", "any", "is", "are", "that", "who", "where",
        "את", "עם", "של", "על", "מצא", "תמצא", "תראה", "לי", "כל", "ש", "ו",
        "frame", "screen", "image", "פריים", "הפריים", "בפריים", "מסך", "המסך", "במסך",
        # clothing words: the colour already describes them
        "shirt", "shirts", "jacket", "coat", "pants", "dress", "clothes",
        "wearing", "wears", "dressed", "חולצה", "מעיל", "מכנסיים", "שמלה",
        "לובש", "לובשת", "בגדים",
        # where one thing is relative to another: the multi-object search already means "together, near"
        "next", "near", "nearby", "beside", "around", "close", "ליד", "שליד", "קרוב", "לצד", "בקרבת", "מול", "משהו", "something"}
# "משהו שהולך" (literally "something that walks") is how people often write "someone": with an
# action of a person it is a person
SOMETHING = {"משהו", "something"}


def tokenize(text: str):
    return [t for t in re.split(r"[^a-z֐-׿]+", text.lower()) if t]


def _match_word(tokens, word):
    parts = tokenize(word)
    for i in range(len(tokens) - len(parts) + 1):
        ok = True
        for j, p in enumerate(parts):
            tok = tokens[i + j]
            if tok == p:
                continue
            if HE_PREFIX.match(tok) and tok[1:] == p:
                continue
            if HE_PREFIX.match(tok) and len(tok) > 2 and \
                    HE_PREFIX.match(tok[1:]) and tok[2:] == p:
                continue
            ok = False
            break
        if ok:
            return i
    return -1


def _to_sec(s: str) -> float:
    out = 0.0
    for part in s.split(":"):
        out = out * 60 + float(part)
    return out


MASS_NOUNS = {"fire", "smoke"}


def english_phrase(objects, colors, actions) -> str:
    """["person", "knife"], ["red"], ["running"] -> "a red person running with a knife"."""
    def noun(o, adj=()):
        words = " ".join([*adj, o])
        return words if o in MASS_NOUNS else f"a {words}"
    if not objects:
        return " ".join([*colors, *actions])
    out = " ".join([noun(objects[0], colors), *actions])
    for o in objects[1:]:
        out += f" with {noun(o)}"
    return out


def parse_rules(text: str, classes: Optional[list[str]] = None) -> dict:
    """Vocabulary parser. `classes`: classes present in the video (optional)."""
    found = {"objects": [], "colors": [], "actions": [], "timeRange": None, "minDuration": None,
             "semantic": "", "unknown": [], "source": "rules"}
    m = DWELL.search(text) or DWELL_MINUTE.search(text)
    if m:       # taken out of the sentence, so "more", "seconds" do not become words for CLIP
        unit = m.group(2).lower() if m.re is DWELL else "minute"
        found["minDuration"] = float(m.group(1) if m.re is DWELL else 1) * (60 if unit[0] in "mד" else 1)
        text = (text[:m.start()] + " " + text[m.end():]).strip()
    tokens = tokenize(text)
    lower = text.lower()
    positions = {}
    for cls, words in VOCAB_OBJECTS.items():
        pos = [p for p in (_match_word(tokens, w) for w in words) if p >= 0]
        if pos:
            positions[cls] = min(pos)
    found["objects"] = sorted(positions, key=positions.get)
    color_pos = {}
    for key, words in VOCAB_COLORS.items():
        pos = [p for p in (_match_word(tokens, w) for w in words) if p >= 0]
        if pos:
            found["colors"].append(key)
            color_pos[key] = min(pos)
    # colours apply to the subject (first object). "character near a tree
    # with a red apple": a colour right next to another object word makes
    # that object the subject, so the colour filters the right thing
    if len(found["objects"]) > 1 and color_pos:
        p = min(color_pos.values())
        nearest = min(found["objects"], key=lambda c: abs(positions[c] - p))
        if abs(positions[nearest] - p) <= 2 and nearest != found["objects"][0]:
            found["objects"].remove(nearest)
            found["objects"].insert(0, nearest)
    for key, words in VOCAB_ACTIONS.items():
        if any(_match_word(tokens, w) >= 0 for w in words):
            found["actions"].append(key)
    found["object_colors"] = {}
    something = [i for i, t in enumerate(tokens) if t in SOMETHING]
    if something and found["actions"] and "person" not in positions:
        positions["person"] = something[0]
        found["objects"] = sorted(positions, key=positions.get)
    # who does the action and what has the colour can be different objects:
    # "someone walking next to the white car" = a walking PERSON near a WHITE CAR.
    # The actor is the object named first; the colour stays on the object it
    # stands next to (object_colors), and the actor becomes the subject.
    if len(found["objects"]) > 1 and found["actions"] and color_pos:
        actor = min(positions, key=positions.get)
        p = min(color_pos.values())
        owner = min(positions, key=lambda c: abs(positions[c] - p))
        if owner != actor:
            found["objects"] = [actor] + [o for o in sorted(positions, key=positions.get) if o != actor]
            found["object_colors"] = {owner: found["colors"]}
            found["colors"] = []

    if classes:
        found["objects"] = [CLASS_FALLBACK.get(c, c)
                            if c not in classes and CLASS_FALLBACK.get(c) in classes
                            else c for c in found["objects"]]

    if {"bicycle", "motorcycle"} & set(found["objects"]):
        if re.search(r"\b(riding|rides|cyclist)\b|רוכב", lower) and \
                "person" not in found["objects"]:
            found["objects"].insert(0, "person")

    times = [_to_sec(m.group(1))
             for m in re.finditer(r"\b(\d{1,2}(?::\d{2}){1,2})\b", text)]
    if len(times) >= 2:
        found["timeRange"] = [min(times[:2]), max(times[:2])]
    elif len(times) == 1:
        if re.search(r"after|from|since|אחרי|מ-|החל", lower):
            found["timeRange"] = [times[0], None]
        elif re.search(r"before|until|לפני|עד", lower):
            found["timeRange"] = [0, times[0]]
        else:
            found["timeRange"] = [max(0, times[0] - 5), times[0] + 5]

    # words we did not understand -> semantic query for CLIP
    known = set()
    for words in (*VOCAB_OBJECTS.values(), *VOCAB_COLORS.values(),
                  *VOCAB_ACTIONS.values()):
        for w in words:
            known.update(tokenize(w))
    for tok in tokens:
        base = tok[1:] if HE_PREFIX.match(tok) and tok[1:] in known else tok
        if base not in known and base not in STOP and len(base) > 2 \
                and not base.isdigit():
            found["unknown"].append(tok)
    # CLIP reads English only: a Hebrew sentence is rebuilt from the words we know
    phrase = text.strip()
    if re.search("[֐-׿]", text):
        phrase = english_phrase(found["objects"], found["colors"], found["actions"]) or phrase
    if found["unknown"]:
        found["semantic"] = phrase
    # a class the detector did not find in this video ("a person with a gun"
    # indexed without security.pt) becomes a CLIP question about the rest
    # ... and an event class (fire, gun, knife) is always worth a CLIP look,
    # even when it was the only thing asked for: the video may have been
    # indexed without the security model, and that model misses a lot.
    # "a dog" in a video where the COCO model saw no dog stays an empty answer.
    if classes:
        absent = [c for c in found["objects"] if c not in classes]
        if len(absent) == len(found["objects"]):
            absent = [c for c in absent if c in EVENT_CLASSES]
        if absent:
            found["objects"] = [c for c in found["objects"] if c not in absent]
            found["unknown"] += [c for c in absent if c not in found["unknown"]]
            found["semantic"] = phrase
    return found


def parse(text: str, classes: Optional[list[str]] = None) -> dict:
    """The local rule-based parser."""
    classes = classes or list(VOCAB_OBJECTS)
    return parse_rules(text, classes)
