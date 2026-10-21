"""
Natural-language query -> structured query.

Two parsers, used in this order:
  1. Claude (Anthropic SDK)  — handles free Hebrew/English, negation,
                               "two people", "a woman with a stroller", ...
                               Needs ANTHROPIC_API_KEY (or `ant auth login`).
  2. Rule-based vocabulary   — offline fallback, same behaviour as the
                               original front-end parser.

Both return:
  {
    "objects":   ["person", "dog"],      # detector classes, subject first
    "colors":    ["red"],                # of the subject
    "actions":   ["running"],            # running/walking/moving/parked/entering/exiting
    "timeRange": [lo, hi] | None,        # seconds, hi may be None (= end)
    "semantic":  "a woman pushing a stroller" | "",   # free text for CLIP
    "unknown":   ["stroller"],           # words we could not map to a class
    "source":    "llm" | "rules",
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
    "clock": ["clock", "שעון"],
    # classes of the fine-tuned game model (train/README.md)
    "character": ["character", "characters", "player", "hero", "avatar",
                  "sprite", "דמות", "דמויות", "שחקן", "גיבור"],
    "tree": ["tree", "trees", "עץ", "עצים"],
    "fruit": ["fruit", "fruits", "apple", "apples", "lemon", "lemons",
              "פרי", "פירות", "תפוח", "תפוחים", "לימון", "לימונים"],
    "sun": ["sun", "שמש"],
}

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
}
ACTIONS = list(VOCAB_ACTIONS)
COLORS = list(VOCAB_COLORS)

HE_PREFIX = re.compile(r"^[הוושבלמכ]")
STOP = {"a", "an", "the", "with", "and", "or", "in", "on", "at", "of", "to",
        "find", "show", "me", "all", "any", "is", "are", "that", "who", "where",
        "את", "עם", "של", "על", "מצא", "תמצא", "תראה", "לי", "כל", "ש", "ו",
        # clothing words: the colour already describes them
        "shirt", "shirts", "jacket", "coat", "pants", "dress", "clothes",
        "wearing", "wears", "dressed", "חולצה", "מעיל", "מכנסיים", "שמלה",
        "לובש", "לובשת", "בגדים"}


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


def parse_rules(text: str, classes: Optional[list[str]] = None) -> dict:
    """Vocabulary parser. `classes`: classes present in the video (optional)."""
    tokens = tokenize(text)
    lower = text.lower()
    found = {"objects": [], "colors": [], "actions": [], "timeRange": None,
             "semantic": "", "unknown": [], "source": "rules"}
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
    if found["unknown"]:
        found["semantic"] = text.strip()
    return found


# ------------------------------------------------------------- Claude LLM
MODEL = os.environ.get("CLAUDE_MODEL", "claude-opus-5-5")

SCHEMA = {
    "type": "object",
    "properties": {
        "objects": {"type": "array", "items": {"type": "string"},
                    "description": "Detector classes to look for, the subject "
                                   "of the sentence first. Only values from "
                                   "the allowed class list."},
        "colors": {"type": "array", "items": {"type": "string", "enum": COLORS},
                   "description": "Colour of the subject's clothes / body."},
        "actions": {"type": "array", "items": {"type": "string", "enum": ACTIONS}},
        "time_start": {"type": ["number", "null"],
                       "description": "Start of the time window in seconds, "
                                      "or null"},
        "time_end": {"type": ["number", "null"],
                     "description": "End of the time window in seconds, or null"},
        "semantic": {"type": "string",
                     "description": "Short English description of the visual "
                                    "appearance that the class list cannot "
                                    "express (e.g. 'a woman pushing a "
                                    "stroller', 'a man wearing a hat'). "
                                    "Empty string when the classes, colours "
                                    "and actions already cover the query."},
        "unknown": {"type": "array", "items": {"type": "string"},
                    "description": "Query terms that do not map to any class."},
    },
    "required": ["objects", "colors", "actions", "time_start", "time_end",
                 "semantic", "unknown"],
    "additionalProperties": False,
}

SYSTEM = """You turn a user's request to search inside a video (Hebrew or English)
into a structured query for an object detector.

Allowed detector classes: {classes}
Colours: {colors}
Actions: {actions}  (running/walking/moving/parked = speed of the object,
entering/exiting = the object enters or leaves the frame)

Rules:
- "objects" must only contain allowed classes. Map synonyms (man, woman, kid,
  pedestrian, איש, ילדה -> person; אוטו, רכב -> car; backpack, suitcase -> bag).
- The subject of the sentence goes first: "a man in red with a dog" ->
  objects ["person","dog"], colors ["red"].
- "riding a bicycle" is person + bicycle.
- Time expressions in the user's text are video timestamps ("after 1:30",
  "בין 0:10 ל-2:00"). Convert to seconds. "at 1:30" -> 85 to 95.
- Put in "semantic" an English phrase describing anything visual that the
  classes/colours/actions do not capture (clothing items, age, objects that
  are not in the list, "holding a phone", ...). Otherwise "".
- Put the words you could not map in "unknown".
"""


def llm_available() -> bool:
    try:
        import anthropic  # noqa: F401
    except ImportError:
        return False
    return bool(os.environ.get("ANTHROPIC_API_KEY")
                or os.environ.get("ANTHROPIC_AUTH_TOKEN")
                or os.environ.get("ANTHROPIC_PROFILE")
                or os.path.exists(os.path.expanduser("~/.config/anthropic")))


def parse_llm(text: str, classes: list[str]) -> Optional[dict]:
    """Returns the parsed query, or None when Claude cannot be used."""
    try:
        import anthropic
    except ImportError:
        return None
    client = anthropic.Anthropic(timeout=20.0, max_retries=1)
    try:
        response = client.messages.create(
            model=MODEL,
            max_tokens=1024,
            system=SYSTEM.format(classes=", ".join(classes),
                                 colors=", ".join(COLORS),
                                 actions=", ".join(ACTIONS)),
            messages=[{"role": "user", "content": text}],
            output_config={"effort": "low",
                           "format": {"type": "json_schema", "schema": SCHEMA}},
        )
    except anthropic.AuthenticationError:
        return None
    except anthropic.RateLimitError:
        return None
    except anthropic.APIStatusError:
        return None
    except anthropic.APIConnectionError:
        return None
    if response.stop_reason != "end_turn":   # refusal / max_tokens -> rules
        return None
    raw = next((b.text for b in response.content if b.type == "text"), "")
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return None

    allowed = set(classes)
    objects = [o for o in data.get("objects", []) if o in allowed]
    lo, hi = data.get("time_start"), data.get("time_end")
    time_range = None
    if lo is not None or hi is not None:
        time_range = [float(lo or 0), None if hi is None else float(hi)]
    return {
        "objects": objects,
        "colors": [c for c in data.get("colors", []) if c in COLORS],
        "actions": [a for a in data.get("actions", []) if a in ACTIONS],
        "timeRange": time_range,
        "semantic": (data.get("semantic") or "").strip(),
        "unknown": list(data.get("unknown") or []),
        "source": "llm",
    }


def parse(text: str, classes: Optional[list[str]] = None) -> dict:
    """Claude when available, otherwise the rule-based parser."""
    classes = classes or list(VOCAB_OBJECTS)
    if llm_available():
        result = parse_llm(text, classes)
        if result is not None:
            return result
    return parse_rules(text, classes)
