"""
Builds the phrase dataset for our alert-request model from templates.

Every example: {"text", "kind", "object", "hands", "split", "template"}
  kind    object / hold / raise_object / hands_up / person / none
          (none = something we cannot detect, or not an alert request at all)
  object  detector class for object / hold / raise_object, else ""
  hands   1 or 2 for hands_up, else 0

Generalisation is measured honestly: some wrappers ("תודיעי לי כש…") and some
words ("מגביה", "brandishes", "סמארטפון") appear ONLY in the test split, so
test sentences are never just training sentences with another object.

Template data is a start. Two more test files are scored by evaluate.py:
  data/test_handwritten.jsonl  free sentences written by hand, not from templates
  data/test_people.jsonl       sentences real people wrote: collect these (see README)

    python nlu/make_dataset.py                 -> nlu/data/train.jsonl, nlu/data/test_templates.jsonl
"""
import argparse
import itertools
import json
import os
import random

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")

# ------------------------------------------------------------ Hebrew
HE_WRAP = ["תתריעי לי אם {x}", "תתריע לי אם {x}", "תתריעי כש{x}", "אני רוצה התרעה כש{x}", "{x}",
           "אם {x} תתריעי", "התרעה אם {x}", "תצפצפי אם {x}", "שלח לי התרעה כש{x}", "תגיד לי אם {x}"]
HE_WRAP_TEST = ["תודיעי לי כש{x}", "תעדכן אותי אם {x}", "בבקשה תתריע ברגע ש{x}", "אני צריכה לדעת אם {x}"]
HE_SUBJ_M = ["מישהו", "אדם", "בנאדם", "איש", "גבר", "אחד"]
HE_SUBJ_F = ["מישהי", "אישה", "בחורה"]
HE_SUBJ_TEST = ["ילד", "שומר"]                    # masculine, test only
HE_RAISE_M = ["מרים", "מניף", "מרים למעלה", "מרים באוויר", "מכוון", "מאיים עם"]
HE_RAISE_F = ["מרימה", "מניפה", "מרימה למעלה", "מכוונת"]
HE_RAISE_TEST = ["מגביה", "מנופף ב"]
HE_HOLD_M = ["מחזיק", "אוחז", "תופס", "מחזיק ביד", "עם", "לוקח", "מסתובב עם", "הולך עם", "מחזיק בידו"]
HE_HOLD_F = ["מחזיקה", "אוחזת", "תופסת", "עם", "לוקחת", "הולכת עם"]
HE_HOLD_TEST = ["שולף", "נושא"]
HE_OBJ = {
    "knife": ["סכין", "סכין מטבח", "להב"],
    "gun": ["אקדח", "נשק", "רובה", "כלי נשק"],
    "cell phone": ["טלפון", "פלאפון", "נייד"],
    "scissors": ["מספריים"],
    "bottle": ["בקבוק"],
    "cup": ["כוס", "ספל"],
    "baseball bat": ["מחבט", "אלה"],
    "laptop": ["מחשב נייד", "לפטופ"],
    "book": ["ספר"],
    "bag": ["תיק", "תרמיל", "ארנק", "תיק יד"],
    "umbrella": ["מטריה"],
}
HE_OBJ_TEST = {"knife": ["סכין יפנית"], "gun": ["אקדח צעצוע"], "cell phone": ["סמארטפון"]}
HE_SCENE = {"fire": ["אש", "שריפה", "להבות"], "smoke": ["עשן"], "dog": ["כלב"], "cat": ["חתול"]}
HE_OBJ_PATTERNS = ["יש {o}", "רואים {o}", "{o} בפריים", "יש {o} בחדר", "מזהים {o}", "{o}", "מופיע {o}"]
HE_OBJ_PATTERNS_TEST = ["נראה {o} במצלמה", "פתאום יש {o}"]
HE_HAND_ONE = ["{s} {r} יד", "{s} {r} את היד", "{s} {r} יד אחת", "יד מורמת", "{s} {r} יד למעלה"]
HE_HAND_BOTH = ["{s} {r} ידיים", "{s} {r} את הידיים", "{s} {r} את שתי הידיים", "ידיים למעלה", "שתי ידיים באוויר"]
HE_HAND_TEST = [("{s} {r} את היד הימנית", 1), ("שתי הידיים למעלה", 2)]
HE_PERSON = ["{s} נכנס", "{s} נכנס לחדר", "יש מישהו בפריים", "{s} מופיע", "יש {s} בחדר", "{n} אנשים בחדר",
             "יותר מ{n} אנשים", "לפחות {n} אנשים", "יש יותר מ-{n} אנשים בפריים"]
HE_PERSON_TEST = ["{s} מגיע", "{n} אנשים או יותר"]
HE_NONE = ["{s} נופל", "{s} נפל", "{s} רץ", "{s} צועק", "{s} גונב", "{s} בוכה", "שני אנשים רבים", "יש מכות",
           "{s} יושב", "{s} ישן", "{s} שוכב על הרצפה", "{s} שותה", "{s} אוכל", "{s} מדבר", "מה השעה", "תודה",
           "היי", "מה שלומך", "איך זה עובד", "מה את יודעת לעשות", "בוקר טוב", "כמה אנשים יש", "תפסיקי",
           "{s} מרביץ", "{s} בועט", "{s} דוחף", "{s} מעשן", "{s} שובר חלון", "{s} רוקד", "{s} שר", "{s} מחייך",
           "יש תאונה", "{s} מתווכח", "מה מזג האוויר", "ספרי לי בדיחה", "שלום", "תודה רבה", "אוקיי", "מה קורה",
           "מה עשית היום", "איך קוראים לך", "מישהו נפגע", "{s} מטפס על הגדר", "{s} פורץ לרכב", "{s} מתנהג מוזר"]
HE_NONE_TEST = ["{s} מחליק", "{s} מתעלף", "יש פה ריב", "מה נשמע"]
HE_NUM = ["שני", "שלושה", "2", "3", "4"]
HE_DUR = ["", "", "", " במשך 2 שניות", " ליותר משנייה", " למשך 3 שניות"]

# ------------------------------------------------------------ English
EN_WRAP = ["alert me if {x}", "alert me when {x}", "let me know if {x}", "notify me when {x}", "warn me if {x}",
           "{x}", "I want an alert when {x}", "send an alert if {x}", "beep if {x}", "tell me if {x}"]
EN_WRAP_TEST = ["ping me as soon as {x}", "could you alert me whenever {x}", "give me a heads up if {x}"]
EN_SUBJ = ["someone", "somebody", "a person", "anyone", "a man", "a woman", "a guy"]
EN_SUBJ_TEST = ["a kid", "the guard"]
EN_RAISE = ["raises", "lifts", "swings", "waves", "holds up", "is raising", "is swinging", "lifts up",
            "points", "is pointing", "threatens with", "waves around"]
EN_RAISE_TEST = ["brandishes", "hoists"]
EN_HOLD = ["holds", "is holding", "has", "grabs", "carries", "picks up", "is carrying", "takes",
           "walks around with", "is using"]
EN_HOLD_TEST = ["pulls out", "is gripping"]
EN_WITH = ["{s} with a {o}"]
EN_OBJ = {
    "knife": ["knife", "kitchen knife", "blade"],
    "gun": ["gun", "pistol", "weapon", "rifle"],
    "cell phone": ["phone", "cell phone", "mobile"],
    "scissors": ["pair of scissors", "scissors"],
    "bottle": ["bottle"],
    "cup": ["cup", "mug"],
    "baseball bat": ["bat", "baseball bat"],
    "laptop": ["laptop"],
    "book": ["book"],
    "bag": ["bag", "backpack", "purse", "handbag"],
    "umbrella": ["umbrella"],
}
EN_OBJ_TEST = {"knife": ["box cutter"], "gun": ["handgun"], "cell phone": ["smartphone"]}
EN_SCENE = {"fire": ["fire", "flames"], "smoke": ["smoke"], "dog": ["dog"], "cat": ["cat"]}
EN_OBJ_PATTERNS = ["there is a {o}", "a {o} appears", "you see a {o}", "a {o} is visible", "{o} in the frame",
                   "{o}", "there's {o} in the room"]
EN_OBJ_PATTERNS_TEST = ["a {o} shows up on camera", "any {o}"]
EN_HAND_ONE = ["{s} raises a hand", "{s} raises his hand", "{s} raises her hand", "{s} puts a hand up",
               "a hand goes up", "{s} lifts one hand", "{s} waves"]
EN_HAND_BOTH = ["hands up", "{s} raises both hands", "{s} puts their hands up", "both arms in the air",
                "{s} raises their arms", "{s} lifts both arms"]
EN_HAND_TEST = [("{s} sticks a hand up", 1), ("{s} throws both hands in the air", 2)]
EN_PERSON = ["{s} enters", "{s} comes in", "{s} appears", "{s} is in the room", "{s} walks in",
             "more than {n} people", "at least {n} people", "{n} people in the frame", "there are {n} people"]
EN_PERSON_TEST = ["{s} shows up", "{n} or more people"]
EN_NONE = ["{s} falls", "{s} falls down", "a fight breaks out", "{s} is running", "{s} screams", "{s} steals",
           "{s} is sitting", "{s} is sleeping", "{s} lies on the floor", "{s} drinks", "{s} is eating",
           "{s} talks", "hello", "thanks", "what time is it", "how does this work", "what can you do",
           "good morning", "how many people are there", "stop", "{s} punches", "{s} kicks", "{s} pushes",
           "{s} smokes", "{s} breaks a window", "{s} dances", "{s} sings", "{s} smiles", "there is an accident",
           "{s} argues", "what's the weather", "tell me a joke", "hi", "thank you", "ok", "how are you",
           "what's your name", "{s} gets hurt", "{s} climbs the fence", "{s} breaks into a car", "{s} acts weird"]
EN_NONE_TEST = ["{s} slips", "{s} faints", "two people are arguing", "what's up"]
EN_NUM = ["2", "3", "4", "two", "three"]
EN_DUR = ["", "", "", " for 2 seconds", " for more than a second", " for 3 seconds"]


def ex(text, kind, obj="", hands=0, split="train", template=""):
    return {"text": " ".join(text.split()), "kind": kind, "object": obj, "hands": hands,
            "split": split, "template": template}


def _join(verb, noun):
    """'מנופף ב' + 'סכין' -> 'מנופף בסכין' (a trailing one-letter prefix sticks to the noun)."""
    return verb + noun if verb.endswith(" ב") else f"{verb} {noun}"


def he_raise_hold(subj_m, subj_f, raise_m, raise_f, hold_m, hold_f, objs):
    out = []
    for cls, words in objs.items():
        for w in words:
            for s, r in itertools.product(subj_m, raise_m):
                out.append(("raise_object", cls, f"{s} {_join(r, w)}"))
            for s, r in itertools.product(subj_f, raise_f):
                out.append(("raise_object", cls, f"{s} {_join(r, w)}"))
            for s, h in itertools.product(subj_m, hold_m):
                out.append(("hold", cls, f"{s} {_join(h, w)}"))
            for s, h in itertools.product(subj_f, hold_f):
                out.append(("hold", cls, f"{s} {_join(h, w)}"))
    return out


def build(seed=0, per_kind=900):
    rnd = random.Random(seed)
    rows = []

    def add(split, lang, kind, obj, body, hands=0, wraps=None, dur=None):
        wraps = wraps or (HE_WRAP if lang == "he" else EN_WRAP)
        w = rnd.choice(wraps)
        d = rnd.choice(dur or (HE_DUR if lang == "he" else EN_DUR))
        rows.append(ex(w.format(x=body) + d, kind, obj, hands, split, f"{lang}:{kind}"))

    for split in ("train", "test"):
        test = split == "test"
        # -------- Hebrew raise / hold
        if test:
            pool = he_raise_hold(HE_SUBJ_TEST + HE_SUBJ_M[:1], [], HE_RAISE_TEST, [], HE_HOLD_TEST, [],
                                 {**HE_OBJ, **HE_OBJ_TEST})
            pool += he_raise_hold(HE_SUBJ_TEST, [], HE_RAISE_M, [], HE_HOLD_M, [], HE_OBJ_TEST)
            wraps = HE_WRAP_TEST
        else:
            pool = he_raise_hold(HE_SUBJ_M, HE_SUBJ_F, HE_RAISE_M, HE_RAISE_F, HE_HOLD_M, HE_HOLD_F, HE_OBJ)
            wraps = HE_WRAP
        n = per_kind // 3 if test else per_kind
        for kind, cls, body in rnd.sample(pool, min(len(pool), 2 * n)):
            add(split, "he", kind, cls, body, wraps=wraps)
        # -------- English raise / hold
        subj = EN_SUBJ_TEST if test else EN_SUBJ
        raises, holds = (EN_RAISE_TEST, EN_HOLD_TEST) if test else (EN_RAISE, EN_HOLD)
        objs = {**EN_OBJ, **EN_OBJ_TEST} if test else EN_OBJ
        pool = []
        for cls, words in objs.items():
            for w in words:
                art = "an" if w[0] in "aeiou" else "a"
                for s, r in itertools.product(subj, raises):
                    pool.append(("raise_object", cls, f"{s} {r} {art} {w}"))
                for s, h in itertools.product(subj, holds):
                    pool.append(("hold", cls, f"{s} {h} {art} {w}"))
                for s in subj:
                    pool.append(("hold", cls, f"{s} with {art} {w}"))
        for kind, cls, body in rnd.sample(pool, min(len(pool), 2 * n)):
            add(split, "en", kind, cls, body, wraps=EN_WRAP_TEST if test else EN_WRAP)
        # -------- objects in the scene
        he_pat = HE_OBJ_PATTERNS_TEST if test else HE_OBJ_PATTERNS
        en_pat = EN_OBJ_PATTERNS_TEST if test else EN_OBJ_PATTERNS
        he_objs = {**HE_OBJ, **HE_SCENE, **(HE_OBJ_TEST if test else {})}
        en_objs = {**EN_OBJ, **EN_SCENE, **(EN_OBJ_TEST if test else {})}
        for _ in range(n):
            cls = rnd.choice(list(he_objs)); add(split, "he", "object", cls, rnd.choice(he_pat).format(o=rnd.choice(he_objs[cls])),
                                                 wraps=HE_WRAP_TEST if test else HE_WRAP)
            cls = rnd.choice(list(en_objs)); add(split, "en", "object", cls, rnd.choice(en_pat).format(o=rnd.choice(en_objs[cls])),
                                                 wraps=EN_WRAP_TEST if test else EN_WRAP)
        # -------- hands
        for _ in range(n):
            if test:
                t, h = rnd.choice(HE_HAND_TEST); body = t.format(s=rnd.choice(HE_SUBJ_TEST), r=rnd.choice(HE_RAISE_M))
            else:
                h = rnd.choice((1, 2))
                female = rnd.random() < 0.3
                t = rnd.choice(HE_HAND_ONE if h == 1 else HE_HAND_BOTH)
                body = t.format(s=rnd.choice(HE_SUBJ_F if female else HE_SUBJ_M),
                                r=rnd.choice(HE_RAISE_F if female else HE_RAISE_M))
            add(split, "he", "hands_up", "", body, hands=h, wraps=HE_WRAP_TEST if test else HE_WRAP)
            if test:
                t, h = rnd.choice(EN_HAND_TEST)
            else:
                h = rnd.choice((1, 2)); t = rnd.choice(EN_HAND_ONE if h == 1 else EN_HAND_BOTH)
            add(split, "en", "hands_up", "", t.format(s=rnd.choice(subj)), hands=h,
                wraps=EN_WRAP_TEST if test else EN_WRAP)
        # -------- people
        for _ in range(n):
            pat = rnd.choice(HE_PERSON_TEST if test else HE_PERSON)
            add(split, "he", "person", "", pat.format(s=rnd.choice(HE_SUBJ_TEST if test else HE_SUBJ_M), n=rnd.choice(HE_NUM)),
                wraps=HE_WRAP_TEST if test else HE_WRAP, dur=[""])
            pat = rnd.choice(EN_PERSON_TEST if test else EN_PERSON)
            add(split, "en", "person", "", pat.format(s=rnd.choice(subj), n=rnd.choice(EN_NUM)),
                wraps=EN_WRAP_TEST if test else EN_WRAP, dur=[""])
        # -------- none: not detectable / not an alert request
        for _ in range(n):
            pat = rnd.choice(HE_NONE_TEST if test else HE_NONE)
            body = pat.format(s=rnd.choice(HE_SUBJ_TEST if test else HE_SUBJ_M))
            rows.append(ex(body if rnd.random() < 0.4 else rnd.choice(HE_WRAP_TEST if test else HE_WRAP).format(x=body),
                           "none", split=split, template="he:none"))
            pat = rnd.choice(EN_NONE_TEST if test else EN_NONE)
            body = pat.format(s=rnd.choice(subj))
            rows.append(ex(body if rnd.random() < 0.4 else rnd.choice(EN_WRAP_TEST if test else EN_WRAP).format(x=body),
                           "none", split=split, template="en:none"))
    # unique sentences; a test sentence that also appears in train is dropped
    seen, out = set(), []
    for r in rows:
        if r["split"] == "train":
            if r["text"] not in seen:
                seen.add(r["text"]); out.append(r)
    test_seen = set()
    for r in rows:
        if r["split"] == "test" and r["text"] not in seen and r["text"] not in test_seen:
            test_seen.add(r["text"]); out.append(r)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--per-kind", type=int, default=900)
    args = ap.parse_args()
    rows = build(args.seed, args.per_kind)
    os.makedirs(DATA, exist_ok=True)
    for split, name in (("train", "train.jsonl"), ("test", "test_templates.jsonl")):
        part = [r for r in rows if r["split"] == split]
        with open(os.path.join(DATA, name), "w", encoding="utf-8") as f:
            for r in part:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        kinds = {}
        for r in part:
            kinds[r["kind"]] = kinds.get(r["kind"], 0) + 1
        print(f"{name}: {len(part)} sentences  {kinds}")


if __name__ == "__main__":
    main()
