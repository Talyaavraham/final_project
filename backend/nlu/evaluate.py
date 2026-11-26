"""
How well does a sentence become the right alert rule? Three parsers: the keyword
parser (the baseline), our model alone, and the two combined (what the server
uses, alert_rules.understand), on every test file that has sentences:

    data/test_templates.jsonl   template sentences with wrappers / words never seen in training
    data/test_handwritten.jsonl free sentences written by hand
    data/test_people.jsonl      sentences real people wrote (collect them!)

    python nlu/evaluate.py                  # all of them
    python nlu/evaluate.py --errors         # + the sentences the model got wrong

"exact" = kind, object and hands all right: the rule the user would get.
"""
import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import alert_rules                      # noqa: E402
from nlu.model import load              # noqa: E402

NEEDS_OBJECT = alert_rules.NEEDS_OBJECT


def read(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def keyword_predict(text):
    r = alert_rules.parse_rules(text)
    if r is None:
        return {"kind": "none", "object": "", "hands": 0}
    return {"kind": r["kind"], "object": r["object"], "hands": r.get("hands", 0) if r["kind"] == "hands_up" else 0}


def model_predict(model, text):
    p = model.predict(text)
    return {"kind": p["kind"], "object": p["object"] if p["kind"] in NEEDS_OBJECT else "",
            "hands": p["hands"] if p["kind"] == "hands_up" else 0}


def score(rows, predict):
    n = len(rows)
    kind = obj = obj_n = hands = hands_n = exact = 0
    errors = []
    for r in rows:
        p = predict(r["text"])
        want_obj = r["object"] if r["kind"] in NEEDS_OBJECT else ""
        want_hands = int(r.get("hands") or 0) if r["kind"] == "hands_up" else 0
        k_ok = p["kind"] == r["kind"]
        kind += k_ok
        if r["kind"] in NEEDS_OBJECT:
            obj_n += 1; obj += p["object"] == want_obj
        if r["kind"] == "hands_up":
            hands_n += 1; hands += p["hands"] == want_hands
        ok = k_ok and p["object"] == want_obj and p["hands"] == want_hands
        exact += ok
        if not ok:
            errors.append((r["text"], f'{r["kind"]}/{want_obj or "-"}/{want_hands}', f'{p["kind"]}/{p["object"] or "-"}/{p["hands"]}'))
    pct = lambda a, b: f"{100 * a / b:5.1f}%" if b else "   — "
    return {"n": n, "kind": pct(kind, n), "object": pct(obj, obj_n), "hands": pct(hands, hands_n),
            "exact": pct(exact, n)}, errors


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--errors", action="store_true")
    ap.add_argument("--model", default=None)
    args = ap.parse_args()
    model = load(args.model) if args.model else load()
    if model is None:
        print("no trained model (nlu/model.npz): run  python nlu/train.py  first. Showing the keyword baseline only.")
    files = ["test_templates.jsonl", "test_handwritten.jsonl", "test_people.jsonl"]
    print(f"{'test set':24s} {'parser':8s} {'n':>4s} {'kind':>7s} {'object':>7s} {'hands':>7s} {'exact':>7s}")
    for f in files:
        path = os.path.join(HERE, "data", f)
        if not os.path.isfile(path):
            continue
        rows = read(path)
        if not rows:
            print(f"{f:24s} (empty: add sentences people wrote)")
            continue
        parsers = [("keywords", keyword_predict)]
        if model:
            parsers.append(("model", lambda t: model_predict(model, t)))
            parsers.append(("combined", lambda t: alert_rules.understand(t)))   # what the server uses
        for name, fn in parsers:
            s, errors = score(rows, fn)
            print(f"{f:24s} {name:8s} {s['n']:4d} {s['kind']:>7s} {s['object']:>7s} {s['hands']:>7s} {s['exact']:>7s}")
            if args.errors and name == "combined" and errors:
                for text, want, got in errors[:40]:
                    print(f"      ✗ {text}   want {want}  got {got}")
    print("\nkind/object/hands = accuracy of each part; exact = the whole rule right")


if __name__ == "__main__":
    main()
