"""
Train our alert-request model: three small classifiers (logistic regression)
on top of a sentence encoder.

    kind    object / hold / raise_object / hands_up / person / none
    object  which detector class (only sentences that need an object)
    hands   1 or 2 (only hands_up sentences)

    pip install scikit-learn
    python nlu/make_dataset.py                       # the template sentences
    python nlu/train.py                              # -> nlu/model.npz  (encoder: hash, seconds on a CPU)
    python nlu/train.py --encoder minilm             # multilingual sentence encoder (download ~470 MB)
    python nlu/train.py --extra nlu/data/people_train.jsonl   # + sentences people wrote
    python nlu/evaluate.py                           # how good is it?
"""
import argparse
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from nlu.features import make_encoder          # noqa: E402

NEEDS_OBJECT = {"object", "hold", "raise_object"}


def read(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def fit_head(X, y, C):
    from sklearn.linear_model import LogisticRegression
    labels = sorted(set(y))
    if len(labels) == 1:                   # one class only: constant head
        return np.zeros((X.shape[1], 1), np.float16), np.zeros(1, np.float32), labels
    clf = LogisticRegression(C=C, max_iter=3000, class_weight="balanced")
    clf.fit(X, y)
    W, b = clf.coef_.T, clf.intercept_
    if W.shape[1] == 1:                    # binary: sklearn keeps one column
        W, b = np.hstack([-W, W]), np.array([-b[0], b[0]])
    return W.astype(np.float16), b.astype(np.float32), [str(c) for c in clf.classes_]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--train", default=os.path.join(HERE, "data", "train.jsonl"))
    ap.add_argument("--extra", nargs="*", default=[], help="more training files (same format)")
    ap.add_argument("--encoder", default="hash", choices=["hash", "minilm"])
    ap.add_argument("--C", type=float, default=10.0, help="inverse regularisation strength")
    ap.add_argument("--out", default=os.path.join(HERE, "model.npz"))
    args = ap.parse_args()

    rows = read(args.train)
    for p in args.extra:
        rows += read(p)
    t0 = time.time()
    enc = make_encoder(args.encoder)
    X = enc.sparse_matrix([r["text"] for r in rows])
    print(f"{len(rows)} sentences encoded with '{args.encoder}' in {time.time() - t0:.1f}s")

    out = {"encoder": np.array(args.encoder), "n_train": np.array(len(rows))}
    heads = {
        "kind": [i for i in range(len(rows))],
        "object": [i for i, r in enumerate(rows) if r["kind"] in NEEDS_OBJECT],
        "hands": [i for i, r in enumerate(rows) if r["kind"] == "hands_up"],
    }
    for name, idx in heads.items():
        y = [str(rows[i][name]) for i in idx]
        W, b, labels = fit_head(X[idx], y, args.C)
        out[f"{name}_W"], out[f"{name}_b"], out[f"{name}_labels"] = W, b, np.array(labels)
        print(f"  {name:6s} {len(idx):5d} sentences, {len(labels)} labels: {', '.join(labels)}")
    np.savez_compressed(args.out, **out)
    print(f"saved {args.out} ({os.path.getsize(args.out) / 1e6:.1f} MB) in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
