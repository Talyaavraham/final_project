"""
Sentence -> feature vector, for our own alert-request model.

Two encoders, same interface (encode(texts) -> rows to score):

* "hash"  (default): character n-grams (2-5) and words, hashed into D buckets,
          log term frequency, L2-normalised. No vocabulary file, no download,
          runs on any CPU in microseconds. Hebrew words are also added with
          their one-letter prefixes stripped (ו/ה/ש/ב/ל/מ/כ, כש), so "כשמישהו"
          shares features with "מישהו".
* "minilm": a multilingual sentence encoder (Hebrew + English) from Hugging
          Face, mean-pooled. Better with unseen words, needs `transformers` and
          a one-time ~470 MB download.

Kept free of scikit-learn so the server only needs numpy at run time.
"""
import re
import zlib

import numpy as np

HASH_DIM = 1 << 16
HE_PREFIXES = ("כש", "וכש", "ש", "ה", "ו", "ב", "ל", "מ", "כ")
MINILM = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"


def normalise(text: str) -> str:
    text = text.lower().replace("״", '"').replace("׳", "'")
    text = re.sub(r"\d+(\.\d+)?", " 0 ", text)            # numbers are read by regex, not learnt
    text = re.sub(r"[^\w֐-׿' ]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _strip_prefix(w):
    for p in HE_PREFIXES:
        if w.startswith(p) and len(w) - len(p) >= 2 and re.match(r"[֐-׿]", w):
            return w[len(p):]
    return None


def tokens(text: str):
    """the strings that become features: words, prefix-stripped Hebrew words,
    word pairs and character n-grams."""
    t = normalise(text)
    words = t.split()
    out = ["w:" + w for w in words]
    out += ["s:" + s for s in (_strip_prefix(w) for w in words) if s]
    out += ["b:" + a + "_" + b for a, b in zip(words, words[1:])]
    padded = " " + t + " "
    for n in (2, 3, 4, 5):
        out += ["c:" + padded[i:i + n] for i in range(len(padded) - n + 1)]
    return out


def hash_features(text: str, dim: int = HASH_DIM):
    """-> (indices, values): one sparse, L2-normalised row."""
    counts = {}
    for tok in tokens(text):
        h = zlib.crc32(tok.encode("utf-8")) % dim
        counts[h] = counts.get(h, 0) + 1
    if not counts:
        return np.zeros(0, np.int64), np.zeros(0, np.float32)
    idx = np.fromiter(counts.keys(), np.int64)
    val = np.log1p(np.fromiter(counts.values(), np.float32))
    val /= np.linalg.norm(val)
    return idx, val


class HashEncoder:
    name = "hash"

    def __init__(self, dim: int = HASH_DIM):
        self.dim = dim

    def sparse_matrix(self, texts):
        """scipy CSR matrix for training (scipy comes with scikit-learn)."""
        from scipy.sparse import csr_matrix
        rows, cols, vals = [], [], []
        for i, t in enumerate(texts):
            idx, val = hash_features(t, self.dim)
            rows += [i] * len(idx)
            cols += idx.tolist()
            vals += val.tolist()
        return csr_matrix((vals, (rows, cols)), shape=(len(texts), self.dim), dtype=np.float32)

    def scores(self, text, W, b):
        """W: (dim, classes), b: (classes,) -> scores of one sentence."""
        idx, val = hash_features(text, self.dim)
        return val @ W[idx].astype(np.float32) + b


class MiniLMEncoder:
    name = "minilm"

    def __init__(self, model_name: str = MINILM):
        from transformers import AutoModel, AutoTokenizer
        self.tok = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).eval()

    def embed(self, texts):
        import torch
        out = []
        with torch.no_grad():
            for i in range(0, len(texts), 64):
                enc = self.tok(texts[i:i + 64], padding=True, truncation=True, max_length=64, return_tensors="pt")
                h = self.model(**enc).last_hidden_state
                m = enc["attention_mask"].unsqueeze(-1).float()
                e = (h * m).sum(1) / m.sum(1).clamp(min=1e-6)
                out.append(torch.nn.functional.normalize(e, dim=1).numpy())
        return np.concatenate(out).astype(np.float32)

    def sparse_matrix(self, texts):      # dense, but the same name keeps train.py simple
        return self.embed(texts)

    def scores(self, text, W, b):
        return self.embed([text])[0] @ W.astype(np.float32) + b


def make_encoder(name: str):
    if name == "hash":
        return HashEncoder()
    if name == "minilm":
        return MiniLMEncoder()
    raise ValueError(f"unknown encoder {name!r} (hash / minilm)")
