"""
Run time: load nlu/model.npz (made by train.py) and predict, for one sentence,
the alert kind, the object and the number of hands, each with a probability.
Only numpy is needed (plus transformers for an "minilm" model).
"""
import os

import numpy as np

from .features import make_encoder

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_PATH = os.environ.get("NLU_MODEL", os.path.join(HERE, "model.npz"))
HEADS = ("kind", "object", "hands")


def softmax(z):
    z = z - z.max()
    e = np.exp(z)
    return e / e.sum()


class NluModel:
    def __init__(self, path=DEFAULT_PATH):
        data = np.load(path, allow_pickle=False)
        self.path = path
        self.encoder_name = str(data["encoder"])
        self.encoder = make_encoder(self.encoder_name)
        self.heads = {}
        for h in HEADS:
            if f"{h}_W" in data:
                self.heads[h] = (data[f"{h}_W"], data[f"{h}_b"].astype(np.float32), [str(x) for x in data[f"{h}_labels"]])
        self.trained_on = int(data["n_train"]) if "n_train" in data else None

    def head(self, name, text):
        W, b, labels = self.heads[name]
        p = softmax(self.encoder.scores(text, W, b))
        i = int(p.argmax())
        return labels[i], float(p[i])

    def predict(self, text):
        out = {}
        for h in self.heads:
            out[h], out[h + "_p"] = self.head(h, text)
        out["hands"] = int(out.get("hands") or 0)
        return out


_cache = {}


def load(path=DEFAULT_PATH):
    """the model, or None when there is no trained model file (or it cannot load)."""
    if path not in _cache:
        try:
            _cache[path] = NluModel(path) if os.path.isfile(path) else None
        except Exception as e:  # noqa: BLE001  (e.g. minilm model without transformers)
            print(f"[nlu] could not load {path}: {e}", flush=True)
            _cache[path] = None
    return _cache[path]
