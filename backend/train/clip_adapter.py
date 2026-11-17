"""
Train a text-side adapter for CLIP on UCA: the UCF-Crime sentences come with
the time window they describe, and the benchmark index already holds a CLIP
vector for every second of every video. The adapter learns to move a sentence
nearer the frames it describes and away from the rest of the video, without
touching the image side, so nothing has to be re-indexed:

    t' = normalise(t + t @ W + b)          W, b start at 0 (= plain CLIP)

Loss, per sentence of a training video:
  * contrast: softmax over the frames of its video (+ frames of other videos),
    the frames inside the window must win (where in the video?)
  * scene: P(sentence) against search.SCENE_NEGATIVES, the softmax
    search.scene_windows actually thresholds: high inside the window, low on
    the rest of the video (is it there at all?)
  * keep: generic prompts ("a red car", "a person") stay where CLIP put them,
    so the adapter does not break the queries it was not trained on.

Videos are split by UCA's own Train / Val / Test lists: train on Train, pick
the epoch on Val, report on Test (eval/offline_eval.py --split Test).

    python train/clip_adapter.py eval/benchmark.db                  -> clip_adapter.npz
    python train/clip_adapter.py eval/benchmark.db --epochs 5       # quick test
"""
import argparse
import json
import os
import random
import sqlite3
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
BACKEND = os.path.dirname(HERE)
sys.path.insert(0, BACKEND)
import db                     # noqa: E402
import embeddings             # noqa: E402
import search                 # noqa: E402

UCA_DIR = os.path.join(BACKEND, "eval", "data", "uca_src")
UCA_ALL = os.path.join(BACKEND, "eval", "data", "uca_all.json")
OUT = os.path.join(BACKEND, "clip_adapter.npz")

COCO = ("person", "bicycle", "car", "motorcycle", "bus", "truck", "traffic light", "bench", "dog",
        "cat", "backpack", "umbrella", "handbag", "suitcase", "bottle", "cup", "chair", "couch",
        "bed", "dining table", "tv", "laptop", "cell phone", "book", "knife", "scissors", "horse")
COLORS = ("red", "blue", "green", "yellow", "black", "white", "gray", "orange")
GENERIC = ([f"a {c}" for c in COCO] + [f"a {col} {c}" for col in COLORS for c in ("car", "shirt", "bag")]
           + ["a person running", "a person walking", "a woman pushing a stroller", "a man with a dog",
              "fire", "smoke", "a person holding a gun", "a person holding a knife", "a crowd of people",
              "a car parked on the street", "a person riding a bicycle", "people fighting"])


def load_splits(uca_dir):
    split = {}
    for s in ("Train", "Val", "Test"):
        p = os.path.join(uca_dir, f"UCFCrime_{s}.json")
        if os.path.exists(p):
            for name in json.load(open(p, encoding="utf-8")):
                split[name] = s
    if not split:
        sys.exit(f"no UCA split files in {uca_dir} (UCFCrime_Train/Val/Test.json)")
    return split


def load_videos(conn, uca, names):
    """-> {name: {"t": (n,), "v": (n, D), "sents": [(text, lo, hi), ...]}} for indexed videos."""
    out = {}
    for name in names:
        rows = db.load_frames(conn, name + ".mp4")
        if not rows or name not in uca:
            continue
        t = np.array([r[0] for r in rows], dtype=np.float32)
        v = np.stack([embeddings.from_blob(r[1]) for r in rows]).astype(np.float32)
        sents = [(s.strip(), float(a), float(b)) for (a, b), s in zip(uca[name]["timestamps"], uca[name]["sentences"])
                 if s.strip() and b > a]
        if sents:
            out[name] = {"t": t, "v": v, "sents": sents}
    return out


def embed_all(texts, bs=256):
    vecs = [embeddings.embed_texts(texts[i:i + bs], raw=True) for i in range(0, len(texts), bs)]
    return np.concatenate(vecs).astype(np.float32)


def build_items(videos, text_vec):
    """one item per sentence: (video name, text vector, positive-frame mask)."""
    items = []
    for name, d in videos.items():
        for text, lo, hi in d["sents"]:
            pos = (d["t"] >= lo - 0.5) & (d["t"] <= hi + 0.5)
            if pos.any():
                items.append((name, text_vec[text], pos))
    return items


def window_hit(videos, items, W, b, neg_vec):
    """Val metric: share of sentences whose best-scoring second (scene softmax, as search does)
    falls inside their window, and the mean P(sentence) inside vs outside the window."""
    hits, p_in, p_out = 0, [], []
    for name, t, pos in items:
        q = t + t @ W + b
        q /= np.linalg.norm(q)
        n = neg_vec + neg_vec @ W + b
        n /= np.linalg.norm(n, axis=1, keepdims=True)
        p = search.match_prob(videos[name]["v"], q[None], n)
        hits += bool(pos[int(np.argmax(p))])
        p_in.append(p[pos].mean())
        if (~pos).any():
            p_out.append(p[~pos].mean())
    return hits / max(1, len(items)), float(np.mean(p_in)), float(np.mean(p_out)) if p_out else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("db", help="benchmark index (eval/benchmark.db)")
    ap.add_argument("--uca-dir", default=UCA_DIR)
    ap.add_argument("--uca-all", default=UCA_ALL)
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--batch", type=int, default=32, help="sentences per step")
    ap.add_argument("--other", type=int, default=256, help="frames of other videos added as negatives")
    ap.add_argument("--scene-w", type=float, default=1.0, help="weight of the scene-softmax loss")
    ap.add_argument("--keep-w", type=float, default=10.0, help="weight of the keep-generic-prompts loss")
    ap.add_argument("--rank", type=int, default=16,
                    help="W = A @ B of this rank: a few thousand sentences cannot fit a full 512x512 matrix")
    ap.add_argument("--wd", type=float, default=1e-4, help="weight decay")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    import torch
    torch.manual_seed(args.seed)
    rnd = random.Random(args.seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    split = load_splits(args.uca_dir)
    uca = json.load(open(args.uca_all, encoding="utf-8"))
    conn = sqlite3.connect(args.db)
    indexed = [r[0][:-4] for r in conn.execute("SELECT video FROM videos") if r[0].endswith(".mp4")]
    videos = load_videos(conn, uca, indexed)
    by_split = {s: {n: d for n, d in videos.items() if split.get(n) == s} for s in ("Train", "Val", "Test")}
    print("indexed videos with UCA sentences:", {s: len(v) for s, v in by_split.items()},
          f"(not in a split: {len(videos) - sum(len(v) for v in by_split.values())})")
    if not by_split["Train"] or not by_split["Val"]:
        sys.exit("need indexed videos from both the Train and the Val split")

    texts = sorted({s for d in videos.values() for s, _, _ in d["sents"]})
    t0 = time.time()
    text_vec = dict(zip(texts, embed_all(texts)))
    neg_np = embed_all(list(search.SCENE_NEGATIVES))
    gen_np = embed_all(GENERIC)
    print(f"{len(texts)} sentences embedded in {time.time() - t0:.0f}s")

    train = build_items(by_split["Train"], text_vec)
    val = build_items(by_split["Val"], text_vec)
    test = build_items(by_split["Test"], text_vec)
    print(f"sentences: train {len(train)}, val {len(val)}, test {len(test)}")

    D = neg_np.shape[1]
    frames_t = {n: torch.from_numpy(d["v"]).to(dev) for n, d in by_split["Train"].items()}
    names = list(frames_t)
    neg = torch.from_numpy(neg_np).to(dev)
    gen = torch.from_numpy(gen_np).to(dev)
    # low rank, B = 0 at the start: the adapter starts as plain CLIP and A still gets gradients
    A = (torch.randn(D, args.rank, device=dev) / D ** 0.5).requires_grad_()
    B = torch.zeros(args.rank, D, device=dev, requires_grad=True)
    b = torch.zeros(D, device=dev, requires_grad=True)
    opt = torch.optim.AdamW([A, B, b], lr=args.lr, weight_decay=args.wd)
    T = search.CLIP_T

    def f(x):
        return torch.nn.functional.normalize(x + (x @ A) @ B + b, dim=-1)

    zero = np.zeros((D, D), np.float32), np.zeros(D, np.float32)
    base = window_hit(by_split["Val"], val, *zero, neg_np)
    print(f"plain CLIP   val: best second inside the window {base[0]:.3f}, "
          f"P(sentence) inside {base[1]:.3f} / outside {base[2]:.3f}")
    best = (base[0], 0, zero)
    for ep in range(1, args.epochs + 1):
        rnd.shuffle(train)
        tot = 0.0
        for i in range(0, len(train), args.batch):
            loss = 0.0
            for name, t, pos in train[i:i + args.batch]:
                q = f(torch.from_numpy(t).to(dev))
                own = frames_t[name]
                other = [frames_t[n] for n in rnd.sample(names, min(8, len(names))) if n != name]
                pool = torch.cat([own] + other)[: own.shape[0] + args.other]
                mask = torch.zeros(pool.shape[0], dtype=torch.bool, device=dev)
                mask[: own.shape[0]] = torch.from_numpy(pos).to(dev)
                s = T * pool @ q
                contrast = torch.logsumexp(s, 0) - torch.logsumexp(s[mask], 0)
                # scene softmax P(sentence) vs "nothing special", as search thresholds it: high on
                # the window, low on the rest of the video (one-sided, it learns to always say yes)
                logits = T * torch.cat([(own @ q)[:, None], own @ f(neg).T], dim=1)
                logp = torch.log_softmax(logits, dim=1)[:, 0]
                pos_t = torch.from_numpy(pos).to(dev)
                scene = -logp[pos_t].mean()
                if (~pos_t).any():
                    scene = scene - torch.log1p(-logp[~pos_t].exp().clamp(max=1 - 1e-6)).mean()
                loss = loss + contrast + args.scene_w * scene
            n_b = len(train[i:i + args.batch])
            keep = ((f(gen) - gen) ** 2).sum(1).mean()
            loss = loss / n_b + args.keep_w * keep
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += loss.item() * n_b
        with torch.no_grad():
            Wn, bn = (A @ B).cpu().numpy(), b.cpu().numpy()
            drift = float(((f(gen) - gen) ** 2).sum(1).sqrt().mean())
        v = window_hit(by_split["Val"], val, Wn, bn, neg_np)
        print(f"epoch {ep:3d}  loss {tot / len(train):.3f}  val: inside {v[0]:.3f}, "
              f"P in {v[1]:.3f} / out {v[2]:.3f}  generic drift {drift:.3f}", flush=True)
        if v[0] > best[0]:
            best = (v[0], ep, (Wn.copy(), bn.copy()))

    acc, ep, (Wb, bb) = best
    if ep == 0:
        print("no epoch beat plain CLIP on Val: nothing saved")
        return
    np.savez(args.out, W=Wb, b=bb, epoch=ep, val=acc)
    print(f"saved epoch {ep} (val {acc:.3f} vs plain CLIP {base[0]:.3f}) -> {args.out}")
    if test:
        tb = window_hit(by_split["Test"], test, *zero, neg_np)
        ta = window_hit(by_split["Test"], test, Wb, bb, neg_np)
        print(f"Test: best second inside the window  plain CLIP {tb[0]:.3f} -> adapter {ta[0]:.3f}")


if __name__ == "__main__":
    main()
