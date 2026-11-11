"""
Search engine: turns raw detections into per-track segments and answers
structured queries (objects / colors / actions / time range / free text).

Everything here is pure Python + numpy, so it is unit-testable without a
model or a video.
"""

import os
from collections import Counter, defaultdict

import numpy as np

from query_parser import EVENT_CLASSES, english_phrase

# Speeds are measured in "body heights per second" (box-center movement
# divided by the box height), so they do not depend on how far the
# object is from the camera. A walking person is ~0.5-1, running ~1.5+.
RUN_SPEED = 1.2
MOVE_SPEED = 0.15   # below this => standing / parked
MAX_GAP = 3.0       # seconds without a detection that split a track
EDGE = 2.0          # % of the frame — a box this close touches the border
NEAR_DIST = 25.0    # % of frame width — "with" / "next to" means closer than this
MIN_SEMANTIC = float(os.environ.get("MIN_SEMANTIC", "0.20"))  # CLIP crop similarity below this => not a match
# words the detector has no class for ("knife", "gun", "package") leave CLIP as the
# only evidence, so the crop must match clearly; measured on UCF-Crime (eval/sweep_crops.py)
MIN_SEMANTIC_UNKNOWN = float(os.environ.get("MIN_SEMANTIC_UNKNOWN", "0.26"))


# ---------------------------------------------------------------- segments
def track_segments(rows, duration=None, W=16, H=9):
    """rows: (t, track_id, cls, conf, x, y, w, h, color) -> list of segments."""
    tracks = defaultdict(list)
    for t, tid, cls, conf, x, y, w, h, color in rows:
        tracks[(tid, cls)].append((t, conf, x, y, w, h, color))

    segments = []
    for (tid, cls), dets in tracks.items():
        dets.sort()
        parts, cur = [], [dets[0]]
        for d in dets[1:]:
            if d[0] - cur[-1][0] > MAX_GAP:
                parts.append(cur)
                cur = []
            cur.append(d)
        parts.append(cur)
        for part in parts:
            seg = make_segment(tid, cls, part, duration, W, H)
            if seg:
                segments.append(seg)
    segments.sort(key=lambda s: s["t"])
    return segments


def make_segment(tid, cls, dets, duration, W, H):
    confs = [d[1] for d in dets]
    if len(dets) == 1 and confs[0] < 0.5:   # a single weak detection is noise
        return None
    t0, t1 = dets[0][0], dets[-1][0]

    steps = []
    for a, b in zip(dets, dets[1:]):
        dt = b[0] - a[0]
        if dt <= 0:
            continue
        dx = (b[2] + b[4] / 2 - a[2] - a[4] / 2) * W / 100
        dy = (b[3] + b[5] / 2 - a[3] - a[5] / 2) * H / 100
        size = max(1e-6, (a[5] + b[5]) / 2 * H / 100)
        steps.append(np.hypot(dx, dy) / size / dt)
    speed = float(np.median(steps)) if steps else 0.0

    def touches_edge(d):
        x, y, w, h = d[2:6]
        return x < EDGE or y < EDGE or x + w > 100 - EDGE or y + h > 100 - EDGE

    entering = touches_edge(dets[0]) and t0 > 1.0 and len(dets) > 1
    exiting = (touches_edge(dets[-1]) and len(dets) > 1 and
               (duration is None or t1 < duration - 1.0))

    colors = Counter(d[6] for d in dets if d[6] != "unknown")
    best = max(dets, key=lambda d: d[1])
    return {
        "t": t0,
        "dur": max(1.0, t1 - t0),
        "conf": float(np.mean(confs)),
        "cls": cls,
        "color": colors.most_common(1)[0][0] if colors else "",
        "speed": speed,
        "entering": entering,
        "exiting": exiting,
        "box": {"x": best[2], "y": best[3], "w": best[4], "h": best[5]},
        "best_t": best[0],
        "path": [[round(d[0], 2), round(d[2], 2), round(d[3], 2),
                  round(d[4], 2), round(d[5], 2)] for d in dets],
        "track_id": tid,
        "entity_id": tid,
    }


# ---------------------------------------------------------- entity merge
def identity(segment):
    """The vector of the track's best crop: the first row when several views are kept."""
    v = segment["embedding_vec"]
    return v if v.ndim == 1 else v[0]


def merge_entities(segments, threshold=0.88):
    """
    ByteTrack ids break whenever an object is occluded for a while, so the
    same person comes back as a new track. Segments of the same class whose
    CLIP embeddings are very similar are given the same entity_id — but only
    when no member of one entity is on screen at the same time as a member
    of the other (two people visible together can never be one entity).
    Segments without embeddings are left alone.
    """
    with_emb = [s for s in segments if s.get("embedding_vec") is not None]
    by_cls = defaultdict(list)
    for s in with_emb:
        by_cls[s["cls"]].append(s)

    for group in by_cls.values():
        group.sort(key=lambda s: s["t"])
        members = defaultdict(list)
        for s in group:
            members[s["entity_id"]].append(s)
        for i, a in enumerate(group):
            for b in group[i + 1:]:
                ea, eb = a["entity_id"], b["entity_id"]
                if ea == eb:
                    continue
                if float(np.dot(identity(a), identity(b))) < threshold:
                    continue
                if any(overlaps(x, y) for x in members[ea] for y in members[eb]):
                    continue
                for s in members.pop(eb):
                    s["entity_id"] = ea
                    members[ea].append(s)
    return segments


# -------------------------------------------------------------- filtering
def action_ok(s, actions):
    checks = {
        "running": s["speed"] >= RUN_SPEED,
        "walking": MOVE_SPEED <= s["speed"] < RUN_SPEED,
        "moving": s["speed"] >= MOVE_SPEED,
        "parked": s["speed"] < MOVE_SPEED,
        "entering": s["entering"],
        "exiting": s["exiting"],
    }
    return all(checks.get(a, True) for a in actions)


def box_at(path, t):
    """Interpolated box (x, y, w, h) of a segment at time t, or None."""
    if not path:
        return None
    if t < path[0][0] or t > path[-1][0]:
        return None
    for a, b in zip(path, path[1:]):
        if a[0] <= t <= b[0]:
            if b[0] == a[0]:
                return a[1:]
            k = (t - a[0]) / (b[0] - a[0])
            return [a[i] + (b[i] - a[i]) * k for i in range(1, 5)]
    return path[-1][1:]


def overlaps(a, b):
    return a["t"] <= b["t"] + b["dur"] and b["t"] <= a["t"] + a["dur"]


def near(a, b, max_dist=NEAR_DIST, W=16, H=9):
    """
    True when the two segments are close in SPACE while they overlap in time.
    Distance = box-center distance in % of the frame width (y scaled by the
    aspect ratio), minimum over the common time samples.
    """
    if not overlaps(a, b):
        return False
    lo, hi = max(a["t"], b["t"]), min(a["t"] + a["dur"], b["t"] + b["dur"])
    ts = [p[0] for p in a["path"] if lo <= p[0] <= hi] or [lo]
    best = float("inf")
    for t in ts:
        pa, pb = box_at(a["path"], t), box_at(b["path"], t)
        if pa is None or pb is None:
            continue
        dx = (pa[0] + pa[2] / 2) - (pb[0] + pb[2] / 2)
        dy = ((pa[1] + pa[3] / 2) - (pb[1] + pb[3] / 2)) * H / W
        # distance between the boxes' edges, not their centers, so a big
        # object (bus) next to a small one (person) still counts as "near"
        gap = np.hypot(dx, dy) - (pa[2] + pb[2]) / 4
        best = min(best, gap)
    return best <= max_dist


def filter_segments(segments, objects, colors, actions, time_range,
                    require_near=True, W=16, H=9):
    """
    objects   : list of class names, first one is the subject
    colors    : apply to the subject (or to everything when no objects)
    actions   : same
    time_range: [lo, hi] in seconds or None
    """
    main = objects[0] if objects else None

    def attrs_ok(s):
        if main and s["cls"] != main:
            return True
        if colors and s["color"] not in colors:
            return False
        return action_ok(s, actions)

    segments = [s for s in segments if attrs_ok(s)]
    if time_range:
        lo, hi = time_range
        hi = float("inf") if hi is None else hi
        segments = [s for s in segments
                    if s["t"] <= hi and s["t"] + s["dur"] >= lo]

    # multi-object query ("person with a dog"): keep the segments that meet a
    # segment of every other requested class — in time AND in space
    if len(objects) > 1:
        by_cls = defaultdict(list)
        for s in segments:
            by_cls[s["cls"]].append(s)
        together = near if require_near else (lambda a, b, **k: overlaps(a, b))
        segments = [s for s in segments
                    if all(any(together(s, o, W=W, H=H) if require_near
                               else overlaps(s, o) for o in by_cls[c])
                           for c in objects if c != s["cls"])]
    return segments


# ------------------------------------------------------------ scene search
# Whole-frame CLIP. A raw cosine similarity means little by itself (every CCTV
# frame is ~0.2 away from every sentence), so a frame is scored the zero-shot
# way: a softmax over the query, phrased a few ways, against prompts for
# "nothing special is going on" (measured with eval/sweep_scene.py).
SCENE_TEMPLATES = ("{}", "a photo of {}", "security camera footage of {}", "a cctv image of {}")
SCENE_NEGATIVES = ("a photo", "a security camera image", "an empty room", "an empty street",
                   "people walking", "people standing", "a person", "a shop", "a parking lot",
                   "a road with cars", "an office", "a street at night", "people talking")
CLIP_T = 100.0      # CLIP's logit scale
MIN_SCENE = float(os.environ.get("MIN_SCENE", "0.2"))  # P(query | frame) below this => no scene
SCENE_Z = float(os.environ.get("SCENE_Z", "1.5"))      # ...and it must stand out: mean + z*std of the video
SCENE_GAP = 1.5     # s between hits that still belong to one window
MIN_CROP_NO_OBJECT = 0.25   # crop similarity needed when the question names no object class


def scene_prompts(text):
    """-> (prompts for the query, prompts for 'nothing special')."""
    return [t.format(text) for t in SCENE_TEMPLATES], list(SCENE_NEGATIVES)


def match_prob(vecs, pos, neg):
    """
    vecs (n, D) image vectors, pos (k, D) query prompts, neg (m, D) negative
    prompts -> (n,) probability of the query in a softmax against the negatives.
    """
    pos, neg = np.atleast_2d(pos), np.atleast_2d(neg)
    vecs = np.asarray(vecs, dtype=np.float32).reshape(-1, pos.shape[-1])
    q = CLIP_T * (vecs @ pos.T).mean(axis=1)
    n = CLIP_T * (vecs @ neg.T)
    top = np.maximum(q, n.max(axis=1))                 # keeps exp() in range
    eq = np.exp(q - top)
    return eq / (eq + np.exp(n - top[:, None]).sum(axis=1))


def scene_windows(frames, pos, neg, min_score=MIN_SCENE, z=SCENE_Z, gap=SCENE_GAP):
    """
    frames: [(t, vec), ...] whole-frame CLIP vectors in time order; a time may
    come with several vectors (views of the same frame), the best one counts.
    Frames whose probability of the query (match_prob) stands out (above
    min_score AND above the video's mean + z*std, so a video that is all fire
    still ranks its strongest seconds) are grouped into time windows. Returned
    as segment-like dicts (cls "scene") so the rest of the pipeline can use them.
    """
    if not frames:
        return []
    probs = match_prob(np.stack([v for _, v in frames]), pos, neg)
    ts, inv = np.unique(np.array([t for t, _ in frames], dtype=np.float32), return_inverse=True)
    sims = np.zeros(len(ts), dtype=np.float32)
    np.maximum.at(sims, inv, probs)
    thr = max(min_score, float(sims.mean() + z * sims.std())) if len(sims) > 3 else min_score
    hit = sims >= thr
    out = []
    i = 0
    while i < len(ts):
        if not hit[i]:
            i += 1
            continue
        j = i
        while j + 1 < len(ts) and hit[j + 1] and ts[j + 1] - ts[j] <= gap:
            j += 1
        k = i + int(np.argmax(sims[i:j + 1]))
        out.append({
            "t": float(ts[i]), "dur": max(1.0, float(ts[j] - ts[i]) + 1.0),
            "conf": float(sims[k]), "score": float(sims[k]), "cls": "scene", "color": "",
            "speed": 0.0, "entering": False, "exiting": False,
            "box": {"x": 0.0, "y": 0.0, "w": 100.0, "h": 100.0}, "best_t": float(ts[k]),
            "path": [], "track_id": -1, "entity_id": -1,
        })
        i = j + 1
    out.sort(key=lambda s: -s["score"])
    return out


def semantic_floor(unknown_words) -> float:
    """The crop-similarity floor for a query: stricter when it names things no detector class covers."""
    return MIN_SEMANTIC_UNKNOWN if unknown_words else MIN_SEMANTIC


def crop_score(vec, text_vec) -> float:
    """Best cosine similarity between the text and the views kept of a track ((D,) or (k, D))."""
    return float((np.asarray(vec).reshape(-1, len(text_vec)) @ text_vec).max())


def semantic_rank(segments, text_vec, min_score=MIN_SEMANTIC):
    """
    Score segments by cosine similarity between their CLIP crop embedding(s)
    and the query text embedding. Segments without an embedding are kept
    with score 0 so a missing model never hides results.
    """
    out = []
    for s in segments:
        v = s.get("embedding_vec")
        score = crop_score(v, text_vec) if v is not None else 0.0
        if v is not None and score < min_score:
            continue
        s = dict(s, score=score)
        out.append(s)
    out.sort(key=lambda s: -s["score"])
    return out


# ------------------------------------------------------------------ query
def run_query(segments, frames, objects=(), colors=(), actions=(), time_range=None,
              semantic="", unknown=(), embed=None, near=True, W=16, H=9):
    """
    The whole answer to one structured query, without a server or a model:

    segments : every segment of the video ("embedding_vec" set or None)
    frames   : [(t, vec), ...] whole-frame CLIP vectors
    embed    : function [texts] -> (n, D) matrix or None (CLIP's text tower)

    -> (results, semantic_used). Three kinds of evidence, in this order:
      1. the detector: tracks of the classes asked for (colour, action, time,
         "with" = near each other)
      2. CLIP on the tracks' crops, when the question says more than classes
      3. CLIP on whole frames ("scenes"), when the tracks do not answer it
    For event classes (fire, gun, knife) the detector's tracks are kept as
    they are and CLIP adds the moments the detector missed: on UCF-Crime the
    fire detector alone finds 14 of 44 fire windows, CLIP alone 17, both 19.
    """
    objects, unknown = list(objects), list(unknown)
    lo = (time_range[0] or 0) if time_range else 0
    hi = time_range[1] if time_range and time_range[1] is not None else float("inf")

    def in_range(s):
        return s["t"] <= hi and s["t"] + s["dur"] >= lo

    def detector(objs):
        cand = [s for s in segments if not objs or s["cls"] in objs]
        return filter_segments(cand, objs, colors, actions, time_range, require_near=near, W=W, H=H)

    events = [o for o in objects if o in EVENT_CLASSES]
    hits = detector(objects)
    text = semantic or (english_phrase(objects, colors, actions) if events else "")
    vecs = None
    if text and embed is not None:
        pos, neg = scene_prompts(text)
        vecs = embed([text] + pos + neg)
    if vecs is None:
        return sorted(hits, key=lambda s: s["t"]), False
    text_vec, pos, neg = vecs[0], vecs[1:1 + len(pos)], vecs[1 + len(pos):]

    def clip(cand, objs, unknown_words):
        ranked = semantic_rank(cand, text_vec, min_score=semantic_floor(unknown_words))
        scenes = [s for s in scene_windows(frames, pos, neg) if in_range(s)]
        if not objs:        # "fire", "someone grabbing a bag": scenes lead, crops must match clearly
            return scenes + [s for s in ranked if s["score"] >= MIN_CROP_NO_OBJECT]
        return ranked or scenes

    if not events:
        return clip(hits, objects, unknown), True
    found = sorted((dict(s, score=s["conf"]) for s in hits), key=lambda s: -s["score"])
    rest = [o for o in objects if o not in events]
    more = clip(detector(rest), rest, unknown + events)
    return found + [s for s in more if not any(overlaps(s, f) for f in found)], True
