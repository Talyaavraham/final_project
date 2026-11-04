"""
Search engine: turns raw detections into per-track segments and answers
structured queries (objects / colors / actions / time range / free text).

Everything here is pure Python + numpy, so it is unit-testable without a
model or a video.
"""

import os
from collections import Counter, defaultdict

import numpy as np

# Speeds are measured in "body heights per second" (box-center movement
# divided by the box height), so they do not depend on how far the
# object is from the camera. A walking person is ~0.5-1, running ~1.5+.
RUN_SPEED = 1.2
MOVE_SPEED = 0.15   # below this => standing / parked
MAX_GAP = 3.0       # seconds without a detection that split a track
EDGE = 2.0          # % of the frame — a box this close touches the border
NEAR_DIST = 25.0    # % of frame width — "with" / "next to" means closer than this
MIN_SEMANTIC = 0.20 # CLIP cosine similarity below this => not a match


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
                if float(np.dot(a["embedding_vec"], b["embedding_vec"])) < threshold:
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


MIN_SCENE = float(os.environ.get("MIN_SCENE", "0.26"))  # whole-frame CLIP similarity below this => no scene
SCENE_Z = float(os.environ.get("SCENE_Z", "1.5"))      # ...and it must stand out: mean + z*std of the video
SCENE_GAP = 1.5     # s between hits that still belong to one window


def scene_windows(frames, text_vec, min_score=MIN_SCENE, z=SCENE_Z, gap=SCENE_GAP):
    """
    frames: [(t, vec), ...] whole-frame CLIP vectors in time order.
    Frames whose similarity to the query stands out (above min_score AND
    above the video's mean + z*std, so a video that is all fire still ranks
    its strongest seconds) are grouped into time windows. Returned as
    segment-like dicts (cls "scene") so the rest of the pipeline can use them.
    """
    if not frames:
        return []
    ts = np.array([t for t, _ in frames], dtype=np.float32)
    sims = np.array([float(np.dot(v, text_vec)) for _, v in frames], dtype=np.float32)
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


def semantic_rank(segments, text_vec, min_score=MIN_SEMANTIC):
    """
    Score segments by cosine similarity between their CLIP crop embedding
    and the query text embedding. Segments without an embedding are kept
    with score 0 so a missing model never hides results.
    """
    out = []
    for s in segments:
        v = s.get("embedding_vec")
        score = float(np.dot(v, text_vec)) if v is not None else 0.0
        if v is not None and score < min_score:
            continue
        s = dict(s, score=score)
        out.append(s)
    out.sort(key=lambda s: -s["score"])
    return out
