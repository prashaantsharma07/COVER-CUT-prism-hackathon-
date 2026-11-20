"""Turns raw head outputs into a JSON that always satisfies the README rules.

Every head can be sloppy; this module is the single place that guarantees
validity: vocabulary, score ranges, segment bounds, sorting, non-overlap,
the 4 s minimum and the 15% budget.
"""
import json
import math
import os

from . import config as C

_HERE = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(_HERE, "supported_classes.json")) as f:
    _CLASSES = json.load(f)
LABELS = set(_CLASSES["labels"])
VERBS = set(_CLASSES["verbs"])
NOUNS = set(_CLASSES["nouns"])


def _r(x):
    return round(float(x), C.DECIMALS)


def _clip01(x):
    x = float(x)
    if not math.isfinite(x):
        return 0.0
    return min(1.0, max(0.0, x))


def clean_label(name, score):
    name = str(name).lower()
    if name not in LABELS:
        name = "other"
    return {"name": name, "score": _r(_clip01(score))}


def clean_actions(actions, duration):
    out = []
    hi = duration + 0.5
    for a in actions:
        s, e = float(a["segment"][0]), float(a["segment"][1])
        if not (math.isfinite(s) and math.isfinite(e)):
            continue
        s, e = max(0.0, s), min(hi, e)
        s, e = _r(s), _r(e)
        if e > hi:
            e = math.floor(hi * 100) / 100
        if not e > s:
            continue
        verb = a.get("verb", "other")
        noun = a.get("noun", "other")
        out.append({
            "segment": [s, e],
            "verb": verb if verb in VERBS else "other",
            "noun": noun if noun in NOUNS else "other",
            "score": _r(_clip01(a.get("score", 0.0))),
        })
    out.sort(key=lambda a: -a["score"])
    return out[:C.MAX_ACTIONS]


def _merge(segs):
    segs = sorted(segs)
    merged = []
    for s, e in segs:
        if merged and s <= merged[-1][1] + 0.01:   # overlapping or touching -> merge
            merged[-1][1] = max(merged[-1][1], e)
        else:
            merged.append([s, e])
    return merged


def clean_highlights(segments, duration, priority=None):
    """segments: list of (start, end). priority: optional same-length scores;
    higher priority survives budget trimming. Returns a valid highlight list."""
    budget = C.TARGET_FRAC * duration
    pr = list(priority) if priority is not None else [0.0] * len(segments)
    cands = []
    for (s, e), p in zip(segments, pr):
        s, e = max(0.0, float(s)), min(duration, float(e))
        if e - s >= C.MIN_SEG:
            cands.append((p, s, e))
    # keep highest-priority segments until the budget is used
    cands.sort(key=lambda t: -t[0])
    chosen, used = [], 0.0
    for p, s, e in cands:
        L = e - s
        if used + L <= budget:
            chosen.append([s, e]); used += L
        elif budget - used >= C.SAFE_MIN_SEG:      # shrink around the centre
            mid, half = (s + e) / 2, (budget - used) / 2
            chosen.append([mid - half, mid + half]); used = budget
    chosen = _merge(chosen)
    # round inward so no segment drops below 4.0 s after rounding
    out = []
    for s, e in chosen:
        s2, e2 = math.ceil(s * 100) / 100, math.floor(e * 100) / 100
        if e2 - s2 >= C.MIN_SEG:
            out.append([s2, e2])
    if not out:
        out = [_fallback_segment(duration, segments, pr)]
    return [{"segment": [_r(s), _r(e)]} for s, e in out]


def _fallback_segment(duration, segments, pr):
    """One 4.05 s segment; used for very short videos where 15% < 4 s."""
    L = C.SAFE_MIN_SEG
    centre = duration / 2
    if segments:
        best = max(range(len(segments)), key=lambda i: pr[i])
        centre = (float(segments[best][0]) + float(segments[best][1])) / 2
    s = min(max(0.0, centre - L / 2), max(0.0, duration + 0.5 - L))
    s = math.ceil(s * 100) / 100
    e = math.floor(min(s + L, duration + 0.5) * 100) / 100
    return [s, e]


def build(video_id, duration, label, actions, highlights, highlight_priority=None):
    return {
        "team_id": C.TEAM_ID,
        "video_id": video_id,
        "duration_sec": _r(duration),
        "label": clean_label(*label),
        "actions": clean_actions(actions, duration),
        "highlights": clean_highlights(highlights, duration, highlight_priority),
    }


def write(path, obj):
    d = os.path.dirname(os.path.abspath(path))
    os.makedirs(d, exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=1)
    os.replace(tmp, path)
