"""Budgeted maximum-coverage highlight selection (the EPIC highlight head, 40% of the score).

Given detections (start, end, pair, prob), choose non-overlapping windows (>= 4 s,
total <= budget) maximising the expected number of DISTINCT pairs captured:
    F(S) = sum_a [1 - prod_{w in S} (1 - q(a, w))]
F is monotone submodular, so greedy by marginal gain per second is near-optimal.
"""
import numpy as np

from . import config as C


def captured(ws, we, ds, de, rule):
    """Boolean matrix [windows x detections]: does window capture detection under `rule`.
    rule: 'any' (any overlap), 'half' (>= 50% of the action inside), 'full' (fully inside)."""
    ws, we = ws[:, None], we[:, None]
    ds, de = ds[None, :], de[None, :]
    inter = np.clip(np.minimum(we, de) - np.maximum(ws, ds), 0, None)
    length = np.maximum(de - ds, 1e-3)
    if rule == "any":
        return (np.minimum(we, de) >= np.maximum(ws, ds))
    if rule == "half":
        return inter >= 0.5 * length
    if rule == "full":
        return inter >= length - 1e-6
    raise ValueError(rule)


def candidate_windows(det_s, det_e, duration, mode, win_len, max_len):
    mid = (det_s + det_e) / 2
    if mode == "center":
        L = np.full_like(mid, win_len)
    elif mode == "span":
        L = np.clip(det_e - det_s + 1.0, win_len, max_len)
    else:
        raise ValueError(mode)
    s = np.clip(mid - L / 2, 0, np.maximum(0.0, duration - L))
    e = np.minimum(s + L, duration)
    keep = (e - s) >= C.SAFE_MIN_SEG - 1e-6
    return s[keep], e[keep]


def select(detections, duration, rule="half", mode="span", win_len=C.SAFE_MIN_SEG,
           max_len=12.0, budget_frac=C.TARGET_FRAC, per_second=True):
    """detections: list of dicts {segment:[s,e], verb, noun, score}. Returns (windows, priorities)."""
    if not detections:
        return [], None
    ds = np.array([d["segment"][0] for d in detections], float)
    de = np.array([d["segment"][1] for d in detections], float)
    p = np.clip(np.array([d["score"] for d in detections], float), 0, 1)
    pairs = [(d["verb"], d["noun"]) for d in detections]
    uniq = {a: i for i, a in enumerate(dict.fromkeys(pairs))}
    pid = np.array([uniq[a] for a in pairs])

    ws, we = candidate_windows(ds, de, duration, mode, win_len, max_len)
    if len(ws) == 0:
        return [], None
    cap = captured(ws, we, ds, de, rule)                     # [W x D]
    # q[w, a] = max prob of any detection of pair a captured by window w
    Q = np.zeros((len(ws), len(uniq)))
    for j in range(len(ds)):
        col = cap[:, j] * p[j]
        np.maximum(Q[:, pid[j]], col, out=Q[:, pid[j]])

    budget = budget_frac * duration
    covered = np.zeros(len(uniq))
    used, chosen, prio = 0.0, [], []
    alive = np.ones(len(ws), bool)
    L = we - ws
    while True:
        alive &= (used + L) <= budget + 1e-9
        if not alive.any():
            break
        gain = Q @ (1 - covered)
        score = gain / L if per_second else gain
        score[~alive] = -1
        k = int(np.argmax(score))
        if gain[k] <= 1e-9:
            break
        chosen.append((float(ws[k]), float(we[k])))
        prio.append(float(gain[k]))
        covered = 1 - (1 - covered) * (1 - Q[k])
        used += L[k]
        # forbid overlap with the chosen window (strict non-overlap, merging would change lengths)
        alive &= (we <= ws[k]) | (ws >= we[k])
    return chosen, prio
