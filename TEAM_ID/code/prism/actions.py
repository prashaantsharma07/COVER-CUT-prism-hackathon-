"""Action head: sliding-window verb/noun probabilities -> temporal (verb, noun) detections.

Input per video: window centres t [T], window length, Pv [T x 97], Pn [T x 300].
1. pair score per window  s(v, n) = Pv[v] * Pn[n] * prior(v, n)   (prior from EPIC TRAIN pairs;
   unseen pairs are down-weighted, not removed: 4% of val instances use pairs unseen in train)
2. keep the top-K pairs per window
3. per pair, merge runs of nearby windows into one segment (plus the single windows themselves)
"""
import json
import os

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(_HERE, "pair_prior.json")) as f:
    _P = json.load(f)
VERBS, NOUNS = _P["verbs"], _P["nouns"]
_PRIOR = None


_PRIORS = {}


def prior_matrix(unseen=0.15, alpha=0.0):
    """Binary pair prior (alpha=0) or frequency-weighted: (count / max) ** alpha for seen pairs."""
    key = (unseen, alpha)
    if key not in _PRIORS:
        m = np.full((len(VERBS), len(NOUNS)), 0.0, np.float32)
        for v, n, c in _P["pairs"]:
            m[v, n] = c
        seen = m > 0
        w = np.where(seen, (m / m.max()) ** alpha if alpha else 1.0, 0.0)
        if alpha:
            w = np.where(seen, w / w[seen].max(), 0.0)
        floor = unseen * (w[seen].min() if alpha else 1.0)
        _PRIORS[key] = np.where(seen, w, floor).astype(np.float32)
    return _PRIORS[key]


def smooth_probs(P, k):
    """Moving average over neighbouring windows (k odd), renormalised."""
    if k <= 1 or len(P) < 2:
        return P
    P = np.asarray(P, np.float32)
    pad = np.pad(P, ((k // 2, k // 2), (0, 0)), mode="edge")
    c = np.cumsum(np.vstack([np.zeros((1, P.shape[1]), np.float32), pad]), axis=0)
    out = (c[k:] - c[:-k]) / k
    return out / out.sum(1, keepdims=True)


def detect(t, win_len, Pv, Pn, duration, topk=5, gap=1.5, min_score=0.01, max_out=1500,
           pad=0.0, unseen=0.15, alpha=0.0, smooth=1, power=1.0):
    t = np.asarray(t, float)
    Pv = np.asarray(Pv, np.float32); Pn = np.asarray(Pn, np.float32)
    if len(t) == 0:
        return []
    prior = prior_matrix(unseen, alpha)
    Pv, Pn = smooth_probs(Pv, smooth), smooth_probs(Pn, smooth)
    half = win_len / 2
    per_pair = {}                                   # (v, n) -> list of (window idx, score)
    kv, kn = min(8, Pv.shape[1]), min(12, Pn.shape[1])
    for i in range(len(t)):
        vi = np.argpartition(-Pv[i], kv - 1)[:kv]
        ni = np.argpartition(-Pn[i], kn - 1)[:kn]
        s = Pv[i, vi][:, None] * Pn[i, ni][None, :] * prior[np.ix_(vi, ni)]
        flat = np.argsort(-s, axis=None)[:topk]
        for f in flat:
            a, b = divmod(int(f), kn)
            sc = float(s[a, b])
            if sc >= min_score:
                per_pair.setdefault((int(vi[a]), int(ni[b])), []).append((i, sc))
    out = []
    for (v, n), hits in per_pair.items():
        run = [hits[0]]
        runs = []
        for h in hits[1:]:
            if t[h[0]] - t[run[-1][0]] <= gap:
                run.append(h)
            else:
                runs.append(run); run = [h]
        runs.append(run)
        for r in runs:
            sc = np.array([x[1] for x in r])
            s0, e0 = t[r[0][0]] - half - pad, t[r[-1][0]] + half + pad
            out.append((float(sc.max() * 0.5 + sc.mean() * 0.5), s0, e0, v, n))
            if len(r) >= 3:                         # also emit the peak window alone (tighter boundary)
                k = r[int(sc.argmax())]
                out.append((float(k[1]) * 0.6, t[k[0]] - half, t[k[0]] + half, v, n))
    out.sort(key=lambda x: -x[0])
    return [{"segment": [max(0.0, s), min(duration, e)], "verb": VERBS[v], "noun": NOUNS[n], "score": sc ** power}
            for sc, s, e, v, n in out[:max_out]]
