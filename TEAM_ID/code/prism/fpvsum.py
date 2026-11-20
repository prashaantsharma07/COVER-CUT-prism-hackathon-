"""FPVSum highlight head: per-second importance -> contiguous >= 4 s segments within budget.

importance = w_motion * z(motion) + w_sem * z(semantic) + w_event * z(feature change),
smoothed; then greedy selection of the best windows with a redundancy penalty."""
import numpy as np

from . import config as C


def _z(x):
    x = np.asarray(x, float)
    if len(x) == 0:
        return x
    s = x.std()
    return (x - x.mean()) / s if s > 1e-8 else np.zeros_like(x)


def _smooth(x, k):
    if k <= 1 or len(x) < k:
        return x
    return np.convolve(np.pad(x, (k // 2, k - 1 - k // 2), mode="edge"), np.ones(k) / k, mode="valid")


def importance(motion, sem, emb, w=(1.0, 1.0, 0.5), smooth=3):
    """All inputs sampled once per second (same length T; emb is [T x D] L2-normalised)."""
    T = len(motion)
    event = np.zeros(T)
    if emb is not None and len(emb) > 1:
        d = 1 - np.sum(emb[1:] * emb[:-1], axis=1)
        event[1:] = d
    imp = w[0] * _z(motion) + w[1] * _z(sem if sem is not None else np.zeros(T)) + w[2] * _z(event)
    return _smooth(imp, smooth)


def select(imp, duration, emb=None, seg_len=5.0, budget_frac=C.TARGET_FRAC, redundancy=0.5):
    """Greedy: repeatedly take the best seg_len window (1 s grid) that does not overlap chosen ones.
    Windows similar to already chosen content are penalised (novelty)."""
    T = len(imp)
    if T == 0:
        return [], None
    L = int(round(seg_len))
    if T < L:
        return [(0.0, min(duration, C.SAFE_MIN_SEG))], [1.0]
    win = np.array([imp[i:i + L].mean() for i in range(T - L + 1)])
    wemb = None
    if emb is not None and len(emb) == T:
        wemb = np.stack([emb[i:i + L].mean(0) for i in range(T - L + 1)])
        wemb /= np.linalg.norm(wemb, axis=1, keepdims=True) + 1e-8
    budget = budget_frac * duration
    n_max = max(1, int(budget // seg_len))
    free = np.ones(T - L + 1, bool)
    chosen, prio, picked = [], [], []
    score = win.copy()
    for _ in range(n_max):
        s = np.where(free, score, -np.inf)
        k = int(np.argmax(s))
        if not np.isfinite(s[k]):
            break
        chosen.append((float(k), float(min(duration, k + seg_len))))
        prio.append(float(len(chosen) and -len(chosen)))
        free[max(0, k - L + 1):k + L] = False
        if wemb is not None:
            picked.append(wemb[k])
            sim = np.max(np.stack(picked) @ wemb.T, axis=0)
            score = win - redundancy * np.clip(sim - 0.8, 0, None) * 5 * max(win.std(), 1e-6)
    return chosen, [float(-i) for i in range(len(chosen))]
