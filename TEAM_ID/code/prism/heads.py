"""Pure-numpy heads: cached features -> label, actions, highlights. No models needed here,
so parameters can be tuned from the cache in seconds."""
import numpy as np

from . import actions, coverage, fpvsum
from . import config as C
from . import label as L


def run(feats, duration, cfg=None):
    g = lambda k: (cfg or {}).get(k, getattr(C, k))
    name, score, _ = L.classify(feats["label_logits"])
    if name == "cooking":
        acts = []
        for s in g("SCALES"):
            if f"t{s}" in feats and len(feats[f"t{s}"]):
                win = C.CLIP_FRAMES * s / C.FPS
                acts += actions.detect(feats[f"t{s}"], win, feats[f"Pv{s}"], feats[f"Pn{s}"], duration, topk=g("ACT_TOPK"),
                                       alpha=g("ACT_PRIOR_ALPHA"), smooth=g("ACT_SMOOTH"), power=g("ACT_POWER"), gap=g("ACT_GAP"),
                                       unseen=g("ACT_UNSEEN"))
        acts.sort(key=lambda a: -a["score"])
        hl, prio = coverage.select(acts[:1500], duration, rule=g("COVER_RULE"), mode=g("COVER_MODE"),
                                   max_len=g("COVER_MAX_LEN"))
        return (name, score), acts, hl, prio
    k = L.SPORTS.index(name)
    T = len(feats["sig_t"])
    motion = np.asarray(feats["motion"], float)[:T]
    motion = np.pad(motion, (0, T - len(motion)), mode="edge") if len(motion) < T and len(motion) else motion
    imp = fpvsum.importance(motion, feats["excite"][:, k], np.asarray(feats["sig_emb"], np.float32), w=g("FPV_WEIGHTS"))
    hl, prio = fpvsum.select(imp, duration, np.asarray(feats["sig_emb"], np.float32), seg_len=g("FPV_SEG_LEN"))
    return (name, score), [], hl, prio
