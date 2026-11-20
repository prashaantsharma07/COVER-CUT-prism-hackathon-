"""One streaming pass over the video -> everything the heads need, cached as .npz.

keys: t{s}, Pv{s}, Pn{s} for each scale s (window centres, verb / noun probabilities);
      sig_t, sig_emb (1 fps SigLIP embeddings), label_logits [N x 15], excite [N x 14];
      motion (mean abs frame difference per second)."""
import os
import sys

import numpy as np

from . import config as C
from .frames import stream_frames

_MODELS = {}


def _models():
    if not _MODELS:
        from . import models
        _MODELS["act"] = models.ActionModel()
        _MODELS["sig"] = models.Siglip()
    return _MODELS["act"], _MODELS["sig"]


def extract(video_path, duration, cache_dir=None, video_id=None):
    if cache_dir and video_id:
        cp = os.path.join(cache_dir, video_id + ".npz")
        if os.path.exists(cp):
            return dict(np.load(cp))
    act, sig = _models()
    bs = int(os.environ.get("PRISM_BATCH", "4"))
    T = C.CLIP_FRAMES
    need = T * max(C.SCALES)
    buf = []                                     # ring of the last `need` frames
    pend = {s: ([], []) for s in C.SCALES}       # scale -> (clips, centre times)
    out = {s: ([], [], []) for s in C.SCALES}    # scale -> (t, Pv, Pn)
    sig_frames, sig_t, sig_emb = [], [], []
    motion_acc, prev_small = {}, None

    def flush(s, force=False):
        clips, ts = pend[s]
        while len(clips) >= bs or (force and clips):
            pv, pn = act(np.stack(clips[:bs]))
            out[s][0].extend(ts[:bs]); out[s][1].append(pv.astype(np.float16)); out[s][2].append(pn.astype(np.float16))
            del clips[:bs], ts[:bs]

    def flush_sig(force=False):
        if len(sig_frames) >= 32 or (force and sig_frames):
            sig_emb.append(sig.image_emb(sig_frames)); sig_frames.clear()

    n = 0
    for fr in stream_frames(video_path, C.FPS):
        small = fr[::4, ::4].mean(axis=2)
        if prev_small is not None:
            sec = n // C.FPS
            a = motion_acc.setdefault(sec, [0.0, 0])
            a[0] += float(np.abs(small - prev_small).mean()); a[1] += 1
        prev_small = small
        if n % C.FPS == 0:
            sig_frames.append(fr.copy()); sig_t.append(n / C.FPS); flush_sig()
        buf.append(fr)
        if len(buf) > need:
            buf.pop(0)
        n += 1
        for s in C.SCALES:
            span = T * s
            if n >= span and (n - span) % (span // 2) == 0:
                clip = np.stack(buf[-span::s])
                pend[s][0].append(clip); pend[s][1].append((n - span / 2) / C.FPS); flush(s)
    if n == 0:
        raise RuntimeError(f"no frames decoded from {video_path}")
    if n < T:                                    # very short clip: pad by repeating the last frame
        clip = np.stack(buf + [buf[-1]] * (T - len(buf)))
        pend[C.SCALES[0]][0].append(clip); pend[C.SCALES[0]][1].append(n / C.FPS / 2)
    for s in C.SCALES:
        flush(s, True)
    flush_sig(True)
    emb = np.concatenate(sig_emb)
    cls, exc, scale = sig.text_side()
    feats = {"sig_t": np.array(sig_t, np.float32), "sig_emb": emb.astype(np.float16),
             "label_logits": (emb @ cls.T * scale).astype(np.float32), "excite": (emb @ exc.T).astype(np.float32),
             "motion": np.array([motion_acc[k][0] / max(1, motion_acc[k][1]) if k in motion_acc else 0.0
                                 for k in range(int(np.ceil(n / C.FPS)))], np.float32)}
    for s in C.SCALES:
        t, pv, pn = out[s]
        feats[f"t{s}"] = np.array(t, np.float32)
        feats[f"Pv{s}"] = np.concatenate(pv) if pv else np.zeros((0, 97), np.float16)
        feats[f"Pn{s}"] = np.concatenate(pn) if pn else np.zeros((0, 300), np.float16)
    if cache_dir and video_id:
        os.makedirs(cache_dir, exist_ok=True)
        tmp = os.path.join(cache_dir, video_id + ".tmp.npz")
        np.savez_compressed(tmp, **feats); os.replace(tmp, os.path.join(cache_dir, video_id + ".npz"))
    return feats
