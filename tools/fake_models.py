"""Stand-in models for testing the pipeline plumbing without weights or a GPU. Never used for real outputs."""
import numpy as np


def install():
    from prism import features
    rng = np.random.default_rng(0)

    class Act:
        def __call__(self, clips):
            assert clips.shape[1:] == (16, 224, 224, 3), clips.shape
            b = len(clips)
            return rng.dirichlet(np.full(97, .05), b), rng.dirichlet(np.full(300, .05), b)

    class Sig:
        def image_emb(self, fr):
            e = rng.normal(size=(len(fr), 32)); e[:, 0] = np.abs(e[:, 0]) + 3; return e / np.linalg.norm(e, axis=1, keepdims=True)

        def text_side(self):
            c = np.zeros((15, 32)); c[:, :15] = np.eye(15) * 0.0; c[14, 0] = 1.0   # always 'cooking'
            return c, rng.normal(size=(14, 32)), 100.0

    features._MODELS.update(act=Act(), sig=Sig())
