"""Phase 1 heads: valid but naive. Each is replaced by a real head later."""
import math

from . import config as C


def label_head(video_path, duration):
    return ("other", 0.0)


def action_head(video_path, duration):
    return []


def highlight_head(video_path, duration, label, actions):
    """Evenly spaced 4.5 s windows using ~14.5% of the duration."""
    L = 4.5
    n = max(1, int(math.floor(0.145 * duration / L)))
    step = duration / n
    segs = [(i * step + (step - L) / 2, i * step + (step + L) / 2) for i in range(n)]
    return segs, None
