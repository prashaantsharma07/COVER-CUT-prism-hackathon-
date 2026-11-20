"""Sanity check for the action model: top-1 verb/noun accuracy on ground-truth EPIC val segments
of whatever videos are on disk. Expected roughly verb 65-70%, noun 55-60%. If far lower, the
preprocessing or checkpoint loading is wrong: stop and fix before extracting features.
usage: python tools/check_action_model.py [--max-clips 60] <video_dir> [...]"""
import argparse, os, random, subprocess, sys
import numpy as np
H = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, H); sys.path.insert(0, os.path.join(H, "..", "TEAM_ID", "code"))
import gt as G

ap = argparse.ArgumentParser(); ap.add_argument("dirs", nargs="+"); ap.add_argument("--max-clips", type=int, default=60)
a = ap.parse_args()
man = G.manifest(); found = {}
for d in a.dirs:
    for root, _, files in os.walk(d):
        for f in files:
            s, e = os.path.splitext(f)
            if e.lower() == ".mp4" and man.get(s, (0, ""))[1] == "epic_kitchens_100":
                found.setdefault(s, os.path.join(root, f))
if not found:
    sys.exit("no EPIC videos on disk yet; re-run this after some have downloaded")
from prism import models, actions
gt = G.epic_gt(list(found))
segs = [(v, s, e, vb, nn) for v, x in gt.items() for s, e, vb, nn in x if e - s >= 0.5]
random.Random(0).shuffle(segs); segs = segs[:a.max_clips]
model = models.ActionModel()
okv = okn = 0


def clip(path, s, e, T=16, size=224):
    vf = f"fps={T / (e - s):.4f},scale={size}:{size}:force_original_aspect_ratio=increase,crop={size}:{size}"
    raw = subprocess.run(["ffmpeg", "-nostdin", "-loglevel", "error", "-ss", f"{s:.3f}", "-t", f"{e - s:.3f}", "-i", path,
                          "-vf", vf, "-an", "-f", "rawvideo", "-pix_fmt", "rgb24", "-"], capture_output=True).stdout
    fr = np.frombuffer(raw, np.uint8).reshape(-1, size, size, 3)
    if len(fr) == 0:
        return None
    idx = np.clip(np.arange(T), 0, len(fr) - 1)
    return fr[idx]


n = 0
for v, s, e, vb, nn in segs:
    c = clip(found[v], s, e)
    if c is None:
        continue
    pv, pn = model(c[None])
    okv += actions.VERBS[int(pv[0].argmax())] == vb; okn += actions.NOUNS[int(pn[0].argmax())] == nn; n += 1
print(f"{n} clips from {len(found)} videos: verb top-1 {100 * okv / max(1, n):.1f}%, noun top-1 {100 * okn / max(1, n):.1f}%")
print("expected about verb 65-70%, noun 55-60% (small samples are noisy)")
