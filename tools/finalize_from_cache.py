"""Rebuilds outputs from the Kaggle feature cache with the current config (heads only; same features
run.sh computes). duration_sec comes from the Kaggle run's own JSON (ffprobe of the original video).
usage: python tools/finalize_from_cache.py <prism_results_dir> <out_dir>"""
import json, os, sys
import numpy as np
H = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(H, "..", "GOLDEN_DAWN", "code"))
from prism import heads, writer
res, out = sys.argv[1], sys.argv[2]
os.makedirs(out, exist_ok=True); n = 0
for f in sorted(os.listdir(os.path.join(res, "cache"))):
    vid = f[:-4]
    d = json.load(open(os.path.join(res, "outputs", vid + ".json")))["duration_sec"]
    feats = dict(np.load(os.path.join(res, "cache", f)))
    lab, acts, hl, pr = heads.run(feats, d)
    writer.write(os.path.join(out, vid + ".json"), writer.build(vid, d, lab, acts, hl, pr)); n += 1
print(n, "outputs written")
