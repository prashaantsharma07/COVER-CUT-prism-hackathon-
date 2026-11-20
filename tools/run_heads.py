"""Re-runs only the numpy heads from cached features (seconds, no GPU) -> JSONs. For tuning.
usage: python tools/run_heads.py <cache_dir> <out_dir> [--set COVER_RULE=any --set ACT_TOPK=8 ...]"""
import argparse, ast, os, sys
import numpy as np
H = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, H); sys.path.insert(0, os.path.join(H, "..", "GOLDEN_DAWN", "code"))
import gt as G
from prism import heads, writer

ap = argparse.ArgumentParser(); ap.add_argument("cache"); ap.add_argument("out"); ap.add_argument("--set", action="append", default=[])
a = ap.parse_args()
cfg = {}
for kv in a.set:
    k, v = kv.split("=", 1)
    try:
        cfg[k] = ast.literal_eval(v)
    except Exception:
        cfg[k] = v
man = G.manifest(); n = 0
for f in sorted(os.listdir(a.cache)):
    vid = f[:-4]
    if not f.endswith(".npz") or vid not in man:
        continue
    feats = dict(np.load(os.path.join(a.cache, f)))
    d = man[vid][0]                      # tools only: run.sh itself always uses ffprobe
    lab, acts, hl, pr = heads.run(feats, d, cfg)
    writer.write(os.path.join(a.out, vid + ".json"), writer.build(vid, d, lab, acts, hl, pr)); n += 1
print(n, "videos")
