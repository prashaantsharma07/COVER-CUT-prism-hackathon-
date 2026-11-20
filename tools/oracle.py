"""Upper bounds: feed GT actions into the REAL selector + writer to see the best achievable
coverage under the 15% / 4 s rules, and to sanity-check score.py (mAP and label must be 1.0).
Outputs go to a scratch dir. Never submit them.
usage: python tools/oracle.py <out_dir> [--rule half --mode span --win 4.05 --max-len 12]"""
import argparse, os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "TEAM_ID", "code"))
import gt as G
from prism import coverage, writer

ap = argparse.ArgumentParser()
ap.add_argument("out"); ap.add_argument("--rule", default="half"); ap.add_argument("--mode", default="span")
ap.add_argument("--win", type=float, default=4.05); ap.add_argument("--max-len", type=float, default=12.0)
a = ap.parse_args()
man = G.manifest()
epic = [v for v, (_, d) in man.items() if d == "epic_kitchens_100"]
fpv = [v for v, (_, d) in man.items() if d == "fpvsum"]
egt, fgt = G.epic_gt(epic), G.fpvsum_gt(fpv)
vocab_miss = {(v, n) for x in egt.values() for _, _, v, n in x if v not in writer.VERBS or n not in writer.NOUNS}
print("GT pairs outside supported vocab:", len(vocab_miss))
for v in epic:
    d = man[v][0]
    acts = [{"segment": [s, e], "verb": vb, "noun": nn, "score": 1.0} for s, e, vb, nn in egt[v]]
    hl, pr = coverage.select(acts, d, rule=a.rule, mode=a.mode, win_len=a.win, max_len=a.max_len)
    writer.write(os.path.join(a.out, v + ".json"), writer.build(v, d, ("cooking", 1.0), acts, hl, pr))
for v in fpv:
    d = man[v][0]
    scores, cat = fgt[v]
    fps = len(scores) / d
    k = int(round(0.15 * len(scores)))
    m = np.zeros(len(scores), bool); m[np.argsort(-scores, kind="stable")[:k]] = True
    segs, i = [], 0
    while i < len(m):                       # GT runs -> segments, pad short runs to 4.05 s
        if m[i]:
            j = i
            while j < len(m) and m[j]:
                j += 1
            s, e = i / fps, j / fps
            if e - s < 4.05:
                c = (s + e) / 2; s, e = c - 2.025, c + 2.025
            segs.append((s, e)); i = j
        else:
            i += 1
    writer.write(os.path.join(a.out, v + ".json"), writer.build(v, d, (cat, 1.0), [], segs, [e - s for s, e in segs]))
print("written", flush=True)
