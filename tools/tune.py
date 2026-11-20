"""In-process sweep over head parameters using the feature cache. Prints per-metric results.
usage: python tools/tune.py <cache_dir> <grid_name>"""
import os, sys, itertools, json
import numpy as np
H = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, H); sys.path.insert(0, os.path.join(H, "..", "GOLDEN_DAWN", "code"))
import gt as G, score as S
from prism import heads, writer
cache = sys.argv[1]
man = G.manifest()
feats = {f[:-4]: dict(np.load(os.path.join(cache, f))) for f in sorted(os.listdir(cache)) if f.endswith(".npz")}
epic = [v for v, (_, d) in man.items() if d == "epic_kitchens_100"]
fpv = [v for v, (_, d) in man.items() if d == "fpvsum"]
egt, fgt = G.epic_gt(epic), G.fpvsum_gt(fpv)


def evaluate(cfg, which=("epic", "fpv"), split=None):
    preds = {}
    for v, f in feats.items():
        ds = man[v][1]
        if ("epic" not in which and ds != "fpvsum") or ("fpv" not in which and ds == "fpvsum"):
            continue
        lab, acts, hl, pr = heads.run(f, man[v][0], cfg)
        preds[v] = writer.build(v, man[v][0], lab, acts, hl, pr)
    out = {}
    if "epic" in which:
        e = [v for v in epic if split is None or ((int(v[1:3]) % 2 == 1) == (split == "A"))]
        g = {v: egt[v] for v in e}
        out["map"] = S.action_map(preds, g, lambda a, b: (a, b))[0]
        for r in ["any", "half", "full"]:
            out["cov_" + r] = S.coverage(preds, g, man, r)[0]
    if "fpv" in which:
        out["f1"] = S.highlight_f1(preds, fgt, man)[0]
        out["acc"] = S.label_acc(preds, fgt)[0]
    return out


if __name__ == "__main__":
    grid = json.loads(sys.argv[2])
    which = tuple(sys.argv[3].split(",")) if len(sys.argv) > 3 else ("epic", "fpv")
    keys = list(grid)
    for vals in itertools.product(*[grid[k] for k in keys]):
        cfg = dict(zip(keys, vals))
        r = evaluate(cfg, which)
        print(cfg, {k: round(v, 4) for k, v in r.items()}, flush=True)
