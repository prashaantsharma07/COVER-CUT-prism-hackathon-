"""Simulated detector: fabricates sliding-window verb/noun probabilities from the GT with a chosen
top-1 accuracy, then runs the REAL action head + coverage selector + writer. Purpose: exercise the
whole EPIC code path and tune head parameters before real features exist. Numbers are indicative only.
usage: python tools/simulate.py <out_dir> [--acc-v 0.65 --acc-n 0.55] [selector/head options]"""
import argparse, os, sys
import numpy as np
H = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, H); sys.path.insert(0, os.path.join(H, "..", "GOLDEN_DAWN", "code"))
import gt as G
from prism import actions, coverage, writer

ap = argparse.ArgumentParser()
ap.add_argument("out"); ap.add_argument("--acc-v", type=float, default=0.65); ap.add_argument("--acc-n", type=float, default=0.55)
ap.add_argument("--win", type=float, default=2.0); ap.add_argument("--hop", type=float, default=1.0)
ap.add_argument("--topk", type=int, default=5); ap.add_argument("--rule", default="half"); ap.add_argument("--mode", default="span")
ap.add_argument("--max-len", type=float, default=12.0); ap.add_argument("--selector", default="coverage", choices=["coverage", "topscore"])
ap.add_argument("--seed", type=int, default=0)
a = ap.parse_args()
rng = np.random.default_rng(a.seed)
man = G.manifest()
epic = [v for v, (_, d) in man.items() if d == "epic_kitchens_100"]
egt = G.epic_gt(epic)
vi = {k: i for i, k in enumerate(actions.VERBS)}; ni = {k: i for i, k in enumerate(actions.NOUNS)}
# confusions are drawn from the empirical class frequency (frequent classes are the usual mistakes)
fv = np.bincount([vi[v] for x in egt.values() for _, _, v, _ in x], minlength=97) + 1.0; fv /= fv.sum()
fn = np.bincount([ni[n] for x in egt.values() for _, _, _, n in x], minlength=300) + 1.0; fn /= fn.sum()


def fake(true, acc, freq, conf):
    k = len(freq)
    p = rng.dirichlet(np.full(k, 0.05)) * (1 - conf)
    top = true if (true is not None and rng.random() < acc) else rng.choice(k, p=freq)
    p[top] += conf
    if true is not None and top != true:
        p[true] += 0.5 * (1 - conf) * rng.random()          # truth often sits in the top few
    return p / p.sum()


for v in epic:
    d = man[v][0]
    t = np.arange(a.win / 2, max(a.win / 2 + 1e-6, d - a.win / 2 + 1e-6), a.hop)
    g = egt[v]
    gs = np.array([x[0] for x in g]); ge = np.array([x[1] for x in g])
    Pv = np.zeros((len(t), 97), np.float32); Pn = np.zeros((len(t), 300), np.float32)
    for i, c in enumerate(t):
        ov = np.clip(np.minimum(ge, c + a.win / 2) - np.maximum(gs, c - a.win / 2), 0, None) if len(g) else np.array([])
        if len(ov) and ov.max() > 0.2:
            j = int(np.argmax(ov)); conf = rng.uniform(0.35, 0.9)
            Pv[i] = fake(vi[g[j][2]], a.acc_v, fv, conf); Pn[i] = fake(ni[g[j][3]], a.acc_n, fn, conf)
        else:
            conf = rng.uniform(0.1, 0.4)
            Pv[i] = fake(None, 0, fv, conf); Pn[i] = fake(None, 0, fn, conf)
    acts = actions.detect(t, a.win, Pv, Pn, d, topk=a.topk)
    if a.selector == "coverage":
        hl, pr = coverage.select(acts, d, rule=a.rule, mode=a.mode, max_len=a.max_len)
    else:                                   # ablation: best-scoring windows, no distinctness
        hl = [(x["segment"][0], max(x["segment"][1], x["segment"][0] + 4.05)) for x in acts[:200]]
        pr = [x["score"] for x in acts[:200]]
    writer.write(os.path.join(a.out, v + ".json"), writer.build(v, d, ("cooking", 1.0), acts, hl, pr))
print("simulated", len(epic), "videos", flush=True)
