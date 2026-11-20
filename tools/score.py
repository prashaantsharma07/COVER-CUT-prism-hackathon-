"""Local re-implementation of the four Track 04 metrics. Assumptions are marked ASSUMPTION;
confirm them with the organizers and change them here.

usage: python tools/score.py --pred GOLDEN_DAWN/outputs [--split A|B] [--coverage-rule half] [--per-video out.csv]
"""
import argparse, json, os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gt as G

WEIGHTS = {"action_map": 0.25, "coverage": 0.40, "highlight_f1": 0.25, "label_acc": 0.10}
TIOUS = [0.1, 0.2, 0.3, 0.4, 0.5]


def load_pred(d, vid):
    try:
        o = json.load(open(os.path.join(d, vid + ".json")))
        assert all(k in o for k in ["label", "actions", "highlights", "duration_sec"])
        return o
    except Exception:
        return None                                            # missing/invalid -> scores 0


def hl_segments(o, min_len=4.0):
    if not o:
        return []
    return [tuple(h["segment"]) for h in o["highlights"] if h["segment"][1] - h["segment"][0] >= min_len]


# ---------------- Action mAP (ActivityNet-style AP per class, averaged over tIoU 0.1..0.5)
def _ap(tp, n_gt):
    if n_gt == 0:
        return None
    if len(tp) == 0:
        return 0.0
    tp = np.asarray(tp, float)
    ctp, cfp = np.cumsum(tp), np.cumsum(1 - tp)
    rec, prec = ctp / n_gt, ctp / np.maximum(ctp + cfp, 1e-12)
    mrec = np.concatenate([[0], rec, [1]])
    mpre = np.concatenate([[0], prec, [0]])
    for i in range(len(mpre) - 2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i + 1])
    idx = np.where(mrec[1:] != mrec[:-1])[0] + 1
    return float(np.sum((mrec[idx] - mrec[idx - 1]) * mpre[idx]))


def _tiou(s, e, gs, ge):
    inter = np.clip(np.minimum(e, ge) - np.maximum(s, gs), 0, None)
    union = np.maximum(e, ge) - np.minimum(s, gs)
    return inter / np.maximum(union, 1e-9)


def action_map(preds, gts, key):
    """key: function (verb, noun) -> class id. ASSUMPTION: classes = (verb, noun) pairs for 'action'."""
    gt_by = {}
    for vid, items in gts.items():
        for s, e, v, n in items:
            gt_by.setdefault(key(v, n), {}).setdefault(vid, []).append((s, e))
    pr_by = {}
    for vid, o in preds.items():
        for a in (o["actions"] if o else []):
            pr_by.setdefault(key(a["verb"], a["noun"]), []).append((a["score"], vid, a["segment"][0], a["segment"][1]))
    res = []
    for thr in TIOUS:
        aps = []
        for c, per_vid in gt_by.items():
            n_gt = sum(len(x) for x in per_vid.values())
            arr = {v: np.array(x, float) for v, x in per_vid.items()}
            used = {v: np.zeros(len(x), bool) for v, x in per_vid.items()}
            tp = []
            for score, vid, s, e in sorted(pr_by.get(c, []), key=lambda t: -t[0]):
                if vid not in arr:
                    tp.append(0); continue
                g = arr[vid]
                iou = _tiou(s, e, g[:, 0], g[:, 1])
                hit = 0
                for j in np.argsort(-iou):
                    if iou[j] < thr:
                        break
                    if not used[vid][j]:
                        used[vid][j] = True; hit = 1; break
                tp.append(hit)
            aps.append(_ap(tp, n_gt))
        res.append(float(np.mean(aps)) if aps else 0.0)
    return float(np.mean(res)), dict(zip(TIOUS, res))


# ---------------- Action coverage
def coverage(preds, gts, man, rule):
    """ASSUMPTION: share of distinct GT (verb, noun) pairs with at least one instance captured
    by the union of >= 4 s highlights; 'any' = overlaps, 'half' = >= 50% of it inside,
    'full' = fully inside. ASSUMPTION: soft penalty x min(1, 0.15 / used_fraction)."""
    per = {}
    for vid, items in gts.items():
        pairs = {}
        for s, e, v, n in items:
            pairs.setdefault((v, n), []).append((s, e))
        if not pairs:
            continue
        o = preds.get(vid)
        segs = hl_segments(o)
        hit = 0
        for inst in pairs.values():
            ok = False
            for s, e in inst:
                L = max(e - s, 1e-3)
                inside = sum(max(0.0, min(e, he) - max(s, hs)) for hs, he in segs)
                if rule == "any":
                    ok = any(min(e, he) >= max(s, hs) for hs, he in segs)
                elif rule == "half":
                    ok = inside >= 0.5 * L
                else:
                    ok = inside >= L - 1e-6
                if ok:
                    break
            hit += ok
        c = hit / len(pairs)
        if o:
            used = sum(h["segment"][1] - h["segment"][0] for h in o["highlights"]) / man[vid][0]
            if used > 0.15:
                c *= 0.15 / used
        per[vid] = c
    return (float(np.mean(list(per.values()))) if per else 0.0), per


# ---------------- Highlight F1 (FPVSum)
def gt_mask(scores):
    """ASSUMPTION: top 15% of frames by consensus score, ties broken by frame order."""
    k = int(round(0.15 * len(scores)))
    m = np.zeros(len(scores), bool)
    m[np.argsort(-scores, kind="stable")[:k]] = True
    return m


def highlight_f1(preds, fgt, man):
    per = {}
    for vid, (scores, _) in fgt.items():
        n = len(scores)
        fps = n / man[vid][0]
        t = (np.arange(n) + 0.5) / fps
        pm = np.zeros(n, bool)
        for s, e in hl_segments(preds.get(vid)):
            pm |= (t >= s) & (t < e)
        g = gt_mask(scores)
        tp = (pm & g).sum()
        per[vid] = 0.0 if tp == 0 else float(2 * tp / (pm.sum() + g.sum()))
    return (float(np.mean(list(per.values()))) if per else 0.0), per


def label_acc(preds, fgt):
    per = {v: float(bool(preds.get(v)) and preds[v]["label"]["name"] == cat) for v, (_, cat) in fgt.items()}
    return (float(np.mean(list(per.values()))) if per else 0.0), per


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", required=True)
    ap.add_argument("--split", choices=["A", "B"], help="EPIC participants: A = odd IDs, B = even IDs")
    ap.add_argument("--coverage-rule", default="half", choices=["any", "half", "full"])
    ap.add_argument("--per-video")
    a = ap.parse_args()
    man = G.manifest()
    epic = [v for v, (_, ds) in man.items() if ds == "epic_kitchens_100"]
    if a.split:
        epic = [v for v in epic if (int(v[1:3]) % 2 == 1) == (a.split == "A")]
    fpv = [v for v, (_, ds) in man.items() if ds == "fpvsum"]
    preds = {v: load_pred(a.pred, v) for v in epic + fpv}
    n_missing = sum(p is None for p in preds.values())
    egt = G.epic_gt(epic)
    fgt = G.fpvsum_gt(fpv)

    m_act, by_t = action_map(preds, egt, lambda v, n: (v, n))
    m_verb, _ = action_map(preds, egt, lambda v, n: v)
    m_noun, _ = action_map(preds, egt, lambda v, n: n)
    covs = {r: coverage(preds, egt, man, r) for r in ["any", "half", "full"]}
    f1, f1_per = highlight_f1(preds, fgt, man)
    acc, acc_per = label_acc(preds, fgt)
    res = {"action_map": m_act, "coverage": covs[a.coverage_rule][0], "highlight_f1": f1, "label_acc": acc}
    total = sum(WEIGHTS[k] * v for k, v in res.items())

    print(f"videos: {len(epic)} EPIC (GT for {sum(bool(x) for x in egt.values())}), "
          f"{len(fpv)} FPVSum (GT for {len(fgt)}); missing/invalid predictions: {n_missing}")
    print(f"  Action mAP   (25%)  {m_act:.4f}   [verb {m_verb:.4f}, noun {m_noun:.4f}; "
          + ", ".join(f"@{t}:{v:.3f}" for t, v in by_t.items()) + "]")
    print(f"  Coverage     (40%)  {res['coverage']:.4f}   [rule={a.coverage_rule}; any {covs['any'][0]:.4f}, "
          f"half {covs['half'][0]:.4f}, full {covs['full'][0]:.4f}]")
    print(f"  Highlight F1 (25%)  {f1:.4f}")
    print(f"  Label acc    (10%)  {acc:.4f}")
    print(f"  TOTAL               {100 * total:.2f} / 100")
    if a.per_video:
        import csv
        with open(a.per_video, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["video_id", "coverage_any", "coverage_half", "coverage_full", "highlight_f1", "label_ok"])
            for v in epic:
                w.writerow([v] + [round(covs[r][1].get(v, 0), 4) for r in ["any", "half", "full"]] + ["", ""])
            for v in fpv:
                w.writerow([v, "", "", "", round(f1_per.get(v, 0), 4), acc_per.get(v, "")])


if __name__ == "__main__":
    main()
