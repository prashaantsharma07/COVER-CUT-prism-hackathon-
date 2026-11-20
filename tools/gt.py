"""Ground-truth loaders. Lives OUTSIDE GOLDEN_DAWN/ on purpose: never import this from code/."""
import csv, os
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")
EPIC = os.path.join(ROOT, "ext", "epic-kitchens-100-annotations")
FPV = os.path.join(ROOT, "ext", "fpvsum")


def _need(path):
    if not os.path.exists(path):
        raise SystemExit(f"missing {path}\nRun: git clone https://github.com/epic-kitchens/epic-kitchens-100-annotations.git ext/epic-kitchens-100-annotations"
                         " && git clone https://github.com/azuxmioy/fpvsum.git ext/fpvsum   (or: bash setup.sh)")
    return path


def manifest():
    return {r["video_id"]: (float(r["duration_sec"]), r["dataset"])
            for r in csv.DictReader(open(os.path.join(ROOT, "data", "manifest.csv")))}


def _ts(t):
    h, m, s = t.split(":")
    return int(h) * 3600 + int(m) * 60 + float(s)


def epic_gt(video_ids):
    """{video_id: [(start, end, verb_key, noun_key), ...]} using EPIC class keys."""
    vk = {int(r["id"]): r["key"] for r in csv.DictReader(open(_need(os.path.join(EPIC, "EPIC_100_verb_classes.csv"))))}
    nk = {int(r["id"]): r["key"] for r in csv.DictReader(open(os.path.join(EPIC, "EPIC_100_noun_classes.csv")))}
    out = {v: [] for v in video_ids}
    for r in csv.DictReader(open(os.path.join(EPIC, "EPIC_100_validation.csv"))):
        if r["video_id"] in out:
            out[r["video_id"]].append((_ts(r["start_timestamp"]), _ts(r["stop_timestamp"]),
                                       vk[int(r["verb_class"])], nk[int(r["noun_class"])]))
    return out


def _norm(name):
    return name.replace("_", "").lower()


def fpvsum_gt(video_ids):
    """{video_id: (frame_scores ndarray, category_lowercase)}"""
    import scipy.io as sio
    m = sio.loadmat(_need(os.path.join(FPV, "annotation", "gt_data.mat")), squeeze_me=True, struct_as_record=False)["gt_data"]
    by = {_norm(e.Name): np.asarray(e.Score, float) for e in m}
    out = {}
    for v in video_ids:
        if _norm(v) in by:
            out[v] = (by[_norm(v)], v.split("_", 1)[0].lower())
    return out
