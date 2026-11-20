"""Stricter local stand-in for the official validate.py (which still has the final say).

Checks the README rules plus things the official script may not: vocabulary
membership, manifest coverage and duration agreement.
usage: python tools/validate_strict.py --dir GOLDEN_DAWN/outputs --manifest data/manifest.csv
"""
import argparse, csv, json, math, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
CLASSES = os.path.join(HERE, "..", "GOLDEN_DAWN", "code", "prism", "supported_classes.json")


def num(x):
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)


def check(obj, fname, man, cls, team_ids):
    errs, warns = [], []
    keys = ["team_id", "video_id", "duration_sec", "label", "actions", "highlights"]
    for k in keys:
        if k not in obj:
            errs.append(f"missing {k}")
    if errs:
        return errs, warns
    if not isinstance(obj["team_id"], str) or not obj["team_id"]:
        errs.append("team_id empty")
    team_ids.add(obj["team_id"])
    if obj["team_id"] == "TEAM_ID_PLACEHOLDER":
        warns.append("team_id is still the placeholder")
    vid = obj["video_id"]
    if vid != os.path.splitext(fname)[0]:
        errs.append(f"video_id {vid} != filename")
    d = obj["duration_sec"]
    if not num(d) or d <= 0:
        errs.append("bad duration_sec"); return errs, warns
    if vid in man and abs(d - man[vid][0]) > 0.5:
        errs.append(f"duration {d} vs manifest {man[vid][0]} (>0.5 s)")
    lab = obj["label"]
    if not isinstance(lab, dict) or not isinstance(lab.get("name"), str) or not num(lab.get("score")) or not 0 <= lab["score"] <= 1:
        errs.append("bad label")
    elif lab["name"] not in cls["labels"]:
        errs.append(f"label {lab['name']} not in supported labels")
    elif vid in man and man[vid][1] == "epic_kitchens_100" and lab["name"] != "cooking":
        warns.append(f"EPIC video labelled {lab['name']}")
    acts = obj["actions"]
    if not isinstance(acts, list) or len(acts) > 2000:
        errs.append("actions must be a list of <= 2000")
    else:
        for i, a in enumerate(acts):
            seg = a.get("segment") if isinstance(a, dict) else None
            if not (isinstance(seg, list) and len(seg) == 2 and all(num(x) for x in seg)):
                errs.append(f"action {i}: bad segment"); continue
            if not (0 <= seg[0] < seg[1] <= d + 0.5):
                errs.append(f"action {i}: segment {seg} out of range")
            if a.get("verb") not in cls["verbs"]:
                errs.append(f"action {i}: verb {a.get('verb')} not in vocab")
            if a.get("noun") not in cls["nouns"]:
                errs.append(f"action {i}: noun {a.get('noun')} not in vocab")
            if not num(a.get("score")) or not 0 <= a["score"] <= 1:
                errs.append(f"action {i}: bad score")
    hl = obj["highlights"]
    if not isinstance(hl, list) or not hl:
        errs.append("highlights must be a non-empty list"); return errs, warns
    prev_end, total, longest = -1.0, 0.0, 0.0
    for i, h in enumerate(hl):
        seg = h.get("segment") if isinstance(h, dict) else None
        if not (isinstance(seg, list) and len(seg) == 2 and all(num(x) for x in seg)):
            errs.append(f"highlight {i}: bad segment"); continue
        s, e = seg
        if not (0 <= s < e <= d + 0.5):
            errs.append(f"highlight {i}: {seg} out of range")
        if s < prev_end:
            errs.append(f"highlight {i}: overlaps or unsorted")
        elif s == prev_end:
            warns.append(f"highlight {i}: touches previous segment")
        if e - s < 4.0:
            warns.append(f"highlight {i}: {e - s:.2f}s < 4.0 s (ignored in scoring)")
        prev_end, total, longest = e, total + (e - s), max(longest, e - s)
    if longest < 4.0:
        errs.append("no highlight segment >= 4.0 s")
    if total > 0.15 * d + 1e-6:
        warns.append(f"highlights use {100 * total / d:.1f}% (> 15%)")
    return errs, warns


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dir", required=True)
    p.add_argument("--manifest", default=os.path.join(HERE, "..", "data", "manifest.csv"))
    p.add_argument("-v", action="store_true", help="print warnings")
    a = p.parse_args()
    cls = json.load(open(CLASSES))
    cls = {k: set(v) for k, v in cls.items() if isinstance(v, list)}
    man = {r["video_id"]: (float(r["duration_sec"]), r["dataset"]) for r in csv.DictReader(open(a.manifest))}
    files = sorted(f for f in os.listdir(a.dir) if f.endswith(".json"))
    team_ids, n_bad, n_warn = set(), 0, 0
    for f in files:
        try:
            obj = json.load(open(os.path.join(a.dir, f)))
            errs, warns = check(obj, f, man, cls, team_ids)
        except Exception as ex:
            errs, warns = [f"unreadable: {ex}"], []
        if errs:
            n_bad += 1
            print(f"FAIL {f}: " + "; ".join(errs))
        if warns:
            n_warn += 1
            if a.v:
                print(f"warn {f}: " + "; ".join(warns))
    missing = sorted(set(man) - {os.path.splitext(f)[0] for f in files})
    extra = sorted({os.path.splitext(f)[0] for f in files} - set(man))
    if len(team_ids) > 1:
        print(f"FAIL team_id differs across files: {team_ids}"); n_bad += 1
    print(f"{len(files)} files, {n_bad} failing, {n_warn} with warnings, "
          f"{len(missing)} manifest videos missing, {len(extra)} not in manifest")
    if missing and a.v:
        print("missing:", " ".join(missing))
    sys.exit(1 if n_bad else 0)


if __name__ == "__main__":
    main()
