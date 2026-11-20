"""Compares ffprobe durations of downloaded videos with the manifest.
usage: python tools/check_durations.py <video_dir> [<video_dir> ...]
A difference > 0.5 s fails validation; > 0.1 s on FPVSum suggests a different YouTube
version than the one annotated, which shifts timestamps against the GT."""
import csv, os, subprocess, sys

HERE = os.path.dirname(os.path.abspath(__file__))
man = {r["video_id"]: (float(r["duration_sec"]), r["dataset"])
       for r in csv.DictReader(open(os.path.join(HERE, "..", "data", "manifest.csv")))}
found = {}
for d in sys.argv[1:]:
    for root, _, files in os.walk(d):
        for f in files:
            stem, ext = os.path.splitext(f)
            if ext.lower() == ".mp4" and stem in man:
                found[stem] = os.path.join(root, f)
bad, warn = [], []
for vid, (dm, ds) in sorted(man.items()):
    if vid not in found:
        continue
    out = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration",
                          "-of", "csv=p=0", found[vid]], capture_output=True, text=True).stdout.strip()
    try:
        dp = float(out)
    except ValueError:
        bad.append((vid, "unreadable")); continue
    diff = dp - dm
    if abs(diff) > 0.5:
        bad.append((vid, f"{dp:.2f} vs {dm:.2f} ({diff:+.2f}s)"))
    elif abs(diff) > 0.1:
        warn.append((vid, f"{diff:+.2f}s"))
missing = [v for v in man if v not in found]
print(f"found {len(found)}/{len(man)}; missing {len(missing)}")
for v, m in bad:
    print("FAIL", v, m)
for v, m in warn:
    print("warn", v, m)
if missing:
    print("missing:", " ".join(sorted(missing)))
