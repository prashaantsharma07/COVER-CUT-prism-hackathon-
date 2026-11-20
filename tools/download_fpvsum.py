"""Downloads the 49 FPVSum manifest videos with yt-dlp as <video_id>.mp4.
usage: python tools/download_fpvsum.py <output_dir>
Matches manifest IDs to YouTube IDs via the FPVSum video list (one manifest ID,
Jumping__mU-p2wuk90, has an extra underscore compared with the annotation file)."""
import csv, os, subprocess, sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")
out = sys.argv[1]
os.makedirs(out, exist_ok=True)
vlist = os.path.join(ROOT, "ext", "fpvsum", "crawler", "FPVSum_videolist.csv")
if not os.path.exists(vlist):
    subprocess.run(["git", "clone", "https://github.com/azuxmioy/fpvsum.git", os.path.join(ROOT, "ext", "fpvsum")], check=True)
yt = {r["video_name"]: r["youtube_id"] for r in csv.DictReader(open(vlist))}
cats = {r["video_name"]: r["category"] for r in csv.DictReader(open(vlist))}


def youtube_id(vid):
    if vid in yt:
        return yt[vid]
    cat, rest = vid.split("_", 1)            # e.g. Jumping__mU-p2wuk90 -> _mU-p2wuk90
    for name, y in yt.items():
        if cats[name] == cat and y == rest:
            return y
    return rest


failed = []
for r in csv.DictReader(open(os.path.join(ROOT, "data", "manifest.csv"))):
    if r["dataset"] != "fpvsum":
        continue
    vid = r["video_id"]
    dst = os.path.join(out, vid + ".mp4")
    if os.path.exists(dst):
        continue
    url = "https://www.youtube.com/watch?v=" + youtube_id(vid)
    cmd = ["yt-dlp", "-f", "bv*[height<=480][ext=mp4]+ba[ext=m4a]/b[height<=480][ext=mp4]/bv*[height<=480]+ba/b[height<=480]/b",
           "--retries", "10", "--fragment-retries", "10", "--concurrent-fragments", "4",
           "--merge-output-format", "mp4", "-o", dst, url]
    print(">>", vid, url, flush=True)
    if subprocess.run(cmd).returncode != 0:
        failed.append(vid)
print(f"done; {len(failed)} failed: {failed}")
