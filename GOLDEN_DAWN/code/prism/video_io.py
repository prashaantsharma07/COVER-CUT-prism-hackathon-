import json
import os
import re
import shutil
import subprocess

FFMPEG = os.environ.get("PRISM_FFMPEG", "ffmpeg")


def probe_duration(path: str) -> float:
    """Container duration in seconds via ffprobe; falls back to parsing `ffmpeg -i` if ffprobe is absent."""
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    if shutil.which("ffprobe"):
        out = subprocess.run(["ffprobe", "-v", "error", "-print_format", "json", "-show_format", "-show_streams",
                              "-select_streams", "v:0", path], check=True, capture_output=True, text=True).stdout
        info = json.loads(out)
        dur = info.get("format", {}).get("duration")
        if dur is None and info.get("streams"):
            dur = info["streams"][0].get("duration")
    else:
        err = subprocess.run([FFMPEG, "-hide_banner", "-i", path], capture_output=True, text=True).stderr
        m = re.search(r"Duration:\s*(\d+):(\d+):(\d+(?:\.\d+)?)", err)
        dur = int(m.group(1)) * 3600 + int(m.group(2)) * 60 + float(m.group(3)) if m else None
    if dur is None:
        raise RuntimeError(f"could not read a duration from {path}")
    dur = float(dur)
    if not dur > 0:
        raise RuntimeError(f"non-positive duration {dur} for {path}")
    return dur
