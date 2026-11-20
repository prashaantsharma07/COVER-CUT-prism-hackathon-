"""Single-video entry point: python -m prism.main <video.mp4> <out.json>"""
import argparse
import os
import sys
import time

from . import baseline, writer
from .video_io import probe_duration


def run(video_path, out_path):
    t0 = time.time()
    video_id = os.path.splitext(os.path.basename(video_path))[0]
    duration = probe_duration(video_path)
    mode = "full"
    try:
        from . import features, heads
        feats = features.extract(video_path, duration, os.environ.get("PRISM_CACHE"), video_id)
        label, actions, hl, prio = heads.run(feats, duration)
    except Exception as ex:                      # never leave a video without a valid JSON
        if os.environ.get("PRISM_STRICT") == "1":
            raise
        print(f"[prism] WARNING: falling back to the naive baseline for {video_id}: {type(ex).__name__}: {ex}",
              file=sys.stderr)
        mode = "BASELINE-FALLBACK"
        label = baseline.label_head(video_path, duration)
        actions = baseline.action_head(video_path, duration)
        hl, prio = baseline.highlight_head(video_path, duration, label, actions)
    obj = writer.build(video_id, duration, label, actions, hl, prio)
    writer.write(out_path, obj)
    print(f"[prism] {video_id}: {mode}, {duration:.1f}s, label={obj['label']['name']}, {len(obj['actions'])} actions, "
          f"{len(obj['highlights'])} highlights, {time.time() - t0:.1f}s", file=sys.stderr)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("video")
    p.add_argument("out")
    a = p.parse_args()
    run(a.video, a.out)


if __name__ == "__main__":
    main()
