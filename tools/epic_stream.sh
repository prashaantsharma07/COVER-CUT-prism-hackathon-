#!/usr/bin/env bash
# Low-disk EPIC download: one video at a time -> 256 px working copy -> delete the 1080p original.
# Needs only a few GB free. Shortest videos first; safe to stop and re-run (skips finished videos).
# usage: bash tools/epic_stream.sh [raw_dir] [out_dir]      KEEP_RAW=1 keeps the originals
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; ROOT="$HERE/.."
RAW="${1:-$ROOT/videos/epic_raw}"; OUT="${2:-$ROOT/videos/epic_256}"; mkdir -p "$RAW" "$OUT" "$ROOT/ext"
[ -d "$ROOT/ext/epic-kitchens-download-scripts" ] || git clone -q https://github.com/epic-kitchens/epic-kitchens-download-scripts.git "$ROOT/ext/epic-kitchens-download-scripts"
python3 -c "
import csv
rows=[r for r in csv.DictReader(open('$ROOT/data/manifest.csv')) if r['dataset']=='epic_kitchens_100']
rows.sort(key=lambda r: float(r['duration_sec']))
print('\n'.join(r['video_id'] for r in rows))" | while read -r vid; do
  [ -s "$OUT/$vid.mp4" ] && continue
  echo "[$(date +%H:%M:%S)] downloading $vid"
  (cd "$ROOT/ext/epic-kitchens-download-scripts" && python3 epic_downloader.py --videos --specific-videos "$vid" --output-path "$RAW") >/dev/null 2>&1
  f="$(find "$RAW" -type f -iname "$vid.mp4" | head -1)"
  if [ -z "$f" ]; then echo "  FAILED to download $vid"; continue; fi
  ffmpeg -nostdin -loglevel error -y -i "$f" -vf "scale=-2:256,fps=30" -an -c:v libx264 -preset veryfast -crf 23 "$OUT/$vid.part.mp4" \
    && mv "$OUT/$vid.part.mp4" "$OUT/$vid.mp4" && echo "  ok $vid ($(du -h "$OUT/$vid.mp4" | cut -f1))"
  [ "${KEEP_RAW:-0}" = "1" ] || rm -f "$f"
done
echo "epic_stream finished: $(ls "$OUT" | wc -l) / 138 videos"
