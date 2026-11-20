#!/usr/bin/env bash
# Makes small working copies (256 px short side, 30 fps, no audio) for fast iteration.
# usage: bash tools/transcode.sh <in_dir> <out_dir>
set -uo pipefail
IN="${1:?in_dir}"; OUT="${2:?out_dir}"; mkdir -p "$OUT"
find "$IN" -type f \( -iname '*.mp4' \) ! -name '*.part*' | sort | while read -r f; do
  b="$(basename "${f%.*}")"
  [ -s "$OUT/$b.mp4" ] && continue
  echo ">> $b"
  ffmpeg -nostdin -loglevel error -y -i "$f" -vf "scale=-2:256,fps=30" -an \
    -c:v libx264 -preset veryfast -crf 23 "$OUT/$b.part.mp4" && mv "$OUT/$b.part.mp4" "$OUT/$b.mp4"
done
