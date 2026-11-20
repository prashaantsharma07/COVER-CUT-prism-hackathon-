#!/usr/bin/env bash
# Downloads the 138 EPIC-KITCHENS-100 validation videos listed in the manifest.
# usage: bash tools/download_epic.sh <output_dir>    (needs ~184 GB free; resumable)
set -euo pipefail
OUT="${1:?usage: download_epic.sh <output_dir>}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$HERE/.."
mkdir -p "$OUT" "$ROOT/ext"
if [ ! -d "$ROOT/ext/epic-kitchens-download-scripts" ]; then
  git clone https://github.com/epic-kitchens/epic-kitchens-download-scripts.git "$ROOT/ext/epic-kitchens-download-scripts"
fi
IDS=$(python3 -c "
import csv
rows=[r for r in csv.DictReader(open('$ROOT/data/manifest.csv')) if r['dataset']=='epic_kitchens_100']
rows.sort(key=lambda r: float(r['duration_sec']))   # shortest first: the pipeline can be tested early
print(','.join(r['video_id'] for r in rows))")
echo "Downloading $(echo "$IDS" | tr ',' '\n' | wc -l) videos to $OUT"
cd "$ROOT/ext/epic-kitchens-download-scripts"
python3 epic_downloader.py --videos --specific-videos "$IDS" --output-path "$OUT"
