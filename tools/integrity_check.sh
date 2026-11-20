#!/usr/bin/env bash
# Fails if anything in the submission looks like a lookup of eval videos or GT files.
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; SUB="$HERE/../GOLDEN_DAWN"
ids=$(python3 -c "import csv;print('|'.join(r['video_id'] for r in csv.DictReader(open('$HERE/../data/manifest.csv'))))")
hits=$(grep -rIlE "$ids|EPIC_100_validation|gt_data|raw_data\.mat|FPVSum_videolist" "$SUB/code" "$SUB/weights" "$SUB/run.sh" 2>/dev/null)
big=$(find "$SUB/code" -type f -size +2M)
if [ -n "$hits$big" ]; then echo "INTEGRITY FAIL:"; echo "$hits"; [ -n "$big" ] && echo "large files in code/: $big"; exit 1; fi
echo "integrity ok: no manifest IDs or GT references in code/, weights/, run.sh"
