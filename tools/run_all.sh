#!/usr/bin/env bash
# Runs run.sh on every manifest video found under the given dirs; skips existing outputs.
# usage: bash tools/run_all.sh <out_dir> <video_dir> [<video_dir> ...]
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUT="$1"; shift; mkdir -p "$OUT"
PRISM_MANIFEST="$HERE/../data/manifest.csv" python3 - "$@" <<'PY' > /tmp/prism_runlist.txt
import csv, os, sys
man = [r["video_id"] for r in csv.DictReader(open(os.environ.get("PRISM_MANIFEST", "data/manifest.csv")))]
found = {}
for d in sys.argv[1:]:
    for root, _, files in os.walk(d):
        for f in files:
            s, e = os.path.splitext(f)
            if e.lower() == ".mp4":
                found.setdefault(s, os.path.join(root, f))
for v in man:
    if v in found:
        print(v, found[v])
PY
fail=0
while read -r vid path; do
  [ -s "$OUT/$vid.json" ] && continue
  "$HERE/../TEAM_ID/run.sh" "$path" "$OUT/$vid.json" || { echo "FAILED $vid"; fail=$((fail+1)); }
done < /tmp/prism_runlist.txt
echo "run_all finished, $fail failures"
