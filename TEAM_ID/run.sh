#!/usr/bin/env bash
# Usage: ./run.sh <video.mp4> <out.json>
set -euo pipefail
if [ "$#" -ne 2 ]; then
  echo "usage: $0 <video.mp4> <out.json>" >&2; exit 2
fi
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PYTHONPATH="$HERE/code${PYTHONPATH:+:$PYTHONPATH}"
exec python3 -m prism.main "$1" "$2"
