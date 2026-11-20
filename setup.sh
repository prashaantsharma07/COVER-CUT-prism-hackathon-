#!/usr/bin/env bash
# One-time setup. Run from the prism folder:   bash setup.sh        (add --no-download to skip downloads)
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
miss=()
python3 -c "import ensurepip" 2>/dev/null || miss+=(python3-venv)
command -v ffmpeg >/dev/null || miss+=(ffmpeg)
command -v git >/dev/null || miss+=(git)
if [ ${#miss[@]} -gt 0 ]; then
  echo; echo "Missing system packages. Run this once, then run 'bash setup.sh' again:"
  echo "    sudo apt update && sudo apt install -y ${miss[*]}"; echo; exit 1
fi
[ -d .venv ] || python3 -m venv .venv
. .venv/bin/activate
pip install -q -U pip && pip install -q -r tools/requirements-dev.txt || { echo "pip install failed"; exit 1; }
mkdir -p ext logs videos
[ -d ext/epic-kitchens-100-annotations ] || git clone -q --depth 1 https://github.com/epic-kitchens/epic-kitchens-100-annotations.git ext/epic-kitchens-100-annotations
[ -d ext/fpvsum ] || git clone -q --depth 1 https://github.com/azuxmioy/fpvsum.git ext/fpvsum
[ -d ext/epic-kitchens-download-scripts ] || git clone -q --depth 1 https://github.com/epic-kitchens/epic-kitchens-download-scripts.git ext/epic-kitchens-download-scripts

FREE_GB=$(df -BG --output=avail . | tail -1 | tr -dc '0-9')
{
  echo "== machine report $(date)"
  (lsb_release -ds 2>/dev/null || grep PRETTY_NAME /etc/os-release); python3 --version
  echo "cpu cores: $(nproc), ram: $(free -g | awk 'NR==2{print $2}') GB, free disk here: ${FREE_GB} GB"
  nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader 2>/dev/null || echo "GPU: nvidia-smi not found (no NVIDIA driver?)"
  ffmpeg -version | head -1
} | tee setup_report.txt

echo "== sanity check (should print TOTAL 9.78 / 100)"
rm -rf work/sanity && python tools/dry_baseline.py data/manifest.csv work/sanity >/dev/null \
  && python tools/score.py --pred work/sanity | tail -1 | tee -a setup_report.txt

if [ "${1:-}" != "--no-download" ]; then
  echo "== starting downloads in the background"
  nohup python tools/download_fpvsum.py videos/fpvsum > logs/fpvsum.log 2>&1 &
  if [ "$FREE_GB" -ge 210 ]; then
    echo "disk ok: full EPIC download (keeps originals)"; nohup bash tools/download_epic.sh videos/epic_raw > logs/epic.log 2>&1 &
  else
    echo "only ${FREE_GB} GB free: low-disk mode (download one video -> shrink to 256 px -> delete original)"
    nohup bash tools/epic_stream.sh > logs/epic.log 2>&1 &
  fi
  echo "watch progress:   tail -f logs/epic.log      and      tail -f logs/fpvsum.log"
  echo "they keep running if you close this terminal; re-running setup.sh resumes them."
fi
echo; echo "Setup done. Paste the contents of setup_report.txt back to Claude."
