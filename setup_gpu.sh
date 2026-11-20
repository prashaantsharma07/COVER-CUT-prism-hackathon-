#!/usr/bin/env bash
# Installs the model stack and downloads weights (about 1.5 GB). Run after setup.sh:   bash setup_gpu.sh
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
. .venv/bin/activate
pip install -q -r TEAM_ID/requirements.txt huggingface_hub || { echo "pip install failed"; exit 1; }
python - <<'PY'
import torch
print("torch", torch.__version__, "| cuda available:", torch.cuda.is_available(),
      "|", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU only (will be slow)")
PY
bash tools/fetch_weights.sh || exit 1
echo "== model check on whatever EPIC videos are already downloaded"
python tools/check_action_model.py --max-clips 60 videos/epic_256 videos/epic_raw 2>&1 | tee logs/model_check.log | tail -6
