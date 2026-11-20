#!/usr/bin/env bash
# Final packaging for team GOLDEN_DAWN. Run from this folder:   bash finish.sh
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
TEAM=GOLDEN_DAWN
[ -d .venv ] || python3 -m venv .venv
. .venv/bin/activate
export PYTHON=python
echo "== 1/5 installing PyTorch (CPU) and helpers into .venv"
pip install -q -U pip
pip install -q torch --index-url https://download.pytorch.org/whl/cpu || { echo "torch install failed"; exit 1; }
pip install -q numpy pandas huggingface_hub || exit 1
echo "== 2/5 setting team_id = $TEAM"
python tools/set_team_id.py "$TEAM"
echo "== 3/5 model weights"
bash tools/fetch_weights.sh || { echo "weights step failed"; exit 1; }
[ -s GOLDEN_DAWN/weights/action_tsfb_ek100.pt ] && [ -d GOLDEN_DAWN/weights/siglip ] || { echo "weights missing"; exit 1; }
echo "== 4/5 checks"
python tools/validate_strict.py --dir GOLDEN_DAWN/outputs | tail -1
bash tools/integrity_check.sh
if [ -f validate.py ]; then python validate.py --dir GOLDEN_DAWN/outputs/ | tail -5
else echo "!! official validate.py not found: copy it from the student kit into $(pwd) and run:  python3 validate.py --dir GOLDEN_DAWN/outputs/"; fi
echo "== 5/5 packaging $TEAM/"
rm -rf "../$TEAM" && cp -r GOLDEN_DAWN "../$TEAM" && (cd .. && rm -f "$TEAM.zip" && zip -qr "$TEAM.zip" "$TEAM")
ls -la "../$TEAM.zip"; du -sh "../$TEAM"
echo "Done. Submission folder: $(cd .. && pwd)/$TEAM  (zip: $(cd .. && pwd)/$TEAM.zip)"
