#!/usr/bin/env bash
# Downloads model weights into TEAM_ID/weights/ so run.sh works offline.
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; W="$HERE/../TEAM_ID/weights"; mkdir -p "$W"
URL="https://dl.fbaipublicfiles.com/lavila/checkpoints/dual_encoders/ek100_cls/clip_openai_timesformer_base.ft_ek100_cls.ep_0100.md5sum_4e3575.pth"
if [ ! -s "$W/action_tsfb_ek100.pt" ]; then
  echo "downloading LaViLa TimeSformer-B (EPIC-100 fine-tuned)"
  wget -q --show-progress -c -O "$W/raw_lavila.pth" "$URL" || { echo "download failed"; exit 1; }
  "${PYTHON:-python3}" - "$W" <<'PY'
import sys, torch
w = sys.argv[1]
ck = torch.load(w + "/raw_lavila.pth", map_location="cpu", weights_only=False)
args = ck.get("args")
print("checkpoint args (check the training split):")
for k in ["model", "dataset", "metadata_train", "metadata_val", "clip_length", "num_classes", "use_vn_classifier"]:
    print("   ", k, "=", getattr(args, k, "?"))
sd = {k.replace("module.", ""): v.half() if v.is_floating_point() else v for k, v in ck["state_dict"].items()}
sd = {k: v for k, v in sd.items() if k.startswith(("visual.", "fc_cls."))}
torch.save(sd, w + "/action_tsfb_ek100.pt")
print("saved", len(sd), "tensors")
PY
  [ -s "$W/action_tsfb_ek100.pt" ] && rm -f "$W/raw_lavila.pth"
fi
if [ ! -d "$W/siglip" ]; then
  echo "downloading SigLIP base"
  "${PYTHON:-python3}" - "$W" <<'PY'
import sys
from huggingface_hub import snapshot_download
snapshot_download("google/siglip-base-patch16-224", local_dir=sys.argv[1] + "/siglip",
                  allow_patterns=["*.json", "*.safetensors", "*.model", "*.txt"])
PY
fi
cat > "$W/SOURCES.md" <<'MD'
# Weights
- `action_tsfb_ek100.pt`: LaViLa TimeSformer-B fine-tuned for EPIC-KITCHENS-100 action recognition
  (verb/noun/action heads), from the official LaViLa model zoo (MIT). Stored as fp16 state dict.
  Training split: see the `metadata_train` printed by tools/fetch_weights.sh (EPIC-100 train).
- `siglip/`: google/siglip-base-patch16-224 (Apache-2.0), used zero-shot.
MD
du -sh "$W"/* 2>/dev/null
