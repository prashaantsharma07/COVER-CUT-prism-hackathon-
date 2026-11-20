# Weights (downloaded by `bash tools/fetch_weights.sh`; too large for GitHub)
- `action_tsfb_ek100.pt`: LaViLa TimeSformer-B fine-tuned for EPIC-KITCHENS-100 action recognition
  (verb 97 / noun 300 / action 3806 heads), official LaViLa model zoo, MIT license. Source:
  https://dl.fbaipublicfiles.com/lavila/checkpoints/dual_encoders/ek100_cls/clip_openai_timesformer_base.ft_ek100_cls.ep_0100.md5sum_4e3575.pth
  converted to an fp16 state dict. Checkpoint metadata: metadata_train = EPIC_100_train.csv.
- `siglip/`: google/siglip-base-patch16-224 (Apache-2.0), used zero-shot.
