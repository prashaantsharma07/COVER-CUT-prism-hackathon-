import os

# Set via env var or edit before submission. Must be identical across all files.
TEAM_ID = os.environ.get("PRISM_TEAM_ID", "TEAM_ID_PLACEHOLDER")

BUDGET_FRAC = 0.15          # official highlight budget
TARGET_FRAC = 0.149         # stay just under the budget
MIN_SEG = 4.0               # segments shorter than this are ignored by the scorer
SAFE_MIN_SEG = 4.05         # what we emit, so rounding never drops below 4.0
MAX_ACTIONS = 2000
DECIMALS = 2

# ---- feature extraction
FPS = 8                      # decode rate; a 16-frame clip spans 2 s
CLIP_FRAMES = 16
SCALES = (1, 2)              # frame step per scale: 1 -> 2 s windows (hop 1 s), 2 -> 4 s windows (hop 2 s)
ACTION_CKPT = "action_tsfb_ek100.pt"
SIGLIP_DIR = "siglip"

# ---- heads (tuned with tools/score.py; see README)
ACT_TOPK = 5
ACT_PRIOR_ALPHA = 0.0      # 0 = binary seen/unseen pair prior; >0 weights pairs by train frequency
ACT_SMOOTH = 1             # moving average over this many neighbouring windows
ACT_POWER = 2.0            # score calibration (score ** power); tuned on EPIC participant halves A/B
ACT_UNSEEN = 0.15
ACT_GAP = 3.0              # merge detections of the same pair closer than this (s); tuned on halves A/B
COVER_RULE = "half"          # capture rule the selector optimises for: any | half | full
COVER_MODE = "span"          # span = window covers the detected action; center = fixed short window
COVER_MAX_LEN = 12.0
FPV_WEIGHTS = (1.0, 1.0, 0.5)   # motion, semantic, event
FPV_SEG_LEN = 5.0
