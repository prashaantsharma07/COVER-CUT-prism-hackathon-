# PRISM 2026 Track 04: Action-Aware Video Summarization

## Run
```bash
./run.sh <video.mp4> <out.json>
```
Tested on Kaggle (Tesla T4, torch 2.11, CUDA). Requires Python 3.10+, `ffmpeg`/`ffprobe` on PATH and `pip install -r requirements.txt`, or use the Dockerfile.
All model weights load from `weights/`; no network access is needed at inference time.

## Architecture
One streaming decode per video (ffmpeg, 8 fps, 224 px) feeds three feature streams, cached per video:
- **Actions:** LaViLa TimeSformer-B fine-tuned on EPIC-KITCHENS-100 (verb + noun heads) over sliding
  16-frame windows at two temporal scales (2 s and 4 s).
- **Semantics:** SigLIP image embeddings at 1 fps, compared with text prompts (zero-shot).
- **Motion:** frame-difference energy per second.

Four heads read the cache (`code/prism/heads.py`):
1. **Label:** zero-shot SigLIP over all sampled frames; also routes the video (cooking vs sport).
2. **Actions:** per-window verb x noun scores with a co-occurrence prior from the EPIC-100 train split,
   merged over time into segments (`actions.py`).
3. **EPIC highlights:** budgeted maximum coverage: greedily pick >= 4 s windows that add the most
   *new* distinct verb-noun pairs per second until 15% of the video is used (`coverage.py`).
4. **FPVSum highlights:** importance = motion + semantic "excitement" + scene change, smoothed, then
   non-redundant 5 s segments within the budget (`fpvsum.py`).

`writer.py` enforces every output rule (vocabulary, ranges, sorting, non-overlap, >= 4 s, <= 15%).
If a model fails to load, `run.sh` still writes a valid (naive) JSON and prints a warning.

## Results (local re-implementation of the four metrics, 184 of 187 videos)
| Action mAP | Action Coverage | Highlight F1 | Label Accuracy | Weighted total |
| --- | --- | --- | --- | --- |
| 0.114 | 0.321 | 0.250 | 0.857 | 30.5 / 100 |

Coverage counts a ground-truth pair as captured when at least half of one instance lies inside the
highlights (0.469 if any overlap counts). For reference, perfect action predictions with this selector
reach about 69 / 100, because the 15% budget and 4 s minimum cap coverage and F1.

## Tuning and evaluation protocol
Head parameters (`code/prism/config.py`) were chosen on cached features. EPIC parameters were accepted
only if they improved both participant halves (odd and even participant IDs) separately. FPVSum weights
and segment length were left at their defaults after a small sweep showed no gain. Label prompts were
written once and not tuned on the evaluation labels. No ground truth is read by the submitted code.

## Missing videos
`Scuba__JnCVbCL5Wg` and `Scuba_KD74Ywuy9ng` are private on YouTube and `Scuba_Um4kqB8bQBg` is unavailable,
so no outputs exist for them: we did not write outputs by hand for videos the system could not process.
Given the videos, `./run.sh` produces their outputs like any other.

## Data used
- EPIC-KITCHENS-100 **train** annotations (verb-noun co-occurrence prior). No validation
  annotations are used by the system.
- Pretrained checkpoints: see `weights/SOURCES.md`. Vendored model code: `code/third_party/` (LaViLa, MIT).
- FPVSum annotations are not used by the system.

## Reproducing outputs/
```bash
for f in /path/to/videos/*.mp4; do ./run.sh "$f" "outputs/$(basename "${f%.*}").json"; done
```
