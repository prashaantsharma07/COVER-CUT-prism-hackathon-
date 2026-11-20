# CoverCut: Action-Coverage Summarization for First-Person Video

**Samsung PRISM XR Hackathon 2026 · Track 04: Action-Aware Video Summarization**
Thapar Institute of Engineering and Technology, Patiala

CoverCut watches a first-person video, works out *what the wearer did* (verb + noun actions), and cuts a
highlight reel of at most 15% of the video that covers as many **different** actions as possible.

```bash
./TEAM_ID/run.sh video.mp4 out.json     # one video in, one JSON out, fully offline
```

![Architecture](docs/architecture.png)

## Results

Measured with our re-implementation of the four official metrics on all 184 videos available to us
(138 EPIC-KITCHENS-100 kitchen videos + 46 FPVSum sports videos). Full report: [`docs/score_report.txt`](docs/score_report.txt).

| Metric | Weight | Naive baseline | **CoverCut** | Ceiling* |
| --- | --- | --- | --- | --- |
| Action mAP (tIoU 0.1–0.5) | 25% | 0.000 | **0.114** | 1.000 |
| Action Coverage | 40% | 0.163 | **0.321** | 0.502 |
| Highlight F1 | 25% | 0.131 | **0.250** | 0.568 |
| Label Accuracy | 10% | 0.000 | **0.857** | 1.000 |
| **Weighted total** | | 9.8 | **30.5 / 100** | ≈ 69 |

\*Ceiling = ground-truth actions fed through our own selector. The 15% budget and the 4 s minimum clip length
mean **no system can score 100**: even perfect action predictions top out near 69. Against that ceiling,
CoverCut reaches **44% of the achievable score**, and **64% of the achievable coverage**, the largest metric.

What the numbers say:

- **3.1× the baseline** on the weighted total, from one forward pass per video.
- **Coverage nearly doubles** (0.163 → 0.321) because we optimise the metric itself: under the
  "any overlap" reading of the rule it is **0.469**.
- **Highlight F1 is ~2× random** (0.25 vs ≈0.13) with zero training on FPVSum.
- **Labels: 85.7% zero-shot** across 14 sports, from pixels only. Every one of the 138 kitchen videos is
  routed correctly to the kitchen pipeline.
- **The action model behaves as published:** spot check on held-out EPIC clips gave verb 60.6% / noun 64.8%
  top-1 (published: 69.0 / 58.4).

### Scale of the run

- **187 videos, 15.6 hours of footage, ~190 GB of 1080p source video**, processed end to end on **one free
  Kaggle Tesla T4 in 6.3 hours**.
- Our parallel downloader ([`tools/epic_fast_dl.py`](tools/epic_fast_dl.py)) took the EPIC download from
  **~0.5 MB/s to ~9 MB/s per video (8 connections, 4 videos in flight)**, which turned a ~100-hour
  download into ~6 hours.
- Every output is produced by `run.sh` from the original video; nothing is hand-edited.

## The key idea

65% of the score is the kitchen videos, and the largest metric (**Action Coverage, 40%**) counts how many
*distinct* ground-truth verb–noun pairs the highlights contain. A second clip of "wash plate" earns nothing.
So highlight selection is a **budgeted maximum-coverage** problem, not importance ranking:

```
F(S) = Σ_pairs [ 1 − Π_{w ∈ S} (1 − q(pair, w)) ]      subject to   Σ|w| ≤ 0.15·T,  |w| ≥ 4 s
```

`F` is monotone submodular, so greedy selection by marginal gain per second is within **(1 − 1/e)** of
optimal. Implementation: [`TEAM_ID/code/prism/coverage.py`](TEAM_ID/code/prism/coverage.py).

## Pipeline

1. **Decode once** (ffmpeg, 8 fps, 224 px), streamed; no frames written to disk.
2. **Feature cache**, one GPU pass:
   - LaViLa **TimeSformer-B fine-tuned on EPIC-KITCHENS-100** (train split only, verified from the
     checkpoint's own metadata): verb and noun probabilities on sliding 2 s and 4 s windows.
   - **SigLIP** image embeddings at 1 fps.
   - Motion energy per second.
3. **Four heads** (pure NumPy, re-run from the cache in seconds):
   - *Label*: zero-shot SigLIP prompt ensemble; also routes kitchen vs sport.
   - *Actions*: verb × noun scores with a verb–noun co-occurrence prior from the EPIC **train** split
     (removes impossible pairs like "cut fridge"), merged over time.
   - *Kitchen highlights*: the coverage selector above.
   - *Sports highlights*: motion + semantic "exciting moment" + scene-change importance, non-redundant
     ≥ 4 s clips.
4. **Constraint-safe writer**: every JSON is valid by construction (vocabulary, ranges, sorting,
   non-overlap, ≥ 4 s, ≤ 15%). If a model fails to load, a valid fallback is still written.

## How we tuned without fooling ourselves

All tuning ran on the cached features. A change was **kept only if it improved both halves of the kitchen
participants separately** (odd vs even participant IDs):

| Change | Half A | Half B | Decision |
| --- | --- | --- | --- |
| Merge same-pair detections within 3 s, score² | **+0.96 pts** | **+0.72 pts** | kept |
| Temporal smoothing of probabilities | worse | worse | rejected |
| Frequency-weighted pair prior | worse mAP | worse mAP | rejected |
| Rewritten label prompts (one attempt only) | 41/46 vs 42/46 | | rejected |
| Sports importance weights | plateau (±0.01) | | defaults kept |

No ground-truth file is read by the submitted code (`tools/integrity_check.sh` enforces this).

## Repository layout

```
TEAM_ID/                 the submission package
  run.sh                 ./run.sh <video.mp4> <out.json>
  code/prism/            pipeline (features, heads, coverage selector, writer)
  code/third_party/      vendored LaViLa TimeSformer (MIT)
  outputs/               184 validated JSON outputs
  weights/               filled by tools/fetch_weights.sh (~1 GB, not stored in git)
  Dockerfile, requirements.txt, README.md
tools/                   team tooling: downloads, scorer, tuning, Kaggle pipeline, integrity check
kaggle/                  the notebook used for the full GPU run
data/manifest.csv        the official video list
docs/                    architecture figure, score report, workspace guide
```

## Reproduce

```bash
bash tools/fetch_weights.sh                          # LaViLa TimeSformer-B + SigLIP into TEAM_ID/weights
./TEAM_ID/run.sh /path/to/P28_20.MP4 out.json        # any manifest video
```

Full run on a free GPU: [`kaggle/prism_kaggle.ipynb`](kaggle/prism_kaggle.ipynb) (instructions in
[`docs/WORKSPACE.md`](docs/WORKSPACE.md)). Score locally with `python tools/score.py --pred TEAM_ID/outputs`.

## Limitations

- **Recognition, not detection:** boundaries come from sliding windows. A temporal detector such as
  ActionFormer is the clear next step for Action mAP.
- **Sports highlights use generic signals:** no public training data matches FPVSum's annotation.
- **Label confusions:** plane vs skydiving, bike vs motorbike / bike polo (4 of 46 errors).
- **3 of 187 videos are unavailable:** `Scuba__JnCVbCL5Wg` and `Scuba_KD74Ywuy9ng` are private on YouTube
  and `Scuba_Um4kqB8bQBg` is removed, so 184 outputs are submitted. Given the files, `run.sh` processes them
  like any other.

## Credits

LaViLa (Zhao et al., CVPR 2023, MIT license) for the EPIC-100 action model; SigLIP (Zhai et al., 2023);
EPIC-KITCHENS-100 (Damen et al.); FPVSum (Ho et al., ECCV 2018).
