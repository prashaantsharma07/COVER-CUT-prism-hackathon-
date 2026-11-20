# AI Disclosure

Team **GOLDEN_DAWN** (Prashaant Sharma, Varun Joshi, Yatharth Joshi), Thapar Institute of Engineering and
Technology, Patiala. Samsung PRISM XR Hackathon 2026, Track 04.

We used AI tools substantially in this project. This page says where.

## AI assistant used during development

**Claude (Anthropic)**, used as a coding and planning assistant throughout:

| Area | What the assistant did | What the team did |
| --- | --- | --- |
| Analysis and planning | Analysed the starter kit, metrics and rules; wrote the implementation plan; proposed the coverage-selector approach | Chose the track, reviewed and approved the plan, set priorities and deadlines |
| Code | Wrote most of the pipeline (`GOLDEN_DAWN/code/prism/`), the tools in `tools/`, the scorer and the Kaggle notebook | Ran setup, downloads and the full GPU run on Kaggle and our own machine; reported errors and logs; reviewed the code |
| Debugging | Diagnosed failures from the logs we supplied (slow downloads, notebook arguments, missing packages) and wrote fixes | Ran each fix and reported back |
| Tuning | Ran parameter sweeps on our cached features and applied the participant-split acceptance rule | Reviewed which changes were kept |
| Documentation | Drafted the README, this disclosure and the submission README | Reviewed and edited; supplied team details |
| Presentation | Wrote the slide content; the layout was generated with **Canva's AI design generator**, then corrected by the assistant | Chose the colour palette and typeface; added team and contact details; reviewed |
| Demo video | Wrote the script (`tools/`-style Python + ffmpeg) that renders the video from our real outputs | Reviewed the video |

## Pretrained models inside the system

These are part of the submitted pipeline, not development aids:

- **LaViLa TimeSformer-B**, fine-tuned on EPIC-KITCHENS-100 action recognition (Zhao et al., CVPR 2023,
  MIT license). Training split confirmed from the checkpoint's metadata: `EPIC_100_train.csv` only.
- **SigLIP base** (`google/siglip-base-patch16-224`, Apache-2.0), used zero-shot for labels and
  sports-highlight semantics.

## What is not AI-generated

- **Every output in `GOLDEN_DAWN/outputs/` was produced by `run.sh`** (the same pipeline, run on Kaggle) from
  the original videos. No output was written or edited by hand or by an assistant.
- **Every number in the README, slides and video was computed by `tools/score.py`** on those outputs.
- No ground-truth annotation is read by the submitted code (`tools/integrity_check.sh` checks this).
