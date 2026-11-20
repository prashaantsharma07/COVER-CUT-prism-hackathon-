# PRISM 2026 Track 04: team workspace

`GOLDEN_DAWN/` is the submission package (team ID already set).
`tools/`, `data/`, `ext/` are team-only (downloads, scoring, ground truth). Never copy them into `GOLDEN_DAWN/`.

## Do these in order
```bash
bash setup.sh          # 1. python env, annotation repos, machine report, starts downloads in background
bash setup_gpu.sh      # 2. PyTorch + model weights (~1.5 GB) + accuracy check of the action model
```
If `setup.sh` says system packages are missing, run the one `sudo apt install ...` line it prints, then re-run it.
Every new terminal: `cd` here and `source .venv/bin/activate`.

## 3. Run the models on Kaggle (free T4 GPU, fast download)
Your laptop has no usable GPU driver and downloads at ~0.8 MB/s (EPIC would take ~60 h), so the heavy run is on Kaggle.
1. kaggle.com: sign in, verify your phone (Settings) so GPU and Internet unlock.
2. Datasets -> New Dataset -> upload `prism_v3.zip` -> title `prism-code` -> Create.
3. Code -> New Notebook -> File -> Import Notebook -> `kaggle/prism_kaggle.ipynb`.
4. Right sidebar: Add Input -> your `prism-code` dataset. Settings: Accelerator `GPU T4 x2`, Internet `On`.
5. Save Version -> `Save & Run All (Commit)`. You can close the browser. Watch the log under the version's Logs.
6. When it finishes: Output tab -> download `prism_results.zip` -> send it to Claude.
   Not finished in 11 h? Open the notebook, Add Input -> Notebook Output -> the previous version, Save & Run All again: it resumes.
It stops early by itself if the action model's accuracy check fails; send `pipeline.log` in that case.
FPVSum videos are downloaded on Kaggle with yt-dlp; if YouTube blocks it, upload the folder your laptop downloaded
(`videos/fpvsum`, 480p) as a second dataset, add it as input and run again.

## 3b. Running on your own GPU instead
```bash
export PRISM_CACHE=$PWD/work/cache
bash tools/run_all.sh work/outputs videos/epic_256 videos/epic_raw videos/fpvsum
```
A `BASELINE-FALLBACK` line means the models did not load for that video (valid but naive JSON).

## 4. Tune the heads from the cache (seconds, no GPU)
```bash
python tools/run_heads.py prism_results/cache work/try --set COVER_RULE=any --set COVER_MODE=center
python tools/score.py --pred work/try --split A      # tune on A (odd participants) ...
python tools/score.py --pred work/try --split B      # ... confirm on B
```
Settable: `ACT_TOPK`, `COVER_RULE` (any|half|full), `COVER_MODE` (span|center), `COVER_MAX_LEN`,
`FPV_WEIGHTS` (motion, semantic, event), `FPV_SEG_LEN`, `SCALES`. Put the winners in `GOLDEN_DAWN/code/prism/config.py`.

## 5. Before submitting
```bash
export PRISM_TEAM_ID=GOLDEN_DAWN
rm -rf GOLDEN_DAWN/outputs && bash tools/run_all.sh GOLDEN_DAWN/outputs videos/epic_256 videos/epic_raw videos/fpvsum
python validate.py --dir GOLDEN_DAWN/outputs/          # official validator from the student kit
bash tools/integrity_check.sh
(cd GOLDEN_DAWN && pip freeze > requirements.txt)
```

## What is tested and what is not (13 Nov)
- Tested: writer/validator on all 187 videos, scorer (oracle = 1.0 mAP), coverage selector, action head,
  feature windowing and caching (with stand-in models), fallback path, setup.sh from a clean copy.
- NOT yet run on real weights: `prism/models.py` (LaViLa TimeSformer + SigLIP loading). `setup_gpu.sh`
  ends with an accuracy check (expect verb 65-70%, noun 55-60%). Do not extract features until it passes.

## Scorer assumptions (confirm with organizers, then edit tools/score.py)
- Action mAP classes are (verb, noun) pairs; verb-only and noun-only mAP are printed too.
- Coverage: a GT pair counts when one of its instances is captured by highlights >= 4 s.
  `--coverage-rule any|half|full`; default `half`. Penalty above 15%: x 0.15 / used fraction.
- Highlight F1: GT = top 15% of frames by consensus score, ties broken by frame order.

## Final results (20 Nov)
Kaggle full run (T4, 6.3 h): 184/187 videos, 29.72. After tuning on the cache: **30.53** (half rule),
36.44 (any-overlap rule), 24.38 (full containment). Final outputs: `GOLDEN_DAWN/outputs/` (184 files).
Rebuild them: `python tools/finalize_from_cache.py <prism_results> GOLDEN_DAWN/outputs`. Tuning: `tools/tune.py`.

## Reference numbers
| Predictions | Action mAP | Coverage (half) | Highlight F1 | Label | Total |
| --- | --- | --- | --- | --- | --- |
| Phase 1 baseline (even 4.5 s windows) | 0.000 | 0.163 | 0.131 | 0.000 | 9.78 |
| Oracle: GT actions + coverage selector | 1.000 | 0.502 | 0.568 | 1.000 | 69.29 |

## Low-disk note
With under 210 GB free, `setup.sh` downloads EPIC one video at a time, keeps a 256 px copy and deletes the
original. Outputs are then produced from the 256 px copies; say so in the submission README, because
organizers re-running on originals will get near-identical but not bit-identical results.
