# WA-JEPA official reproduction and untrained sparse interface

Status: preflight passed; GPU smoke in progress, **not a full evaluation result**. Baseline fd5fc5f;
Drive-JEPA baseline 578be6e unchanged. No training or learned selection.

## Official target fixed before inference scores

| Item | Pinned target / evidence |
|---|---|
| WA source | Official `404d8afd4f5e3334fe4b15a6ca7587b98cc35243` |
| Preserved latest reference | `bec29660f5ea46ac73db8e2d0c33c8d1a72c23ad`, untouched |
| Planning artifact | [AFARI-Research/WA-JEPA](https://huggingface.co/AFARI-Research/WA-JEPA), revision `15c0770ebd233665214590bb2190a90907499f9e`, model_state_dict.pt + state.pt |
| Encoder | [Meta V-JEPA2.1 ViT-L](https://dl.fbaipublicfiles.com/vjepa2/vjepa2_1_vitl_dist_vitG_384.pt), separate initialization weight, not a planning checkpoint |
| Protocol | NAVSIM v1.1 navtest PDMS; devkit `3e8291bfa89ff247231e0227778840cd0a036896` |
| Paper row | [Table3](https://arxiv.org/html/2608.20974v1): NC99.5, DAC98.3, TTC97.7, Comfort100, EP85.0, PDMS91.8 |
| Input | [cam_l0,cam_f0,cam_r0,cam_b0], history4/future8 at0.5s,256×512 |
| Sampling/precision | original BF16/TF32; **12** denoising steps; official flow seed0 |

Latest official commit changes only five presets12→4. Published state.pt and paper AppendixC specify12.
We therefore pin the earlier **official**12-step preset before any scores, not a tuning override.
Paper/README report10-seed means; exact10-seed list is unconfirmed. Our single preset seed is not that
average, and no reproduction tolerance is invented. Official installation uses the same stage2 artifact for
v1 PDMS/v2 EPDMS. We run only v1 and do not compare its score to v2 EPDMS91.7.
Published checkpoint config uses navtest validation and best metric pdms, not a newly independent held-out.
State metadata does not contain original run-specific scores; that correspondence remains documentation-based.

## Isolation and gates

New dedicated Conda prefix `runtime/environments/wa_jepa_official_evaluation` cloned from the known-working
Drive Conda, with changes confined to the new prefix. Python3.9.23 meets source>=3.9 (guide recommends3.10),
Torch2.1.0+cu121 retained. Official WA worktree and NAVSIM checkout are separate. No pilot ROI, planner,
predictor or image rectification imported. Downloads/derived outputs are under the workspace, public dataset RO.
The cloned Conda ICU requires its own libstdc++; use an execution-local `LD_LIBRARY_PATH` pointing to
that prefix/lib. No system libraries or other environments changed. NAVSIM's dependency metadata pins
Torch2.0.1/torchvision0.15.2/numpy1.23.4; this runtime keeps WA-compatible Torch2.1/torchvision0.16/numpy1.24.
`pip check` reports these declared-version conflicts and unused notebook/selenium/tensorboard absences.
These are documented environment deviations, not suppressed successful dependency resolution.

Harness `scripts/evaluate_official_wa_jepa.py`: source/size/SHA checks,4-view history completeness, cache
dataclass/scorer/config comparison and actual unpickling under official devkit; original agent/PDMS scorer calls.
Only evaluation scheduling differs: sorted sequential scenes, one loaded agent/scorer, incremental JSONL records.
Upstream generator resets each call, so scene ordering does not change the official noise. Failed scenes explicit.
Preflight fixes first token from each of six sorted logs before inference. Full gate: smoke success, compatible
cache, measured extrapolation≤24single-GPU-hours; extra disk≤50GiB. GPU1 first; foreign GPU0 CARLA untouched.

```bash
cd /rhome/junseong/PlanningAwareFuturePrediction
runtime/environments/wa_jepa_official_evaluation/bin/python scripts/evaluate_official_wa_jepa.py preflight
CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 LD_LIBRARY_PATH=/rhome/junseong/PlanningAwareFuturePrediction/runtime/environments/wa_jepa_official_evaluation/lib runtime/environments/wa_jepa_official_evaluation/bin/python scripts/evaluate_official_wa_jepa.py smoke
# Only after smoke/cost gate:
CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 LD_LIBRARY_PATH=/rhome/junseong/PlanningAwareFuturePrediction/runtime/environments/wa_jepa_official_evaluation/lib runtime/environments/wa_jepa_official_evaluation/bin/python scripts/evaluate_official_wa_jepa.py full
```

Original equivalent: `scripts/evaluation/run_navsim_pdms.sh` with pinned CONFIG, CHECKPOINT, VJEPA2_CKPT,
NAVSIM_ROOT, OPENSCENE_ROOT and METRIC_CACHE_PATH. Exact resolved configuration/import paths recorded.

## Registered sparse check, not implemented yet

Canonical camera→future tubelet→spatial IDs, not object IDs. Dense context4096/future8192/ego8.
128spatial tubes per camera×4future tubelets→future2048, all currentcontext/ego retained. Pack before
predictor projection/QKV/attention/FFN. Original source preserved; separate sparse patch after smoke passes.
All-ID requires **bitwise equality** of positions/noises/predictor outputs/per-step states/final trajectory;
ordering, shapes, dtype, weights and kernels are unchanged. Investigate mismatch rather than loosen tolerance.
Fixed lattice/random29: same six scenes, one warmup/three timed repeats each, CUDA sync, packing included.
No learned selector and no sparse full benchmark. Untrained removal is distribution-shift sensitivity, not H1/H2 proof.

Position-mediated ST only differentiates positional tensor mixing/scores; hard noisy-state/GT target replacement
is not approximated. Positional gradient does not establish useful native WA selection learning. Future training
GT flow inputs are distinct from online selector inputs; future-derived command fallback must later be blocked.

## Results

Preflight: expected/loaded12146, missing/extra/cache/image failures0;71,488 unique4-view history image files.
MetricCache dataclass/PDM scorer/proposal config byte-identical to the preserved v1 evaluation and actual
sample cache unpickles under new official NAVSIM. All registered checkpoint fields match, including12steps.
HF model/state published SHA match; Meta encoder measured SHA256
`7ea9b7cb4a75d10644a8a8d42cff9e177b10dca8f02173f0eaf2b0bed82838c6` (no public SHA claimed).
5CPU provenance/protocol tests passed. Initial smoke startup stopped before inference due to process-local
ICU C++ library lookup, then a Path/mmap harness compatibility issue (fixed using string filename).
Upstream agent already reported complete checkpoint key/shape match; original code unchanged.
Smoke/full results pending. Missing/failed gates will be reported explicitly.
