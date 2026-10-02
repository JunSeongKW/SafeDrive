# Drive-JEPA selective future extension — connection experiment v1

## Question and scope fixed before execution

Can a small learned patch-target branch receive planning gradient through the **existing official
Drive-JEPA planner**, while disabling the branch preserves the evaluated model? This is an implementation
gate, not evidence for better planning, useful target selection, future-prediction accuracy or novelty.
WA-JEPA is user-paused; its partial scores, raw scenes and sparse cost/equality evidence are preserved.
Pilot tuning, data expansion, adaptive budgets, full evaluation and training sweeps remain out of scope.

Baseline: official `548bb8215e3aae18e162a0f12f1ba83b4d3eb57e`, full PF ViT-L planning checkpoint
SHA256 `982960aed9e82900584d20c676ba9a7498add912bed94a3e7fa5f4d22bff6237`.
The existing strict-encoder-loading patch is retained and recorded, not silently reverted.
No original source, weights, Conda packages or official evaluation results are modified.

## Calculation graph

```text
official two front images + current ego status/command
  -> frozen FULL planning encoder: [batch,512,1024], 16x32 camera grid
  -> learned current-content/position/ego scorer -> unique hard K4, soft-backward ST
  -> selected current content AND position -> lightweight predictor [batch,4,4,1024]
  -> future memory -> cross-attention residual over original 128 image memories
  -> existing status memory + existing positional embeddings
  -> original transformer AND original trajectory head -> [batch,8,3]

training-only future image pairs -> same frozen encoder -> [batch,4,512,1024]
  -> DETACHED hard-ID target gather -> auxiliary MSE
```

The branch-off path calls the original model directly. Branch-on uses the same normalization,
encoder, pooling, image/status projections, positional embeddings, transformer and trajectory head.
It adds no trajectory decoder. The residual output projection is zero-initialized: at update0,
branch-on must also equal the original trajectory bitwise in deterministic execution.
At exact zero initialization planning gradient reaches the residual output projection but not earlier
branch parameters. One train-only diagnostic optimizer step opens this path; then selector and predictor
planning gradients must be finite/nonzero. This is not yet evidence that hard patch choices learn well.

## Target and selection semantics

- Candidates are 512 **front-camera image patch positions**, not entities, BEV cells or object tracks.
- Current clip uses official history indices2/3, crop28, resize512x256, official RGB normalization.
- Four future tubelets use frame pairs(4,5),(6,7),(8,9),(10,11), ending at+1,+2,+3,+4seconds.
- Target uses the full planning checkpoint encoder, frozen/eval/stop-gradient. It is not a copied
  pretraining-only encoder. Future input is never an argument of online `forward`.
- The same image-grid ID is supervised in each future tubelet. This is an Eulerian camera-grid target,
  **not the same physical object/world location**. Ego motion and object motion can change its content.
  No GT association, ROI, extra undistortion, future command inference or spatial-state head is used.
- Missing future images mask auxiliary loss only; they never change current scores/selection/planner input.
- Sequential ST excludes each hard-chosen patch from subsequent slots. The surrogate differentiates
  current feature mixing and selected coordinate mixing; it does not differentiate argmax, hard exclusion
  or changes in which future target is gathered. It is biased, not an optimal-selection guarantee.

## Gradient and preservation contract

| Signal/control | Selector | Predictor | Residual bridge | Original model |
|---|---|---|---|---|
| Planning, after opening zero residual | yes | yes | yes | frozen parameters; input autograd retained |
| Future auxiliary, separate detached-selection call | no | yes | no | no |
| Detach predicted future for planning | no | no | yes | frozen |
| Disable future branch | not called | not called | not called | original forward |

Frozen planner execution must not be wrapped in `no_grad` on the enabled path: its input Jacobian is needed.
Detach preserves forward values and changes backward only. Zero/swap are dependence diagnostics, not
equivalent to retraining a no-future model. Official length-normalized imitation loss is the planning signal;
PDMS is not treated as differentiable. Auxiliary target/mask are supplied only to the loss function.

## Minimum experiment and bounds

Config: `configs/drive_jepa_selective_future/connection_v1.json`.
Two sorted, distinct **official navtrain recording segments**, first nonoverlapping history4/future8 window;
selection fixed before model output. Navtest/navhard/private_test are excluded from fitting/diagnostic update.
Also require `train` in the preserved project native-recording3-way manifest, not just official navtrain membership.
Window admission uses current image existence, not future image validity. Future missingness is recorded.
Raw timestamps have sub-millisecond recording jitter (also observed in the completed official navtest
preflight). Preserve original frame order and actual offsets; use the existing adapter's0.05s cadence
tolerance for validation, not resampling. An initial exact-500000us diagnostic check stopped before model
forward and was corrected; the failed output directory is preserved. No windows changed to improve results.
Only one optimizer step, no predictor tuning/sweep/cache creation, one GPU0/1 process, sequential target
encoding to avoid multi-clip VRAM amplification. Admission12GiB free, peak allocated cap6GiB, wall900seconds.
Measure baseline, disabled, enabled forward latency with1warmup/5repeats, selection/predictor/bridge hooks,
and backward memory. Shared-GPU timings are not isolated deployment latency or benchmark speed claims.

Required checks: strict full weights; original module identities and before/after weight hashes; off/init-on
bitwise preservation; K unique current IDs; explicit input/target separation; auxiliary mask invariance;
planning/aux/detach gradients; actual Kx4 predictor input count; finite official trajectories; no training
outside the one diagnostic step. Synthetic CPU tests do not substitute for a real official-model batch.

Later performance experiments, **not run here**: same K random/fixed-rule/learned selection, same current
feature bridge versus future-supervised branch, all-patch reference and branch-off original. Use paired seeds,
train/dev/independent evaluation and official safety/progress metrics. Content gradients alone do not
establish discrete choice utility; useful choice changes must be tested on real train/dev batches next.

## Execution results

**Completed** on2026-10-02, GPU0, official real images from2distinct project-TRAIN navtrain recordings;
no navtest fitting/evaluation. Whole CPU suite103/103 passed (11new extension +2split tests), touched-code Ruff passed.
Original309,981,955parameters frozen;1,271,489new trainable parameters. One optimizer step updates
the zero-initialized bridge output projection only. Selector/predictor receive planning gradient afterward
but have NOT yet been optimized into useful future predictions or good discrete selections.

| Contract / loss | Selector gradient norm | Predictor | Bridge | Original model |
|---|---:|---:|---:|---:|
| Planning at zero residual |0|0|0.101464|0|
| Planning after1bridge diagnostic step |0.000009395|0.000010501|0.101464|0|
| Future auxiliary |0|1.598178|0|0|
| Detach predicted future |0|0|0.101464|0|
| Detach selection |0|0.000010501|0.101464|0|

Strict full planning loading: missing/unexpected0. Disabled outputs before/after step and enabled outputs
at update0 (same no-grad inference mode) are bitwise identical to original. Frozen original state hash before
and after matches. Future detach keeps forward trajectory identical; hard current IDs are unique.
Actual predictor queries `[2,4,4,2074]`, outputs `[2,4,4,1024]`, ego `[2,8,3]`; not512future queries.
Teacher target `[2,4,512,1024]` is dense **training-only**. Encoder still processes the whole scene.
Five of8future tubelets available:20selected patch-time targets supervised, other12masked only in loss.
The window missing its+2/+3/+4s images was retained; future availability did not filter current candidates.
Shared source input files and reference diff hashes are unchanged; existing environments/weights/results preserved.

| Forward timing,2window batch | Mean(s) | Peak allocated(GiB) |
|---|---:|---:|
| Official original |0.126048|1.255|
| Extension disabled |0.110916|1.255|
| Extension enabled |0.120065|1.261|

One warmup/5repeats, preprocessed inputs on GPU; excludes disk I/O, future teacher and scoring.
**Different intervals on a shared GPU: an enabled mean lower than original is NOT evidence of speedup.**
One enabled CUDA-hook trial: encoder118.530ms, selector0.232ms, predictor0.044ms, bridge2.439ms,
original transformer0.848ms. This is not an isolated latency/efficiency benchmark; reserved memory includes
allocator carryover and allocated memory does not include all CUDA-context/NVML usage.
Final train-only diagnostic wall45.176s includes loading/hashes/targets/tests; no OOM, process exited/GPU released.
Initial exact-cadence loader failure occurred before forward, preserved separately; it is not a failed research result.

Shared [result JSON](../results/drive_jepa_selective_future/connection_v1_20261002.json),
[implementation](../src/planning_aware_future_prediction/models/drive_jepa_selective_patch_future.py),
[validation runner](../scripts/validate_drive_jepa_selective_future_connection.py),
[CPU tests](../tests/test_drive_jepa_selective_patch_future.py).
Local `outputs/drive_jepa_selective_future/connection_v1c_train_split_20261002/` contains raw report, fixed pre-forward
window metadata, loading log and **diagnostic-only** extension weights after1step. No original weights copied into it.

### Split audit and preserved rejected diagnostic

The preceding v1b run passed the computational contracts but admitted official-navtrain recordings reserved
internally as held_out(`2021.05.12.19.36.12_veh-35`) and development(`2021.05.12.22.00.38_veh-35`).
It performed one bridge diagnostic update: this is an exposure, even though not benchmark performance evaluation.
Its raw logs/state and [rejected report](../results/drive_jepa_selective_future/connection_v1b_project_split_rejected_20261002.json)
remain preserved, and its weights were NEVER reused by the final train-only run. Selection now requires
the original project assignment==train before loading windows. The final run resets original checkpoint/new
branch seed29 and uses `2021.05.12.22.28.35_veh-35` and `2021.05.12.23.36.44_veh-35`, both project train.
[Exposure audit](../results/drive_jepa_selective_future/project_split_exposure_audit_20261002.json) records that
the exposed held-out recording must not be called untouched independent holdout for this extension.
The original manifest is preserved, not silently reassigned. A future final evaluation protocol must account for
this exposure; correcting the run cannot undo it. Two regression tests protect admission.

```bash
cd /rhome/junseong/PlanningAwareFuturePrediction
# CPU regression, does not run a GPU experiment:
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 PYTHONPATH=src \
  runtime/environments/future_prediction_cpu/bin/python -m pytest -q tests
# Only for a newly authorized replay, after GPU0 free-memory/allocation check; fresh output path required:
timeout 900 env CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  LD_LIBRARY_PATH=/rhome/junseong/PlanningAwareFuturePrediction/runtime/environments/drive_jepa_official_evaluation/lib \
  PYTHONPATH=/rhome/junseong/PlanningAwareFuturePrediction/src \
  /rhome/junseong/envs/kjs-drive-jepa-extension/bin/python -u \
  scripts/validate_drive_jepa_selective_future_connection.py \
  --output-directory outputs/drive_jepa_selective_future/another_authorized_connection_run
```

**Gate passed: original model preservation + actual official-planner autograd + bounded execution.**
Unproven: predictor's future accuracy, selector learning/discrete utility, planning improvement, equal-budget
benefit, independent holdout performance and novelty. Do not automatically resume WA/pilot or start training.
