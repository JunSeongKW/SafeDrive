# WA-JEPA inference bottleneck — measured module boundaries

Baseline source `404d8afd4f5e3334fe4b15a6ca7587b98cc35243`, official FP32/TF32,12sampling steps.
This diagnoses inference cost; no model/perception/training/scoring settings were tuned.

## Main finding

**The joint future/ego predictor is the bottleneck, not the encoder or ID packing.**
Earlier isolated-GPU six-scene/18dense trials: predictor6.41256s versus agent6.69444s,95.79%.
New shared-GPU0 one-scene probe: encoder0.54474s, predictor24.10485s/12calls,
model24.69828s,97.60% in predictor. Mean actual predictor step2.00874s (range1.73688–2.24856s).
The encoder runs once over the multiview context; the12-block predictor is called12times.
Dense tokens: current4096 + future8192 + ego8 =12296. Memory capacity is not a measure of compute cost.

## New direct measurements (seconds)

| Boundary | Original dense | Packed-all | Fixed sparse |
|---|---:|---:|---:|
| Future tokens |8192|8192|2048|
| Encoder CUDA elapsed |0.544744|0.871273|0.943929|
| Joint predictor CUDA elapsed, sum12steps |24.104851|43.036248|13.195149|
| Predictor mean per sampling step |2.008738|3.586354|1.099596|
| ID selection CPU wall, transfer/sync included |not used|0.009062|0.009918|
| ID validation/transfer/two gathers CUDA elapsed |not used|0.033806|0.027957|
| Whole model wall, no preprocessing |24.698277|44.029279|14.235846|
| Official preprocessing + transfer wall |0.157270|0.240458|0.215388|
| Whole agent wall, selection/preprocessing/model |24.855954|44.279092|14.461350|

**These columns were measured at different shared-GPU concurrency and are not a controlled speedup comparison.**
GPU0 evaluation concurrency increased2→5 while profiling. CUDA event intervals include other-process
competition and CPU launch gaps, not just kernel compute. One fixed preflight scene, one warmup and one
timed call per condition. No JPEG disk I/O or official scorer in these timings. The diagnostic process has exited.
Original dense has no sparse future-ID selector/packing. Dense's near-zero no-op selection timing is not
a learned-selector cost. Canonical positional lookup is inside predictor time, not the pre-predictor packing row.

For stable standalone cost, reuse the earlier controlled six-scene results:
dense total6.69444s/predictor6.41256s; fixed sparse total2.07849s/predictor1.81117s.
Those untrained sparse measurements do not establish learned-selector performance.

## How measured and what remained unchanged

- Encoder and predictor use forward pre/post CUDA events;12individual predictor intervals are saved.
- Model call and preprocessing/transfer use synchronized wall-clock boundaries.
- The existing sparse packing `if future_token_ids is not None` block is wrapped in an in-memory AST
  timing context only; its validation/gather math is unchanged. No tracked official source file was edited.
- Dense/packed-all final trajectories were bitwise identical, including the reused dense reference.
- Initialization metadata was redirected to the diagnostic directory, not the evaluation result paths.
- First attempt completed dense timing but packing instrumentation used `no_grad` decorator wrapper globals
  and raised `NameError`. Fixed with `inspect.unwrap`; dense measurements were preserved/reused, not rerun.
  This was a diagnostic-tool error, not a failed official evaluation scene. Two CPU wrapper tests passed.
- No scorer invocation, checkpoint download, training, precision switch or sampling-step change.

## Reproduction and evidence

```bash
cd /rhome/junseong/PlanningAwareFuturePrediction
# Only after checking GPU0 free>=12GiB and shared allocation; choose a fresh output path.
timeout 900 env CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  LD_LIBRARY_PATH=/rhome/junseong/PlanningAwareFuturePrediction/runtime/environments/wa_jepa_official_evaluation/lib \
  /rhome/junseong/envs/kjs-wa-jepa-eval/bin/python -u scripts/profile_official_wa_jepa_modules.py \
  --output-directory outputs/official_wa_jepa_reproduction/module_profile_another_authorized_run
```

Shared [summary JSON](../results/official_wa_jepa_reproduction/module_timing_summary.json),
[diagnostic script](../scripts/profile_official_wa_jepa_modules.py),
[CPU tests](../tests/test_wa_jepa_module_profiler.py),
[earlier isolated cost](../results/official_wa_jepa_reproduction/sparse_cost_summary.json).
Local raw record `outputs/official_wa_jepa_reproduction/module_profile_shared_gpu_packing_v2/module_timing_results.json`;
raw SHA and source hash are in the shared summary. Every step's timing is available there.
