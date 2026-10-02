# WA-JEPA official reproduction and untrained sparse interface

Status: smoke/all-ID/sparse profiling complete; **official dense full evaluation user-paused at9253/12146**. Baseline fd5fc5f;
Drive-JEPA baseline 578be6e unchanged. No training or learned selection.

Latest user decision: preserve WA and implement a small selective-future extension on the evaluated
Drive-JEPA planner. Seven own WA workers and CPU helpers were stopped; no automatic WA resume.
All9253saved scenes successful, raw SHA256 verified; partialPDMS91.102506 versus matched Drive89.019762.
This is not a full benchmark result. [Partial scene CSV/summary](../results/official_wa_jepa_reproduction/partial_navtest_at_drive_extension_20261002.json).
New local snapshot `outputs/official_wa_jepa_reproduction/paused_snapshot_before_drive_extension_20261002.tar.gz`,
SHA256 `6aed87ec3feea7ecb7c9813ab0855af3fea84c3036e21f5b894eadb14f71249b`.
Earlier interruption snapshots and sparse measurements are unchanged. Resume commands below are historical,
require a new explicit user request and must NOT be run for the Drive extension.

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
| Sampling/precision | YAML BF16, actual original agent **FP32/TF32** (no autocast); **12** denoising steps; official flow seed0 |

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

User initials `kjs`: worker launches use `/rhome/junseong/envs/kjs-wa-jepa-eval/bin/python`, a symlink
to the preserved Conda prefix. Initially this was for future launches only. After the user explicitly requested
pause/relaunch, all14UID/command/GPU/shard-verified workers received SIGINT at10:32KST. All562completed scenes
(failed0) were preserved; the same14shards resumed with the new label, skipping those records. Only any interrupted
scene is recomputed. No Conda relocation, dependency/model/precision/scorer/seed change. Original paths/PIDs remain
in the previous manifest entries and pause report; guard validates only live registered workers.
Post-resume check:590completed/28new, all562preserved, duplicate/failed0,14active workers; [shared record](../results/official_wa_jepa_reproduction/process_label_resume.json).

Harness `scripts/evaluate_official_wa_jepa.py`: source/size/SHA checks,4-view history completeness, cache
dataclass/scorer/config comparison and actual unpickling under official devkit; original agent/PDMS scorer calls.
Evaluation scheduling differs: sorted disjoint scene shards, one loaded agent/scorer per worker, incremental JSONL.
Harness also applies the pinned YAML `tf32: true` to CUDA global flags and fixes CPU threads/initialization seed;
the bare upstream NAVSIM launcher does not explicitly apply that TF32 setting. This execution-control difference
is recorded, not a post-score adjustment. No autocast or model parameter dtype conversion is added.
Upstream generator resets each call, so scene ordering does not change the official noise. Failed scenes explicit.
Preflight fixes first token from each of six sorted logs before inference. Full gate: smoke success, compatible
cache; extra disk≤50GiB. Initial single-GPU24h estimate gate was exceeded by measured25.4499h.
Before full launch, resource plan revised to≤30single-GPU-hours/≤18h two-GPU wall: authorized GPU0 has since
become free (we did not stop CARLA). Initial two-worker launch did not persist; no full scene result was lost.
Dedicated tmux then ran two GPU0 workers under a4-shard plan, saving32scenes. User requested more workers
and permitted45GB per GPU. Only these confirmed own PIDs2207310/2207311 were SIGINT-stopped; their32records
were retained and are reused under disjoint14-way modulo shards. Final plan **7workers per GPU,14total**.
Measured NVML usage≈39,880,491,008bytes per GPU (~39.88decimalGB), below45,000,000,000bytes; OOM0.
The eighth worker would risk the conservative decimal45GB limit when including CUDA context overhead.
Guard checks every10s, validates owned PID/UID/command/shard before any SIGINT, never stops foreign processes.
Before model loading free GPU≥12GiB/host≥24GiB is required; OOM stops the affected worker without precision/model sweeps.
CPU threads1 per worker; current host available RAM remains>300GiB. VRAM headroom does not imply proportional
throughput: dense predictor already saturates the GPU. Original two-GPU estimate≈12.7h is not a14x speed claim.

```bash
cd /rhome/junseong/PlanningAwareFuturePrediction
runtime/environments/wa_jepa_official_evaluation/bin/python scripts/evaluate_official_wa_jepa.py preflight
CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 LD_LIBRARY_PATH=/rhome/junseong/PlanningAwareFuturePrediction/runtime/environments/wa_jepa_official_evaluation/lib runtime/environments/wa_jepa_official_evaluation/bin/python scripts/evaluate_official_wa_jepa.py smoke
# Only after smoke/cost gate:
CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 LD_LIBRARY_PATH=/rhome/junseong/PlanningAwareFuturePrediction/runtime/environments/wa_jepa_official_evaluation/lib runtime/environments/wa_jepa_official_evaluation/bin/python scripts/evaluate_official_wa_jepa.py full
```

Original equivalent: `scripts/evaluation/run_navsim_pdms.sh` with pinned CONFIG, CHECKPOINT, VJEPA2_CKPT,
NAVSIM_ROOT, OPENSCENE_ROOT and METRIC_CACHE_PATH. Exact resolved configuration/import paths recorded.

## Registered sparse check and implemented interface

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
Smoke6/6 succeeded: finite[8,3]trajectories, scorer success, peak allocated3.440GiB/reserved4.980GiB,
scene total7.006–8.849s, mean≈7.543s. **Smoke scores are not a paper reproduction result.**
All787,692,035 parameters FP32 despite YAML dtypeBF16; original agent does not convert/autocast.
Strict PyTorch loading1162keys, missing/unexpected/shape errors0. Full evaluation running; no aggregate published yet.

## Native sparse results (six preregistered scenes only)

All-ID:6/6scenes,12steps each, raw canonical position embeddings/normalized positioner outputs/scene and ego
pre-states/predictor outputs/final ego trajectories **bitwise identical (atol=rtol=0)**. Diagnostic Euler post-state
recomputation equals actual next pre-state for11transitions; final post-state recomputed because original API
does not expose final scene state. Full raw evidence [all-ID JSON](../results/official_wa_jepa_reproduction/all_id_equivalence.json).
Repeated dense trajectories also match exactly. Parameters are one shared loaded set; temporary class substitution
uses the separate worktree and restores original classes after each call, avoiding double-model VRAM.

| Condition | Future/joint tokens | Predictor CUDA(s) | Whole inference(s) | Peak allocated(GiB) | Six-scene PDMS(%) | XY change from dense(m) |
|---|---:|---:|---:|---:|---:|---:|
| Original dense |8192/12296|6.413|6.694|3.440|96.641|0|
| Packed-all |8192/12296|6.431|6.713|3.455|96.641|0|
| Fixed lattice |2048/6152|1.811|2.078|3.400|96.207|0.344|
| Seeded random |2048/6152|1.809|2.064|3.400|96.353|0.393|

One warmup per condition/3repeats per scene (18timed trials each); same device1/weights/input/noise/steps/precision.
IDs generated from geometry or scene-token hash+seed29 only, with128spatial tubes per camera across all4future tubelets.
Hooks verify **2048future tokens enter projections/QKV/scene FFNs**, while context4096/ego8 remain unchanged.
This is actual skipped predictor work, not a loss mask or post-prediction planner bottleneck. Dense Gaussian noise
and future condition are still constructed before gathering, preserving canonical scene noise and subsequent ego RNG offset.

Predictor time reduces≈71.8%, whole inference≈69% (~3.2x), not a full-system training efficiency result.
Whole timing includes ID construction/transfer, packing, official preprocessing, encoder and12-step prediction,
but excludes preloaded sensor disk I/O and scorer. Predictor CUDA interval excludes CPU rule generation; future-pipeline
interval includes noise/packing/Euler. CUDA allocated peak barely drops because weights/encoder dominate.
**Reserved peak4.980GiB stays the same for all conditions due to allocator high-water carryover from dense**;
no physical reserved-VRAM reduction claim. Concurrent GPU0 full workers were active; GPU1 timing had no other GPU1 worker.
Six-scene PDMS is a small untrained distribution-shift diagnostic, **not paper reproduction, learned-selector performance,
or statistical evidence that random beats fixed**. No sparse full benchmark and no training executed.

Evidence: [cost summary](../results/official_wa_jepa_reproduction/sparse_cost_summary.json),
[72trial records](../results/official_wa_jepa_reproduction/sparse_cost_trials.json),
[upstream diff](../patches/wa_jepa_sparse_future_token_interface.patch).
Cost hook initially misclassified a scene-FFN time-modulation Linear as token FFN; only the hook filter was corrected.
Native patch unchanged, completed all-ID results reused for cost-only retry; no model/result tuning.

## User pause / preservation / later resume

2026-10-02 19:12KST: user requested GPUs0/1 for another researcher. Only own verified14evaluation workers
received SIGINT; own CPU guard/aggregate were also stopped after verifying UID/resolved script/subcommand.
GPU0/1 each25MiB, no own compute; dedicated tmux server has no sessions. No foreign process stopped.
8686/12146completed (71.513%),3460remaining, failed/duplicate0. All16scene JSONL SHA256 checks pass.
Original weights/Conda/metric cache/config/12steps/flowseed/14waypartition and earlier32records remain unchanged.
Per-scene files flush incrementally; only interrupted, not-yet-recorded scenes need recomputation.

Local immutable snapshot `outputs/official_wa_jepa_reproduction/paused_snapshot_20261002_1912.tar.gz`,
2885497bytes, SHA256 `b11b8ac6c34b7034d1ffd572f4a0300c9e76a3c60f6ac6760bc65ab70033f967`.
Includes scene rows/logs/worker inventory/config/pause inventory/source/checkpoint metadata, not large weights/cache.
Shared [pause state](../results/official_wa_jepa_reproduction/paused_evaluation_state.json) and
[backup metadata](../results/official_wa_jepa_reproduction/paused_snapshot_backup.json).

**Do not resume automatically.** Pause marker blocks both launcher and direct full/smoke. Explicit user resume
and renewed physical GPU allocation are required. GPUutil0 alone does not establish availability.
The first pause invocation stopped workers but bookkeeping hit an argparse/command variable-name collision;
names were separated, inventory recovered while all workers were already stopped, and all saved files verified.
The recovered inventory therefore correctly reports0workers newly stopped in its recovery invocation.

### Shared-GPU bounded resume, 2026-10-02 20:40 KST

The user explicitly requested fewer workers on GPUs0/1 while other users remain active.
Verified all16scene-file SHA256 values and the unchanged official evaluation configuration before resume.
At admission, GPU0/1 free memory was25539/18834MiB. The queue caps concurrency at2workers per GPU (4total),
retains all14original shards, and skips completed8686scenes. Only unfinished3460scenes remain.
Camera/preprocessing/checkpoint/source/FP32/TF32/12steps/noise seed/scorer are unchanged. No training.

New scheduling-only profile: `configs/official_wa_jepa/shared_gpu_resume_v1.json`.
Admission requires12GiB free minus reservations for workers still loading (estimated6GiB peak per worker),
and24GiB available host RAM. Running reserve6GiB; CPUguard polls every5s and interrupts only verified own
UID/script/shard/GPU workers. Pressure marker blocks further queue launches; failed/interrupted workers are not
retried automatically. Other users' sudden allocations can still cause OOM; this is precaution, not isolation.
Launcher manifest updates are atomic to avoid partial JSON reads by the guard.

Shared [resume metadata](../results/official_wa_jepa_reproduction/shared_gpu_resume_state.json).
Runtime queue state `outputs/official_wa_jepa_reproduction/bounded_scheduler_status.json`;
logs `bounded_scheduler.log`, `memory_guard_shared_gpu.log`, `aggregate_shared_gpu.log`.
Six new CPU scheduling checks and the three existing pause-contract checks passed before GPU resume.

21:03KST user then requested more GPU0 workers. GPU0 limit2→5, GPU1 remains2;
new `shared_gpu_resume_v2.json`. Only the CPU scheduler was replaced; it adopted existing4workers,
then launched registered shards4/6/8 onGPU0. No GPU worker or foreign process was interrupted.
Guard running reserve/cap unchanged. Do not run the older all-seven-per-GPU launcher on shared cards.
[Module profiling](wa_jepa_inference_module_timing.md) separately confirmed predictor as main bottleneck.
The short diagnostic process exited; remaining evaluation is unchanged.

Use these commands only for a later explicitly authorized resume (the current sessions are already running):

```bash
cd /rhome/junseong/PlanningAwareFuturePrediction
/rhome/junseong/envs/kjs-wa-jepa-eval/bin/python scripts/pause_official_wa_jepa_workers.py --verify-only
nvidia-smi
# Start the memory guard before workers; guard/evaluator need the dedicated Conda library path.
tmux -L planning-aware-wa-jepa new-session -d -s official_wa_jepa_memory_guard 'cd /rhome/junseong/PlanningAwareFuturePrediction && exec env LD_LIBRARY_PATH=/rhome/junseong/PlanningAwareFuturePrediction/runtime/environments/wa_jepa_official_evaluation/lib /rhome/junseong/envs/kjs-wa-jepa-eval/bin/python -u scripts/guard_official_wa_jepa_memory.py --safety-profile configs/official_wa_jepa/shared_gpu_resume_v1.json >> outputs/official_wa_jepa_reproduction/memory_guard_shared_gpu.log 2>&1'
tmux -L planning-aware-wa-jepa new-session -d -s official_wa_jepa_bounded_scheduler 'cd /rhome/junseong/PlanningAwareFuturePrediction && exec /rhome/junseong/envs/kjs-wa-jepa-eval/bin/python -u scripts/schedule_official_wa_jepa_workers.py --safety-profile configs/official_wa_jepa/shared_gpu_resume_v2.json --resume-user-paused >> outputs/official_wa_jepa_reproduction/bounded_scheduler_gpu0_increase.log 2>&1'
tmux -L planning-aware-wa-jepa new-session -d -s official_wa_jepa_aggregate 'cd /rhome/junseong/PlanningAwareFuturePrediction && exec env LD_LIBRARY_PATH=/rhome/junseong/PlanningAwareFuturePrediction/runtime/environments/wa_jepa_official_evaluation/lib /rhome/junseong/envs/kjs-wa-jepa-eval/bin/python -u scripts/evaluate_official_wa_jepa.py wait-and-aggregate >> outputs/official_wa_jepa_reproduction/aggregate.log 2>&1'
```

Launcher skips active named sessions, appends logs and reuses scene records. Do not manually change the shard count
while workers run. `outputs/official_wa_jepa_reproduction/parallel_worker_manifest.json` holds exact owned PIDs/sessions.
All14shards must complete; aggregation asserts exact12146tokens/no omissions, extra IDs or duplicates and lists failures.
Partial scores are not published as full benchmark numbers. Existing Drive/pilot/results untouched.
On completion, review `results/official_wa_jepa_reproduction/full_navtest_results.json` and scene CSV, compare all
submetrics to Table3, audit any difference in checkpoint/split/scorer/preprocessing/noise/precision order, then commit/push.
Do not repeat the benchmark to match91.8 or auto-resume pilot/selector learning.

## Selector design: unresolved, no learned selector implemented

Potential native gradient graph: current history patches+ego status/intent → policy scores → ST positional
dictionary mixture → packed scene hidden/QKV → ego trajectory flow-MSE. Gradient could reach policy parameters
through continuous position tensors; PDMS is not differentiable training loss. Hard IDs still gather noisy scene state
and detached teacher targets, so this surrogate **does not differentiate their replacement when a selected ID changes**.
It is biased and may learn position shortcuts, not useful future information. Gradient existence is insufficient.

Minimum next-stage check (requires separate training approval): frozen official encoder/teacher and native WA
predictor, real train-only windows, fixed quota/noise, verify autograd boundaries then compare position-mediated
policy updates to random/fixed IDs on a small preregistered dev split. Track actual selected-ID changes, planning
flow loss and independent rollout scores, paired seeds, future-target/noise identities, and whether gains survive
held-out selection tests. Use native WA modules, not a structurally different toy task. Failure demands revisiting
surrogate design, not declaring the research hypothesis false or silently adding soft GT routing/new predictors.

Original training's flow-interpolated future GT is training supervision; it must not become selector input.
Online selector sees only current/history/ego intent. Official training adapter's future-derived command fallback
must be disabled explicitly in a future training adapter. Scene auxiliary policy gradient blocking needs an audited
separate detached-selection call; shared context/time/predictor parameters and single-forward scene_out exceptions
remain distinguished. No full native training-gradient/selection-learning evidence has been produced in this task.

## Four requested questions at this stage

1. Public original WA runs? **Yes, strict6-scene/scorer smoke passed; full navtest resumed from8686/12146, nowGPU0:5/GPU1:2 bounded workers.**
2. All tokens same as original? **Yes,6-scene native bitwise equality,12steps.**
3. Sparse reduces compute? **Yes, QKV/FFN work actually shrinks and six-scene latency≈69% lower; VRAM gain small.**
4. Selector learning established? **No. Position-only ST utility/content-replacement surrogate, GT/command boundaries,
   real native training and independent planning validation remain open.**
