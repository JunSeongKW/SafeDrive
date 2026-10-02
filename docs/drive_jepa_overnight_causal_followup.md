# Overnight cause analysis and controlled retraining

## Scope and stop conditions

Starting commit: `8159aad`. User authorized autonomous experiments through the
next morning, preserving the planning-aware future-selection question. Deadline:
**2026-10-03 09:00 KST**. Routine choices are recorded, not referred back to the user.
No WA restart, old GT/ROI pilot, held-out/navtest tuning, official checkpoint edits,
new dependencies or public dataset writes. GPU1, one process, at least 16 GiB free
at admission, 6 GiB running reserve and 8 GiB own allocated-memory cap. Memory
pressure/OOM stops the job with a recovery checkpoint; no automatic OOM restart.

This extension remains front-camera **patch** selection, K=4, four future tubelets.
It is neither instance selection nor proof of a paper contribution. The original
full Drive-JEPA planner/encoder remain frozen and preserved.

## Completed diagnosis (read-only existing checkpoints)

Reused 192 cache windows and two architectures × three seeds × updates 50/200.
149.06 seconds, 1.571 GiB peak allocated, official parameter hash unchanged.
Raw data: `outputs/drive_jepa_selective_future/learning_limitations_diagnosis_20261003/`.

| Development scene-macro XY ADE, m (3-seed mean) | MLP, update 200 | Ego-query residual, update 200 |
|---|---:|---:|
| Normal trained forward | 0.209975 | 0.216938 |
| Branch disabled (preserved baseline) | 0.220644 | 0.220644 |
| Half of trained memory residual | 0.210888 | 0.209855 |
| Replace predicted future with current feature | 0.205721 | 0.214940 |
| Replace available targets with future GT | 0.204675 | 0.209679 |

These are post-training interventions, not separately trained models. Future-GT
replacement is privileged/OOD and not a deployable oracle upper bound. Current
feature replacement also changes the input distribution. Neither proves that
future prediction is useless or beneficial. The half-gain observation motivates
a separately trained matched condition, not a tuned inference setting.

On eight recorded training batches per seed (24 per checkpoint), negative
predictor planning/auxiliary gradient cosine occurs in 62.5% of MLP and 54.2% of
ego-query batches at update 200. Mean cosine is −0.0203 / −0.0782; scaled auxiliary
gradient norm averages 0.344 / 0.110 of planning gradient norm. This is a local
optimization measurement, not proof of the full cause of poor generalization.
Official-IL versus uniform-ADE output-gradient cosine is about 0.63: objectives
are related but not equivalent. The official training loss is not a coding bug.

Ego-query development selected-patch retention relative to warmup falls from
68.6% at update 50 to 55.5% at update 200. Memory residual RMS/current-memory RMS
rises from 7.36% to 12.81%. These motivate selection-stationarity and fusion
controls. Train/dev mean ego speed is similar (4.15/4.36 m/s), which does not rule
out other distribution differences. Only eight development recordings were used.

## Source evidence and adaptation decisions

- [ForeDrive, §3 and Appendix H](https://arxiv.org/html/2609.26299v1#A8): the paper
  detaches injected future representations; planning reaches the online encoder
  through its current path. It does **not** establish an input-gradient-preserving
  predictor-parameter freeze. Our variant freezes predictor parameters only for
  the planning forward, preserving the selector's input gradient. Auxiliary
  prediction still updates the predictor. This is an inspired experiment, not a
  reproduction of ForeDrive. The paper also uses current-primary gated fusion.
- [PCGrad, §2.3](https://papers.neurips.cc/paper_files/paper/2020/file/3fe78a8acf5fda99de95303940a2420c-Paper.pdf):
  conflicting task gradients are projected. Our one-sided variant preserves the
  planning gradient and projects only its conflicting auxiliary component, over
  all predictor parameters jointly. It is not original symmetric/random-order
  PCGrad and does not inherit a general convergence or performance guarantee.

Primary texts were read; no author code was independently executed for either.
ResWorld was seen at abstract level only and is not an implementation source here.

## Registered first comparison

Config: `configs/drive_jepa_selective_future/overnight_causal_followup_v1.json`.
Each condition starts from the preserved per-seed auxiliary-warmup checkpoint.
New joint optimization is 400 updates; report update 400, not best development
checkpoint. Evaluate 0/50/100/200/400. Seeds 29/47/83, batch 8, identical joint
sample order; common module initialization and warmup are matched. AdamW,
predictor/bridge LR 1e-4, selector LR 2e-5, weight decay .01, clip 1, cosine to
10%, future raw-MSE weight .01. No auxiliary weight sweep.

| Condition | One change from reference | Question |
|---|---|---|
| `mlp_reference` | Existing MLP architecture, longer fixed schedule | Is complexity necessary? |
| `ego_reference` | Existing contextual-residual + ego-query model | Matched reference |
| `ego_predictor_aux_only` | Planning treats predictor parameters as constants | Does task coupling hurt? |
| `ego_planning_priority_projection` | One-sided auxiliary gradient projection | Does destructive interference matter? |
| `ego_bridge_half` | Fixed .5 memory-residual gain during train and inference | Is fusion too aggressive? |
| `ego_selector_frozen` | Freeze warmup selector parameters | Does moving selection destabilize learning? |
| `ego_uniform_ade` | Uniform XY ADE + .1 circular heading error | Does objective alignment affect measured ADE? |

21 runs / 8,400 joint updates. These are exploratory tests; using a development
set to choose a follow-up is not independent confirmation. The frozen-selector
control is not an optimized fixed policy. All conditions still receive current
ego status and command. Raw future targets/valid masks are auxiliary-loss-only.
No changed predictor/planner architecture is hidden inside an optimization row.

Runner: `scripts/run_drive_jepa_causal_followup.py`. It saves deltas, full optimizer,
scheduler, RNG, batch order and completed update every 25 updates and on a caught
stop. Explicit `--resume` recovers this runner exactly; initializing a new schedule
from historical warmup is **not** called exact resume of the previous study.

Next stages are conditional: inspect these causes, then fix a small matched
selection comparison (fixed/random/learned) and, if needed, a bounded training-
recording coverage check using the existing navtrain manifest. Any new config and
sample list must be saved before their results. Do not repeat the old pilot or
expand automatically to full cache/benchmark sweeps. Retain negative results.

## Execution status

### Registered recording coverage and selection followup

Before seeing any added-window model score, fixed 512 train windows from 64
recordings and 192 additional development windows from 24 recordings. Train128
is nested inside train512. Excluded the prior extension's eight development
recordings, all four old mini development recordings, and the known exposure
exceptions. Assignments remain in the original three-way manifest; held-out is
untouched. Sampling uses hashes, current-file eligibility and non-overlap, never
future availability or model errors. The official baseline itself was pretrained
on navtrain, so these are not an independent baseline benchmark or final test.

Cache uses the same official frozen encoder/crop/resize, without the old pilot's
ROI or rectification. Prior128 training cache files are reused by checksum.
Maximum logical cache 9 GiB, one hour generation cap, no dataset download. This
is a bounded extension data-coverage check, not a full dataset cache or old pilot
restart. Sample IDs and exclusions are already saved in
`outputs/drive_jepa_selective_future/overnight_recording_coverage_v1_20261003/`.

Config `overnight_coverage_training_v1.json` fixes nine conditions × three seeds
×800 joint updates, all initialized from the same preserved per-seed warmup:
MLP on128 / MLP on512 / ego-query / predictor-aux-only / half-gain / uniform-ADE /
MLP fixed-lattice / MLP seeded-random / MLP current-feature control. Current-feature
control learns selector/bridge only; predictor output is replaced by current
selected features and auxiliary supervision is disabled. Its active parameter
count differs and the implementation still executes the unused predictor, so
neither matched active capacity nor compute saving is claimed.

The paired same-K MLP fixed/random/learned comparison returns directly to the
research question. Per-window evaluation random IDs are stable across batch
order and checkpoint, while training random IDs change by update. Added-dev
is evaluated at0/100/200/400/800; final800 is the endpoint, not best checkpoint.
The128/512 comparison holds updates constant, not epochs. The128 condition's
training metrics use only its128 training windows. All rows share added-dev192.

`run_drive_jepa_overnight_sequence.py --detach` waits for the existing first study
without spawning another one, then exports results, builds the bounded cache,
trains the registered comparison and exports paired-seed/window/cluster evidence.
Only one child uses GPU1. A failure, explicit `pause.json`, or09:00 deadline stops
the queue. It never resumes WA or silently retries memory failures. A queue
completion is not itself an agent's scientific review of its results.

CPU suite: 121 tests passed; dedicated Python 3.9 compile and Ruff passed.
The first detached launch produced no run artifacts. A subsequent sandbox run
failed at CUDA initialization before any model update. Host GPU1 execution with
the same registered configuration is now launched. No new completed retraining
result is claimed yet. Source/checkpoint/cache provenance is saved per run.

## First controlled comparison results

The first21 runs are complete:8,400 joint updates,1,096.94 seconds total, all
official baseline parameters preserved. Subsequent coverage/selection extensions
bring the CPU suite to124 passing tests. This table reports fixed update400,
not the lower intermediate development values.

| Condition | Train ADE m | Development ADE m | Development seed standard deviation m |
|---|---:|---:|---:|
| Original frozen baseline | 0.273974 | 0.220644 | — |
| MLP reference | 0.187435 | 0.232000 | 0.015184 |
| Ego-query reference | 0.141326 | 0.239397 | 0.023509 |
| Predictor auxiliary-only updates | 0.146359 | 0.236099 | 0.013620 |
| Planning-priority gradient projection | 0.141960 | 0.239264 | 0.025172 |
| Half bridge gain during training | 0.158500 | 0.233743 | 0.024833 |
| Frozen selector | 0.130282 | 0.234913 | 0.002932 |
| Uniform ADE objective | 0.129398 | 0.242933 | 0.018298 |

Training fit improves substantially while added updates hurt development for
most runs. This supports a generalization concern but does not identify data
quantity, diversity, architecture capacity or regularization as the sole cause.
Predictor-gradient separation and projection do not solve the issue here.
Training with half gain differs from applying half gain after training: the
adapter can compensate for the scale while optimizing. The inference-only
improvement must not be substituted for this negative trained comparison.

All six comparisons against the ego reference have recording-cluster95% intervals
covering zero on the eight development groups. Do not read that as proof of
equivalence. Seeds and recording uncertainty are reported separately. No row is
promoted as a confirmed performance improvement; additional-development and
matched selection results are pending.

Evidence: [summary and paired intervals](../results/drive_jepa_selective_future/overnight_causal_followup_v1_20261003/summary.json),
[window data](../results/drive_jepa_selective_future/overnight_causal_followup_v1_20261003/window_results.csv),
[gradient contracts](../results/drive_jepa_selective_future/overnight_causal_followup_v1_20261003/gradient_and_dependence.json).
The local preregistration was8f5b18d; coverage/queue code is9b28ebc. GitHub push
failed because the configured credential socket is unavailable. Local commits
and results remain intact; remote visibility must not be claimed.
