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

Launch history:121 CPU tests initially passed; the first detached launch produced
no run artifacts. A subsequent sandbox run failed at CUDA initialization before
any model update. Host GPU1 execution subsequently completed the first21 runs
below. The later projection/regularization controls bring the suite to130 tests.
Projection identity tests initially compared different grad/fast-path execution
modes; with matched no-grad execution they pass bitwise. Gradient tests are
separate. Source/checkpoint/cache provenance is retained, including failed checks.
Inherited `source.json` describes the historical official-checkpoint reproduction;
the actual new training split and authorization are in this plan and each run's
`specification.json`, not its inherited navtest/evaluation-only metadata.

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

## Future projection transfer hypothesis

Direct code inspection shows that the bridge initializes a new1024→256 future
projection, although the frozen original planner already has an `image_fc` with
that shape. The original projection was trained on pooled current features;
selected future patch features need not share that distribution. Reusing it is
therefore a hypothesis, not an established alignment or guaranteed improvement.

Register three further ego-query conditions on the same512/192 cache and exact
800-update/3-seed schedule, after the coverage study finishes:

- Frozen random projection: isolate removing projection update capacity.
- Frozen copy of official `image_fc`: isolate pretrained projection initialization
  relative to the frozen random control.
- Official projection plus rank4/alpha4 LoRA: test limited trainable adaptation
  relative to frozen official projection.

The existing coverage ego reference supplies the learned-random projection row;
do not retrain it. Other module initialization, preserved predictor warmup,
sample order, targets, losses and original planner remain matched. All copied
official parameters stay frozen. LoRA applies to the **future projection**, not
the encoder or original trajectory planner. Active capacities are reported.

The [LoRA paper, §4.1](https://arxiv.org/html/2106.09685v2#S4.SS1) supplies the
frozen-weight plus scaled low-rank-update idea. Our existing local implementation
uses Kaiming input-factor initialization and a zero output factor rather than
the paper's stated Gaussian input initialization; this is not a paper result
reproduction. Zero output factor makes initial projection exactly official.

Config `overnight_projection_transfer_v1.json`:9 additional runs/7,200 updates,
same09:00 deadline, two-hour series cap, no new cache. CPU contracts test initial
identity, gradient flow, and preservation of original parameters. The second
queue waits for the first queue's completion, so the two do not use GPU1 together.
This comparison targets how selected futures enter the preserved planner, not a
new predictor research question. Results are not yet available at registration.

## Conservative adaptation controls

The longer fits degrade after early development improvements on both the small
and added-recording sets. Inference half-gain helped one historical model but
training half-gain did not reliably preserve that benefit. A learnable bridge
can offset a fixed gain; limiting its residual energy is a different test.

Register `overnight_conservative_adaptation_v1.json`: MLP and ego-query each get
two separate conditions, never combined: all learning rates scaled by0.2, or
an extra relative-memory-energy penalty with weight1. The latter is
mean(residual²)/mean(frozen-current-memory²), averaged per sample, and equals0.01
for a residual with10% of current RMS. The coefficient is fixed before these
runs; there is no grid search. Targets and future auxiliary weight stay fixed.
This is a dimensionless L2 regularizer, not a formal trust-region guarantee.

Twelve paired-seed runs use the same512/192 data, same800 updates and matching
warmup/batch schedules as their MLP/ego references. Report the final fixed step
and separate regularization loss. Added-dev has already informed this hypothesis,
so this stage is explicitly further development, not independent confirmation.
The projection queue executes it only after the prior stages succeed and before
09:00. In total this night's registered work is69 runs/46,800 joint updates
(21×400 +27×800 +9×800 +12×800), with time and memory caps taking precedence.

When all stages finish, the supervisor commits only the twelve known generated
evidence files from the three later result directories, never unrelated staged
work. It attempts the authorized `mine junseong/main` push once; credential
failure is recorded and does not erase local results. It does not edit scientific
conclusions automatically or declare the research hypothesis established.

## Completed recording coverage and selection comparison

The second27 runs completed21,600 joint updates in1,940.08 seconds, maximum
allocated1.568 GiB. The new cache took301.14 seconds and added6.04 GB, reusing
128 old files. Official weights are unchanged. An independent CPU recomputation
from window records confirms the scene-macro aggregation, paired batch schedules,
baseline outputs and auxiliary-gradient boundaries for all48 completed runs.
The following table uses the preregistered final update800 and the same192
development windows /182 scenes /24 recordings throughout.

| Condition | Train ADE m | Development ADE m | Seed std m |
|---|---:|---:|---:|
| Original | — | 0.352210 | — |
| MLP,128-window training subset | 0.119066 | 0.372616 | 0.004365 |
| MLP learned selection,512 train | 0.234251 | 0.361504 | 0.007782 |
| Ego-query learned selection | 0.194074 | 0.371554 | 0.013255 |
| Ego predictor auxiliary-only updates | 0.198389 | 0.367154 | 0.007870 |
| Ego half bridge gain | 0.207424 | 0.366605 | 0.007685 |
| Ego uniform ADE objective | 0.187032 | 0.370875 | 0.008822 |
| MLP fixed lattice | 0.236439 | 0.355226 | 0.004661 |
| MLP seeded random selection | 0.249884 | 0.351212 | 0.003054 |
| MLP current-feature control | 0.196653 | 0.365208 | 0.014428 |

Do not compare this original0.352210 directly with the earlier0.220644: the
development population changed. The128-row train metric also has a different
training population from the512-row conditions. Compare their common dev only.

Random minus learned MLP ADE is−0.010292m, recording-cluster95% interval
[−0.018142,−0.003216]. Random minus original is−0.000998m with interval
[−0.010777,+0.007704]; this is not confirmed improvement over the original.
These are exploratory, seed-averaged recording intervals without multiplicity
correction, not a general finding that random selection is superior.
Larger training coverage reduces MLP error by0.011112m relative to the nested
128-window pool, but its interval includes zero. No condition is promoted yet.

The learned MLP predicts its selected future targets with MSE1.9090 versus its
matched persistence2.2062; ego-query gives2.2101 versus2.3976. Thus poor planning
cannot simply be labeled failure to beat visual persistence. Conversely, these
MSEs concern different selected patch sets across policies and cannot rank the
policies' forecasting quality on a common target set.

There are46/121/25 windows for command IDs0/1/2; command means in the JSON are
window-weighted diagnostics, not scene-macro results. Learned selections are
more spatially concentrated within a frame than random (MLP mean normalized
pair spacing0.191–0.232 vs random0.373–0.386), but span266–305 distinct IDs
across development. This is not global selection collapse. Whether nearby
patches are redundant or legitimately relevant remains an untested hypothesis.

Evidence: [coverage summary](../results/drive_jepa_selective_future/overnight_coverage_training_v1_20261003/summary.json),
[window records](../results/drive_jepa_selective_future/overnight_coverage_training_v1_20261003/window_results.csv).
Projection transfer and conservative adaptation are still running/queued; no
performance result for those stages is claimed here.

## Registered train-only selection-surrogate diagnosis

Because learned selection underperformed seeded random selection, add a read-only
check of the local routing surrogate after all registered training finishes.
This does **not** increase the69-run /46,800-update training cap. Fixed final800
MLP and ego-query checkpoints, seeds29/47/83, are tested on32 training recordings,
one hash-selected window each. Candidate substitutions are seeded without target
scores: eight unselected IDs per each of four slots,32 substitutions per window.
No development/held-out/navtest examples or future visual GT enter online inputs.

At the selected hard routing matrix compute the planning-loss derivative with
respect to its entries. For replacing ID j by j' in slot k, the first-order
surrogate is dL/dW[k,j']−dL/dW[k,j]. Compare that number with the actual loss
change after the full predictor, bridge and preserved planner process the
replacement. Report sign agreement, correlation, exact substitution counts and
all individual changes; exclude changes of at most1e-6 from sign agreement.
The continuous derivative need not approximate a large discrete jump well.
This checks the downstream local W derivative, **not** the full selector-score
Jacobian, an optimizer update, a world-causal importance or deployable policy.
Ground-truth ego trajectories are used only to compute this training-only loss;
the best hindsight substitution is not an oracle bound or reported planning score.

The reference row is included in each batched no-grad counterfactual forward.
Its output is checked against the gradient-enabled reference with an absolute
FP32 tolerance1e-5, fixed before execution. Batch/grad-mode differences are
reported, not silently attributed to a patch replacement. No tolerance sweep.
Official weights must remain unchanged. Single GPU1,16GiB admission/6GiB free
reserve/6GiB allocated cap,15-minute wall cap and09:00 deadline. Failure stops
the diagnostic without repeating training or changing its checkpoints.
Config `overnight_selection_surrogate_diagnosis_v1.json`, runner
`diagnose_drive_jepa_selection_surrogate.py`, serial queue `--series surrogate`.
The queue first independently audits all69 completed experiments on CPU, then
runs this diagnosis; it cannot overlap GPU training.

### Numerical guard and matched-execution revision

V1 stopped with a reference trajectory difference1.1444091796875e-5, just above
the preregistered1e-5 guard. It compared a single-window grad-enabled reference
against a33-window no-grad call, confounding batch and autograd/attention paths.
The failure and exact source/config hashes are [preserved](../results/drive_jepa_selective_future/selection_surrogate_v1_guard_failure_20261003.json).
No training was rerun or changed. V2 evaluates each counterfactual with batch1
and gradients enabled, as for the reference, but performs no backward/optimizer
update for the counterfactual. Sample selection, checkpoint, substitutions and
tolerance are unchanged. A fixture test checks that an explicit identical ID
list preserves the learned-ID output under matched execution. The actual GPU
test is still required;137 CPU tests do not replace it.

V2 config `overnight_selection_surrogate_diagnosis_v2.json`, queue
`run_drive_jepa_overnight_sequence.py --series surrogate_matched --detach`.
Outputs use a separate v2 directory. The completed69-run CPU evidence audit is
reused, not recomputed or overwritten.

## All registered training completed

All69 runs /46,800 joint updates finished. The four serial training series took
5,027.30 seconds combined, plus301.14 seconds for the bounded cache and149.06
seconds for the earlier read-only checkpoint diagnosis. These are process wall
times, not exclusive GPU compute times. Training peak allocated memory was
1.573GiB. The immutable original parameter hash and independent window/seed/
batch-schedule/gradient-boundary checks passed for all69 runs.

The remaining seven conditions use the same192 development windows and fixed
800-update endpoint as the coverage comparison:

| Condition | Dev ADE mean m | Seed std m | Difference vs original m | Recording-cluster95% interval m |
|---|---:|---:|---:|---|
| Frozen random future projection | 0.362767 | 0.013671 | +0.010557 | [−0.004736,+0.025555] |
| Frozen official future projection | 0.359388 | 0.013887 | +0.007178 | [−0.006943,+0.020092] |
| Official future projection +LoRA | 0.360904 | 0.011637 | +0.008694 | [−0.006289,+0.022388] |
| MLP learning rates ×0.2 | 0.347694 | 0.000392 | −0.004515 | [−0.009774,+0.001198] |
| Ego-query learning rates ×0.2 | 0.346338 | 0.001650 | −0.005872 | [−0.011676,+0.000194] |
| MLP relative-memory penalty | 0.349090 | 0.003654 | −0.003120 | [−0.010691,+0.003659] |
| Ego-query relative-memory penalty | 0.358561 | 0.004929 | +0.006351 | [−0.001441,+0.014002] |

Lower learning rates improve relative to their original optimizer settings:
MLP−0.013809m (cluster interval[−0.024027,−0.004267]), ego−0.025216m
([−0.037982,−0.012743]). This is evidence that adaptation step size matters in
this setup, not proof that optimizer choice is the sole cause. Stronger data
coverage, gradient routing, projection copying or complexity alone did not
establish improvement over the frozen original. Every listed original-baseline
interval includes zero; absence of a confirmed difference is not equivalence.

The lowest development mean is ego-query with lower rates, but the1.67% ADE
reduction is only about5.9mm and comes from repeatedly used development data.
Do not equate it with PDMS improvement or rank it as a validated method. The
random/fixed policies have not been retrained at these lower rates, so comparing
their prior rows directly to the lower-rate learner cannot establish learned
selection superiority. That matched comparison is a next-stage question, not
permission to change this completed experiment's endpoint or grow its sweep.

Evidence: [projection transfer](../results/drive_jepa_selective_future/overnight_projection_transfer_v1_20261003/summary.json),
[conservative adaptation](../results/drive_jepa_selective_future/overnight_conservative_adaptation_v1_20261003/summary.json),
[independent full evidence audit](../results/drive_jepa_selective_future/overnight_evidence_audit_20261003.json).
Automatic result commit313a3ab was created; push failed with return code128.
Original69 checkpoints, failed diagnostic logs and all window records remain local.

## Completed actual-model routing diagnosis

The matched-execution v2 completed all6 model/seed cases:32 train recordings,
32 substitutions each,6,144 total. Reference output differences were exactly0
in all cases, without changing the1e-5 tolerance. Original weights are unchanged;
no optimizer updates. Wall128.75 seconds, peak allocated1.189GiB.

| Model / seed | Local derivative sign agreement | Pearson loss-change correlation | Sampled substitutions improving train loss |
|---|---:|---:|---:|
| MLP29 | 91.70% | 0.815 | 50.59% |
| MLP47 | 89.75% | 0.681 | 50.49% |
| MLP83 | 86.43% | 0.578 | 37.89% |
| Ego-query29 | 75.59% | 0.475 | 20.21% |
| Ego-query47 | 80.37% | 0.603 | 35.55% |
| Ego-query83 | 73.54% | 0.635 | 27.34% |

All6,144 changes exceeded the1e-6 exclusion floor. The local routing derivative
contains useful directional information in these training examples, but neither
perfectly predicts finite changes nor establishes that the selector's parameter
updates learn a good policy. This is not a comparison with a random sign/null
model, not independent observations for significance testing, and not a measure
of world-causal relevance. Training-loss-reducing substitutions still exist;
they were not used to update a model or to report a deployable oracle score.
Evidence: [all substitutions and checkpoint hashes](../results/drive_jepa_selective_future/overnight_selection_surrogate_diagnosis_v2_20261003.json).

## Next bounded stage: matched conservative selection

The original69-run stage is complete and immutable. Under the user's continuing
overnight authorization, register a **separate18-run /14,400-update** next stage
to answer the selection question under matched learning rates, rather than grow
the earlier optimizer/projection sweep. Deadline09:00 and a two-hour stage cap
remain. No new cache, architecture, target definition or official benchmark.

Reuse the completed three-seed MLP/ego lower-rate learned references. Add only:

- MLP fixed lattice and seeded random at the same lower rates.
- MLP joint future auxiliary weight0 after the **same preserved future warmup**.
  This isolates continued joint auxiliary supervision, not absence of all future
  knowledge. All predictor/bridge/selector modules stay present and trainable.
- MLP current-feature bypass at the same lower rates. Its inactive predictor
  means this is not a capacity-identical no-auxiliary comparison or a speed claim.
- Ego-query fixed lattice and seeded random at the same lower rates.

All use the same512/192 development setup, K4/horizon4, seeds29/47/83,800 updates,
fixed evaluation times and final checkpoint; effective rates are predictor/bridge
2e-5 and selector4e-6. Initial S/P/bridge hashes and batch schedules must exactly
match the saved learned reference before training. Reference config, architecture,
rate scale, projection mode and memory-penalty settings are checked at export.
Gradient reports describe unweighted diagnostic losses; effective joint weights
and weighted auxiliary losses are recorded separately, especially for aux-off.

Config `matched_conservative_selection_v1.json`; queue `--series matched_selection`.
This is repeated-development analysis informed by previous results, not a new
independent validation set. Lower-rate learned references are not retrained.
The original69-run audit remains unchanged; the new18-run audit is separate.
Across both stages the registered training total is87 runs /61,200 updates,
but only69 runs are complete at this registration. The supervisor exports,
audits and commits only known results, tries the authorized push once, then
stops; it does not launch further unregistered training or claim success.
