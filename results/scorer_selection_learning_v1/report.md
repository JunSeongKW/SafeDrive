# Selection-error acquisition and frozen-scorer correction pilot

## Question

At equal official-label budgets, does scene-conditioned acquisition of selection errors improve a frozen planner's trajectory ranking?

## Actual locked development results

192 previously used NAVVAL development scenes; recording-disjoint from fitting and tuning. One seed, not independent NAVTEST.

| Method | Selected PDMS | Fixed oracle | Selection regret | Collision/offroad |
|---|---:|---:|---:|---:|
| unchanged | 78.4753 | 97.5683 | 19.0930 | 12.50% |
| A_uniform | 78.0014 | 97.5683 | 19.5669 | 12.50% |
| B_boundary | 77.5620 | 97.5683 | 20.0063 | 13.02% |
| C_selection_error | 77.5871 | 97.5683 | 19.9811 | 13.02% |

C minus A: -0.4142 points; paired recording 95% interval [-1.3968951318303895, 0.0755573365904786]. Replication gate passed: False.

## Controls and interpretation

All rankers use identical frozen encoder/generator/scorer weights and native64 candidate contexts. The six corrected logits use the original scalar composition. A/B/C have identical correction initialization, parameter count, optimizer, batch order, pairs, 500 fitting steps and 96 official labels per training scene. Each method uses 64 native and 32 acquired candidates, with the first128 common scouts identical.

The first128 training scenes supervise an eight-family sampler using true utility gaps weighted by soft frozen-scorer selection probability. Its300 CPU updates use no development labels. C chooses later acquisition families using scene/ego/base controls/spread. The sampler is used during data acquisition only; inference uses the same frozen native proposals.

Boundary baseline uses training GT, map and future occupancy for coarse geometric boundary scans. It does not reproduce the Versatile Samples simulator-margin solver. Geometry probes cost time but consume no official trajectory-score labels; their counts are recorded. Boundary paths are not guaranteed comfort compliant; official tracked execution supplies the labels.

The component loss is the mean six-subscore BCE (soft progress target); the additional ranking loss is actual-score-gap-weighted pairwise softplus. This is an adaptation objective, not an exact re-execution of original DrivoR training. Frozen score composition retains NAVSIM v1 DDC weight zero; DDC is reported separately.

The sampler reward is a proxy for useful data acquisition, not a directly differentiated final selection-regret objective. Reparameterization is not used to differentiate through the official evaluator or argmax. A single learning cycle does not establish optimal active sampling or novelty.

The base checkpoint has only3200 small-corpus updates. Correcting this weak scorer does not establish benefit on a fully trained published planner. Candidate self-attention means representations depend on the candidate context: each evaluation context is kept identical, while training acquired contexts vary as part of the acquisition intervention.

Tuning selects one of100/250/500 steps on64 separate-recording scenes; locked192 results are read once after selection. All500 steps are executed for every method. These recordings have been used in previous development work; this is not a pristine final holdout. Bootstrap describes recording variation, not learning-seed uncertainty.

## Costs

Cache GPU occupancy 35.18s, peak allocated 0.226GiB. Sampler and correction training GPU0s. Official candidates physically evaluated: 188416; shared labels counted once physically and equally in each method's budget.

Actual per-method fitting, correction inference, official labeling and sampler costs are in results.json. Cached correction latency excludes the already-frozen encoder and original scorer; total deployment latency must be profiled later.

## Next experiment

If the preregistered gate passes, repeat this same controlled comparison with another training seed and a separate evaluation-recording panel. If it fails, inspect native-progress/DAC ranking and acquired-pair coverage before adding modules or scaling training.

## Related work (verified on2026-10-11)

- CLOVER: https://arxiv.org/html/2605.15120 (alternating scorer fitting and conservative generator distillation)
- Designing Versatile Samples for Learned Trajectory Scoring: https://arxiv.org/html/2609.01799 (training-only boundary candidates)

## Replay

```bash
cd /rhome/junseong/PlanningAwareFuturePrediction
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 PYTHONPATH=src runtime/environments/kjs-scorer-selection-pilot/bin/python -u scripts/run_scorer_selection_pilot.py --configuration configs/scorer_selection_learning/pilot_v1.json --phase run
```

Completed phases are not repeated; new scientific settings require a new config/output registration.

![Comparison](selection_regret_comparison.png)
