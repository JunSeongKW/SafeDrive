# LPWM epoch 14: common NAVTEST results

Same fixed NAVTEST1,024 / 44 recordings, official NAVSIM v1 evaluator. This is a repeatedly exposed development subset; not full NAVTEST or independent final evaluation.

| Model | Planning epoch | PDMS | ADE m | FDE m |
|---|---:|---:|---:|---:|
| lpwm | 14 | 85.2586 | 1.2824 | 3.2062 |
| drivor | 5 | 83.9662 | 1.8092 | 4.3526 |
| jepa | 5 | 85.3751 | 1.6929 | 3.9388 |

## Same-scene changes

- LPWM minus lpwm_epoch9: -0.6731 points; recording bootstrap 95% CI [-2.1965846557541995, 0.8606758738232583].
- LPWM minus drivor_epoch5: +1.2924 points; recording bootstrap 95% CI [-1.5174382211468729, 3.9957666506722958].
- LPWM minus jepa_epoch5: -0.1165 points; recording bootstrap 95% CI [-2.323951022675038, 1.9849063710103176].

Against LPWM epoch9, ADE/FDE decrease from 1.4212/3.5233 to 1.2824/3.2062 m, while mean DAC decreases from 0.957031 to 0.942383. This is an observed submetric change, not an isolated causal attribution of the whole PDMS difference.

## Current ego command

| Command | Scenes | LPWM | DrivoR | Drive-JEPA-style |
|---|---:|---:|---:|---:|
| left_turn | 205 | 83.3711 | 82.4200 | 84.3096 |
| straight | 710 | 86.7564 | 86.3969 | 86.1840 |
| right_turn | 109 | 79.0523 | 71.0416 | 82.1105 |

## Official mean submetrics

| Metric | LPWM | DrivoR | Drive-JEPA-style |
|---|---:|---:|---:|
| no_at_fault_collisions | 0.978027 | 0.987793 | 0.985840 |
| drivable_area_compliance | 0.942383 | 0.940430 | 0.947266 |
| ego_progress | 0.790738 | 0.761232 | 0.777863 |
| time_to_collision_within_bound | 0.941406 | 0.943359 | 0.949219 |
| comfort | 1.000000 | 1.000000 | 0.999023 |
| driving_direction_compliance | 0.960449 | 0.968750 | 0.972656 |

## Controls and limits

LPWM planning epoch14 versus preserved small-corpus DrivoR and Drive-JEPA-style epoch5 checkpoints. Same NAVTEST1024 scenes and official NAVSIM v1 evaluator; unequal training datasets, presentations, history, resolution, memory, seed and SSL. Not a matched-budget representation comparison, full NAVTEST, or original-paper reproduction.

LPWM planning: 75,297 scenes ×14 passes, 16,478 updates, effective64, seed2; front1/history4/128×128; Stage1 23,126 clips×20 epochs. Preserved DrivoR/Drive-JEPA-style: 10,240 scenes×5 passes, 3,200 updates, effective16, seed47; front1/history2/512×256. These are study baselines, not public original full-model checkpoints.

Baseline predictions and official scores were copied byte-for-byte. LPWM inference uses the existing strict-loading registered code, original precision and batch4. No original training setting or queue was modified. All three predictions are selected top1 trajectories; these numbers are not candidate-average scores or oracle candidate-best scores.

The panel scene_type uses expert terminal yaw and is separate from current ego command. Earlier epoch9 visualization mislabeled these scene_type strata as current command. Historical numeric results are preserved; this report explicitly calculates current-command strata separately.

Bootstrap reflects recording-level variation on this panel and does not quantify training-seed variability. Repeated NAVTEST exposure makes this a development diagnosis. A higher score with unequal training budgets does not isolate particle representation or establish superiority over the original papers.

![Shared NAVTEST results](common_navtest_results.png)

## Replay

```bash
cd /rhome/junseong/PlanningAwareFuturePrediction
runtime/environments/kjs-lpwm-drivor-joint/bin/python scripts/report_lpwm_common_navtest_checkpoint.py --config configs/lpwm_front_history_stage1_lora/epoch14_common_navtest.json
```
