# LPWM results at the user-requested pause

Recorded: 2026-10-13T20:16:47.366189+09:00

Training and queue are paused. Physical GPU0/1 each returned to 15 MiB and 0% utilization. Resume requires another explicit user request.

Saved full resume state: 24,377 updates, 20 completed epochs and 837 updates of epoch21. Model, 796 AdamW states, scheduler and both rank RNG states were checked. Checkpoint: /rhome/junseong/PlanningAwareFuturePrediction/outputs/lpwm_front_history_stage1_lora_v1/stage2/latest.pt

## Internal development history

Same 1,024 predicted scenes / 1,021 available official scores, not NAVTEST.

| Epoch | Updates | PDMS | ADE m | FDE m |
|---|---:|---:|---:|---:|
| 1 | 1177 | 69.7437 | 3.5841 | 8.9248 |
| 2 | 2354 | 77.4678 | 1.8605 | 4.5543 |
| 3 | 3531 | 81.7225 | 1.6245 | 3.9800 |
| 4 | 4708 | 81.8300 | 2.0900 | 5.2255 |
| 5 | 5885 | 83.9882 | 1.7463 | 4.2489 |
| 6 | 7062 | 85.4884 | 1.6864 | 3.9850 |
| 7 | 8239 | 85.6399 | 1.6395 | 4.0769 |
| 8 | 9416 | 85.8215 | 1.6022 | 3.9951 |
| 9 | 10593 | 87.4335 | 1.3611 | 3.4214 |
| 10 | 11770 | 87.4052 | 1.4209 | 3.5435 |
| 11 | 12947 | 87.5119 | 1.2510 | 3.1992 |
| 12 | 14124 | 87.4761 | 1.3841 | 3.5798 |
| 13 | 15301 | 88.1786 | 1.3202 | 3.3408 |
| 14 | 16478 | 88.0149 | 1.2163 | 3.0565 |
| 15 | 17655 | 87.8737 | 1.2536 | 3.2201 |
| 16 | 18832 | 88.3220 | 1.3685 | 3.4660 |
| 17 | 20009 | 88.5973 | 1.2455 | 3.1805 |
| 18 | 21186 | 88.3545 | 1.2198 | 3.1437 |
| 19 | 22363 | 88.5371 | 1.2349 | 3.1982 |
| 20 | 23540 | 89.1006 | 1.2012 | 3.1151 |

## Common NAVTEST history

Repeatedly exposed fixed 1,024 scenes /44 recordings, not full NAVTEST or independent final evaluation.

| Epoch | PDMS | ADE m | FDE m |
|---|---:|---:|---:|
| 1 | 70.1667 | 3.7751 | 9.4406 |
| 3 | 80.2828 | 1.6974 | 4.1122 |
| 4 | 81.5610 | 2.1684 | 5.3802 |
| 5 | 82.3243 | 1.8408 | 4.4376 |
| 9 | 85.9317 | 1.4212 | 3.5233 |
| 14 | 85.2586 | 1.2824 | 3.2062 |

## Latest shared NAVTEST comparison

| Model | Epoch | PDMS | ADE m | FDE m |
|---|---:|---:|---:|---:|
| lpwm | 14 | 85.2586 | 1.2824 | 3.2062 |
| drivor | 5 | 83.9662 | 1.8092 | 4.3526 |
| jepa | 5 | 85.3751 | 1.6929 | 3.9388 |

Baselines are preserved small-corpus models, not original full pretrained paper systems. LPWM uses 75,297 planning scenes, front1/history4/128×128 and reused Stage1 (23,126 clips×20 epochs). Baselines use 10,240 planning scenes, front1/history2/512×256. These are unequal training budgets.

## Interpretation

Internal PDMS peaks at epoch20 (89.1006), but latest measured NAVTEST remains epoch14 (85.2586). Its epoch9→14 change is −0.6731 points with recording bootstrap95%CI[−2.1966,+0.8607]. Internal development improvement does not establish NAVTEST improvement. No NAVTEST evaluation was run after the pause.

On 16 world-model diagnostic scenes, reconstruction MSE changes +1.0994% and future MSE +4.3108% relative to Stage1.

Fixed four-scene particle diagnostics at epoch20: mean center shift 0.8910 input pixels; mean relative size change 11.4481%. Geometry change is not proof of planning-relevant object information.

![Training history](planning_training_history.png)
