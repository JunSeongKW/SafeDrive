#!/usr/bin/env bash
set -euo pipefail
cd /rhome/junseong/PlanningAwareFuturePrediction
variant="${1:?raw or rotation_stabilized}"
gpu_index="${2:?GPU 0 or 1}"
case "$variant:$gpu_index" in raw:0|rotation_stabilized:1) ;; *) exit 2 ;; esac
lpwm_python=/rhome/junseong/envs/kjs-lpwm-navsim/bin/python
artifact_root=outputs/lpwm_navsim_adaptation_v1
exec 9>"$artifact_root/${variant}_queue.lock"
flock -n 9
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export PYTHONUNBUFFERED=1
for seed in 29 47 71; do
  run_name="${variant}_seed${seed}"
  if [[ ! -f "$artifact_root/runs/$run_name/training_summary.json" ]]; then
    "$lpwm_python" scripts/run_lpwm_navsim_adaptation.py --variant "$variant" --seed "$seed" --gpu "$gpu_index" --updates 300
  fi
  if [[ ! -f "$artifact_root/evaluation/$run_name/metrics.json" ]]; then
    "$lpwm_python" scripts/evaluate_lpwm_navsim_adaptation.py --name "$run_name" --variant "$variant" --gpu "$gpu_index" --checkpoint "/rhome/junseong/PlanningAwareFuturePrediction/$artifact_root/runs/$run_name/checkpoint.pt"
  fi
done
if [[ ! -f "$artifact_root/evaluation/zero_shot_$variant/metrics.json" ]]; then
  "$lpwm_python" scripts/evaluate_lpwm_navsim_adaptation.py --name "zero_shot_$variant" --variant "$variant" --gpu "$gpu_index"
fi
echo "PILOT_VARIANT_DONE $variant"
