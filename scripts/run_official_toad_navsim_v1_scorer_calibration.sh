#!/usr/bin/env bash
set -euo pipefail
cd /rhome/junseong/PlanningAwareFuturePrediction
configuration=configs/frozen_planner_postprocessing/official_toad_navsim_v1_scorer_calibration.json
output=outputs/official_toad_navsim_v1_scorer_calibration_20261011
mkdir -p "$output"
model_python=runtime/environments/kjs-lpwm-drivor-joint/bin/python
official_python=runtime/environments/drive_jepa_official_evaluation/bin/python
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTHONPATH=/rhome/junseong/PlanningAwareFuturePrediction/src
if [[ ! -f "$output/registration.json" ]]; then
    CUDA_VISIBLE_DEVICES='' "$model_python" -u scripts/calibrate_official_toad_navsim_v1_scorer.py --phase prepare --configuration "$configuration" > "$output/prepare.log" 2>&1
fi
if [[ ! -f "$output/profile_complete.json" ]]; then
    CUDA_VISIBLE_DEVICES=0 "$model_python" -u scripts/calibrate_official_toad_navsim_v1_scorer.py --phase profile --configuration "$configuration" > "$output/profile.log" 2>&1
fi
if [[ ! -f "$output/inference_complete.json" ]]; then
    CUDA_VISIBLE_DEVICES=0 "$model_python" -u scripts/calibrate_official_toad_navsim_v1_scorer.py --phase infer --configuration "$configuration" > "$output/infer.log" 2>&1
fi
if [[ ! -f "$output/official_evaluation_complete.json" ]]; then
    CUDA_VISIBLE_DEVICES='' LD_LIBRARY_PATH=/rhome/junseong/PlanningAwareFuturePrediction/runtime/environments/drive_jepa_official_evaluation/lib OMP_NUM_THREADS=1 "$official_python" -u scripts/calibrate_official_toad_navsim_v1_scorer.py --phase score --configuration "$configuration" > "$output/score.log" 2>&1
fi
if [[ ! -f "$output/calibration_complete.json" ]]; then
    CUDA_VISIBLE_DEVICES='' "$model_python" -u scripts/calibrate_official_toad_navsim_v1_scorer.py --phase report --configuration "$configuration" > "$output/report.log" 2>&1
fi
if [[ ! -f "results/official_toad_navsim_v1_scorer_calibration_20261011/cpu_gradient_validation.json" ]]; then
    CUDA_VISIBLE_DEVICES='' "$model_python" -u scripts/verify_official_toad_cached_gradients.py --configuration "$configuration" > "$output/cpu_gradient_validation.log" 2>&1
fi
if [[ ! -f "$output/inspection_packaged.json" ]]; then
    CUDA_VISIBLE_DEVICES='' "$model_python" -u scripts/package_official_toad_scorer_inspection.py --configuration "$configuration" > "$output/package.log" 2>&1
fi
