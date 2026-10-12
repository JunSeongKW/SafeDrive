#!/usr/bin/env bash
set -euo pipefail
cd /rhome/junseong/PlanningAwareFuturePrediction
configuration=configs/scorer_compatible_generator/accuracy_only_v2.json
output=outputs/scorer_compatible_generator_accuracy_only_v2
mkdir -p "$output"
export CUDA_VISIBLE_DEVICES=''
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=2
export PYTHONPATH=/rhome/junseong/PlanningAwareFuturePrediction/src
analysis_python=/rhome/junseong/PlanningAwareFuturePrediction/runtime/environments/kjs-lpwm-drivor-joint/bin/python
evaluation_python=/rhome/junseong/PlanningAwareFuturePrediction/runtime/environments/drive_jepa_official_evaluation/bin/python
if [[ ! -f "$output/training_complete.json" ]]; then
    "$analysis_python" -u scripts/run_scorer_compatible_accuracy_only.py --configuration "$configuration" --phase train > "$output/train.log" 2>&1
fi
if [[ ! -f "$output/prediction_complete.json" ]]; then
    "$analysis_python" -u scripts/run_scorer_compatible_accuracy_only.py --configuration "$configuration" --phase predict > "$output/predict.log" 2>&1
fi
if [[ ! -f "$output/evaluation_complete.json" ]]; then
    LD_LIBRARY_PATH=/rhome/junseong/PlanningAwareFuturePrediction/runtime/environments/drive_jepa_official_evaluation/lib "$evaluation_python" -u scripts/run_scorer_compatible_accuracy_only.py --configuration "$configuration" --phase labels > "$output/labels.log" 2>&1
fi
"$analysis_python" -u scripts/run_scorer_compatible_accuracy_only.py --configuration "$configuration" --phase report > "$output/report.log" 2>&1
