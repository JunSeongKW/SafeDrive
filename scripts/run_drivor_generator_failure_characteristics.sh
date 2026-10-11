#!/usr/bin/env bash
set -euo pipefail
cd /rhome/junseong/PlanningAwareFuturePrediction
configuration=configs/frozen_planner_postprocessing/drivor_generator_failure_characteristics.json
output=outputs/drivor_generator_failure_characteristics_20261012
mkdir -p "$output"
export CUDA_VISIBLE_DEVICES=''
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=2
export PYTHONPATH=/rhome/junseong/PlanningAwareFuturePrediction/src
analysis_python=/rhome/junseong/PlanningAwareFuturePrediction/runtime/environments/kjs-lpwm-drivor-joint/bin/python
navsim_python=/rhome/junseong/PlanningAwareFuturePrediction/runtime/environments/drive_jepa_official_evaluation/bin/python
if [[ ! -f "$output/analysis_complete.json" ]]; then
    "$analysis_python" -u scripts/diagnose_drivor_generator_failure_characteristics.py --configuration "$configuration" > "$output/analysis.log" 2>&1
fi
if [[ ! -f results/drivor_generator_failure_characteristics_20261012/visualization_manifest.json ]]; then
    LD_LIBRARY_PATH=/rhome/junseong/PlanningAwareFuturePrediction/runtime/environments/drive_jepa_official_evaluation/lib "$navsim_python" -u scripts/visualize_drivor_generator_failure_characteristics.py --configuration "$configuration" > "$output/visualization.log" 2>&1
fi
"$analysis_python" -u scripts/report_drivor_generator_failure_characteristics.py --configuration "$configuration" > "$output/report.log" 2>&1
