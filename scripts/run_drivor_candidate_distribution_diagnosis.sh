#!/usr/bin/env bash
set -euo pipefail
cd /rhome/junseong/PlanningAwareFuturePrediction
configuration=configs/frozen_planner_postprocessing/drivor_candidate_distribution_diagnosis.json
output=outputs/drivor_candidate_distribution_diagnosis_20261011
mkdir -p "$output"
model_python=/rhome/junseong/PlanningAwareFuturePrediction/runtime/environments/kjs-lpwm-drivor-joint/bin/python
export CUDA_VISIBLE_DEVICES=''
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=2
export PYTHONPATH=/rhome/junseong/PlanningAwareFuturePrediction/src
if [[ ! -f "$output/rescoring_complete.json" ]]; then
    "$model_python" -u scripts/diagnose_drivor_candidate_distributions.py --phase rescore --configuration "$configuration" > "$output/rescore.log" 2>&1
fi
if [[ ! -f "$output/analysis_complete.json" ]]; then
    "$model_python" -u scripts/diagnose_drivor_candidate_distributions.py --phase analyze --configuration "$configuration" > "$output/analyze.log" 2>&1
fi
