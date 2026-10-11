#!/usr/bin/env bash
set -euo pipefail
cd /rhome/junseong/PlanningAwareFuturePrediction
configuration=configs/frozen_planner_postprocessing/common_candidate_scorer_comparison.json
output=outputs/common_candidate_scorer_comparison_20261011
mkdir -p "$output"
model_python=runtime/environments/kjs-lpwm-drivor-joint/bin/python
official_python=runtime/environments/drive_jepa_official_evaluation/bin/python
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTHONPATH=/rhome/junseong/PlanningAwareFuturePrediction/src
if [[ ! -f "$output/registration.json" ]]; then
    CUDA_VISIBLE_DEVICES='' "$model_python" -u scripts/compare_public_drivor_toad_scorers.py --phase prepare --configuration "$configuration" > "$output/prepare.log" 2>&1
fi
if [[ ! -f "$output/inference_complete.json" ]]; then
    CUDA_VISIBLE_DEVICES=0 "$model_python" -u scripts/compare_public_drivor_toad_scorers.py --phase infer --configuration "$configuration" > "$output/infer.log" 2>&1
fi
if [[ ! -f "$output/official_evaluation_complete.json" ]]; then
    CUDA_VISIBLE_DEVICES='' LD_LIBRARY_PATH=/rhome/junseong/PlanningAwareFuturePrediction/runtime/environments/drive_jepa_official_evaluation/lib OMP_NUM_THREADS=1 "$official_python" -u scripts/compare_public_drivor_toad_scorers.py --phase score --configuration "$configuration" > "$output/score.log" 2>&1
fi
if [[ ! -f "$output/comparison_packaged.json" ]]; then
    CUDA_VISIBLE_DEVICES='' "$model_python" -u scripts/report_common_candidate_scorer_comparison.py --configuration "$configuration" > "$output/report.log" 2>&1
fi
if [[ ! -f "$output/navsim_visualization_complete.json" ]]; then
    CUDA_VISIBLE_DEVICES='' LD_LIBRARY_PATH=/rhome/junseong/PlanningAwareFuturePrediction/runtime/environments/drive_jepa_official_evaluation/lib OMP_NUM_THREADS=1 "$official_python" -u scripts/visualize_common_candidate_scorer_comparison.py --configuration "$configuration" > "$output/visualize.log" 2>&1
fi
