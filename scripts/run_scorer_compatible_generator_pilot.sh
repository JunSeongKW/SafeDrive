#!/usr/bin/env bash
set -euo pipefail
cd /rhome/junseong/PlanningAwareFuturePrediction
configuration=configs/scorer_compatible_generator/pilot_v1.json
output=outputs/scorer_compatible_generator_v1
mkdir -p "$output"
export CUDA_VISIBLE_DEVICES=''
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=2
export PYTHONPATH=/rhome/junseong/PlanningAwareFuturePrediction/src
analysis_python=/rhome/junseong/PlanningAwareFuturePrediction/runtime/environments/kjs-lpwm-drivor-joint/bin/python
evaluation_python=/rhome/junseong/PlanningAwareFuturePrediction/runtime/environments/drive_jepa_official_evaluation/bin/python
evaluation_library=/rhome/junseong/PlanningAwareFuturePrediction/runtime/environments/drive_jepa_official_evaluation/lib
[[ -f "$output/cache_complete.json" ]] || { echo 'Run the authorized prepare_scorer_compatible_generator.py cache phase first.' >&2; exit 1; }
if [[ ! -f "$output/native_labels_complete.json" ]]; then
    LD_LIBRARY_PATH="$evaluation_library" "$evaluation_python" -u scripts/run_scorer_compatible_generator.py --configuration "$configuration" --phase label_native > "$output/label_native.log" 2>&1
fi
if [[ ! -f "$output/teacher_and_gradient_complete.json" ]]; then
    "$analysis_python" -u scripts/run_scorer_compatible_generator.py --configuration "$configuration" --phase teachers > "$output/teachers.log" 2>&1
fi
if [[ ! -f "$output/training_complete.json" ]]; then
    "$analysis_python" -u scripts/run_scorer_compatible_generator.py --configuration "$configuration" --phase train > "$output/train.log" 2>&1
fi
if [[ ! -f "$output/prediction_complete.json" ]]; then
    "$analysis_python" -u scripts/run_scorer_compatible_generator.py --configuration "$configuration" --phase predict > "$output/predict.log" 2>&1
fi
if [[ ! -f "$output/evaluation_labels_complete.json" ]]; then
    LD_LIBRARY_PATH="$evaluation_library" "$evaluation_python" -u scripts/run_scorer_compatible_generator.py --configuration "$configuration" --phase label_predictions > "$output/label_predictions.log" 2>&1
fi
"$analysis_python" -u scripts/report_scorer_compatible_generator.py --configuration "$configuration" > "$output/report.log" 2>&1
LD_LIBRARY_PATH="$evaluation_library" "$evaluation_python" -u scripts/visualize_scorer_compatible_generator.py --configuration "$configuration" > "$output/visualization.log" 2>&1
"$analysis_python" -u scripts/diagnose_scorer_compatible_generator_outcomes.py --configuration "$configuration" > "$output/outcome_diagnosis.log" 2>&1
LD_LIBRARY_PATH="$evaluation_library" "$evaluation_python" -u scripts/run_scorer_compatible_generator.py --configuration "$configuration" --phase baseline_precision > "$output/baseline_precision.log" 2>&1
LD_LIBRARY_PATH="$evaluation_library" "$evaluation_python" -u scripts/visualize_scorer_compatible_road_boundary.py --configuration "$configuration" > "$output/road_boundary.log" 2>&1
