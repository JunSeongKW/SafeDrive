#!/usr/bin/env bash
set -euo pipefail
cd /rhome/junseong/PlanningAwareFuturePrediction
experiment_python=/rhome/junseong/PlanningAwareFuturePrediction/runtime/environments/kjs-scorer-selection-pilot/bin/python
experiment_configuration=configs/metric_visual_evidence/pilot_v1.json
experiment_output=outputs/metric_visual_evidence_v1
export PYTHONPATH=/rhome/junseong/PlanningAwareFuturePrediction/src
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 CUBLAS_WORKSPACE_CONFIG=:4096:8
if [[ ! -f "$experiment_output/prepare_complete.json" ]]; then
  CUDA_VISIBLE_DEVICES='' "$experiment_python" -u scripts/run_metric_visual_evidence_pilot.py --configuration "$experiment_configuration" --phase prepare
fi
for experiment_phase in cache validate profile train report; do
  case "$experiment_phase" in
    cache) experiment_marker=patch_cache_complete.json ;;
    validate) experiment_marker=validation_complete.json ;;
    profile) experiment_marker=profile_complete.json ;;
    train) experiment_marker=training_complete.json ;;
    report) experiment_marker=report_complete.json ;;
  esac
  if [[ ! -f "$experiment_output/$experiment_marker" ]]; then
    CUDA_VISIBLE_DEVICES=0 "$experiment_python" -u scripts/run_metric_visual_evidence_pilot.py --configuration "$experiment_configuration" --phase "$experiment_phase"
  fi
done
if [[ ! -f "$experiment_output/diagnostics_complete.json" ]]; then
  CUDA_VISIBLE_DEVICES='' "$experiment_python" -u scripts/diagnose_metric_visual_evidence.py --configuration "$experiment_configuration"
fi
