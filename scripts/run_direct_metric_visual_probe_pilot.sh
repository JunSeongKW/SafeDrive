#!/usr/bin/env bash
set -euo pipefail
cd /rhome/junseong/PlanningAwareFuturePrediction
experiment_python=/rhome/junseong/PlanningAwareFuturePrediction/runtime/environments/kjs-nc-dac-visual-probe/bin/python
experiment_configuration=configs/direct_metric_visual_probe/pilot_v1.json
experiment_output=outputs/direct_metric_visual_probe_v1
export PYTHONPATH=/rhome/junseong/PlanningAwareFuturePrediction/src
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 CUBLAS_WORKSPACE_CONFIG=:4096:8
for experiment_phase in prepare validate profile train report; do
  case "$experiment_phase" in
    prepare) experiment_marker=prepare_complete.json ;;
    validate) experiment_marker=validation_complete.json ;;
    profile) experiment_marker=profile_complete.json ;;
    train) experiment_marker=training_complete.json ;;
    report) experiment_marker=report_complete.json ;;
  esac
  if [[ ! -f "$experiment_output/$experiment_marker" ]]; then
    case "$experiment_phase" in
      prepare|validate) CUDA_VISIBLE_DEVICES='' "$experiment_python" -u scripts/run_direct_metric_visual_probe_pilot.py --configuration "$experiment_configuration" --phase "$experiment_phase" ;;
      *) CUDA_VISIBLE_DEVICES=0 "$experiment_python" -u scripts/run_direct_metric_visual_probe_pilot.py --configuration "$experiment_configuration" --phase "$experiment_phase" ;;
    esac
  fi
done
if [[ ! -f "$experiment_output/diagnostics_complete.json" ]]; then
  CUDA_VISIBLE_DEVICES='' "$experiment_python" -u scripts/diagnose_direct_metric_visual_probe.py --configuration "$experiment_configuration"
fi
