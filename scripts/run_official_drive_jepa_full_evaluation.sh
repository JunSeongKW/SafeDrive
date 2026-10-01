#!/usr/bin/env bash
# Full official NAVSIM v1 navtest only; no pilot model, training, or parameter tuning.
set -euo pipefail
WORKSPACE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
EVALUATION_PYTHON="$WORKSPACE_ROOT/runtime/environments/drive_jepa_official_evaluation/bin/python"
OFFICIAL_ROOT="$WORKSPACE_ROOT/reference_repositories/DriveJEPAOfficialEvaluation/navsim_v1"
RUN_DIRECTORY="${1:?usage: run_official_drive_jepa_full_evaluation.sh <new-absolute-run-directory>}"
PHYSICAL_GPU="${2:-0}"
EXECUTION_SHARD="${3:-all}"
[[ "$PHYSICAL_GPU" = 0 || "$PHYSICAL_GPU" = 1 ]] || { echo 'Only physical GPU0/1 authorized'; exit 2; }
[[ "$RUN_DIRECTORY" = "$WORKSPACE_ROOT/outputs/official_drive_jepa_reproduction/"* ]] || exit 2
[[ ! -e "$RUN_DIRECTORY" ]] || { echo 'Preserve previous run; supply a new directory'; exit 2; }
"$EVALUATION_PYTHON" -c 'import json,sys; from pathlib import Path; root=Path(sys.argv[1]); assert json.loads((root/"results/official_drive_jepa_reproduction/evaluation_preflight.json").read_text())["complete"]; assert json.loads((root/"results/official_drive_jepa_reproduction/inference_smoke.json").read_text())["failed_scenes"]==0' "$WORKSPACE_ROOT"
CACHE_ROOT="$("$EVALUATION_PYTHON" -c 'import json,sys; print(json.load(open(sys.argv[1]))["metric_cache_root"])' "$WORKSPACE_ROOT/results/official_drive_jepa_reproduction/verified_assets.json")"
export CUDA_VISIBLE_DEVICES="$PHYSICAL_GPU"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1
export NAVSIM_DEVKIT_ROOT="$OFFICIAL_ROOT"
export NAVSIM_EXP_ROOT="$WORKSPACE_ROOT/outputs/official_drive_jepa_reproduction"
export OPENSCENE_DATA_ROOT="$WORKSPACE_ROOT/dataset"
export NUPLAN_MAPS_ROOT="$WORKSPACE_ROOT/dataset/maps"
export NUPLAN_MAP_VERSION=nuplan-maps-v1.0
export PYTHONPATH="$OFFICIAL_ROOT"
SHARD_OVERRIDES=()
WORKER_COUNT=4
if [[ "$EXECUTION_SHARD" != all ]]; then
  [[ "$EXECUTION_SHARD" = 0 || "$EXECUTION_SHARD" = 1 ]] || exit 2
  SHARD_METADATA="$WORKSPACE_ROOT/results/official_drive_jepa_reproduction/evaluation_shard_${EXECUTION_SHARD}.json"
  SHARD_LOG_NAMES="$("$EVALUATION_PYTHON" -c 'import json,sys; print(json.dumps(json.load(open(sys.argv[1]))["log_names"],separators=(",",":")))' "$SHARD_METADATA")"
  SHARD_OVERRIDES=("train_test_split.scene_filter.log_names=$SHARD_LOG_NAMES")
  WORKER_COUNT=2
fi
mkdir -p "$RUN_DIRECTORY"
cd "$OFFICIAL_ROOT"
date --iso-8601=seconds
"$EVALUATION_PYTHON" "$OFFICIAL_ROOT/navsim/planning/script/run_pdm_score.py" \
  train_test_split=navtest agent=drive_jepa_perception_free_agent \
  agent.pretrain_pt_path="$WORKSPACE_ROOT/runtime/checkpoints/drive_jepa/vitl_merge_3dataset_e50.pt" \
  agent.image_architecture=vit_large agent.num_keyval=129 agent.front_only=true \
  agent.tf_dropout=0.0 agent.freeze_encoder=false agent.double_image=true \
  worker=single_machine_thread_pool worker.max_workers="$WORKER_COUNT" worker.use_process_pool=true \
  agent.checkpoint_path="$WORKSPACE_ROOT/runtime/checkpoints/drive_jepa_official_evaluation/drive_jepa_perception_free_agent_vitl.ckpt" \
  metric_cache_path="$CACHE_ROOT" experiment_name=official_drive_jepa_perception_free_vitl_navtest \
  output_dir="$RUN_DIRECTORY" "${SHARD_OVERRIDES[@]}"
date --iso-8601=seconds
echo OFFICIAL_DRIVE_JEPA_EVALUATION_DONE
