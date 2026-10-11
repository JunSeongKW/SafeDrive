#!/usr/bin/env bash
set -euo pipefail
SCORER_PILOT_PROJECT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$SCORER_PILOT_PROJECT"
SCORER_PILOT_PYTHON="$SCORER_PILOT_PROJECT/runtime/environments/kjs-scorer-selection-pilot/bin/python"
SCORER_PILOT_CONFIG="configs/scorer_selection_learning/pilot_v1.json"
if [[ ! -e "$SCORER_PILOT_PYTHON" ]]; then
  ln -s kjs-lpwm-drivor-joint runtime/environments/kjs-scorer-selection-pilot
fi
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 PYTHONPATH="$SCORER_PILOT_PROJECT/src"
CUDA_VISIBLE_DEVICES='' "$SCORER_PILOT_PYTHON" -u scripts/build_scorer_selection_cache.py --configuration "$SCORER_PILOT_CONFIG" --phase prepare
if [[ ! -f outputs/scorer_selection_learning_v1/cache_complete.json ]]; then
  CUDA_VISIBLE_DEVICES=0 CUBLAS_WORKSPACE_CONFIG=:4096:8 "$SCORER_PILOT_PYTHON" -u scripts/build_scorer_selection_cache.py --configuration "$SCORER_PILOT_CONFIG" --phase cache
fi
CUDA_VISIBLE_DEVICES='' "$SCORER_PILOT_PYTHON" -u scripts/run_scorer_selection_pilot.py --configuration "$SCORER_PILOT_CONFIG" --phase run
CUDA_VISIBLE_DEVICES='' "$SCORER_PILOT_PYTHON" -u scripts/visualize_scorer_selection_pilot.py --configuration "$SCORER_PILOT_CONFIG"
CUDA_VISIBLE_DEVICES='' "$SCORER_PILOT_PYTHON" -u scripts/analyze_scorer_selection_acquisition.py --configuration "$SCORER_PILOT_CONFIG"
CUDA_VISIBLE_DEVICES='' "$SCORER_PILOT_PYTHON" -u scripts/diagnose_scorer_correction_fit.py --configuration "$SCORER_PILOT_CONFIG"
