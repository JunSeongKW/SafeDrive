#!/bin/bash
set -e
source /home/kaist5/miniconda3/etc/profile.d/conda.sh
conda activate safedrive
BASE=/home/kaist5/data/junseong/SafeDrive
DATA_ROOT=$BASE/dataset
cd $BASE
export PYTHONPATH=$BASE
export NUPLAN_MAP_VERSION=nuplan-maps-v1.0
export NUPLAN_MAPS_ROOT=$DATA_ROOT/maps
export OPENSCENE_DATA_ROOT=$DATA_ROOT
export NAVSIM_EXP_ROOT=$BASE/exp
export NAVSIM_DEVKIT_ROOT=$BASE/navsim
export HYDRA_FULL_ERROR=1
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
date
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
python navsim/planning/script/run_dataset_caching.py \
    agent=SafeDrive_Phase3_Planner_FullTrain \
    experiment_name=caching/safedrive_navmini \
    train_test_split=navmini \
    cache_path=$BASE/exp/feat_cache_navmini \
    worker.threads_per_node=32
date
echo "FEATCACHE_DONE"
