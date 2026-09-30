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

CONFIG=SafeDrive_Phase3_Planner_FullTrain   # agent_token_ids 를 써야 하므로 phase3 config
FEATURE_CACHE=$BASE/exp/safedrive_train_cache
TRAIN_METRIC_CACHE=$BASE/exp/train_metric_cache_navtrain

echo "########## STEP 1 : navtrain feature cache ##########"; date
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
python navsim/planning/script/run_dataset_caching.py \
    agent=$CONFIG \
    experiment_name=caching/safedrive \
    train_test_split=navtrain \
    cache_path=$FEATURE_CACHE \
    worker.threads_per_node=48
date
echo "  용량: $(du -sh $FEATURE_CACHE | cut -f1)"

echo "########## STEP 2 : navtrain metric cache ##########"; date
python navsim/planning/script/run_train_metric_caching.py \
    train_test_split=navtrain \
    cache.cache_path=$TRAIN_METRIC_CACHE \
    worker.threads_per_node=40
date
echo "  용량: $(du -sh $TRAIN_METRIC_CACHE | cut -f1)"
echo "CACHE_ALL_DONE"
