#!/bin/bash
# feature cache 에서 사라진 토큰만 다시 만든다.
# dataset.py:257 이 force_cache_computation=False 일 때
#   tokens_to_cache = 전체 - 이미 유효한 캐시
# 로 계산하므로, 있는 것은 건드리지 않는다. yaml 기본값이 true 라 반드시 명시해야 한다.
set -e
source /home/kaist5/miniconda3/etc/profile.d/conda.sh
conda activate /home/kaist5/miniconda3/envs/kjs-SafeDrive-exp2
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
export CUDA_VISIBLE_DEVICES=0,1
PY=/home/kaist5/miniconda3/envs/kjs-SafeDrive-exp2/bin/python

FEATURE_CACHE=$BASE/exp/safedrive_train_cache
echo "########## 복구 전 토큰 수 ##########"
find $FEATURE_CACHE -mindepth 2 -maxdepth 2 -type d | wc -l
date

OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
$PY navsim/planning/script/run_dataset_caching.py \
    agent=SafeDrive_Phase3_Planner_FullTrain \
    experiment_name=caching/repair \
    train_test_split=navtrain \
    cache_path=$FEATURE_CACHE \
    force_cache_computation=False \
    worker.threads_per_node=48

date
echo "########## 복구 후 토큰 수 ##########"
find $FEATURE_CACHE -mindepth 2 -maxdepth 2 -type d | wc -l
du -sh $FEATURE_CACHE | cut -f1
echo "CACHE_REPAIR_DONE"
