#!/bin/bash
# 사용법: train_navmini.sh <CONFIG> <EXP_TAG> <GPUS> <BATCH> <RAY_THREADS> <EPOCHS>
set -e
CONFIG=$1; EXP_TAG=$2; GPUS=$3; BATCH=$4; RAY=$5; EP=$6
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
export OMP_NUM_THREADS=4
export CUDA_VISIBLE_DEVICES=$GPUS
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "### $CONFIG -> $EXP_TAG | GPU=$GPUS batch=$BATCH ray=$RAY ep=$EP"; date
python navsim/planning/script/run_training.py \
    agent=$CONFIG \
    experiment_name=safedrive/$EXP_TAG \
    trainer.params.max_epochs=$EP \
    agent.checkpoint_path=$BASE/ckpts/safedrive_phase1_90ep.ckpt \
    train_test_split=navmini split=mini \
    cache_path=$BASE/exp/feat_cache_navmini \
    use_cache_without_dataset=True force_cache_computation=False \
    dataloader.params.batch_size=$BATCH dataloader.params.num_workers=4 \
    ++trainer.params.precision=bf16-mixed \
    ++trainer.params.check_val_every_n_epoch=1 \
    +ddp_find_unused_parameters=True +second_lidar=True \
    ++agent.config.safety_metric_cache_path=$BASE/exp/metric_cache_navmini \
    ++agent.config.safety_ray_threads=$RAY
date
echo "NAVMINI_TRAIN_DONE"
