#!/bin/bash
# 사용법: train_phase2.sh <CONFIG> <EXP_NAME> <GPUS>
#   예:   train_phase2.sh SafeDrive_Phase2_Planner_FreezePerception phase2_baseline 4,5
set -e
CONFIG=$1; EXP_TAG=$2; GPUS=$3
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
export OMP_NUM_THREADS=8
export CUDA_VISIBLE_DEVICES=$GPUS
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

PHASE1_CKPT=$BASE/ckpts/safedrive_phase1_90ep.ckpt
CACHE_PATH=$BASE/exp/safedrive_train_cache
METRIC_CACHE=$BASE/exp/train_metric_cache_navtrain

echo "########## RESUME Phase2: $CONFIG -> safedrive/$EXP_TAG (GPU $GPUS) ##########"; date
/home/kaist5/miniconda3/envs/kjs-SafeDrive-exp2/bin/python navsim/planning/script/run_training.py \
    agent=$CONFIG \
    experiment_name=safedrive/$EXP_TAG \
    trainer.params.max_epochs=5 \
    agent.checkpoint_path=$PHASE1_CKPT \
    +resume_from=true +ckpt_path=$BASE/exp/safedrive/$EXP_TAG/lightning_logs/checkpoints/last.ckpt \
    train_test_split=navtrain split=navtrain \
    cache_path=$CACHE_PATH \
    use_cache_without_dataset=True force_cache_computation=False \
    dataloader.params.batch_size=${BATCH:-16} dataloader.params.num_workers=4 \
    ++trainer.params.precision=bf16-mixed \
    ++trainer.params.check_val_every_n_epoch=1 \
    +ddp_find_unused_parameters=True \
    +second_lidar=True \
    ++agent.config.safety_metric_cache_path=$METRIC_CACHE \
    ++agent.config.safety_ray_threads=${RAY:-8}
date
echo "TRAIN_DONE -> $NAVSIM_EXP_ROOT/safedrive/$EXP_TAG/lightning_logs/checkpoints"
