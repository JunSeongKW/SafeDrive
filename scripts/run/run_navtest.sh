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
export OMP_NUM_THREADS=8
export CUDA_VISIBLE_DEVICES=7          # 공유 머신: GPU 1장만

CONFIG=SafeDrive_Phase3_Planner_FullTrain
CKPT=$BASE/ckpts/safedrive_phase3_10ep.ckpt
METRIC_CACHE=$BASE/exp/metric_cache_navtest
AGENT_NAME=safedrive_navtest
EXP=safedrive/eval_$AGENT_NAME

echo "########## STEP 0 : navtest metric cache ##########"
date
if [ ! -f "$METRIC_CACHE/metadata/metric_cache_navtest_metadata_node_0.csv" ]; then
  python navsim/planning/script/run_metric_caching.py \
      train_test_split=navtest \
      cache.cache_path=$METRIC_CACHE \
      worker.threads_per_node=40
else
  echo "(이미 존재 -> 건너뜀)"
fi
date

WEIGHTS=(
    +imi_test_weight=0.3  +NC_test_weight=16.0  +DAC_test_weight=48.0
    +EP_test_weight=0.75  +TTC_test_weight=15.0 +PwNC_test_weight=5.0
    +TwDAC_test_weight=1.5 +W_test_weight=1.0   +pdm_score_test_weight=1.0
    +DDC_test_weight=1.0  +TLC_test_weight=1.0  +LK_test_weight=1.0
    +TwDAC_bbox_margin=[1.4,1.6] +no_EP_TC_sum_scoring=True
)

VAL_LOGS=$(python3 -c "
import os
logs = sorted(f[:-4] for f in os.listdir('$DATA_ROOT/navsim_logs/test') if f.endswith('.pkl'))
print('[' + ','.join(logs) + ']')")

mkdir -p $NAVSIM_EXP_ROOT/training/$AGENT_NAME

echo "########## STAGE 1 : GPU forward (navtest 12,146) ##########"
date
python navsim/planning/script/run_training.py \
    agent=$CONFIG \
    train_test_split=navtest split=test \
    experiment_name=safedrive/eval_fwd_$AGENT_NAME \
    cache_path=null use_cache_without_dataset=False force_cache_computation=False \
    dataloader.params.batch_size=16 dataloader.params.num_workers=8 \
    trainer.params.max_epochs=1 ++trainer.params.precision=32 \
    ++train_logs=$VAL_LOGS ++val_logs=$VAL_LOGS \
    +debug=false +second_lidar=True +ddp_find_unused_parameters=True \
    +ckpt_path=$CKPT +agent_name=$AGENT_NAME \
    +test_traj_save=True +test_save_name=traj \
    +metric_cache_path=$METRIC_CACHE \
    +scoring_test=True +pair_NC_scoring=true +twdac_scoring=true +TwDAC_bevseg_pred=true \
    "${WEIGHTS[@]}"
date

TRAJ_PKL=$(ls -t $NAVSIM_EXP_ROOT/training/$AGENT_NAME/traj_*.pkl | head -1)
echo "########## STAGE 2 : PDM scoring ##########"
date
python navsim/planning/script/run_evaluation_gpu.py \
    agent=$CONFIG experiment_name=$EXP \
    train_test_split=navtest \
    metric_cache_path=$METRIC_CACHE \
    worker.threads_per_node=40 \
    pred_traj_path=$TRAJ_PKL \
    +second_lidar=True +save_csv=result.csv
date
echo "NAVTEST_DONE -> $(ls -t $NAVSIM_EXP_ROOT/$EXP/*.csv 2>/dev/null | head -1)"
