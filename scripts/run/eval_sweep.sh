#!/bin/bash
# eval_ckpt.sh <CONFIG> <CKPT> <TAG> <SPLIT: navmini|navtest> <GPU>
set -e
CONFIG=$1; CKPT=$2; TAG=$3; SPLIT=$4; GPU=$5
source /home/kaist5/miniconda3/etc/profile.d/conda.sh
conda activate /home/kaist5/miniconda3/envs/kjs-SafeDrive-exp2
BASE=/home/kaist5/data/junseong/SafeDrive; DATA_ROOT=$BASE/dataset; cd $BASE
export PYTHONPATH=$BASE NUPLAN_MAP_VERSION=nuplan-maps-v1.0 NUPLAN_MAPS_ROOT=$DATA_ROOT/maps
export OPENSCENE_DATA_ROOT=$DATA_ROOT NAVSIM_EXP_ROOT=$BASE/exp NAVSIM_DEVKIT_ROOT=$BASE/navsim
export HYDRA_FULL_ERROR=1 PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python OMP_NUM_THREADS=8
export CUDA_VISIBLE_DEVICES=$GPU PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=/home/kaist5/miniconda3/envs/kjs-SafeDrive-exp2/bin/python

if [ "$SPLIT" = "navmini" ]; then
  LOGDIR=mini; TTS=navmini; SPL=mini; MC=$BASE/exp/metric_cache_navmini
else
  LOGDIR=test; TTS=navtest; SPL=test; MC=$BASE/exp/metric_cache_navtest
fi
W=( +imi_test_weight=0.3 +NC_test_weight=16.0 +DAC_test_weight=48.0 +EP_test_weight=${EPW:-0.75}
    +TTC_test_weight=15.0 +PwNC_test_weight=5.0 +TwDAC_test_weight=1.5 +W_test_weight=1.0
    +pdm_score_test_weight=1.0 +DDC_test_weight=1.0 +TLC_test_weight=1.0 +LK_test_weight=1.0
    +TwDAC_bbox_margin=[1.4,1.6] +no_EP_TC_sum_scoring=True )
VAL_LOGS=$($PY -c "
import os; logs=sorted(f[:-4] for f in os.listdir('$DATA_ROOT/navsim_logs/$LOGDIR') if f.endswith('.pkl'))
print('['+','.join(logs)+']')")
mkdir -p $NAVSIM_EXP_ROOT/training/$TAG
echo "### STAGE1 $TAG ($SPLIT)"; date
$PY navsim/planning/script/run_training.py agent=$CONFIG \
  train_test_split=$TTS split=$SPL experiment_name=safedrive/fwd_$TAG \
  cache_path=null use_cache_without_dataset=False force_cache_computation=False \
  dataloader.params.batch_size=16 dataloader.params.num_workers=4 \
  trainer.params.max_epochs=1 ++trainer.params.precision=32 \
  ++train_logs=$VAL_LOGS ++val_logs=$VAL_LOGS \
  +debug=false +second_lidar=True +ddp_find_unused_parameters=True \
  +ckpt_path=$CKPT +agent_name=$TAG +test_traj_save=True +test_save_name=traj \
  +metric_cache_path=$MC +scoring_test=True +pair_NC_scoring=true +twdac_scoring=true \
  +TwDAC_bevseg_pred=true "${W[@]}"
TRAJ=$(ls -t $NAVSIM_EXP_ROOT/training/$TAG/traj_*.pkl | head -1)
echo "### STAGE2 $TAG"; date
$PY navsim/planning/script/run_evaluation_gpu.py agent=$CONFIG \
  experiment_name=safedrive/eval_$TAG train_test_split=$TTS metric_cache_path=$MC \
  worker.threads_per_node=32 pred_traj_path=$TRAJ +second_lidar=True +save_csv=result.csv
date; echo "EVAL_DONE_$TAG"
