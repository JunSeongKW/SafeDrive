#!/bin/bash
# queue.sh <GPU_A> <GPU_B> <TAG:CONFIG> [TAG:CONFIG ...]
#
# 한 레인(GPU 2장)에서 항목을 순서대로 돌린다. 항목마다
#   ① 두 GPU 가 실제로 빌 때까지 대기
#   ② 체크포인트가 있으면 재개, 없으면 신규 학습
#   ③ navtest 평가
# 를 수행한다.  로그 경로를 규칙으로 고정해(train_<TAG>.log / ev_<TAG>.log) 파일명을
# 추측하다 무한 대기하는 과거 실수를 막는다.
set -u
S="${SD_SCRIPTS:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
BASE=/home/kaist5/data/junseong/SafeDrive
GA=$1; GB=$2; shift 2
used () { nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$1"; }
wait_gpus () { while :; do [ "$(used $GA)" -lt 2000 ] && [ "$(used $GB)" -lt 2000 ] && break; sleep 60; done; }

for item in "$@"; do
  TAG=${item%%:*}; CFG=${item#*:}
  CKDIR=$BASE/exp/safedrive/$TAG/lightning_logs/checkpoints
  echo "=== [$TAG] 시작 $(date '+%m-%d %H:%M') ==="
  if grep -q TRAIN_DONE $S/train_$TAG.log 2>/dev/null; then
    echo "[$TAG] 학습 이미 완료 — 건너뜀"
  else
    wait_gpus
    cd $BASE
    if [ -f "$CKDIR/last.ckpt" ]; then
      echo "[$TAG] 체크포인트 발견 → 재개 $(date '+%H:%M')"
      BATCH=24 RAY=24 bash $S/resume_phase2.sh "$CFG" "$TAG" "$GA,$GB" > $S/train_$TAG.log 2>&1
    else
      echo "[$TAG] 신규 학습 $(date '+%H:%M')"
      BATCH=24 RAY=24 bash $S/train_phase2.sh "$CFG" "$TAG" "$GA,$GB" > $S/train_$TAG.log 2>&1
    fi
    echo "[$TAG] 학습 종료 $(date '+%m-%d %H:%M')"
    # 학습이 정말 끝났는지 확인한다. 외부 SIGTERM 이나 크래시로 죽어도 위 명령은
    # 조용히 반환되므로, TRAIN_DONE 없이 평가로 넘어가면 덜 학습된 체크포인트를
    # 채점해 잘못된 결과를 만든다(2026-09-28 에 실제로 그럴 뻔했다).
    if ! grep -q TRAIN_DONE $S/train_$TAG.log 2>/dev/null; then
      LASTEP=$(tr '\r' '\n' < $S/train_$TAG.log | grep -oE "Epoch [0-9]+:" | tail -1)
      echo "★ [$TAG] 학습이 완료 표시 없이 끝났다 ($LASTEP). 평가하지 않고 큐를 멈춘다."
      echo "★ 재개하면 마지막 체크포인트에서 이어진다."
      exit 1
    fi
  fi
  if grep -q EVAL_DONE $S/ev_$TAG.log 2>/dev/null; then
    echo "[$TAG] 평가 이미 완료 — 건너뜀"
  else
    echo "[$TAG] 평가 시작 $(date '+%H:%M')"
    bash $S/eval_ckpt.sh "$CFG" "$CKDIR/last.ckpt" "${TAG}_ev" navtest "$GA" > $S/ev_$TAG.log 2>&1
    echo "[$TAG] 평가 종료 $(date '+%m-%d %H:%M')"
  fi
done
echo "=== 큐 전부 완료 $(date '+%m-%d %H:%M') ==="
