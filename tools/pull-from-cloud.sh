#!/usr/bin/env bash
# 새 서버(AXE-080 등)에서 실행해, 이전 서버(cloud-orO3Hf)의 자산을 당겨온다.
#
#   bash tools/pull-from-cloud.sh <tier>
#     tier 1  필수·소용량   메모리/대화 + 평가 결과 CSV + 재개용 last.ckpt   약 7 GB
#     tier 2  SafeDrive 재개용   사전학습 ckpt + navtest/navmini 캐시       약 8 GB
#     tier 3  전체 재학습용     navtrain metric cache                      약 22 GB
#     tier 4  (권장하지 않음)    navtrain feature cache                    444 GB
#
# tier 1 은 `exp/safedrive` 전체(52 GB)를 받지 않는다. 중간 epoch 체크포인트가 run 당
# 7.8 GB 씩 쌓여 있는데 재개에는 `last.ckpt` 하나만 필요하다.
#
# 왜 pull 인가: cloud-orO3Hf 는 사설 IP(192.168.0.2)만 가진 NAT 뒤 VM 이라 outbound 가
# 막혀 있다. 반대로 외부 IP 61.107.200.100 으로 들어오는 것은 된다(이전에 성공한 경로).
#
# tier 4 를 권하지 않는 이유: 그 캐시는 SafeDrive 의 feature builder 산출물이라
# Drive-JEPA 등 다른 모델이 재사용할 수 없다. JEPA 전환에는 원본 데이터셋만 필요하다.
set -euo pipefail

SRC_HOST="${SRC_HOST:-61.107.200.100}"     # cloud-orO3Hf 의 외부(NAT) IP
SRC_USER="${SRC_USER:-kaist5}"
SRC_PORT="${SRC_PORT:-22}"                 # 포트포워딩이 다르면 SRC_PORT 로 덮어쓴다
SRC_ROOT="${SRC_ROOT:-/home/kaist5/data/junseong/SafeDrive}"
DST_ROOT="${DST_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"

TIER="${1:?usage: pull-from-cloud.sh <1|2|3|4>}"
SSH="ssh -p $SRC_PORT -o StrictHostKeyChecking=accept-new"
# -a 보존, -h 사람이 읽는 크기, --partial 중단 시 이어받기, --info 진행률
R="rsync -ah --partial --info=progress2,stats1 -e"

echo "출처 : $SRC_USER@$SRC_HOST:$SRC_PORT  $SRC_ROOT"
echo "대상 : $DST_ROOT"
echo "tier : $TIER"
echo
echo "-- 연결 확인 --"
$SSH "$SRC_USER@$SRC_HOST" "echo '  OK: '\$(hostname)" || {
  echo "★ 접속 실패. SRC_PORT 를 확인하라(포트포워딩이 22 가 아닐 수 있다):"
  echo "   SRC_PORT=2222 bash $0 $TIER"
  exit 2
}
echo

pull () {  # pull <원격 상대경로> <설명>
  local rel="$1" desc="$2"
  echo "-- $desc  ($rel) --"
  mkdir -p "$DST_ROOT/$(dirname "$rel")"
  $R "$SSH" "$SRC_USER@$SRC_HOST:$SRC_ROOT/$rel" "$DST_ROOT/$(dirname "$rel")/" || \
    echo "  ★ 실패: $rel (건너뜀)"
  echo
}

case "$TIER" in
1)
  # Claude 메모리·대화 (git 으로 오지 않는다. pin 된 연구 명제가 들어 있어 가장 중요)
  echo "-- Claude 메모리·대화 묶음 (2.5 MB) --"
  mkdir -p "$HOME/claude-context"
  $R "$SSH" "$SRC_USER@$SRC_HOST:/home/kaist5/data/junseong/claude-context-*.tar.gz" \
     "$HOME/claude-context/" || echo "  ★ 실패"
  echo "  → 풀기: mkdir -p \$CLAUDE_CONFIG_DIR/projects/<슬러그> && tar xzf ~/claude-context/claude-context-*.tar.gz -C 거기"
  echo

  # 평가 결과 CSV 만 (17 MB). 분석·재현에 실제로 쓰는 것은 이것뿐이다.
  echo "-- 평가 결과 CSV (17 MB) --"
  mkdir -p "$DST_ROOT/exp/safedrive"
  $R "$SSH" --include='*/' --include='traj_*.csv' --exclude='*' \
     "$SRC_USER@$SRC_HOST:$SRC_ROOT/exp/safedrive/" "$DST_ROOT/exp/safedrive/" || echo "  ★ 실패"
  echo

  # 재개용 last.ckpt 만 (중간 epoch 체크포인트는 받지 않는다)
  for run in o0_nofuture f3_nopairnc; do
    echo "-- $run 재개용 last.ckpt (1.3 GB) --"
    d="exp/safedrive/$run/lightning_logs/checkpoints"
    mkdir -p "$DST_ROOT/$d"
    $R "$SSH" "$SRC_USER@$SRC_HOST:$SRC_ROOT/$d/last.ckpt" "$DST_ROOT/$d/" || echo "  ★ 실패(없을 수 있음)"
    echo
  done
  echo "  ※ 중간 epoch 체크포인트(run 당 7.8 GB)는 받지 않았다. 필요하면"
  echo "     rsync 로 exp/safedrive/<run>/lightning_logs/checkpoints/ 를 따로 받는다."
  ;;
2)
  pull "ckpts"                        "사전학습 체크포인트 (3.0 GB)"
  pull "exp/metric_cache_navtest"     "navtest metric cache (3.1 GB) — 평가 필수"
  pull "exp/metric_cache_navmini"     "navmini metric cache (149 MB)"
  pull "exp/feat_cache_navmini"       "navmini feature cache (1.8 GB)"
  ;;
3)
  pull "exp/train_metric_cache_navtrain" "navtrain metric cache (22 GB) — 학습 safety GT"
  ;;
4)
  echo "★ tier 4 는 444 GB 이고 SafeDrive 전용이다. JEPA 작업에는 불필요하다."
  read -rp "  정말 진행하려면 'yes' 입력: " ok
  [ "$ok" = yes ] || { echo "중단"; exit 0; }
  pull "exp/safedrive_train_cache" "navtrain feature cache (444 GB)"
  ;;
*)
  echo "tier 는 1|2|3|4"; exit 2;;
esac

echo "================ tier $TIER 완료 $(date '+%F %H:%M') ================"
echo "중단되면 같은 명령을 다시 실행하면 --partial 로 이어받는다."
