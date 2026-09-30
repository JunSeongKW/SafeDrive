#!/usr/bin/env bash
# 새 서버(예: AXE-28)에서 실행해, 무엇이 이미 있고 무엇을 옮겨야 하는지 알려준다.
#
#   bash tools/check-new-server.sh
#
# cloud-orO3Hf(이전 서버)는 사설 IP 뿐이라 양방향 rsync 가 막혀 있었다. 그래서
# 대용량을 옮기기 전에 "이미 있는 것"을 먼저 확인해 불필요한 전송을 줄인다.
set -u
echo "================ 새 서버 현황 ================"
echo "host: $(hostname)   user: $(whoami)   date: $(date '+%F %H:%M')"
echo

echo "---- GPU ----"
nvidia-smi --query-gpu=index,name,memory.total,memory.used --format=csv 2>/dev/null || echo "  nvidia-smi 없음"
echo

echo "---- 디스크 ----"
df -hT . | tail -1
echo

echo "---- NAVSIM 데이터셋이 이미 있나 (가장 중요) ----"
FOUND=""
for d in /home/user/data/Dataset/navsim ~/Dataset/navsim ~/data/Dataset/navsim /data/navsim /mnt/navsim \
         ~/navsim/dataset /workspace/navsim; do
  [ -d "$d" ] && { echo "  발견: $d"; FOUND="$d"; }
done
[ -z "$FOUND" ] && echo "  ★ 못 찾음 — find 로 넓게 찾아본다 (30초)" && \
  timeout 30 find / -maxdepth 5 -type d -name "sensor_blobs" 2>/dev/null | head -3
if [ -n "$FOUND" ]; then
  for sub in maps navsim_logs sensor_blobs navhard_two_stage; do
    p=$(find "$FOUND" -maxdepth 3 -name "$sub" 2>/dev/null | head -1)
    [ -n "$p" ] && printf "    %-26s %s\n" "$sub" "$(du -shL "$p" 2>/dev/null | cut -f1)"
  done
fi
echo

echo "---- conda / 환경 ----"
which conda 2>/dev/null || echo "  conda 없음"
conda env list 2>/dev/null | grep -iE "safedrive|jepa|navsim" || echo "  관련 env 없음 (새로 만들어야 함)"
python -c "import torch,sys; print(f'  torch {torch.__version__}  cuda {torch.version.cuda}  arch {torch.cuda.get_device_capability() if torch.cuda.is_available() else \"-\"}')" 2>/dev/null || echo "  torch 없음"
echo

echo "---- 이 레포가 git 으로 잘 왔나 ----"
for f in HANDOFF.md AGENTS.md RESUME_NOTES.md EXPERIMENT_DESIGN.md; do
  [ -f "$f" ] && printf "  %-24s %s줄\n" "$f" "$(wc -l < $f)" || echo "  ★ $f 없음"
done
printf "  %-24s %s개\n" "scripts/run/"      "$(ls scripts/run 2>/dev/null | wc -l)"
printf "  %-24s %s개\n" "scripts/analysis/" "$(ls scripts/analysis 2>/dev/null | wc -l)"
printf "  %-24s %s개\n" "analysis/*.csv"    "$(ls analysis/*.csv 2>/dev/null | wc -l)"
echo

echo "---- 이전 서버에서 옮겨야 할 것 (없는 것만) ----"
[ -d ckpts ] && [ "$(ls ckpts 2>/dev/null | wc -l)" -gt 0 ] \
  && echo "  ckpts/ 있음" || echo "  ★ ckpts/ (3.0 GB) — SafeDrive 재개 시에만 필요"
[ -d exp/metric_cache_navtest ] \
  && echo "  exp/metric_cache_navtest 있음" || echo "  ★ exp/metric_cache_navtest (3.1 GB) — navtest 평가에 필요"
echo "  ※ exp/safedrive_train_cache (444 GB) 는 SafeDrive 전용이라 JEPA 작업에는 불필요"
echo "  ※ Claude 메모리/대화는 git 으로 오지 않는다 — claude-context-*.tar.gz 를 따로 옮긴다"
echo
echo "================ 끝 ================"
