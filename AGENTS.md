# 이 저장소에서 에이전트가 지킬 것 (SafeDrive, junseong 작업 규칙)

Claude Code 는 `CLAUDE.md`(= `@AGENTS.md`)로, Codex 는 이 파일로 같은 내용을 읽는다.
**현재 상태**는 `HANDOFF.md`, 상세 실험 기록은 `RESUME_NOTES.md`(시간순 실험 일지)와
`EXPERIMENT_DESIGN.md`(실험 설계·근거) 에 있다. 이 파일에는 바뀌지 않는 규칙만 둔다.

## 세션 시작 루틴

```bash
git status --short --branch && git log --oneline -10
cat HANDOFF.md
ps -eo pid,etimes,args --no-headers | grep -E 'kjs-SafeDriv[e]|safedriv[e]/bin/python' | cut -c1-140
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv
ls -lt exp/safedrive | head; ls -lt exp/training | head
```

## 세션 종료 루틴

1. `HANDOFF.md` 의 1(실행 중)·2(결과)·3(변경)·4(다음)·5(미결) 절을 갱신하고, 실험 결과의
   상세는 `RESUME_NOTES.md` 맨 아래에 날짜 절로 덧붙인다(기존 절은 고치지 않는다).
2. 커밋한다: `tools/handoff-commit.sh "<에이전트 이름>" "[exp] 제목 한 줄"`. 그리고 `git push mine` (원격 `mine` = GitHub JunSeongKW, 다른 서버가 같은 상태를 보도록). HANDOFF.md 3절이
   커밋 본문이 된다. 접두어는 `[exp]` `[code]` `[infra]`. `junseong/*` 브랜치에만 커밋한다(스크립트가 거부).
3. 장시간 학습·평가는 nohup 으로 띄우고, 끝나면 다음 작업이 자동으로 이어지도록 체이닝한다
   (`until grep -q EVAL_DONE <로그>` 방식; 프로세스 이름 grep 은 자기 명령줄과 일치해 무한 대기한다).

## 바뀌지 않는 규칙

- **연구 명제**: "planning 에 필요한 미래 정보는 상황마다 다르고 사람이 미리 정하면 안 된다."
  모든 실험 제안은 이 명제의 어느 하위 질문에 답하는지 한 줄로 밝힌다. 가설이 기각되면 다음
  가설은 원래 명제의 하위 질문 중에서 고른다. 효과가 큰 곁가지보다 원래 질문을 겨냥한 실험이 우선.
- **평가는 집계 점수를 쓰지 않는다**: 난이도 계층·시나리오 유형별 분해로 보고한다. 난이도 정의를
  여러 개 두고 정의를 바꿔도 유지되는 것만 주장한다. baseline 점수로 계층을 나눈 뒤 비교하는 것은
  평균 회귀 때문에 무효(RESUME_NOTES "난이도 정의가 결론을 바꾼다").
- **결과 보고**: 성능 향상 크기가 아니라 논문 기여가 되는지를 기준으로 말한다. 여러 실험은 한 표로 모은다.
- **GPU**: 이 계정의 카드는 4~7. 0~3 은 사용자가 명시적으로 허락한 경우에만. util 0% 는 비어 있다는
  뜻이 아니다. H100 80 GB 기준 Phase 2 학습은 GPU 당 batch 8 이 적정(16 은 OOM). 학습은 2 GPU 이상
  (baseline 이 2 GPU × batch 24 라 1 GPU 면 유효 배치가 깨진다).
- **프로세스 이름**: nvidia-smi 에 식별되도록 conda env 심볼릭 링크(`kjs-SafeDrive-exp2` 식)로
  python 을 절대경로 호출한다.
- **공유 머신**: 다른 연구원(junhyeok, hanbin, dogun, uisung)의 프로세스·컨테이너·폴더는 건드리지
  않는다. `pkill -f` 금지, PID 를 먼저 확인하고 죽인다. 대량 삭제는 `.trash-*/` 로 옮겼다가 실행 중인
  작업이 없을 때 지운다.
- **환경**: conda env `safedrive`(py3.10 / torch 2.1.0+cu121 / mmcv 2.1.0 소스빌드 sm_90 / spconv-cu120 /
  mmdet 3.2.0). 새 의존성은 새 env 에.
- **데이터 안전(AXE-080)**: `/home/user/data/Dataset/` 전체는 연구실 공용 원본이다. 읽기와 프로젝트
  심볼릭 링크만 허용하며, 그 안의 파일·디렉터리를 직접 생성·수정·이동·이름 변경·삭제하지 않는다.
  SafeDrive 는 `dataset -> /home/user/data/Dataset/navsim` 링크로 읽는다. 데이터 변환 결과, metric/feature
  cache, 새로 다운로드하는 데이터셋은 모두 개인 경로 `/home/user/data/processed_dataset/junseong/` 아래에 둔다.
- **실행 방식**: 확인 질문으로 멈추지 말고 합리적 기본값으로 진행한 뒤 가정을 결과와 함께 보고한다.
  되돌릴 수 없는 삭제만 예외.
- **경로**: 체크포인트 `ckpts/`, 캐시·학습 산출물 `exp/`(499 GB, git 밖), 설정
  `navsim/planning/script/config/common/agent/SafeDrive_Phase2_*.yaml`, 결과 CSV `exp/safedrive/eval_*/traj_*.csv`.
