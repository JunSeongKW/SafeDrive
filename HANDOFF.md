# HANDOFF — 이 파일 하나로 다음 에이전트가 이어받는다

마지막 갱신: 2026-09-23 21:44 KST (Codex)

세션 **시작**: 이 파일 + `git log -10` + `AGENTS.md`. 세션 **끝**: 이 파일 갱신 + `tools/handoff-commit.sh`.
상세 실험 일지는 `RESUME_NOTES.md`(2026-09-17~20, 시간순), 설계·근거는 `EXPERIMENT_DESIGN.md`.

## 1. 실행 중인 작업

없음 (2026-09-23 21:35 확인: SafeDrive 프로세스 0개). GPU 0-3 은 AlpaSim 평가가 쓰고 있고 4-7 은 다른 연구원 작업이 점유 중.

## 2. 최근 결과 (요약 — 상세는 RESUME_NOTES.md)

- baseline 재현: navtest PDMS 90.96 (논문 91.6), navmini 93.59.
- α(보행자를 sparse world 에 포함): 효과 없음, 가설 기각 (RESUME_NOTES "α 실험 최종 결과").
- E5 EP 테스트 가중치 스윕, E2 perception 실제 동결: ΔEP 가 난이도 전 구간 −2.5~−3.0 으로 균일 → EP 손실은 상황 의존이 아니라 전역 편향. 난이도 정의 5종 비교표 있음.
- F1·F2 평가 CSV는 기준선/E2와 동일한 12,147개 유효 토큰을 담는다. 분해 전 짝차 진단값: F1 ΔPDMS +0.249 (95% 구간 −0.011~+0.509), F2 −0.327 (−0.570~−0.083), F2 ΔEP −0.597. 난이도 B·E 및 시나리오 유형으로 분해하기 전에는 연구 결론을 내리지 않는다. 상세는 RESUME_NOTES.md 4차.
- F3(pair-NC 제거)는 epoch 0 체크포인트만 있고 F4(TW-DAC 제거)는 체크포인트가 없다. 두 run 모두 현재 프로세스가 없으며 평가 CSV도 없다. 완료로 취급하지 않는다.
- 체크포인트: ckpts/ = safedrive_phase1_90ep.ckpt,safedrive_phase2_5ep.ckpt safedrive_phase3_10ep.ckpt

## 3. 마지막 커밋 이후 바뀐 것

- F1/F2 평가 CSV를 기준선·E2와 토큰별로 대조해 표본 일치와 지표별 짝차·구간을 검증했다. RESUME_NOTES.md 4차에 진단 표와 해석 한계를 기록했다.
- F3/F4 프로세스·체크포인트·평가 파일 상태를 확인했다. F3는 epoch 0까지, F4는 체크포인트 없음. 완료로 오인하지 않도록 HANDOFF를 정정했다.
- 기존 난이도 라벨 `scratchpad/navtest_labels.csv`가 현재 경로에서 없어 B·E·시나리오 유형별 분해는 다음 작업으로 남겼다. GPU 4-7 점유 중이라 학습은 시작하지 않았다.

## 4. 다음 단계

1. `scratchpad/navtest_labels.csv`를 찾아 복구하거나 navtest에서 동일한 기준으로 재생성한다. F1·F2·E2를 난이도 정의 B·E와 시나리오 유형별로 **한 표**에 분해한다. 라벨 정의와 토큰 수를 검증하고 baseline 점수 계층화는 쓰지 않는다.
2. F3/F4 중단 원인을 확인하고 2 GPU 이상이 확보되면 학습을 재개·평가한다. F3는 epoch 0 체크포인트가 있고 F4는 없다. batch 8/GPU, 평가 자동 체이닝, nvidia-smi 프로세스 이름 규칙을 적용한다.
3. EXPERIMENT_DESIGN.md 의 남은 셀을 선택하기 전, 각 실험이 "상황마다 필요한 미래 정보가 다른가"의 어떤 하위 질문에 답하는지 한 줄로 명시한다.

## 5. 미결 질문 (사용자 결정 필요)

- PDM 롤아웃 병목 대응: num_proposal_2stage 128→32 로 baseline·실험 양쪽 동일 단축할지 (절대 성능은 낮아짐).
- F1/F2의 난이도·시나리오 분해가 끝난 뒤 다음 실험 축을 정보 종류 라우팅(상황별 선택)과 예산 스윕(K=1..N) 중 어디에 둘지.

## 6. 다른 서버에서 재구성 (git 으로 오지 않는 것)

| 항목 | 이 서버 위치 | 옮기는 방법 |
|---|---|---|
| conda env `safedrive` (6.7 GB) | /home/kaist5/miniconda3/envs/safedrive | docs/install.md 로 재설치. mmcv 2.1.0 은 대상 GPU arch 로 소스빌드(sm_90 = H100) |
| 체크포인트 3.0 GB | ckpts/safedrive_phase{1_90ep,2_5ep,3_10ep}.ckpt | rsync |
| 데이터셋 | dataset -> /home/kaist5/Dataset/navsim/dataset | 대상 서버의 navsim 데이터 경로로 심볼릭 링크 |
| navtrain feature cache 444 GB, metric cache 22 GB | exp/safedrive_train_cache, exp/train_metric_cache_navtrain | rsync 하거나 cache.sh 로 재생성(수 시간~일) |
| navtest/navmini metric·feature cache 5 GB | exp/metric_cache_navtest, exp/feat_cache_navmini, exp/metric_cache_navmini | rsync |
| 실험 산출물 | exp/safedrive/*, exp/training/* | 결과 CSV(traj_*.csv)만 rsync 하면 충분 |

git 으로 오는 것: 코드, 설정 yaml, 스크립트, RESUME_NOTES/EXPERIMENT_DESIGN/HANDOFF, trajectory_anchors.
