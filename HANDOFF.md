# HANDOFF — 이 파일 하나로 다음 에이전트가 이어받는다

마지막 갱신: 2026-09-23 21:23 KST (Claude Code)

세션 **시작**: 이 파일 + `git log -10` + `AGENTS.md`. 세션 **끝**: 이 파일 갱신 + `tools/handoff-commit.sh`.
상세 실험 일지는 `RESUME_NOTES.md`(2026-09-17~20, 시간순), 설계·근거는 `EXPERIMENT_DESIGN.md`.

## 1. 실행 중인 작업

없음 (2026-09-23 21:20 확인: SafeDrive 프로세스 0개). GPU 0-3 은 AlpaSim 평가가 쓰고 있고 4-7 은 다른 연구원 작업이 점유 중.

## 2. 최근 결과 (요약 — 상세는 RESUME_NOTES.md)

- baseline 재현: navtest PDMS 90.96 (논문 91.6), navmini 93.59.
- α(보행자를 sparse world 에 포함): 효과 없음, 가설 기각 (RESUME_NOTES "α 실험 최종 결과").
- E5 EP 테스트 가중치 스윕, E2 perception 실제 동결: ΔEP 가 난이도 전 구간 −2.5~−3.0 으로 균일 → EP 손실은 상황 의존이 아니라 전역 편향. 난이도 정의 5종 비교표 있음.
- F1(미래 BEV 감독 제거)·F2(motion 감독 제거) 학습·평가 완료 폴더 존재. F3(pair-NC 제거)·F4(TW-DAC 제거) 는 9/21 에 시작됐으나 **RESUME_NOTES 에 결과 미기록**:
  - exp/safedrive/eval_f1_nofutbev_ev: 37 파일, 최신 run_evaluation_gpu.log
  - exp/safedrive/eval_f2_nomotionsup_ev: 37 파일, 최신 run_evaluation_gpu.log
  - exp/safedrive/f3_nopairnc: 9 파일, 최신 lightning_logs
  - exp/safedrive/f4_notwdac: 7 파일, 최신 train_ddp_process_1.log
- 체크포인트: ckpts/ = safedrive_phase1_90ep.ckpt,safedrive_phase2_5ep.ckpt safedrive_phase3_10ep.ckpt

## 3. 마지막 커밋 이후 바뀐 것

- AGENTS.md 종료 루틴에 `git push mine` 추가 (원격: GitHub JunSeongKW, deploy key `~/.ssh/id_ed25519_junseong*`, ssh 별칭 `github-junseong`, `github-junseong-safedrive`).
- 미커밋이던 코드(9/13~9/16: transformer_encoder, backbone, features, config/loss/model 의 include_pedestrian 분기 등 6파일 +200/−35)와 Phase2 설정 7종(Alpha, E2, E3, F1~F4), RESUME_NOTES.md, EXPERIMENT_DESIGN.md 를 이 커밋에 포함.
- AGENTS.md / CLAUDE.md / HANDOFF.md / tools/handoff-commit.sh 추가 (에이전트 인수인계 구조).

## 4. 다음 단계

1. F1~F4 결과를 RESUME_NOTES.md 에 정리하고 E2 와 함께 **한 표**로 (난이도 정의 B·E 분해 포함). 위 폴더의 traj_*.csv 확인.
2. EXPERIMENT_DESIGN.md 의 실험 행렬 중 남은 셀 결정 — 각 실험이 명제의 어느 하위 질문에 답하는지 한 줄 먼저.
3. 학습을 다시 걸 때: 2 GPU 이상, batch 8/GPU, 종료 후 평가 자동 체이닝, nvidia-smi 프로세스 이름 규칙.

## 5. 미결 질문 (사용자 결정 필요)

- PDM 롤아웃 병목 대응: num_proposal_2stage 128→32 로 baseline·실험 양쪽 동일 단축할지 (절대 성능은 낮아짐).
- 다음 실험 축: 정보 종류 라우팅(상황별 선택)으로 갈지, 예산 스윕(K=1..N)으로 갈지.

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
