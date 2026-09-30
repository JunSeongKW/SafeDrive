# HANDOFF — 이 파일 하나로 다음 에이전트가 이어받는다

마지막 갱신: 2026-09-30 11:00 KST (Claude Opus 5)

세션 **시작**: 이 파일 + `git log -10` + `AGENTS.md`. 세션 **끝**: 이 파일 갱신 + `tools/handoff-commit.sh`.
상세 실험 일지는 `RESUME_NOTES.md`(2026-09-17~29, 시간순), 설계·근거는 `EXPERIMENT_DESIGN.md`.

## 0. ★ 연구 플랫폼이 바뀌었다 — SafeDrive → JEPA 계열 (2026-09-29 사용자 결정)

사용자가 **"SafeDrive 는 변경 가능 범위가 좁아서 새로운 novelty 를 넣기가 어렵다"** 고
판단해 **JEPA 기반으로 전환**하기로 했다. 근거: SafeDrive 에서 돌린 실험 5건이 전부
loss 가중치·개수·불리언 하나를 바꾼 것이고 아키텍처를 건드린 것이 없었다.

- **SafeDrive 결과는 논문의 motivation 근거로만 쓴다.** 주 실험으로 쓰지 않는다.
- 새 실험을 제안할 때 SafeDrive 를 기본 플랫폼으로 가정하지 말 것.
- JEPA 가 명제에 맞는 이유: **"무엇을 예측할지"가 마스크로 표현된다.** 기존 JEPA 마스크
  (random, multi-block, progressive schedule, object-level)는 전부 입력과 무관하게 사람이
  고정한 것이므로, **마스크를 주행 맥락 조건부로 planning objective 가 학습**하게 하는 것이
  명제의 두 질문(사람이 정하지 않기 / 상황마다 다르기)을 동시에 겨냥한다.

## 1. 실행 중인 작업

**없음** (2026-09-30 확인). GPU 0~3 비어 있고 4~7 은 junhyeok 의 pycena/physics_server 스택 점유.

중단된 큐: `scripts/run/queue.sh 0 1` 로 8개 항목(O0 → F3 → F4 → F2b → O1×4)을 돌리던 중
사용자가 다른 작업을 위해 여러 번 중단했다. **O0 는 epoch 1 체크포인트까지 있다**(3 epoch 남음).
JEPA 전환에 따라 **O0 · O1×4 는 취소를 권한다**(미래 표현 4종 틀이 JEPA 에서 성립하지 않음).
**F3·F4 만 남기면 "동적 vs 정적" 대비가 JEPA 마스크 설계의 직접 근거가 된다**(각 약 13시간).

## 2. 최근 결과 (요약 — 상세는 RESUME_NOTES.md)

navtest 12,147 전수, 각 조건 Phase 2 를 5 epoch 재학습(batch 24 × 2 GPU 고정).

| 실험 | PDMS | Δ vs P2 | 뜻 |
|---|---:|---:|---|
| phase3 논문 재현 | 90.96 | +1.96 | 논문 91.6 |
| **P2 baseline** | **89.00** | — | 기준 |
| α 보행자 **추가** | 89.09 | +0.09 | 더 넣어도 효과 없음 |
| **F1 미래 BEV 제거** | **89.25** | **+0.25** | **빼도 손해 없음** |
| F2 agent궤적 제거\* | 88.67 | −0.33 | \*pair_Disp 누출 있던 버전 |
| E3 월드 25→5 | 88.62 | −0.38 | 1/5 로 줄여도 0.38 |
| E2 perception 114M→19M | 87.98 | −1.02 | 6배 줄여도 1.02 |

- **5개 실험 전부 DAC 가 올랐다**(+0.29~+0.63). 용량이 남아 planning loss 에 과적합되며
  도로 구조 이해를 망치고 있었다.
- 손실의 88.7% 가 EP 하나. 안전 지표는 97~99.5 로 포화. 충돌 agent 는 98.1% 가 이미 월드 안.
- scoring 상수 하나(EP_test_weight 0.75→300)가 **+1.22** — 95M 파라미터 추가 학습(+1.02)보다 크다.
  단 하이퍼파라미터 튜닝이므로 기여로 주장하지 않는다.
- ★ **상황별 요구 정보 차이 확인**(2026-09-28). 입력 기반 맥락 축으로 분해하니 단조 추세 3건과
  부호 역전이 나왔다. 특히 **월드 25명이 직진에서는 필요(−0.67★)하고 좌회전에서는 불필요(+0.20)**.
  3차에서 "미확립" 이라 보고한 것은 *난이도* 한 축으로만 봤기 때문이었다.

## 3. 마지막 커밋 이후 바뀐 것

- **`/tmp` 스크래치패드에만 있던 자산을 레포로 옮겼다.** 다른 서버로 넘어갈 때 작업이
  끊기던 가장 큰 구멍이었다. `scripts/run/`(14개 실행 스크립트), `scripts/analysis/`(10개 분석
  스크립트), `analysis/`(7개 라벨·측정 CSV, 3.5 MB). 하드코딩된 `/tmp/claude-1000/...` 경로를
  `SD_SCRIPTS` / `SD_LOGS` / `SD_ANALYSIS` 환경변수 기반으로 바꿨고 문법 검사를 통과했다.
  이전 HANDOFF 4절의 "`navtest_labels.csv` 를 찾을 수 없다" 는 이 경로 문제였고, 이제
  `analysis/navtest_labels.csv` 로 레포에 있다.
- **연구 플랫폼을 JEPA 계열로 전환하기로 했다**(0절). SafeDrive 결과는 motivation 으로만 쓴다.
- **맥락 축을 사전 확정했다**(`scripts/analysis/context_axes.py` → `analysis/context_axes.csv`).
  입력만으로 7축(회전·agent밀도·보행자·ego속도·요구진행량·곡률·정적물)을 뽑았다. 모델 성능으로
  계층을 나누면 평균 회귀로 허상이 생긴다(3차에서 baseline 점수 3분위가 +4.93 이라는 허상을 만들었다).
- **"상황별로 필요한 정보가 다르다" 를 예비 확인했다**(`scripts/analysis/context_check.py`).
  단조 추세 3건이 핵심 증거. 다중비교(약 76 검정) 때문에 개별 ★은 과신하지 않는다.
- **오염 요인 2건을 고쳤다.** (a) feature cache 4,321 토큰 소실(step/epoch 1774→1695 이 단서),
  `force_cache_computation=False` 로 15분에 복구. (b) `pair_Disp` 가 `prediction_loss_weight` 와
  무관하게 `gt_motion_traj` 를 계속 감독하던 누출 — motion 을 끄는 5개 config 에
  `pair_Disp_loss_weight: 0.0` 추가.
- **queue.sh 가 학습 실패를 감지하지 못하던 버그를 고쳤다.** O0 가 외부 SIGTERM 으로 죽었는데
  큐가 그것을 모르고 epoch=1 체크포인트로 평가를 시작했다. 이제 `TRAIN_DONE` 을 확인하고
  없으면 평가하지 않고 멈춘다. 긴 작업은 `setsid` 로 띄워 셸 정리에 휩쓸리지 않게 한다.
- 선행연구 정독: **DA-WAM(2608.19085) NAVSIM v1 93.7 PDMS — SafeDrive(91.6)는 더 이상 SOTA 가 아니다.**
  PerceptDrive(2607.20175)가 가장 가깝지만 현재 프레임 prior 를 라우팅하고 future 는 단일 latent,
  예산 스윕 없음, 저자가 *"probes reliance rather than isolated causal effects"* 라고 한계를 자인.

## 4. 다음 단계

1. **JEPA 선행연구 3건 정독 — "마스크를 무엇이 정하는가" 표로 정리.** 최우선.
   `Causal-JEPA`(2602.11389, Object-Level Latent Masking — 최근접 선행, 맥락 조건부인지 확인),
   `AD-E2E-JEPA`(2609.34085, 3주 전), `Drive-JEPA`(2601.22032, **코드 공개**).
2. **Drive-JEPA 를 작업 베이스라인으로 검토.** 코드를 받아 데이터 파이프라인이 NAVSIM 과
   호환되는지, 기존 캐시를 재사용할 수 있는지 확인하고 보고된 수치를 재현한다.
   DA-WAM 은 SOTA 지만 질문 1 의 최대 경쟁자여서 그 위에 쌓으면 delta 가 애매해진다.
3. **F3·F4 만 재개**(각 약 13시간, 2 GPU). `scripts/run/queue.sh <gpuA> <gpuB> f3_nopairnc:SafeDrive_Phase2_F3_NoPairNC f4_notwdac:SafeDrive_Phase2_F4_NoTwDAC`.
   체크포인트가 있으면 자동 재개, `TRAIN_DONE` 있으면 건너뜀. 결과는 JEPA 마스크 설계 근거로 쓴다.
4. 마스크 ablation 설계: random / block / agent영역 / 도로영역 / horizon 단축을 `analysis/context_axes.csv`
   로 분해한다. 이 축은 metric cache 기반이라 NAVSIM 계열이면 그대로 재사용된다.

## 5. 미결 질문 (사용자 결정 필요)

- JEPA 베이스라인을 Drive-JEPA(코드 공개, 1월) / DA-WAM(SOTA, 경쟁자) 중 무엇으로 할지.
- SafeDrive 큐의 O0·O1×4 를 취소할지(약 65시간 절약) 아니면 분석 완결성을 위해 끝낼지.
- 벤치마크를 navtest 유지 / navhard(NAVSIM v2, SOTA 55.5~56.6 EPDMS, devkit 이전 필요) 중 어디로 둘지.

## 6. 다른 서버에서 재구성 (git 으로 오지 않는 것)

| 항목 | 이 서버 위치 | 옮기는 방법 |
|---|---|---|
| conda env (6.7 GB) | /home/kaist5/miniconda3/envs/safedrive (링크 `kjs-SafeDrive-exp2`) | `scripts/run/build_env.sh` + `build_mmcv.sh`. **mmcv 2.1.0 은 대상 GPU arch 로 소스빌드**(sm_90=H100). 사전빌드 wheel 은 sm_90 커널이 없어 deformable attention 이 조용히 전부 0 이 된다 |
| 체크포인트 3.0 GB | ckpts/safedrive_phase{1_90ep,2_5ep,3_10ep}.ckpt | rsync |
| 데이터셋 | dataset -> /home/kaist5/Dataset/navsim/dataset | 대상 서버 navsim 경로로 심볼릭 링크 |
| navtrain feature cache 444 GB / metric cache 22 GB | exp/safedrive_train_cache, exp/train_metric_cache_navtrain | rsync 또는 `scripts/run/cache_navtrain.sh` 재생성(수 시간). **일부만 없으면 `scripts/run/repair_cache.sh` 로 없는 것만 채운다** |
| navtest/navmini 캐시 5 GB | exp/metric_cache_navtest, exp/feat_cache_navmini, exp/metric_cache_navmini | rsync |
| 실험 산출물 | exp/safedrive/*, exp/training/* | 결과 CSV(`eval_*/traj_*.csv`)만 rsync 하면 충분 |
| 학습 체크포인트(진행 중) | exp/safedrive/{o0_nofuture,f3_nopairnc}/lightning_logs/checkpoints/ | 재개하려면 rsync. O0 는 epoch 1, F3 는 epoch 0 까지 |

**git 으로 오는 것**: 코드, 설정 yaml, `scripts/run`·`scripts/analysis`, `analysis/*.csv`(라벨·측정값),
RESUME_NOTES / EXPERIMENT_DESIGN / HANDOFF / AGENTS, trajectory_anchors.

### 서버 이전 실전 메모 (2026-09-30)

이전 서버 `cloud-orO3Hf` 는 사설 IP(192.168.0.2)만 가진 **NAT 뒤 클라우드 VM** 이라
outbound 가 막혀 있다(AXE-28·AXE-080 모두 ping·포트22 도달 불가). 그러나
**외부 IP `61.107.200.100` 으로 들어오는 것은 된다** — 이전에 새 서버에서 당겨오기(pull)로
성공한 이력이 있다.

→ **`tools/pull-from-cloud.sh` 를 새 서버에서 실행한다.** tier 로 나눠 두었다.

| tier | 내용 | 크기 |
|---|---|---|
| 1 | Claude 메모리·대화 + 평가 결과 CSV + 재개용 `last.ckpt` | **2.6 GB** |
| 2 | 사전학습 ckpt + navtest/navmini 캐시 | 8 GB |
| 3 | navtrain metric cache | 22 GB |
| 4 | navtrain feature cache (**권장하지 않음**) | 444 GB |

```bash
bash tools/pull-from-cloud.sh 1          # 포트가 22 가 아니면 SRC_PORT=xxxx 를 앞에 붙인다
```

tier 1 이 `exp/safedrive` 전체(52 GB)를 받지 않는 이유: 중간 epoch 체크포인트가 run 당
7.8 GB 씩 쌓여 있는데 재개에는 `last.ckpt` 하나만 필요하다.

그 외 원칙:

1. **코드·스크립트·분석 CSV 는 git 으로만 옮긴다** (이미 `mine` 원격에 push 됨, 16 MB).
2. 새 서버에서 **`bash tools/check-new-server.sh`** 를 먼저 돌려 GPU·디스크·데이터셋·conda·
   레포 상태를 확인하고, **없는 것만** 옮긴다.
3. **`exp/safedrive_train_cache`(444 GB) 는 옮기지 않는다** — SafeDrive 전용 feature builder
   산출물이라 Drive-JEPA 등 다른 모델이 재사용할 수 없다. JEPA 전환에는 원본 데이터셋만 필요하다.
4. Claude 메모리·대화는 git 으로 오지 않는다. `/home/kaist5/data/junseong/claude-context-0930.tar.gz`
   (2.5 MB, `memory/` 14개 + 이 세션 transcript)를 별도 경로로 옮긴다. **메모리에 pin 된 연구
   명제가 들어 있으므로 이것만은 꼭 옮긴다.**

원본 데이터셋 실제 크기(`/home/kaist5/Dataset/navsim/dataset`, 심볼릭 링크 모음):

| 항목 | 크기 |
|---|---|
| sensor_blobs | 2.5 TB |
| **navhard_two_stage** | **31 GB** ← NAVSIM v2 navhard 데이터가 이미 내려와 있다 |
| navsim_logs | 16 GB |
| private_test_hard_two_stage | 10 GB |
| maps | 1.4 GB |
| warmup_two_stage | 1.2 GB |

### 환경변수

새 서버에서 경로가 다르면 다음만 맞춘다(기본값은 이 서버 경로).

```bash
export SD_SCRIPTS=<레포>/scripts/run      # 스크립트끼리 서로를 찾는 경로
export SD_LOGS=<레포>/exp/logs            # 학습·평가 로그 출력
export SD_ANALYSIS=<레포>/analysis        # 라벨·측정 CSV
```
