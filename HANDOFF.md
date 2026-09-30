# HANDOFF — 이 파일 하나로 다음 에이전트가 이어받는다

마지막 갱신: 2026-09-30 16:19 KST (Codex)

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

- **AXE-080 Tier 1 이전 완료.** Claude 메모리 14개와 세션 JSONL 1개, 평가 CSV 22개,
  `o0_nofuture`·`f3_nopairnc` 재개용 `last.ckpt` 2개를 받았다. 체크포인트 크기는 AICA 원본과
  일치하고 받다 만 임시 파일은 없다.
- **새 서버 데이터 경로 규칙을 확정했다.** `/rhome/junseong/`은 코드·변환 결과·metric/feature
  cache를 포함한 작업공간이다. `/home/user/data/Dataset/`의 연구실 공용 원본은 절대 직접 수정하지
  않고 `/rhome/junseong/` 아래 프로젝트에 심볼릭 링크로만 연결한다. 새로 필요한 데이터셋 원본은
  `/home/user/data/processed_dataset/`에 총 1 TB 한도 안에서 다운로드한다. 이 규칙을 `AGENTS.md`와
  pinned Claude 메모리에 기록했다.
- SafeDrive 에 `dataset -> /home/user/data/Dataset/navsim` 심볼릭 링크를 만들고 maps·navsim_logs·
  sensor_blobs 를 읽을 수 있음을 확인했다. 링크는 `.gitignore` 대상이고 공용 원본에는 쓰지 않았다.

## 4. 다음 단계 — 새 서버에서의 순서

**원칙: SafeDrive 는 재학습 없이 뽑을 수 있는 것만 마무리하고, GPU 는 JEPA 에 쓴다.**
SafeDrive 에 남은 가치는 대부분 이미 있는 평가 CSV 안에 있다(22개 run, `exp/safedrive/eval_*/traj_*.csv`).
반면 JEPA 는 아직 코드 한 줄도 없다. 따라서 GPU 를 SafeDrive 재학습에 26~65시간 더
쓰는 것은 우선순위가 아니다.

### 4-0. 세팅 (반나절, GPU 0장)

```bash
git clone -b junseong/main git@github.com:JunSeongKW/SafeDrive.git && cd SafeDrive
bash tools/check-new-server.sh          # GPU·디스크·데이터셋·conda 확인
bash tools/pull-from-cloud.sh 1         # 2.6 GB (메모리/대화 + 평가 CSV + last.ckpt)
```

`check-new-server.sh` 의 답이 이후 분기를 결정한다 — **sensor_blobs(2.5 TB) 와 디스크
500 GB 여유가 있는가.** 없으면 SafeDrive 재학습(4-3)은 아예 선택지가 아니다.

### 4-1. ★ 최우선: 선행연구 novelty 생사 확인 (1~2일, GPU 0장)

논문의 존재 여부가 여기서 갈린다. **GPU 작업보다 먼저 한다.**

| 논문 | 확인할 것 |
|---|---|
| `Causal-JEPA` 2602.11389 | Object-Level Latent Masking 이 **맥락 조건부**인가, 고정 규칙인가 |
| `AD-E2E-JEPA` 2609.34085 | 3주 전 논문. 맥락별 예측 대상 선택을 이미 했다면 명제 재설계 필요 |
| `Drive-JEPA` 2601.22032 | 코드 공개. 마스크 생성 코드 경로를 직접 읽는다 |

산출물은 **"마스크를 무엇이 정하는가" 표** 한 장이다. 열은
(무엇을 마스킹 / 누가 정함: 사람·랜덤·스케줄·입력조건부 / planning loss 가 마스크에
gradient 를 주는가 / 맥락별 분해 평가가 있는가). **마지막 두 열이 전부 "아니오" 면
이 연구의 자리가 확보된다.** 하나라도 "예" 면 그 논문과의 차이를 먼저 정의해야 한다.

### 4-2. SafeDrive 마무리 — 재학습 없는 것만 (2~3일, GPU 0장)

1. **★ 상황별 oracle 조합 (split-half).** 이 연구 명제의 가장 강한 단일 숫자이고 비용이 0 이다.
   맥락 버킷마다 6개 조건(baseline · F1 · F2 · E3 · E2 · α) 중 최선을 고른 조합 점수가
   **어떤 단일 고정 구성보다 높은가**를 본다. 버킷별 승자는 navtest 의 **절반에서 고르고
   나머지 절반에서 측정**한다(같은 데이터로 고르고 재면 허상이다 — 3차의 평균 회귀 사고와
   같은 종류의 함정). 이 숫자 하나가 논문 Figure 1 이 된다.
2. **다중비교 보정.** 맥락 분해에서 약 76개 검정을 했으므로 BH-FDR 를 적용하고, 부호 역전
   3건(특히 직진 −0.67 / 좌회전 +0.20)이 보정 후에도 남는지 확인한다. 남지 않으면
   motivation 을 "예비 관찰" 로 낮춰 쓴다.
3. 위 두 결과로 **motivation 절(논문 Sec.1/3) 초안**을 쓴다. SafeDrive 는 여기서 끝낸다.

### 4-3. SafeDrive 재학습 — **조건부. 조건이 안 맞으면 버린다**

- **O0 · O1×4 는 취소한다**(약 65시간 절약). 미래 표현을 4종으로 나눈 틀 자체가 JEPA 에서
  성립하지 않으므로, 그 조합표는 쓸 곳이 없다.
- **F3·F4 만** 값이 있다(각 약 13시간, 2 GPU). 얻는 것은 *동적 상호작용(F3) vs 정적 도로(F4)*
  대비 한 줄이고, 이것이 JEPA 마스크를 "agent 영역 / 도로 영역" 으로 나누는 설계 근거가 된다.
- 실행 조건: sensor_blobs 존재 + 디스크 500 GB + GPU 2장 유휴. **feature cache 444 GB 는
  전송하지 말고 `scripts/run/cache_navtrain.sh` 로 새로 만든다**(수 시간, NAT 뒤 VM 에서
  444 GB 를 받는 것보다 빠르다). 그 전에 tier 3(metric cache 22 GB)이 필요하다.
  ```bash
  bash tools/pull-from-cloud.sh 2   # ckpt + navtest 캐시 (평가용, 8 GB)
  bash tools/pull-from-cloud.sh 3   # navtrain metric cache (22 GB)
  bash scripts/run/cache_navtrain.sh
  bash scripts/run/queue.sh <gpuA> <gpuB> \
       f3_nopairnc:SafeDrive_Phase2_F3_NoPairNC f4_notwdac:SafeDrive_Phase2_F4_NoTwDAC
  ```
  체크포인트가 있으면 자동 재개하고 `TRAIN_DONE` 이 있으면 건너뛴다.
- **조건이 안 맞으면 미련 없이 버린다.** F3·F4 없이도 4-2 로 motivation 은 성립한다.

### 4-4. JEPA 착수 — 이것이 본편 (4-2 와 병행)

1. **Drive-JEPA 클론 → 재현.** 데이터 파이프라인이 NAVSIM 과 호환되는지, navtest 평가
   인프라(`exp/metric_cache_navtest`)를 그대로 붙일 수 있는지 먼저 본다. 새 conda env 로
   만든다(`python-env-isolation` 원칙).
2. **1차 실험 = SafeDrive 결과의 JEPA 재현.** 고정 마스크 4~5종(random / multi-block /
   agent 영역 / 도로 영역 / horizon 단축)을 각각 학습하고 `analysis/context_axes.csv` 로
   분해한다. 이 축은 metric cache 기반이라 NAVSIM 계열이면 그대로 재사용된다.
   **노리는 결과: 마스크 종류에 따라 상황별 승자가 바뀐다.** 이것이 나오면 method 로 간다.
   안 나오면 명제가 SafeDrive 특수성이었다는 뜻이므로 그 사실을 보고한다.
3. **2차 = method.** 맥락 조건부 마스크 생성기를 planning loss 로만 학습하고, 예산 K 를
   제약해 정보 요구 곡선을 그린다. 1차의 고정 마스크 성적이 gate 가 도달해야 할 상한이 된다.
   기술 리스크는 gate collapse 이며 load-balancing 또는 명시적 sparsity budget 이 필요하다.
4. **navhard 이전은 GPU 가 바쁜 동안의 CPU 작업으로 끼워 넣는다.** 데이터(31 GB)는 이미
   내려와 있고 이 fork 에 v2 코드가 없을 뿐이다.

## 5. 결정된 것 / 남은 미결

**결정했다** (근거는 4절):

- **베이스라인은 Drive-JEPA.** 코드가 공개돼 있어 재현 가능하고, DA-WAM 은 SOTA 지만
  질문 1 의 최대 경쟁자라 그 위에 쌓으면 우리 delta 가 그 논문 기여와 섞인다.
  DA-WAM 은 related work 비교 대상으로만 쓴다.
- **O0 · O1×4 취소, F3 · F4 는 조건부.** SafeDrive 재학습에 GPU 를 쓰지 않는 쪽이 기본값.
- **벤치마크는 navtest 로 반복하고 navhard 로 최종 주장.** navtest 집계는 포화지만
  맥락 분해용 인프라가 이미 있고, navhard 는 55.5~56.6 EPDMS 로 포화되지 않았다.

**남은 미결** — 4-1 의 결과를 봐야 답할 수 있다:

- `AD-E2E-JEPA`(3주 전)가 맥락 조건부 마스킹을 이미 했다면 명제의 어느 부분으로 좁힐지.
  후보: 예산 제약(K 스윕)으로 좁히기 / 마스크가 아니라 *예측 horizon* 의 맥락 적응으로 옮기기.

## 6. 다른 서버에서 재구성 (git 으로 오지 않는 것)

**AXE-080 데이터 규칙(2026-09-30 확인):** `/rhome/junseong/`이 코드·변환·cache를 포함한
작업공간이다. 연구실 공용 원본 `/home/user/data/Dataset/`은 절대 직접 수정하지 않고 작업공간의
프로젝트에 심볼릭 링크로만 연결한다. 새로 필요한 데이터셋은 `/home/user/data/processed_dataset/`에
총 1 TB 한도 안에서 내려받는다. 현재 SafeDrive 링크는 `dataset -> /home/user/data/Dataset/navsim`이다.

| 항목 | 이 서버 위치 | 옮기는 방법 |
|---|---|---|
| conda env (6.7 GB) | /home/kaist5/miniconda3/envs/safedrive (링크 `kjs-SafeDrive-exp2`) | `scripts/run/build_env.sh` + `build_mmcv.sh`. **mmcv 2.1.0 은 대상 GPU arch 로 소스빌드**(sm_90=H100). 사전빌드 wheel 은 sm_90 커널이 없어 deformable attention 이 조용히 전부 0 이 된다 |
| 체크포인트 3.0 GB | ckpts/safedrive_phase{1_90ep,2_5ep,3_10ep}.ckpt | rsync |
| 데이터셋 | `dataset -> /home/user/data/Dataset/navsim` | 공용 원본은 읽기 전용으로 취급하고 심볼릭 링크로만 사용 |
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
