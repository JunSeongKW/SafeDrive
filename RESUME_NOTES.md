# SafeDrive 실험 재개 노트 (2026-09-17 중단 시점)

## 확보된 자산 (전부 재사용 가능, 재생성 불필요)

| 항목 | 경로 | 크기 |
|---|---|---|
| 실행 환경 | conda env `safedrive` (py3.10 / torch2.1.0+cu121 / mmcv 2.1.0 소스빌드 sm_90 / spconv-cu120 / mmdet 3.2.0) | 6.7 GB |
| 체크포인트 | `ckpts/safedrive_phase{1_90ep,2_5ep,3_10ep}.ckpt` | 3.0 GB |
| 데이터 링크 | `dataset -> /home/kaist5/Dataset/navsim/dataset` | (심볼릭) |
| navtrain feature cache | `exp/safedrive_train_cache` (103,288 토큰) | 444 GB |
| navtrain metric cache | `exp/train_metric_cache_navtrain` (103,288) | 22 GB |
| navtest metric cache | `exp/metric_cache_navtest` (12,146) | 3.1 GB |
| navmini feature cache | `exp/feat_cache_navmini` (396) | 1.8 GB |
| navmini metric cache | `exp/metric_cache_navmini` (396) | 149 MB |

## 재현 완료된 baseline

navtest 12,147 시나리오, phase3 체크포인트, test.sh 가중치 그대로:

| 지표 | 논문 | 재현 |
|---|---|---|
| NC | 99.5 | 99.39 |
| DAC | 99.0 | 98.90 |
| TTC | 97.2 | 97.00 |
| EP | 84.3 | 83.15 |
| Comfort | 100 | 99.87 |
| **PDMS** | **91.6** | **90.96** |

결과 CSV: `exp/safedrive/eval_safedrive_navtest/traj_*.csv`
navmini(397) 결과: `exp/safedrive/eval_safedrive_navmini/traj_*.csv` → PDMS 93.59

## α 실험 (보행자를 sparse world 에 포함) — 코드 수정 완료

sparse world 구성원을 사람이 "차량만"으로 못박은 것을 풀고, 고정된 query 예산 50개
안에서 Hungarian matching 이 차량/보행자 배분을 스스로 결정하게 하는 실험.

수정 3곳 (모두 `include_pedestrian` 플래그 분기, 기본값 False 라 baseline 동작 불변):

- `navsim/agents/safedrive/safedrive_config.py` — `include_pedestrian: bool = False` 추가
- `navsim/agents/safedrive/safedrive_loss.py` — `gt_labels == 0` → 플래그 시 `>= 0`
- `navsim/agents/safedrive/safedrive_model.py` — `_extract_gt_for_dn_2d` 동일 분기

새 config: `navsim/planning/script/config/common/agent/SafeDrive_Phase2_Alpha_Pedestrian.yaml`
(Phase2 baseline 과 `include_pedestrian: True` 한 줄만 차이)

검증: 200 scene 기준 차량 2,091 + 보행자 1,679 가 새로 감독 대상이 됨.
navmini 스모크 테스트에서 에러 0 으로 정상 동작 확인 (검출·motion·pair-NC·PDM safety GT).

## 병목 진단 결과

step 시간의 지배 요인은 GPU 가 아니라 **PDM 시뮬레이터 롤아웃**이다.

- 샘플 1개(후보 궤적 128개) 롤아웃 = **1.33 초** (직접 벤치마크)
- 유휴 GPU(util 0%) 가 경합 GPU 보다 오히려 느렸음 → GPU 경합은 병목 아님
- `safety_ray_threads` 를 batch 크기 이상으로 올려도 무의미 (worker 가 놀게 됨)
- batch 8 × 2 GPU 기준 5,320 step/epoch × 약 12 초 = **1 epoch 17.7 시간, 5 epoch 약 3.7 일/실행**

### 단축 옵션
- **A. `num_proposal_2stage: 128 → 32`** → PDM 4배 단축, 실행당 약 22시간.
  baseline·α 양쪽에 동일 적용하면 비교는 유효(절대 성능은 논문보다 낮아짐).
- B. 128 유지 → 3.7일/실행
- C. navtrain subset(20k) → 약 17시간

## 재개 절차 (2026-09-17 14:5x 중단, epoch 0 의 22% 지점 — 체크포인트 없음, 처음부터)

스크립트: `train_phase2.sh <CONFIG> <EXP_TAG> <GPUS>` (BATCH / RAY 환경변수)

```bash
S=<scratchpad>
cd /home/kaist5/data/junseong/SafeDrive
BATCH=24 RAY=24 nohup bash $S/train_phase2.sh \
    SafeDrive_Phase2_Planner_FreezePerception phase2_baseline  4,5 > $S/train_baseline.log 2>&1 &
BATCH=24 RAY=24 nohup bash $S/train_phase2.sh \
    SafeDrive_Phase2_Alpha_Pedestrian        phase2_alpha_ped 6,7 > $S/train_alpha.log 2>&1 &
```

예상: 1 epoch 약 8~9 시간, 5 epoch 약 1.8 일 (두 실험 병렬)

### 배치 크기 실측 (H100 80GB 기준)

| batch | GPU 메모리 | 결과 |
|---|---|---|
| 8 | ~30 GB | OK |
| 16 | ~36 GB | OK (여유 35 GB 였을 때는 OOM) |
| **24** | **~75 GB** | **OK — 채택** |
| 32 | > 79 GB | OOM |

메모리가 비선형 증가한다(pair-wise 텐서 `pwnc_pred (L,B,50,128,8)` 때문으로 추정).
`ray_threads` 는 batch 와 같게 맞춘다(batch 보다 크면 worker 가 논다).

### 주의사항

- GPU 는 **4~7번만** 사용 (0~3 은 다른 연구원 몫). util 0% 여도 메모리 점유 중이면 건드리지 않는다
- `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` 유지
- 학습 후 평가: navmini 약 3분, navtest 약 1시간 5분 (metric cache 재사용)

### nvidia-smi 프로세스 이름 표시

다른 연구원이 구분할 수 있도록 `alpasim-sd` 로 보이게 해두었다. conda env 를 옮기면
pip entry point/shebang 의 절대경로가 깨지므로 **심볼릭 링크만** 걸었다.

```bash
ln -sfn ~/miniconda3/envs/safedrive ~/miniconda3/envs/alpasim-sd
```

`train_phase2.sh` 는 이 경로로 activate 하고, python 도 절대경로로 호출한다
(rank 0 의 argv[0] 까지 바꾸려면 절대경로 호출이 필요하다).

## 코드 리뷰에서 발견한 것 (논문과 코드의 간극)

1. Phase 2 의 `freeze_perception_prefixes` 10개 중 실제 모듈명과 맞는 것은 2개뿐
   (`_backbone` vs 실제 `ProposalNet_BEV` 등). 실행 로그상 114.33M/116.66M 이 학습됨
   → "perception 동결"은 사실상 동작하지 않고 end-to-end 미세조정이 됨
2. `fut_bev_map`(미래 BEV seg)은 planner 가 입력으로 받지 않음. loss 로만 존재
3. agent future GT 가 128개 후보 궤적 전부에 동일하게 적용됨
   (`all_motion_predidction_loss: true`) → reactive prediction 이 감독되지 않음
4. `twdac_bev_seg_out`(4채널 BEV seg)에 학습 loss 가 없음. 추론 시에만 사용됨
5. 논문의 "instance queries **and their trajectories** are passed to SWNet" 과 달리,
   코드에서는 궤적이 전달되지 않고 SWNet 의 `swnet_ins_reg_branch[0]` 가 자체 생성
6. safety loss 는 6개 layer 중 마지막(SWNet layer 3) 하나만 감독
7. `pwnc_pred (L,B,50,128,8)` vs `twdac_pred (L,B,128,8)` — 동적은 대상별, 정적은
   시점별만. fine-grained reasoning 이 정적 요소에는 절반만 적용됨

---

## α 실험 최종 결과 (2026-09-18)

Phase 2 를 baseline / α 두 갈래로 5 epoch 씩 학습(09-17 17:08 → 09-18 08:35, batch 24,
GPU 2장씩, 에러 0)한 뒤 navtest 12,147 시나리오로 평가했다.

### 통합 비교표 (navtest, n=12,147, valid 100%)

| 지표 | 논문 phase3 | 재현 phase3 | P2 baseline | **P2 α** | Δ(α−base) |
|---|---:|---:|---:|---:|---:|
| NC | 99.5 | 99.39 | 99.51 | 99.45 | −0.07 |
| DAC | 99.0 | 98.90 | 98.45 | 98.65 | +0.20 |
| TTC | 97.2 | 97.00 | 97.51 | 97.46 | −0.06 |
| EP | 84.3 | 83.15 | 78.74 | 78.75 | +0.01 |
| Comfort | 100 | 99.87 | 99.43 | 99.44 | +0.01 |
| DDC | — | — | 98.07 | 98.00 | −0.07 |
| **PDMS** | **91.6** | **90.96** | **89.00** | **89.09** | **+0.09** |

P2 가 phase3 보다 낮은 것은 정상이다(phase3 는 10 epoch 을 더 돈 결과).

### 결론: α 는 효과가 없다 — 가설 기각

paired bootstrap(5,000 회, 같은 시나리오 짝지음)으로 ΔPDMS 의 95% 신뢰구간을 구했다.
**모든 구간이 0 을 포함한다.**

| 구간 | n | ΔPDMS | 95% CI | 동일 점수 |
|---|---:|---:|---|---:|
| 전체 | 12,147 | +0.09 | [−0.17, +0.36] | 24.6% |
| 보행자 ≤20 m | 6,444 | **−0.02** | [−0.35, +0.32] | 29.1% |
| 보행자 없음(>20 m) | 5,702 | +0.22 | [−0.22, +0.63] | 19.6% |
| 보행자 ≤30 m | 8,327 | **−0.10** | [−0.39, +0.20] | 27.9% |
| 보행자 없음(>30 m) | 3,819 | +0.51 | [−0.05, +1.04] | 17.6% |
| 보행자 4명 이상 | 2,328 | **−0.50** | [−1.02, +0.02] | 31.9% |

### navmini 결과는 노이즈였다

먼저 돌린 navmini(397 장면)에서는 전체 +0.59, 보행자 장면 +0.70 / 비보행자 +0.44 로
가설에 부합하는 것처럼 보였다. navtest 로 30배 키우자 **부호가 뒤집혔다**(보행자 장면
−0.02, 비보행자 +0.22). 397 장면은 ΔPDMS 0.5 수준을 판별할 검정력이 없다.
**앞으로 이 종류의 비교는 navmini 로 결론짓지 않는다.**

### 왜 실패했는가 — 용량이 병목이지 클래스 목록이 아니다

보행자 밀도별로 쪼개면 단조 감소 추세가 보인다.

| 보행자(≤20 m) | n | ΔPDMS |
|---|---:|---:|
| 1명 | 1,958 | +0.15 |
| 2–3명 | 2,158 | +0.35 |
| **4명 이상** | 2,328 | **−0.50** |

원인은 `select_topk`(safedrive_model.py:1266)에 있다. sparse world 구성원 25명을
**plan anchor 궤적까지의 최소 L2 거리만으로** 고른다(`l2.topk(k, largest=False)`).
보행자를 검출 대상에 넣으면 인도·횡단보도의 보행자가 **거리상 가깝다는 이유만으로**
계획에 더 중요한 원거리 차량의 슬롯을 밀어낸다. 보행자가 많을수록 이 잠식이 커지므로
4명 이상 구간에서 손해가 난다.

즉 "월드에 무엇을 넣을지"를 넓히는 것만으로는 안 되고, **고정된 25 슬롯을 거리순으로
채우는 규칙 자체**가 사람이 정한 진짜 제약이다. 이는 오히려
"context-adaptive world representation" 방향을 지지하는 근거다.

### 산출물

- 결과 CSV: `exp/safedrive/eval_test_{baseline,alpha}/traj_*.csv`
- 시나리오 라벨(보행자·차량 밀도, 20/30/50 m): scratchpad `navtest_labels.csv` (12,146행)
- 집계 스크립트: scratchpad `analyze.py`

---

## 2차 실험 (2026-09-18) — 프로세스명 `kjs-SafeDrive-exp2`

env 심볼릭 링크: `~/miniconda3/envs/kjs-SafeDrive-exp2 -> safedrive`

### 가설 3개가 연속으로 반증되고, 네 번째가 확정됐다

| # | 가설 | 방법 | 결과 |
|---|---|---|---|
| α | 월드 구성원이 "차량만"인 것이 제약 | Phase2 재학습 5ep | ❌ +0.09, CI [−0.17,+0.36] |
| E1 | 거리 top-25 규칙이 중요한 agent 를 놓친다 | CPU 분석 12,146 | ❌ 충돌 agent 의 **98.1% 가 이미 월드 안** |
| E4 | 월드 범위 64 m 가 부족하다 | CPU 분석 12,146 | ❌ 64 m 초과 장면 2.1% 뿐 |
| **E5** | **월드 정보를 결정으로 바꾸는 가중치가 병목** | 추론 10회 | ✅ **두 체크포인트 모두에서 재현** |

### E1 — 실패 귀인

`pdm_score_pairwise` 의 `collision_tokens` 로 at-fault 충돌 agent 를 특정하고,
`select_topk` 와 같은 거리 규칙으로 순위를 매겨 25 슬롯 안이었는지 판정했다.

```
NC 실패 장면        66 / 12,146 (0.54%)
충돌 agent 특정됨    52
25 슬롯 '밖'          1  (1.9%)
순위 중앙값 2등 | 상위5등 69.2% | 상위25등 98.1%
```

### 손실 분해 — 엉뚱한 곳을 파고 있었다

| 지표 | 평균 | 완점 비율 | 가중 손실 기여 |
|---|---:|---:|---:|
| NC | 99.51 | 99.5% | — |
| DAC | 98.45 | 98.5% | — |
| TTC | 97.51 | 97.5% | 10.4% |
| **EP** | **78.74** | **27.5%** | **88.7%** |

안전 지표는 포화라 개선 여지가 없다. EP 는 **복잡한 장면에서 오히려 높다**
(agent 0–25명 → 69.76 / 100명↑ → 84.98). E4 에서 원인이 드러났다:
**예측 궤적이 기준보다 중앙값 9.21 m 짧다. 89.7% 의 장면에서.**

### ★ E5 — EP 테스트 가중치 스윕 (재학습 없음, 추론만)

후보 선택 공식은 로그 공간이라 가중치가 **지수**로 작동한다
([safedrive_model.py:1198-1207](navsim/agents/safedrive/safedrive_model.py#L1198-L1207)):

```
score ∝ P(NC)^16 · P(DAC)^48 · P(EP)^0.75 · P(TTC)^15
```

DAC 확률이 1% 좋아지면 1.62배, EP 가 80% 좋아지면 1.51배 — EP 는 사실상 무시된다.

**Phase 2 (자체 학습 5ep, n=12,147)**

| EP 가중치 | NC | DAC | TTC | EP | Comf | DDC | PDMS | ΔPDMS | 95% CI |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 0.75 | 99.51 | 98.45 | 97.51 | 78.74 | 99.43 | 98.07 | 89.00 | — | 기준 |
| 3 | 99.49 | 98.43 | 97.48 | 79.04 | 99.51 | 98.04 | 89.10 | +0.11 | [+0.05,+0.16] |
| 8 | 99.47 | 98.43 | 97.36 | 79.61 | 99.60 | 97.97 | 89.30 | +0.30 | [+0.22,+0.38] |
| 20 | 99.46 | 98.40 | 97.35 | 80.43 | 99.67 | 97.95 | 89.64 | +0.64 | [+0.53,+0.75] |
| 50 | 99.46 | 98.38 | 97.25 | 81.23 | 99.72 | 97.85 | 89.93 | +0.93 | [+0.80,+1.07] |
| 100 | 99.43 | 98.34 | 97.09 | 81.62 | 99.75 | 97.76 | 90.01 | +1.01 | [+0.87,+1.17] |
| 120 | 99.42 | 98.32 | 97.09 | 81.71 | 99.74 | 97.76 | 90.03 | +1.03 | [+0.88,+1.18] |
| 300 | 99.40 | 98.32 | 97.08 | 82.10 | 99.61 | 97.72 | 90.17 | +1.17 | [+1.01,+1.33] |
| 1000 | 99.40 | 98.30 | 97.07 | 82.26 | 99.62 | 97.69 | 90.22 | +1.22 | — |

**Phase 3 (논문이 배포한 최종 체크포인트, n=12,147)**

| EP 가중치 | NC | DAC | TTC | EP | Comf | DDC | PDMS | ΔPDMS | 95% CI |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 0.75 (논문) | 99.39 | 98.90 | 97.00 | 83.15 | 99.87 | 98.05 | 90.96 | — | 기준 |
| 100 | 99.32 | 98.77 | 96.49 | 84.95 | 99.60 | 97.80 | 91.37 | +0.41 | [+0.29,+0.54] |
| 300 | 99.30 | 98.74 | 96.44 | 85.20 | 99.46 | 97.74 | 91.42 | +0.46 | [+0.32,+0.59] |

핵심:

- **300 부근에서 포화**한다 (300 → 1000 이 +0.05).
- 논문 체크포인트에서도 재현되므로 특정 학습 상태의 문제가 아니다.
- 재현치 90.96 과 논문 주장 91.6 의 격차 0.64 중 **0.46 을 이 한 줄이 메운다.**
- 덜 학습된 모델일수록 이득이 크다(Phase2 +1.22 vs Phase3 +0.46). 추가 학습이 하는 일의
  상당 부분이 진행량 회복인데 그걸 공짜로 당겨올 수 있다는 뜻. 다만 Phase2+EP1000(90.22)이
  Phase3 기본값(90.96)에 못 미치므로 추가 학습에는 대체 불가능한 가치도 있다.
- **논문화 주의**: navtest 위에서 가중치를 골랐으므로 엄밀히는 test-set tuning 이다.
  navmini 나 별도 val split 에서 최적값을 정한 뒤 navtest 로 한 번만 보고해야 한다.

### 부수 발견 — Phase 2 의 "freeze perception" 은 동작하지 않는다

yaml 의 prefix 10개 중 8개가 실제 모듈명과 다르다(`_backbone` → 실제 `ProposalNet_BEV`).
exact-module 매칭이라 **116.66M 중 2.30M(2.0%)만 동결**되고 나머지는 end-to-end 학습된다.

```
baseline : trainable 114.33M / 116.66M
E2 (수정): trainable  19.21M / 116.66M
체크포인트 크기도 1.3GB vs 593MB 로 차이난다(optimizer state).
```

### 진행 중인 학습

- **E2** `SafeDrive_Phase2_E2_RealFreeze` — 올바른 prefix 로 perception 실제 동결
- **E3** `SafeDrive_Phase2_E3_World5` — `num_filtering_instance: 25 → 5`

둘 다 batch 24, 5 epoch, GPU 2장씩. baseline(PDMS 89.00)과 직접 비교.

### 재사용 자산

- 시나리오 라벨(보행자·차량 밀도): scratchpad `navtest_labels.csv`
- 실패 귀인: scratchpad `e1_attribution.csv`
- 궤적 길이·속도: scratchpad `e4_range.csv`
- 스크립트: `eval_sweep.sh`(EPW 환경변수), `e1_failure_attribution.py`, `e4_range.py`, `analyze.py`
- **주의**: 체크포인트 파일명의 `=`(`epoch=0-step=1774.ckpt`)는 Hydra override 파싱을 깨뜨린다.
  `exp/safedrive/ckpt_links/` 에 `=` 없는 심볼릭 링크를 만들어 쓴다.

---

## 3차 (2026-09-20) — 미래 정보 필요성 실험

연구 명제 *"planning 에 필요한 미래 정보는 상황마다 다르고 사람이 미리 정하면 안 된다"*
의 질문 1(사람이 정한 표현이 실제로 필요한가)에 대한 재학습 기반 인과 ablation.

### E2 — perception 을 실제로 동결하면

yaml 의 prefix 8/10 이 실제 모듈명과 달라 baseline 은 116.66M 중 2.30M(2.0%)만 동결된다.
올바른 이름으로 고쳐 97.4M 을 동결하고 5 epoch 학습했다.

| 모델 | 학습 파라미터 | NC | DAC | TTC | EP | Comf | DDC | PDMS |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| baseline | 114.33M | 99.51 | 98.45 | 97.51 | 78.74 | 99.43 | 98.07 | 89.00 |
| E2 동결 | 19.21M | 99.25 | 99.08 | 97.29 | 76.00 | 98.90 | 98.31 | 87.98 |
| Δ | −95M | −0.27 | **+0.63** | −0.22 | **−2.73** | −0.53 | **+0.24** | **−1.02** |

ΔPDMS −1.02, 95% CI [−1.33, −0.73] (유의).

**지표마다 부호가 갈린다.** perception 미세조정은 진행량(EP)만 끌어올리고 정적 장면
준수(DAC·DDC)는 오히려 망친다. Phase 1 에서 학습된 표현은 DAC/DDC 에 이미 충분하다.

비용 대비로 보면 더 인상적이다.

| 방법 | 비용 | PDMS 이득 |
|---|---|---|
| perception 미세조정 (95M 추가 학습) | 13 h × 2 GPU | +1.02 |
| EP scoring 상수 하나 변경 | 0 | +1.22 |

### ★ 난이도 정의가 결론을 바꾼다 — 주의

E2 를 난이도 계층으로 쪼갰더니 정의에 따라 패턴이 달라졌다 (값 = ΔPDMS).

| 난이도 정의 | 쉬움 | 보통 | 어려움 | 패턴 |
|---|---:|---:|---:|---|
| A. agent수 + 요구변위 | −0.80 | −1.46 | −0.53 | 비단조 |
| B. agent 수만 (20 m) | −0.86 | −1.21 | −1.29 | **단조** |
| C. 요구 전방변위만 | −0.75 | −2.09 | −0.54 | 비단조 |
| D. baseline 점수 3분위 | −4.04 | −3.95 | +4.93 | **무효** |
| E. 전체 객체 수 | −0.62 | −1.42 | −1.35 | 거의 단조 |

- **비단조성은 요구 전방변위를 정의에 섞을 때만 나타난다.** agent 밀도로 정의하면 단조다.
  한때 "어려울수록 동결이 덜 손해"라고 보고했으나 정의의 artifact 였다.
- **D 는 방법론적으로 무효다.** baseline 점수로 계층을 나눈 뒤 두 모델을 비교하면
  평균 회귀가 생겨, baseline 이 우연히 낮았던 집단에서는 다른 모델이 무조건 좋아 보인다.
  +4.93 은 실체가 없다. **이 함정을 앞으로도 피한다.**
- 정의를 바꿔도 유지된 것: **ΔEP 가 전 구간 −2.5~−3.0 으로 균일**. EP 손실은 상황 의존적
  정보 부족이 아니라 전역적·체계적 편향이다(E4 의 "89.7% 장면에서 9.21 m 덜 감"과 일치).

**교훈**: 명제의 핵심이 "상황별로 다르다"인 만큼 *상황을 어떻게 정의하느냐*가 급소다.
사후에 고른 정의로 패턴을 보고하면 안 되고, 여러 정의를 전부 보고해 정의를 바꿔도
유지되는 것만 주장한다.

### 진행 중

| 실험 | 변경점 | GPU |
|---|---|---|
| **F1** `SafeDrive_Phase2_F1_NoFutureBEV` | `fut_bev_semantic_weight: 7.0 → 0.0` | 4,5 |
| **F2** `SafeDrive_Phase2_F2_NoMotionSup` | `prediction_loss_weight: 1.0 → 0.0` | 6,7 (E3 평가 후) |
| E3 평가 | 월드 25→5 체크포인트 navtest | 6 |

F1 과 F2 는 비대칭이다. `fut_bev_map` 은 planner 에 도달하지 않으므로(model:585-586 에서
output 에 넣고 끝) loss 를 끄면 영향이 완전히 사라지지만, motion 출력은 sparse world 로
흘러가므로 F2 는 "감독이 필요한가"만 묻는다.

### 운영 교훈

- **학습이 끝나면 다음 작업을 체이닝해 둔다.** E3 가 13:03 에 끝났는데 후속을 안 걸어
  GPU 6,7 을 6시간 놀렸다. `chain_f2.sh` 처럼 `until grep -q EVAL_DONE` 로 이어붙인다.
- **단일 GPU 로 학습을 돌리지 않는다.** baseline 이 2 GPU × batch 24(유효 48)이므로
  1 GPU 면 유효 배치가 절반이 되어 비교가 깨진다.
- `until` 루프의 grep 패턴이 **자기 명령줄과 일치**해 무한 대기하는 일이 있었다.
  패턴을 좁히거나 프로세스 이름 대신 로그 파일 내용으로 판정한다.

---

## 4차 (2026-09-23) — F1/F2 결과 파일 검증 및 남은 분석

연구 질문: 사람이 정한 미래 표현의 감독을 하나씩 제거했을 때, 어떤 상황에서 성능 변화가 생기는가? F1은 미래 BEV semantic 감독, F2는 agent motion 감독을 제거한다.

`eval_test_baseline`, `eval_e2_final`, `eval_f1_nofutbev_ev`, `eval_f2_nomotionsup_ev`의 `traj_*.csv`는 모두 같은 유효 토큰 12,147개를 담는다. 아래 값은 동일 토큰을 짝지어 계산한 **분해 전 진단값**이며, 난이도·시나리오별 평가가 끝나기 전 연구 결론으로 쓰지 않는다. 단위는 점수의 percentage point, 95% 구간은 토큰별 짝차의 정규 근사다.

| 모델 | ΔPDMS (95% 구간) | ΔNC | ΔDAC | ΔTTC | ΔEP | ΔComf | ΔDDC |
|---|---:|---:|---:|---:|---:|---:|---:|
| E2 실제 동결 | −1.023 (기존 분석 [−1.33, −0.73]) | −0.268 | +0.626 | −0.222 | −2.735 | −0.527 | +0.243 |
| F1 미래 BEV 감독 제거 | +0.249 [−0.011, +0.509] | −0.029 | +0.354 | +0.074 | +0.108 | −0.156 | −0.016 |
| F2 motion 감독 제거 | −0.327 [−0.570, −0.083] | −0.132 | +0.288 | −0.370 | −0.597 | −0.329 | +0.107 |

F1의 전체 점수 차는 0을 포함한다. F2는 전체적으로 EP 하락이 두드러지지만, 변화가 상황 의존적인지는 아직 모른다. 기존 E2 분석의 난이도 정의 B(20 m agent 수)·E(전체 객체 수)와 시나리오 유형 라벨로 세 실험을 한 표에 분해해야 한다. 앞서 사용한 `scratchpad/navtest_labels.csv`가 현재 저장소와 인근 작업 경로에서 확인되지 않았다. 라벨을 동일 정의로 복구하거나 재생성한 뒤 결론을 내린다. baseline 점수로 계층을 나누는 정의 D는 평균 회귀 때문에 사용하지 않는다.

F3/F4 학습은 현재 프로세스가 없다. F3에는 `exp/safedrive/f3_nopairnc/lightning_logs/checkpoints/epoch=0-step=1774.ckpt`와 `last.ckpt`만 있다. F4에는 체크포인트가 없다. 두 `run_training.log`는 Trainer 시작 이후 종료 이유를 담지 않는다. 완료·평가 결과로 취급하지 않으며, GPU 2장 이상이 비면 재개 상태와 종료 원인을 먼저 확인한다.

---

## 4차 (2026-09-28) — ★ "상황별로 필요한 정보가 다르다" 예비 확인

### 먼저 고친 것 두 가지 (둘 다 조용히 결과를 오염시키던 문제)

1. **feature cache 4,321 토큰 소실** (103,288 → 98,967, 444 → 426 GB). step/epoch 이
   1774 → 1695 로 바뀐 것이 유일한 단서였다. `force_cache_computation=False` 로
   없는 것만 재생성해 15분에 복구. **앞으로 학습 시작 시 step 수가 1774 인지 확인한다.**
2. **`pair_Disp` 누출.** `prediction_loss_weight: 0` 으로 motion 감독을 껐다고 믿었지만
   `pair_Disp` 가 별도 가중치로 `gt_motion_traj` 를 계속 감독했다
   (safedrive_loss.py:367-379, 로그에 `pair_Disp_loss_GT=66.70`). motion 을 끄는 모든
   config 에 `pair_Disp_loss_weight: 0.0` 을 추가했다. **이미 끝난 F2 는 따라서
   "agent 미래 궤적 제거" 가 아니라 "motion head 자체 loss 만 제거" 로 해석해야 한다.**

### 맥락 축을 사전 확정했다

`context_axes.py` 로 navtest 12,147 장면에서 **입력만 보고** 7개 축을 뽑았다
(`context_axes.csv`). 모델 성능으로 계층을 나누면 평균 회귀로 허상이 생긴다 — 3차에서
baseline 점수 3분위가 +4.93 이라는 허상을 만든 적이 있다. 그 함정을 구조적으로 배제했다.

축: 회전(dheading) · agent 밀도 · 보행자 유무 · ego 속도 · 요구 진행량 · 곡률(bow) · 정적물

### 결과 — 단조 추세 3건이 핵심 증거

개별 ★(95% CI 가 0 제외)은 7축 × 4모델 × 3버킷 ≈ 76 검정이라 우연히 4개쯤 나온다.
따라서 **순서 있는 버킷을 따라 단조로 움직이는 것**만 주장한다.

| 축 | ablation | 버킷별 ΔPDMS | 해석 |
|---|---|---|---|
| 요구 진행량 | E3 월드 25→5 | **+0.29 → −0.13 → −1.36★** | 멀리 갈 때만 큰 월드가 필요 |
| 요구 진행량 | F2 agent궤적 제거 | **+0.09 → −0.40★ → −0.68★** | 같은 방향 |
| ego 속도 | E2 perception 동결 | **−2.39★ → −0.97★ → −0.74★** | 정지 상태에서 인식이 가장 중요(직관과 반대) |

### 부호가 뒤집히는 사례

- **미래 BEV**: 한산 −0.31 → 보통 **+0.64★** → 밀집 **+0.41★**. 보행자 없음 −0.03 →
  있음 **+0.50★**. 붐빌수록 빼는 게 낫다.
- **회전 축에서 순위가 완전히 바뀐다.**

| | 직진(8,458) | 좌회전(2,258) | 우회전(1,430) |
|---|---:|---:|---:|
| F1 미래BEV 제거 | +0.23 | −0.17 | +0.99★ |
| F2 agent궤적 제거 | −0.30★ | −0.94★ | +0.50 |
| E3 월드 25→5 | **−0.67★** | **+0.20** | +0.42 |
| E2 perception 동결 | −0.93★ | −2.25★ | +0.36 |

**같은 정보(월드 25명)가 직진에서는 필요하고(−0.67★) 좌회전에서는 불필요하다(+0.20).**
우회전에서는 무엇을 빼도 아프지 않다. 이것이 명제의 핵심 주장에 대한 직접 증거다.

### 왜 3차에서는 안 보였는가

3차에서는 *난이도* 라는 한 축으로만 봤고, 정의를 바꾸면 패턴이 뒤집혀 "미확립" 으로
보고했다. **회전·요구진행량·속도·밀도로 나누면 신호가 분명하다.** 축을 잘못 골랐던 것이다.

### 남은 유보

- 미래 표현이 2종(F1·F2)뿐이다. **F3(동적 충돌)·F4(정적 도로)** 가 큐에 있고, 그 둘이
  붙으면 "동적 vs 정적" 대비가 생긴다.
- E2·E3 는 엄밀히 미래 표현이 아니라 **용량**(perception 파라미터 · 월드 구성원 수)이다.
- 축끼리 상관이 있다(회전↔곡률, 요구진행량↔속도). 논문에서는 상관을 명시하고 독립 축만 주장.

### 선행연구 정독 결과 (2026-09-28)

| 논문 | 상태 |
|---|---|
| **DA-WAM** (2608.19085) | 질문 1 거의 점유. 후보별 future latent 이지만 **종류는 하나**, 맥락 적응 없음, 난이도 분해 없음. **NAVSIM v1 93.7 PDMS** — SafeDrive(91.6)는 더 이상 SOTA 가 아니다 |
| **PerceptDrive** (2607.20175) | 가장 가까움. 다만 **현재 프레임** prior 를 라우팅하고 future 는 단일 latent. 예산 스윕 없음. 저자 자인: *"probes reliance rather than isolated causal effects"* |
| 위협 신호 추적 | "auxiliary task 효과가 시나리오 복잡도에 따라 다르다" 는 문장의 출처는 2021년 논문(2103.01039)으로, NAVSIM 도 미래 표현 선택 문제도 아니다. **경쟁 아님** |

→ 논문 무게중심을 **질문 2(상황 의존성)** 로 옮긴다. 질문 1 은 related work 에서
이미 알려진 사실로 인용하고 도입부 근거로만 쓴다.

### 진행 중 (GPU 0,1)

`queue.sh` 가 8개 항목을 순서대로 돌린다(로그: `chain_after_repair.log`).
O0 → F3 → F4 → F2b → O1×4. 약 100시간.

---

## 5차 (2026-09-29~30) — JEPA 전환 결정과 인수인계 정비

### 연구 플랫폼 전환

사용자가 **"SafeDrive 는 변경 가능 범위가 좁아서 새로운 novelty 를 넣기가 어렵다"** 고
판단해 JEPA 기반으로 전환하기로 했다. 근거가 분명하다 — SafeDrive 에서 돌린 실험 5건의
실제 변경 내용이 전부 스칼라·불리언 하나였다.

| 실험 | 실제 변경 |
|---|---|
| α | `include_pedestrian: True` (불리언) |
| F1~F4 | loss 가중치 하나를 0 으로 |
| E2 | `freeze_perception_prefixes` 오타 수정 |
| E3 | `num_filtering_instance: 25 → 5` (숫자) |

아키텍처를 건드린 실험이 하나도 없다.

### JEPA 가 이 명제에 맞는 이유

JEPA 에서는 **"무엇을 예측할지" 가 마스크로 표현된다.** 즉 연구 질문이 그대로
기술적 질문이 된다 — *"planning 에 필요한 미래 정보는 무엇인가"* = *"어떤 마스크를 쓸 것인가"*.

그리고 기존 JEPA 의 마스크는 전부 입력과 무관하게 사람이 고정한 것이다.

| 방식 | 정하는 주체 |
|---|---|
| random masking | 사람이 정한 무작위 규칙 |
| multi-block (I-JEPA) | 사람이 정한 블록 구조 |
| progressive schedule | 사람이 정한 학습 커리큘럼 |
| object-level (Causal-JEPA) | 사람이 정한 객체 단위 |

→ **마스크를 주행 맥락 조건부로 planning objective 가 학습하게 하는 것**이 명제의 두 질문을
동시에 겨냥한다. SafeDrive 와 달리 손댈 표면이 아키텍처 수준에 있다.

### 위험 — 이 분야는 매우 붐빈다

| 논문 | 시점 | 비고 |
|---|---|---|
| Drive-JEPA (2601.22032) | 2026-01 | **코드 공개**, V-JEPA + trajectory distillation |
| Auto-JEPA (2607.29031) | 2026-07 | 미래 ego 궤적 latent 를 planner retrieval key 로 |
| DA-WAM (2608.19085) | 2026-08 | dense JEPA supervision, **NAVSIM v1 93.7 PDMS**, 코드 공개 |
| AD-E2E-JEPA (2609.34085) | 2026-09 | 3주 전. NAVSIM v2 EPDMS 67.3/72.9 (zero-shot) |

**아키텍처로 경쟁하면 불리하다.** "마스크를 무엇이 정하는가" 축에 집중하고 JEPA 자체를
개선하려 하지 않는다.

### 인수인계 정비 — `/tmp` 의존을 제거했다

다른 서버로 넘어갈 때 작업이 끊기던 가장 큰 구멍을 막았다. 스크립트 31개와 분석 CSV 7개가
세션 임시 폴더(`/tmp/claude-1000/.../scratchpad`)에만 있어서 git 으로 오지 않았다.
이전 HANDOFF 4절의 "`navtest_labels.csv` 를 찾을 수 없다" 도 이 문제였다.

```
scripts/run/       14개  학습·평가·큐·캐시 실행 스크립트
scripts/analysis/  10개  분석 스크립트
analysis/           7개  라벨·측정 CSV (3.5 MB)
```

하드코딩된 `/tmp/claude-1000/...` 경로를 환경변수로 바꿨다
(`SD_SCRIPTS` / `SD_LOGS` / `SD_ANALYSIS`, 기본값은 이 서버 경로). 셸·파이썬 문법 검사 통과.

### queue.sh 버그 — 학습 실패를 감지하지 못했다

O0 가 외부 SIGTERM 으로 epoch 2 의 32% 에서 죽었는데, 큐가 그것을 모르고
**epoch=1 체크포인트(3 epoch 부족)로 평가를 시작했다.** "54분 만에 학습 종료" 라는 이상을
눈치채지 못했다면 잘못된 O0 결과를 보고했을 것이다.

- `queue.sh` 가 `TRAIN_DONE` 을 확인하고 없으면 **평가하지 않고 큐를 멈춘다**.
- 긴 작업은 `setsid` 로 띄워 셸 프로세스 그룹 정리에 휩쓸리지 않게 한다.
- 잘못된 평가 산출물(`eval_o0_nofuture_ev` 등)은 삭제했다.

### 중단 시점 상태

O0 는 `epoch=1-step=3548.ckpt` 까지, F3 는 `epoch=0-step=1774.ckpt` 까지 있다. F4 이후는 미시작.
사용자가 다른 작업을 위해 GPU 를 여러 번 회수했고, 매번 체크포인트가 보존돼 실제 손실은
총 2.5시간 정도다.

## 2026-10-01 — Codex/ChatGPT 공유용 중간 코드 감사 체크포인트

### 실제 수행

- 최신 사용자 연구 범위를 H1/H2, 고정 K·horizon, entity 미래 target 선택으로 정리했다.
- SafeDrive의 거리 top-K, query gather, world/planning 경로, target track 정렬 및 loss를 읽었다.
  top-K index를 사용하는 현재 선택에 학습된 선택 정책의 미분 경로가 없음을 확인했다.
- Drive-JEPA 공식 저장소를 `/rhome/junseong/research_sources/Drive-JEPA`에 clone했다.
  기준 commit: `548bb8215e3aae18e162a0f12f1ba83b4d3eb57e`.
  perception-free model/agent/features/encoder loader를 감사했다. 이 downstream 분기는
  pretrained encoder→trajectory decoder이며 별도 entity 미래 predictor를 호출하지 않는다.
  perception-based 내부 감사는 아직 미완료다.
- CSV 22개와 O0/F3 checkpoint 두 파일의 존재·바이트 크기를 재확인했다. checksum 미검증.
- 공용 NAVSIM 링크와 작업/다운로드 경로를 확인했다. 공용 원본에는 쓰지 않았다.
- 호스트 GPU 0·1은 RTX A6000 약 48 GB다. 기존 프로세스가 있는 상태였고 새 GPU 작업은 시작하지 않았다.
- `docs/RESEARCH_STATUS.md`, `docs/SELECTIVE_FUTURE_GRAPH.md`를 추가하고 AGENTS/HANDOFF를 정정했다.
  최신 사용자 승인 GPU(0·1)와 집계+상황별 평가를 반영했다.

### 이전 해석 정정과 한계

- 기존 context 축 중 fwd/lat/dheading/bow는 미래 PDM reference 기반이므로 online selector 입력이 아니다.
- cross-fitted 선택 규칙을 oracle upper bound로 부르지 않는다.
- AD-E2E-JEPA v1은 2026-09-28 공개다. 원문 §3.4 downstream IL의 predictor 제거와
  goal-conditioned zero-shot 경로를 구분한다. 모든 기존 마스크가 입력과 무관하다는 일반화는 철회한다.
- 과거 SafeDrive 점수·부호 역전·원인 해석은 이번에 재계산하거나 학습 provenance를 확인하지 않았다.

### 미수행 / 이어할 일

모델·loss 코드는 변경하지 않았다. selector 구현, synthetic gradient test, 실제 batch 재현,
환경 설치, 학습, cache 생성, baseline 최종 선정은 미수행이다. 먼저 그래프 초안을 함수 경계로
구체화하고 planning→selector 및 auxiliary 차단을 CPU autograd로 검증한다.
이 기록은 연구 성능 결과가 아니라 사용자 요청에 따른 **중간 인수인계 커밋**이다.

## 2026-10-01 — 계산 그래프 v1 CPU 계약 검사 및 합성 선택 학습

협업 출발 기준은 `95015df`다. 사용자/ChatGPT 피드백을 반영해 관측 마스킹이 아닌 미래 target
선택의 최소 synthetic fixture를 `scripts/research/`에 구현했다. 실제 NAVSIM agent 통합은 미수행이다.

### 구현과 계약 검사

순차 조건부 ST는 각 slot에서 이전 hard 선택을 제외하고 remaining valid 후보의 softmax로
backward한다. Forward는 중복 없는 hard 집합이며 tie는 stable ID 순서다. 이 exclusion은 미분하지
않으므로 편향된 추정이다. 모든 valid 후보가 K 안에 들어가면 부분집합 surrogate를 차단한다.
13개 unittest가 모두 통과했다. Detach는 forward 값을 유지하면서 backward만 차단한다.
미래 latent 제거/다른 샘플과 교환은 값 자체를 바꾸는 별도 개입이다.

실측 gradient norm (selector/predictor/planner):
planning = 0.009558/0.275470/1.282172;
auxiliary = 0/0.554858/0;
future detach = 0/0/1.282172.
Target encoder와 current encoder는 fixture에 없으므로 해당 encoder까지 검증한 결과가 아니다.

### 작은 합성 학습

N=6, K=2, 고정 analytic 미래값 x+2v, 고정 합산 planner. Intent가 원하는 semantic key 두 개를
명시하고 entity 순서는 무작위다. Selector만 planning MSE로 학습한다. 정답 selection loss는 없다.
3-seed, 각 1,000 step×batch 128, 각 별도 생성 holdout 4,096개. 사전 코드 기준 3/3 통과.
Exact-set 정확도는 100/99.9756/100%. 3-seed 평균 MSE는 learned 0.001660, random 6.734545,
command 교란 6.643663, no-intent entity-only learned 7.167637이다.

명령이 relevant key를 직접 지정하고 미래도 현재 state의 알려진 함수인 쉬운 과제다.
선택기의 기술적 학습 가능성만 점검했으며 미래 예측 필요성·NAVSIM 성능·H1/H2·novelty를
입증하지 않는다. Joint predictor/planner 학습과 같은 용량의 current-feature 대조는 남아 있다.

### 실행 환경·자산·다음 작업

CPU 1 thread, Python 3.12.13, torch 2.8.0+cu128, 최종 전체 실행 약 38.82초.
`exp/graph_cpu_env`는 기존 torch를 읽기 전용 참조하는 venv다. 기존 환경 설치/업그레이드는 없었다.
GPU·공용 원본·다운로드·cache는 사용/수정하지 않았다. Source hash/seed/config/raw 수치는
`analysis/research/graph_v1_validation.json`, 재현 명령과 해석은 `docs/GRAPH_V1_VALIDATION.md`.
다음은 실제 entity/target adapter와 Drive-JEPA perception-based 내부 감사 후 baseline 결정이다.

## 2026-10-01 — 현재 연구 작업공간·코드 명칭의 가독성 정비

사용자 요청: 다른 사람/에이전트가 이름만 보고 목적을 이해할 수 있도록 폴더뿐 아니라
함수·변수·설정·결과 key에도 읽기 쉬운 명명 규칙을 적용한다.

### 실제 변경

- Checkout 이름을 `/rhome/junseong/PlanningAwareFuturePrediction/`으로 바꿨다.
  현재 package `src/planning_aware_future_prediction/`, 검사 `tests/`, 실행 `scripts/`로 분리했다.
- Scorer / selection operator / future predictor / ego planner를 이름으로 구분했다.
  h/c/u와 batch/toy 같은 축약명은 entity_features/scene_context/ego_intent와 역할 있는 함수명으로 바꿨다.
- 현재 후보 / selected slot / 미래 target의 valid mask를 명칭으로 구분했다.
  합성 시간 범위와 future token 개수도 다른 명칭으로 표현한다.
- 공식 Drive-JEPA는 `reference_repositories/Drive-JEPA/`로 옮겼지만 코드를 수정하지 않았다.
- README는 현재 연구 진입점으로 바꾸고 이전 내용을 `README_SAFEDRIVE_ARCHIVE.md`에 보존했다.
  연구 문서·CLI·환경·result 경로를 갱신하고 naming_conventions/directory_migration 문서를 추가했다.
- 규칙은 project AGENTS와 docs에 commit 대상이며 home AGENTS/WORKSPACE_GUIDE에도 기록했다.
  원격 repo명/branch/Git 이력은 유지했다. SafeDrive baseline 연구는 계속 잠정 중단 상태다.

### 검사와 보존

재명명 중 테스트에서 표준 외부 API 참조 변경을 발견해 `torch.autograd.grad` 등 원래 API를
복구한 뒤 전체 검증을 재실행했다. 새 경로에서 13/13 테스트가 통과했다.
같은 설정의 3-seed 학습을 다시 실행했으며, 63개 policy metric·9개 초기 metric·15개 gradient
norm이 기존 실행과 정확히 같다. 새 코드와 environment/source hash는
`results/synthetic_diagnostics/readability_refactor_validation_20261001.json`에 기록했다.

옛 raw report는 수치·timestamp·source hash를 바꾸지 않고
`results/synthetic_diagnostics/future_prediction_graph_v1_before_readability_refactor_20261001.json`로 옮겼다.
기존 SafeDrive 모델·loss·CSV·checkpoint, 공식 Drive-JEPA 코드, 공용 원본, 다른 프로젝트는
변경하지 않았다. GPU는 사용하지 않았다. 절대경로가 박힌 옛 venv 대신 새 경로의
`runtime/environments/future_prediction_cpu`를 생성했다. 외부 dependency upgrade/install은 없다.
빈 이전 디렉토리만 정리하며 실제 파일/데이터는 삭제하지 않는다.

다음 연구 작업의 우선순위는 바뀌지 않았다: 실제 entity/target adapter 및 baseline 코드 감사.

## 2026-10-01 — ChatGPT fe8c930 검토 반영, baseline 감사와 실제 GT-state adapter 진단

실행 전 기준 commit은 `9b7d6e1`, 검토 대상은 `fe8c930`이었다. 이번 작업은 SafeDrive 재학습이 아니다.

### 확인·정정·설계

합성 relevance는 현재 key·intent 내적으로 계산 가능하므로 이전 hindsight/배포 불가 해석을 정정했다.
`input_exact_match`와 `relevance_oracle`은 이 과제에서 같은 선택이다. No-intent scorer는 entity input을
보므로 입력 독립 global mask가 아니다. 과거 raw 결과는 보존하고 코드/문서를 정정했다.
추가 synthetic selector 학습·accuracy tuning은 하지 않았다.

Drive-JEPA source548bb8215e3aae18e162a0f12f1ba83b4d3eb57e의 v1 perception-based feature/backbone/
refiner/scorer/target/agent loss를 감사했다. 기본 query는 object instance가 아니라 ego proposal/time
query다. Collision-object state head는 train-only auxiliary이며 그 출력을 planner/score가 입력받지 않는다.
권고는 official front-video encoder와 단순 trajectory decoder를 재사용한 새 scaffold + visual ROI
instance/target + 미래 memory adapter다. Official Drive-JEPA 전체 재현과는 구분한다.
Source/shape/gradient/privileged input/association 및 current-feature 대조 설계는
`docs/baseline_and_target_adapter_audit.md`에 있다. 본 학습 baseline은 official visual batch gate 후 결정한다.

### 실제 구현·실행

Raw NAVSIM GT-state adapter는 history/current와 future label builder를 분리한다. 현재 GT 차량·보행자
32개 cap, K4, future8/약4s, track-token 정렬 및 lidar/global/current-ego 변환을 구현했다.
GT 기하 state target은 시각 latent/JEPA가 아니다. 기존 graph에 8 waypoint 출력 옵션을 추가했다.
초기 실제 데이터 대조에서 yaw Euler convention 차이가 ego lateral label에 최대 약0.00021m 오차를
만들어 실패했다. 공식 pyquaternion 규약으로 수정하고 nonzero roll/pitch 회귀 검사를 추가했다.

28/28 unittest 통과: 기존 graph13 + state adapter13 + 합성 reference/gradient 회귀2.
이전 synthetic norm15개가 정확히 같았다. 실제 mini 한 log/scene의 window0/12에서 state forward/loss/
backward 및 Adam 한 번 update가 통과했다. 첫 구간 planning S/P/D norm =
0.00350536 / 0.15334630 / 0.94902224, state auxiliary = 0 / 0.32624449 / 0.
두 번째도 같은 zero/nonzero 계약 통과. 두 구간은 독립 성능 표본이 아니다.
학습 후 성능·수렴·H1/H2 검증으로 보고하지 않는다. 공식 visual encoder는 아직 실행하지 않았다.

실제 log와 두 front image의 SHA256이 전후 같았다. 공유 원본 쓰기·환경 설치/upgrade·GPU·cache
재생성·대규모 학습은 없었다. CPU venv(Python3.12.13/torch2.8.0+cu128)만 사용했다.
Ruff와 git diff --check 통과. Source hash/config/raw norm/shape는
`results/adapter_diagnostics/navsim_tracked_state_validation_20261001.json`, 명령/해석은
`docs/navsim_state_adapter_validation.md`에 남겼다.

다음: 독립 encoder 환경/선택 official weight 한 파일의 key 검증 → 실제 current/future visual ROI
batch → planner memory 연결 → 작은 공동 학습과 미래 경로 무시/선택 collapse 검증. GPU0·1만 사용하며
점유는 실행 직전 재확인한다. SafeDrive 주 baseline 연구는 계속 잠정 중단이다.

### 같은 세션의 최종 소스 대조 정정

주요 구현/결과 commit은 `3b44be1`. SafeDrive default query shape의 config30을 실제 constructor와
대조하니 `_query_splits=[1,num_bounding_boxes]`의 합31이었다. 감사 문서의 shape를31로 정정했다.
원본 SafeDrive 코드나 실제 GT-state 실행 결과는 변경하지 않았다.

## 2026-10-01 — 35fbdcf 검토 이후 실제 front-video / GT ROI pilot (Codex)

### 목적과 설계

실제 영상에서 selector→선택적 미래 예측→planner의 gradient 경계를 확인하는 하위 질문이다.
SafeDrive는 재개하지 않으며 합성 selector tuning이나 GT-state 테스트 추가 확대는 하지 않았다.
ROI appearance만으로 미래 위치/운동이 보존된다고 가정하지 않고 visual1024dim과 현재 ego 기준
future state6dim을 분리했다. 이것은 frozen visual+explicit spatial 혼합 감독 pilot이지 순수 JEPA,
GT 없는 deployment perception, 공식 Drive-JEPA 전체 모델/점수 재현이 아니다.

### 실제 자산·환경 준비

새 venv `runtime/environments/visual_future_prediction_pilot`은 기존 alpasim-cuda128을 읽기 전용
상속한다. timm1.0.30만 이 overlay에 추가하고 기존 환경은 upgrade하지 않았다.
공식 HF dataset LinhanWang/Drive-JEPA revision65e0d7284f69bf29d1a4864affcfd84ca4e97a2e의
vitl_merge_3dataset_e50.pt 한 개만 받았다. 5,127,748,765bytes, SHA256
4649182770ef68f84a001780c6579435345948fd80f186fa3616ab078ced668f 일치.
Source548bb8215e3aae18e162a0f12f1ba83b4d3eb57e clean 확인, target_encoder292tensor strict load.
weights_only/mmap 안전 로딩 성공; missing/mismatch 초기값 대체와 unsafe fallback 없음.
전체80GB weight bundle/cache를 받지 않았다. 새 dataset 원본 다운로드 없음.

### 실제 실행

승인 GPU0·1의 점유를 재확인한 뒤 GPU0만 사용했다. 기존 타인 프로세스/카드1을 건드리지 않았다.
Mini log2021.05.12.22.00.38_veh-35_01008_01518, window0, scene165060762e765a5a,
현재/미래2frame front clip9개를 FP32 frozen encoder로 처리했다. 기존 state smoke와 같은 scene다.
현재32radius/cap 후보 중 front projected13, 그중K4·8시점. Actual offset끝4.0013s.
Current H[1,32,1034], future visual target[1,32,8,1024], future state[1,32,8,6], ego[1,8,3].

Planning S/P/D norm0.02964600/0.46665784/25.91781228;
visual aux0/3.90319871/0; spatial aux0/6.38596213/0.
Future detach는 출력 동일이며 S/P gradient0. No future branch는 S/P forward 생략/gradient0.
Current target으로 label만 교체할 때 forward 동일. Frozen teacher/target gradient 없음.
공동 backward finite와 Adam 단일 selector update 통과; 수렴/중요 객체 학습/성능을 의미하지 않는다.

33/33 unittest(기존28+visual5)와 Ruff 통과. 최초 명령은 PYTHONPATH=src 누락으로 기존module2개
import 실패했으며 올바른 명령으로 수정해 통과했다. 공식 timm/CUDA attention의 deprecation
경고는 남았고 기능 실패가 아니므로 official source는 수정하지 않았다.
최초 영상검사23.67s, 최종 format/시각화 정리 후 재검사20.24s(공식 weight hash/load 포함).
CUDA peak allocated1,324,247,552bytes(~1.23GiB), whole process/reserved VRAM은 아님.
Frozen encoder303,885,312params와 새 scaffold2,487,988params를 구분한다. FLOPs/속도 절감 미검증.

ROI 그림을 직접 확인했다. 영역은 차량과 대체로 정합하나 일부 occluded/overlap이다.
미래 projected 유효수12,10,10,10,10,10,11,12; projection-valid는 실제 가시성 보장이 아니다.
Appearance cosine평균0.89360와 current-ego metric 이동평균0.91031m/최대13.81093m는 설명 통계다.
Pinhole distortion/rectification 규약과 상황별 multicamera coverage는 미확인이다.
공유 log1개와 사용 image10개의 SHA256은 전후 같았다. 공유 원본에는 어떠한 쓰기도 하지 않았다.

### 공유 기록·다음 단계

`results/visual_diagnostics/visual_future_pilot_verified_20261001.json`은 최종 실행의 provenance,
source/dataset SHA256, shape, budget, gradient, 단일 update, runtime을 담는다.
`docs/visual_future_prediction_pilot_validation.md`에 해석·한계·재현 명령·최소 대조 후보를 작성했다.
PNG는 로컬 outputs/visual_pilot에 두고 공용 영상/weight/env는 git에 넣지 않는다.

다음은 visual/spatial 감독 및 branch 대조 확정, novelty 원문/코드 표 완성, 여러 log의 coverage/
split 조사, 작은 공동 학습이다. 미래 감독 없음/current-target/no-future-branch를 구분하고
visual 감독 on/off에는 spatial supervision을 동일하게 유지하는 대조를 둔다.
All-entity 예측은 다른 예산 reference로, 선택 연구의 필수 진행 조건이 아니다.
본 학습·공식 score·미래 활용/collapse·추론 association·H1/H2·효율은 아직 확인하지 않았다.

## 2026-10-01 — 1231767 검토 이후 연구 결정·여러-log 유효율·최소 학습 계획 (Codex)

### 요청과 연구 결정

영상 연결 검사는 충분하다는 ChatGPT/사용자 검토를 받아 추가 smoke/합성 tuning 대신
target 효과를 분리하는 작은 학습의 준비로 전환했다. Fixed K는 개발 기반이고 최종 novelty가 아니다.
현재 pilot은 frozen visual latent+explicit 객체 spatial-state 감독, official encoder 재사용과 신규 작은
planner다. 객체 대상은 고르지만 정보 종류를 선택하는 모델은 아니다. SafeDrive 주 baseline/retraining은 중단 유지.

`docs/research_question_and_target_decision.md`에 첫 질문을 고정했다: 같은 현재 정보/거리 규칙/K4/branch에서
visual·spatial·mixed 미래 보조 감독 중 무엇이 log-held-out ego 경로 예측에 기여하는가?
이후 target을 좁힌 뒤 선택 비교와 고정 K별 맥락적 추가 예산 효과를 측정한다. 동적 K/horizon 구현 없음.

### 직접 선행연구 확인

EgoFSD arXiv2409.09777v6(2026-02-09)의 §3.4/3.5, Eq.(2)(3), Table3을 확인했다.
Intention/attention 및 geometry 기반 hierarchical selection과 joint motion/planning은 직접 중복한다.
Official shallow/no-checkout tree `23fec8aba3e828ef228939e30e3020240d8b0cae`는 README/assets only이며
모델 source/config/weight가 없어 discrete selection의 planning autograd는 독립 코드 감사 불가다.
GitHub network는 sandbox DNS 실패 후 승인된 escalation으로 작은 clone/README만 읽었다.

ForeDrive arXiv2609.26299v2(2026-09-23)의 Eq.(1)-(4)/(7), Appendix F/H를 확인했다.
Visual patch future+future ego status의 current-anchored fusion, fixed horizon/sample-shared gate다.
Eq.(7)은 planning→encoder/fusion/planner, planning↛predictor를 명시한다. 구현 링크는 제한된 확인에서 찾지 못했다.
우리의 주변 객체 state/P까지 planning gradient는 그 연구와 다르지만 engineering 차이만으로 기여를 확정하지 않는다.
Version/source/미확인 경계는 `docs/egofsd_foredrive_evidence_audit.md`에 기록했다. 전체 novelty 표는 여전히 미완료.

### 실제 CPU 데이터 조사와 split

`scripts/survey_navsim_visual_target_coverage.py`, `configs/exploration/data_survey.json`을 추가했다.
Encoder/학습/GPU 없이 기존 GT-state adapter와 cuboid pinhole projection을 이용했다.
Mini64segment를capture timestamp+vehicle 기준52recording group으로 묶고16group(1segment/group)을 hash sampling.
Train12/dev4, 과거 smoke recording은 development-only, 모든 segment alias도 같은 split으로 고정했다.
Coverage/outcome 이전 pre-survey manifest를 저장했다. Same native log/scene 및0.5±0.05s cadence,
비중첩 history4+future8 window를 metadata로 선정했다. Pilot train277/dev96window, survey128window.
Native log token/segment alias split 중복 없음. 미래 target validity로 현재 후보/window를 필터하지 않았다.

128window의현재GT 후보2,647개(40m 최근접cap32) 중front projection-valid621개=23.46%.
Front count>K4는58/128=45.3125%,0개는7/128. Side-only current projection은candidate의26.8984%.
현재front 객체를 분모로+4s visual406/621=65.3784%, spatial581/621=93.5588%.
고정nearest-front-K4는391active slot이며+4s visual238/391=60.8696%, spatial368/391=94.1176%.
Future-heading left proxy11window/8group은+4s visual12/33=36.3636%; right proxy0, merge label 미확인.
Future-heading proxy는사후분석값이며online입력으로사용하지않는다. Command 의미도추정하지않았다.

4contact sheet(train/dev×left/low-turn,각최초관측)를직접읽었다. ROI는대체로맞지만가려진GT box도
유효하여background/occluder feature를담을수있다. 공식load/pinhole code에undistort가없다는것은
stored JPEG의rectification여부를확정하지못한다. 이문제는미해결이며공유원본/calibration변경없음.
최종run CPU약8.08s,실패0. 공용log16file/reviewimage16file의SHA256은전후동일했다.
모든미래이미지의byte hash나실제가시성을검증했다는뜻은아니다.

Raw JSON/PNG는 `outputs/data_surveys/navsim_visual_target_coverage_v1_20261001/`.
기존조사output을덮어쓰지않았다. 공유summary/manifest는 `results/data_surveys/`, 조사해석은
`docs/navsim_visual_target_coverage.md`. Shared JSON은실행본과byte단위동일,source hash도최종script와일치한다.

### 최소 학습 계획 — 미실행

`docs/minimal_target_ablation_plan.md`와 `configs/exploration/target_ablation_plan.json`을작성했다.
Fixed-current-distance K4/horizon8/현재입력을고정하고 selector는학습하지않는다.
A branch없음/B planning만/C visualaux0.1/D spatialaux0.1/E mixed각0.05.
B–E 두head/branch폭동일. Visual-only/spatial-only는감독의종류이며planner 정보채널제거가아니다.
C/D/E공통 visual∩spatial mask,train-only normalization,같은초기weights/mini-batch sequence,
seed29/200update/조건/batch8계획/마지막fixed checkpoint로고정했다. E는총계수균형의trade-off이며
혼합기여주장에는matched-weight후속확인이필요하다.
주지표scene-macro ADE와FDE/heading/recording별결과,사전정의context및native coverage,
persistence와futurezero/swap의존도,시간/메모리/activeparameter를보고한다. 공식planning지표는아직없다.
373window작은featurecache만계획(2GiB cap),train8windowprofile후30분/조건·2시간총학습cap.
이한도는예상GPU-hour가아니다. GPU0·1만허용,실행전점유확인. Novelty/효과확정없이큰학습으로확대하지않는다.

### 실제 검사와 미수행

Ruff check/format check 통과. JSON/config/source hash,128window수치,exact 공유파일,
train/devnative-log/alias분리,373pilot 비중첩window,5고유조건/총aux계수일치 검사를통과했다.
기존33modeltests는이번에재실행/확대하지않았고modelsource도변경하지않았다.
새training CLI/fixed-rule forward/cache/normalization/evaluator는아직미구현이다.
학습·GPU실행·공식baseline재현·새원본download·공유dataset쓰기·기존환경upgrade없음.
다음은이번계획공유/검토후고정규칙runner와제한cache/metrics구현,이후작은A–E비교학습이다.

## 2026-10-01 15:28 KST — Codex: 실제 미래 감독 A–E 200-update 비교 완료

출발 `09913b5`. 최신 사용자 승인 범위는 영상 rectification/투영 정리, 고정 작은 split cache,
현재 거리 K4/A–E 각200update다. 이번 질문은 **선택된 객체에 어떤 미래 감독을 주면 planning
학습에 도움이 되는가**이며 selector 선택/최종 target/novelty가 아니다. GPU0만 사용하고 종료했다.

### 투영과 원본 보호

`audit_front_camera_projection.py`로 train/dev×직진/회전4표본을 조사했다. 원본 nuPlan DB의
K/D/1920×1080가 NAVSIM pickle과 일치한다. DB는 read-only/immutable로 열고 CameraIntrinsic
list wrapper만 제한적으로 해독했다. Stored K/D로 full-resolution in-memory rectification 후
상하28px crop, cv2 linear512×256, pixel-center affine/pinhole ROI로 규약을 고정했다.
원본/보정 ROI sheet를 직접 검토하고 calibrated-FOV distort/undistort 왕복 최대1.14e-12px 확인.
이는 실제 visibility/3D 정합 정확도 측정이 아니다. Raw distorted corner의 FOV 밖 외삽은
거짓 큰 ROI를 만들 수 있어 cache에 쓰지 않았다. 기존 단일 pilot의 기본 전처리는 보존했다.

OpenScene/nuPlan schema/MTGS Appendix A.1/OpenCV primary source를 확인했다.
별도 원본 sensor JPEG가 없어 local export의 byte-level rectification 이력은 증명 못했다.
Original-distorted로 취급하는 운영 가정을 명시했으며 occlusion/시간 차이는 여전히 미해결이다.
공유 원본을 변경하지 않고 image remap/cache/output은 모두 작업공간에 생성했다.

### 제한 cache와 runner

`cache_target_supervision_features.py`: 기존 manifest train277/dev96window만 생성, 373개 파일
합계616,266,713bytes. Full directory 약617MB로2GiB cap 아래. Train8 profile58.65s(첫 kernel
시작43.95s 포함), 나머지365개188.50s; clip9/window 총3357. Peak allocated1,479,913,472bytes,
reserved1,648,361,472bytes. Current grid/entity와 future ROI target을 분리 저장했다.
Current feature geometry10도FP16 저장/FP32 학습으로 미세한 거리 quantization이 있다.
Teacher는 같은 pinned official frozen encoder이고 미래 clip은 학습 target에만 사용한다.

`FixedDistanceFutureSupervisionPilot`/`train_target_supervision_ablation.py`: 현재 거리+stable ID
K4만 사용, learned scorer freeze/미호출. 같은초기state SHA와 recording-balanced batch sequence,
AdamW1e-4/weight decay1e-4/clip1, 조건당200update/batch8/1600draw. B–E는동일두head/branch,
A는branch가없어활성용량이작다. A762627/B–E2223283active params, stored scaffold2487988,
encoder303885312frozen을별도로구분한다. No official full-stack baseline/pretraining 주장.

Train-only selected common-valid4785slot/time에서 population std/floor0.05로 normalization;
floor에닿는채널0. Predictor는whitened좌표를출력해planner에전달하고forecast지표는역변환한다.
C/D/E common mask는 loss에만 사용하며 future validity로 현재 후보/window/planner를 거르지 않았다.
모든조건의공통availability27659slot/time/zero-common133sample draw가같음을hash로확인했다.
Aux→plannergradient정확히0, planning→P/D 정상, scorer변화0, P/D 실제update 확인.
Step0/50/100/150/200 평가, dev-best를고르지않고last200checkpoint보관. 추가seed/학습없음.

### 결과와 연구 해석

| 조건 | Train planning loss | Dev planning loss | Train scene-macro ADE(m) | Dev scene-macro ADE(m) | Dev FDE(m) |
|---|---:|---:|---:|---:|---:|
| A branch없음 | .14899 | .17995 | 4.950 | 5.921 | 12.065 |
| B planning만 | .14782 | .17940 | 4.853 | 5.881 | 12.212 |
| C visual aux | .14951 | .18031 | 4.968 | 5.921 | 12.067 |
| D spatial aux | .14517 | .17615 | 4.753 | 5.749 | 12.035 |
| E mixed aux | .14730 | .17869 | 4.846 | 5.872 | 12.046 |

모두dev ADE약9.25에서낮아졌고150→200에도0.74~0.88m 감소하여 아직수렴하지않았다.
D–B −0.132m는 초기관찰이며 target선정/공식안전성능/novelty의근거로확정하지않았다.
Scene86/devrecording4, paired1000recording bootstrap은설명용만. Context 표본부족flag유지.

**C의평균회귀·branch무시경고**: visual prediction/target분산비율0.011997, future swap의
window mean ΔADE+0.000572m, branch제거+0.010632m. Frozen encodercollapse는아니다.
E도visual분산비율0.120223. Commonnormalized visualMSE B/C/D/E1.737/.951/1.616/.977,
spatial1.445/.760/.355/.386. Currentpersistence visual.395/spatial.153보다아직모두높다.
Auxloss감소만으로좋은미래표현이라고해석하지않는다. 다른scene/recording예측swap이며미래GT아님.
Zero는time/type token이남고, 제거는futurememory전체를뺀다. 의존도는재학습대조의대체물이아니다.

Updatewall합26.22s, 전체invocation250.21s에eval/진단/IO포함. Cached-head속도이며전체pipeline효율아님.
Peakallocated172~239MB, processVRAMsample최대550~626MiB(일부측정unavailable).
GPU0종료후1MiB, GPU1기존1945MiB보존. 타인process중지없음.

### 검사·산출물·다음

38/38unittest(기존33+fixed-rule/normalization/loss/gradient/resize계약5), Ruff check/format 통과.
실행전GPU sandbox접근/초기CUDAmemory초기화/OpenCV5API오류를수정했으며실패output도보존했다.
독립summary 단계에서공유log16/image3730의pre/postSHA와실행source hash 일치를확인했다.
기존envupgrade/새의존성설치/공유원본쓰기/download/SafeDrive재학습/동적K구현없음.

Raw: `outputs/feature_caches/target_supervision_rectified_v1b/`,
`outputs/target_supervision_exploration/seed29_updates200_v1/`,
`outputs/projection_audit/front_camera_rectified_verified_20261001/`.
Shared: `results/target_supervision_exploration/seed29_updates200_20261001.json`,
`docs/target_supervision_exploration_results.md`. Code/config/지침/status/HANDOFF/README 갱신.
원본이미지/ROI sheet/checkpoint는Git원격에올리지않고summary/source만공유한다.

다음은C/E평균회귀·branch무시/학습량을검토하고충분한동일조건학습·다른seed,
current-target/current-feature대조의우선순위를정하는것이다. 초기순위로조건을탈락시키지않는다.
Target근거이후selector비교로돌아가며이번커밋에는자동추가학습/대규모cache/최종target선정없음.

### 2026-10-01 15:43 KST — 사용자 전원 중단 보고 후 복구 점검

README/AGENTS/HANDOFF와 git 상태를 다시 읽었다. HEAD09913b5이고 이번 구현·결과·문서는
아직 커밋되지 않은 상태였다. Cache FEATURE_CACHE_DONE, A–E CONDITION_DONE 및
TARGET_COMPARISON_DONE이 모두 남아 있었다. 이번 cache/training process는 없었고 현재
GPU compute process 목록도 비어 있었다. 완료된 학습을 다시 실행하지 않았다.

Cache373파일의 SHA256, checkpoint5개의 기록 SHA256과 CPU weights_only/strict model loading,
finite state 및 optimizer_updates200을 재검증했다. 공유 JSON/원본 실행 report/실행 source hash도
기록과 같다. CPU38/38tests 재통과, Ruff 및 git diff --check 통과.
따라서 손실된 학습 구간은 없고 남은 작업은 최종 커밋·push였다. 그 부분부터 재개했다.
기존 raw 결과·cache·체크포인트를 덮어쓰거나 추가 학습하지 않았다.

## 2026-10-01 16:25 KST — 9353acf 이후 C/E 저분산 진단·제한 추가학습

최신HEAD9353acf/clean worktree를확인하고README/AGENTS/HANDOFF 및실제runner/model/config/JSON을읽었다.
기존cache373/train277/dev96와A–E200 완료작업·복구검증을반복하지않았다.
GPU0/1 각각15MiB/compute없음, 타인GPU4–7 PID21618/33392/33393/33394를확인하고GPU0만사용했다.

기존C/E checkpoint로mean/persistence/horizon/분산축/교란입력RMS/currentROI입력을측정했다.
C devvisual MSE.950855 vs train-mean.949572/persistence.395004;target분산.916786은남는다.
Cswap visual입력RMS.1204/ego변화.00611m, E.4171/.3183m; donor120slot inactive confound확인.
작은swap만으로branch무시를확정하지않는다. Normalize inverse1.91e-6/manualloss5.96e-8오차.
Raw373구간 track순서/spatialtarget/valid/gather일치, nominaltimestamp최대편차.012979s.
Source/cache일관성검사이지GT좌표현실정합의독립증명은아니다. Visualteacher전부재계산하지않았다.

Projection기존4표본감사를재사용하고대표2sheet를다시직접봤다. 원본DB K/D/크기일치 및왕복오차는
이미보정된JPEG에추가보정이필요하다는증거가아님을명시했다. OpenScene문서/collect_data.py/MTGS A.1을
직접읽었지만로컬JPEG export이력은확정못했다. 조건부가정유지;근거없는adapter변경/cache재생성없음.
iPad arXiv2505.15111v1 §3.4/4.3 Table4와README를직접읽었다.90.5→91.7은General→Proposal-centric
prediction교체이고89.8→91.7은mapping까지바뀐다. Learned selector/adaptive budget효과로인용하지않음.
기존EgoFSD/ForeDrive감사의원문/구현미확인수준을구분;새모델재현은없다.

사전config `future_prediction_diagnostic_followup_v1.json`: 추가7400update/whole1800s/run600s상한,
A seed11/47 각200을먼저실행. 이후seed29 A–E총1000(model/AdamW200복원+800), F29/E11/F11각1000.
기존initialhash/첫200sampler/window순서/AdamWstep200검증. 구RNG없으므로optimizer-state continuation,
bitwise exact resume으로부르지않는다. 새checkpoint에는optimizer/sampler/Python·NumPy·Torch CPU/CUDA RNG저장.
F는planner입력만detach하고반환두head는attached. Planning→P0/aux→P양수, aux→D0확인.
Forward불변/두head학습과분산축계약새3test통과,관련기존5test/Ruff/diff통과. 기존전체38/recovery재실행없음.

총7400update/235.33s완료. DevADE seed29 A/B/C/D/E/F=1.6717/1.7445/1.6485/1.6991/1.5747/1.5536m。
A200 seed29/11/47=5.9208/6.3224/5.6591,std.3341m(A다른조건변동을대신하지않음).
E/F29 F이득.0211m,11 .0508m;11의planningloss는F가소폭나쁨. 두seed/4devrecording으로F우월성주장안함.
C200→1000 visualMSE.9509→.7936, 분산비.0120→.2137; E.9773→.8344/.1202→.2025。
모든horizon persistence보다MSE나쁘며encoder collapse/visual무효/유용한시간예측/H1·H2/novelty를확정안함.
초기저분산에학습량이관련됐지만capacity/조건부평균/aux regularization/현재side-channel/실제미래활용은미분리.

새final10checkpoint strict/finite/optimizerstep/samplernext/CPU RNGreplay검증통과. CUDA snapshot은저장만검사,
CUDA RNG실제replay는CPU검사에서안했다. 종료후GPU0/1 15MiB/타인PID보존,추가학습없음.
Raw `outputs/future_prediction_diagnostics/bounded_followup_1000_v1/` 약767MB,
초기진단 `seed29_update200_v2/`, share `results/future_prediction_diagnostics/`4JSON。
보고서 `docs/future_prediction_variance_followup_results.md`, scope/evidence/config/재현명령/HANDOFF/status갱신.
원본/cache/구checkpoint/env/타인process/SafeDrive자산/저장소공개설정보존。

다음후보는현재feature+delta residual한가지, 기존C target/계수/K유지하고별도1000update/600s상한을
사전고정하는안(미실행). 더실패하면target진단을계속연장하지않고front/target제약을검토한다.
K연구전multiview/중복제거/turn·interaction coverage와JPEG근거를보완해야한다.
확률적mean/covariance/calibration/계획제약은후속설계만; learned selector/동적K/horizon/대규모cache/학습없음.

16:36 추가정리: B/D200의zero/swap 실제입력RMS도CPU-only로측정해A–E교란표를완성했다.
추가optimizer update없음. As-run source4개/config SHA는현재파일과일치한다.

## 2026-10-01 — `607da52` 이후 기준선 재현·bounded residual·공개 기반 결정

기존373cache/normalization/seed29 A–F1000checkpoint를 재사용했다. 전부 CPU 실행.
Dev scene-macro ADE: 정지9.1239/CV1.0951/CA0.9260/train-fitted ridge0.7677m.
Ridge는 train-only3 recording fold로 lambda0.0001 선택, fold별 표준화/최종277train fitting.
공간+4초 common215관측: CV2.0197m vs D/E/F6.3530/5.5511/5.6630m; yaw/velocity도 horizon별 계산.

고정 설정의 한 가지 visual-only residual 대조: seed29/11/47, 1000update, aux0.1/K4 유지.
기존 C29 재사용, 나머지5run 총5000update/364.26 CPU초. Ego/spatial residual은 이 학습에서 미사용.
Absolute C dev ADE1.6485/1.7126/1.7588, visual residual1.6189/1.8104/1.8341m.
대응 차이−0.0297/+0.0978/+0.0753m, 평균+0.04783m. 4recording 조건부 cluster95%CI[+0.01372,+0.08644].
Visual MSE0.3789/0.3815/0.3833으로 persistence0.3950보다 조금 낮지만 큰 개선은 구조적 skip 효과.
G1 약세는 pilot 판단이며 미래 예측 가치/선택 가설 기각이 아니다. 새5checkpoint strict/finite/sampler/RNG presence 확인.

현재 rear-axle 고정 frame/ego status 채널과 scaled entity state의 단위를 코드로 재확인했다.
Prior+zero-output-head/invalid slot/초기 upstream gradient0→1update후양수/detach-aux 계약6tests.
Recording/group cap/미래 이미지 미필터/scene-macro/cluster estimand6tests를 추가했다.
GPU graphics 메모리 guard1test도 추가해 최종 전체54tests 통과(PYTHONPATH=src:scripts), Ruff 통과.
처음 path 없는 discovery의 기존 import1건 실패도 기록했다. Compute-apps 없음만으로 GPU를 비었다고 판단하지 않는다.

Navtrain trainval1192segment/162group, cap24 보존한 고정3-way:train1857/82group,dev865/40,held898/40.
Held는mini52group 제외, native recording/log/current token 경계와 비중첩 검사. Held 모델 평가는 미실행.
Cap 이론상3888으로 요청12000에 부족하므로 수를 강제로 맞추지 않았다.
JPEG4표본 stored/pinhole vs stored/distortion vs undistorted/pinhole 민감도39.69 CPU초.
OpenScene helper source는 경로/K/D 복사이고 local export 이력은 없어 보정 타당성은 여전히 미확정.
Fail-closed cache runner는 구현했지만200encoder profile/확대cache/확대학습은 미실행.

WA-JEPA source bec29660f5ea46ac73db8e2d0c33c8d1a72c23ad를 조사했다.
공식 joint attention의 tiny CPU backward에서 trajectory→scene flag true는 input0.01519/QKV0.11358,
false는0. Official HF revision15c0770ebd233665214590bb2190a90907499f9e의 public/ungated weight
1,575,763,741bytes/LFS SHA 확인. Weight 다운로드·strict loading·full model 추론·공식 score는 미수행.
Official future8192/context4096 token의 dense 처리, 고정 shape, future positional index 변경 필요를 확인했다.
ForeDrive v2 원문을 다시 읽었지만 제한 검색에서 공식 code/checkpoint 링크는 확인하지 못했다.

사용자 후속 지시로 predictor 튜닝/대규모 pilot 확대는 보류하고, 공개 future-planning 기반 재현 후
같은 예산의 상황별 차이→학습형 선택→평균 예산 배분으로 복귀한다. WA-JEPA 첫 재현 후보 권고.
GPU0/1은 조사 초반 타인 CARLA 점유, 마지막 확인25MiB/compute 없음. 이번 GPU 사용/타인 중지 없음.
공용 원본/이전 cache·결과/환경/SafeDrive 자산/공개 설정 보존.
두 보고서와 공유 JSON/manifest·window별 결과를 추가하고 README/AGENTS/HANDOFF/status 갱신.

## 2026-10-01 — 공식 Drive-JEPA planning checkpoint 재현 준비 / GPU0·1 전체 평가 시작

기준5c6e6d5(clean), 실행중우리pilot학습/cache없음. 기존자산보존/확대·pilot·WA보류.
Source548bb82/HF65e0d728/NuPlanv1.2ce3c323/논문v2Table2PF89.0을고정했다.
독립Conda Python3.9.23/Torch2.1cu121, 기존base/pilot환경·OSdriver변경없음.
Fullplanning3.72GB/공식metriccache3.18GB새다운로드, 기존encoder5.13GB재사용, 모두SHA256일치.
공식source별도worktree의encoder로더만strict강화하고전처리/planner/scorer는공식그대로사용한다.
12146token/log136/front14247image/cache12146누락0. OldYAML도12146이고sampleCSV12147행은average1행포함.
3scene fullplanner·encoder strict/유한8×3trajectory/scorer통과. Epoch36/globalstep49173 체크포인트.
사용자추가요청으로전체공식split을GPU0·1 execution shard6075/6071(68log씩)로병렬실행.
각worker2/총4, CPU스레드1/worker, 겹침0/합집합전체. source/scorer/precision변경없음.
전용tmux socket drive-jepa-official-evaluation, gpu0/gpu1. 최종수치아직미확정.

## 2026-10-01 — 공식 Drive-JEPA full planning 전체 평가 완료

준비18663fc의 독립Conda/공식source548bb82/공식weight·cache/SHA와preflight·smoke를재사용했다.
GPU0·1 각각6075/6071scene 성공, 원본평균행을제외하고전체12146scene을합쳐집계했다.
실패/누락/추가/중복/비유한metric0. NC99.082002/DAC96.558538/EP83.034487/Comfort99.983534/
TTC96.023382/DDC98.196937/PDMS89.224320. 논문v2Table2 PF89.0 대비+0.224320점.
원논문run별체크포인트card/환경lock/sceneCSV/공식허용오차없음으로정확원인은미확정이다.
공식source·checkpoint·변형/split completeness/cache·scorer/전처리·센서/실행환경 순으로 감사했다.
저장된각Hydra의agent/simulator/scorer/sampling/센서/split/path가고정설정과일치했다.
수치를맞추기위한추가학습/설정변경/다른ckpt/sweep는없었다.

실제평가21:14:44→GPU1 21:26:34/GPU0 21:27:05 KST, wall약741초.
GNUtime740.59/710.22초, shardwall합0.403GPU-allocation-hour(실제SM active시간아님).
GPU0·1총worker4; 도중시작15초표본VRAMpeak7810/7811MiB, 전체연속peak아님.
Worker증가요청에는upstream 메모리누적/CSV마지막저장과거의완료진행률을확인하고재시작하지않았다.
Worker scaling benchmark/추가평가없음. 종료후GPU0·1 25MiB/우리compute process없음, tmux세션정상종료.
다른연구원의GPU3–7 작업/공용데이터/기존cache·환경은건드리지않았다.

Fail-closed 집계5CPUtests(평균행제외·unequal shard/누락/중복/invalid/NaN)와관련Ruff통과.
OfficialCSV별SHA+정렬sceneCSV/결과JSON/실행비용·Hydrahash/telemetry/report·재현명령을공유한다.
이것은encoder-only가아닌full planning checkpoint 추론·평가재현이다. 논문정확수치재현/선택·예산효과는미확인.
PF 추론에는별도future predictor가없다. 자체pilot/확대/WA-JEPA는사용자와결과검토전까지보류한다.

## 2026-10-01 — 578be6e 보존 / 미래 예측 기반 코드 감사·현재 상황 현황·최소 선택 실험 명세

시작HEAD578be6e/clean, 기존공식PFViT-L12146성공/PDMS89.224320과+0.224320미확정원인을보존했다.
공식결과디렉토리+고정config13개trackedfile이578be6e와byte hash동일함을검사했다.
Source Drive548bb82/WA bec2966 reference tree도변경없이읽었다.

DrivePB의ImgEncoder/Traj_refiner/Bev_refiner/Scorer/TargetBuilder/agentloss/Lightning/eval를읽었다.
Scorer futurecollision/areahead는train-only이고planner는그출력을소비하지않는다.
WA positioner/jointattention/predictor/teacher/flowloss/inference/trainer/loader/config/strictload를읽었다.
Inference에서scene+ego4stepjoint갱신,trajectoryloss→scenehiddenK/V경로확인.
Single-forward scene_out은trajectoryloss의ancestor가아니라는예외를명시했다.
Official dataset의storedcommand누락시future-derived fallback도찾아선택실험에서는금지하기로설계했다.
ForeDrive v2 원문/제한공식링크조사는paper detach만확인,실제공식code/weight위치미확인으로남겼다.
공식HF pinnedrevision의작은tree JSON만조회,PB v1/v2·WAweight name/size/LFShash공유.
Weight 다운로드/full loading/새환경/학습/GPU/추가평가없음.

Current speedbins0.1/2/5/10mps와raw4onehotcommand/min100scene·10recording을scorejoin전에기록했다.
모든12146scene currentframeindex3에서metadatalabel생성완료후에만저장scoreCSV를join했다.
12146scene/136native recording/136exportlog,metadata/speed/command/logtoken결손0.
Speed5bin+invalid/command4+invalid/교차30cell을exhaustive집계했다.
Command left2501/forward8070/right1575scene,PDMS88.806043/90.153576/85.127191.
이는coverage/기술통계이고H1·미래필요성·selector개선증거아니다. Navtest tuning없음.
CPU16.286952초,기존공식Conda사용/sensor로딩없음. 원본CSV SHA
56010fc01ae91818daef49ec63f38614bf1db6913b6e672c354d650891f99ce5 전후동일.

WA nativecamera/spatialpatch-tube를첫추천기반으로명세했다. 객체instance범위유지안과비용·위험을비교했다.
Camera당128tube/총2048futuretoken/4step/고정horizon,현재context4096유지.
Prediction전packing/indexinterface변경, position-mediatedST/auxpolicydetach경로는설계만이고미구현.
Original/random/fixed-rule/learned/all-future참조와matchedfine-tune/currentfeature대조/zero-swap역할을분리했다.
Strictweight/config/fullgradient/actualcost/4viewcoverage/all-ID동등성은미확인gate로남겼다.
범위승인후에만공식WA소수train/dev동작검사를진행하고새selector·대규모학습을자동실행하지않는다.
DynamicK/horizon/확률적predictor/navhard/pilot·확대·residual/SafeDrive는계속보류.

Newcontext7+기존officialaggregation5CPUtests통과, 관련Ruff/diffcheck/새문서link/JSON검사통과.
Shared산출물약1.72MB,currentmetadataCSV와summary/config/metadataJSON/두report/상태문서갱신.
최신보고 docs/future_prediction_foundation_decision.md, docs/official_navtest_current_context_summary.md.

## 2026-10-02 — WA-JEPA official reproduction preparation (baseline fd5fc5f)

사용자 승인으로camera별 spatial patch-tube/공식원본재현/sparse진단을준비한다. 학습형selector/학습금지.
Sourcelatestbec2966은4-step,직전공식404d8af/publishedstate.pt/paper는12-step. 결과조회전404d8af고정.
전용Conda를known-workingDrive환경에서독립clone하고새officialNAVSIMv1/sourceworktree에연결한다.
HF고정revisionplanning/state와Metaencoder다운로드완료. Preflight/strictloading/원본agent·scorerharness추가.
GPU0타인CARLA는건드리지않고GPU1우선. Full은등록된completeness/smoke/costgate통과후한번만실행.
새config: configs/official_wa_jepa/reproduction_v1.json;진입보고 docs/official_wa_jepa_reproduction.md.

## 2026-10-02 — WA native smoke/interface complete; dense benchmark running

Preparation commit b81b1bf를mine에push했다. Official12-step404d8af strict1162keys shape/missing0.
6scene원본agent/scorer성공,현재4-view[4,4,3,256,512]/historytraj[4,3]/ego8,출력[8,3]/0.5s.
YAMLBF16이나원본agentparameter787692035개FP32/noautocast. HarnessTF32는pinnedYAMLtrue명시적적용;
bareupstreamlauncher가globalTF32flag를설정하지않는실행차이도기록했다. NumPy/Torchdeclaredversion차이는공개.
All-ID6scene×12step bitwise동일,rawPE/normalizedPE/noise/predictor/Eulertransition/ego검사.
Originalreference trackedsource보존,수정은별도worktree;Git공유patch를추가했다.
Fixedlattice/random29camera당128tube→future2048;실제QKV/FFN2048/current4096/ego8shapehook통과.
72timedtrial(6scene×4condition×3repeat),original6.694s/packedall6.713/fixed2.078/random2.064;
predictor6.413→1.81s. 6scenePDMS96.641/96.641/96.207/96.353은untrained민감도만,선택학습성능주장금지.
Allocated3.440→3.400GiB,Reserved4.980GiBallocatorcarryover불변;전체VRAM절감주장불가.
첫costhook이시간modulationLinear를tokenFFN으로오분류해수정,완료all-ID는재사용했다.
초기nohupbackground는유지되지않아결과손실0. Dedicatedtmux2GPU0worker가32scene저장 후사용자증설요청.
확인된우리PID2207310/2207311만SIGINT,32records그대로보존/14-waypartition에서재사용한다.
사용자45GB승인→각GPU7/총14worker,실제NVML39880491008bytes각카드/OOM0/RAM>300GiB.
Memoryguard는등록UID/PID/command/shard검증후에만SIGINT. 타인프로세스/공용데이터수정0.
전14shard자동집계는정확12146token/중복·누락·추가검사뒤에만fullJSON/CSV생성. Full은아직진행중.
다음세션은완료여부/완전성/Table3차이원인을확인하고보고/commit/push. 학습/pilot/selector재개금지.

## 2026-10-02 10:32 KST — 사용자 요청 kjs 명칭으로 평가 pause/resume

사용자이니셜kjs에맞춰실행별칭 `/rhome/junseong/envs/kjs-wa-jepa-eval`을추가했다.
실제Conda prefix `runtime/environments/wa_jepa_official_evaluation`은이동/재설치하지않는다.
처음에는새실행부터적용하려했으나사용자가현재평가도잠시중단/재개하도록명시적으로요청했다.
Manifest의UID/script/shard/physicalGPU를검증한우리14worker만SIGINT하고30s이내정상종료확인.
562unique완료scene/실패0,기존JSONL/로그그대로보존. `kjs_process_label_pause.json`에PID/명령/token기록.
동일14-waypartition/원본agent/12step/seed/scorer/weight로각GPU7worker재개,완료scene재평가없음.
새launcher는별칭Python절대경로를사용하며guard/자동aggregate는그대로작동한다.
명명검사용15sCUDA컨텍스트306MiB에서nvidia-smi가kjs별칭을표시함을확인했고임시프로세스는종료했다.
타인프로세스/공용데이터/과거결과수정없음. Full전체평가결과는아직미완료다.
재개후추가28scene/총590완료·562전부보존·중복/실패0,14active/각GPU39880491008bytes확인.
환경별칭/기존6검사/순서partition 포함7tests+canonicalID5tests=12통과,compileall/diffcheck통과.
공유기록 `results/official_wa_jepa_reproduction/process_label_resume.json`;전체평가완료보고와구분한다.

## 2026-10-02 19:12 KST — 사용자요청GPU0/1반환 / 재개가능상태보존

다른연구원GPU사용을위해우리14평가worker에만UID/script/shard/GPU검증후SIGINT했다.
우리CPUguardPID2345068/aggregatePID2477320도UID/실제상대script경로/subcommand검증후정상중단했다.
GPU0/1각25MiB/util0/우리compute없음;dedicatedtmuxserver종료확인. 타인process수정0.
완료8686/12146(71.513255%),남은3460,실패·중복0. 16JSONL SHA256+token/rowcount검증통과.
원본source/weights/Conda/cache/seed/12step/14waypartition불변;과거32scene도그대로보존했다.
첫pause는SIGINT완료후argparse変数와cmdline변수충돌로summary작성오류,이름분리후이미종료된상태에서
inventory를복구했다. 복구invocation은추가stop0이정확하며완료scene손실없음을SHA/token으로확인했다.
Pause marker와명시적resume gate를추가;기본launcher/directfull/smoke는기동차단한다.
`--resume-user-paused`는나중에사용자가재개요청하고GPU배정확인했을때만사용한다.
CPUguard/aggregate는재개시함께복원. 완료scene는skip하고in-flightscene만다시계산한다.
2.89MB별도tar snapshot보존;SHA/파일크기/경로는paused_snapshot_backup.json에기록했다.
Shared paused_evaluation_state.json/상태문서/재현명령갱신. CPUpause3+기존official7tests통과.
GPU재시작/학습/튜닝/새평가없음. 사용자의재개요청전어떤GPU작업도자동재개하지않는다.

## 2026-10-02 20:40 KST — 사용자요청 공유 GPU0/1에서 bounded 재개

기준5ed8a60/clean tree. 다른연구원GPU작업이남아있지만사용자가여유범위에서워커를줄여재개하도록요청했다.
16JSONL SHA256/8686scene·실패0/config SHA확인 후pause marker를acknowledged로보존했다.
원본reproduction_v1.json/source/weight/Conda/12step/noise/scorer/14-waypartition을바꾸지않았다.
새shared_gpu_resume_v1 profile: 각GPU 최대2/총4worker, 입장12GiB(로딩중worker추가reservation차감),
worker 예상peak6GiB/running reserve6GiB/host24GiB/poll5s. OOM 완전보장이나타인메모리예약은아니다.
새CPUqueue는현재구간이끝나면같은GPU의다음구간을처리하고완료scene를재평가하지않는다.
Failed/guard-stop 자동재시도없음. Guard가압력marker를남기고등록된UID/script/shard/GPU검증우리worker만SIGINT.
Launcher는특정기존shard를선택할수있고manifest atomic write로감시JSON읽기경합을방지했다.
Dedicatedtmux guard/queue/aggregate복원, GPU0 workers55343/56687, GPU1 workers58708/59289기동확인.
시작직전GPU free25539/18834MiB; warmup이후21875/7976MiB, 네worker가strict model loading을통과했다.
초기38scene추가/총8724/실패0/pressure0; 기존8686행prefix SHA256는전부일치한다.
새6CPU scheduling tests+기존3pause tests통과. 결과는진행상태이며최종공식평가완료가아니다.
Runtime bounded_scheduler_status.json/로그로추적; 공유shared_gpu_resume_state.json은재개시점의기록이다.
다른userprocess/공용데이터/기존Drive·pilot결과를수정하지않았다. 학습·selector·pilot재개없음.

## 2026-10-02 21:03 KST — GPU0증설 / 모듈별추론병목진단

사용자요청으로GPU0상한2→5/GPU1은2유지. v2 GPU별상한profile/CPUqueue liveworker adopt추가.
UID/script확인된우리CPUqueue55310만SIGINT/교체; 기존GPUworker중단0.
구간4·6·8의worker278797/280724/281681추가. 기존14-waypartition/config/12step/seed/scorer/가중치불변.
Launch12GiB/reserve6GiB/우리PID만guard/pressure-stop 자동retry금지 정책을유지한다.
공유gpu0_worker_increase.json/runtimequeue로그와상태로실측추적. Userprocess변경없음.

사용자추론시간질문: 이전단독6scene18dense측정predictor6.41256s/agent6.69444s(95.79%)를재사용했다.
Encoder와packing은미분리였으므로짧은공식model1개diagnostic을GPU0여유범위에서실행했다.
Shareddense1scene encoder0.54474s/predictor24.10485s(12actualsteps,mean2.00874)/model24.69828s(97.60%).
첫dense측정후packing AST진단이no_grad wrapperglobals를사용해NameError; inspect.unwrap으로수정하고
완료dense측정을재사용/packed-all·fixed만수행했다. Officialevaluation실패가아니며기존결과변경0.
Packed-all finaltrajectory bitwise동일. Fixedpacking27.957ms/ID생성9.918ms; sparsepredictor13.195s/model14.236s.
동시worker수2→5/타인workload변동이있어new조건간ratio를통제된speedup으로해석하지않는다.
CUDAelapsed는경쟁/CPUlaunchgap포함. SensorIO/scorer제외,onewarmup+onetime/condition.
최초dense2call+추가packed/fixed4call,15분상한내완료/diagnosticprocess종료. 평가는계속실행중이다.
새2CPUprofiletests통과, namespace오류는테스트로회귀방지. JSON module_timing_summary와보고서에한계기록.

## 2026-10-02 — WA 보존·중단 / 공식 Drive-JEPA 선택적 미래 연결 gate

기준865be79/clean tree. 사용자최신지시로WA를정리하고공식Driveplanner를재사용하는extension을승인받았다.
우리7WAworker만UID/script/shard/GPU검증후SIGINT, CPUqueue자체종료/우리guard·aggregate종료.
9253/12146성공/실패·중복0/76.181%에서중단, raw16JSONL SHA256검증. 남은2893은자동재개금지.
PartialPDMS91.102506 vs동일9253scene Drive89.019762(+2.082744점). 전체점수로해석하지않는다.
원본scene별부분CSV·JSON와새2835084bytes snapshot보존, snapshot SHA256
6aed87ec3feea7ecb7c9813ab0855af3fea84c3036e21f5b894eadb14f71249b.
WA sparse/all-ID/비용 및원본Drive전체89.224320/weights/Conda/pilot자산은그대로유지했다.

실행전connection_v1 config/design고정: front512patch중learnable K4/uniquehard·softbackwardST,
current feature·position·ego8채널→경량4tubelet latent predictor→zero-init residual128image memories
→원본DriveTransformer/trajectoryhead. 원본planner를새로만들지않고모듈identity/weights hash검사했다.
원본309981955params freeze/eval, 신규1271489params; futuretarget은동일FULL planning encoder의fixedcamera-grid latent.
GT/미래validity는loss밖online forward에없다. Aux는detachedhardselection으로predictor별도호출→selector/bridge직접gradient차단.
ST는current내용·좌표mixing의biasedsurrogate이며hardID교체의미래target/중복제외gradient는근사하지않는다.

Conda패키지변경없이기존Drive전용환경을읽기전용재사용; 별칭envs/kjs-drive-jepa-extension.
첫실제run은strictweights통과후exact500000us cadence검사에서중단; 원본jitter약0.5ms확인.
기존adapter0.05s검사tolerance로수정/원본frame·offset유지, 처음실패디렉토리보존/결과튜닝아님.
같은사전선정규칙의2개officialnavtrain native recording으로실행/val·navtest fitting없음.
현재파일존재조건만입장에사용; 한window의+3/+4s future사진누락도window를유지하고loss만mask했다.

공식strictfull missing/unexpected0, off전후/init-on(no-grad)원본bitwise동일.
Planning atzero: selector0/predictor0/bridge.123523. 1bridge diagnosticstep후S1.30145e-5/P1.14573e-5/bridge.123518.
Futureaux: S0/P1.247975/bridge0; futuredetach forward동일/S/P0/bridge양수; selectiondetach S0/P양수.
원본gradient0/weights hash전후동일. Predictor실제query[2,4,4,2074]→latent[2,4,4,1024], ego[2,8,3].
Futuretubelet6/8valid/selected24targets만감독. Selector/predictor는아직유용한선택·미래예측으로최적화되지않았다.
Optimizer1step은초기zero bridge출력projection만갱신; 이후nonzerogradient검사가학습성공을뜻하지않는다.

1warmup/5repeat/2windowbatch sharedGPU0:원본.136506s/off.129595s/on.196451s, peakallocated1.261GiB.
CUDAhook1trial encoder176.909ms/selector.241ms/predictor.045ms/bridge.106ms/원본Transformer.866ms.
공유GPU점유변동을포함해전체차이를순수moduleoverhead로주장하지않는다. SensorIO/targetteacher/scorer제외.
성공run전체56.673s/최대15분·peak6GiB상한준수/OOM없음/종료후우리GPUprocess없음.
공유JSON results/drive_jepa_selective_future/connection_v1_20261002.json, local원시로그/1stepstate는outputs에보존.
최종CPU전체101tests(11신규)/Ruff/gitdiffcheck통과. 구현/runner/config해시와원본source/input해시를기록했다.
Gate는원본보존·계산그래프·실행가능성만통과했다. 성능/최적선택/미래정확도/안전향상/novelty는미검증.
WA/pilot/큰학습자동재개없음. 다음은같은K random/규칙/learned/currentfeature대조의작은train/dev 계획을검수한다.

### 후속 split 감사 — 위 첫 contract 진단은 내부 train-only 요건을 만족하지 않아 기각

Officialnavtrain 안에도기존projectheldout/dev가있음을뒤늦게확인했다. 위56.673s진단의두recording은
2021.05.12.19.36.12_veh-35(held_out)/2021.05.12.22.00.38_veh-35(development)였다.
단일bridgeupdate/gradient·loss열람을이미수행했으므로해당heldoutgroup을향후extension의미사용독립평가로취급하면안된다.
원본split은그대로보존했고무단재배정하지않았다. rejected_shared_report와exposureaudit로공개/사용자에게알렸다.
위report/state는보존만하고재사용하지않았다. 새필터는기존manifest assignments.split_by_recording==train을
window로딩전에확인한다. Config에manifest경로/필수train조건추가, 회귀2tests로이전recording입장차단을확인했다.

최종v1c_train_split은원본checkpoint/branch seed29에서새초기화. 실제recording은
2021.05.12.22.28.35_veh-35 / 2021.05.12.23.36.44_veh-35, 둘다projecttrain/서로다른native log.
Source/splitmanifest SHA기록, off/init-on bitwise/원본weights hash불변/gradient계약은다시통과했다.
Bridge1step후S9.394805e-6/P1.050055e-5/bridge.101464; auxS0/P1.598178/bridge0.
Target5/8tubelets valid/selected20targets. +2/+3/+4future사진없는trainwindow를그대로유지해loss만mask했다.
최종two-windowbatch original.126048s/off.110916s/on.120065s. Encoder118.530ms/selector.232ms/predictor.044ms/bridge2.439ms.
SharedGPU평균on이원본보다작아도speedup으로해석하지않는다. 최종진단45.176s/peak1.261GiB/종료.
전체CPU103tests/Ruff통과. 공유connection_v1_20261002.json은이최종train-only결과이고,
v1b결과는connection_v1b_project_split_rejected_20261002.json으로이동/보존했다. 과거측정값은바꾸지않았다.
진행조건/미검증주장범위는동일하다. 학습형선택성공/성능개선/미래정확도를선언하지않는다.

## 2026-10-02–03 — Drive-JEPA 선택 비교 보존 및 구조별 추가 학습

기준2185ce5 이후 완료한 선택 비교를 보존했다. Train128/dev64, recording16/8,
fixed/random/learned/no-aux×seed29/47/83×200update, 원본 frozen planner를 재사용했다.
Dev scene-macro ADE는 원본0.220644m, fixed0.215851m, random0.217723m,
learned0.242582m, learned-no-aux0.238094m. Learned는 초반 개선 후 dev가 악화했다.
같은 patch의 미래 MSE가 persistence보다 나빠 유용한 미래 예측이라고 주장하지 않았다.
Raw `selection_comparison_v1c_20261002`, 공유 `selection_comparison_v1_20261002`.
실행 약576초/peak allocated1.476GiB, 원본 hash 불변. 이 결과를 재계산하지 않았다.

사용자가 아키텍처 개선 방향의 추가 학습을 승인했다. `d3bbced`에 코드·사전 계획·기존결과를
커밋하고 같은192window에서5조건×3seed를 실행했다. 현재 encoder-prefix만402,888,384bytes
추가했고192개 원본 최종 feature와 차이0을 확인했다. 데이터 확대·미래cache 재생성은 없다.
조건은 기존MLP/contextual residual/ego-query/last4 QKV LoRA/현재target대조다.
각각 randomK4 auxwarmup100+learnedjoint200, 고정LRcosine/aux가중치/최종checkpoint를 유지했다.
15run/4500update 완료, 전체1300.33초/학습run합1205.60초, GPU1만/peak allocated2.674GiB/OOM0.
LoRA는 별도 future branch tail의65,536factor parameter만 학습한다.
Teacher는 full planning checkpoint의 frozen encoder이고 EMA를 도입하지 않았다.

Dev ADE3seed평균±표준편차:

- MLP+새절차:0.209975±0.008071m
- Contextual residual:0.223808±0.006583m
- Ego-query:0.216938±0.008604m
- Ego-query+LoRA:0.216459±0.009297m
- LoRA+현재target:0.219851±0.009602m

MLP 원본대비 차이-0.010669m이나 recording cluster95%CI[-0.025015,+0.000101]이다.
이전 실험 대비 절차와 초기화가 함께 달라져 warmup 단독 효과로 해석하지 않는다.
Residual의 같은 patch 미래 MSE는 persistence를 넘었으나 planning 이득과 같지 않았다.
LoRA 추가 차이-0.000479m/CI[-0.001177,+0.000231]로 추가 이득 미확인이다.
계획상 마지막joint200을 보고했고 좋은 중간 checkpoint를 고르지 않았다.

실제 공식 모델 planning→selector/predictor/bridge/LoRA, aux→predictor/LoRA만 확인했다.
Aux selector/bridge0, 원본parameter gradient없음/hash 전후 동일이다.
학습 후 다섯 조건 seed29의 trainable delta 엄격 key/shape 복원과 RGB2window 검사도 완료했다.
최초 postflight 참조와 후속출력의9.5367e-7 차이로 bitwise gate 실패를 기록했다.
동시 반복 원본과off차이0/hash동일; 동결·warmup 조건 일치 후off bitwise 통과했다.
Online/cache 최대차이1.90735e-6은 사전FP32 기준 안이다. 실패JSON도 공유/보존한다.
RGB2window 원본81.1ms/MLP90.5ms/LoRA83.6ms이나 shared GPU 변동으로 speedup 주장 금지.
Cached branch+planner는 MLP5.14ms/ego-query8.16ms/LoRA20.32ms다.
검증 종료 후 우리 GPU 프로세스 종료. 공용데이터/다른연구원/환경/원본checkpoint 불변이다.

Raw `outputs/drive_jepa_selective_future/architecture_followup_v1_20261002/`.
Shared `results/drive_jepa_selective_future/architecture_followup_v1_20261003/`.
보고서 `docs/drive_jepa_architecture_followup.md`, 원문학습 source는d3bbced/SHA 기록.
완료 후 formatting과 동기 lambda의 loop binding 정리만 추가했다. 추가 학습/sweep은 없다.
CPU116tests/Ruff/전용Python3.9compile 검사. 복잡한 구조의 승격을 보류하고 MLP를 저비용 개발
참조로 보존한다. 후속은 같은 절차의 fixed/random/learned 비교 여부를 검토한다.
Held-out/navtest 공식평가·선택우월성·안전성 개선은 이번 범위에서 검증하지 않았다.
GitHub push는 기존 VSCode credential socket 오류 및No anonymous write access로 실패했다.
공개설정/원격/credential은 변경하지 않았다. 로컬 커밋은 보존하며 인증 복구 후 push한다.

## 2026-10-03 밤샘 원인 분리와 후속 선택 비교

사용자가명제를유지하고다음오전09:00KST까지질문없이문헌조사/실험/기록을진행하도록승인했다.
기준8159aad. 8f5b18d에서7조건×3seed×400jointupdate를사전등록했다.
기존checkpoint읽기전용진단149.06초/1.571GiB. Future경로half-gain개입은ego .216938→.209855,
gradient충돌은24trainbatch중ego54.2%/MLP62.5%이나인과원인확정아니다.
ForeDrive원문§3/AppendixH의outputdetach와우리parameter-onlyfreeze를구분했다.
PCGrad원문§2.3을읽고planning-priority one-sided변형으로명시했다. Authorcode재현주장없음.

기본격리환경CUDA차단으로업데이트전실패했고호스트GPU1로실행했다. 첫21run은1096.94초완료.
Fixed400 dev mean:MLP .232000,ego .239397,aux-only .236099,projection .239264,
halfbridge .233743,frozenS .234913,uniformADE .242933m. 원본 .220644m보다모두높다.
Train은줄어드나작은데이터과적합으로원인을단정하지않고coverage검사를이어간다.
원본parameterhash불변/gradient계약통과/현재첫학습종료,실패/낮은성능도그대로공유한다.
Shared `results/drive_jepa_selective_future/overnight_causal_followup_v1_20261003/`.

9b28ebc는후속표본train512/64group,추가dev192/24group을결과전에고정하고단일GPUqueue를구현했다.
Oldtrain128포함/기존개발recording제외/heldoutnavtest불사용.9GiBcache상한,기존128cache체크섬재사용.
9조건×3seed×800update:학습표본크기/기존구조/gradient분리/half-gain/목적함수/
fixed·random·learned선택/현재feature대조. 전체CPU124tests통과.
01:14KST queuePID2134994/cachePID2135322실행확인. 실제현재단계는active_stage.json참조.
09:00deadline/메모리압력/단계실패시중단하며원인미확인상태에서자동재시도하지않는다.
GitHub push는기존credential socket거절로실패했으며remote설정/인증정보는변경하지않았다.

### 2026-10-03 01:47KST — 등록된 후속 queue

704-window cache 완료: 논리7,385,149,376bytes, 신규6,042,394,944bytes,128파일 재사용.
301.14초/peakallocated1.207GiB/원본hash불변. GPU1에서27run coverage/selection 비교 진행 중.
4355bf2는 future projection9run, lower-LR/메모리 잔차 penalty12run을 결과 전에 등록했다.
동일512train/192dev/3seed/800update, 원본 planner 보존, 새로운 cache/benchmark 없음.
첫 queue PID2134994 뒤 두 번째 PID2296713이 대기한다. PID는 실시간 명령과 함께 재확인한다.
두 번째 queue는 모든 후속 완료 시12개 지정 결과 파일만 커밋하고 push를 한 번 시도한다.
등록 상한69run/46,800jointupdate;09:00/pressure/failure stop이 우선. 새 실험 무제한 확대 없음.
130 CPU tests 통과. Projection identity 신규 테스트는 처음에 grad-mode를 맞추지 않아 실패했고,
동일 no-grad 실행 조건으로 수정 후 bitwise 통과했다. Gradient 검사는 별도로 유지/통과했다.

### 2026-10-03 01:55KST — coverage/동일K 선택 비교 완료

27run/21,600update/1,940.08초/peakallocated1.568GiB/원본hash불변.
추가dev192window/182scene/24recording에서 원본ADE0.352210m,
MLPsmall .372616/MLPlearned .361504/ego .371554/aux-only .367154/
halfbridge .366605/uniform .370875/fixed .355226/random .351212/current .365208m.
Random-learned delta−.010292m/recordingCI[−.018142,−.003216],
random-original−.000998m/CI[−.010777,+.007704]. 다중비교보정/독립확증/PDMS검증 아님.
MLP 미래MSE1.9090<persistence2.2062이나 planning 개선과 동일하지 않다.
학습 선택의 장면내 patch간 거리는 작지만 전체dev에서266–305IDs를 써 전역collapse로 부르지 않는다.
독립CPU 감사에서 첫48run의 집계·batch순서·원본출력·aux경계 검사 통과.
Projection queue는01:52KST 후속 학습 시작. 다음12run은 이전 단계 완료 뒤 실행한다.

### 2026-10-03 — 학습 종료 후 선택 근사 진단 등록

69run/46,800update 상한은 그대로 두고 train32 recording·각1window에서 읽기전용 진단을 등록했다.
Coverage의 MLP/ego 최종800 checkpoint ×3seed, 각window32개의 seeded single-slot교체.
실제 planning loss차이와 dL/dW의 one-hot변위 근사를 비교한다. 선택 score 전체 Jacobian이나
실제 optimizer 효과와 동일하지 않으며, GT를 쓰는 반사실적 loss진단을 배포정책/oracle로 부르지 않는다.
학습은0, 미래GT input 없음, 마지막 queue 완료 뒤 GPU1단독으로만 실행,15분/6GiB cap.
동일조건 reference FP32비교 tol1e-5와sign검사loss noise floor1e-6을 실행 전에 고정했다.

### 2026-10-03 02:26KST — 등록학습69run 완료 / 선택진단 v1 guard

69run/46,800jointupdate, 시리즈wall 합5,027.30초, 최대allocated1.573GiB.
전체CPU evidence audit 통과: original hash/scene-macro/seed순서/pairedbatch/aux경계.
LowLR MLPdev .347694±.000392, ego .346338±.001650m; 원본 .352210.
이전LR 대비 개선은 있지만 원본대비 recording CI가0포함, 반복개발자료/다중비교를 유의한다.
최저mean을 선택학습 우월성/PDMS 개선으로 부르지 않는다. 동일LR fixed/random은 미실행.
공유 결과 자동commit313a3ab, push128(인증오류). 기존 checkpoint/result 보존.
후속읽기전용진단 v1은single-grad vsbatch33-no_grad의출력차이1.144409e-5>tol1e-5에서중단.
실패로그/source/confighash보존. v2는batch1/grad-enabled를일치시키며tol/표본/checkpoint는바꾸지않는다.
실제GPU검사는별도진행하며137CPUtests통과만으로대체하지않는다.

### 2026-10-03 02:35KST — 선택진단 완료 및 다음 통제 비교 등록

Matched-mode 실제모델6×32trainwindow×32교체=6144건 완료. 기준output차이전부0/tol불변.
128.75초/peakallocated1.189GiB/원본hash불변/optimizerupdate0.
MLPseed29/47/83 signagreement .917/.897/.864, ego .756/.804/.735.
LocalW미분과finite교체의정합이며fullscoreJacobian/실제업데이트/일반화/인과적중요도 보장은아니다.
공유 `overnight_selection_surrogate_diagnosis_v2_20261003.json`에개별교체6144개와checkpointhash보존.
다음별도실험 matched_conservative_selection_v1은6조건×3seed×800=18run/14400update다.
기존낮은LR학습형6run재사용, 같은LR고정/무작위(MLP/ego),MLPjointaux-off/currentfeature비교.
Jointaux-off도공통100updatefuturewarmup은유지하므로미래지식이전혀없는모델로부르지않는다.
실행전초기모듈hash·batchschedule같음확인, effectiveauxweight와rawgradient진단역할구분.
기존69run상한은그대로종료했고이번다음단계를별도등록했다. 총등록87run/61200update,현재완료69.
두시간/09:00/공유GPUreserve상한, 추가cache/target정의변경/공식benchmark/heldout 없음.

### 2026-10-03 오전 — 87run 완료 확인 및 실제 학습 모듈 시각화

후속18run도03:45완료, 자동결과commit8182f6c. Push는인증오류로실패했다.
사용자 요청은 selector/predictor가 학습됐는지 눈으로 확인할 시각화다. 추가학습은하지않았다.
낮은LR seed29 MLP/ego-query final800을 CPU에서 strict submodule복원했다.
기존공식encoder캐시를읽고 fullplanner/encoder/scorer는재실행하지않았다.
Dev192/24recording 전부GPU저장selected IDs일치, window MSE 최대차이는
ego4.768e-7/MLP9.537e-7(tol1e-5 사전설정). 출력finite/캐시checksum검사통과.
6개명령별hash표본의현재선택전후/softmax/미래실제영상·crop/1024-Dlatent값·MSE를그렸다.
Before는100auxwarmup후이고동일한final-selected ID에대해predictor를비교한다.
사진은생성prediction이아니라GT참조이며,고정camera-grid는객체track이아니다.
결측future는제외표본선정에사용하지않고오차공백/GT회색으로표시한다.
Ego predictor유효patch-time MSE2.379169→2.271626(current-copy2.396446),
MLP4.374582→2.876284(copy2.164686). Selector선택교체43.10%/26.56%.
이는학습진행관측이며최적선택/공식planning성능향상은검증하지않는다.
주최종render56.10초/CPU1thread/GPU0, 초기errorcurve렌더도보존했다.
144pytest통과/Ruff통과/HTML65파일로컬참조·ZIP CRC검사통과,ZIP30.03MB.
이미지·latent NPZ·전체JSON은Git에넣지않고소형summary만공유한다.
갤러리 `outputs/drive_jepa_selective_future/selector_predictor_visualization_20261003/index.html`.
ZIP `outputs/drive_jepa_selective_future/selector_predictor_visualization_20261003.zip`.
공유 `results/drive_jepa_selective_future/learned_module_visualization_20261003.json`.

재현(새 output-directory 필요):
```bash
PYTHONPATH=src OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 CUDA_VISIBLE_DEVICES='' \
  runtime/environments/future_prediction_cpu/bin/python \
  scripts/visualize_drive_jepa_learned_modules.py \
  --config configs/drive_jepa_selective_future/learned_modules_visualization_v1.json \
  --output-directory outputs/drive_jepa_selective_future/selector_predictor_visualization_<run_id>
```

## 2026-10-03 11:40 KST — Spatial region selection comparison registered

Latest user requested more/larger meaningful patches rather than visualization alone.
Base122e885; config `configs/drive_jepa_selective_future/spatial_region_selection_v1.json`.
5conditions×3pairedseeds×800jointupdates; same704cache and commonwarmup100.
K16native, K4pooled2×2, K8pooled random/planning/planner-retention.
Current planner always receives full original grid. Retention masks only a training readout,
uses current full-planner teacher, and gradients go only to selector. Futureaux goes only to predictor.
No new target-validity candidate filtering or semantic GT importance assertion.
151CPUtests pass,GPU0/1both48GBfree at11:39. Registered2hwhole/20mincondition/8GiBown/6GiBfree caps.
No performance result yet. Resume restores this runner optimizer/scheduler/RNG; old runs preserved.

## 2026-10-03 — Registered region training completed; user redirects to location diagnosis

15runs/12,000updates completed in2008.58seconds, peakallocated1.665GiB, GPU1only, noOOM, originalhashunchanged.
At user redirection, knownPID2624673hadalreadyexited; nootherprocesswasstopped. Noadditionalcount/sizeexperiment.
Dev ADE: native16 .346896; region4 .346354; region8 random .347460; region8 planning .347129;
region8 retention .346951m. SameK8 lasttwoCIcontains0. Retention moves74–89%ofsites but often towarddistantcenter,
not demonstrably toward all important nearby agents. Same6scene gallery exported without outcome selection.

Read-onlydiagnosis registered9dcf80c; 16hashchosen trainrecordings×2conditions×3seeds,768hardreplacements.
Score instrumentation and explicit reference exactlymatch productionoutput. Final planning-score gradients nonzero.
STscore signagreement81.6–93.7%islocalfiniteprobe, notglobaloptimality. Retentionbranch-offchanges5.6–7.0cm,
slotshuffle0.24–0.27mm despitefeatureRMS1.12–1.14. Weightedretentiongradient4.3–10.5×planning,
directioncosine-0.067..+0.021. Supportsweakspatialidentityusage and surrogateobjective mismatch, notdeadgradient.
63seconds/1.260GiB, optimizerupdates0, extension+officialhashunchanged. Noautoadditionaltraining.
Nextcandidate: fixedK/size, location-aligned future injection into current spatial memory; NOTimplementedyet.

Firstdiagnostic failed beforemeasurement becauseofficialimport changescwdandoutputwasrelative; preservedv1.
Absolute-pathv1bcompleted. Original source_commit fields accidentallyrefer toofficial548bb82 duecwd.
Keeprawrecordsunchanged; `provenance_clarification.json` identifies projecttraining8e8e49a/diagnostic9dcf80c+pathfix
and executedfilehashes. Futurecodeusescwd=WORKSPACEforgit. SharedJSON/code/docs inlocalcommits;
pushstillfailsVSCodecredential socket/anonymouswrite. CPU154tests+Ruffpass.

Reproduction commands (new output paths; do not rerun completed learning by default):
```bash
CUDA_VISIBLE_DEVICES=1 LD_LIBRARY_PATH=/rhome/junseong/PlanningAwareFuturePrediction/runtime/environments/drive_jepa_official_evaluation/lib \
  OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONPATH=src \
  /rhome/junseong/envs/kjs-drive-jepa-extension/bin/python -u scripts/diagnose_drive_jepa_location_learning.py \
  --config configs/drive_jepa_selective_future/location_learning_diagnosis_v1.json \
  --output-directory outputs/drive_jepa_selective_future/location_learning_diagnosis_<run_id>
```


## 2026-10-03 — SPARTAN/C-JEPA/IA-JEPA 착안 적용 등록 (Codex)

사용자 명시적 적용·성능 확인 요청. region_research_v1.json에 9조건×3seed×800update를 결과 확인 전 고정했다. 위치 bridge→희소 관계→관측 history mask→관측 motion 순서이며 current-only/random/unmasked 대조군 포함. 같은 704window/원본 frozen planner와 warmup/batch순서 재사용. 과거 관측 4frame 존재704/704, dev 5초 state192/192. 코드 관련 CPU40검사통과. 공식 논문 재현이 아닌 region 구조 변형이며 자세한 범위는 docs/drive_jepa_region_research.md. 학습은 등록 commit 뒤 GPU1 여유 재확인 후 실행, navtest/WA는 재개하지 않는다.


## 2026-10-03 — 세 논문 착안27run 및개발PDM완료 (Codex)

사용자요청에따라등록5b85a01/region_research_v1의9조건×3seed×800=21600update를모두완료했다.
GPU1단일process/전체56.39분/학습peak1.719GiB/CPU162통과. 원본 frozen parameter SHA-256 유지(validation.json 참조).
관측4frame704/704확인후earlieranchor/motioncache를workspace에만추가했고기존cache·모델·공용데이터보존.
공식PDM save_buffer의sandbox정체를10초probe timeout/밖에서SAVE_OK로확인했고우리PID3276645만종료후PID3289792재개. 빈미완성cache는이름바꿔보존했다.
Dev192window/182scene/24recording에대해새27+보존6+원본=34조건6528score완료. 기존6checkpoint추론은modulehash와저장ADE(1e-6)대조통과/optimizer0.
ADE/PDM: 원본0.352210/87.119134, 기존global learned0.347129/88.826031,
새denselearned0.349222/87.640696,sparselearned0.349238/87.742199,current-only0.349207/87.742499,
C-JEPA unmasked0.349226/87.742247,masked0.349230/87.742244,motion0.348680/87.691769.
세기법추가효과와learned-vs-random의대응95%CI모두0포함. 원본대비일부ADE감소는관측되나current-only도동일하고PDM개선CI는0포함한다.
Sparse연결밀도19.63%이나dense연산이므로FLOPs효율주장없음. 객체slot/causal discovery/독립test/동적K결과가아니다.
새방법채택·추가sweep없음. 현재특징전달과미래변화정보의planning기여분리만후속설계로권고하며미실행.
상세docs/drive_jepa_region_research.md; 공유results/drive_jepa_region_research_v1/(summary.json,comparison.csv,validation.json,planning_comparison.png,development_pdm_results.json).
실행train_drive_jepa_region_research.py; 평가evaluate_drive_jepa_region_research_pdm.py/evaluate_preserved_region_research_controls.py; 집계report_drive_jepa_region_research.py.

결과 commit `215cb6b`. `git push mine`은 sandbox DNS 실패 후 밖에서 재시도했으나 기존 VS Code Git 인증 소켓 ECONNREFUSED / No anonymous write access로 실패했다. 로컬 결과·코드는 모두 커밋됐고 작업 트리는 정리됐다. 원격 인증 복구 후 push만 남으며 GPU/학습/평가 작업은 없다.

## 2026-10-03 Encoder 자체 미래 표현 학습 승인 및 사전등록

사용자승인: encoder 입력/학습목적 변경 및 불필요한selector/predictor제거 가능. LoRA사용아님: 원본last2block+norm full update와FiLM.8조건×3seed×512update,같은512/192window;현재준비단계,CPU5검사통과.설계/근거는docs/encoder_future_learning.md.

사용자가last2충분성을질문해학습전last6대조4조건을추가했다. 총12조건×3seed. 첫sandbox실행은CUDA불가로모델로드전에종료했고sandbox밖GPU권한으로재시도한다.

CPU전체검사는Python3.9공식환경에서기존Python3.10문법때문에collection실패해코드를바꾸지않고기존CPU환경으로실행했다:167passed. 입력마스킹actualpixel교란불변/공식maskedencoder동일성통과. 준비GPU0 PID613492. 공식devPDM/공통미래delta ridge probe/paired분석runner추가.

704prefix준비완료:552.29초/13,671,549,056bytes/최대feature차이0/64train24devrecording.12조건GPUgradient계약모두통과(각blocknonzero,planning→head0),peak5.538GiB. GPU검사원본hash보존.36run학습/공식PDM·probe기동예정.

입력마스킹비교의targetschedule혼입(마스킹fixed2views vsunmasked매번sampling)을코드감사로확인했다. 개발성능미조회상태에서same-target unmasked2조건×3seed를별도등록. 기존36run보존/새6run추가/총42run. 기존일반마스킹대조는복합recipe효과로표기한다.

초기train8window의weighted auxiliary/plan encodergradnorm이1.2–2.6%임을확인해개발성능조회전last6uniform/plan-mask의auxweight0.5대조2조건×3seed추가등록. 기존0.05와다른설정불변. 추가통제설정additional_controls_v1.json에fixedtarget6+strength6을통합,총48run. 기존matched2조건등록은보존하며중복실행하지않는다.

본36run완료:GPU0worker1862.38초/GPU1worker2575.77초,원본hash보존.36모델raw replay최대성분절대차이1.1444e-5/validcommands0,1,2검증통과. 추가12통제run시작:GPU0 PID1283398/GPU1 PID1283399;공식dev평가대기1283637/raw대기1284511. 본evalPID1074518진행중.


## 2026-10-03 Encoder 미래 표현 학습 48회 완료

사용자 요청에 따라 encoder 자체를 planning에 필요한 미래 정보를 보존하도록 학습했다.
마지막 두 블록으로 충분한지 질문을 반영해 학습 전에 여섯 블록 대조를 등록했다.
LoRA 없이 원래 마지막 2/6개 ViT block과 norm을 직접 갱신하고 내부 intent FiLM을 적용했다.
기존 selector/future bridge를 추론에서 제거했고 미래 head는 학습 전용이다.

본 source ae5c2da의 36회와 추가 source e7d57b1의 12회 모두 완료했다.
16조건 × seed29/47/83 × 512update = 48회 / 24,576update. Train512/64recording, dev192/24recording.
추가는 동일 fixed target의 마스킹 대조 6회와 미래 loss 0.05→0.5 대조 6회다.
모두 코드 감사·train 초기 gradient를 근거로 개발 성능을 읽기 전에 등록했다.

원본 ADE/PDM 0.352209793m / 87.119134167%; planning-only 2블록 0.347628701 / 88.396320378,
6블록 0.346061389 / 88.718099738. 2블록 ADE 변화 −4.581091mm CI[−6.595189,−2.570006].
PDM 원본 대비 개선 구간은 0 포함. 6블록 대 2블록 우월성도 미확정이다.
6블록 intent planning 88.437487974, uniform future λ0.05 88.438068996,
λ0.5 88.438153327. 미래 감독·움직임/planner target·마스킹 추가 효과는 실질적으로 매우 작다.
일부 tiny PDM 차이 CI는 0을 포함하지 않으므로 모두 유의하지 않다고 단정하지 않았다.
강한 미래 loss probe MSE .150730104 vs intent planning .150741347로 미세한 차이만 관측했다.

48개 encoder raw 영상 replay(모델당 현재명령별 3개 window) 최대trajectory차이1.144409e-5,
future head 미생성 확인. Planner status를 고정한 encoder 명령 개입에서 intent모델42개 표현변화,
비조건부6개 불변. 표현의 조건부 변화가 의미적으로 올바른 미래·계획을 뜻하지는 않는다.
평균 PDM이 가장 높은 no-intent6block도 우회전명령−2.4648pp, 정지근처두정의모두악화,
귀책충돌없음100→99.826389. 상황별/구성점수/CI/동일용량probe를 보고서에 포함했다.

CPU167pytest 통과. 원본model hash보존/모든blockgradient/pairedschedule/마스킹입력계약통과.
Official PDM9600score(두단계original중복192;고유9408), 실패없음.
학습process합5711.109초/최대trainingallocated5.641023GiB. 두GPU0·1만사용하고모든작업종료.
추가PDM/probe완료뒤validation의input_mask_contract경로가추가root를참조해FileNotFoundError발생.
공용prepared_cache_directory를참조하도록고치고저장predictionhash대조후report만복구;
기존점수/학습/metric cache불변,재학습/재scoring없음. 원래오류로그보존.

통합 results/encoder_future_learning_v1/combined_summary.json, combined_comparison.csv,
combined_validation.json, paired_planning_comparison.svg/pdf.
한국어 docs/encoder_future_learning_results.md, 설계 docs/encoder_future_learning.md.
추가원자료 results/encoder_future_additional_controls_v1/;체크포인트outputs/각family/.
Config 설명의 all24 잔여문구를validation/report에정정했고과거confighash는바꾸지않았다.
작업문서저장은 pages:write-page 지침의기존저장소문서경로존중에따라로컬docs에수행했다.
전체사전학습/독립test/학습형동적target선택검증은아니다. 이후sweep미실행.
원격push는세션말재시도하고인증결과를별도로기록한다.


### Encoder 결과 커밋과 원격 공유 상태

결과 commit `955e82f`에 코드·48회 전체 표·검증·보고서·그림을 저장했다.
`git push mine`은 sandbox에서 DNS 실패, 제한 밖 재시도에서는 기존 VS Code Git 소켓
ECONNREFUSED와 No anonymous write access로 인증 실패했다.
학습·평가는 완료했으며 로컬 자료는 보존됐다. 인증 복구 후 push만 남았다.


## 2026-10-03 Encoder 실험 결과의 실제 장면 시각화

사용자가 PDMS 숫자만으로 감이 오지 않는다며 이미지 시각화를 요청했다.
`visualize_encoder_planning_results.py`를 추가해 저장된 48회 실험 중 핵심 조건을 CPU에서 렌더링했다.
세 이미지: 전체 ADE/개선악화분포/명령별 PDM, 실제 전방영상+BEV+시간별GT오차,
미래 감독의 추가 경로 이동량과 ADE변화. 한국어 NotoSansCJK를 사용하고 모두 육안 확인했다.

구간별3seed평균ADE에표시용±1mm경계를적용: 2block112/11/69, 6block113/4/75,
intent+strongfuture105/8/79(개선/작은변화/악화). 통계적 동등성 기준이 아니다.
4초예측끝점모델간거리중앙값: original→planning6 6.262623cm,
intentplanning6→strongfuture6 1.789567mm. 출력이동량이지GT오차개선량이아니다.
사진사례는seed29 planning6의ADE변화가최소/상위중앙/최대인3개로의도적선택:
dfaf7f0318b25029(-11.693442cm),7c72be317cca5e4a(-0.443103cm),fe38f82d16e35220(+5.612510cm).
사례대표성주장없음. 실제currenttoken사진/GTcachehash와재계산ADE일치검증,
참조sourcehash불변. Camera사진은실제관측;BEV는ego좌표계경로로closedloop재생아님.

로컬PNG outputs/encoder_future_learning_v1/visualization/,
공유PDF/JSON results/encoder_future_learning_v1/visualization/.
새학습·모델inference·PDM재계산·GPU사용0. 기존weights/실험수치/원본데이터수정없음.
결과보고서의실험결과이미지절과README/research_status/HANDOFF갱신.


## 2026-10-03 18:48 KST LPWM 객체 표현의 NAVSIM 적응 완료

**완료: LPWM의 NAVSIM 객체 표현 적응 실험.**
공식 main `4cf53c4`와 49쪽 논문을 조사하고 Sketchy checkpoint를 strict loading했다.
원영상·회전 보정 × 3 seed × 300 update, 90 train/30 development clip의 학습·평가를 완료했다.
원영상 적응의 복원 MSE는 0.05680→0.01640이지만, 객체 박스 대응률은 17.71→18.47%로 추가 개선 미확정이다.
과거만 사용하는 미래 MSE는 0.03015, 마지막 영상 유지 0.03032로 차이 CI가 0을 포함한다.
회전 보정은 시야 손실이 커 채택하지 않는다. Planning/PDMS 이득은 평가하지 않았다.
보고서 `docs/lpwm_navsim_adaptation_results.md`, 논문 검토 `docs/lpwm_paper_and_driving_assessment.md`.
실제 이미지·GIF `outputs/lpwm_navsim_adaptation_v1/visualization/`, 공유 PDF/JSON `results/lpwm_navsim_adaptation_v1/`.
좌표 검사 3개, 미래 입력 교란 검사 8개 모델 통과. 등록 작업 종료, 기존 Drive/WA/공용데이터 보존.

공식원본코드불변, horizon scalar7·KL contiguous adapter만외부에서적용. 109545263개전체parameter직접적응.
학습6회합계807.39초/peak10.145GiB/원본보존. 새전용venv에pycryptodome만추가.
첫2평가에마스크와복원LPIPS를추가저장하기위한재추론은기존metric차이0; 초기결과보존.
결과source/run config/환경/selection/checkpoint hash는 results/lpwm_navsim_adaptation_v1/에있다.


## 2026-10-03 LPWM planning 연결과 사용자 방향 수정

**최신 방향 수정: LPWM encoder·context·dynamics·planner 공동 학습.** 사용자가 원본 context/dynamics를 제외한 선택을 지적하고 공동 학습을 요청했다.
기존 encoder-only는 완료7run(frozen3/planning3/uniform seed29)만 보존하고 우리 worker3425264/3425265/scorer3425266을 SIGINT 종료했다.
`outputs/lpwm_planning_v1/superseded_by_joint_world_model.json`을 따른다. 이전18run queue/finalizer를 자동 재개하지 않는다.
새 코드 `src/planning_aware_future_prediction/object_centric/lpwm_joint_world_planner.py`: 공식 encoder6.035M/context39.389M/dynamics59.869M+planner0.821M, RGB decoder만 제외.
과거2영상→관측transition posterior→미래8step은policy prior만으로 autoregressive rollout, activation checkpointing으로 gradient 보존.
GPU0 batch1 3update에서 planning/future loss 각각 세모듈gradient>0, 미래label변경시예측동일을 통과했다. bf16 batch4 profile 진행/새 데이터·규모 준비 중이다.
기존512train 중13개/192dev 중2개가 공식navtrain token필터밖임을 발견했다(로그는전부navtrain). 새 학습은 공식token까지엄격필터한다.

초기 등록 commit e1086f1. Encoder-only1000update/seed29 planning공동학습239.75s,3.67GiB, 실제 가중치변화238state tensors를 확인했다.
User는 데이터/학습량 및 작은VRAM에 문제를 제기했고, 원본context/dynamics 공동학습을 요청했다. 이전실험은 완료대조군으로만보존하며 전체LPWM학습으로 해석하지 않는다.


## 2026-10-03 21:47 KST — Full NAVSIM LPWM training, batch/worker timing and process naming

**이번 세션의 확인 결과:**
- 공식 navtrain log AND token 필터, 기존40개 development recording 유지. 나머지122개 recording 모두 학습 사용.
- 이전 navtrain heldout recording도 이번 사용자 전체학습 승인으로 train에 포함됐으므로 독립평가로 부르지 않는다. navtest 학습 사용 없음.
- 공유 RGB cache152,495장/약7.0GiB. Stage1 train23,126/dev7,745; Stage2 train75,297/dev27,076.
- Stage2 ego 상태·미래 경로 생성 시 기존 공식 cached target과 max차이0.0.
- 공개 원본 Sketchy SHA6d62bf5a...부터 시작. 전체4모듈 gradient>0/weight change 검사 통과.
- batch2/accum4/worker0:1.729s; batch4/accum2/worker0:1.612s. worker2:1.627s,4:1.634s,8:1.612s,0재확인:1.624s. 각8update 중 초반2개 제외; 짧은 공유GPU 측정이며 전역최적 증거가 아니다.
- 본 학습 batch4 peakallocated29.10GiB, nvidia process약36.1GiB. 6개 처리량 측정에서 OOM0, worker증가 추가이득 미확인.
- planning gradient 독립 CPU audit: encoder4.015/context0.590/dynamics2.472/planner157.929, 미래GT교란 출력차이0, 1진단update 후 intent particle차이0.01828. 진단weight미저장/Stage2성능결과아님.
- 프로토콜/카메라기하/분산데이터 재개·RNG 검사8개 통과. Stage2 GPU경로는 gate 후 검증 예정.
- 공유 기록 `results/lpwm_navsim_full_posttraining_v2/`; 아직 전체 학습 완료나 planning 개선 결과 없음.


**2026-10-03 21:47 KST 최신: 전체 navtrain Stage1 본 학습이 GPU0·1에서 계속 실행 중이다.**
- Supervisor919150 / torchrun920080 / ranks920132,920133; 프로세스와 interpreter alias `kjs-lpwm-stage1`.
- 마지막 확인 update208 이상 / 총28,920, 20epoch; train23,126/122recording, dev7,745/40recording.
- batch4/GPU × accumulation2 × 2GPU =16; FP32; DataLoader worker0, torch CPU threads4/rank.
- GPU 각각 사용률100%, process VRAM36.1GiB/free11.3GiB, rank RSS 약3.21GiB씩 실측. GPU reserve6GiB.
- 최근 1.55s/update, 남은 순수학습12.35h; 중간진단 포함13~14h 추정. 최종적응평가/Stage2 시간 제외.
- 사용자는 더 이상의 속도 실험으로 중단하지 말고 빨리 본 학습을 계속하라고 지시했다. 필요 없는 GPU profile/worker 변경 금지.
- 최초 첫-update 이전 정체는 재기동으로 해소됐으나 정확 원인은 미확정. source amendment1~3에 변경 이력 보존.
- `outputs/lpwm_navsim_full_posttraining_v2/active_stage.json`, `stage1/progress.json`, `stage1_full_training.log`를 먼저 확인한다.
- particle gallery는 동일 장면 update0/128/512 및 epoch1/5/10/15/20. 현재 update128 저장 완료.
- 자동 체인: 전체Stage1 → 원본/적응본 dev7,745 평가 → 적응 gate → 통합Stage2 두조건 학습 → 전체 dev 및 navtest12,146 공식PDM.
- Stage2 GPU preflight는 gate 통과 후 실행. 아직 Stage2 학습·PDMS 결과 없음. 원래 18run encoder-only queue와 WA는 재개하지 않는다.


User requested no further interruptions for optional profiling. Current supervisor919150 and GPU ranks920132/920133 remain running during this handoff. Stage2 pipeline code is connected but GPU validation waits for stage1 adaptation gate; no Stage2 results yet.

Health check while training continued: {"update": 400, "training_loss": 25.616626262664795, "fixed_scene_means": {"initial_reconstruction_mse": 0.04461017088033259, "update128_reconstruction_mse": 0.01081353472545743, "initial_forecast_mse": 0.05366304004564881, "update128_forecast_mse": 0.035020887618884444, "last_frame_persistence_mse": 0.040329359006136656, "initial_presence_sum": 41.96882390975952, "update128_presence_sum": 45.00778150558472}}
All four core module gradients nonzero; GPU0/1 utilization100percent, free11.3GiB each. Do not equate early reconstruction improvement with object binding or planning improvement. No extra training interruption.

Commit dea246b saved full training implementation and health records. git push mine: sandbox DNS failed; escalated retry reached GitHub but VSCode credential socket was unavailable and remote rejected authentication. Background GPU training continues independently.


## 2026-10-03 22:49 KST — LPWM candidate metrics/refinement and validated queue

사용자 추가요청: E2E planner 논문 활용, 단일human회귀이외loss, SafeDrive식Stage1표현활용후보보정, 각stage검증과자동대기열, 상세설계보고.
Stage1 supervisor919150/ranks920132,920133은 그대로계속, 이기록시update2432/28920. GPU실험추가없음.
공식DrivoR fc6e5aa/DriveSuprim80fe792clone/논문, Hydra-MDP/Drive-JEPA 및기존SafeDrive읽기검토.
LPWM768particlememory→512train-onlycandidate scoring→32distinct shortlist→future-particlecrossattention boundedoffset→actualrefinedtrajectoryre-scoring 구현.
기본softimitation CE+6metricBCE+.02officialworldELBO, refine조건은WTA회귀+6live-metricBCE+.5prefixNC/DACBCE+.01acceleration/jerkproxy 추가.
지도/미래GT는teacher전용, particle depth/ID의3D객체가정없음. 기존SafeDrivefutureBEV는auxiliary임을명시.
전용source/client IPC로공식Python3.9 CPUoracle와LPWMPython3.12 GPU를분리, 후보좌표detach/원래anchor점수재사용금지.
CPU6test통과; CPU실제학습영상/공개weight1update gradient image23.0228/context5.0042/dyn15.0257/refiner>0, 미래교란0, intentparticle차이.02015. 검사용weight미저장.
후보oracledevADE.34544m/p95.72152m는모델성능아님. 512배치/단일공식scoreparity 최대2.14e-8. 전체teacher에서도segment별확인.
CPUprofile sandbox save_buffer정체를확인한CPU3PID만종료후host재실행. Stage1은중단없음.
queue1481082 / teacher1481089 CPU16worker실행, source registration고정. 현재전체102373scenes teacher준비와기존Stage1완료대기.
새queue는원래Stage1의train/eval완료를adopt하고originalgate+8causal/coverage/noncollapse gate를추가, teacheroraclecoverage/quality gate후Stage2.
metric_plus_world→검증→imitation_plus_world→검증→metric_refinement_plus_world→검증→locked fullnavtest12146. 각조건20epoch94140update, 동일stage1weight, lowLR1e-6/new3e-4.
world-off비교는최신refinement요청우선으로보류. Gate실패는diagnostics/차단, 강제넘김/무제한retry없음.
설계 docs/lpwm_planning_experiment.md, 설정 metric_distillation_v2.json, 공유 results/lpwm_metric_planning_v2/.
Stage2GPU/DDP및PDMS미실행. 본학습성능향상주장없음.

## 2026-10-03 23:00 KST — Stage1 목적함수·gradient·실패 원인 진단 설명

- 실제 공식 calc_dyn_elbo/encoder/context/DDP forward를 감사했다. 12프레임 posterior 복원, 11전이 teacher-forced particle/context KL, 첫프레임 prior/presence 규제. 총 loss=(.01/12)*(rec+.08static+.2dyn+.2context+.08presence).
- decode_with_ctx=False, detach_dyn_inputs=False; 복원과 context/dynamics gradient 경로를 구분. LPIPS frozen VGG는 생성영상 gradient를 유지. kl_balance=.01은 static appearance 가중치이며 gradient balance=.5와 다름.
- dev512 epoch1 loss23.3991/PSNR20.4611, 고정8장면 causal forecastMSE0.029535 vs persistence0.040329. 전체개발 검증 전이며 실제 적응/PDMS 성공 주장 없음. 공유 snapshot JSON에 source hash/기여항/모듈norm/scene별값 보존.
- 현재 queue 검증과 미등록 후속 frozen-LPWM/oracle-future/gradient분해 대조를 구분. 문서·README·HANDOFF만 보완했고 Stage1/Stage2 runtime source hash불변.
- 본학습 update2832/28920, queue waiting_for_stage1_full_training_and_validation; GPU학습 중단 없음.

### 2026-10-03 23:02 KST 원격 공유 시도

- e968b1a commit 완료. git push mine: sandbox DNS 오류 후 require_escalated host 재시도, 기존 VSCode Git credential socket ECONNREFUSED 및 GitHub Authentication failed(exit128). 원격 반영을 주장하지 않는다. GPU학습/queue 계속 유지.

## 2026-10-04 01:05 KST — 사용자 요청 Stage1 학습 상태 점검

- 7280/28920 update, dev512 epoch5 loss21.132790/PSNR20.834908dB. finite objective/gradient, module audit56회 전부양수. GPU0/1 87/100%,36.1GiB사용/11.3GiB여유, RAMavailable354GiB.
- 최근1.586s/update, 학습ETA10/04 10:40~11:30KST(전체최종검증제외). Checkpoint갱신/queue heartbeat/CPUteacher 20361scene진행확인. Source hash등록일치.
- 원시근거 `results/lpwm_navsim_full_posttraining_v2/health_check_20261004_0104.json`. 이전SIGINT traceback3개를현재오류로집계하지않음. 학습중단/재시작/설정변경없음. 전체적응/PDMS개선은미확정.

## 2026-10-04 11:17 KST — Stage1 종료 직전 재점검

- 28912/28920update, dev512 epoch19 loss19.692797/PSNR21.100133dB. GPU0/1 100%,VRAM36.1GiB/여유11.3GiB, RAMavailable362GiB. 225개modulegradient감사모두양수/finite.
- 전체ELBO/복원은좋아지지만dev dynamicsKL은epoch5 3840→epoch19 4210반등. 고정8장면causal 미래MSE epoch10 .023120→15 .023096으로정체. 실제원인/전체적응성공은최종검증전미확정.
- CPUteacher complete/coverage gate true,train75165/dev27034,coverage99.82%/99.84%,official parity max2.98e-8. Queue는원래Stage1전체검증대기라node status가아직running인것이며parent완료는오류아님.
- `results/lpwm_navsim_full_posttraining_v2/health_check_20261004_1117.json`에근거보존. runtime source등록hash일치/학습·queue변경없음. 이전pipeline_stopped 및SIGINT로그는현실행전이력.

### 2026-10-04 11:21 KST — 점검 도중 Stage1 학습 완료 및 자동 평가 전환

- 전체28920update/20epoch 완료, checkpoint SHA71478ee548376bec21a4a22bc219929955aa0bfdd876f704676ade169fb4831f. 네모듈parameter sample변화/gradient정상. Dev512 loss19.696784/PSNR21.076768,dynKL4221.361.
- 최종고정8장면미래MSE0.0210534로epoch15보다추가개선. 직전기록의epoch10→15정체가끝까지지속됐다고해석하지않는다. 전체개발gate는미완료.
- 공개/적응full evaluation자동기동로그확인. `results/lpwm_navsim_full_posttraining_v2/training_completion_check_20261004.json` 및각evaluation records/log를다음확인. b7a42d3 push기존인증실패,학습에는영향없음.

## 2026-10-04 11:42 KST — 현재 작업 및 완료예정시간 점검

- published: 3248/7745 (41.94%), 11:21이후2.381clip/s, ETA2026-10-04T12:13:30.870112+09:00.
- posttrained: 3208/7745 (41.42%), 11:21이후2.350clip/s, ETA2026-10-04T12:14:13.359031+09:00.
- Supervisor/queue/평가2process정상,등록sourcehash불변. Stage2는Stage1gate후GPUprofile전이며전체완료시각미측정. `results/lpwm_navsim_full_posttraining_v2/evaluation_progress_20261004_1142.json`. 학습/queue변경없음.

## 2026-10-04 11:46 KST — Particle 시각화 범례 설명

- draw_particles/capture_particle_snapshot 및공식decode_objects와실제straight PNG를대조.64중심점, learned-scale/presence상위16box,HSV색particle index,점면적7+16presence/alpha.25+.75presence확인.
- Box는지역glimpse배치범위이며object detector GT box/분산아님; 내부alpha/depth가실제합성을결정. Presence sum은객체수가아니고top16은시각화만. GIF는동일장면의checkpoint변화. 기존문서에범례추가/실행코드와출력불변.

## 2026-10-04 11:53 KST — 배경 중심 particle과 Stage2 planning 적응 가능성 설명

- 실제particle_attributes에position/scale/presence/features경로가detach없이연결, LPWMfull-low-LR/egoFiLM/worldELBO.02확인. 객체중심배치직접loss없고현재6개metrics에정지선준수독립항없음.
- RGB목적·화면면적·질감·patch기원·128해상도로배경표현을학습할가능성은있으나시각화만으로원인확정불가. Geometry이동과feature/future/활용변화는분리해검증필요.
- 현재queue는worldretention/futurepersistence검사까지만포함. Stage2semantic재배치/면적보정/particle개입/frozenLPWM대조는후속진단설계이며미구현. 문서갱신만수행/학습·평가·queue불변.

## 2026-10-04 12:06 KST — Planner loss·stop-gradient·선행연구 대조

- `lpwm_candidate_planner.py`/`lpwm_planning_finetuning.py`/trainer/config의 실제 실행 경로를 대조. Coarse512 soft CE+6metric BCE; refined32 WTA SmoothL1/circular heading+6metric BCE+0.5 temporal BCE+0.01 normalized acceleration/jerk penalty; 별도clip worldELBO0.02.
- LPWM/planner는 연속 역전파. 좌표→score embedding 및 CPU oracle에 detach, top-k/argmax/argmin index는 비미분. Gather된 coarse feature와 WTA pose는 gradient 유지. Refined score BCE가 고유 refiner로 직접 전달되지 않지만 공유 LPWM/coarse feature는 갱신하므로 완전 독립은 아님.
- DrivoR§3.4는 생성 latent를 채점기에 주지 않지만 현재코드는 `score_queries+candidate_features`; 원형 재현으로 보고하지 않는다. Hydra-MDP의 vocabulary/privileged metric distillation, DriveSuprim coarse/fine scoring, Drive-JEPA MTD 범위도 원문 재확인. Temporal safety는 보조감독이며 최종순위 직접항 없음.
- 기존연구문서/HANDOFF만 수정. 추가학습·GPU진단·runtime변경 없음. Queue `waiting_for_stage1_full_training_and_validation`, Stage2 본학습은 적응gate 후. 기존 CPU 연결검사는 합산 planning loss 수준이며 loss별 autograd 실측을 새로 했다고 주장하지 않음.

## 2026-10-04 12:29 KST — 충돌회피 loss 의미 설명 및 Stage1 gate 실패 확인

- 사용자 BCE/SG 질문에 따라 unsafe정답0을 정확히 예측하는 BCE와 안전좌표를 생성하는 목적을 구분. Detach만 제거하면 안전확률을 낮추는 gradient가 좌표로 전달될 수 있어 해결책이 아님. 기하학적미분가능clearance/Drive-JEPA식 안전pseudo경로회귀를 보완설계로 설명했으며 runtime 미구현/실행변경 없음.
- 확인 중 전체7745clip 평가 종료 및 queue_failed.json 발견. 유일한 원래gate 실패는 object_box_recall_noninferiority: top16 recall@IoU0.1 0.3219947→0.2893251, paired CI[-0.0415200,-0.0236445], 등록하한기준-0.02. 추가causal/noncollapse/coverage는전부통과. Stage2 진입차단 유지.
- 자동생성 `results/lpwm_navsim_full_posttraining_v2/summary.json`을 그대로보존하고 README/AGENTS/HANDOFF현재상태갱신. 학습/queue/source/threshold 변경없음. 다음은box대응proxy실패원인진단. 이번 설명을 직접안전loss구현이나실험완료로해석하지않음.

## 2026-10-04 12:42 KST — E2E 선행연구 안전 loss·refinement 실제 구현 비교

- 공식 VAD1688c4b/UniAD532fc33/SafeDriveea7791d/DiffusionDrive9b52ed0 소스와설정 hash보존. 기존DrivoRfc6e5aa/DriveSuprim80fe792 및Hydra-MDP/Drive-JEPA 논문대조. 재현근거 `results/e2e_planner_safety_audit_20261004/`.
- SafeDrive Phase3 TwDAC reference detach=True이나 motion/plan query는FRNet에연결돼BCE가SWNetdecoder를학습. 우리refined_score_decoder는coarsefeature를읽으므로고유refinementdecoder가안전BCE를직접받지않는차이확인. Safety score calibration/직접collision좌표loss를구별.
- 공식kernel AST에서등록/reduction decorator만제거한 CPU검사: VADcollision1.3→1.26875/grad.790569, boundary.363616→.302604/grad1.104161. UniADloss2.85는requires_grad=False이나수치x미분2.00009; tensor재생성에의한공개함수gradient단절. Full planner재현/논문전체오류주장아님.
- 기존문서에8방법비교/직접감독·안전pseudoGT·공유latent·추론후처리의차이와추천ablation기록. 신규학습/GPU/PDMS/runtime변경없음. Stage1gate실패에따른Stage2차단유지.

## 2026-10-04 13:00 KST — LPWM과 planner를 연결하는 연구 문제·방법론 제안

- 사용자 요청에 따라 downstream planner의 목적을 재검토. 현재 구현은 후보512 metric scorer+선택적refiner이며 하나의 연구질문을 식별하는 통제실험이 부족하므로, 같은 후보/관측/예산에서 미래 정보가 선택 regret를 줄이는가를 중심질문으로 제안.
- DriveSuprim/DrivoR/DiffusionDriveV2/VAD 원문과 가장 가까운 SafeDrive/WorldDrive/ResWorld/World4Drive/EgoFSD/ForeDrive를 확인. WorldDrive의 미래latent preference ranking, EgoFSD 객체선택, CAPO의 control변화 기반 prediction weighting과 중복을 기록. 논문 요약을 성능 재현으로 해석하지 않음.
- `docs/lpwm_planning_experiment.md` 첫 절에 선택형 기준planner, fixed candidate oracle와 ranking regret, current/persistence/future, frozen/joint, encoder intent, utility weighted causal future 감독, 이후 동일 실제예산 선택 비교를 제안. Particle-GT association/metric geometry/GT target SG/주변차량 반응 counterfactual의 경계 명시.
- CAPO 착안 utility는 객체×시간 미래를 persistence로 교체했을 때 선택한 후보를 원래 관측미래로 재채점한 손실. Offline proxy부터 검증하며 신규 novelty나 실제학습으로 주장하지 않음. Stage1 box gate실패 진단이 실행의 선행조건, 등록runtime/config/queue/결과 모두불변. 문서만 수정/구문·공백·diff 범위 검토.
