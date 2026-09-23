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
