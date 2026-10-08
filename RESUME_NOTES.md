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

## 2026-10-04 13:20 KST — Stage1 완료 검증결과 재확인

- 사용자 요청으로 원본 summary/adaptation gate/transition gate와 공개·적응 metrics를 직접 검토. 각7,745고유token/누락·추가·중복0,평가metadata hash가training과일치,공유summary와gate내용일치/transition의원본SHA일치. `validation_review_20261004.json`에검토결과보존. 최종gate들은12:17KST생성.
- 원래16항목중15통과/현재top16boxIoU.1 recall비열등성1실패.6,926유효clip/40recording paired차이−3.267pp,CI[−4.152,−2.364]pp,등록하한≥−2pp미충족. 추가8장면causal/비붕괴/coverage통과.Stage2미시작유지.
- 미래MSE .020829 vs유지.036617(43.12%감소),LPIPS.382994 vs.399487(4.13%감소),객체ROI .029812 vs.042921(30.54%감소). IoU.3와4초미래box는개선되어전체객체표현악화로단정불가. 현pointcoverage/presence합감소원인미확정.
- 등록직진126개LPIPS악화CI양수,회전1289개개선,겹침5958개우월성CI0포함. Overlap우선scenario정의/위험계층10%비열등성제한명시. 저장horizon평균은0.5초LPIPS악화/2·4초개선이며새시간별CI없음.
- 기존문서에표/해석추가,HANDOFF갱신. 학습·추론재실행/임계값완화/소스변경없음. 원본결과/모델보존. 이전설계commit aa44380의원격push는기존VSCode credential socket/GitHub인증오류로실패했다.

## 2026-10-04 13:30 KST — 객체 검증 명칭·타당성 정정

- 분해요청후원시metrics/manifest/riskmetadata/시각화NPZ구조확인. 전체64particle저장은고정8시각화장면뿐이고나머지7745평가에는집계치만있음. GPU0·1조회시각여유각약28GiB/타인사용20GiB확인;추론프로세스기동없음. 이후사용자가LPWM에detection/segmentation없는데객체검증이무엇인지질문하여정의·타당성검토를수행.
- 공식modules.py:5332–5397은RGBAglimpse와alpha/presence/depth합성;semantic/GTinstance head아님. Stage1객체GT입력/loss없음. 평가용NAVSIMvehicle/pedestrian/bicycle GT투영box와상위16particle의position/scale사각형을Hungarian일대일매칭한IoU≥.1 recall임.
- ‘객체표현검증실패’는지나치게넓은해석으로정정. Top16/presence/객체부분표현/복원box와instance경계차이때문에proxy감소가정보손실을입증하지않음. 필수Stage1gate로쓰는타당성미확립인정,원래실패수치/threshold/source보존. README/기존연구문서/HANDOFF에정정추가;gate자동완화나Stage2기동없음.
- 원인분해결과·featureprobe·PDMS는미실행. 0726071 원격push도기존VSCode credential socket ECONNREFUSED/GitHub인증오류로실패했다.

## 2026-10-04 13:47 KST — 객체 구분·정보 보존의 검증 방법 제안

- 사용자 질문에 따라 LPWM 원문과 Dittadi ICML2022/공식 object-centric-library를 확인했다. Semantic detector가 없더라도 비지도 객체 분해는 검증 대상이며, native particle ID의 영속적 tracking이나 객체당 하나의 particle을 보장하지 않는다는 점을 명시했다.
- 기존 `docs/lpwm_planning_experiment.md` 첫 절에 5개 검사 제안: frozen linear/ridge 상태 판독, full64 실제 alpha 기반 instance 분리, temporal correspondence, causal future 상태/객체 영역 예측, 학습된 planner의 객체 정보 개입. 현재 공개/적응 모델의 동일 예산 probe를 최우선으로 권고했다.
- GT-localized probe는 자동 발견과 다름. Geometry/background/GT-ROI 대조, 미대응 객체 coverage, parts 분해와 merge 분리, GT oracle union의 한계, 미래 GT association/ego pose 누출 방지, zero-out의 OOD 혼란을 기록했다. Recording 단위 분할·paired CI, probe seed와 encoder seed 불확실성도 구분했다.
- 현재 cache는 projected box/category/track 주석이며 검수된 pixel instance mask는 확인되지 않았다. 전체 feature/alpha 신규 추출과 약200프레임 mask audit는 제안으로 남겼다. 기존 full-dev 저장 metrics만으로 probe를 이미 실행했다고 보고하지 않는다.
- 문서/HANDOFF/RESUME만 변경. 새 학습·추론·GPU 실행·gate 변경·Stage2 기동 없음. 원본 결과/실패 로그 보존. 직전 6e20108의 원격 공유는 기존 credential socket/GitHub 인증 오류로 실패한 상태다.

## 2026-10-04 14:20 KST — GT 감독 위치 판단과 승인된 객체 판독 검증 기동

- 사용자 요청은 GT box를 Stage1 입력/loss에 사용하는 안 vs Stage2 공동학습의 적합성 판단이며, 앞서 제안한 검증 방법 적용에도 동의했다. VAD§3.4의 class/attribute/motion 감독+planning loss를 원문 확인. GT는train-only감독으로 활용하고, 현재정보가 유지되면Stage2 planning+world+object/future 보조loss, 부족하면Stage1추가적응을 우선하는 설계를 기존 연구 문서에 기록했다. 학습조건 변경·Stage2기동은 아직 없음.
- `object_readout_validation_v1.json`, `object_readout_diagnostics.py`, `validate_lpwm_object_readouts.py`, `launch_lpwm_object_validation.py`를 추가했다. 공유 원본을 읽어 관측4장 GT track/ROI, 현재ego 위치·속도/camera depth를 준비. 총30,871clip/객체관측312,611건, 준비342초. Train222,646(차량137,526/보행자83,806/bicycle1,314), dev89,965(55,924/33,682/359). 98fit/24validation/40dev recording, 동일train/dev split 유지.
- 7입력대조의 선형ridge class/state 판독: ROIgeometry/particlegeometry/appearance/combined/background/combined+background/shuffle. 정규화강도train내선택 후전체train재적합. 분류점추정과상태오차recording bootstrap2000. 주석/미래GT는encoder입력에없고observed GT는probeassociation만 사용. 자동detection/tracking이 아니며단독hardgate로사용하지않음.
- CPU5검사통과. 초기pytest는PYTHONPATH 누락으로collection실패했고명시한후정상. GPU0 4clip실행검사2.95초/peak2.69GiB/LPWMupdate0; 정확도검사는아님. nohupqueue504595, 공개GPU0worker504614/적응GPU1worker504615 기동. 14:20적응4356/30871/223초;free22.34GiB/peak2.69GiB. `kjs-lpwm-object-validation` 실행별칭추가,원환경이동없음. 추출→CPU판독→sharedsummary 자동연결.
- Source/config/checkpoint hash는registration.json에보존. 추출도중공식decoder가64중variance기준30개선택임을추가확인. 이번판독의filter_key=None은full64재합성에의한pooling이며native30복원기여와다름을probe결과확인전 `decoding_scope_amendment.json`으로명시했다. Runtime변경은없으며nativeinstance검사는별도로30선택ID를고려해야함.
- 원본Stage1/Stage2 source·gate·checkpoint·실패로그보존. 새학습loss미구현,Stage2미시작. 수동mask검수/causal future판독/학습된planner개입은후속. 직전0a8fd30 원격push는기존VSCode credential socket/GitHub인증오류로실패했다.

## 2026-10-04 14:33 KST — Stage1 SSL 유지, GT 감독은 Stage2로 한정

- 사용자는 GT 감독으로 particle이 주석 객체에 편중되고 taxonomy/개수/라벨 품질에 의존할 수 있음을 지적하며 Stage1 SSL 유지+Stage2 GT 결합을 제안했다. 해당 방향을 채택하고 직전 Stage1 GT 추가 적응 권고를 현재 경로에서 철회했다. 이전 실험 수치/로그는 보존하고 설계 이력에 superseded 표시했다.
- GT fine-tuning이 SSL의 이득을 반드시 모두 없애지는 않지만, Stage2에서도 의존성/forgetting은 남는다고 명시. GT 중심/일대일particle/개수/presence 강제 없음, 미주석=unknown, 전체 영상 SSL loss 유지, state/future 보조 head와planning gradient, GT target/association SG를 권고했다. 직접boxloss가 없어도 learned crop 경로로 위치가 바뀔 수 있음.
- 동일SSLcheckpoint의 Stage2 planning+world vs +GTauxiliary, label subset/특정auxiliary종류제외 대조를 설계 제안으로 기록. 라벨제외 객체도 RGB/planning에 노출될 수 있어 unseen-category 일반화로 과장하지 않음. 라벨량 sweep를 신규 실행하지 않았다.
- Frozen readout 확인: 공개18,692/적응19,076 of30,871, 각각약973/977초, queue_failed없음/LPWMupdate0. 기존진단은GT로평가head만학습하므로최신원칙과일치. Source/config/학습/gate/queue/Stage2기동변경없음. 문서4개만갱신. dffe9f5 원격push는기존Git인증오류로실패했다.

## 2026-10-04 14:41 KST — 검증 절차 요약과 실제 객체 주석 확인

- 사용자 질문에 따라 현재queue가관측4장/GT-localized현재상태의frozenreadout임을정리. 종류·ego위치/속도·camera depth,train98fit/24validation/dev40recording,7대조이며미래판독/nativeinstance/PDMS개입은후속. LPWMupdate0.
- 원본sample log의gt_names에서vehicle/pedestrian/bicycle외traffic_cone/barrier/czone_sign/generic_object확인. anns는3Dbbox/class/3Dvelocity/instance/track,BoundingBoxIndex와xyz/length/width/height/heading순서대조. 검증2D ROI는3D투영이며pixelmaskGT가아님. sample클래스빈도를전체분포로일반화하지않음.
- 공개29,828/적응30,340 of30,871clip,queue_failed없음. `status_and_annotation_scope_20261004.json`에시각별상태/주석범위보존. Runtime/config/source/queue/가중치변경없음. 기존문서/HANDOFF/RESUME만설명갱신. e501352 push는기존Git인증실패.

### 같은 턴 후속 — frozen current-state readout 자동 완료 확인

- Queue complete/공개·적응각30,871clip/312,611객체관측/LPWMupdate0,추출1,502.9/1,490.0초. CPU판독과집계도완료했고shared summary와로컬summary동일. `completion_review_20261004.json`에시각별근거보존. Stage2미시작/완료queue재실행없음.
- Appearance macro-F1 .32018→.38669,combined .38453→.41766,GT-ROI-only .55473. 분류CI없는점추정이며GT위치조건부검사. Combined state오차는y+0.044m(CI[+.006,+.085]),vx−.053m/s(CI[-.078,-.024]);x/depth/vy는CI0포함. 전체객체이해/미래/PDMS개선으로일반화하지않음.
- 자동생성summary보존,문서/README/HANDOFF완료상태반영. 현재probe의full64재합성과native30alpha차이amendment유지. 학습/runtime/gate변경없음.

## 2026-10-04 14:55 KST — 완료된 객체 판독 결과 상세 보고

- 사용자의 결과 요청에 따라 저장된 summary를7대조/전체클래스confusion/상태CI/크기·거리·scenario별로 분석했다. 새 추출·판독학습·GPU·gate변경 없음. Appearance macro-F1 32.02→38.67/combined38.45→41.77/GT-ROI-only55.47(×100);분류CI없는점추정이다.
- Combined 클래스 F1 차량67.13→73.26/보행자47.08→50.72/자전거1.15→1.32. 자전거정답133/359이나예측19,807로오탐다수. Class-balanced ridge·불균형·용량과encoder원인을분리해야하며정보전무로단정하지않는다. 단일클래스strata는타클래스오탐을제거하므로전체confusion의클래스F1과다름.
- Combined 위치x11.094→10.914m(CI0포함),y4.097→4.141m(차이CI양수),vx1.751→1.698m/s(CI음수),vy/depth개선불확정. Appearance-only 위치/vx는개선CI가0제외하나절대오차크다. GT-localized linear정보검사이고자동검출성능이아님.
- 작은box33.64→36.18/40m이상32.97→34.77 F1. 직진313/회전1783/겹침87333객체관측이며겹침우선scenario분류는전체직진·회전을대표하지않음. 회전분류상승에도위치x12.321→12.858/y5.080→5.364m악화. 그룹별CI없음/투영겹침은가림GT아님.
- 연구문서에평가조건/수치표/제한/판단추가,HANDOFF1–5갱신. Stage1 SSL유지/GT보조감독Stage2원칙유지,causal future/native mask/planner개입·PDMS미완료. 원본결과·모델보존. bb3a6be원격push는기존인증오류로실패한상태다.

## 2026-10-04 17:59 KST — Stage2 본학습 시작, 객체 GT 감독 on/off 비교

- 사용자 GPU0·1 Stage2시작승인. 후속정정: 객체정답사용은아직확정아니며없음/있음어블레이션후결정. 46GB제한은VRAM기준이며CPU RAM아님을확인했다. `object_future_joint_v3.json`, 조건 `metric_plus_world`→검증→`metric_object_future_plus_world`→검증·paired비교로등록. 기존3조건v2queue는실패상태보존/재개하지않음.
- Top16 box기하proxy의hardgate타당성정정/나머지15기준과causal/noncollapse통과/readout부분개선/사용자승인을근거로새stage1_admission_amendment를작성. 기존failure값/기준/hash불변. 새admission코드는수정허용proxy가정확히한항목인지,나머지검증·evidence/checkpoint hash가일치하는지확인. 완전한적응·미래객체이해를입증했다고하지않음.
- ParticleObjectStateHead와loss전용detached Gaussian ROI다대다association추가. Current/future normalizedstate SmoothL1가중치.2/.2,종류CE.02;GT center/개수/presence강제없고미주석unknown. GT/futurepose는loss만,현재영상4장·ego/command→causal8step만planner입력. 직접객체aux없음조건도공통privileged PDMteacher는사용하므로label-free로부르지않음.
- TargetCPU4worker544.8초,102,373planning record/1,216,641현재객체관측,최대149. 기존readout3,223객체비교현재box차이0/상태최대3.82e-6. CPU14검사통과. GPU미래objectloss단독gradient encoder.2705/context.0666/dynamics1.1311/planner·aux.4065/decoder0;GTfuture교란trajectory/logit차이0,명령별particle변화>0. 진단가중치저장/본학습사용없음.
- Batch4 profile은공유GPU를보호한20GiB allocatorcap에걸려OOM예외(당시GPUfree6.91GiB),실패checkpoint/log보존. Batch2누적4로변경,유효16유지.5update두검사완료,peak11.88GiB/steady6.9–7.6초,입력준비0.03–0.11초.14검사/구문/diff확인후등록source14개고정.
- Queue1131167기동,첫조건본학습update16및48/94,140확인.75,297train/20epoch,LPWM1e-6/planner3e-4/SSL.02,111.76M전체학습,GPU0·1 DDP,kjs-lpwm-stage2.256update복구/epochcheckpoint/전중후시각화. 조건당초기예상7.5–8.5일+검증;고정최종epoch사용.
- GPU마다전체46,000,000,000byte상한감시(타사용자포함),우리reserved20GiB/allocated19.5GiB/최소6GiB여유. 실제각35.9GB,CPUprocess-tree RSS8.7GB(공유mapping중복포함). CPU46GB제한은사용자정정후제거. 메모리문제시우리child만checkpoint정지,타인process불변.
- 각조건27,076dev초기/학습후/persistent미래+공식후보PDM,7,745clip world유지평가자동연결. Runtime/누출/불완전학습실패는중단;성능가설실패는보존하고나머지어블레이션진행. 노출navtest를독립test로자동사용하지않음. 실제미래객체독립probe/native mask/객체planning개입은후속. active_pipeline을v3queue로갱신하여이전launcher도중복기동없이현재queue에join.
- README/AGENTS/연구문서/HANDOFF및공유launch_verification·audits·targetmetadata갱신. 직전2131e24원격push는기존VSCode credential socket/GitHub인증오류로실패한상태다.

## 2026-10-04 21:44 KST — Stage2 중간점검과 고정 개발 장면 진단

- 첫 조건 `metric_plus_world` 본학습은 1,792/94,140 update(0.3807 epoch)까지 진행. Queue1131167/torchrun1135635 유지, failure/stopped 없음, 등록 source14개/config hash 불변. 본학습을 중단하거나 설정을 변경하지 않았다.
- 초기16–256과 최근1504–1744 update의 각16개 로그 평균: total loss9.6775→7.0208, imitation5.9555→4.5907, metric BCE3.3568→2.0600. Raw SSL18.2615→18.5083은 비슷한 규모이나 world 성능 유지 검증을 대신하지 않는다. Loss/측정 gradient 유한, update 단조 증가 확인.
- 저장 update1536 checkpoint를 hardlink로 고정해 동일 seed 초기 모델과 CPU 비교했다. Optimizer746개 state 모두 step1536, 모든 모듈의 parameter tensor 변경 확인. 상대 L2 변화 encoder0.0509%/context0.0644%/dynamics0.0677%/decoder0.0714%/planner·command16.80%. Weight decay 효과도 포함하므로 모든 scalar의 task gradient를 입증하는 검사는 아니다.
- 독립 진단 `scripts/check_lpwm_stage2_training_health.py` 추가. 예측 전 기존 epoch-monitor 순서의 첫128개 development 장면/31recording을 고정하고 초기 무작위 planner와 update1536을 비교. ADE8.4098→1.5844m, FDE17.6252→3.7796m, 기존 공식 후보 PDM cache의 선택 점수1.9397→78.5124/100. Paired recording bootstrap2000회 PDMS 차이 CI[+70.17,+81.40]점. 강한 baseline 대비 이득/LPWM 미세조정의 별도 이득/전체 개발셋 또는 독립 test 결과가 아니다.
- 진단은 GPU0 batch1/allocator4GiB/전체44GB 중단선/10분 제한으로 287.9초 완료. Peak allocated0.661GiB, 관측 GPU0 전체 최대36.489GB. 종료 후 본학습 GPU0/1 전체35.266/35.219GB로 사용자46decimalGB 제한 이내. 본학습 process-tree RSS18.66GB는 공유 mapping 중복을 포함하며 CPU46GB 제한은 없다.
- 프로토콜/checkpoint hash/예측/weight audit/로그 검토/최신 상태 JSON과 학습 곡선 PNG를 `results/lpwm_object_future_planning_v3/health_check_20261004_2132/`에 보존. PNG 직접 확인, 문서/HANDOFF 갱신. 진단 결과로 checkpoint/epoch/config를 선택하지 않았다.
- 약7.75초/update 기준 첫 epoch512장면 ADE/FDE 모니터는 10월5일04시 전후, 첫 조건 본학습 종료는 약8.3일 후 예상(향후 검증/공유 부하 제외). 객체 GT auxiliary on 조건은 첫 조건 학습·검증 후 자동 실행하며 채택 여부 미정. 전체 world retention/전체 개발 PDMS는 아직 미완료다. 직전1eb9471 push는 기존 Git 인증 오류로 실패한 상태다.

## 2026-10-04 21:57 KST — 현재 Stage2 학습의 상세 실행 명세 보고

- 사용자 요청으로 config/forward/loss/optimizer/teacher/검증 queue와 공식 LPWM 구현을 읽어 대조. 첫 조건1,888/94,140 update(0.4011epoch), queue1131167 계속, source14개/config hash 일치, failure/stopped 없음. 새 GPU 실행·학습·추론·runtime 수정 없음.
- 연구 문서 첫 절에 관측4장/ego8D/명령 FiLM→full64 particle×12시점×14D→768개256D memory→512고정 train medoid 후보의2층Transformer scorer 구조를 기록했다. Context는 미래 particle에 영향을 주며 planner14D에는 직접 포함되지 않는다. 예측 미래는512후보 공통이고 후보 action별 rollout/상황별 particle·horizon 선택은 미구현이다.
- 실제 loss는 soft XY-L1-distance candidate CE+6공식subscore BCE+0.02공식world ELBO. SSL은 별도로 뽑은12장 posterior 복원+11전이 KL이며 past-only RGB rollout loss가 아니다. MSE+0.1LPIPS, 내부0.01/12와beta(.08,.2,.08), KL balance 기본.5/feature kl_balance.01 차이, 주요 no-detach 경로와 부분 SG를 문서화했다.
- LPWM109.55M 저LR1e-6/새planner·command LR3e-4, 총111.76M 전체 학습. Planning16장면과SSL8clip/update, AdamW/warmup941/cosine/clip5, DDP batch2×accum4×2GPU, worker0/RGB mmap/FP32 encoder·world+BF16 planning dynamics를 확인했다. RGBdecoder는SSL만 받고 planner loss 단독gradient는0인 기존 감사를 인용했다.
- 두 번째 조건은 같은 Stage1/같은초기planner에서 새로 시작하며 current/future state와class 보조 항을 추가한다. GT/다대다Gaussian ROI association SG, 위치/개수/presence 강제 없음, 현재 투영3종의track만future유효시점 감독. 현재 두조건 모두refiner없음. GT감독채택 미정 유지.
- 매epoch512dev/epoch0·1·5·10·15·20시각화/조건후전체27076dev(공식PDM유효27034)/7745world유지/미래persistence대조/두조건paired비교를 정리했다. 조건당1seed/고정최종epoch,10%world유지기준,과거노출navtest와미완료독립probe 한계 명시.
- 상태근거 `results/lpwm_object_future_planning_v3/training_execution_review_20261004_2200.json` 저장. Config자유서술의first-condition 객체GT 오기와구refiner/stop문구는actual condition/분기와구분해문서에주석했고,등록config는불변이다. GPU각35.266/35.219GB<46GB,CPU RSS18.83GB(공유중복포함). 직전75687c2 push는기존인증오류로실패한상태다.

## 2026-10-04 22:23 KST — 사용자 요청 Stage2 일시중단 및 재개 상태 보존

- 사용자가 학습 시간 때문에 잠시 중단하고 나중에 재개할 수 있는지 요청했다. 기존 queue의 pause 경로를 확인하고22:22:36 `pause.requested` 작성. Queue는22:22:52 이를 확인해 후속작업을 차단하고우리torchrun/worker에SIGINT를 전달했다. 직접타인process를제어하지않았다.
- Worker는진행중update를끝내고22:22:56 **2,095/94,140update(0.4451epoch)** signal checkpoint를atomic저장했다. CPU로드에서model1129stateentry/optimizer746state모두step2095,reason=signal,source14개/config hash일치확인. 경과학습시간16,275.53초. latest.pt1,342,277,703bytes,SHA aef7ab37bfdfcf3a7c032774b0aaf4f4bbd76e8f84849a24a5b5fac2664b9bac.
- `user_pause_20261004/checkpoint_update002095.pt` hardlink보존,이전queue/progress/registration와요청/queue_failed/검증을별도저장. 공유결과 `results/lpwm_object_future_planning_v3/user_pause_20261004.json`,현재상태 `outputs/lpwm_object_future_planning_v3/user_pause_status.json`. 마지막로그2080보다checkpoint가최신이다.
- 호스트ps로queue1131167/torchrun1135635/worker1135737·1135738모두없음확인,nvidia-smi GPU0·1은기존다른사용자process만남음. 우리GPU메모리반환확인. Queue의 `RuntimeError('Queue paused by marker')` 원문은보존하며수치학습실패로해석하지않음. Training stopped/완료sentinel없음.
- README/AGENTS/HANDOFF/연구문서에사용자pause우선/자동재개금지및재개절차명시. 명시적요청후pause/failure를이력보존하고같은queue를실행하면본학습 --resume으로2096부터계속. LR/데이터순서/난수는update/config로재구성. GPU재개시험이나학습방법변경은없음. 기존ETA는pause로무효. 직전92509d9push는기존Git인증오류로실패했다.


## 2026-10-04 22:54–22:59 KST — 사용자 승인 Stage2 부분 미세조정 시작

- 사용자 요청: LoRA 또는 일부 계층만 학습해 빠른 효과/경향을 먼저 확인. 별도 `partial_output_layers_v1.json`을 등록하고 전체navtrain75,297개/조건당1epoch4,707update를 실행했다. 기존20epoch fullrun2,095update checkpoint SHA aef7ab37bfdfcf3a7c032774b0aaf4f4bbd76e8f84849a24a5b5fac2664b9bac와14개source/config/pause를 그대로 보존했다.
- Native LPWM출력모듈12개만선택학습, Transformerblocks/RGBdecoder등고정. LPWM5,558,389/109,545,263개, planner·command2,211,975개 전체학습, 총7,770,364/111,757,238개(6.95%). LPWM1e-5/planner3e-4. 명령FiLM을attribute CNN conv_in에서conv_out뒤로이동해고정prefix backward를줄였다. LoRA없음/원본전체구조와동일budget비교아님.
- 처음depth_head=None으로초기화실패(0optimizerupdate) 후실제존재하는계층만선택하도록수정. Batch2누적4 4update통과/peak7.04GiB/정상6.36–7.07초. Batch4future checkpoint해제는1update후21.5GiB allocatedguard초과로정지·기록보존. Batch4누적2/checkpoint유지는5update통과/peak12.0596GiB/마지막3회평균3.7553초로선택. Profile가중치는본학습초기화에쓰지않는다.
- CPU개발panel격리·recording균형·순서불변·GT teacher결측paired처리2검사통과. Profilecheckpoint4planning/2world평가실행완료, engineering_only로분리. 이소수점수를성능으로보고하지않는다. 첫조건GPU감사에서planninggradient가encoder/context/dynamics/planner로전달, frozen모든weight SHA불변/grad없음/optimizer중복없음. 미래GT교란prediction/logit0, commandparticle변화6.98e-5 확인.
- Queue1601034/torchrun1602577, GPU0·1에서본학습시작. GPU당batch4×accum2/유효16, SSL8clip/update, worker0/RGBmmap. FP32encoder·SSL/BF16planningdyn, checkpointing유지,46decimalGB/GPU전체(타인포함),6GiB여유/allocator22GiB/allocated21.5GiB상한. User요청은VRAM이며CPURAM46GB제한없음.
- 첫조건metric_plus_world학습→고정dev평가→metric_object_future_plus_world감사·학습→평가→paired결과자동연결. 두조건각같은Stage1/같은초기planner/seed47로독립시작,loss는softcandidateimitation+6PDM BCE+.02SSL에객체현재·미래상태/종류보조감독만OFF/ON. 최신질문에두조건이LoRA대partial비교가아님을설명했다. GT채택미정유지.
- 예측전에고정planning1024/world256각40devrecording panel등록,초기및매512update의128dev실제캐시PDMS/ADE/FDE모니터,particle0/2353/4707시각화. 최종world LPIPS/persistence개입/recordingbootstrap2000회paired비교. 작은dev경향/1seed1epoch이며전체dev/test/표현독립효용입증아님. Frozen-LPWM+trained-planner대조는후속설계,현재자동queue에추가하지않음.
- 22:58KST본학습48/4707update,정상로그평균3.496초/peakallocated12.073GiB,GPU전체36.444/36.311GB,source20개불변/failure없음. 초기128dev(유효127)의randomplanner PDMS2.41294 확인. 첫512모니터예상23:25–23:30,조건당학습약4.6시간+검증여유/공유부하,두조건약10–12시간계획. 진전은후속평가필요.
- 사용자후속질문에원문/공식코드조사: LPWM은frozen LPWM+2층mappingL1, V-JEPA2-AC는frozenencoder+actionconditionedpredictor, Drive-JEPA는encoder1e-5/planner1e-4공동학습, UniAD는perception선행후task공동학습하되공개Stage2 image/BEVencoder고정, OpenVLA는LoRA/전체FT경로제공. 현재출력계층5.07%는우리의빠른실험설정이며일반적최적/논문직접재현주장없음.
- Sharedconfiguration/engineering_review/evaluation_protocol/gradient_audit/launch_verification와문서갱신. 이전3673974까지원격push는GitHub인증오류이력;이번push결과는별도확인한다.


## 2026-10-04 23:07 KST — 부분 Stage2 학습 요약 및 ETA 점검

- 사용자요청으로실행중queue1601034/torchrun1602577와첫조건metric_plus_world로그를읽어확인.192/4707update(4.079%),전체train75,297/조건당1epoch/두조건GT보조OFF·ON그대로. LPWM5.56M부분학습+planner2.21M전체/기존SSL유지, GPU0·1 batch4×accum2=16/worker0. Runtime/config변경·추가GPU진단·재시작없음.
- 최근128update경과시간기준3.843초/update(최근64도3.842),loss초기11.309→최근7.540,update128에서측정한모듈gradient유한/양수·decoder0. 등록20source/config불변,queue/training실패marker없음. GPU0·1사용률83/84%,전체VRAM36.444/36.311GB<46decimalGB,peakallocated12.073GiB.
- 학습후dev모니터는아직없고초기128dev만완료. 첫512update평가23:30경예상. 최근속도기준학습계산만첫조건10/5 03:56/두조건08:58,모니터·저장·각조건검증을고려해사용자에게첫조건04:00–04:30/전체10:00–11:00KST로보고. GT ON속도/전체panel평가소요미실측·공유부하/입장대기로변동가능명시.
- 근거 `results/lpwm_partial_planning_v1/status_review_20261004_2307.json` 보존,HANDOFF1–5/RESUME갱신. 직전8294fc3push는GitHub인증실패였으며학습영향없음.


## 2026-10-04 23:10 KST — 일부 계층 학습과 LoRA 차이 설명

- 사용자개념질문에현재LPWM부분학습구현/LoRA원논문(2106.09685)/microsoft공식구현을대조. 선택계층을직접갱신하는것과그계층변화량을저랭크로제한하는것은별도축이며조합가능. 원본W고정상태에서도LoRA유효W+BA가encoder표현을수정함을설명했다.
- 현재native출력12모듈직접학습과중간attention LoRA의가설을구분. LoRA로학습파라미터/optimizer메모리는줄일수있으나앞쪽적용시backward/activation비용이남고우리rollout8step/SSL12frame계산도유지되므로속도/PDMS우열은직접비교필요.1024²/rank8예시는64배파라미터감소만의계산이며속도수치아님.
- 기존queue running/첫조건240update/source20개·config불변을파일조회로확인. GPU실험/LoRA구현/대기열변경없음. 문서/HANDOFF갱신;직전b2c48fdpush는GitHub인증실패상태다.


## 2026-10-04 23:56 KST — 네 가지 미세조정 대기열과 기존full 이어학습


최신 사용자 지시에 따라 **일부 계층 → LoRA → Adapter → 낮은 학습률 전체 미세조정** 순서로 실행한다.
각 학습 뒤에 planning/world 검증을 완료하고 다음 조건으로 넘어간다. 직접 객체 GT 보조 loss는 모두 OFF이며 후순위다.
현재 부분 학습을 재시작하지 않고 CPU supervisor만 교체했다. 학습 torchrun1602577은 계속 실행 중이다.


Partial torchrun1602577를 CPUqueue1675463이 adopt; old supervisor1601034만 종료. 후속queue1709131은 CPU 대기 중.
Full2095checkpoint SHA aef7ab37bfdfcf3a7c032774b0aaf4f4bbd76e8f84849a24a5b5fac2664b9bac, optimizer746states 복원/출력차0. 원본pause 보존.
Partial912/4707, monitor512 PDMS70.8503/ADE1.86759, 메모리46GB 이내/모든logged loss finite.
LoRA1.343M/Adapter0.705M nativefrozen, full109.545M trainable, planner공통2.212M.7개tests/CPU감사통과.
현재partial 학습범위 image2.377M/context0.803M/dynamics2.378M/RGB0; feature to_logvar Identity.
새후속34source/10config hash 등록. Full command conv_in+20epoch LR schedule 차이,1seed/노출dev/no frozencontrol 한계 기록.
LoRA/Adapter/full GPUprofile/학습/최종검증은 아직 대기. 문서 docs/lpwm_planning_experiment.md 최신절 및 results/lpwm_four_method_queue_v1/ 참조.

원격 공유: c4cc73e에 코드·설정·보고서를 로컬 commit했다. 이번 git push mine은 기존 VSCode credential socket ECONNREFUSED/GitHub 인증 실패로 끝났다. 원격 반영은 미완료이며 실행 중 학습과 두 queue에는 영향이 없다.


## 2026-10-05 00:14KST — Stage2 워커·배치 단축 가능성 검토

CPU loading0/2/4/8worker를각2회측정:0.746/1.946/1.699/1.582ms/update. GPU전송/연산제외. 실제logged입력0.01408/전체3.742초=0.376%, 현재남은학습에입력부분제거시약49초상당.
기존batch2→4profile6.678→3.755초로개선이미적용. 두peak점으로batch8 allocated22.091GiB/카드전체47.20GB추정,46GB/6GiBreserve불충족. 현재batch4/accum2/worker0/checkpoint유지.
현재trainer는DataLoader미사용으로workers설정숫자만바꿔도효과없음. LoRA/Adapter/full의기존safe batch profile대기열유지. Runtime34source/10config hash불변. 새GPU작업/학습중단/재시작없음.
Partial1232update;1024monitorPDMS76.6794%,ADE1.40314m. Pmon8회에서타작업과연산공유관측,카드GPUutil을우리전용util로해석하지않음.
첫CPUbench sandbox IPC소켓권한거절로진단만종료후호스트재실행완료. 결과results/lpwm_throughput_review_20261005/,script benchmark_lpwm_stage2_input_loading.py.


## 2026-10-05T00:20:56.240475+09:00 — 48GB VRAM 상한 질문

48GB 상한 검토: batch8 중심 추정47.07–47.20GB는 수치상 들어가지만 상한 여유0.80–0.93GB뿐이며, 추가 workspace1GiB를 포함하면48.14–48.28GB다. 예상 free3.45–3.58GiB로 현재6GiB guard도 충족하지 못한다.6GiB는 우리가 정한 보수적 운용 여유이며 물리적 불가능을 뜻하지 않는다. 현재코드 GB는10진(48GB=44.70GiB). 질문에대한계산검토만수행했고 실제batch8/제한변경없음.

## 2026-10-05: 48GB 상한과 배치8 실행 전환

사용자가 VRAM48GB로 상한을 올리고 배치/워커를 늘려 실제 진행하도록 승인했다.
기존46GB/6GiB 여유 및 학습을 중단하지 않는 이전 조회 범위는 이 요청으로 갱신됐다.
다른 사용자의 GPU process나 원본 데이터는 변경하지 않았다.

### 실행 변경과 실측

| 항목 | 기존 | 새 실행 |
|---|---:|---:|
| 카드 전체 VRAM 상한(다른 사용자 포함) |46decimalGB|48decimalGB|
| GPU당 planning microbatch |4|8|
| Gradient accumulation |2|1|
| GPU 수 / 유효 planning batch |2 /16|2 /16|
| SSL clip 수/update |8|8|
| 입력 worker/rank |0|0|
| 운영 중 최소 물리 free |6GiB|3GiB|
| CUDA allocator 상한 |22GiB|23.2GiB|
| Tensor allocated 상한 |21.5GiB|22.75GiB|

동일한1576 checkpoint 모델/AdamW에서 배치4/8을 각각8update 측정했다. 첫2회를 제외한
평균은4.4078→2.9921s/update(시간32.12% 감소, 처리량1.473배)였다. 공유 GPU 부하에
따라 달라질 수 있는 짧은 순차 실측이며 전역 최적이나 장기 속도 보장은 아니다.
Batch8의 rank0 tensor peak22.0778GiB, 두 카드 전체 최대 관측45.5921decimalGB,
최소 물리 free4.9521GiB,48GB 대비 여유2.2425GiB. OOM/메모리 guard 중단 없음.
Profile 동안0.5초 간격의 카드 통계이며 순간적인 모든 peak를 측정한 것은 아니다.
선택 기준은 유한 loss/gradient,48GB 대비 최소0.5GiB 여유,물리free3GiB와 속도다.
개발 성능은 배치 선택에 쓰지 않았다. 워커0/2/4/8의 앞선 CPU 입력 비교에서0이
가장 빨랐고 입력 준비가 학습시간의약0.38%여서 worker0을 유지했다.

### 학습 보존과 이어가기

Partial1576에서 모든 rank에SIGINT를 보내 update 경계에서 model+AdamW를 저장했다.
125optimizer state가 모두step1576임을 확인했고, 원본을 보존한 별도 복사본 SHA는
`b3d06344cb3fe3c67cdc7708d6f30a36539f98f2d5bece64502f69968b5a3337`이다.
Profile 가중치는 버리고 본학습은1577부터 시작했다. 본학습은 총4707update/75,297train장면,
원래LR/loss/모델/분할/SSL비중/activation checkpointing을 유지한다. Microbatch 그룹이 바뀌면
Dropout 및 개별 SSL clip의 RNG는 달라질 수 있으므로 bitwise 동일한 학습 연속성 주장은 하지 않는다.
기존 로그와 초기 particle 시각화도 새 실행 폴더에 복사해 전·중·후 비교를 이어간다.

### 검증과 대기열

- CPU 검사3개 통과: 과학적 설정 변경 거부, 재개 전후 다음AdamW update 동일성, 빠르지만 메모리 여유가 부족한 profile 탈락.
- 실행 설정9개에서 유효planning16/SSL8 및 기존 scientific field 동일성 확인.
- 두 GPU profile에서 encoder/context/dynamics/planner gradient가 유한하고 양수이며 frozen RGB gradient0 확인.
- 새 wrapper의4planning/2world engineering evaluation 통과. 이는 성능평가 결과가 아니다.
- 새 queue1902774, 현재 본학습 torchrun1978684. 원래 CPU queue1675463/1709131 및 training1602577은 정상인계 후 종료.
- 순서: Partial학습→1024planning/256world검증→LoRA4/8 profile·감사·학습·검증→Adapter2/4/8→full2/4 재개·검증→네 방법paired보고.
- 후속 방법도 성공한 실측 중 안전하고 빠른 배치 선택. 메모리만의 profile 실패는 작은 배치로 fallback, 다른 실행 오류는 중단.
- Scientific 효용 gate 실패는 보고하고 다음 독립 비교를 계속한다. 불완전 학습/실행/인과검사 오류는 의존 작업을 차단한다.
- 객체 GT 보조loss OFF/후순위. Full은 원본2095checkpoint와20epoch scheduler/conv_in명령 위치를 유지하고4707까지 이어간다.

현재 상태: `outputs/lpwm_48gb_planning_v1/queue/queue_state.json`.
실측: `results/lpwm_48gb_planning_v1/execution_review.json`.
설정: `configs/lpwm_planning/execution_48gb_v1/queue.json`.
새 실행 source38개/config20개를 hash등록했다. 실행 중 해당 source/config를 바꾸지 않는다.
본학습 중 카드 통계는2초마다 감시하며48GB 또는free3GiB guard가 걸리면 우리 작업만 중단한다.
공유GPU의 외부 메모리 급증이나 순간 할당까지 막는 하드웨어 격리 보장은 아니다.

### 2026-10-05 00:47 KST — Stage2 배치8 중간점검

현재 부분계층 미세조정1712/4707update(36.4%), queue1902774/torchrun1978684/worker1979560·1979561 정상.
배치8/누적1/2GPU/유효16/worker0/객체GT OFF. 재개 후 기록된step평균2.747s, wall평균2.883s/update.
전체GPU0/1 약45.592/45.570GB, free4.952/4.974GiB로48GB/3GiB guard 이내. OOM/NaN/오류 marker 없음.
Source38/config20 hash 모두일치. 현재학습/queue/설정변경·추가GPU진단 없이 저장로그와process를조회했다.
초반16개로그평균total loss8.7177→최근16개6.9469, planning8.3387→6.5685, SSL18.9490→18.9191.
서로다른training batch의진단평균이며held-out world보존을의미하지않는다. 실제최근gradient검사는1664update,
encoder23.798/context4.631/dynamics27.539/planner+command21.005,모두유한·양수; frozenRGB0. 전역norm5 clipping 전값이다.
고정dev128장면(유효PDMS127)의512/1024/1536 update PDMS70.8503/76.6794/79.1126,
ADE1.8676/1.4031/1.5267m,FDE4.2098/3.2788/3.5224m. 최근PDMS는상승했지만ADE/FDE는악화해모든지표개선주장불가.
최신dev1536은batch변경1576 이전 결과다. 변경후첫monitor는2048, 최종1024planning/256world검증은학습후대기.
학습전미학습planner대비진전이지LPWM미세조정단독효과는아직분리되지않았다. Frozen-LPWM학습대조군/독립test결과없음.
Partial학습완료예상10/5 03:10–03:40KST(공유GPU변동/중간검증·시각화여유, 종료후검증시간제외).
후속LoRA→Adapter→full재개대기; 각방법학습후검증연결유지.
근거 `results/lpwm_48gb_planning_v1/health_review_20261005_0045.json`.

## 2026-10-05: LPWM 적응과 planner 학습 효과를 구분하는 대조 설계(제안)

사용자는 현재 PDMS 상승에서 LPWM 미세조정과 planner 학습의 효과를 어떻게 구분하는지 질문했다.
현재 initial 대비 trained 평가는 LPWM과 미학습planner가 함께 변하므로 두효과를분리하지못한다.
기존 네방법queue에는 별도학습 frozen-LPWM대조군이 없다. 이번에는 설명과설계기록만수행하며
GPU추가실행/queue변경/새실험등록은하지않았다.

코드상 command_feature_modulation(FiLM)은 optimizer의planner_and_command_input 그룹에속하지만
LPWM image encoder의particle 속성CNN에개입하며SSL과planning gradient를받는다. 따라서 native
LPWM가중치만고정하고FiLM을학습하는조건을엄밀한고정표현/순수planner학습으로부르면안된다.

| 제안 조건 | 원래LPWM가중치 | ego-intent FiLM | planner |
|---|---|---|---|
| A: 고정표현대조 | Stage1상태고정 | 초기identity상태고정 | 새로학습 |
| B: native LPWM고정대조 | Stage1상태고정 | 학습 | 새로학습 |
| C: 현재부분적응 | 지정native출력계층학습 | 학습 | 새로학습 |

B-A는고정native LPWM에서명령조건화학습을허용한효과, C-B는명령조건화와planner학습에더해
native LPWM적응을허용한추가효과다. 동일Stage1/동일planner초기화/동일학습목적·학습률·데이터순서·
update수·유효배치·평가장면으로비교한다. 현재partial은1576까지batch4/그후8이므로정확대조는그
실행일정도맞추거나별도matched재실행으로해야한다. FiLM을학습하는B에서는SSL도FiLM으로gradient가
흐르므로SSL항을임의제거하면C-B가미세조정유무단독대조가되지않는다. FrozenLPWM은
requires_grad=False로고정해도FiLM까지의연산graph는유지해야한다.

연구의planning-aware목적을검증하려면추가D조건도유용하다: C와동일한LPWM학습계층+FiLM을
SSL로만갱신하고, planningbranch의particle표현에서stop-gradient하여planner만planningloss로학습.
C-D는같은SSL후속학습에planninggradient를표현까지전달하는것의추가효과를검증한다.
각차이는해당조건에서의시스템효과이며모듈기여를보편적인가산비율로분해하는것은아니다.

평가는동일장면의paired PDMS/ADE/충돌·도로이탈subscore와recording단위bootstrap CI,가능하면3seed,
고정개발패널및최종독립평가로확인한다. Frozen대조군대비PDMS차이의CI가0을포함하면우월성미확정.
원래/적응표현을각각고정하고같은용량·초기화·예산의새readout/planner를따로학습하는추가진단은
특정공동학습planner와의호환성너머표현의읽기쉬움/유용성을검사할수있다. readout학습예산이동일해도
표현을얻기위한전체학습비용이동일한것은아니므로별도로보고한다.

이미학습된planner에Stage1 LPWM만끼워넣어성능이떨어지는것은표현분포/particle의미불일치때문일수
있으므로미세조정표현의품질개선증거로충분하지않다. Gradient가흐르거나가중치가변했다는사실도
PDMS기여를보장하지않는다. 기존persistent_future평가는학습된planner가미래입력에의존하는지의
보조개입이며LPWM미세조정대조를대체하지않는다.

### 2026-10-05 01:59 KST — 학습 진행 조회

Partial3184/4707update(67.64%), queue1902774/torchrun1978684/두rank 정상. 기존batch8/accum1/유효16/worker0유지.
최근로그평균2.780s,최근256update wall2.913s/update;남은학습약1.23h,완료03:15–03:30KST예상(최종검증별도).
고정dev128/유효PDM127:2048update PDMS78.9990/ADE1.35675m,2560update82.3708/1.39076m,
3072update81.8502/1.28197m/FDE2.97124m. 최근PDMS0.52점하락과ADE/FDE개선을함께관찰하며단조개선주장없음.
모든loggedloss유한,3072gradient유한·양수/frozenRGB0,OOM·오류marker없음,source38/config20hash일치.
GPU0/1총45.592/45.570GB,각free4.95GiB이상.2353update중간particle시각화저장확인.
최종1024planning/256world평가와LoRA→Adapter→full학습검증대기. Frozen대조군은제안만있고미실행.
기존GPU학습이나queue/source/config변경없음. 근거 `results/lpwm_48gb_planning_v1/status_20261005_0159.json`.

### 2026-10-05 — NAVSIM PDMS 최고 보고치 원문 확인

사용자SOTA질문에공식문서·논문·저자저장소를조회했다. 이번조사에서확인한NAVSIM-v1 navtest
논문최고보고치는 **DrivoR+SimScale+Traffic-Element Awareness95.1**이다.
[원문Table2](https://arxiv.org/html/2608.18035v1#S5.T2):DrivoR93.1/TE추가94.4/SimScale추가94.6/둘다95.1.
합성학습데이터와외부detector로만든교통요소pseudo감독이포함되고추론에는예측TE표현을사용한다.
[iDriveVLA](https://arxiv.org/abs/2609.30818)는94.95와공개리더보드1위를저자보고(2026-09-25).
[실험설정](https://arxiv.org/html/2609.30818v1#S4.SS1)은103k실데이터+134kSimScale와DrivoR/InternVL3-2B.
[TOAD저자repo](https://github.com/valeoai/TOAD)는94.9,
[ChainFlow-VLA](https://arxiv.org/abs/2605.23270)는94.85(trainval103k),
[DriveSuprim](https://arxiv.org/abs/2506.06659)은93.5를보고한다.
공식HF리더보드는앱shell만읽혀실시간순위직접검증불가. 따라서95.1은이번조사에서확인한논문최고치이며
공식실시간1위로단정하지않는다. NAVSIM-v2 EPDMS와v1 PDMS구분,
우리81.85는고정dev128(유효127)중간점수여서전체navtest SOTA와직접순위비교불가.
근거 `results/lpwm_48gb_planning_v1/navsim_pdms_literature_review_20261005.json`.
학습/queue/config/방법론변경없음.


## 2026-10-05 09:36 KST — VRAM 감소와 현재 배치 확인

현재 LoRA 4352/4707, GPU당batch4/누적2/GPU2/유효16. Partial8은 완료됐고, LoRA8은03:37:55 backward중23.20GiB allocator allowance에서20MiB 할당을거부해profile탈락. 당시물리free4.63GiB/카드45.935decimalGB이므로물리메모리전량사용에따른OOM으로설명하지않는다. 현재GPU0/1우리process14452/14456MiB,카드34658/34663MiB,util100/84%. 최근update4304→4320→4336으로증가했고읽기전용점검중계속학습.

Partial 최종dev1024/유효PDMS1021/40recording PDMS81.6341/ADE1.28250m; world256 reconstruction 유지통과/forecast LPIPS 실패. 이는 전체검증통과가 아니며 사전등록 독립방법비교 계속 규칙으로 LoRA 새초기화학습진행. 이번턴실행설정·source·queue·cap변경없음. 근거 `results/lpwm_48gb_planning_v1/batch_verification_20261005_0935.json`.


## 2026-10-05 09:45 KST — 모든 후속 학습 batch8 우선과48GB 카드예산

**최신(2026-10-05): 사용자 요청으로 모든 후속 학습에 batch8 우선 정책 적용.**
현재 진입점은 `configs/lpwm_planning/card_budget_batch8_v2/queue.json`과
`scripts/queue_lpwm_card_budget.py`(queue408762). GPU0·1 각각 전체48decimalGB(다른 사용자 포함)를 감시한다.
이전 고정allocator23.2GiB/allocated22.75GiB/free3GiB는 이번 승인으로 대체됐다.
현재 NVML점유에서 우리allocator를 제외한 사용량과 workspace512MiB/반올림128MiB를 뺀 잔여예산을 적용한다.
LoRA·Adapter·full 모두batch8 먼저실측,48GB내통과하면8채택; 메모리실패시에만4/2fallback.
LoRA4422 model+AdamW226state를 별도보존하고4423부터재개. Partial완료결과재사용,full원본2095재개유지.
유효planning16/SSL8/worker0/기존loss·LR·데이터·4707목표유지. Profile학습결과는본학습에사용하지않는다.
새queue등록후source/config변경금지, 이전queue1902774는superseded이며재기동금지.
이전 48GB v1의 고정프로세스예산/배치선택 설명은 보존이력이다.

이전CPUqueue만종료후LoRA두rank에SIGINT,4422 update/226AdamWstate원자저장. Checkpoint SHA04efad377910390cb4ba947eb5b410f5972438d10199eb2b071b1bac7e104f61. 기존등록source38/config20불변,CPU6tests통과. 새profile은진짜forward/backward/optimizer8update이며discard;LoRA4423재개와full2095재개는보존source를쓴다.

### 같은 요청의 메모리실측 후속 — v4

v2 기본allocator는47.449GB/23.38GiB allocated/583.93MiB reserved-unallocated에서실패. v3 expandable은47.456GB/23.71GiB allocated/252.42MiB reserved-unallocated로단편화는감소했으나보수적23.956GiB allowance에걸림. 비allocator추가사용약122MiB측정에근거해v4는growth192MiB+rounding64MiB로조정,shared48GB상한유지. v2/v3 queue정상반환확인후v4 queue421603시작. 보존LoRA4422원본은모든시험동안불변.

### 2026-10-05 09:59 KST — 배치8 본학습 재개 성공

v4 profile8회통과/카드최대47.598010368GB/peak allocated23.835232GiB/last6평균3.483267s. Gradient audit+engineering통과후421603queue/427197torchrun이4423부터재개하여4448/4707확인,loss6.572179,update3.293407s. 본학습2GPU100%/카드각약47.60GB. Native model/AdamW226state복원원본SHA보존. 42source/29config불변,CPU9검사통과. 다음Adapter/full도8우선이며48GB초과/메모리실패시에만작은배치로진행한다.
기록 `results/lpwm_card_budget_measured_v4/resumed_batch8_status_20261005.json`. v2/v3실패와중단로그는보존,완료Partial 재실행없음. 최종LoRA PDMS/전체후속방법성능은아직대기.


## 2026-10-05 10:06 KST — LPWM 학습 중간결과 비교

2026-10-05 10:06 KST 사용자 중간결과 요청: 저장로그·개발검증을 읽고 같은128monitor/동일4096update 비교 및곡선PNG 작성. Partial1epoch4707 최종1024개발 PDMS81.6341/ADE1.28250/FDE3.04333. LoRA4528/4707(10:04snapshot),최신공통4096 monitorPDMS partial80.6169/LoRA84.5584(+3.9415점),ADE1.23115/1.11915. Adapter미시작/full2095보존재개대기. Runtime/queue/source/config변경없음.

Partial world256 같은장면 LPIPS 복원0.304335→0.305230(+0.294%,유지통과),미래0.392249→0.438393(+11.764%,등록10%유지gate실패). 회전/작은객체/박스겹침상황군에서도미래LPIPS악화. Partial미래particle을현재표현반복으로교체하면PDMS81.6341→76.2361,paired차이5.398점/95%CI4.139–6.680;추론입력분포교란검사이므로미래예측학습·LPWM미세조정의독립효과단정금지. Stage1전체7745의LPIPS는복원0.770675→0.300016/미래0.807938→0.382994,SSL영상적응확인이나객체/완전적응증거는별개. JSON/PNG `results/lpwm_card_budget_measured_v4/interim_review_20261005/`.

128monitor(유효PDMS127)과1024최종개발(유효PDMS1021)을분리했으며대응token동일검사완료. 학습곡선은update0을가독성때문에생략,완료partial4707과현재LoRA4096최신monitor를표시. 이후실행은기존queue가수행한다. 학습·GPU·source설정수정/추가실험없이CPU집계·그림만생성.


## 2026-10-05T10:12:20.347189+09:00 — 공식 평가 분할과 내부 개발 분할 명확화

사용자 지적에 따라 "전체 navtest 성능은 아니다"라는 설명을 명확히 한다. 현재 LPWM의81.63/84.56점은
**navtest 일부의 점수도 아니며, navtrain에서 recording 단위로 분리한 내부 개발 장면의 점수**다.
공식 navtrain103,288token filter에서 현재 유효planning manifest는 train75,297/dev27,076이고,
train/dev token과recording 중복은각0이다. 고정최종개발panel1,024token을공식filter와대조한결과
navtrain1,024/navtest0;중간monitor는이panel의첫128이다. 유효PDMS는각1,021/127이다.

현재PDMS는모델이선택한고정후보의공식v1 PDM시뮬레이터cache점수를조회한다. 모델이예측한점수를
실제평가점수로보고하는것은아니다. 다만공식평가split을사용하지않았으므로논문/리더보드와비교불가다.
"최종 검증"은현재조건학습종료후내부개발검증이라는뜻이며공식benchmark완료가아니다.
현재configs에는automatic_navtest=false이며네방법queue에도navtest/navhard실행단계가없다.
최종벤치마크비교에는고정checkpoint/설정으로v1 navtest전체12,146token의공식평가가추가로필요하다.
navhard는별도v2/EPDMS프로토콜을맞춘다. 이번질문에서는분할검증·문서정정만수행하고활성학습queue는보존했다.

근거: `results/lpwm_card_budget_measured_v4/evaluation_split_clarification_20261005.json`,
[공식분할](https://github.com/autonomousvision/navsim/blob/main/docs/splits.md),
[공식지표](https://github.com/autonomousvision/navsim/blob/main/docs/metrics.md).

## 2026-10-05 10:18 KST — E2E 논문의 학습·평가 데이터 관행 확인

- 사용자 질문에 답하기 위해 DiffusionDrive/DrivoR/DriveSuprim/PARA-Drive와 NAVSIM 공식 분할·지표 문서를 확인했다.
- DiffusionDrive는 navtrain100epoch/navtest, DrivoR는 navval에서 어블레이션 후 v1 competition split25epoch/navtest. NuScenes planning은 공개 val 비교도 쓰며 지표 구현 일치가 필요하다.
- v1 navtest/PDMS와 v2 navhard_two_stage/EPDMS는 별도 프로토콜이다. 우리 개발 분할을 DrivoR navval로 부르지 않는다.
- 현재 LPWM은 navtrain 내부 개발 검증이고 공식 테스트는 미실행·현재 queue 미포함. 1epoch 결과는 빠른 경향 비교이다.
- 권고안과 원문 링크를 docs/lpwm_planning_experiment.md에 기록했다. 고정표현 대조와 동일 조건 공식 navtest 전체 평가를 추천하되 신규 실행으로 등록하지 않았다.
- 활성 학습/queue/등록 source/config 변경 없음. 이번 턴 현재 학습 진척 재조회 없음.

## 2026-10-05 10:26 KST — navval 정의와 논문 공정 비교 권고

- 사용자 요청은 개념 설명과 권고이며 신규 학습·평가 실행은 하지 않았다.
- DrivoR fc6e5aa의 navtrain filter103288token/1192log segment를 확인. 기본 train978/val214 log segment로 중복0. Competition 학습은1192개이며 val도 포함한다.
- run_training.py는 동일 filter에 train_logs/val_logs를 각각 적용하고, README v1 full 명령은 비캐시 run_training_full.py로 합집합을 학습한다. 앞선 navtrain+navval 설명이 별도 추가 원본 데이터셋을 뜻하는 것으로 오해되지 않도록 보충했다.
- 권고는 개발 단계 Stage1/2 공통 split→설정 고정→동일 navtrain 전체 풀 최종 학습→전체 navtest/v1 공식 평가, navhard/v2 별도 확장이다. Frozen-LPWM 대조 및 센서/사전학습/감독/학습량을 함께 보고한다.
- 공개 LPWM 사전학습을 쓰므로 전체 학습을 navtrain only라고 부르지 않는다. 기존 자체dev와 DrivoR 기본val은 서로 다른 구성이다.
- 상세 근거/링크는 docs/lpwm_planning_experiment.md 최신절. 학습·queue·등록 source/config 변경 없음.

## 2026-10-05 10:31 KST — LoRA 학습 완료 확인

- 2026-10-05 10:31 KST 확인: LoRA 본학습은10:15:22KST 4707/4707update,1epoch 정상종료(returncode0). checkpoint/latest/epoch01 보존. 마지막128개발monitor PDMS83.6380/ADE1.14796/FDE2.72267. Queue421603/evaluation457092 실행,학습427197 종료. 현재 검증 단계 {'stage': 'persistent_future', 'completed': 132, 'total': 1024}. Adapter/full은 이후 등록 대기. 읽기전용점검으로runtime변경없음.
- 체크포인트453,050,300bytes, 저장10:15:14; latest와epoch01도존재. 완료summary의profile_only=false.
- 마지막monitor는128개/유효127개로 최종1024점수나navtest성능이아님.
- GPU0·1전체점유조회는각21966/20201MiB,util28/26%;학습종료후검증중인상태.
- 근거 results/lpwm_card_budget_measured_v4/lora_training_completion_20261005.json;활성queue변경이나추가학습없음.

## 2026-10-05 10:35 KST — 현재 navtrain 학습·개발 사용량 확인

- Stage2는공식103288장면중train75297/dev27076,프레임시간간격부적합915제외. 비율72.900/26.214/0.886%.
- 같은원본주행기록을묶어122개recording학습/40개개발로고정;token·recording중복0. 기존개발40개보존,나머지학습정책이며예전heldout40개도현재학습에포함.
- Stage1은동일recording분할에서12RGB완전가용train23126/dev7745 clip. 각token은Stage2동일split부분집합임을검사했다.
- 현재Stage2는개발27076전체가아닌고정128monitor/1024planning/256world평가. 40recording을고르게선정하며모델예측무관. 공식navtest/navhard미사용.
- 근거results/lpwm_card_budget_measured_v4/navtrain_usage_breakdown_20261005.json. 학습·queue·source/config변경및새GPU작업없음.

## 2026-10-05 10:39 KST — Adapter 시작 여부 확인

- 2026-10-05 10:39 KST: Adapter는 아직미시작. 현재queue attention_lora_evaluation, LoRA학습완료후 world_retention 97/256검증중. Adapter node/실행선택파일없음. Heartbeat정상갱신. 근거results/lpwm_card_budget_measured_v4/adapter_start_check_20261005.json.
- LoRA의planning/미래표현개입 결과파일은생성됐으며world유지검증과최종집계가남았다. 검증후기존대기열이Adapter8우선profile→학습으로이어진다.
- 추가GPU작업/학습재기동/대기열/source/config변경없음.

## 2026-10-05 10:52 KST — Adapter 본학습 시작 및 적용 구조 확인

- 2026-10-05 10:52 KST: Adapter본학습10:46:36KST시작,update80/4707. GPU당batch8/accum1×2=16,전체VRAM각45.89GB. Queue421603/torchrun866051. LoRA검증10:43:53완료후profile/gradient/engineering검사통과해자동진입. 이번턴runtime변경없음.
- Native LPWM109.55M고정. 11개Transformer block출력뒤64차원bottleneck nonlinear residual을추가. 위치는particle interaction1/context4/dynamics6.
- Adapter704960/planner+FiLM2211975/총2916935학습. FiLM은attributeCNN conv_out에유지되며최대LR각1e-5/3e-4,planning+metric+0.02SSL공동loss. 직접객체GT OFF.
- Stage1동일checkpoint+동일planner초기화의독립조건. 이전LoRA학습을이어가는것아님. 초기출력차0,진단frozen해시보존,planning/SSL gradient검사통과. Decoder는고정이지만SSL입력gradient유지.
- 진행log의초기대형gradient는clip전값이며gradient_clip5/error_if_nonfinite=True를적용한다. 로그의image_encoder gradient그룹은interaction도포함;CNN native가학습된다는뜻아님.
- LoRA검증완료와모든과학적gate통과는별개이며본기록JSON에trend_checks와checks원문을보존했다.
- 근거 results/lpwm_card_budget_measured_v4/adapter_training_and_architecture_20261005.json. 학습/queue/source/config변경없음.

## 2026-10-05 11:01 KST — 완료 LoRA 결과 및 일부계층 비교

- 동일1024개발/유효PDMS1021/40recording,world256. CPU로paired recording bootstrap2000회/seed20261003 집계.
- LoRA PDMS81.9141 vsPartial81.6341;차+0.2799점95%CI[-0.7341,1.3308],우월성미확정. ADE1.18220vs1.28250(-0.10030m),FDE2.84057vs3.04333(-0.20276m)는pairedCI0미포함.
- Stage1/Partial/LoRA 미래LPIPS .392249/.438393/.391846. LoRA는-0.103%수준유지,추가개선CI0포함. 복원.304335/.305230/.304335. LoRA등록경향gate전부통과.
- Persistent미래교체시LoRA81.9141→79.2509,차2.6632점CI[1.3160,4.0355]. 추론교란이며LoRA학습효과의독립증거아님.
- 학습가능총parameter7.770M→3.555M이나기록학습시간4h17m→6h19m. 공유부하·배치이력·adaptation위치차이로속도인과주장금지.
- 추가감사초기후보62/1024불일치/최종567불일치. 동일Stage1·seed설정이지만저장초기출력동일성은성립하지않음;원인은미확정. 현재queue의four-method집계는초기동일성assert없이paired최종평가를비교한다.
- scripts/report_lpwm_completed_lora.py 및 results/lpwm_card_budget_measured_v4/completed_lora_review_20261005/summary.json,comparison.png/pdf 생성. 그래프를직접열어축·수치·가독성확인.
- 이전84.56은4096update/128monitor이며이번81.91은4707/1024최종개발. Frozen-LPWM학습planner대조·seed반복·공식navtest는여전히미완료.
- 활성Adapter학습/queue/등록source/config변경및새GPU작업없음.


## 2026-10-05 11:09 KST — seed·epoch 개념과 문헌 학습량 비교

현재 Stage2 방법별 실행은 seed47 한 번이며, 75,297개 학습 장면을 1epoch 순회해
유효batch16 기준4,707optimizer update를 수행했다. Seed는 초기화·shuffle·dropout 등의
난수 설정이고, 독립seed 반복은 동일Stage1에서 새planner/적응모듈 학습을 다시 시작하는 실험이다.
Stage1은별도로23,126clip을20epoch/28,920update 학습했으므로전체모델학습이1epoch라는뜻이아니다.
현재Stage2는빠른방법선별이며수렴·최종순위를입증하지않는다. Recording bootstrap CI는학습seed변동을포함하지않는다.

문헌확인: [DiffusionDrive §4.2](https://arxiv.org/html/2411.15139v2)는NAVSIM100epoch/totalbatch512,
[DrivoR §4.2.4·Appendix D](https://arxiv.org/html/2601.05083v2)는v1최종25epoch/v2 10epoch를사용한다.
DrivoR는v1개발성능이25epoch에서plateau하며v2는장기학습시악화됨을보고한다.
데이터·batch·사전학습이달라epoch만으로계산량이나충분성을비교하지않는다.
확인한본문에서반복seed수는명확히검증하지못했으므로두논문이1seed/3seed라고단정하지않는다.

권고는현재방법선별후유망한1–2조건과동일planner의frozen-LPWM대조를수렴까지비교하고,
핵심조건은예컨대3개seed로반복해평균·편차를보고하는것이다. 학습량은개발곡선으로선정한다.
이는이번질문에대한권고이며활성queue/epoch/seed/source/config변경이나새학습은없다.


## 2026-10-05 11:14 KST — Stage1/2 epoch 재확인

Stage1완료summary는20epoch/28920update/23126clip,완료Stage2 LoRA는1epoch/4707update/75297장면이다. 1epoch제한은현재Stage2방법비교의학습량이며Stage1에적용되지않는다. 읽기전용확인과인수인계기록만수행,활성학습·queue·source/config변경없음.


## 2026-10-05 11:19 KST — 학습 충분성과 추가 epoch 검토

연구 하위 질문: 현재 학습 예산으로 planning 목적의 LPWM 적응 효과를 판단할 수 있는가?
Stage1 고정512개발clip의temporal SSL 검증loss는epoch10/15/20에서20.35810/19.88686/19.69678.
15→20은0.956%추가감소,PSNR21.05743→21.07677dB로후반개선폭이작다.
20epoch까지적응은관측되지만최적학습량을입증한것은아니다. 이monitor는full영상temporal목적함수이며
epoch별past-only미래예측곡선이나planning유용성곡선이아니므로그수렴을대신판정할수없다.

LoRA 동일128개발monitor(유효PDMS127)의update3072/4096/4707은PDMS81.4273/84.5584/83.6380,
ADE1.12896/1.11915/1.14796m다. 전체적으로학습됐지만후반에는지표가오르내려지속개선·완전수렴을모두단정못한다.
학습loss의기록minibatch평균은updates3073–4096에서6.58144,4097–4707에서6.53389로소폭감소.
정확한전체epoch평균이아니며학습loss감소자체는PDMS개선증거가아니다.
기존1024최종평가PDMS81.9141과이128monitor를혼합해하락추세로판정하지않는다.

권고: Stage1은현20epoch를고정하고우선유망Stage2조건+frozen-LPWM대조를총3→5epoch로확장해
매epoch동일1024개발planning/256world 및시나리오별안전지표를검증한다. 숫자는제안이며등록실행아니다.
이후필요시더확장하고핵심결과를복수seed로반복한다. Stage1추가학습의효과는별도대응planner실험으로분리한다.
LoRA현재1epoch cosine끝LR은LPWM1e-6/planner3e-5로초기최대의10%다.
연장시모델·optimizer를보존하되학습률스케줄과비교예산을사전명시해야하며epoch설정만바꾸는것으로취급하지않는다.

문헌도추가학습효과가일률적이지않다: [DrivoR §4.2.4](https://arxiv.org/html/2601.05083v2)는v1에서는25epoch까지
개선후plateau, v2에서는학습연장시EPDMS악화를보고한다. 이결과는우리모델의추가개선을보장하지않는다.
새CPU script `scripts/report_lpwm_training_convergence.py`,원자료hash/집계/PNG/PDF는
`results/lpwm_card_budget_measured_v4/convergence_review_20261005/`에저장했다.
활성학습·queue·등록source/config변경이나새GPU평가는없다.


## 2026-10-05 14:10 KST — 입력 이미지 해상도 감사

현재공개Sketchy checkpoint hparams는image_size128/normalize_rgb=false다.
NAVSIM CAM_F0 전방단일카메라1920×1080 RGB에서위아래28px씩제거하여1920×1024로만든뒤,
cv2.INTER_AREA로128×128에직접축소한다. 가로세로비율을유지하는letterbox가아니며,
가로1/15·세로1/8배율로변환한다. 원본30px폭은입력에서약2px폭에해당한다.
실제rgb_frames.npy헤더shape=(152495,128,128,3),dtype=uint8를읽기전용확인했다.

Stage1은이캐시12프레임을(batch,12,3,128,128) float/[255]로전달한다.
Stage2 planning은동일캐시관측4프레임(batch,4,3,128,128)과ego status를입력하며,
별도world보조학습은12프레임을사용한다. 이미지값은[0,1],ImageNet평균/표준편차정규화없음.
모든프레임간격0.5초;미래RGB는planning입력이아니며planning은예측particle을받는다.
Stage2에서별도고해상도원본경로를사용하지않는다.

128해상도는공개모델구성을유지한설정이며자율주행최적해상도라는검증결과가아니다.
원거리작은객체/차선정보손실과종횡비왜곡은가능한제약이나현재PDMS원인으로확정못한다.
해상도확장의효과를보려면원본에서캐시를다시생성하고모델/체크포인트/particle구조·메모리를검증해야한다.
이미128로줄인캐시를확대하는것만으로원본세부정보를복구할수없다.
이번질문으로해상도·학습·queue·등록source/config를변경하지않았다.
근거 `results/lpwm_card_budget_measured_v4/image_preprocessing_audit_20261005.json`;
코드 `scripts/prepare_lpwm_navsim_posttraining.py:169`, `scripts/train_lpwm_partial_planning.py:67`.


## 2026-10-05 14:16 KST — 해상도와 LPWM 구조 관계

현재공개Sketchy checkpoint의image_size128을유지한것은모델구성/가중치호환을위한선택이다.
LPWM개념이128만가능하다는뜻은아니다. 공식DLP는image_size를인자로받고기본구현은정사각형을가정한다.
models.py:249에서패치수=(image_size//patch_size)^2,261에서객체glimpse크기를image_size로정한다.
modules/modules.py:2760의attribute head Linear입력차원,2541이후background latent projection,
4812의patch_centers buffer,decoder출력공간도구성크기에연결된다. 따라서공개가중치를그대로둔채
입력resize만256으로바꾸는작업으로고해상도적응이완료되지않는다. 공유가능가중치이식과달라지는구성검증이필요하다.

상하28px crop와가로세로비율을바꾸는resize는우리NAVSIM adapter의전처리선택이다.
동일상하crop는로컬Drive-JEPA/TransFuser공식NAVSIM코드에도존재하지만LPWM자체요구조건은아니다.
특정28값의독립효과나현재왜곡resize의최적성을검증한실험은없다.
정사각형입력도비율유지padding등으로구성할수있으나128캔버스에서는유효영상영역이줄어들므로
왜곡해소와작은객체해상도확보는별개로검증해야한다. 고해상도/직사각형입력확장은가능한연구방향이며미구현이다.

공식근거: https://github.com/taldatech/lpwm/blob/main/models.py 및 configs/sketchy.json;
사용중고정checkout의models.py/modules/modules.py와실제hparams를확인했다.
이번질문은설명/코드감사이며활성학습·queue·등록source/config·해상도변경없음.


## 2026-10-05 14:24 KST — 타 E2E 해상도와 정보 보존 비교

연구 하위 질문: 작은 객체의 입력 정보를 유지하면서 planning에 필요한 entity 표현으로 압축할 수 있는가?
공식코드/논문을확인한예시(가로×세로): 현재LPWM전방1개128×128;
Drive-JEPA PF공식코드의front_only분기512×256,3카메라결합분기1024×256;
DiffusionDrive NAVSIM은좌/전/우crop을이어붙인전체1024×256;
VAD nuScenes Tiny는카메라당640×360,Base1280×720(각32배수padding후640×384/1280×736);
DrivoR논문Table11은카메라당1148×672/4카메라,공개config와PILresize순서도확인했다.
카메라수·화각·센서구성·벤치마크가다르므로이미지면적비로성능차이를설명하지않는다.

원본축소는비용절감을위해쓰이지만planning에필요한작은객체/차선세부정보가사라지면성능저하가능.
같은1920폭영상의30px객체는128폭에서2px,512폭에서8px이다. 이는기하학적예시이며감지성능측정이아니다.
모든축소가동일하게나쁘거나원본해상도가항상최적이라는근거는없다.
VAD Tiny/Base는해상도외모듈깊이·BEV query수도달라성능차이를해상도단독효과로주장하지않는다.
우리128설정의PDMS손실량은해상도통제실험전에는미확정이다.

DrivoR Table4(b)의LoRA조건에서전체scene tokens약16k는navval PDMS90.2,
register압축64scene tokens는90.0이다. 이는입력이미지를극단적으로축소한실험이아니라
인코딩된표현을학습적으로압축한사례이며우리particle설계에참고할수있다.
후속설계에서는영상해상도·종횡비처리·particle예산을분리하고원본재처리/체크포인트호환을검증한다.
선별후학습량확장필요성은유지되나128입력정보가충분하다고가정하지않는다.
이번문헌/코드조사로활성학습·queue·등록source/config변경이나새GPU실험없음.

출처:
- https://raw.githubusercontent.com/hustvl/DiffusionDrive/main/navsim/agents/diffusiondrive/transfuser_features.py
- https://raw.githubusercontent.com/hustvl/VAD/main/projects/configs/VAD/VAD_tiny_e2e.py
- https://openaccess.thecvf.com/content/ICCV2023/papers/Jiang_VAD_Vectorized_Scene_Representation_for_Efficient_Autonomous_Driving_ICCV_2023_paper.pdf
- https://arxiv.org/html/2601.05083v2 (Table4/11)
- 로컬Drive-JEPA: reference_repositories/DriveJEPAOfficialEvaluation/navsim_v1/navsim/agents/drive_jepa_perception_free/drive_jepa_features.py:49


## 2026-10-05 14:31 KST — 공개 LPWM 해상도 지원 범위

공식config들을확인하면64×64(bair64/balls/ogbench/shapes)와128×128(sketchy등)설정이존재한다.
따라서LPWM아키텍처가128만가능한것은아니다. 다만오늘확인한공식README Model Zoo의
Sketchy/SketchyAction/BAIR/LanguageTable/Bridge 공개checkpoint표는모두128×128이다.
64config존재를64공개weight확인으로표현하지않고,256dataset링크를256모델weight로해석하지않는다.

현재Sketchy checkpoint를구성변경없이재사용하는경로는128입력이다.
256×256은기존weight를출발점으로해상도관련head/bg/decoder/patch좌표및particle구성을조정해
적응학습하는확장방향이다. 실제256호환·학습성능검사는이번턴수행하지않았다.
512×256등직사각형은기본정사각형가정수정이추가로필요하다.
호환되는weight재사용이가능한범위를검증해야하며모든가중치를새로학습해야한다고단정하지않는다.

출처: https://github.com/taldatech/lpwm#model-zoo---pretrained-models 및
https://github.com/taldatech/lpwm/blob/main/configs/bair64.json .
이번사용자질문은가능범위설명이며신규학습/해상도확장구현/활성queue변경없음.


## 2026-10-05 14:37 KST — GPU0·1 공동 사용자 종료 시각 확인

14:34:23 카드점유각23591MiB,호스트compute조회와ps대조에서GPU0·1은우리866456/866457 junseong kjs-lpwm-stage2만사용한다. GPUaccounting은양쪽Disabled. 최신저장공동점유snapshot10:52:35는각45.89GB이나소유자별내역없음. Queue는resource최신값만덮어쓰므로junheok의정확종료시각/지속부재시간은확정할수없다. 추가공동점유가10:52~14:34사이에사라진관측과개인별종료시점을구분한다. Adapter14:35:32 progress4176/4707,학습·queue·source/config변경없음. 근거results/lpwm_card_budget_measured_v4/gpu_shared_user_departure_review_20261005.json.


## 2026-10-05 14:42 KST — pjh 프로세스 이력 확인

사용자가 제공한 pjh- 이름으로 자체 세션의 과거 GPU 조회 원본을 대조했다. 09:35:49KST GPU0(UUID95f20c84...) PID290957, GPU1(UUIDa987cdb2...) PID290958의 실행파일은 /rhome/junhyeok/miniconda3/envs/pjh-wamvla-v2/bin/python, 각20168MiB였다. 09:45/09:49 저장 OOM조회에도 같은PID가 등장하나 로그조회시각을 실시간 생존 확인으로 사용하지 않는다. 14:42 신규GPU조회에서는 두GPU에 우리 kjs-lpwm-stage2 PID866456/866457 각23558MiB만 있다. 기존10:52공동점유~14:34감소관측은 유지하며 정확한 pjh 종료시각/부재경과시간은 미확정. 학습/queue/등록source/config 변경없음. 근거 results/lpwm_card_budget_measured_v4/pjh_process_history_review_20261005.json.


## 2026-10-05 14:58 KST — 배치 확대 검토 중 공유 GPU 점유 복귀

사용자가GPU0·1여유활용을요청했다. 시작시Adapter4512/4707,배치8/누적1/GPU2=유효16,각24.74GB·100%사용이었다. 완료임박한학습은유지하고검증중빈GPU1에서full2095원본을복사해배치8/12/16및checkpoint재계산비교용스크립트를준비했다. 프로파일controller는sandbox NVML조회exit9에서실패했으며GPUchild는기동하지않았다. 호스트읽기전용재조회14:58에서pjh-wamvla-v2 PID290180/290181가GPU0·1각19522MiB로복귀한것을확인해실측을재시도하지않았다. 두스크립트는자동대기열에등록하지않았고Python구문검사만완료,속도/gradient동일성은미검증이다.

Adapter는4707/4707정상완료·기존queue가전체개발검증으로전환했다. 등록source42개및config전체hash보존. 기존full8우선profile/48GB감시/필요시4·2fallback/원본2095재개를유지한다. 유효배치16에서12·16perGPU로변경하면24·32가되므로학습조건변경임을사용자에게설명했으며실제로변경하지않았다. 근거results/lpwm_available_vram_throughput_20261005/capacity_review.json.


## 2026-10-05 15:22 KST — 현재 full low-LR 재개 학습 확인

사용자현재학습질문으로queue/progress/config/복구기록읽기전용조회. Adapter4707학습·전체개발검증이완료됐다. 다음full은15:12:12에기동,원본2095체크포인트의model과AdamW746state를보존재개해2192/4707진행. GPU당batch8profile이공유점유하에서24.96GiB allocator제한OOM으로실패했고프로파일가중치는폐기,정상통과한batch4×누적2×GPU2=유효16/SSL8을선택했다. LPWM전모듈1e-6와planner3e-4기본LR,기존planning+0.02SSL/GT보조OFF. 현재GPU전체각43.45GB,학습loss유한. 새학습이나실행조건변경없음. 근거results/lpwm_card_budget_measured_v4/full_resume_status_20261005_1522.json.


## 2026-10-05 15:29 KST — Adapter 최종 검증 보고

연구하위질문:미래표현을유지하면서planning에유용하게적응할수있는가? Adapter4707최종과LoRA/부분계층의동일1024개발장면(유효cachedPDMS1021/40recording),256worldclip을저장원자료로CPU집계했다. AdapterPDMS82.4852/ADE1.15513/FDE2.75672,LoRA대비PDMS+0.5711점95%CI[-0.3703,1.5020],부분계층대비+0.8511[-0.3869,2.2565]. PDMS우월성은미확정. LoRA대비FDE-0.08385mCI[-0.17458,-0.00570],ADE차이는CI에0포함.

미래LPIPS:Stage1.392249/partial.438393/LoRA.391846/Adapter.393758. AdapterStage1대비+0.385%로기존10%유지검사통과. 예측미래를관측반복으로바꾸면PDMS82.4852→79.6509,차이+2.8342[1.1428,4.5169]이나표현활용입력개입이며Adapter미세조정자체인과효과는아님. 기존scenario직진9/회전36/겹침197/기타14개world지표와대안큰회전flag등함께보고. 원거리객체flag211개에서전체영상미래오차+0.001943으로소폭악화,객체상태보존검사로해석하지않음.

NativeLPWM고정·adapter704960+planner명령2211975학습. 기록시간약4h10m/평가11m48s,공유GPU부하차이로기법속도이득확정금지. LoRA/Adapter초기후보1024모두일치·최종325개차이. 1epoch1seed/frozen-LPWM학습planner대조부재/공식navtest미실행한계유지. 재현scripts/report_lpwm_completed_adapter.py,결과results/lpwm_card_budget_measured_v4/completed_adapter_review_20261005/summary.json및comparison.png/pdf. 그래프육안검사완료·원자료token정렬/체크포인트출처검사통과. 활성full학습/queue/source42/config29불변.


## 2026-10-05 16:17 KST — 고정 LPWM과 동일 planner 대조군 자동 후속 등록

연구하위질문:LPWM표현미세조정이동일planner학습만하는것보다추가planning이득을주는가? 사용자현재학습후자동실행지시로새CPUqueue871940을기동했다. 기존v4queue421603/full학습은중단·수정하지않았다. 새상태waiting_for_full_training_and_validation,heartbeat정상,nodes비어있음으로GPU작업미기동확인. 선행4방법학습·검증·집계완료→GPU감사→batch8우선/48GB범위실측→engineering→대조군75297장면/1epoch4707→동일개발1024/256→기법별paired recording CI자동보고다.

Stage1의nativeLPWM전체가중치/buffer·encoder명령FiLM4480개고정/eval/no_grad,planner2207495개만학습한다. Ego명령은planner에유지. Seed47/동일초기planner·vocabulary·teacher·loss·LR3e-4·1epoch스케줄·유효planning16/world8·worker0·객체GT OFF. SSL0.02는detach된모니터라어떤가중치도갱신하지않는다. Adapter/LoRA와의대조는LPWM적응+FiLM+train/eval차이를포함하며full의conv_in/20epoch스케줄차이도남는것을등록했다.

실제체크포인트CPU감사에서초기planner및4종출력최대차이0,optimizer2step후LPWM·FiLM전체state와particle표현불변,world各gradient0/planner6.94398,SSL無gradient·미래입력독립성·planner명령반응통과. CPU첫검사비연속영상view오류는새고정모델SSL입력contiguous정리로해결후통과했다. 진단weights폐기. 프로토콜음성검사7개통과. 새등록source49/config33 hash확인,기존42/29도불변. 설정configs/lpwm_planning/frozen_control_v1/queue.json,검사results/lpwm_frozen_control_v1/queue/cpu_audit.json,기동근거registration_status.json. GPU검사/처리량은미측정이며선행작업종료후자동수행한다.

## 2026-10-05 16:25 KST — 현재 전체 미세조정 종료 예상

2026-10-05 16:25 KST: full low-LR 학습 2864/4707 (60.85%), 남은 1843 update. 최근 128/256/512 update의 경과시간 기준 약 5.61–5.63s/update; 학습 종료 19:15–19:40, 최종 개발 검증 종료 19:30–20:10 KST 예상.

최근 log의 elapsed_seconds 차분을 사용해 중간 검증·저장 시간을 포함했다. 이전 개발 평가 실측은 partial 19.9분, LoRA 28.4분, Adapter 11.8분이다. 추정 범위는 공유 GPU 부하 변동을 고려한 계획값이며 보장/통계적 신뢰구간이 아니다. Main queue는 학습 중, frozen 대조군 queue는 선행 학습·검증·집계 완료를 기다린다. 대조군 완료 ETA는 GPU profile 후 계산한다.

근거: `results/lpwm_card_budget_measured_v4/full_training_eta_20261005_1624.json`. 학습·queue·source/config 변경 없이 CPU 로그 확인과 기록만 수행했다.

## 2026-10-05 16:31 KST — 전체 LPWM end-to-end 학습 범위 확인

사용자가 현재 모든 LPWM 가중치를 end-to-end 갱신하는지 질문했다. Full condition의 실제 inventory에서 전체/학습 가능 파라미터가 모두111,757,238개이며 LPWM109,545,263개와 planner+command2,211,975개다. Native LPWM 전체 requires_grad=True 및 optimizer 포함을 코드에서 확인했고, 최근 주기적 모듈별 gradient 검사(128 update마다 측정, 중간 log에는 직전 측정값 유지)도 모두 양수였다. Planning loss는 particle memory를 통해 encoder/context/dynamics로 전달되고 RGB decoder는 SSL reconstruction 경로에서 갱신된다. 기본LR LPWM1e-6/planner3e-4, imitation+metric BCE+0.02SSL, 객체GT보조OFF 유지. 모든 scalar의 매 step 변화 여부까지 새로 검증한 것은 아니다.

근거: `results/lpwm_card_budget_measured_v4/full_parameter_scope_20261005.json`. 학습·대기열 변경 없이 기존 full 완료/검증 후 frozen 대조군 자동 실행을 유지한다.

## 2026-10-05 16:47 KST — DrivoR register와 LPWM particle 비교

DrivoR §3.2–3.5/§4.2.1 및 공식 fc6e5aa의 register/ego 경로, LPWM A.3–A.4와 현재 planner 코드를 확인했다. Register도 planning에 필요한 영역으로 특화되므로 객체/중요정보 이해 불가로 설명하지 않는다. LPWM의 구조화된 속성·명시적 미래 전이·SSL과 명령 FiLM 차이를 정리했다. Particle는 patch identity이고 depth는 합성 순서이며 객체 추적/물리3D 보장이 없다. 현재 memory768 대 DrivoR기본64, 전방저해상도 대4카메라, 고정512후보 대연속생성 차이로 동일조건 단순교체 비교가 아니다. Fixed64/future8 전부를 예측하므로 원래 상황별 예측 예산 선택 목표는 아직 미구현이다.

자료와 상세: `docs/lpwm_planning_experiment.md` 최신절. 기존 full 학습과 후속 frozen 대조군 유지. 추가 register 비교나 미래/명령 ablation은 제안만 기록하고 자동 등록하지 않았다.

## 2026-10-05 16:55 KST — 표현 비교의 공통 planner 통제 원칙 정정

사용자가 DrivoR과 같은 planner를 두어야 표현 비교가 성립한다고 지적했다. 공통 planner 구조/초기값/학습 규칙을 두고 각 조건을 별도 학습하는 기준을 연구 문서에 반영했다. Planner 가중치 freeze와 구분했으며, 입력 센서/해상도/관측 이력/token예산/학습목표/gradient/평가까지 통제하도록 했다. 공식 코드의 trajectory/scoring decoder·detach 경로를 확인했다. DrivoR vs LPWM은 backbone과 사전학습 차이가 남으므로 시스템 비교이며, particle 구조 단독효과는 같은 기반의 일반 latent dynamics 비교가 필요하다. 미래·encoder intent 조건도 별도 ablation으로 분리한다.

상세: `docs/lpwm_planning_experiment.md` 최신절. 기존 full/frozen은 LPWM 내부 적응 비교로 해석하며 register 우월성 비교로 사용하지 않는다. 공통 DrivoR 구현/새 학습은 시작하지 않았고 기존 full→검증→frozen 대기열과 source/config를 유지했다.

## 2026-10-05 17:02 KST — 현재 LPWM 비교 완료 우선, DrivoR 대조실험은 대기열 없이 보관

사용자는 현재 “같은 LPWM 기반에서 표현까지 planning에 맞춰 수정하면 추가 이득이 있는가” 실험을 먼저 마무리하도록 했다. 현재 full 학습·검증과 이미 승인된 frozen LPWM/encoder FiLM+동일 planner 대조군 학습·검증·비교는 이어간다. 후속 DrivoR 대조실험의 목적은 “주행 의도에 맞춰 구조화된 particle의 미래 정보를 보존하는 것이 일반적인 압축 feature보다 planning에 도움이 되는가”다. 공통 DrivoR planner 통제 원칙을 유지한다.

**명시적 지시: DrivoR 비교는 지금 대기열에 등록하지 말고 후속 과제로 기억한다.** 현재 작업 완료를 trigger로 새 DrivoR profile/학습/평가를 자동 기동하지 않는다. 기존 main queue 학습/frozen queue 대기 heartbeat 정상 확인, source/config/queue 변경 없음. 연구 문서 및 HANDOFF에 우선순위와 실행 제한을 기록했다.

## 2026-10-05 18:35 KST — frozen-LPWM planner 대조군 자동 연결 재확인

사용자가 현재 학습 완료 후 LPWM 완전 고정·planner만 학습하는 실험이 이어지는지 확인했다. 실제 main queue full_low_learning_rate_training/frozen queue waiting_for_full_training_and_validation, 양쪽 heartbeat 약7초 이내 정상이다. 등록 sequence는 현재 학습/최종검증/집계 완료 후 GPU freeze 감사와 batch profile/평가구동확인을 거쳐 planner 학습/검증/paired 보고로 연결된다.

대조군은 Stage1 완료 LPWM에서 시작하고 encoder 명령 FiLM까지 eval/no_grad 및 parameters/buffers 불변으로 고정한다. Planner는 같은 seed47 초기값으로 새로 학습하며 현재 Stage2 최종 가중치를 대조군에 넘기지 않는다. 기존 source/config/queue 변경 없음. DrivoR은 후속 과제로만 보관하고 대기열 미등록을 유지한다.

## 2026-10-05 20:27 KST — 네 미세조정 완료, frozen 대조군 학습 중

2026-10-05 20:27 KST: 네 가지 미세조정 학습·검증 완료. Full 학습19:06:33/검증19:16:44/집계19:16:48 완료. Frozen 대조군은19:18:08 자동 시작, 현재3616/4707 (76.8%) planner만 학습 중.

GPUfreeze감사·배치8profile·평가구동검사통과후GPU0·1에서batch8×누적1×2=유효16으로학습중이다. LPWM가중치와encoderFiLM고정,trainableworld0/planner2,207,495개이며주기적gradient검사LPWM0/planner양수다. Frozen 최종검증은미실행으로engineering결과를학습후성능으로보고하지않는다. 최근속도와기존평가시간으로학습20:45–20:55/최종검증·비교21:00–21:20KST예상이며공유부하에따라변동가능하다.

네방법내부개발PDMS partial81.6341/LoRA81.9141/Adapter82.4852/full81.2282. Full의5개등록검사는통과,partial미래LPIPS유지기준미달은보존한다. 전체navtest/수렴결과가아니다. 기존main complete/frozen running을확인했고source/config/queue변경없음. DrivoR미등록유지. 상태근거 `results/lpwm_frozen_control_v1/status_20261005_2025.json`.

## 2026-10-05 20:39 KST — 네 미세조정 결과 비교 완료

사용자요청에저장1024planning/256world원자료를CPU재집계,scene/recording일치·checkpoint provenance·4707완료를확인하고6쌍paired CI의기존집계와정확동일성을검사했다. Adapter평균planning최상/LoRA미래영상유지최상이나PDMS6쌍모두CI에0포함한다. Partial미래LPIPS+11.764%로유지기준미달,LoRA-0.103/Adapter+0.385/full+0.802%다. Full복원성능최상이나planning최상은아니다. Full20epoch스케줄유지로종료plannerLR2.989e-4 대나머지3e-5차이를확인해단독방법인과결론을제한했다.

Scenario별world9직진/36회전/197투영겹침/14other와위험분해를보존했다. 학습비용은partial4h17/LoRA6h19/Adapter4h09/full누적8h25이며공유부하비교한계를명시했다. `scripts/report_lpwm_four_adaptation_results.py`, `results/lpwm_card_budget_measured_v4/completed_four_method_review_20261005/`의JSON/PNG/PDF와연구문서를작성했다. 최초그림확인후legend겹침을해결했다. Frozen학습은진행중이며DrivoR미등록유지,새GPU작업없음.

## 2026-10-05 20:46 KST — Frozen 대조군 메모리 감소 확인 및 학습 완료

사용자가낮은VRAM사용량을관찰했다. LPWM전체/encoderFiLM고정및no_grad추론,planner2,207,495개만역전파함을확인했다. 학습peakallocated3.906GiB로full20.735GiB보다작고배치8×GPU2×누적1=16,75297장면/4707update/1epoch를유지했다. 전체GPU점유와PyTorch할당메모리,타사용자포함GPU연산사용률은구분한다.

확인중20:45:37학습정상종료(returncode0),최종training_summary frozen_state_unchanged=true와checkpointSHA9a62eb85d4ff41874cb5baac8127ba929391a57dc189235b936d3c1d37bb72ee확인. 누적5226.38초약1시간27분,검증PID1766544자동시작. 최종1024/256평가는진행중이며128monitor83.389PDMS를최종결과로보고하지않는다. 공유summary: `results/lpwm_frozen_control_v1/batch8/metric_plus_world_training_summary.json`. 새학습/배치변경/DrivoR기동없음.

## 2026-10-05 21:04 KST — Frozen 최종검증과 네 미세조정 대비 비교 완료

사용자 질문은 같은 LPWM 기반에서 planner 학습에 더해 표현까지 planning에 맞춰 수정하면 추가 이득이 있는지다. Frozen 검증20:56:12/집계20:56:16 정상완료,main/frozen queue 모두complete 및 해당학습·평가PID종료확인. 최종체크포인트의LPWM/encoderFiLM가중치와persistent buffer hash초기값일치검증통과.

Frozen PDMS82.5238/ADE1.163467/FDE2.782463. Partial/LoRA/Adapter/full minus frozen PDMS점 차이 각각 -0.8897[-2.2018,+0.2507],-0.6097[-1.4261,+0.1113],-0.0386[-1.1642,+1.0071],-1.2956[-3.2034,+0.3145]. 모든95%CI가0포함하여이번설정의표현미세조정추가이득미확인. 대조군우월성/동등성/미세조정일반적무효도입증아님. Adapter ADE차이-0.00834m CI[-0.04080,+0.02423]도불확실. Partial/Full은ADE/FDE차이구간양수. Partial미래LPIPS+11.764%유지기준실패보존.

Frozen에서도예측미래→마지막관측반복교체시PDMS82.5238→79.7709,차이+2.7529[1.7547,3.8131]점. 기존미래표현활용진단이며표현미세조정효과/객체보존/재학습no-future비교와구분. Frozen이미지LPIPS는Stage1과평균약1e-7차이. 평가1024장면/40recording/world256정확일치,PDMS공통teacher무효3개제외1021,1epoch4707/75297train/seed47. PDMS는변형없는고정후보의cache된공식PDM이며전체navtest아님. Frozen모드/FiLM,full명령위치/20epoch스케줄및microbatch이력차이를명시했다.

새 CPU script `scripts/report_lpwm_adaptation_vs_frozen.py`에서 원자료/공통누락/hash/provenance를확인하고평균·4pairedCI원집계와정확일치검사통과. `results/lpwm_frozen_control_v1/completed_comparison_20261005/{summary.json,comparison.png,comparison.pdf}` 생성/이미지확인,직진9/회전36/투영겹침197/other14 영상오차분해포함. 최종자동집계와trend summary공유보존,연구문서최상단/HANDOFF1–5갱신. 등록source/config/학습조건변경없음. DrivoR추가실행/seed·epoch/navtest자동등록없음.

## 2026-10-05 21:36 KST — 공개 LPWM 고정 조건 시작 및 배치 변경 이력 감사

사용자가Stage1없이공개LPWM을고정해planner만학습하는조건을요청했다. 배치확대제안에두조건재학습을제안했으나사용자가82.52완료대조군재사용/같은조건하나만실행으로정정했다. 실제로새공개조건하나만GPU당8×누적1×2=유효16,75,297장면/seed47/1epoch4707update로등록했다. 초기공개weight SHA6d62bf5a2f8977c8e4cea10250ac73fea4dbe61dad997607e7df959a0a9aa731,Stage1cache는동일입력준비용이며NAVSIM적응weight는로드하지않는다.

실제CPU2step검사/GPU검사에서공개weights/buffers완전동일,Stage1고정조건과초기planner정확일치,optimizer후LPWM/FiLM/particle불변,plannergradient만양수/명령반응/미래보조입력독립을확인했다. 7개조건변경음성검사통과. Tool sandbox detach자식이종료돼첫기동은본학습이없었으며권한승인후host queue1869615정상기동. GPU감사/배치8실측최대전체26.36GB/평가구동검사통과후본학습21:35현재304/4707,약0.93s/update. 최종평가와기존고정군대비paired비교/PNG/PDF자동연결,추가DrivoR/GT/navtest없음.

사용자가과거배치변경때문에기존실험이무효인지질문했다. 최초config/resume parent/최종summary/전체training_log에서모두유효planning16/world8/4707update유지를확인했다. Partial4×2→8×1(update1576),LoRA4×2→8×1(4422),Full2×4→4×2(2095),Adapter/Frozen8×1전구간이며GPU2개다. Loss/누적횟수로backward하고누적뒤clip/optimizerstep하는코드도확인했다. 그러나dropout RNG/SSL보조clip추출/부동소수점누적차이와full20epoch LR/conv_in차이는남는다. 기존결과는경향비교로보존하되기법단독우월성·완전실행동일성으로주장하지않는다. `results/lpwm_card_budget_measured_v4/batch_history_audit_20261005.json`과연구문서에정정했다.

사용자가남은VRAM을활용한가속가능성을추가질문하여큰배치의고정표현계산과detached SSL monitor 계산생략의gradient동일성·속도진단을준비했다. 본학습은유지하며읽기전용/optimizerupdate0/진단10GiB상한/카드48GB보호다. 결과확인전실행조건을바꾸지않는다.

## 2026-10-05 21:44 KST — 고정 표현 가속 진단 완료 및 결과 ETA

일반sandbox의CUDA접근차단후승인된host실행으로32scene/batch8·16·32의고정표현계산을확인했다. 동일seed반복시간3.199/2.942/2.785초로약8–13%단축이나batch8대비혼합particle최대차이1.464/1.216,같은batch반복에서는0이었다. 물리batch출력동일성이성립하지않으므로큰batchcache미채택. 이값은물리객체오차가아니며정확원인/PDMS영향은미분리. Detached worldmonitor생략의planninglogits는동일,gradient최대차이0.000488은동일조건반복에서도관측됐다. 단일batch속도2.292→0.925초이나공유GPU부하측정이며장기간동등실행미검증으로현재등록monitor를유지했다. 두진단결과모두보존하고추가GPU진단종료,본학습/source/config/optimizer는바꾸지않았다.

현재본학습768/4707,최근128/256/512update1.127/1.069/1.054s,queueheartbeat정상. 사용자에게학습22:55–23:10,최종결과23:05–23:25KST예상이라고보고했다. 근거 `results/lpwm_stage1_effect_v1/frozen_batching_repeatability_benchmark.json`, `training_eta_20261005_2144.json`. 학습종료후동일1024/256평가및기존82.5238고정대조군대비Stage1효과자동비교가이어진다.

## 2026-10-05 21:58 KST — Adapter 대 고정 LPWM particle 분포 시각화

사용자 요청에 따라 완료된 NAVSIM Stage1 적응 고정군과 Adapter군의 동일 개발 8장면을 비교했다.
현재 학습 중인 공개 LPWM 고정 조건과 구분한다. 기존 0·2353·4707 update의 48개 snapshot만 CPU로 읽고
새 추론·학습 없이 `scripts/visualize_lpwm_adapter_vs_frozen_particles.py`로 PNG12개/PDF3개/HTML/JSON을 생성했다.
저장 위치는 `results/lpwm_adapter_vs_frozen_particles_20261005/`이며 독립 HTML에서 8장면·학습시점 선택,
박스/인덱스/presence 표시 전환, 이미지 클릭 후 동일 영역 확대와 실제 크기의 중심 겹침을 제공한다.

입력 RGB·초기 중심/크기/presence 정확일치, 고정군 세 시점 geometry 정확불변, 모든 수치 유효성을 검사했다.
전체64개 점과 고정군 초기 presence 상위16개 인덱스의 박스를 양쪽·세 시점에 동일 적용했다.
8장면512particle 최종 중심 이동 평균0.084663/중앙값0.070458/p950.207563/최대0.347623px로 모두1px미만,
크기 평균절대차0.271845px,presence 평균변화+0.003996/평균절대차0.004925다. 단위는128×128입력픽셀이다.
뚜렷한 객체방향 재배치나 정보보존 개선은 이 그림에서 입증되지 않는다. Particle ID는 객체track이 아니며
박스는 learned support,presence는 planning중요도가 아니다. Adapter조건에 명령FiLM·SSL유지도 포함되어
planning loss 또는 adapter 단독 인과효과로 해석하지 않는다. Native 위치head고정/후단adapter구조도 기록했다.
Snapshot은관측4프레임만의FP32단일장면진단이며 BF16배치평가/미래particle이동/latentfeature비교가 아니다.

Python·HTML내JavaScript구문 검사와 대표비교/변화량 PNG 육안확인을 완료했다. 원본snapshot의 SHA48개를 남겼다.
현재public_control_training은1584/4707,heartbeat정상으로 계속되고 등록source/config/모델/대기열은변경하지않았다.

## 2026-10-05 22:06 KST — planning 미세조정 효과 해석 재검토

사용자가 particle이 거의 이동하지 않으니 planning loss 미세조정이 무효인지 질문했다. 코드·감사·paired평가를 읽어 확인했다.
Adapter는 native CNN/geometryhead고정, interaction1/context4/dynamics6 residualadapter+명령FiLM+planner학습이다.
공식encoder의interaction출력은feature/depth를바꾸며현재위치/크기를덮어쓰지않고interaction_obj_on=False다.
따라서현재geometry변화는주로FiLM경로이며후단표현학습을현재점이동만으로검증할수없다.
Planning-only기존감사gradient는encoder그룹(interaction포함).030906/context.020353/dynamics.225220으로연결확인.
이는nativeCNN/좌표head가학습됐거나일반화성능이개선됐다는뜻은아니다.
AdapterPDMS82.4852/frozen82.5238,차이−.03859점CI[−1.16421,+1.00706],ADE차이−.00834mCI[−.04080,+.02423].
이번1seed1epoch개발비교에서추가이득미확인으로판정하며일반적무효/동등성은주장하지않는다.
후속은현재·미래속성별변화/동일상태planning대가중SSLgradient/Adapter·FiLM개입/geometryhead해제통제제안이다.
기존planning과SSL감사norm은시점·모드가달라그비율로SSL지배를확정하지않는다. 추가GPU진단/학습/대기열등록은없다.
현재공개고정queue1869615 public_control_training update1904/heartbeat정상확인,등록source/config유지.

## 2026-10-05 22:20 KST — 일부 계층·전체 미세조정의 particle 위치 시각화

사용자 요청으로 두 조건의 particle 배치가 함께 planning fine-tuning됐는지 확인하고 같은8장면 학습전0/중2353/후4707을 CPU로 시각화했다. Full초기snapshot은원래v3실행에서읽고중후는2095checkpoint재개실행을사용했다. RGB/초기geometry정확일치와frozen3시점불변,72개snapshot 수치유효성/SHA를확인했다.

새 script `scripts/visualize_lpwm_geometry_finetuning.py`는 기존그림helper를재사용하고실제Stage1·최종checkpoint를CPU mmap으로읽어xy_head/scale_xy_head/obj_on_head의trainable등록과가중치변화를검사한다. 두조건3head전부실제로바뀌었다. Planning-only기존감사의encoder그룹gradient도partial1.26456/full1.81744로양수다. Head별loss기여율을별도측정한것은아니다.

최종512particle중심이동 partial평균0.550627/중앙값0.524997/p951.019065/최대1.713751px,full평균0.460710/중앙값0.330322/p951.219635/최대2.529015px. 1px이상이동 partial5.46875%/full10.15625%,크기평균절대차1.205988/1.576844px,presence평균절대차.014118/.021439다. 이전Adapter중심평균.084663보다크지만방법별학습률·스케줄·FiLM·배치이력차이도있다.

결과 `results/lpwm_geometry_finetuning_visualization_20261005/`: PNG13/PDF4/독립확대HTML/summaryJSON. 고정기준공통16box인덱스/전체64점,학습전중후·8장면전부·실제크기위치겹침을제공한다. Python·JS구문과대표비교PNG육안검사완료. Planning+0.02SSL·정규화의합산변화로planning단독원인/주요객체재배치/추가PDMS이득은입증하지않는다. 현재공개고정본학습2784/4707·heartbeat정상이며GPU/등록source/config변경없음.

## 2026-10-05 22:30 KST — planning 신호 강화 제안과 기존 clipping 감사

사용자가복원사전학습으로배경particle이많은것같으니planningloss신호강화를테스트하면어떨지질문했다. 현재L_plan+.02SSL과globalclip5→AdamW코드를확인하고기존로그만읽었다. 마지막gradient값을재사용하는로그를제외하고update1/128배수실측기록만집계했다. Partial37/37(중앙norm87.96),Adapter31/37,full재개20/20이clip5를넘었다. `results/lpwm_planning_loss_balance_review_20261005/existing_training_audit.json`에기록했다. Loss값은planning약6/가중SSL약.35이나gradient기여를뜻하지않고SSL지배/충돌도미확정이다.

후속설계:같은모델상태·trainingminibatch에서planning/가중SSL의geometryhead별norm/cosine/clip/실제update분리진단후,geometryhead가열린partial구조고정으로planning1·SSL.02/.02÷3/.002(상대계수1/3/10배)을비교한다. LPWMLR/학습가능계층은함께바꾸지않는다. 완료partial4707+AdamW에서동일분기해모두같은추가navtrain1epoch/배치8×2/seed/샘플순서/공통schedule로비교하는후속적응안을제안했다. Baseline도같은추가학습을해야하며기존완료점수로대신하지않는다. 공통schedule구체값·분기SHA는실행전등록사항으로남겼다. Freeze대비새학습량효과를주장하려면대응frozen추가학습도필요하다.

동일개발평가/particle·feature·미래변화/개입효용/world유지를검증하도록제안했다. 배경이무조건불필요하거나particle이동량자체가성공기준이라고보지않는다. 이번에는새backward/재학습/queue등록을하지않았고기존공개고정3184/4707·heartbeat정상확인후유지했다.


## 2026-10-05 22:49 KST — DrivoR 학습 절차의 논문·코드 확인

사용자 요청으로 Driving on Registers v2와 공식 fc6e5aa를 읽었다. 외부 DINOv2 사전학습 뒤 NAVSIM에서는 단일 공동 학습이며, 논문의 Stage1/2는 NAVSIM-v2 평가 단계다. 원래 ViT는 고정하고 Q/V LoRA32·새 카메라 register·trajectory decoder·scoring decoder를 같은 optimizer로 학습한다. WTA L1+6개 oracle BCE가 기본이며 별도 복원SSL/직접 객체loss는 없다. 후보 좌표→scorer의 detach와 두 loss→공유 perception 경로를 확인했다. 안전 BCE는 후보 좌표를 직접 교정하지 않는다.

최종v1 navtrain+navval25epoch/v2 navtrain10epoch, README batch16×4·AdamW2e-4와 warmup/cosine 코드를 확인했다. run_training_full non-cache 경로가 train+val을 합치고 cache-only는 train만 쓰는 차이도 기록했다. Register와 particle/미래world model의 차이 때문에 DrivoR 결과로 LPWM Stage1/SSL 불필요를 단정하지 않는다. 상세 docs/lpwm_planning_experiment.md 최상단.

현재 공개 LPWM 고정 학습은22:44에4192/4707·queue1869615 heartbeat정상이었다. 이번에는 문서만 갱신했고 활성 source/config/GPU조건/queue를 바꾸거나 DrivoR/새학습을 실행하지 않았다.


## 2026-10-05 22:54 KST — 공개 고정 대조군 완료 예상 재확인

사용자 ETA 질문에 queue/progress/최종monitor를 읽었다. 22:52에는4704였고22:53에는4707/4707 optimizer update 및128scene monitor완료, queue heartbeat정상이었다. 최종저장·검사·process종료와후속개발평가는별도확인해야한다. 직전동일평가634.823초를근거로1024planning/256world평가와Stage1효과paired비교까지23:05–23:15KST예상으로갱신했다. 정확snapshot results/lpwm_stage1_effect_v1/training_eta_20261005_2253.json. 실행설정/GPU/queue는변경하지않았다.

같은턴22:54추가확인:학습은22:53:37 returncode0로종료했고training_summary가저장됐다. 22:53:38 public_control_evaluation(PID2514642)이자동시작됐다. 현재최종검증단계이며예상완료23:05–23:15KST를유지한다.


## 2026-10-05 23:15 KST — NAVSIM Stage1 효과 대조군 최종 검증 보고

최종검증23:11:21,paired비교23:11:28 정상종료/queue complete를확인했다. 공개고정78.9160799→적응고정82.5237848 PDMS,+3.6077049점CI[+1.5064775,+5.8538573]. ADE1.3728555→1.1634666/FDE3.3423478→2.7824634,복원LPIPS.7672489→.3043349/미래.8044382→.3922484. 같은planner초기화·학습조건/seed47/planner1epoch4707/75,297train/batch8×2를유지했고기존82.52군재학습없다. 두조건LPWM/FiLM고정과공개최종LPWM의원본weight/buffer동일성검사통과.

원자료8SHA·동일1024장면순서/40recording/동일3PDM누락/256world/평균재집계를독립확인하고비교PNG육안검사했다. 시나리오·7개risk별미래영상LPIPS도개선이나객체별정보보존/상황별PDMS증거는아니다. 내부개발1seed1epoch조건의Stage1이득이며navtest/seed불확실성/수렴은미검증. 기존Stage2표현미세조정추가이득미확인결론과구분한다. 결과results/lpwm_stage1_effect_v1/queue/stage1_effect_summary.json 및비교PDF/PNG. README/연구문서/인수인계갱신,모든등록작업종료,새GPU/후속queue없음.


## 2026-10-06 00:13 KST — 공개 LPWM + DrivoR joint 본 학습 및 공식 평가 대기열

사용자가 Stage1/2를 합쳐 planning 중심으로 학습하고 official DrivoR backend/data/loss를 활용하라고 새로 승인했다. 이 범위의 과거 DrivoR/추가epoch/navtest 금지를 갱신했다. 공개 Sketchy LPWM에서 native encoder/xy/scale/presence/context prior/dynamics를 LR2e-5로 갱신하며 새 official DrivoR planner/commandFiLM/projection은2e-4. SSL/RGBdecoder/직접객체aux없음;4현재카메라128square→현재+8prior→native64particles고정4개pool→64scene tokens. 공식 generator4/scorer4/64연속후보/8poses, 원본 WTA L1+6BCE 및proposaldetach유지. Source DrivoRfc6e5aa, LPWM4cf53c4, officialNAVSIMv2.2 359c7f7.

v1공식train85,109+val18,179/25epoch40,350updates, seed2/effective64=8×4×2. 23:56본학습parent2788260,queue2839064는v1fullnavtest후v2별도public-inittrain85,109/10epoch13,300→warmup/navhardEPDMS. 검사단계단일batch8최대44.15GB,DDP/본학습약45.26GB,48decimalGB내. 조회24update완료;최근37.3초/update이면v1약17.4일잔여,완료시각확정아님. 등록source/config는불변이다.

Gradientroute두loss모두geometry/representation도달,scorerBCE의coord/generatorgradient없음;GT/longer공식오차<3.1e-7,2장면cache변환vsfresh공식7score차이0. 2GPU/누적/체크포인트저장2update engineering가중치는본학습에미사용. 공식v1evalscorer실행확인,officialv2warmup220cache/두단계집계0누락검사완료(고정0trajectory,모델성능아님). Navtest12,146/current4camera입력완료,warmup220완료,navhard5,912입력준비중.

샌드박스DNS로pip/GitHub최초실패했으나권한재시도로성공,timm1.0.15/piqa1.3.2전용venv설치. 샌드박스async파일저장대기는trainingcache atomic local I/O로대체(직렬화/computation동일),중단된불완전freshauditcache보존. 지도1.4GB workspace사본으로공용원본write방지. 사용자질문에DrivoR가DINOv2pretrained+Q/VLoRA32이고모든가중치scratch아님을코드로확인했다. LPWM은active nativeFT이므로미세조정/해상도/증강/초기분포까지동일하지않음을명시했다. 성능향상/정확한미래의미/순수register-vs-particle효과는아직미검증. 상세docs/lpwm_drivor_joint_training.md, 공유results/lpwm_drivor_joint_v1/.


## 2026-10-06 00:49 KST — DrivoR 방식 LPWM LoRA, ego 동일성 감사, batch16 재개

사용자요청에따라 native joint35update checkpoint를보존중단하고 public Sketchy에서 Q/V rank32 LoRA로전환했다.
기존 native109,545,263 parameter/buffer고정, LoRA1,343,488, 전체trainable18,413,566.
초기출력/native출력및planner초기동일,2step후frozen digest보존, LoRA/FiLM/planner gradient·변경/DDP검사통과.
공식 DrivoRFeatureBuilder와 navtrain8+navval8 ego11D 입력비교오차0;공식generator/scorer주입동일.
추가LPWM command4D FiLM은공식DINO에없으며향후register통제비교에서분리해야하는차이로문서화.

LoRA1update 후사용자VRAM활용요청으로 model+AdamW334state step1/scheduler/RNG보존하고 execution-only증설실측.
Batch8 31.56초/16.30GB → batch16(2loader/4oracle)18.29초/29.59GB.
Batch24(4loader/8oracle)19.35초/43.00GB, batch16(4loader/8oracle)19.93초/29.59GB.
짧은공유자원진단이며본학습속도보장아님. 배치32는48GB초과예측으로실행하지않음.
선택batch16+16/GPU2=effective64, 데이터/loss/LR/epoch불변;microbatch/dropout분할은변경됨.
본학습2994997/queue2994998로재개,조회시update6/40350,loss유한/총VRAM29.59GB.
후속v1전체navtest→독립publicv2navtrain10epoch→warmup/navhardEPDMS동일override자동연결.
기존등록source/config불변,새parallelism_registry까지hash재검사통과. 공식PDMS/EPDMS아직없음.
새보고서 docs/lpwm_drivor_lora_training.md, 실행증거 results/lpwm_drivor_lora_v1/.


## 2026-10-06 01:30 KST — 학습 중 particle 유용성 진단 및 DrivoR 비교 감사

사용자 요청: 학습하면서 particle의 driving/intent 유용성과 Register–Particle 공정 비교를 함께 검토.
기존 LoRA 본학습2994997/queue2994998 및 등록 source/config는 변경하지 않았다.
별도 읽기 전용 monitor3351138이 저장 checkpoint를 진단한다. Root outputs/lpwm_drivor_representation_monitor_v1.
GPU0/allocator4GiB상한/카드40GB미만 진입·46.5GB중단, main pause 존중. 추가 학습/DrivoR baseline 큐 없음.

- Panel: 모델 결과를 보지 않고 고정한24recording×4=96trainval장면(직진42/좌30/우24), 객체-camera관측2974.
- 모든64particle/camera의 GT투영 분포·presence·면적정규화, 평면지도 도로proxy3높이, 학습전후좌표/feature 변화.
- Readout fit18/eval6recording; 둘 다 upstream train분포. 차량/보행자/자전거 종류,2s/4s GT track displacement.
- 동일12장면 encoder-only/planner-only/both command변경, 미래반복/시간반전/관련particle/카메라·presence·크기대조개입.
- Native memory복원→proposal/고정후보score bitwise동일성, 원본native SHA동일성 확인. CPU검사4개통과.
- 초기0update 96scene288.7초/100update257.0초, peakreserved0.654GB. 초기원본등록파일을 run별보존 후watch등록sealed.
- 100update 중심 평균0.00305px/최대0.05446px이동, 차량/보행자/도로분포 거의같음.
- Encoder-only 명령변경 궤적0→0.2834m: 의도 경로는작동. 대안GT없으므로 적합성/정확성을뜻하지않음.
- 전체14D 클래스F1 .3799→.3778, 미래2s/4s readout3.8933→3.9021m/8.0206→8.0322m;정보보존개선미확인.
- 미래-minus-current readout CI와미래순서/반복planning개입 CI는0포함. 관련particle개입은5scene뿐,성공증거로채택하지않음.
- 전체40,350update중100은0.25%warmup이므로실패판정금지. 500/1000/2000간격·epoch부근·final자동추적.

공식DrivoR fc6e5aa 코드/논문AppendixD 대조. 동일planner/loss/split/epoch/effective64이나 순수 표현비교는아님.
해상도672×1148vs128², DINO사전학습vsSketchy, GridMask/정규화,encoder명령FiLM유무,미래추가연산차이.
실제Q/VLoRA rank32 파라미터589824vs1343488 확인. Default FP16 vsBF16+encoderFP32,
drop_last True/False 및backend생성순서RNG 차이도추가발견. 표준DDP/공식4GPUrecipe유도updates40325vs40350(v1),13290vs13300(v2).
기존 '동일protocol' 표현은split/epoch/기본optimizer수준으로한정. 시스템benchmark/엄격통제재학습/LPWM intent×future ablation 분리설계.
DrivoR register도planning으로학습되는표현이며일반압축=planning무관이라고가정하지않음.
문서 docs/lpwm_drivor_representation_and_fair_comparison.md, 공유수치 results/lpwm_drivor_representation_monitor_v1/.
학습분포진단과독립navtest/EPDMS 구분, 도로proxy/투영bbox/가림proxy의한계명시. 기존공식평가큐유지.


## 2026-10-06 — Particle 위치를 집중 영역으로 해석할 수 있는가

Native LPWM feature encoder의 position/scale 기반 RGB spatial_transform crop 및 interaction 경로를 읽었다.
Particle 점은 지역 시각 표현의 중심이며 planner attention/주행 중요도 그 자체가 아니다.
최신 monitor는 노란점=현재중심, 점크기=presence, 색box=평가용GT투영이다. Box를glimpse/detection으로해석하지않는다.
Current+future attributes에는 공유background feature도 포함되며 projection후4particle씩평균하여planner에전달한다.
하늘위점→하늘만사용, 보행자위점→정지판단기여 등의추론은성립하지않는다. Scale/readout/개입으로보완한다.
설명과문서정정만수행;본학습·monitor·등록source/config·queue변경없음.


## 2026-10-06 — LoRA 학습과 현재 particle 위치 경로 원인 확인

사용자질문: planning LoRA인데왜현재particle위치가같고학습이정상인가.
등록본학습/monitor/queue를유지하고100update snapshot(64e304f0...)을읽는독립진단을실행했다.
새script audit_lpwm_drivor_lora_geometry.py, 결과results/lpwm_drivor_lora_geometry_audit_v1/report.json.
직진/좌회전/우회전각1장면×4cam, optimizer0, peakreserved0.736GB/카드총30.683GB. 원본freeze/checkpoint SHA불변.
LoRA84/84tensor갱신. 실제main100update gradient interaction/context/dynamics .03895/.04668/.06041,FiLM.14440,두rank동일.
LoRA OFF current xy bitwise동일;FiLM OFF initial xy bitwise동일;둘OFF 전체attrs initial과동일.
현재좌표projection autograd에서LoRA84개는모두None,FiLM4개연결(.80924). 이는planningloss와별개의연결성검사.
LoRA OFF future xy 성분RMS.492px변화는의존성증거이며성능향상아님.
원인: 현재Q/V LoRA는좌표생성뒤interaction/context/dynamics만대상. CNN/xyhead에는없어현재위치를직접바꾸지못함.
새commandFiLM만현재geometry에연결돼작은변화가생김. 초기warmup만으로위치불변을설명했던해석보완.
수정방향은xy_head/scale_xy_head/obj_on_head의Linear LoRA(출력차원에맞는rank),필요시CNN ConvLoRA.
현재설정중간변경/추가학습은하지않았다. 기존실험은attention-onlyfeature/future적응조건으로보존한다.
보고서docs/lpwm_drivor_lora_geometry_audit.md와기존LoRA/표현검증문서에한계를추가했다.


## 2026-10-06 08:55 KST — LoRA 본학습 진행 및 ETA 점검

요청: 현재학습진행과종료예상. 학습·설정·queue변경없이상태만읽었다.
Host train2994997/queue2994998/monitor3351138생존. Sandbox ps로안보이는것은namespace이며host확인완료.
진행1400/40350(3.47%),epoch1 86.74%,batch16×accum2×2=effective64,loader2/oracle4perrank.
GPU0·1각29.59decimalGB/util82·100%조회. 최신1400checkpoint08:54:43저장. Active pause/error완료marker없음.
양rank1400로그NaN/Inf0,모든gradient그룹양수. First100→last100 total28.0402→4.9325,trajectory24.3438→2.4956,scoreloss3.6963→2.4368.
Recent100 20.0415sec/update,300 mean20.6659/wall20.7067. Wall기준v1 ETA10/15 16:57KST(약9.3일잔여),epoch1오늘10:09KST.
공유부하에따라변동하며navtest평가·v2독립10epoch·EPDMS평가미포함. V2학습만같은속도약3.19일추가.
표현96scene진단0/100/500/1000완료,다음1614부근자동. 1000중심mean.2508px/max3.1373px(FiLM),F1initial.3799→.3475.
미래current대비readout CI0포함;loss감소와연구목표달성은별개. Geometry LoRA는아직미적용.
Snapshot results/lpwm_drivor_lora_v1/status_20261006_0855.json 및500/1000monitor공유artifact보존. Main83/queue250/parallelism4소스hash불변.


## 2026-10-06 09:40 KST — 첫epoch 검토 후24epoch 재개 제어 등록

사용자제안:1epoch후검증·DrivoR비교후나머지24epoch가능한가. Reversible중간검토제어를준비·기동했다.
새PID2949030/scripts/queue_lpwm_drivor_epoch_review.py/configs/lpwm_drivor_review/epoch1.json.
Root outputs/lpwm_drivor_epoch1_review_v1, target1614/epoch1,현재1541로대기중. 최근속도19.21s→10:04KST경계예상.
원래등록trainer/model/config/25epochLR는변경하지않고epoch_01.pt marker후latest fullstate를열린inode로확보·복사한다.
그후기존pause신호로본학습/oldqueue/watch종료대기,다음epoch1update진행시별도보존하고exact1614 checkpoint를검토/재개에쓴다.
Fullresume에는model/AdamW/scheduler/양rankRNG/epoch1,next0가있어추가24epoch38736update재개가능.
CPU검사3개(atomicreplace경쟁/pause덮어쓰기방지/mismatchedepoch거부)통과,259개sourcehash등록.
중단후같은96scene표현진단자동실행→DrivoR비교·해석대기,24epoch자동재개안함.
사용자에게동일DrivoR1epoch새학습(추천) vs공개최종모델참고비교선택을async요청했으며아직응답없음.
공식README/release확인. 현재첫epoch는25epoch warmup3322중1614이므로1epoch완결scheduler와비교금지.
현재navtrain+navval모두학습하므로navval독립검증주장금지;기존96scene도training분포. Navtest자동튜닝/새DrivoR학습은하지않음.
상세docs/lpwm_drivor_epoch1_review.md;검토후resume시실중단latest가아닌epoch_01_resume.pt복원,oldlaunch/pause보존후동일execution재개.
기존10/15 ETA는새검토대기시간미포함으로갱신필요. GeometryLoRA는여전히미적용.


## 2026-10-06 — 첫 epoch 검증의 타당성과 선행연구 설정 확인

사용자 질문:1epoch만 학습 후 비교·검증해도 되는지, 다른 연구에서도 사용하는지.
공식 공개 코드/문서에서 DrivoR v1/v2 25/10epoch, check_val_every_n_epoch=1 및 trainer.fit(val_dataloaders) 연결 확인.
validation_run=false는 학습 중 검증을 끄는 것이 아니라 검증 전용 분기 대신 fit을 선택한다.
DiffusionDrive 공식 train_eval recipe는100epoch. VAD base stage1/2는48/12epoch이며 매epoch checkpoint 저장,
기본 evaluation.interval=total_epochs로 각 단계 마지막 평가다. 저장 주기와 평가 주기를 혼동하지 않는다.
첫epoch검증은 정상작동·초기표현변화 확인에 적합하며 동일예산비교는 그 시점의 성능만 보여준다.
현재1614update는warmup3322중이므로 낮은 점수로최종실패를단정하지않고25epoch스케줄을유지한다.
Geometry LoRA경로부재는초기에판단가능한구조문제이며더많은epoch만으로해결되지않는다.
DrivoR1epoch추가학습은초기비교선택지이지현재연결성진단의필수선행단계는아니다.
docs/lpwm_drivor_epoch1_review.md에 공식6개출처와해석추가. 독립검증부재/navtest반복튜닝금지한계유지.
Review status조회1561/1614 waiting_for_epoch_boundary. 기존train/hold/queue/source/config무변경,추가GPU작업없음.
이번질문은비교방식선택응답이나24epoch자동재개요청으로간주하지않았다.


## 2026-10-06 — 첫epoch particle 시각화·head LoRA 후보 검사·사용자 계층 선택 대기

사용자처음지시:DrivoR비교25epoch후,첫epoch시각화후geometry변화없으면headLoRA추가재학습.
기존제어가정확1614/fullmodel+optimizer+scheduler+rankRNG를보존했다. 실제process는1615중단,추가1update별도보존.
96scene진단완료checkpointSHA3c4a60918e32a7c92d10099b51998ad27b5b74cc0e83ccc1aa0c535450662440.
직진/좌회전/우회전첫고정scene의before/afterPNG와겹침PNG작성. 64center전부,사각형은initialpresence상위16ID고정/실제glimpse크기.
96×4×64=24576전체통계위치mean.302314px/median.207698/p90.658991/max3.975736,1px초과3.6377%.
크기축mean2.02025%/median1.35067%,presence평균절대변화.046125. 완전동일아님,대다수시각적재배치작음.
F1전체.379859→.372994,appearance.292610→.310808. 미래2/4s추가readoutCI0포함.
12scene미래반복/순서역전개입시oracle점수감소초기신호도있어기존방식전체실패로판정하지않음.

별도geometryhead후보모델/학습/추론/queue/monitor/config준비. 원본소스변경없음.
xy/scale/presence두Linear에rank8→4/4/1,alpha=rank,57633param추가. 기존QV포함LPWM LoRA1401121/native109545263고정.
실제이미지2장면/공식onlineoracle planningloss2update검사:세headgradient양수,현재출력→adapter직접연결,
adapter만OFF하면좌표·크기·presence차이,원본SHA/zero-init출력/이전planner초기state동일성통과.
첫검사JSON직렬화np.int64오류를inventory Pythonint로수정하고전체재검사통과. 검사weights폐기.
GPU0·1batch16/effective64/DDP2update통과,최대카드30.6163/29.6149GB,두rankheadgradient일치.
267source/config해시등록을준비했으며본학습기동은하지않음.

진행중사용자가두차례계층설명을요청했고마지막에는"본학습전에보고서내가적용계층을알려주겠다"고명시.
즉시새본학습보류: outputs/lpwm_drivor_geometry_lora_v1/pause.requested 및pending_user_layer_selection.json작성.
새본학습/queue/monitor PID없음. 기존epoch1hold유지. 계층선택전어떤본학습도재개하지않음.
현재객체순회원래Linear234/Conv2d70개CSV/JSON저장. ctx/prior공유alias중복제외.
docs/lpwm_lora_layer_catalog.md에현재head,영상CNN5종,interaction/context/dynamics의projection/attention/FFN/조건변환/출력,
RGBdecoder,미사용posterior/선택옵션/비행렬튜닝구분과현재planninggradient한계설명.
결과 results/lpwm_drivor_epoch1_particle_review_v1/, results/lpwm_drivor_geometry_lora_v1/.
DrivoR비교25epoch이후는확정,다음LoRA대상은사용자선택대기. Conv-LoRA는미구현.


## 2026-10-06 10:35 KST — 세 경로 LoRA 승인 및 본학습 시작

**2026-10-06 최신 사용자 승인: geometry·appearance·future 세 경로 LoRA 본학습 시작.**
사용자가 대상 계층을 선택하고 실행을 요청했으므로, 이전 계층 선택 대기는 이 새 조건에 한해 해제됐다.
[설계·gradient 검사·실행 설정](docs/lpwm_drivor_planning_path_lora_training.md).
Linear229 + Conv44 / LPWM LoRA4,683,650개, native109,545,263개·buffer고정, 공식 DrivoR planner 전체학습.
새 공개 초기화 / navtrain+navval103,288 /25epoch40,350update / GPU0·1 /batch16×누적2×2=유효64.
Loader2·oracle4/rank, 총48decimalGB 상한. 실제DDP2update 최대40.16GB·모든273adapter planning gradient 검사통과.
본학습3186133 /후속queue3186134 /monitor3186135. Root `outputs/lpwm_drivor_planning_path_lora_v1/`.
고정96장면 전·중·후 geometry/readout/intent/개입검사; DrivoR 비교는25epoch 이후.
새source/config279개 등록불변. 이전attention1614/1615와geometry-only hold는 보존하고 재개하지 않는다.
아래 새학습보류/과거실행중 문장은 이전 이력이다.


최신: `results/lpwm_drivor_planning_path_lora_v1/`에 실제 loss2update와DDP2update 증거를보존했다.
273adapter모두 output-gradient>0/unused0,head3개와prior·attributeCNN에서currentgeometry연결확인.
초기 native 출력/초기planner동일, 원래가중치·buffer digest a5dd2345…5d 유지.
Linear229/Conv44 LoRA4,683,650/전체학습21,753,728개; DDP카드최대40,161,509,376bytes.
이는연결·실행검증이지객체이해/주행성능향상결과가아니다. 본학습공식PDMS/EPDMS아직없음.

상세: `docs/lpwm_drivor_planning_path_lora_training.md`. 본학습첫update검증, finalscore미확인.


## 2026-10-06 10:45 KST — 이전/새 LoRA 첫 epoch 직접 비교 예약 및 계층 역할

**2026-10-06 후속 요청: 첫 epoch에서 이전 LoRA 조건과 직접 시각화 비교를 예약했다.**
별도CPU watcher3317230, `outputs/lpwm_drivor_epoch1_lora_scope_comparison_v1/status.json`.
공통초기 / 이전Q·V LoRA1614 / 새세경로LoRA1614를 같은96장면·카메라·명령·particle번호로비교한다.
현재새epoch1을기다리는중이며최종비교PNG는아직없다. 기존25epoch학습과monitor는지속한다.


CPU검사결과 `results/lpwm_drivor_epoch1_lora_scope_comparison_v1/preflight.json`. 계층별Conv/Linear합계273및각역할은`docs/lpwm_drivor_planning_path_lora_training.md`마지막절.


## 2026-10-06 11:56 KST — 명령조건부 current geometry 확인

동일영상에서도command4D→FiLM→attribute CNN→xy/scale/presence head 경로로현재geometry가달라질수있다.
LoRA가명령별다른가중치를선택하는것은아니며동일LoRA가명령에따라달라진feature를처리한다.
완료100update진단12장면/24명령교체에서현재중심이동평균0.002942px,최대0.026271px(128²입력)로반응확인이나매우작음.
초기0update명령교체위치변화는0이었다. 크기/presence도구조적으로영향경로가있지만기존intent JSON에는별도변화량집계없음.
실제좌회전에유용한객체배치·정보보존개선은미확정. 이번턴GPU추가실험/학습설정변경없음.


## 2026-10-06 12:29 KST — 입력 카메라 공정비교 재확인

## 2026-10-06 재확인: 카메라 입력과 현재 세 경로 LoRA의 비교 범위

공식DrivoR NAVSIM recipe와현runtime을대조했다. 카메라는F0/B0/L0/R0의4개와현재1frame로같다.
해상도는 **가로×세로 DrivoR1148×672, LPWM128×128**로다르며카메라당pixel수는771,456대16,384(47.0859배)다.
이는pixel수비율이며실제정보량/성능/연산량비율이아니다. 양쪽모두전체이미지를resize하지만LPWM은정사각형으로변환한다.
LPWM원체크포인트와연결구현을유지하려고128을사용한선택이지DrivoR과동일입력조건을맞춘것이아니다.

공식DrivoR은ImageNet정규화와학습GridMask를쓰며LPWM은RGB/255,normalize_rgb=False,GridMask없음이다.
현재LPWM은Q/V-only단계에서확대되어Linear229+Conv44/LoRA4,683,650개를학습한다.
**아래기존표의LPWM Q/V42개·1,343,488개수치는이전attention-only조건이다.**

같은공식NAVSIM evaluator·split의시스템성능비교는가능하지만설정/해상도/사전학습/계산량을함께보고해야한다.
사용자연구명제인register대particle의구조효과를분리하는엄밀한통제비교는현재조건만으로충족되지않는다.
차이가나면해상도·사전학습·명령FiLM·추가future연산·LoRA예산·증강의효과가섞인다.

후속대조설계는(1)공식DrivoR설정의성능기준,(2)동일camera/raw정보·해상도·종횡비·증강·planner초기값/
loss/update수·평가를맞춘조건,(3)동일pretrained visual features에서register/particle표현을비교하는기전대조로구분한다.
공통해상도에서도DINO/LPWM의사전학습차이는남는다. LPWM고해상도적응은checkpoint/feature head/particle수호환검사가필요하다.
128영상의단순upsampling은원래detail을복구하지않는다. DrivoR저해상도대조는해상도효과분리용이며공식DrivoR재현점수로부르지않는다.

현재학습은128²조건에서planning에따른LPWM표현적응을확인하는실험으로유지한다.
본학습설정을중간에바꾸거나DrivoR대조학습을추가기동하지않았다. 사용자의25epoch후DrivoR비교지시는유지한다.
근거: `results/lpwm_drivor_planning_path_lora_v1/camera_input_fairness_audit.json`.
공식설정: <https://github.com/valeoai/DrivoR/blob/main/navsim/planning/script/config/common/agent/drivoR.yaml>.


## 2026-10-06 12:43 KST — 원본/전처리 입력 영상 비교

현재학습입력시각화완료: `outputs/lpwm_camera_input_visualization_v1/four_camera_original_vs_input.png`.
전/후/좌/우원본1920×1080·실제cache128×128·3배최근접확대를함께표시. 장면c94ee8ade05a5b12/navtrain.
원본4개는바이트그대로복사,입력4개는실제cache에서읽고공식현재BICUBIC전처리와pixel완전일치확인.
각원본/128PNG와`index.html`,`manifest.json`도같은폴더에있음. CPU작업만수행,학습/279source등록불변.

## 2026-10-06 — LPWM 공개 사전학습 가중치의 해상도 확인

사용자 질문: 다른 해상도로 공개된 LPWM 사전학습 가중치가 있는가?
공식 Model Zoo를 웹에서 다시 확인했다. 공개 목록은 아래 5종이며 모두 128×128이다.

| 모델 | 학습 데이터셋 | 공개 목록의 해상도 |
|---|---|---|
| LPWM | Sketchy | 128×128 |
| LPWM-Action | Sketchy | 128×128 |
| LPWM | BAIR | 128×128 |
| LPWM-Language | LanguageTable | 128×128 |
| LPWM-Language | Bridge | 128×128 |

출처: <https://github.com/taldatech/lpwm#model-zoo---pretrained-models>.
공식 Releases 페이지에는 별도 release가 없다: <https://github.com/taldatech/lpwm/releases>.
로컬 공식 configs에는 bair64/balls/balls_occlusion/ogbench/shapes의 image_size=64 설정이 있다.
그러나 설정 파일의 존재는 해당 사전학습 가중치 배포의 증거가 아니다.
공식 README가 연결하는 bair_256/bridge_256은 Hugging Face 데이터셋이며 모델 가중치가 아니다.

결론은 '공식 공개 목록에서 다른 해상도의 가중치를 확인하지 못했다'로 한정한다.
모든 외부 배포처에 없다고 단정하거나 LPWM 구조가 오직 128만 지원한다고 해석하지 않는다.
고해상도 적용은 기존 가중치 재사용·구조 호환성·추가 적응 필요성을 따로 검토해야 한다.
이번 턴에는 checkpoint 다운로드, 고해상도 실험, 현재 학습/queue/등록 source/config 변경을 하지 않았다.

## 2026-10-06 — 첫 epoch 후 DrivoR 동일 해상도 본학습 제안 검토

사용자 제안은 1epoch 비교에서 개선이 확인되면1148×672·DrivoR 전처리로 본학습하는 것이다.
방향은 타당하지만 LPWM의 H/W·patch 좌표·공간 의존 Linear/glimpse·정규화 호환성 변경과 실측이 먼저 필요하다.
현재 ParticleSceneEncoder에는4×3×128×128 assert가 있고 공식 LPWM에는 정사각 image_size 가정이 있다.
공개 hparams patch16에서1148경계 처리가 필요하고 geometry head의 fc_in_dim은 crop 공간 크기에 의존한다.
공식 DrivoR ImageNet정규화와 LPWM RGB[0,1] 입력 차이는 같은 resize만으로 해결되지 않는다.

현재 firstepoch comparison watcher snapshot은285/1614, exact 표현을기다리는중이었다.
1epoch에서 점 이동/크기 변화와 표현 readout/미래/명령 반응/개입 효과를 함께 보되 trainval 진단을 독립 성능으로 부르지 않는다.
고해상도47.0859배 pixel을 VRAM/학습시간 배수로 해석하지 않는다.48GB/card 실측과 effective batch64 유지가 필요하다.
공정 최종비교는 공개 가중치에서 별도 고해상도 학습을 시작하는 것을 권한다.128²1epoch warmup을이어받으면 별도curriculum/학습량을기록해야한다.
같은구조의128²대조 없이 구조변경+해상도변경의 성능차를 해상도효과로만 주장하지 않는다.
상세: docs/lpwm_drivor_representation_and_fair_comparison.md 최신 절.
이번 턴은 제안 검토와 기록이며 구현·GPU실측·학습·queue/registered source/config 변경을 하지 않았다.

## 2026-10-06 13:06 KST — 첫 epoch 전 실제 particle 시각화

사용자의1epoch종료전시각화요청으로300update의현재표현을고정96장면×4카메라에서읽기전용으로추론했다.
공통초기재현/실제forward현재표현대조오차0, native원본digest동일,등록279source/config전후불변.
새스크립트scripts/visualize_lpwm_intermediate_particle_geometry.py;현재/미래의분리중현재geometry만진단했다.
위치평균0.0314985px/중앙0.0240560px/최대0.747336px,크기축평균절대변화율0.236948%,presence평균절대변화0.0059874.
중심1px초과0%,크기축5%초과0%. 고정되지않았지만뚜렷한육안재배치/주행효용은아직주장할수없다.
초기고정top16presence박스+전체64중심,사전고정첫직진/좌회전/우회전장면.같은ID는객체tracking아님.
PNG: outputs/lpwm_drivor_intermediate_geometry_update300_v1/before_after_overview.png.
각장면4카메라PNG/index.html/CSV/복사한checkpoint와현재attributes도같은폴더.공유결과는results/.../intermediate_geometry_update300/.
GPU0전체표본최대41,137,733,632bytes,진단reserved637,534,208bytes,추론약61초.본학습변경없음.

직전사용자의ETA/완료알림요청도확인했다. host에서train3186133/queue3186134/monitor3186135/comparison3317230생존확인.
최신313/1614,최근100update 28.564s/update,첫epoch예상2026-10-06T23:25:55.946009+09:00,남은10.32시간.
현재128²설정25epoch지속시예상2026-10-19T18:46:46.743695+09:00,남은13.24일;후속진단/benchmark/고해상도학습미포함.
첫epoch완료시비교이미지는기존watcher가생성한다.현재연결도구에이대화의예약알림도구는없어push알림을등록하지못했다.
OpenAI Docs공식예약작업문서를확인했으나현재도구가능성과구분한다:https://learn.chatgpt.com/docs/automations?surface=app.
자동채팅알림이설정됐다고주장하지않는다.

## 2026-10-06 13:38 KST — geometry 변화가 작은 이유 진단

300update 시각화가거의동일하다는질문에대해scheduler/main log와실제snapshot을CPU검사했다.
Warmup3322update=2.058epoch,300step LR1.80616e-5=peak9.03%,평균사용LR9.00079e-6=peak4.50%.
첫epoch1614도warmup중(peak48.59%)이므로1epoch를LoRA수렴/실패판정시점으로쓰지않는다.
양수geometrygradient확인,273adapter모두유효DeltaW비영;||DeltaW||F/||W0||F 중앙0.054164%.
최종xy/scale/presence Linear각0.002749/0.011352/0.001668%,단matrix비율은output기여가아니다.
현재RGBreconstruction0으로복원loss경쟁은없다. planner/feature경로에서도loss를줄일수있어geometry이동은보장되지않는다.
기존firstepoch진단및warmup후4000진단을함께볼것을권고. 현재등록본학습변경/GPU추가사용없음.
Script:scripts/audit_lpwm_lora_update_strength.py;근거:results/lpwm_drivor_planning_path_lora_v1/update300_lora_strength_audit.json.
공식LoRA원논문4.1의zero-B/additive W0+BA 설명확인:https://arxiv.org/html/2106.09685#S4.SS1.

## 2026-10-06 13:41 KST — 4000 update 추세진단 ETA

사용자질문에대해진행로그와이전진단시간을읽었다.393/4000,남은3607update.
최근50/100/200wall속도25.572/26.500/27.364초,4000까지25.62/26.55/27.42시간.
중앙예상10/7 16:14KST.첫epoch10/6 22:40,3322warmup완료10/7 11:15KST.
기존96장면진단본체309.10/280.46초.모델로딩/시각화/상황변동을위해추가10~20분정도여유를본다.
전체예상은대략26~28시간,10/7오후4~5시전후이며공유서버부하나메모리대기로변동가능.
이는표현변화추세진단이며최종25epoch/공식benchmark완료예상이아니다.학습·대기열설정변경없음.
근거:results/lpwm_drivor_planning_path_lora_v1/eta_update393_20261006_1341.json.

## 2026-10-06 15:05 KST — 매500 update 자동 진단 등록·500결과 확인

사용자요청으로scripts/monitor_lpwm_particle_trends_every500.py와별도config/registration추가.
기존read-onlymonitor3186135만검증후종료하고새568996기동. Train3186133/queue3186134/epoch비교3317230은유지.
원래279source/config·monitorregistration보존, 새controller는원래watch.lock/evaluator/output을재사용한다.
매500+기존epoch/final106시점, exactatomiccheckpointhardlink보존;나중update대체금지.
CPU4검사통과,0/100/500보고서/PNG/CSV/HTML생성확인. Gallery outputs/lpwm_drivor_particle_trends_every500_v1/index.html.
채팅push도구미연결로자동메시지발송은설정하지못했음을사용자에게알렸다. 로컬결과자동갱신.

교체전원래evaluator의정확500결과완료를재사용:중심평균.074530px/최대1.603768px,크기.5592%,presence.015460.
300보다변화증가하지만차량/보행자중심비율동일,도로proxy미세감소. 전체currentreadoutF1 .379859→.363157,
appearance-only .292610→.304012. 명령교체중심변화.018508px. 주행효용개선은미확인.
582update/최근100속도26.49초,1000학습도달18:09KST+진단여유10~20분예상. 500간격약3.68시간.
500시점보고서/PNG와새등록·상태근거를results/lpwm_drivor_planning_path_lora_v1/particle_trends_every500/에보존.

## 2026-10-06 19:20 KST — 1000 update 중간 결과

현재1090/40350(첫epoch67.53%),본학습3186133/queue3186134/monitor568996/epoch비교3317230생존확인.
GPU0·1전체41.14GB내외/95·100%util표본. exact1000진단완료,1500대기. source/config/학습조건무변경.
500→1000: 중심평균.07453→.66717px,크기평균변화.55918→5.83110%,presence변화.01546→.18522.
18.71%particle1px초과/52.27%particle한축크기5%초과. Current xy/scale/presence gradient양수,원래native digest불변.
전체current판독F1 초기.37986→1000.40072,appearance-only .29261→.24782. 정보향상은혼재하며GT ROIcontrol .41481미만.
차량중심8.883→8.931%,보행자1.200→1.200%,도로proxy20.033→19.753%로뚜렷한재배치는아직없다.
초기도로중심particle고정부분집합presence평균.59382→.19596;도로presence가중비중18.408→8.184%.
전체presence평균.63221→.47481,lowpresence<.1비중19.71→37.89%. 별도semantic/dropout/유용성판정아님.
2초readout현재3.8203/미래3.8403m,4초현재7.8223/미래7.8584m,현재대비차이CI0포함. 미래모듈추가효용미입증.
미래를현재로반복하는개입은ADE .0852m감소/scoreCI0포함;작은12scene교란검사이며독립PDMS아님.
양rank100update평균loss28.0283→14.5374→7.7566,planner공동학습전체결과이며LPWM단독효과아님.
1000LRpeak30.10%/warmup3322중. 첫epoch학습23:38~00:01,4000은10/7 19:08~21:16속도기반범위(진단여유별도).
근거/시각화: results/lpwm_drivor_planning_path_lora_v1/intermediate_update1000_20261006/.

## 2026-10-06 — 고정95장면 중간 공식 PDMS

사용자PDMS확인요청으로scripts/evaluate_lpwm_intermediate_panel_pdms.py를추가하고평가완료.
공통96패널중standardMetricCache없는20c5f1c678e7548a를모든시점에서동일제외,95장면/24recording.
정확0/500/1000particleattributes를동일checkpointplanner로재생하고공식NAVSIM v1 pdm_score/LQR40×.1s로채점.
DrivoRtrainingoracle로이미계산한12장면점수를정식평가로바꾸어부르지않았다.
기존12baseline×3의ADE재생일치36개/최대오차0,원래nativehash보존,평가GPU전체최대41.98079GB.
Sandbox NVML조회실패뒤host재시도2617416으로완료. 추가모델학습없음/기존source/config/queue/monitor변경없음.

PDMS초기30.429935/500:63.405664/1000:67.831505,zero score46/16/11,expertADE9.55123/4.54123/3.17766m.
500→1000좌회전63.29317→74.34462,직진63.24252→70.76221,우회전63.82499→54.68349.
500→1000+4.42584점,24recordingpairedbootstrap5000회seed71의95%CI[-2.81579,11.01668].
전체초기→1000+37.40157점CI[29.95645,45.10453]. 학습분포공동학습진단이며독립navtest/LPWM단독이득은아님.
285scene-checkpoint채점성공/실패0. 결과/CSV/등록/replay검사/그래프 results/.../intermediate_pdms_95_v1/.
이번PDMS는완료된일회평가이며기존500표현watcher에는source불변원칙에따라변경을넣지않았다.

## 2026-10-06 23:29 KST — 첫epoch완료·1500/1614 중간결과

Epoch_01.pt23:00:28저장,현재1682update/epoch2.정확1614old Q/V vs 새세경로비교watcher완료.
평균geometry old .302314px/크기2.02025% vs new1.236072px/7.71424%.42.12%가1px초과/67.43%가한축크기5%초과.
Whole currentreadout 초기.379859→.420517(old.372994),appearance초기.292610→.266910(old.310808).
Same scene명령변경중심평균.255803px, 도로presence가중비중초기18.4076→7.2731%로낮음.
미래readout2초현재3.90490/미래3.83489m, 4초현재8.01687/미래7.83373m.
current대비차이95%CI2초[-.09651,-.03744], 4초[-.34250,-.05745].6recording의예비신호며baseline0변위보다오차높음.

동일95장면공식PDMS재평가3487344완료: 1500=63.519656, 1614=67.986416.
1000=67.831505대비첫epoch+0.15491점, 24recording pairedbootstrap5000/seed71 CI[-7.14543,7.16457].
도로준수.90526→.84211, zero score11→17, expertADE3.17766→2.10819m.
직진70.7622→74.5582, 좌회전74.3446→63.7964, 우회전54.6835→61.9971.
190채점성공/실패0, 기존baseline24ADE재생검사오차0, GPU전체최대41.72913GB, nativefreeze/source보존.
학습분포진단이며navtest/LPWM단독효과아님.본학습/queue/정기표현monitor변경없음.
최근100 wall24.90초, 2000은10/7 01:41, warmup3322는10:49, 4000은15:31학습도달예상/진단시간별도.
결과/누적PDMS/비교PNG: results/lpwm_drivor_planning_path_lora_v1/epoch1_intermediate_report_v1/.

## 2026-10-07 09:18 KST — 3,000 update 중간 결과·25 epoch ETA·포화 판단

본학습3,020/40,350update(7.4845%), 1.8711epoch. Train3186133/queue3186134/monitor568996 host생존.
양rank3,020행 비유한loss0, native/실행digest각1개. Update3,000의14LoRA경로와plannergradient유한·양수.
본학습batch16/누적2/effective64·loader2/oracle4·source/config/loss/LR/queue변경없음.
추가평가2301619: exact2,000/2,500/3,000 × 동일95장면 공식NAVSIM v1 PDMS 완료.
285성공/실패0, 기존baseline12×3 ADE재생검사36개최대오차0, GPU전체최대41.72913GB/nativefreeze유지.
PDMS70.16709/72.92701/74.90627. 첫epoch대비3,000 +6.91986점CI[1.02676,12.38134].
1000대비+7.07477 CI[-.78508,14.36931], 2500대비+1.97926 CI[-5.66113,9.02419]; 비교시점별차이숨기지않음.
직진epoch1 74.5582→84.4096/좌63.7964→68.2930/우61.9971→66.9380. 도로준수.84211→.88421.
공식채점방법이지만학습분포패널이고navtest아님; 공동학습효과이며LPWM단독미세조정이득분리아님.

Geometry초기대비2,000 1.385px/10.879% →2,500 2.163px/13.527% →3,000 1.813px/10.957%.
3,000의63.29%가1px초과이동,86.19%는한축크기5%초과. 명령교체중심.399664px.
WholecurrentF1 .419269/.386249/.393593, appearance .273758/.352573/.356225. 초기whole.379859/appearance.292610.
차량중심8.88265→8.79720%,보행자1.20036→1.21257%,roadproxy중심20.03289→20.68257%.
Roadpresence가중비중18.40762→5.34835%지만presence는plannerattention이아니고feature는그대로입력된다.
미래readout첫epoch의예비이득지속안됨:3,000 2초current3.97808/predicted4.01345m(차이+.03536,CI[.00270,.10121]),
4초8.25388/8.21429m(차이-.03959,CI[-.26095,.23276]). 절대오차0변위baseline보다높음.
Future를current반복으로교체시12장면trainingoracle선택점수+.007808CI[.002938,.014411]; 공식PDMS나인과효과로해석안함.

사용자후속질문25epoch잔여와saturation방법답변:23.13epoch/약11.6–12.1일,10월19일전후학습종료(평가별도).
최근50/100/200 wall27.93/27.32/26.90초. Epoch2경계10:51–10:55, warmup3,322 11:33–11:39,4,000 16:37–16:55예상.
Warmup전이면서PDMS상승이므로현재포화근거없음. 독립개발/실질최소개선delta/3–5epoch patience/LR감소구간과안전·표현진단권고.
현재navtrain+navval모두학습으로독립validation없음. 95훈련진단을일반화포화근거로쓰지않고navtest반복tuning도하지않음.
자동조기중단/새학습조건미등록. DrivoR25epoch비교는같은학습예산유지;조기종료실험은동일budget/중단규칙필요.
결과/생성스크립트/CSV/PDMS·future·particle PNG: results/lpwm_drivor_planning_path_lora_v1/intermediate_update3000_20261007/.

## 2026-10-07 09:30 KST — Particle 겹침·이동 화살표와 향후 자동 출력

사용자가3,000 비교그림에학습전후particle을겹친그림도항상함께요청했다.
새CPU전용 scripts/publish_lpwm_particle_geometry_overlays.py를추가했다. 기존학습/monitor/source/config는불변.
입력은완료diagnostic의particle_attributes.npy/panel/images/geometry_report; 새학습·GPU모델추론없음.
직진2/좌회전0/우회전3 front카메라에서현재64particle을같은번호로대응했다.
청록빈점·점선박스가전,주황점·실선박스가후. 흰화살표의끝점은실제변경된중심이다.
Glimpse박스는초기presence상위16개를고정. 겹침dot반경은고정하여위치변화를비교하며presence표시와구분.
512tile또는384tile로영상·좌표를같이확대했으며displacement자체추가증폭1.0.
기존3열PNG를복사하고4열겹침을추가해사용자가본기존그림과직접대응되도록했다.
3,000처음PNG둘을시각적으로검사한후CPUwatchpublisher2507743 실행,100/500/1000/1500/1614/2000/2500/3000 자동생성확인.
Sandboxbackground시도PID5는작업종료와함께끝났으므로host에서재실행했다. GlobalPID5를signal하지않음.
초기preview/source도별도보존,실제publisher source SHA b97ab198b03057788f51108d77de94c6123dc75c4b67a2469d6047f405c57f85.
향후매500/epoch기존diagnostic완료마다새시각화가자동추가된다. Watcher final40,350처리후종료/자체stop.requested지원.
Root outputs/lpwm_drivor_particle_geometry_overlays_v1/의index.html에기존비교·겹침·4열링크.
현재PNG/overlay metadata는results/lpwm_drivor_planning_path_lora_v1/intermediate_update3000_20261007/에추가했고
launch/status는results/lpwm_drivor_planning_path_lora_v1/particle_geometry_overlay_visualization/에보존했다.
Geometry변화는planningbenefit/attention이나semantic tracking의증거가아니다. 앞으로전후와겹침을함께보고한다.

실제PNG검사: 통합그림의header아래기존3열pixel이원본과완전히일치. 8시점출력과CPUheartbeat, source/launch/metadata hash일치를확인했다.

## 2026-10-07 09:52 KST — 차량 밀집 장면3개의 전후 particle 비교·겹침

사용자가기존scene보다주변차량이많은다른이미지도요청했다.기존96고정패널에서front투영vehicle GT수로screening,
차량박스최소4×3input pixel로큰객체수도계산했고rawcontactsheet를육안검토했다.
실제차량밀집교통으로scene41(c3ba85bd489a5e47)/21(2516f0fa67f9535f)/27(2b2c80d7c63e5ea4)를선정.
각기다른recording의도심버스·승용차교통/고가도로아래대기/근거리차량대기. 모두straight유형이며유형대표분해는아님.
투영차량GT31/15/13은가림포함수로완전히보이는차량수·모델검출수로해석하지않는다.
선택기준은GT밀도와영상의traffic내용이며particle변화량/PDMS로선정하지않음.
새script scripts/visualize_lpwm_vehicle_rich_particle_changes.py로exact3,000savedattributes를읽어CPU렌더.
청록전/주황후/동일64particle/초기presence상위16glimpse박스/실제이동화살표·고정overlaydotradius를유지.
이미지encoder/planner재추론·학습·GTloss추가없음. 기존source/config/queue/monitor/overlaypublisher변경없음.
Front평균중심이동scene41 2.212736px /21 2.484261 /27 1.126128;사례효용증거로해석안함.
원본1920×1080 RGB를BICUBIC128²로resize한값이3장면모두실제cache input과bitwise동일함을검사했다.
초기BILINEAR가정에서일치검사가실패했고실제prepare_lpwm_drivor_joint_data.py의BICUBIC으로정정해통과.
완료results/lpwm_drivor_planning_path_lora_v1/vehicle_rich_particle_visualization_update3000/에는
vehicle_rich_particle_overlays.png, vehicle_rich_before_after_with_overlay.png, vehicle_rich_original_camera_images.png,
scene041/021/027_four_camera_overlays.png, selection_and_geometry_report.json(source/checkpoint/panel/attributes hash).
출력PNG3개를육안검토했다. 향후같은3장면은 --updates와새 --output을주어재생성하고전후+겹침을함께제공한다.

## 2026-10-07 10:01 KST — 현재 particle 시각화 해석

이 그림은 같은 현재 이미지에서 학습 전후 모델이 추정한 particle의 2D 중심, glimpse 크기, 일부 presence를 보여준다.
현재프레임만 표시하며 dynamics의 미래particle이나 실제차량의시간상움직임을그린것은아니다.

| 표시 | 실제 코드의 의미 | 해석 범위 |
|---|---|---|
| 점 위치 | particle의학습된영상좌표 / glimpse중심 | 객체 중심정답이나semantic object ID로확정하지않음 |
| 전후별그림의초기상위16개점크기 | 표시반경2+4×presence, presence는0–1 | obj_on활성도이며planner중요도/검출confidence아님 |
| 전후별그림의나머지48개점크기 | 고정반경2px | presence가낮다는뜻아님 |
| 겹침그림점크기 | 청록빈점반경4px /주황실점반경2px로고정 | 크기차이에서presence/중요도변화를읽지않음 |
| 박스위치·폭·높이 | 중심±학습된scale/2의glimpse범위 | particle이외관정보를얻는영상crop범위; 객체GT/detection bbox아님 |
| 박스개수16개 | 각scene/camera별초기presence상위16개번호고정 | 학습후새로운top16을선택한그림아님;나머지48개도내부scale존재 |
| 색·점선·실선 | 청록/점선은학습전,주황/실선은학습후 | 객체종류/안전도/attention을표시하지않음 |
| 흰화살표 | 같은particle번호의학습전후중심차이 | 실제좌표끝점/이동추가증폭없음.물체의물리적속도나미래궤적이아님 |

Presence는LPWM의obj_on latent로원래foreground활성도/alpha compositing에관여한다.
공식RGBdecoder는obj_on으로particle alpha를곱하고depth로혼합하며,appearanceencoder도obj_on을받으면CNNfeature를곱한다.
현재planner입력은64particle의현재+미래14Dattributes전체를projection/pooling하므로작은점/낮은presence만으로삭제되지는않는다.
같은차에여러particle이걸칠수있고한glimpse에차량·도로·배경이함께있을수있다.
큰박스는더넓은범위,작은박스는더좁은범위를뜻할뿐정보량/중요도/정확도를보장하지않는다.
Particleinteraction과context로다른particle정보가추가되므로박스가표현의유일한정보출처도아니다.
차량쪽중심이동은기하학적변화의증거이며객체정보보존·미래예측·planning기여는readout/개입/PDMS로따로검증한다.
위치·크기가비슷해도appearance/context/dynamics feature는달라질수있다. 현재이미지에서미래예측성능을판정하지않는다.

확인코드: scripts/visualize_lpwm_planning_path_geometry.py:45(draw_particles),
scripts/publish_lpwm_particle_geometry_overlays.py:64(draw_geometry_overlay),
reference_repositories/LPWM/modules/modules.py:3010(appearancecrop), :5344(alpha obj_on),
src/planning_aware_future_prediction/object_centric/lpwm_drivor_joint.py:108(전체particle projection/pooling).
이번작업은코드확인·해석설명과문서기록이며학습/시각화source/config/실행중publisher변경없음.


## 2026-10-07 10:25 KST — GPU0·1 현재 학습 속도 진단·CPU oracle 증설

사용자 요청은 현재 학습의 단축 가능성 확인/기존48decimalGB 예산 활용이다.
README/AGENTS/HANDOFF와 실제execution/phase timing/이전실측/host CPU·GPU를읽었다.
3038–3137 rank0평균26.238s, forward6.491/oracle4.783/backward14.710s, loader대기.0676s, wholecard40.325GB.
Host CPU32core/64thread, availableRAM298GiB;공용원본/타인프로세스변경없음.
Loader는병목이아니고micro16이이미40.3GB여서단순batch32를48GB안전범위로가정하지않았다.

새CPU-only benchmark scripts/benchmark_lpwm_drivor_oracle_parallelism.py를실행했다.
두rank각16scene/실제후보고정/두서비스동시호출,4→8→6→4 각5회(초기화제외),평균2.620/1.669/2.103/2.851초.
7subscore exact equality통과, CPU서비스시간약39%감소. 고정후보·warmcache·공유CPU범위임을기록했다.
출력 outputs/lpwm_drivor_oracle_parallelism_20261007/;보고 results/.../throughput_20261007/.

새execution_batch16_loader2_oracle8.json은micro16/누적2/effective64/loader2/모델/학습loss/LR/데이터/25epoch를유지하고oracle만4→8.
새283source override등록/queue wrapper/안전checkpointresume controller 추가, 기존279연구source/config/registration 불변.
준비controller첫filename검사가경로prefix때문에pause전에실패. 실패소스·registration·이유를oracle8_resume_20261007에보존,
고친검사와이전등록hash/amendment를명시해oracle8_resume_checked_20261007에서재실행했다. 조용한기존연구seal갱신은없음.

3159update정상경계저장후원래model/AdamW/scheduler/2rankRNG/full latest hardlink보존·복원.
Optimizer796state step min=max3159, checkpoint611c9853...,nativea5dd2345... 동일.
새train3144180/queue3144181/500monitor3144182로재개, 원래3186133/3186134/568996종료.
Overlaypublisher2507743계속. 기존v1→fullnavtest→freshv2→EPDMS queue유지, v2도oracle8으로실행.
실제부모launch.json/queue_launch.json/500launch.json은이전사본보존후새PID메타데이터로갱신.
GPU입장/매update48GB전체카드guard유지. 새로운LPWM초기화/추가학습조건/DrivoR실험없음.
DataLoader새iterator생성때문에state복원과bitwise동일한dropout흐름을구분했다.

Resume첫3160은iterator건너뛰기53.22초+CUDA/oracle초기화36.24초로안정비교에서제외.
3161–3170 각rank10개의실제학습으로wall26.3176→23.5018s,10.699%시간감소.
Rank별update느린쪽평균26.2568→23.7326s, rank0oracle4.7832→3.2447s,최대wholecard40.1615GB.
3170기준37180update잔여약10.113일/10월17일13:08외삽,이전속도대비29.081시간절약.
짧은표본·서버부하·scene비용·GPU역전파시간변화때문에장기개선량이나worker단독인과효과를보장하지않음.
후속평가시간/V2학습별도. 두rankfinite검사/old279·new283hash/queueheartbeat/next3228monitor상태통과.
샌드박스NVIDIA조회1회실패는host조회로확인해정상GPU학습85/72% 표본을확인했다. 드라이버/다른작업변경없음.

문서AGENTS/README/학습문서/HANDOFF1–5/본일지갱신, 결과3JSON과측정script를공유한다.
이후100/300update속도추세·메모리·500/epoch진단상태확인. 기존25epoch과학습목표를유지한다.


## 2026-10-07 10:42 KST — 중간 결과 요청: 최신 학습상태와 완료된 검증 구분

README/AGENTS/HANDOFF/실제progress/queue/500monitor/trend와완료3000보고를읽었다.
현재3213/40350(7.963%)/epoch2의1599/1614, GPU0·1/micro16×accum2×2effective64/loader2/oracle8 유지.
Oracle8 steady3161–3213/53표본 wall23.3497초, 최대40.1615GB; 전체양rank로그NaNloss0.
3200의모든gradient그룹norm유한·양수, old279/new283sourcehash불변. Queue3144181/500monitor3144182fresh.
NVIDIA표본memory38301MiB/GPU, util62/0은짧은CPUoracle구간표본이며그후업데이트지속확인.
새평가/모델추론/학습조건변경없음. 최신완료표현·공식95장면PDMS는3000이고현재update성능값은아직없음.
Epoch1/2000/2500/3000 PDMS67.9864/70.1671/72.9270/74.9063, 학습분포95장면24recording/navtest아님.
Firstepoch→3000 +6.9199,CI[1.0268,12.3813];3000직진84.4096/좌회전68.2930/우회전66.9380.
Particlegeometry평균1.813px/10.957%,wholeF1firstepoch.4205→.3936,appearance.2669→.3562.
미래2s readout current3.9781/future4.0134m,도로presenceproxy하락. 움직임/학습은확인되지만driving객체·미래정보의추가가치확정불가.
새status JSON results/lpwm_drivor_planning_path_lora_v1/intermediate_status_20261007_1040.json에runtime와완료결과를구분보존.
3228epoch경계도달약10:48, warmup3322약11:24,3500약12:34,4000약15:48;표현진단처리시간별도.
현재속도V1최종ETA10월17일11:34전후,서버부하변동·후속평가/V2제외. 기존25epoch/자동진단/queue유지.
HANDOFF1–5/README/본일지갱신,다음완료된3228/3500결과를보고하도록정리.


## 2026-10-07 11:07 KST — LPWM→planner foreground/background/context 인터페이스 확인

사용자는현재LPWM의무슨정보가planner로넘어가는지질문했다. README/AGENTS/HANDOFF와실제encoder/helper/공식DrivoR forward/LPWM sample/hparams를읽었다.
현재인터페이스FG10D=영상xy2/scale2/presence1/depthlatent1/appearance4 + BG4D=14D.
BG는카메라·시점별공통4D를64foregroundparticle에붙이고현재1+미래8시점모두사용한다.
Mmap읽기검사: update3000완료attributes96×4×9×64×14,마지막4채널64particle사이최대차이0.
별도64개BG점/GTobjectclass/semantic도로·차량분해라는해석은틀리다. LPWM의FG도도로·나무를표현할수있다.
Encoded context출력tensor를planner에직접전달하지않으며sample z_context=None으로내부contextprior생성→dynamics조건→미래FG/BG경유.
None을contextoff라고해석하지않는다. Context LoRA는future출력경유planninggradient가있다.
현재+미래9×14=126D→Linear256/LayerNorm/GELU→고정번호4개mean→16token/camera→카메라embedding→총64×256scene memory.
공식trajectorygenerator/scorer가같은memory를사용. 64FG전부참여하지만개별particle토큰은그대로보존되지않는다;pooling은학습된selector/spatialneighbor/objecttracking이아님.
Ego11D는planner별도,command4D는encoderFiLM. 큰hidden/variance/posterior/RGB는직접안넘김;z_score는내부rollout에만사용.
이구조의압축/bottleneck가능성은서술하되실패원인/계획성능효과로확정하지않았다.
학습문서와HANDOFF1–5/본일지에기록. 현재3278/40350 epoch3/3228검증완료/3500대기.
새학습/평가/모델구조/실행config/등록source/queue/monitor/publisher변경없음.

## 2026-10-07 12:05 KST — epoch2 중간 결과 및 공식 PDMS 평가

사용자 중간 결과 요청에 따라 정확한3,228update /epoch2의 완료된 표현 진단을 읽고,
기존 scripts/evaluate_lpwm_intermediate_panel_pdms.py로 동일95학습장면 /24recording을 공식 NAVSIM v1으로 채점했다.
GPU0에서 저장attributes→같은checkpoint planner 재생, CPU scorer2workers 사용. 학습 업데이트 없음.
12baseline장면ADE차이 전부0 /native digest유지 /평가전체카드최대41.261466GB /실패0.
PDMS74.5769543,3000의74.9062726대비−.3293184점 CI[−5.4791674,+4.9163912].
1614첫epoch67.9864165대비+6.5905378점 CI[+.4949005,+13.2121105].
Paired 동일recording24개 /5000bootstrap /seed71. 직진85.455099/좌71.897053/우59.343334.
우회전은3000의66.937989보다낮다. Expert ADE2.318900→2.005668m 감소에도PDMS는비슷하다.
기하초기대비중심1.675429px/크기15.032439%,3000→3228실제중심변화1.038864px/크기6.944177%.
전체현재판독F1.418250/appearance.364822. GT ROI를쓰는probe이며LPWM자체detection성능이아니다.
2s current/future3.949206/3.975663m,4s8.149321/8.266721m. 차이CI둘다0포함/zero-displacement보다높은오차.
첫epoch의미래readout이득방향이유지되지않음. Future개입에서도추가이득미확인,geometry변화만으로주행효용주장안함.
차량중심9.0739%/도로proxy중심19.8479%/도로presence가중4.9499%;집중적주행영역재배치미확인.
CPU reportscript 및 수치·pair·등록·전후겹침·상황별PDMS그림을 새results/intermediate_epoch2_20261007에보존했다.
12:04:51본학습3428/40350 epoch3/warmup3322완료. 양rank각3428로그의비유한loss0/nativehash한개유지.
최근18gradient그룹모두유한·양수,원래279/oracle8실행283개sourcehash모두일치.
Oracle8동일실행최근50/100update23.13646/22.81048초,다음3500오늘12:32/4000오늘15:42–15:45.
V1학습완료10월17일06–09시KST외삽이며최종평가·V2시간별도. 단일구간에서포화·조기중단판정안함.
Train3144180/queue3144181/500monitor3144182/CPUoverlay2507743생존;batch16/accum2/GPU2/loader2/oracle8유지.
본학습source/config/대기열/monitor/25epoch계획변경없음. 새 PDMS는완료했고GPU해제,새baseline은미기동.
Sandbox NVML 실패는호스트query재시도에서정상응답했고실제GPU0·1각40.325GB. 드라이버장애로해석하지않음.

## 2026-10-07 13:16 KST — 학습의 유의미성 검사 및3,500 PDMS

사용자질문에맞춰최신3500표현진단·전체양rank로그·planning-only목적을확인했다.
양rank3611로그비유한loss0/nativehash한개,최근18개gradient그룹모두유한·양수. Source279/283hash일치.
GPU0에서저장된attributes의동일checkpoint planner재생/CPU2workers로공식95PDMS일회평가를완료했다.
PDMS75.9039356,직진85.088196/좌69.885614/우67.737059,expertADE1.965246m/NC.952632/DAC.905263/TTC.936842.
첫epoch대비+7.917519 CI[+.795418,+15.207135],3000대비+.997663 CI[−6.549471,+8.457738],3228대비+1.326981 CI[−4.024176,+7.194000].
동일95training scene/24recording/5000bootstrap/seed71이며navtest/일반화검증아님. 실패0/재생12ADE차이0/카드최대41.261466GB.
새3,500신호:12scene future-repeat-current selectedoracle score−.084639 CI[−.196006,−.002792],fixedcandidate−.042448 CI[−.119362,−.001258].
ADE+.480654m CI[+.140652,+.839179]. 미래분기에대한의존/활용초기신호로해석하며물리적미래정확도증명과구분한다.
이oracle변화를공식PDMS의8.46점하락으로부르지않음. 300bootstrap/OOD대체/12scene/반복검사한계명시.
Geometry초기대비1.847200px/18.004615%,appearanceF1.382788/wholeF1.426689.
차량중심9.1634%/도로proxy19.0872%.2s current/future4.064171/4.040740,4s8.364325/8.216459m.
Readout차이CI0포함/zero-displacement보다높은오차. 정확한미래정보보존향상과planning효용의안정적이득미확정.
현재 planning-only loss이므로 미래 상태 정합을 직접 감독하지 않음. Planner+LPWM 공동학습의 전체 PDMS 향상과 LPWM 단독 효과는 미분리.
Rank0 rolling100 loss4.238733(1601–1700)→3.144724(3401–3500)/3.184487(3501–3600),상이한batch/online목표한계.
13:16:31학습3611/40350 epoch3/약2.24epoch, warmup이후289update. 기존25epoch계획·queue·500진단유지.
새source/config/목표/학습조건변경없음. 추가일회PDMS는종료·GPU해제,새baseline/실험조건미기동.
통합 results/lpwm_drivor_planning_path_lora_v1/learning_meaningfulness_update3500_20261007/에 assessment script/수치/등록/officialscore 보존.
학습의 정상 진행·planning 개선·미래분기 활용 신호는 보고하되 미래 정확도/LPWM 단독 기여/heldout 일반화 결론은 보류.

## 2026-10-07 13:26 KST — 미래분기 개입 설명도와실제경로시각화

사용자가 0.48m 증가의 의미를 이미지와 함께 쉽게 설명해 달라고 요청했다. scripts/visualize_lpwm_future_branch_intervention.py를 추가했다.
정확한 3500 checkpoint/nativehash와 기존 panel·attributes·interventions SHA를 등록했다. 원래 미래/현재 반복 두 조건을 12장면에서 재생했다.
현재 camera/ego/command/model은 같고, 미래 8step의 foreground/background 14D attributes만 현재값으로 대체했다.
Encoder 재추론/optimizer/학습/새 oracle 채점은 없다. 저장된 24개 ADE와 차이 모두 0 /maxcard41.250980GB /nativehash 유지.
12장면 GT 궤적 ADE는 1.510779→1.991433m /평균 +.480654m. 8악화·4개선. 두 모델의 계획 경로 사이 거리와 구분한다.
0.5~4초 8개 시점에서 GTxy와 planxy 거리의 평균을 장면별 계산한 후 12장면 평균을 구했다. GT는 plot/평가에만 사용한다.
한글 NotoSansCJK 폰트로 입력 변경 설명도/실제 우회전 사례/세 유형 사례/12장면 전체 barplot을 생성하고 육안 검사했다.
유형별 첫 panel scene8 우회전/16 직진/0 좌회전을 선택했다. 결과 방향으로 고르지 않았고, scene0의 오차 감소 반례도 포함한다.
초록 GT/파랑 future 사용/주황 current 반복. 평면도 horizontal−ego_y/verticalego_x(m). 전방 사진은 참고이며 4camera 입력을 유지한다.
Source·checkpoint·24재생검사·trajectories.npz/visualization_report.json를 별도 results/future_branch_intervention_explained_update3500에 보존했다.
기존 25epoch 학습/queue/500monitor/publisher/sourceconfig는 변경하지 않았다. 추가 GPU 일회 재생은 완료·GPU 해제, 본학습3635/40350 epoch3 계속.
해석은 미래 분기에 대한 의존의 초기 증거다. 실제 물리적 미래 정확도·운전 객체 이해·안전 이득은 별도 검증이 필요하다.
실제 참조 camera와 checkpoint 출력을 표준 plot으로 렌더했다.

## 2026-10-07 16:39 KST — 4,000-update 중간 결과

사용자 중간 결과 요청에 따라 완료된4000 표현진단·readout·geometry·미래개입을 확인했다.
기존 evaluate_lpwm_intermediate_panel_pdms.py로 동일95training장면/24recording의 공식PDMS를 평가했다.
PDMS75.8302944,3500대비−.0736412 CI[−5.2169817,+5.2610609],첫epoch대비+7.8438780 CI[+.7064257,+15.1245060].
같은token 대응/5000recording bootstrap/seed71. 직진89.044452/좌69.508018/우61.158954,expertADE2.031890m.
NC.973684/DAC.884211/TTC.905263. 직진상승과우회전하락이함께관측되어평균만보고일관된개선을주장하지않는다.
공식평가95성공/실패0/12재생ADE차이0/nativehash유지/최대카드41.261466GB,일회평가완료·GPU해제.
Geometry초기대비1.979098px/14.098437%,직전3500→4000실제중심이동1.788838px.
Appearance readout F1 .382788→.411411,전체attributes F1 .426689→.422994. GT ROI probe이며deployment detection이아니다.
Future-repeat-current ADE+.244425m CI[−.198056,+.835376],selectedoracle−.007215 CI[−.214978,+.210078].
3500의ADE+.480654/selectedoracle−.084639 긍정신호는이번에는CI0을포함한다. 안정적효과·효과소멸을단정하지않는다.
2s current/future3.997089/4.016425m,4s8.233153/8.146505m. 차이CI0포함/zero-displacement보다높은오차.
차량중심9.0088%/도로proxy20.0206%/도로presence가중4.6124%,주행영역집중미확인.
본학습4116/40350 epoch3/양rankloss비유한0/4100샘플18gradient그룹유한양수/279·283source hash 정상.
Train3144180/queue3144181/monitor3144182/publisher2507743 생존,본학습조건·모델·loss·유효64·source/config 변경없음.
최근50/100/200update wall23.6104/23.8062/23.8282초. 4500오늘19:10–19:11/4842오늘21:25–21:27(학습경계도달 기준).
V1학습끝10월17일14–16시외삽,진단처리/최종평가/V2시간별도. 기존25epoch·후속queue·정기진단 유지.
새 results/lpwm_drivor_planning_path_lora_v1/intermediate_update4000_20261007/에report/script/score/등록근거와PDMS·미래개입·전후겹침그림 보존.
학습패널진단이며navtest/독립일반화/LPWM단독효과증거가아니다. Source수정·중단gate·새baseline 실행없음.

## 2026-10-07 16:50 KST — 현재 학습 포화 여부 판단

사용자가 현재 학습이 포화됐고 더 이상 성능 개선이 안 되는지 질문했다. 상태 판단 요청이며 중단 요청이 아니다.
16:49 조회 학습4,144/40,350, 약2.568/25epoch. Warmup3,322 이후822update(약0.509epoch), LR0.000199627/peak0.0002.
Monitor 마지막 완료4,000/다음4,500 대기. 기존 완료report에서 동일95학습장면 PDMS 추세를 확인했다.
첫epoch67.99/2,500=72.93/3,000=74.91/3,500=75.90/4,000=75.83. 최근 상승 둔화는 관측됐다.
3,500→4,000 차이−0.0736, paired recording bootstrap95%CI[−5.217,+5.261]. CI0 포함은 무효과나 포화의 증명이 아니다.
직진85.09→89.04/우회전67.74→61.16으로 평균 정체 안의 상황별 변화가 존재한다.
현재 navtrain+navval 모두 학습에 사용해 독립 planning validation이 없다. 일반화 포화와 추가 개선 불가능은 미확정이다.
Warmup 이후 짧은 구간/LR 감소 초기이며 gradient·particle 변화가 실제 추가 성능 개선을 보장하지 않는다.
미래정보 보존과 미래분기의 안정적인 planning 기여도 여전히 미확인이다. 단순 추가 학습으로 해결된다고 약속하지 않는다.
기존25epoch/후속 queue/매500·epoch 진단 유지. 새 GPU 작업/모델·loss·source/config 변경/중단 gate 없음.
HANDOFF 1–5절과 학습 문서에 근거를 추가했다. 다음4,500/4,842 이후 여러 epoch 추세를 확인하되 단일 진단으로 포화를 확정하지 않는다.

## 2026-10-07 17:00 KST — Particle의 planning 이득 검증 설계

사용자가 particle 정보가 planning에 주는 이득을 어떻게 확인할지 물었다. 방법 설명 요청이며새실험시작요청이아니다.
실제lpwm_drivor_joint.py와monitor_lpwm_drivor_representations.py/particle_planning_diagnostics.py를읽어입력·pooling·개입을확인했다.
Current/future foreground10D와background4D가함께전달되고4particle고정평균으로16token/camera를만든다.
기존12training장면개입은future-repeat-current/reverse-time/GTcorridor연관particle평균대체/matched다른particle대체다.
고정64후보 재채점으로scorer선택/regret을검사하는구현이있다. 모든개입의새후보전체품질비교와FG/BG분리확대는후속항목이다.
ROAR NeurIPS2019 원문을읽어추론중정보제거의분포변화한계와재학습대조원칙을참고했다.
제안 A현재+미래(현재본학습),B처음부터현재만반복,C초기LoRA고정,Dencoder명령off. Planner와projection은모두학습한다.
C는FiLM을A처럼학습해LoRA만의효과를분리한다. 완전LPWM+FiLM고정planner-only대조와구분한다.
Publicinit/planner초기state/token순서/유효64/학습량/LR/loss/입력/evaluator를맞춘다. Seed반복과recording CI를권고했다.
B는token/channel budget을유지하나dynamics가빠져연산량·활성parameter차이가남는다. Future물리정보주장에는계산량대조/시간별readout을추가한다.
GT연관은glimpse박스proxy,matched다른particle도무관하다고보장안됨. Background·다른particle에정보가중복될수있다.
Probe정보접근성/개입사용의존성/독립벤치마크planning이득을구분하고안전·도로준수·진행·상황별지표를함께보고한다.
GT라벨은진단/probe감독에만쓰는설계,본학습객체GT loss추가확정없음. 다른명령GT없는장면에원래GT로정답판정을하지않는다.
현재navval도학습에포함돼독립development으로쓸수없다. 새로운개발split은처음부터제외해야한다. Navtest반복튜닝금지.
설계는docs/lpwm_drivor_representation_and_fair_comparison.md 마지막절에기록했다. 새학습/평가/대조queue/source변경없음.
16:59조회4170/40350 epoch3,monitor4000완료→4500대기. 기존25epoch·후속평가및매500/epoch진단유지.

## 2026-10-07 18:01 KST — 중간 상태 확인, 새 성능 검증 대기

사용자 중간 결과 요청에README/AGENTS/HANDOFF와현재학습·monitor·queue상태를읽었다.
18:00:55 기준4,324/40,350, epoch3/약2.679epoch/전체10.716%. Micro16×accum2×2=유효64/loader2/oracle8.
GPU0·1query38457MiB각(40.325GB)/util100%. Host train3144180/queue3144181/monitor3144182/publisher2507743생존확인.
Sandbox ps에host PID가없어승인된읽기전용host ps로재확인했다. 다른사용자프로세스변경없음.
양rank4324전체로그비유한loss0/nativehash동일,4300최근18gradient그룹모두유한·양수.
원래279/현재oracle8실행283sourcehash모두일치. Monitor4,000완료→4,500대기,새완료진단/공식PDMS없음.
최신완료4,000의PDMS75.8303,직진89.0445/좌69.5080/우61.1590,95training패널이며navtest아님.
Particle초기중심1.979px/크기14.098%,appearanceF1.4114/wholeF1.4230,미래대체12scene ADE+.2444m CI[−.1981,+.8354].
기존수치를현재4324성능으로부르지않는다. 포화·안정적미래정보기여에대한새결론은없다.
최근50/100/200update wall24.1770/23.8734/23.7642초,4500오늘19:11/4842오늘21:26–21:30경계예상.
V1학습완료10월17일16–20시외삽,진단처리/fullnavtest/V2등후속시간별도. 공유부하로변동가능.
결과 results/lpwm_drivor_planning_path_lora_v1/intermediate_status_20261007_1800.json에조회시각·progress·rankhealth·sourceaudit·ETA·최신기존평가근거보존.
README/HANDOFF1–5/학습문서/RESUME_NOTES갱신. 모델/loss/학습량/source/config/queue/monitor/publisher변경없음.

## 2026-10-07 18:07 KST — 이전 Adapter 82.49점의 epoch와시간범위 재확인

사용자가Stage1/2분리후Adapter만추가한82.49/ADE1.155/FDE2.757/미래오차+.38%/LPWM70.5만/4h9m이1epoch인지질문했다.
원래residual_adapter_batch8 config epochs1,완료training_summary epochs1/completed4707/train75297/seed47및review값을확인했다.
Stage1 actualsummary는20epoch/28920update/23126clip,Stage2 inherited stage1 checkpoint SHA가Stage1최종SHA와일치.
Stage1은공개SketchyLPWM의NAVSIM SSL posttraining,Stage2는Adapter704960+planner/command2211975학습/native고정.
총2916935학습parameter. Stage2목표는planning+0.02공식temporalELBO,직접객체GT보조loss OFF.
Stage2기록시간14974.3994초=4h9m34s,Stage1기록49245.3658초=13h40m45s. 표의4h9m은Stage1/공개pretraining/최종평가미포함.
PDMS82.4852는내부개발1024중1021유효/40recording,ADE/FDE1024장면/world256clip진단. 공식navtest아님.
Stage2 1epoch/1seed의경향비교이며수렴·최적예산으로해석하지않는다. 현재DrivoR joint95training패널과단순수치비교금지.
docs/lpwm_planning_experiment.md 마지막절과HANDOFF1–5에재확인근거추가. 역사적결과/config/체크포인트변경없음.
18:06조회현재학습4338/40350 epoch3계속. 새학습/추가epoch/queue변경없이과거이력만확인했다.

## 2026-10-07 18:15 KST — Stage1 20 epoch의 실행 속도 감사

사용자가 Stage1 20epoch가 어떻게 빨리 끝났는지 질문했다. 실제 summary/manifest/log/trainer와 공식 LPWM forward 및 현재 joint 코드를 확인했다.
Stage1은23,126개 완전한 전방12프레임128×128 RGB 클립 /122recording /유효batch16 /epoch당1446 /총28920update였다.
기록49245.3658초=13시간40분45초, epoch 약41분. 로그 update 중앙값1.5705초, 검증·시각화·저장 포함 평균1.7028초.
원래 공개 Sketchy 가중치에서 SSL post-training, 전체 시퀀스 latent transition을 dyn_module 한 호출로 계산했다.
Encoder6035191/context39389417/dynamics59869416/decoder4251239 모두 학습 대상이고 각 모듈 parameter 변경 샘플218/154/224/92 확인.
로그 비유한loss0. 입력 cache는 RGB 픽셀이지 고정 encoder feature가 아니다. Planner/online candidate oracle은 Stage1에 없었다.
Manifest train/missing_sequence_RGB52749개 제외. 20epoch는 가용 전방12프레임 클립 기준이며 현재4카메라 trainval103288장면 기준이 아니다.
현재4카메라 순차 encode·각8단계 contextprior/dynamics rollout·activation checkpoint backward 재계산·DrivoR candidate generator/scorer/oracle 수행.
18:12:38조회4354/40350, 최근100update23.5139초(약10.54h/epoch). 마지막 phase forward5.9901/oracle2.1112/backward13.1293초.
서로 다른 시점/부하 관측이므로 원인별 속도배수는 미확정. LoRA parameter 감소로 forward/backward activation 계산이 없어지는 것은 아니다.
근거 results/lpwm_navsim_full_posttraining_v2/stage1_training_speed_explanation_20261007.json을 새로 보존했다.
HANDOFF1–5 및 기존 실험문서에 설명 추가. 새 GPU 작업·과거결과수정·실행 source/config/학습량/queue 변경 없음.
기존25epoch·매500/epoch 진단·후속평가를 유지하며 이번 질문으로 Stage1 확대/재학습/중단을 자동 실행하지 않는다.

## 2026-10-07 18:50 KST — 이전 방식과 현재 방식의 선택 판단

사용자가 이전의 짧은 학습/높은PDMS를 근거로 이전 방식이 더 나은지 질문했다. 상태 변경 요청이 아닌 방법 판단 요청으로 처리했다.
README/AGENTS/HANDOFF와 완료 네방법·frozen·Stage1 효과 대조 및 현재4000 report/실제config를 읽었다.
82.49는1021내부개발장면/40recording,75.83은95학습장면/24recording이고 planner·입력·초기화·학습량/시드가 다르다.
이전은512고정후보 모방/subscore BCE,현재는공식DrivoR generator/scorer. 원시PDMS의 차이로방법우열을 판정하지 않는다.
동일이전planner/개발조건 public-frozen78.916080→Stage1-frozen82.523785,차이+3.607705점 CI[+1.506478,+5.853857].
Stage1 Adapter82.485196−Stage1-frozen82.523785=−.038589점 CI[−1.164209,+1.007058]. Adapter PDMS 추가이득미확인/동등성미입증.
Stage2 frozen기록5226.381초=1h27m6s,Adapter14974.399초=4h9m34s. 공통Stage1 약13h41m와 준비·최종평가가 별도다.
Stage1+frozen-planner를 유용한 실용기준선으로 권고. Frozen은planninggradient로particle을 수정하는 연구가설의 증거가 아니다.
현재joint의 추가비용에따른성능이득미확인이고추가학습으로해결될것이라보장하지 않는다.
후속은공통DrivoR planner/입력/학습예산 아래 Stage1 초기화와planning적응을분리. FiLM조건명시,SSL유지는별도요인검증.
동일토큰/상황별지표·pairedCI/표현검사·최종고정checkpoint 독립navtest가필요. 현재학습navval/95패널은독립validation아님.
이번조회에서저장progress4450/40350 epoch3확인. 새GPU학습·평가·중단·교체·queue등록·source/config변경없음.
HANDOFF1–5와실험문서에판단근거를추가했으며기존25epoch/매500·epoch진단/후속평가를유지한다.

## 2026-10-07 19:01 KST — 중간 결과 점검, 4,500 대기

사용자 중간 결과 요청에 README/AGENTS/HANDOFF와 학습progress·queue·monitor·기존4000 report를 읽었다.
19:00:43 기준4480/40350 epoch3,약2.776/25epoch/전체11.103%. 유효64(micro16×accum2×GPU2)/loader2/oracle8 유지.
GPU0·1각38457MiB=40.325GB,util34/95% 단일시점표본. 전체48decimalGB 이내다.
Sandbox ps에host PID가노출되지않아승인된읽기전용host ps로train3144180/queue3144181/monitor3144182/publisher2507743 생존확인.
양rank4480 전체로그 비유한loss0,4400 최근18gradient그룹 모두유한·양수, nativehash 유지.
19:01:19 원래279/현재oracle8실행283sourcehash 모두일치. 실행source/config변경없음.
Monitor4000완료→4500대기이며 새PDMS/표현성능평가 없음. 마지막4000 PDMS75.8303/직진89.04·좌69.51·우61.16은95학습패널이다.
Particle초기중심1.979px/크기14.098%,appearanceF1.4114/전체.4230,미래대체12scene ADE+.2444m CI[−.1981,+.8354]를기존근거로구분.
최근50/100/200wall23.0486/22.9308/23.2408초,4500오늘19:08/4842오늘21:19–21:21 학습경계 예상(진단처리별도).
V1끝10월17일07–11시외삽이며후속fullnavtest/V2/EPDMS시간미포함/공유부하변동가능.
결과 results/lpwm_drivor_planning_path_lora_v1/intermediate_status_20261007_1900.json에조회시각·progress·health·process·GPU·hash·ETA근거보존.
README/HANDOFF1–5/학습문서/RESUME_NOTES 갱신. 새GPU작업·학습/queue/monitor/publisher/source/config변경없음.
기존25epoch/매500·epoch진단/후속평가유지. 새상태로포화·미래정보이득·방법우열을판정하지 않는다.

## 2026-10-07 19:10 KST — Stage1 epoch별 기록과 속도 차이 감사

사용자가과거20epoch Stage1의epoch별PDMS를현재와비교해달라고한뒤큰LPWM인데속도차이가왜나는지추가질문했다.
실제validation_log0–20/summary와trainervalidate_epoch,stage1root전체JSON/JSONL/LOGPDMS검색으로epoch별PDMS없음을확인.
당시SSLonly/no planner/official_elbo true/planning_loss false. 매epoch512개발clip의loss/KL/PSNR검사였고주행경로출력없음.
Loggedloss0/1/5/10/20=64.8380/23.3991/21.1328/20.3581/19.6968,loggedPSNR13.1449/20.4611/20.8349/20.9450/21.0768.
같은epoch0동일행중복1개는새CSV에서만제거/원래로그보존. Stage1epoch01–20checkpoint모두존재확인.
현재PDMSepoch1=67.9864/2=74.5770/4000(2.4783epoch)=75.8303,95training패널. 82.49는Stage1epoch20후Stage2Adapter1epoch개발결과다.
CPUplot3패널(과거SSLloss/PSNR·현재PDMS),PNG/PDF/전체epochCSV/currentCSV/감사JSON/sourcehash/generator를새results디렉토리에보존.
경로results/lpwm_navsim_full_posttraining_v2/stage1_epoch_metrics_vs_joint_pdms_20261007/. PNG시각검사완료,GPU작업없음.
현재도RGB npy pixelcache사용을확인. 과거cache가상대속도의주원인이라는해석은정정,128²해상도/cache공통이다.
Update당카메라시퀀스16(batch16×1)→256(batch64×4),각시퀀스시간입력/graph가달라16배FLOPs라고주장안함.
Stage1실제영상latent전이한dyn_module호출 vs현재4camera각8step sample/history반복context·particleTransformer/activationcheckpointbackward재계산.
추가DrivoRgenerator/scorer/CPUonlineoracle/native동결·gradient검사. 최근forward약6/oracle2–3/backward약13초근거확인.
Stage1native전체109.545M학습/현재native동결+LoRA4.684M이며학습가중치수와gradient전달계산량을구분했다.
Stage1epoch별PDMS확인은각checkpoint뒤동일planner동일예산학습필요. 0/20기존결과있고1/5후속검토안미실행.
Epoch20학습planner에encoder만교체하는검사는분포불일치가섞이므로각epoch최종planning성능으로부르지않는다.
19:07 저장progress4498/40350 epoch3/monitor4000완료→4500대기확인. 이후상태는새로조회해야함.
HANDOFF1–5/실험문서/RESUME_NOTES 갱신. 기존25epoch·source/config·학습/queue/monitor변경 및새GPU작업/epoch별planner재학습없음.

## 2026-10-07 19:20 KST — 4,500-update 최신 PDMS 직접 평가

사용자 최신업데이트기준PDMS 요청에README/AGENTS/HANDOFF·training/monitor·GPU를확인,4520본학습/4500표현진단완료/새PDMS없음확인.
Preservedexact4500checkpointSHA9adeab74b561fa785927c9d86be6493582000b11b7da85d9d2d61832804613bd와attributes SHA검증.
기존evaluate_lpwm_intermediate_panel_pdms.py --updates4500 --workers2로동일95training장면/24recording의공식NAVSIMv1 평가실행.
Sandbox첫시도는NVMLexit9로GPU초기화전실패,require_escalated host권한동일명령으로완료. 본학습중단/설정변경없음.
PDMS79.8876786,4000대비+4.0573841 CI[−3.0805198,+10.9129468]. 첫epoch대비+11.9012621 CI[+4.4353382,+19.6440688].
Epoch2대비+5.3107243 CI[−.7217334,+12.4843968],3500대비+3.9837429 CI[−.6487153,+8.6572802].
같은tokenpaired/5000recordingbootstrap/seed71. 95training패널이며navtest·독립validation·학습시드불확실성추정아님.
직진91.009912/좌74.945567/우67.064835,expertADE1.957827m. 4000의89.044452/69.508018/61.158954/ADE2.031890m보다좋은pointestimate.
NC.963158/DAC.936842/progress.685725/TTC.915789/comfort.989474. NC·comfort소폭감소도기록해평균개선만으로안전일괄개선주장안함.
95성공/실패0/12재생ADE차이0/nativehash유지/최대card41.259368GB/등록279·283sourcehash모두일치. 일회평가종료·GPU해제.
19:20학습4530/40350 epoch3,양rank비유한loss0/4500의18gradient그룹유한양수/nativehash불변. 새79.89는4500모델결과이며4530성능으로부르지않음.
Particle초기중심2.170891px/크기15.947790%,직전4000→4500중심1.284820px/size7.618933%,wholeF1.420444(직전.422994).
미래대체12scene selectedoracle−.377887 CI[−.557280,−.172739]/fixedoracle−.123822 CI[−.260985,−.010708]/ADE+.392341m CI[−.076664,+.985058].
Oracle개입은공식PDMS차이가아니며물리미래정확도·LPWM단독효과·독립이득미확인/OOD및반복검사한계유지.
새results/lpwm_drivor_planning_path_lora_v1/intermediate_update4500_20261007/에정확score/summary/replay/registration/pairedCI/표현/전후겹침/PDMSplot/reportgenerator보존.
PDMSplot시각검사완료. README/HANDOFF1–5/학습문서/RESUME_NOTES갱신,기존25epoch/유효64·loader2/oracle8/queue/monitor/source/config변경없음.


## 2026-10-07 19:43 KST — LoRA planning 학습 속도 개선 가능성 점검

**2026-10-07 19:43 KST — LoRA 학습 속도 병목과 SDPA 후보 점검.**
본학습 최근100update는23.268초/update, 순전파26.42%·역전파60.74%·CPU oracle11.85%·기타0.99%다.
Loader 대기는 미미하고 GPU0·1 각40.325GB, micro16×누적2×2=유효64/loader2/oracle8 유지.
별도4500checkpoint 한 장면·한 카메라·8단계 particle VJP 검사: attention21개 SDPA의2.002초 대 기존2.120/2.215초(탐색적5.6–9.6% 단축).
FP32 dropout-off 출력 최대차1.72e-5,15 LPWM/명령 gradient그룹 최대상대L2오차0.0576%, native불변. 실제loss/DDP속도·PDMS 검증은 아니다.
진단용4GiB allocator 두 번의OOM은본학습에영향없음;6GiB허용 재검사완료·최대카드45.009GB·GPU반환. 본학습 4590/40350 계속.
원래279/실행283source 불변, 본학습설정·queue·monitor 변경없음. SDPA→선택적checkpoint→compile→oracle중첩의별도검증을권고.
근거 `results/lpwm_drivor_planning_path_lora_v1/training_speed_audit_20261007/assessment.json`.

진단전4455–4554의wall23.2677초/update. SDPA는본학습미적용이며동일batch16/DDP/officialloss의보존재개비교가다음gate다.
선택적checkpoint/compile/oracle중첩/microbatch증설을후속우선순위로기록했다. 학습목표/gradient/horizon/데이터축소없음.


## 2026-10-07 21:33 KST — 실행 최적화 적용 및 본학습 계속

**2026-10-07 21:33 KST — 속도 최적화 적용, 4,633에서 본학습 재개 후 4,807 확인.**
사용자 적용승인에같은fullstate/실제DrivoRloss/GPU2/micro16/유효64로44개DDP비교update(12+12+12+8)를실행했다.
기존전후중앙값평균25.4215초 vs oracle16+일괄gradient유한성검사24.1568초,약4.98%단축. SDPA포함24.0000초는추가.65%라미채택.
채택설정 `configs/lpwm_drivor_optimized_execution/batched_checks_oracle16.json`,새trainer `scripts/train_lpwm_drivor_optimized_execution.py`.
4,633 model/AdamW796state/scheduler/양rankRNG보존·복원,비교weights폐기. 20:24부터train3437947/queue3437948/monitor3437949 실행.
21:33확인174실제update추가,wall23.4624초/update·최대카드40.1615GB,원래279/실행283/새291source정상.
LoRA전체경로·명령·4카메라·미래8단계·loss·25epoch동일. Native/evalattention/정기진단유지,다음4842대기.
남은V1약9.65일(현재부하가정10/17 13:12KST),최종평가/V2별도. 새로운PDMS/성능향상검증은아니다.
근거 `results/lpwm_drivor_planning_path_lora_v1/optimized_execution_20261007/report.json`.

복원checkpoint SHA fafd39b3fd68f47577db7e4c0ef08abc505ead221a74e88bea352b0747b58cab.
Oracle16+일괄유한성채택,SDPA는실배치추가.65%라미채택. 원본재검도후속loss수치차이가있어bitwise동일학습을주장하지않는다.
새등록291source는실행중수정금지. 비교용44updateweights는폐기했고원본4,633부터재개했다.

21:40 추가검사: 새실행4800체크포인트의AdamW796state step·scheduler step이모두4800,양rankRNG2개저장확인.
실제state_dict의 `planner.image_backbone.world_model.` 아래native1070항목으로재계산한SHA가원본a5dd2345…와일치했다.
`production_checkpoint_integrity.json`에보존. 첫수동검사에서property이름을stateprefix로쓴조회오류는실제저장prefix로정정했으며모델가중치변경이아니었다.
21:40:56 본학습4827/40350계속,원래279/283·새291source불변확인.


## 2026-10-07 22:08 KST — epoch3 공식PDMS와표현검사중간보고

**2026-10-07 22:08 KST — epoch3 공식95장면PDMS80.3450, 학습4,898/40,350 계속.**
Exact4,842 checkpoint 평가95성공/실패0,첫epoch67.9864→epoch2 74.5770→epoch3 80.3450.
4,500대비+.4573점 CI[−3.4353,+4.9538],직진87.6751/좌77.9405/우70.8283·ADE1.6212m. 직진은91.0099에서하락.
동일95training scene/24recording,navtest나독립validation아님. 12scene미래반복개입ADE+.5028m CI[+.0642,+1.1667]이나인과기여/미래정확도미확정.
96scene geometry초기대비2.1631px/크기17.8007%,whole판독F1.40909(4500 .42044),도로집중미확인.
양rank비유한loss0/최근4800의18gradient그룹유한양수/nativehash·원래279/283/새291source불변. GPU평가최대41.2594GB·종료반환.
본학습train3437947/queue3437948/monitor3437949/publisher2507743 유지,기존micro16×누적2×GPU2=유효64/loader2/oracle16.
최근50–200update22.81–23.04초,V1남은약9.36–9.45일(10/17 07–09시KST외삽). 후속평가·V2시간별도.
근거 `results/lpwm_drivor_planning_path_lora_v1/intermediate_epoch3_20261007/report.json` 및전후·겹침/PDMS그림.

Exact4842 PDMS80.3450018,epoch1대비+12.3585853 CI[+6.3251604,+18.5155639],epoch2대비+5.7680475 CI[+.0731959,+11.9065914].
4500대비+.4573232 CI[−3.4352546,+4.9538021],4000대비+4.5147074 CI[−.5885301,+9.3744147]. 24recording paired bootstrap5000/seed71,반복중간검사.
직진87.675135/좌77.940529/우70.828282,ADE1.621154m. NC.984211/DAC.926316/TTC.936842/comfort1.0. 4500대비직진/DAC는하락했다.
12scene future-repeat-current ADE+.502778m CI[+.064170,+1.166741],selectedoracle−.119726 CI[−.255945,−.014017]. 의존성신호/대체OOD한계.
96scene geometry중심2.163119px/크기17.800714%,의도변경중심.443236px. Whole readoutF1초기.379859→4500 .420444→4842 .409089.
4s 미래변위readout오차 current8.242523→future7.943212m,차이−.299311 CI[−.583566,−.062750]이나zero-displacement6.726962m보다큼. 미래정확도성공으로부르지않는다.
차량중심비율초기8.88265→9.11051%,보행자1.20036→1.29801%,도로proxy20.03289→19.43257%. 뚜렷한주행중요영역재배치미확인.
평가12ADE 재생차이0/nativehash유지/최대카드41.2594GB. 원래279/실행283/새291source 전부일치,학습양rank비유한loss0.

Sandbox NVML조회exit9로GPU초기화전에실패한첫시도후같은평가를host승인으로재실행해완료했다. 평가완료후GPU반환,본학습무중단.


## 2026-10-07 23:03 KST — 이전 Adapter Stage2 추가 epoch 실험 준비 및 병행 실행 구성

사용자 요청은 과거 Adapter 모델을 이어서 더 학습하면 PDMS가 상승하는지 확인하는 것이다.
해당 최종 기록은 81점이 아니라 **82.4852점**이다. Stage1 20epoch 이후 Stage2 1epoch/4,707update/seed47,
75,297개 navtrain 학습 장면을 사용했다. 기존 1,024개 development 장면 중 공식 candidate score가 있는 1,021개,
40개 recording의 PDMS이며 전체 navtest가 아니다.

### 고정한 비교와 추가 학습

- 기존 model과 AdamW 124개 state를 그대로 이어 받는다. 원본 `latest.pt` SHA는
  `01f6d28a0d374ae28cb229ca8092a9a9210a2e274689ca36ccfac9a3868a01e4`이고, 평가된 `checkpoint.pt`와 모든 tensor가 일치했다.
- 우선 **2epoch를 추가해 총 3epoch**까지 진행한다. 총 2epoch(9,414update) 학습→검증→총 3epoch(14,121update) 학습→검증 순서다.
- 1epoch cosine schedule의 마지막 학습률인 LPWM 약1e-6, planner 약3e-5를 유지한다.
  새 3epoch cosine으로 재계산하거나 warmup을 재시작하지 않는다. 처음부터 3epoch로 학습한 실험과 구분한다.
- 기존 Adapter704,960개와 planner/command2,211,975개를 학습하며 native LPWM은 고정한다.
  동일 candidate imitation + 6개 PDM subscore BCE +0.02 SSL을 유지하고 직접 객체 GT를 넣지 않는다.
- 초기 두 GPU 구상은 batch1×누적8×GPU2였으나 아래 자원 충돌 이후 **GPU0 단독, batch1×누적16**으로 변경했다.
  유효 planning16/world8을 유지한다. SSL은 두 번째 microbatch마다 1clip을 계산하고 그 microbatch의 SSL weight를2배로 해
  전체 optimizer update에서 원래 global mean과 샘플 수를 유지한다.
- 각 epoch에서 이전과 같은 planning1,024/world256 panel, 같은 inference batch4로 PDMS/ADE/FDE/metric calibration,
  미래를 현재로 대체하는 개입 및 영상 복원·미래 LPIPS를 평가한다. 완료 검증 뒤 다음 epoch로 진행한다.
- GPU1 및 GPU0의 각각16장면 원본 checkpoint 재생에서 candidate 선택이 모두 동일했고 ADE 차이가0이었다.
  미세 배치·GPU 수 변경으로 dropout/SSL 샘플 묶음이 달라져 bitwise 동일한 학습 연장은 아니다.

### 메모리 검사와 실제 충돌 이력

당시 본학습은 각 카드40.33GB를 사용했다. 이전 Adapter batch8은 peak allocation22.31GiB라 병행하기 어려웠다.
배치2/누적4를 프로세스당5.5GiB 제한으로 검사했으나 역전파에서 그 제한에 도달했다.
먼저 LPIPS를 2프레임씩 계산하고 activation checkpoint를 적용했다. 실제 LPIPS CPU검사에서 loss 최대차1.49e-8,
입력 gradient 차이0이었다. 영상 loss나 gradient 연결은 유지했다. 이 방식도 batch2의 미래 rollout 역전파에는 부족했다.
배치1/누적8의 실제 DDP4update는 성공했고 최대 카드점유45.654GB, native 가중치·buffer 불변,
image encoder/context/dynamics/planner gradient 유한·양수였다. 이 profile의 학습 가중치는 폐기했다.

이후 두 GPU 병행 학습을 시작했지만 **다른 사용자의 GPU1 프로세스(PID202240)가 진입해 약5.49GiB를 추가 점유**했다.
GPU1 전체 점유50.8475GB에서 새 Adapter rank1이 OOM을 만났고, 기존 본학습도 4,988update 경계에서 보호 중단했다.
새 Adapter에서 완료·저장된 추가 update는0이다. 다른 사용자의 프로세스는 수정하지 않았다.
우리 새 torchrun198149만 종료하고 원래 두 GPU 실험은 pause 상태로 보존했다.

기존 본학습의4,988 model/AdamW796state/scheduler/양rank RNG를 보존하고 native1070항목 hash를 재검사했다.
모두 정상이라 원래 micro16×누적2×GPU2=유효64/loader2/oracle16 설정으로 복구했다.
새 본학습240919/후속queue240920/표현monitor240921이며4,990 이후 재개와5,000 정기검사 진입을 확인했다.
`paused.json`의 memory_exceeded=false는 rank0의 국소 값이었다. 실제 rank1 로그는48GB 초과이므로 이력을 함께 보존했다.

### 채택한 GPU0 대기열

- 설정: `configs/lpwm_planning/adapter_epoch_extension_single_gpu_v2.json`.
- 실행: `scripts/queue_lpwm_adapter_epoch_extension_single_gpu.py`, queue326218.
- Root: `outputs/lpwm_adapter_epoch_extension_single_gpu_v2/`. 원래 두 GPU root v1은 중단된 이력이다.
- 새 source310개를 등록했으므로 실행 중 해당 trainer/evaluator/queue/memory helper/config를 수정하지 않는다.
- GPU0 점유가120초간 안정되고 기존 정기검사가 없을 때 실행한다. 기존500update/epoch 표현검사가 다가오면
  새 Adapter가 update 경계에서 저장·종료해 GPU를 비우고 검사 완료 후 재개한다.
- GPU0의 다른 compute process 또는47.2GB 이상 점유를 감지하면 **자신이 소유한 새 process group만** 즉시 중단한다.
  마지막 온전한 checkpoint로 돌아가며 부분 update는 폐기한다. 다른 사용자 작업은 건드리지 않는다.
- 새 학습은 첫 추가 update와 이후8update마다 model/optimizer를 저장한다. 각 저장에서 native hash·optimizer step을 검사한다.
- 공유 GPU를 다른 작업의 미래 진입으로부터 예약하는 기능은 없다. 이 감시는 이후의 충돌 가능성을0으로 보장하지 않는다.

설정·profile·실패·복구·baseline 재생 근거는
[실험 준비 보고](../results/lpwm_adapter_epoch_extension_single_gpu_v2/setup_report.json)에 있다.
추가 epoch의 PDMS 결과는 아직 없으며 기존82.49점을 새 결과로 보고하지 않는다.


## 2026-10-07 23:17 KST — GPU0 Adapter 추가 학습 시작과 저장 상태 검증

- GPU0 단독 Adapter queue326218/train400195로 원래4,707에서4,716까지 추가9update 진행했다. 기존 GPU0·1 본학습은5,043이며5000정기표현검사를마치고5500을기다린다.
- 4,712 fullstate를 별도 hardlink `outputs/lpwm_adapter_epoch_extension_single_gpu_v2/startup_audit_resume.pt`로 보존하고 CPU 검사했다. SHA `d3db67f3645fbfe78682022e6fa17e37a154d0319a55ddc36e9b0e2bf5461197`, AdamW124상태모두step4712, 원본대비학습대상124tensor모두변화, frozen weight/buffer변화0이다.
- 단순 state 이름 대 inventory 비교는 shared context Adapter alias48개를 별도 가중치로 오인했다. 동일 storage pointer/shape로 canonical학습tensor와동일함을확인해진단을정정했다. 학습가중치변경이나freeze수정은없다. Native SHA `18e5ac965f13c9be0a7930a64664639c291c75c32018adc9f16ae904b47c14cb` 유지.
- 추가9update loss/gradient norm은모두finite. 첫update preclipnorm16983.10에clip5를적용했고이후관측최대31.605다. 훈련손실만으로planning개선을판정하지않는다.
- Batch1×누적16/GPU0단독=유효planning16/world8,원래마지막LR 약1e-6/3e-5를상수유지한다. LPIPSframe chunk의기존gradient검사와weighted global objective대응검사는통과했지만microbatch/RNG재구성이므로역사적batch8실행과bitwise동일하지않다.
- 현재GPU0약45.78GB,병행관측최대46.90GB;GPU1본학습약40.16GB. 앞선두GPU시도의50.8475GB/OOM/primary4988저장중단·복구는별도실제사건으로보존하며이번단독실행메모리수치로덮지않는다.
- 초기후속8update wall62.8405초/update. 추가epoch2완료까지약82시간(10/11오전),epoch3까지약164시간(10/14저녁)으로외삽된다. 정기진단GPU양보/epoch평가/공유부하시간이제외돼있으므로확정ETA가아니다.
- Epoch2(9,414)→같은1024planning/256world평가→epoch3(14,121)→동일평가대기열유지. 기준82.4852와대응PDMS차이·상황별지표·미래유지검사를비교한다. 전체navtest아니며,추가epochPDMS는아직없다.
- 기존279/283/291source와새307/310source등록hash전부불변확인. 실행중trainer/evaluator/queue/helper/config추가수정없음.
- 공유근거: `results/lpwm_adapter_epoch_extension_single_gpu_v2/runtime_started_report.json`, `parameter_inventory.json`, `resume_check.json`, `effective_batch_weighting_check.json`.


## 2026-10-07 23:29 KST — Adapter 추가 학습 조건 동일성 설명

사용자는 추가 학습이 기존과 동일하고 성능에 영향을 주는 변경이 없는지 물었다. 코드를 대조한 결과 모델·학습대상·75,297장면·planning 순서·loss항/가중치·유효planning16/world8·optimizer복원·precision·평가패널은 유지되지만 완전한 실행 동일성은 성립하지 않는다.

- 기존8×누적1×GPU2에서1×누적16×GPU1로 바뀌었다. Seed47명칭은 같아도 실제 per-update seed공식이47+update×2+rank에서47+update로 달라지고 dropout·SSL latent/random clip 추출도 달라진다. 유효배치 동일성만으로 최종PDMS 동일성을 보장하지 않는다.
- 기존1epoch의1%warmup/cosine은끝났다. 추가epoch는마지막Adapter약1e-6/planner약3e-5를상수유지한다. 처음부터3epoch용cosine을설계한학습과다르며,낮은LR로개선이없어도추가epoch의가능성이나모델상한을기각할수없다.
- LPIPS2frame chunk/checkpoint는국소loss·입력gradient검사를통과했으나전체학습gradient나PDMS비열등성을검증한것은아니다. 기존16scene재생일치는재개전inference확인이다.
- 현재실험은기존checkpoint의낮은LR연장효과로해석한다. 엄밀한epoch수만의통제효과나성능중립적실행변환으로표현하지않는다.
- 조회4727update,추가epochPDMS아직없음. 이번요청은조건설명이며기존학습/queue/config/source변경없음. 등록310source모두일치.
- 근거 `results/lpwm_adapter_epoch_extension_single_gpu_v2/condition_comparison_audit.json`.


## 2026-10-08 00:06 KST — Adapter 원래 batch 복원과 병행 자원 재배치

사용자 요청: 기존약40GB본학습의batch를줄여서라도Adapter를기존설정으로GPU0·1에서이어가되두작업의성능에영향을주지않는범위로재개. 후속질문에최종성능동등성은미검증이라고명시했다.

### 보존과 실제 재개

- 본학습5,084 model/AdamW796state/scheduler5084/양rankRNG를보존했다. SHA `9d8c7a3098576cbf568623be4b30487c4575458eb29f9f2e186f659968a38498`.
- Adapter4,733 model/AdamW124state보존. SHA `3b3d685c35142fa9b2dbff395a07eeea7acb0ad17d71e7262f22959fc4b92d1f`. 이전4708–4733의singleGPU26update를되돌리지않았다.
- 본학습micro16×acc2×GPU2에서micro4×acc8×GPU2로변경,유효64·학습률/scheduler·데이터·학습목표·25epoch·LoRA/FiLM/planner그대로다. Nativehash유지. PID875422/queue875423/monitor875424.
- Adapter는micro8×acc1×GPU2,SSL4clip/rank/update,원래fullLPIPS,원래rank별seed공식으로복원했다. AdamW/LR1e-6/3e-5상수연장정책보존,profile가중치미사용. Queue918561/train921639,root `outputs/lpwm_adapter_original_batch_shared_v4/`.
- 00:03조회본학습5,090,Adapter4,811까지실제진행. 추가PDMS는아직없다. 같은1024dev/1021유효PDM·256world에서epoch2→평가→epoch3→평가.

### 검사와 해석 한계

- 본학습batch16 vs4는같은checkpoint/scene/학습률이며첫loss차rank0 .05454/rank1 .06794,gradientcosine .685716/relativeL2 .76772. Dropout실제draw가달라져동일gradient검사가아니며최종PDMS비열등성입증도아니다.
- 원래batch16 CPU saved-tensor offload는첫loss일치/gradientrelativeL2 2.043e-5이나메모리40.16GB로거의감소하지않아미채택. 원래batch16+Adapter8동시상주를48GB에서보장하는방안은확보하지못했다.
- Adapterbatch8실제4updateprofile은단독약24.74GB/GPU,본학습batch4는12.345GB/GPU. 병행4update검사에서최대GPU0 37.0525GB/GPU1 43.4656GB(타인작업포함),OOM0. 모든profileweights폐기.
- 현재본학습손실·gradient는유한하지만성능무영향으로확정하지않는다. 사용자에게엄밀하게양쪽GPU별batch까지보존하려면교대실행이필요하다고설명했다. 동시/교대선호질문을제시했으나명시응답이없어기존배치축소승인범위의동시실행으로재개했다.
- Adapter첫profilepreclipnorm659448.625에기존clip5적용,이후22.18/8.64/6.96;초기재개gradient크기를누락하지않는다. 현재학습성능은epoch검증을기다린다.

### 큐와 후속 복구

- Adapterprimary315/새319sourcehash불변. 준비v3는profile만실행;production은shared_v4다. 이전v2singleGPU는4733pause보존.
- baseline전체카드21GB미만의기존외부작업은안정30초후허용. 신규외부PID 또는47GB도달시Adapter자기child process group만중단하고8update주기의온전한checkpoint에서재개. 타인작업수정없음. 독립적인외부할당을예약차단할수는없다.
- 본학습정기500/epoch진단은Adapter가checkpoint양보하고재개한다. 기존fullnavtest→freshV2→EPDMS후속유지.
- CPUwatcher967796 `scripts/restore_lpwm_primary_batch_after_adapter.py`/316sources는Adapterepoch2/3학습·평가완료후본학습의최신model/AdamW/scheduler/RNG를저장하고원래batch16×acc2로복원한다. 사용자pause/다른실행변경을존중한다.
- 공유근거 `results/lpwm_adapter_original_batch_shared_v4/rebalance_and_resume_report.json`,각profile로그/체크포인트무결성/gradient비교.


## 2026-10-08 00:16 KST — 두 학습 중간 결과 조회

**2026-10-08 00:16 KST — 두 학습 중간 결과 확인.**
본학습 5,100/40,350 update(4번째 epoch), Adapter 5,014/14,121 update(2번째 epoch, 기존4,707 이후 +307회). 두 GPU에서 함께 진행 중이다.
최근 완료 PDMS는 본학습 epoch3/4,842의80.3450(고정95학습장면); Adapter82.4852는기존1epoch 기준(내부dev1,021유효장면)이며추가epoch평가는아직없다. 서로직접순위비교하지않는다.
5,000 표현진단: 초기대비중심2.277px/크기14.084%,readout F1 .37986→.43893. 미래반복개입ADE−.0201m CI[−.2825,+.2560]으로일관된미래이득은미확인.
현재GPU당약37.05GB/48GB, 양rank재개후기록loss비유한0. Adapter최근wall약3.9초/update; epoch2학습잔여약4.8시간(진단대기·평가별도) 초기외삽.
본학습 micro4/유효64,Adapter batch8/유효16과기존queue·복원watcher유지. 이번요청은조회이며설정·학습·평가작업을새로변경하지않았다.
근거 `results/lpwm_adapter_original_batch_shared_v4/intermediate_both_runs_20261008_0016.json`.

두 학습 양rank의 현재재개이후 loss 기록에서 비유한값0. 등록된다음검증을기다리며새평가를시작하지않았다. GPU별기존batch와동일성능은여전히보장하지않는다.


## 2026-10-08 07:08 KST — Adapter 추가1epoch 완료와 두 학습 중간 결과

**2026-10-08 07:08 KST — Adapter 2 epoch 검증 완료: PDMS 82.4852 → 83.5279.**
동일 내부 개발1,024장면/유효1,021점수/40recording, +1.0427점(대응95%CI[+.1716,+2.0117]). ADE1.1551→1.1433m/FDE2.7567→2.7021m이며두오차차이CI는0포함.
9,414 checkpoint 평가05:09완료 후현재Adapter 11,380/14,121,3번째epoch학습중. 미래현재반복80.4210 대비정상83.5279(+3.1068점);미래분기의존성검사이며재학습대조가아니다.
미래LPIPS는1epoch대비+.0003524(+.0895%,CI0포함),Stage1대비+.0018623(CI양수)로소폭악화. 기존모델전체/표현의단독기여가분리된결과는아니다.
본학습 5,393/40,350,4번째epoch;최신PDMS는기존4,842의80.3450(95학습장면). 5,000표현진단이후새결과없고다음5,500대기.
두학습양rank현재재개이후loss비유한0,본학습5,300의18gradient그룹유한양수. GPU0·1각37.05GB. 기존설정·queue·원래batch복원watcher유지.
Adapter3epoch학습잔여약2.7시간,본학습5,500경계약2.5시간(최근200회속도외삽,진단양보·평가별도).
근거 `results/lpwm_adapter_original_batch_shared_v4/intermediate_both_runs_20261008_0708.json` 및 `results/lpwm_adapter_original_batch_shared_v4/epoch02_summary.json`.

양rank loss와최근gradient유한성을확인했다. 본학습과Adapter의평가패널이달라직접비교하지않으며,world위험계층별값은원래epoch02_summary에보존했다. 실행중인등록source/config는읽기만했다.


## 2026-10-08 07:30 KST — 공통navtest비교기동

**2026-10-08 07:30 KST — 동일 장면 비교 실행 중: Adapter2epoch vs LoRA5,400update.**
사용자는 같은 평가 장면을 요청했고, 이어 Adapter는2epoch으로 고정하라고 명시했다. 3epoch동일학습량 비교안은미실행준비이력이며현재조건이우선한다.
Root `outputs/lpwm_shared_navtest_adapter2_primary5400_v1/`, config `configs/lpwm_shared_navtest_comparison/adapter_epoch2_vs_primary_update5400.json`.
Navtest1,024scene/44recording을점수확인전고정,양쪽학습및Stage1과token/recording중복0. 입력은각학습전처리를보존한다. 전체navtest가아니다.
Primary5400SHA d7addc72…/Adapter9414SHA9df6bd02…고정. 서로다른planner·카메라/시간입력·데이터·학습량의시스템비교이며LoRA/Adapter단독효과로해석하지않는다.
323source/config등록불변. Queue966120,primaryeval970996. Adapter원래queue918561/train3842032는11,699fullstate저장후정상pause종료했다.
Pause소유권은새comparisonqueue이며`adapter_yield_ready.json`에기록한다. GPU추론완료또는오류시기존v4queue를동일config로자동재개,CPU공식채점은재개와병행한다.
본학습875422는계속실행. 평가중Adapterpause를임의삭제/별도재기동하지말고새queue상태를먼저확인한다. 원래batch복원watcher967796은일시대기한다.
기존83.53/80.35는서로다른장면점수이며공통결과가나오기전직접비교하지않는다. 새root/evaluation_complete.json이완료근거다.


## 2026-10-08 07:46 KST — 공통평가 완료 및 원래학습재개

**2026-10-08 07:46 KST — 공통1,024 navtest장면 비교 완료: Adapter2epoch81.6141 vs LoRA5,400update81.6096.**
두학습및Stage1과token/recording중복없는44recording/1,024scene,동일공식NAVSIMv1 scorer와8×.5s출력/40×.1s시뮬레이션. 양모델실패0.
LoRA−Adapter PDMS−.004515점,대응recording95%CI[−1.9981,+1.9968]. 우열근거없음이며통계적동등성입증으로부르지않는다.
ADE Adapter1.1632/LoRA2.3644m,FDE2.7367/5.4760m. LoRA무과실충돌·도로준수·TTC평균높음,Adapter진행률높음. 상황별PDMS 직진84.1161/84.4131,좌75.6400/76.2832,우73.8255/69.9490(Adapter/LoRA).
전체navtest아닌고정부분집합이다. 이전83.5279(Adapterdev)·80.3450(LoRA학습장면)과구분한다. 카메라/과거입력·planner·데이터·학습량·Stage1차이남음,두시스템비교다.
비교중Adapter11,699의124AdamW/마지막LR상태를완전저장해잠시GPU양보,추론완료후동일v4queue로자동재개했다. 현재queue1208639/train1212507,Adapter11,744;본학습875422는계속진행해5,419.
평가queue966120와GPU추론970996/1176048·CPU채점1208640은완료. 새source323개불변검사통과. 기존batch복원watcher967796은Adapter3epoch/검증완료를기다린다.
결과 `results/lpwm_shared_navtest_adapter2_primary5400_v1/evaluation_complete.json`,양모델CSV및resume검사동일폴더. 완료평가를재기동하지않는다.

## 2026-10-08 08:02 KST — 본학습 입력 명세 확인

사용자가 Drive-JEPA의 카메라/해상도/시간 프레임 표기처럼 현재 본학습 입력을 요청했다.
`scripts/prepare_lpwm_drivor_joint_data.py:24,68,85,153` 및 scene cache manifest에서 CAM_F0/B0/L0/R0 현재 RGB 각1장, 전체 이미지 BICUBIC128×128, 과거/미래 영상 입력 없음 확인.
`src/planning_aware_future_prediction/object_centric/lpwm_drivor_joint.py:83,105,148`에서 이미지 [batch,4,3,128,128], encoder 단일 관측, dynamics prior 미래8단계 생성 및 command FiLM 경로 확인.
Ego11은 현재 ego 좌표계 pose3(실제0), velocity2, acceleration2, command4다. 병행 Adapter는 전방1카메라의 과거3+현재1프레임(0.5초 간격), 각128×128이므로 두 종류의 4장 입력을 혼동하지 않는다.
학습 코드/설정/대기열은 변경하지 않았다. 설명을 위해 학습·평가를 새로 실행하지 않았다.

## 2026-10-08 08:05 KST — Adapter와 본학습의 관측 프레임 차이 이유

사용자는 두 실험의 시간 입력이 다른 이유를 물었다. 이전 Adapter는 NAVSIM Stage1의 전방 관측4장/미래8장 체계를 유지한 planner 적응이다. 새 LPWM+DrivoR는 사용자의 DrivoR 기반 비교 목적에 맞춰 공식 센서 입력의4카메라·현재1시점을 따랐다.
근거: `reference_repositories/DrivoR/navsim/planning/script/config/common/agent/drivoR.yaml:32`의cam_*:[3], `drivor_features.py:74`의cameras[-1], `configs/lpwm_navsim_adaptation/full_posttraining_v2.json`의observed_frames4, `lpwm_planning_finetuning.py:81`의4프레임 및 관측 context 기반 rollout.
LoRA/Adapter 자체가 시간축을 강제하는 것은 아니다. 단일 관측 미래 prior는 가능하지만 관측 간 변화를 사용하지 못하며, 과거 제거에 대한 성능 동등성 검증은 없다. 두 시스템의 공통 평가 점수를 미세조정 방식 단독 효과로 해석하지 않는다.
설명과 기록만 수행했으며 학습 입력/설정/대기열을 변경하지 않았다.

## 2026-10-08 08:09 KST — Drive-JEPA 사전학습과 planner 학습 확인

질문: Drive-JEPA도 이전 LPWM Adapter처럼 Stage1 사전학습 뒤 planner를 추가 학습하는가?
논문 https://arxiv.org/html/2601.22032v2 §3.2/4.2와 로컬공식repo548bb8215e3aae18e162a0f12f1ba83b4d3eb57e를 확인했다. 공개 V-JEPA2로 초기화하고 CoVLA/DrivingDojo/OpenScene trainval의330h 주행 영상에 JEPA SSL 적응(8프레임512×256/2Hz,50epoch)을 한 뒤 NAVSIM planner를 학습한다.
후속 학습에서는 encoder 원래 가중치도 갱신한다. Perception-based agent get_optimizers는 backbone_lr_mult .1×lr1e-4=1e-5와나머지1e-4;20epoch script다. Perception-free yaml 기본freeze_encoder:true를 실제train script가false로 override하며40epoch/Adam전체1e-4다. 논문perception-free MSE와공개코드 length-normalized L1 차이도 있어 loss를둘에서완전히동일하다고설명하지않는다.
우리 Adapter는 공개LPWM→NAVSIM SSL20epoch→native고정/Adapter+command+planner갱신 및SSL보조유지다. 단계적전이는유사하지만 SSL목표(latent JEPA vsRGB복원/예측),가중치갱신범위,planner입력(encoderfeatures vs명시적particle rollout)이 다르다.
웹논문은열람성공;개별GitHub웹blob은cache miss였으나같은파일을로컬공식clone에서직접확인했다. 학습/대기열/등록소스변경이나새평가없음.

## 2026-10-08 08:13 KST — 두 학습 중간보고

본학습5,437/40,350(3.369epoch),Adapter12,178/14,121(2.587epoch),GPU0·1 각각37.05GB. 재개후본학습353/Adapter479기록의양rank loss유한,본학습5,400의18gradient그룹유한양수,Adapterencoder/context/dynamics/planner gradient유한양수. RGBdecoder는원래고정이다.
기존본학습875422/queue875423/monitor875424,Adapter1212507/queue1208639,복원watcher967796 호스트실행확인. 샌드박스ps에서는PID가보이지않아호스트읽기전용재확인했다.
새평가점수는없음. 공통navtest1,024scene/44recording에서는기존Adapter9414 PDMS81.6141/ADE1.1632/FDE2.7367,LoRA5400 PDMS81.6096/ADE2.3644/FDE5.4760. Adapter내부dev1epoch82.4852→2epoch83.5279는별도패널이다.
최근50–200회벽시계속도본학습84.79–88.15초/Adapter3.63–3.69초. 5500경계09:42–09:46,Adapterepoch3학습10:11–10:13KST외삽이며진단양보·평가시간별도. 본학습의34일내부ETA는병행micro4속도를그대로외삽하므로예정batch16복원후ETA로사용하지않는다.
Particle최신5000진단은평균중심2.277입력px/크기14.084%변화,F1.43893이며다음5500대기. 실행·설정변경없이 `results/lpwm_adapter_original_batch_shared_v4/intermediate_both_runs_20261008_0814.json`과인수인계에기록했다.

## 2026-10-08 08:20 KST — 본학습 particle 변화 시각화 제공

사용자는본학습의particle변화를그림으로요청했다. 현재학습5,440이나완료진단은5,000이므로정확히그시점으로표시했다. 기존publisher의직진/좌회전/우회전4열(입력/전/후/겹침)과겹침단독PNG를 `results/lpwm_drivor_planning_path_lora_v1/particle_visualization_update5000/`에동일바이트로복사했다.
기존CPUscript `visualize_lpwm_vehicle_rich_particle_changes.py --updates 5000 --output results/lpwm_drivor_planning_path_lora_v1/vehicle_rich_particle_visualization_update5000` 실행완료. 이전3,000과같은scene41/21/27,동일입력/64particle번호,학습전presence상위16박스고정. 원영상→실제128입력pixel일치·유한attributes·checkpoint완료검사통과.
학습후checkpoint SHA971e8724f99f13b0ee024e28e9e3e19db8a820ea42ef8918144a188ba25ecef2. 차량장면앞카메라평균중심변화2.853/2.876/1.453입력px,크기변화13.138/12.625/9.466%. 전체96×4에서는2.277px/14.084%.
실제표준4열/차량4열/차량겹침PNG를열어확인했다. 청록학습전/주황학습후/흰화살표동일번호학습전후차이,겹침점반경고정·박스glimpse로설명한다. 물리적이동/검출/attention/planning이득으로해석하지않는다. 새GPU작업/코드수정/학습조건변경없음.

## 2026-10-08 — 변화량이 큰 particle 이미지 재시각화

사용자가큰변화사례를명시적으로요청하여5,000update의96장면×4카메라384이미지를평균중심이동/평균가로·세로상대크기변화/개별최대중심이동으로순위화했다. 기준별1위는scene21/left(4.0416px),scene94/left(23.4248%),scene84/front(최대21.9107px,평균3.1408px)다.
CPU전용 `scripts/visualize_lpwm_largest_particle_changes.py` 추가, `--updates 5000 --output results/lpwm_drivor_planning_path_lora_v1/largest_particle_changes_update5000` 실행완료. 과거출력/등록학습코드불변. 원본전체이미지BICUBIC128입력pixel일치·64×14attributes유한·완료checkpoint검사통과.
각입력/학습전/5,000/겹침4열PNG3개및통합이미지,384이미지전체CSV,선택기준/체크포인트및속성SHA/표시particle별presence보고서를생성했다. PNG3장을직접열어가독성과화살표/박스를확인했다. 동일5,000SHA971e8724…사용,추가GPU추론없음.
이그림에서는64점반경고정,각선택기준별변화상위8particle의박스만같은번호로표시한다. 이전초기presence상위16박스와다르므로명시한다. 특히최대이동particle들은낮은presence가포함된다. 큰geometry변화가주행중요도/유용성증거라는해석은하지않는다.

## 2026-10-08 — 더 많은 particle 시각화와 전체 분포 갤러리

사용자는다른이미지를추가해전체경향을파악하고자했다. 동일5,000checkpoint/저장particle을CPU에서읽는 `scripts/build_lpwm_particle_comparison_gallery.py`를추가하고 `results/lpwm_drivor_planning_path_lora_v1/particle_gallery_update5000/`에전체96장면×4카메라384개의개별전후4열PNG와겹침PNG를생성했다. localindex.html은카메라/시나리오필터와위치·크기·presence변화정렬,24개씩페이지를지원한다.
이전에제공한scene0/2/3/21/27/41/84/94를제외하고직진/좌/우각6전방이미지를평균위치변화rank10/25/40/60/75/90분위근처로선택했다. 총18장면/14recording이며IDs10,13,61,37,35,51/86,89,63,69,91,93/83,60,79,23,40,32. 18장overview PNG와시나리오별3장씩2페이지총6상세PNG를제공한다.
이번갤러리는모든64점반경고정,초기presence상위16개박스번호고정으로원래전후그림과박스예산을맞췄다. 직전극값그림의변화상위8박스와구분한다. 미래GT/이미지편집/새추론없음,움직임과이미지를같은비율로확대했다.
전체384의이미지별평균위치2.27696px(P10–P90:1.78201–2.81629),평균크기14.08376%(10.68341–17.78140),presence.63221→.46835. 상황별평균위치직2.26388/좌2.28705/우2.28723px;전/후/좌/우카메라2.17383/2.28042/2.35940/2.29417px. 각그룹의서로다른관측이미지기술통계이며같은장면명령변경효과가아니다.
384개unique장면-카메라조합,768asset링크/PNG크기,18예시중복·이전예시제외검사를완료했다. overview/좌회전상세2/분포그래프PNG를직접확인했다. artifact_checks.json/전체CSV/gallery_report.json과PNG/PDF보존. 현재등록학습코드/설정/queue는변경하지않았다.

## 2026-10-08 08:53 KST — 현재까지의 중간 결과 확인

본학습5,464/40,350(3.385epoch), Adapter12,828/14,121(2.725epoch). 양rank 기록에서 비유한 loss 없음; 최신 본학습5,400의18gradient그룹 유한·양수, Adapter encoder/context/dynamics/planner gradient 유한·양수. GPU0·1 최근200update 최대각37.05GB.
공통navtest1,024장면 PDMS는 기존본학습5,400의81.6096/Adapter2epoch81.6141 그대로다. 새가중치 점수로 부르지 않는다. 최신 particle 진단5,000, 다음5,500 대기.
최근50–200update 본학습85.78–90.69초, Adapter3.58–3.66초. 5,500 학습경계09:45–09:48, Adapter3epoch학습10:10–10:12KST 추정; 진단양보·평가시간 별도다. Adapter3epoch→dev평가→본학습batch16복원 대기열 유지.
근거 `results/lpwm_adapter_original_batch_shared_v4/intermediate_both_runs_20261008_0853.json`. 이번 확인은 로그/진행 파일 기반이며 새 host PID조회는 하지 않았다. 실행코드·설정·대기열 변경 없음.

## 2026-10-08 09:00 KST — 본학습의 단일 관측과 JEPA 추론 history 확인

사용자는 본학습이 과거 영상을 사용하지 않는 이유와 Drive-JEPA/WA-JEPA 추론의 history 사용을 질문했다.
현재 `lpwm_drivor_joint.py:83`은 현재 RGB를 길이1 sequence로 encoder에 넣고 `dyn_module.sample(..., z_context=None, steps=8, return_context_posterior=False)`을 호출한다. LPWM `modules/modules.py:6417`에서 context가 없거나 길이가 부족하면 context prior를 예측한다. 과거 관측으로 motion context를 추론하는 경로는 이번 조건에서 사용하지 않는다. 미래 rollout 내 예측 state는 이어지지만 이전 실제 관측 시점의 상태를 전달하는 online memory는 없다.
이는 DrivoR NAVSIM `drivoR.yaml:32`의cam_*:[3]과 `drivor_features.py:74`의cameras[-1]을 맞춘 입력 선택이다. 과거 프레임 불필요성이나 1프레임의 최적성을 실험으로 검증한 결과가 아니다.
공식 Drive-JEPA NAVSIM perception-free eval은front_only=true/double_image=true이고 feature builder가cameras[-1]/[-2]를512×256으로 처리한다. 현재+직전1장=총2장이다. Perception-based JEPA경로도front[2,3]이며ResNet대안과구분한다. WA-JEPA 공식PDMS YAML은4camera×4history/512×256(W×H), agent가마지막4시점영상을실제로stack한다. 0.5초간격으로현재+과거3장이다. 논문 https://arxiv.org/html/2608.20974v1 의Planning inference/Implementation details와 https://arxiv.org/html/2601.22032v2 의Implementation details도열람했다.
이전 Adapter는front4실관측frame과encoded context를dynamics에전달하므로시간관측정보가본학습과다르다. 자차속도/가속도는주변객체속도관측을대체하지않으며,단일이미지학습prior로미래출력을만드는것과실제motion을관측하는것은구분한다. 과거부재가현재성능차이의원인인지는통제실험전미확정이다.
후속연구권고는동일LPWM초기화/4camera/해상도/planner/학습조건에서1시점대현재+직전2시점비교,동적객체·교차로·가림분해및미래개입평가다. Register대particle비교에는같은시간정보와처리설계통제가필요하다. 설명요청이므로현재등록학습/평가/대기열은변경하지않았다.

## 2026-10-08 09:05 KST — 原 LPWM 학습·추론의 시간 관측 감사

원본commit4cf53c403433e64c01652ac2adbec66231a46dea와논문 https://arxiv.org/html/2603.04553 §4/A.4/A.5를확인했다. 이미지particle encoder는프레임별독립처리이고context/dynamics가시공간attention으로시간정보를결합한다. 원본train_lpwm.py:164의학습입력길이는timestep_horizon+1이며실제관측sequence로teacher forcing한다. models.py:1170에서현재측particle[:-1]와transition inverse context[1:]를dynamics에전달한다. Loss는frame복원+staticKL+particle dynamicsKL+context inverse/priorKL+presence정규화,optimizer는model.parameters()로네모듈공동학습이다. 장기free rollout의RGB오차만으로학습한다고설명하지않는다.
원본sample_from_x(:865)는cond_steps만큼실프레임을encode하고observed context를전달한다. 이후context prior→dynamics로autoregressive particle생성,필요하면RGB복원한다. 이전실제관측과이후생성된states를사용하며standard use_all_ctx=False에서는미래RGB를encode하지않는다. use_all_ctx=True/--ctx는전체실영상에서inversecontext를얻는별도조건부재생검사다. animate_trajectory_lpwm의deterministic패널기본경로에는이검사가포함되고다른stochastic패널은표준forecast다. 모든원본시각화를causal forecast라고부르지않는다.
공개config와다운로드Sketchyhparams는horizon20/학습21frames/cond_steps10이다. README의Sketchy예시-c6과생성CLI기본fallback=horizon20도있으므로실행별override를구분한다. Traffic21/6,BAIR17/1,Bridge25/2로관측수는설정별이며1frame도정식지원된다. num_static_frames=1은KL의static구간수이지추론관측1장지정이아니다.
정책학습논문A.5에서는LPWM고정후실영상inverse latent actions→action mapping을L1로학습한다. 배포예시는현재obs+goal에서prior/dynamicsrollout후생성particletrajectory의inverse latent actions를mapping에넣으므로현재1관측출발자체는원본과불일치가아니다. 우리의planningloss LoRA/DrivoRplanner와원본정책방식은다르다.
현재본학습1frame/기존Adapter4frame의관측차이는유지했다. 원본이무조건여러frame을요구한다거나원본1frame지원이NAVSIM충분성증거라는주장을하지않는다. 근거 `results/lpwm_drivor_planning_path_lora_v1/original_lpwm_temporal_protocol_20261008.json`. 등록학습/평가/대기열변경없음.


## 2026-10-08 — Drive-JEPA 주행 영상 사전학습 재사용 가능성

검증할 하위 질문은 더 넓은 주행 영상의 자기지도 적응이 의도 조건부 planning에 필요한 particle의 움직임 정보를 개선하는가이다. 이번 요청은 가능성 조사이며 기존 두 학습·평가·대기열을 변경하지 않았다.

논문 https://arxiv.org/html/2601.22032v2 는 V-JEPA2 초기화, CoVLA/DrivingDojo/OpenScene trainval의 전방 영상 330시간, 8frame·2Hz·512×256, H800 8대에서 50epoch 약3일을 보고한다. 공개 Drive-JEPA checkout의 generic V-JEPA encoder/predictor/EMA 학습 코드는 확인했지만 동일330시간 clip manifest·데이터별 혼합률·주행 전용 SSL YAML은 확인하지 못했다. 포함된 generic YAML은16frame/4Hz/정사각256/generic video sources여서 논문 주행 설정이 아니다. RGB 복원 대신 가려진 위치의 target latent를 예측하며 target branch는stop-gradient/EMA다. 무작위 시공간 마스킹을 엄격한 과거→미래 예측과 동일시하지 않는다.

- CoVLA 공식 HF는10,000×30초/80시간 이상, 저장소452GB. 로그인·연락처 공유·조건 동의가 필요하며 현재 계정 승인 여부는 확인하지 않았다. 데이터 이용 조건은 학술/비상업 목적이며 코드 라이선스와 구분한다.
- DrivingDojo 공식 문서는약18,000영상/45archive다. 확인한 base/Extra1/Extra2만283+283+291GB이고 모두gated다. Base Apache-2.0와Extra CC-BY-NC-SA-4.0가 달라 전체를단일license/283GB로표기하지않는다. CoVLA와 이세저장소만 합해도약1.309TB로신규원본1TB정책을넘는다. 전체다운로드는실행하지않았다.
- OpenScene 공식trainval카메라archive는1.1TB이며RGB SSL에LiDAR는필요없다. 로컬trainval1,310log는메타데이터규모이지연속영상보유량이아니다. 7log의메타데이터와실제CAM_F0를대조했고다수누락을확인했다. 현재NAVSIM current/history subset만으로330시간확보를주장할수없다.
- CPU검사에서첫log의339번부터8장을실제로decode하고1920×1080→512×256 RGB BICUBIC변환에성공했다. 인접간격약0.5초,첫끝3.498873초다. 이것은전처리가능성확인이며논문crop을재현했다는검사는아니다. 전체유효시간은아직집계하지않았다.

LPWM에는동일주행영상으로원래particle/영상SSL을적용하는것이가장직접적인확장이다. 현재128×128 LPWM에512×256을넣거나ViT가중치를그대로로드할수있다고가정하지않는다. JEPA목적을particle에적용하려면target대응·collapse방지·gradient경로설계가별도로필요하다. 주행사전학습50epoch의ViT `runtime/checkpoints/drive_jepa/vitl_merge_3dataset_e50.pt`(5,127,748,765bytes)는이미존재한다. 이를별도baseline또는고정featureteacher로활용하면ViT주행재학습을생략할수있지만LPWM증류효과는미검증이다.

우선기존OpenScene에서실제연속clip목록과recording별분리를확정하고LPWM SSL조건을설계하는것을권고한다. Navtest/Navhard 등평가recording과인접frame은새SSL학습목록에서제외하고,외부데이터노출·해상도·관측수·planner·배치·seed를대조조건에명시한다. 외부영상SSL만으로명령별선택표현이학습되는것은아니므로planning/ego intent 적응은별도검증한다. 8frame사전학습은추론8frame을강제하지않는다. GPU0·1은A6000이며현재기존학습중으로새SSL GPUprofile을하지않았고소요시간을단정하지않았다.

근거: [조사 JSON](../results/lpwm_drivor_planning_path_lora_v1/drive_jepa_video_pretraining_feasibility_20261008.json), [CoVLA](https://huggingface.co/datasets/turing-motors/CoVLA-Dataset), [DrivingDojo 데이터 문서](https://github.com/Robertwyq/Drivingdojo/blob/main/docs/DATASET.md), [OpenScene 데이터 문서](https://github.com/OpenDriveLab/OpenScene/blob/main/docs/getting_started.md).


## 2026-10-08 10:51 KST — native512×256 LPWM·particle 최소후보 학습 시작

**2026-10-08 10:51 KST — 사용자 지시로 기존 두 학습 중단, 512×256·소수 particle Stage1 시작.**
기존 본학습5,493 / Adapter13,480 fullstate 저장·pause 및 자동재개 watcher 종료. 과거 25epoch 대기열 재기동 금지.
새 연구: front1·512×256·2Hz·8frame LPWM SSL → native backbone1e-5 + DrivoR planner1e-4 공동학습.
Foreground8/16/32/64(+background1) 각200update·유효4·동일32held-out recording 비교 → 잠정 최소개수로 로컬 전체1epoch 자동학습.
GPU1 queue2819995, root `outputs/lpwm_driving_video_512x256_v1/particle_budget_study/`; CPU overlay2926659. GPU0 타인작업 유지.
로컬 SSL train14.290h/10,480비중복clip, val3.050h/61recording중고정32평가. 8particle 실제 forward/backward·causal예측·저장 확인. 최종planning 성능보존은 미검증.
OpenScene downloader2826493: 공용원본 읽기전용, 누락front만 별도processed_dataset/junseong 소유root에 stream, 총1TB제한/압축archive미보관/SHA검사.
CoVLA·DrivingDojo 및Extra1–5는 현재HF계정403 GatedRepo. 사용자 이용조건 동의·접근승인 필요; 대신 승인하지 않는다.
**330h 본사전학습 및 Stage2는 아직 미시작.** 원Drive-JEPA의 정확330h clip manifest도 미공개이며 동일3source의 자체curation으로구분한다.
[시작 근거](results/lpwm_driving_video_512x256_v1/transition_and_start_report.json) · [확정 목표·남은 단계](configs/lpwm_driving_video_512x256_v1/research_plan.json).
아래 기존실행중·새학습금지·geometry선택대기 문장은 이번 사용자 승인 이전 이력이다.


### 최신 — 축소 particle native-resolution 구현과검사

- 128×128 공개LPWM을strict-load후 CNN은실제512×256 RGB를입력받음. prior grid8개2×4/16개4×4/32개4×8/64개8×8, foreground glimpse64×128. CNN후adaptive pooling으로FC dimensions보존. 위치embedding공간평균, background1보존.
- foreground sprite renderer는공개32×32를유지하고512×256canvas에합성; background latent seed를16×32로확장하여CNN복원. 전체RGB를128로축소하지않음. 이는새아키텍처적응조건이며원본동일아키텍처성능보장아님.
- 8·16·32·64개공식loss+역전파모두성공. 8개encoder/context/dynamics/decoder gradient모두유한양수. 8개공개초기future MSE.0664,현재반복.0323으로초기public모델주행예측은미적응. 최종경향은각200update후검증해야함.
- 로컬SSL은train102,890 unique frames(14.2903h),val21,958(3.0497h),image/recordingoverlap0. metadata전체100.419h지만실제영상부족하므로가용시간과구분. 전체330h아님.
- 최초planning index가82 scene을missing으로기록한원인은camera파일누락이아닌history간격약1초. 전부두이미지존재. `corpus/planning_history_audit_amendment.json`에정정; 등록SSLmanifest불변. 새indexsource는missing/cadence분리. 미래Stage2는실timestamp보존하고예외를기록해야함.
- 학습전후재구성/causal6미래(2관측) MSE·LPIPS·아래절반오차·presence/spatial/appearance분산검사. 64대비15%내는exploratory후보screen일뿐객체정보/PDMS 비열등성증명아님. 미래GT를context입력하지않음.
- HF토큰있지만CoVLA와DrivingDojo base/Extra1–5 모두403 GatedRepo,OpenScene200. sandbox밖에서도동일,네트워크설치실패와구분. 사용자동의·승인이필요하다.


### 최신 승인된 다음 단계 (과거 자동학습보류보다 우선)

1. 새particle queue/학습/quality결과·PNG 확인. 구조검사통과와주행정보보존검증을구분. 8개가실패하면16/32를검토하며작은객체/가림정보도후속진단.
2. HF계정CoVLA/DrivingDojo 접근승인대기. 사용자가동의했다면catalog script재확인; 토큰출력·약관자동동의금지. OpenScene download진행/실패/SHA검사. 원본330h manifest미공개조건을숨기지않음.
3. 완료archive marker의파일만새expanded manifest에추가. 기존실행중localmanifest/등록source수정금지. CoVLA·Dojo video ingest·2Hz/crop28/512×256변환·330uniquehour audit 추가구현필요. 새원본총1TB제한.
4. 330h 코퍼스완성후full Stage1 DDP0·1 학습·검증설정확정/구현. 현재queue는로컬1epoch후대기하며330h학습·Stage2로자동넘어가지않는다.
5. Stage1검증후front1/2observed512×256용generic-count particle encoder↔공식DrivoR planner연결(Stage2새구현필요). native backbone1e-5/planner1e-4; intentionCNN경로의checkpoint-safe FiLM을연결하고gradient검사. 기존4cam128코드를그대로재사용하지않음.
6. 공정split·fullNAVTEST/EPDMS평가후새로받은데이터만owner/ledger확인후삭제. 자동cleanup은아직구현/실행되지않았음; shared Dataset/checkpoints/manifest/results절대삭제금지.

후속실측: 8개200update 검증완료, 미래MSE .066418→.029759/현재반복.032324. Presence평균.999648로거의모두활성: adaptive sparse selection이나객체검출성공으로부르지않음. 16개비교계속. OpenScene첫archive SHA검증통과/누락front2085장확보. CPUpublisher는패널축512×256고정수정후2926659로교체하고 v2그림에저장. 기존그림보존.


## 2026-10-08 11:45 KST — 데이터셋계정승인검증

**2026-10-08 11:45 KST — 사용자 동의 후 데이터셋 접근 승인 확인 완료.**
서버계정 `JunseongKwak` / CoVLA·DrivingDojo기본·Extra1–5·OpenScene 모두 HEAD200, 대표파일실제GET206·1,024bytes수신 성공.
기존토큰으로통과, 추가사용자동의·토큰교체불필요. 토큰값미출력. [검사근거](results/lpwm_driving_video_512x256_v1/dataset_access_approved.json).
새승인catalog `outputs/lpwm_driving_video_512x256_v1/download_access_after_user_approval.json`, ready marker `dataset_access_ready.json` 사용. 과거403catalog는이력보존.
OpenScene다운로드계속. CoVLA/DrivingDojo본영상수집·전처리·330h확정코퍼스는아직구현/준비가남아있으며, 권한해결을다운로드완료로부르지않음.
Particle200update 8/16/32/64비교완료, 8/16/32는64대비전체재구성MSE15%기준초과로queue `held_for_quality_review`; 자동로컬1epoch 미시작.
8future오차+21.2%,16+4.45%,32+.55% vs64. 짧은SSL오차검사이므로최종PDMS·최소적정개수결론불가. 규칙을조용히완화하거나학습중이라고보고하지않음.


## 2026-10-08 12:35 KST — 실제 데이터 수집 복구와 두 GPU 로컬 SSL 시작

**2026-10-08 12:35 KST — 다운로드와 병행하여 기존 데이터 Stage 1을 GPU 0·1에서 시작.**
사용자가 기존 데이터로 즉시 시작하고 두 GPU 배치를 늘리라고 승인했다. 새 trainer/queue는 `scripts/train_lpwm_local_stage1_distributed.py` / `queue_lpwm_local_stage1_distributed.py`.
Root `outputs/lpwm_driving_video_512x256_v1/local_stage1_distributed/`, queue3516125 / torchrun3519630. 실제 양rank 12update·loss/gradient동일·43개 source hash 일치 확인.
공개Sketchy LPWM 새초기화, front1·512×256·2Hz·8frame·foreground16+background1. encoder/context/dynamics/decoder native 전체학습, ego명령·planning·객체GT loss 없음.
GPU당micro4×누적2×GPU2=유효16, worker4/rank, FP32 Adam8e-5. 실측 batch2 3.276s/22.44GB vs batch4 3.100s/43.04GB; batch4채택, batch8은예상메모리초과로미시도. 전체카드48decimalGB상한.
고정로컬10,480비중복clip/655update=1epoch. 인덱스학습frame14.29h 중 완전8frame clip에 실제소비되는양11.644h; 미래정답을입력하지않는32held-out recording 검증을초기/100update/끝에실행, 전후·겹침PNG자동생성. Adam/양rank RNG/진행cursor 저장.
16particle은잠정추가학습후보다. 앞선200update의15%품질gate실패를변경/통과처리하지않았고64대비planning성능보존미검증. 앞선유효4와이번유효16을동일조건이라고부르지않음.
다운로드완료아님: OpenScene3352751(HTTP Range재연결), CoVLA3352752, 실제JPEG형식Dojo3499336 병행. 이전Dojo MP4가정실행3352753은수집0건으로종료·이력보존.
확장코퍼스는archive SHA/시간/중복·split검증후별도등록해야하며실행중manifest에혼합금지. 330h본학습·새Stage2는미시작. 로컬1epoch→검증까지자동이며확장학습/Stage2자동기동은아직연결되지않음.
[시작·메모리·gradient 근거](results/lpwm_driving_video_512x256_v1/local_stage1_distributed/start_report.json). 아래작업중상태/PID는과거기록이며기존LoRA/Adapter 중단은유지한다.


Loss = 0.01/8 × [0.125 reconstruction + 0.08 static KL + 0.2 dynamics KL + 0.2 context KL + 0.08 object regularization]. reconstruction은MSE+0.1LPIPS의공식pixel-count scaling이며0.125는128²대비8배pixel-count보정. 미래RGB 직접rolloutloss를새로추가한것은아니다. 공개LPWM의posterior/prior KL경로를유지,loss계산용VGG는고정하지만입력gradient전달. Decode_with_ctx=False이므로context는주로KL경로로학습. 공식dynamics입력detach=False이고KL balancing내부stop-gradient는원본유지.


## 2026-10-08 12:49 KST — Drive-JEPA 조건 일치 감사와 프로토콜 확정

**2026-10-08 12:49 KST — 사용자 목적 확정: Drive-JEPA와 입력·330h 데이터가 일치하는 LPWM 사전학습 비교.**
[비교용 본학습 프로토콜](configs/lpwm_driving_video_512x256_v1/drive_jepa_matched_pretraining_protocol.json)을 등록했다. 현재 GPU0·1의 로컬1epoch는 준비실험으로만계속하며, 그checkpoint/optimizer/추가노출을비교용본학습으로넘기지않는다. 본학습은공개LPWM원초기화부터새로등록한다.
공식논문 front1·512×256·2Hz·8frame·330h·50epoch 확인. 저자GitHub issue7/12/17에서 실제설정을추가발견: OpenScene/CoVLA/DrivingDojo sampling0.5/0.2/0.3,8GPU×batch64=global512,300update/보고epoch. 이확률은원본시간비율이아니다.
저자posted config max100epoch, 논문50epoch, 공개e50checkpoint metadata epoch51/Adam15,300step이므로서로동일값으로고치지않는다. 비교시epoch명칭보다실제clip노출량을명시한다. 공개checkpoint대상노출7,833,600clip;paper50기준7,680,000clip. 본학습global512/누적64후보는계획이며아직실행아님.
정확한세CSV내용과저자의영상생성script는미확보.28px crop은공식downstream에서확인한것이며pretraining에서도같다는근거는없다.현재준비자산을exact-equivalent라고부르지않음.원논문목록미확보시동일자체330h목록으로LPWM·V-JEPA2를양쪽재학습하는통제비교가대안이며새JEPA학습은미시작.
OpenScene trainval을쓴다는저자답변확인;로컬준비실험의navval61recording제외를본실험동일분할이라고가정하지않음.사전학습에포함되는navval을독립SSL검증으로부르지않고공식test누출제외.
다운로드검사에서DrivingDojo35만ZIP이라기존44tar목록에빠진것발견. 별도catalog `download_catalog_with_dojo_zip35.json`, 새zip수집3594914/`drivingdojo_zip35_download`시작·실제8MB수신확인.원catalog/기존세수집source불변,zip변환/admission은아직남음.
학습원본43source불변,기존LoRA/Adapter중단유지,330h완료/동일영상비교성립/Stage2시작을주장하지않는다. [저자설정·checkpoint 감사](results/lpwm_driving_video_512x256_v1/drive_jepa_matching_audit/findings.json).


공식저자설정출처: https://github.com/linhanwang/Drive-JEPA/issues/7#issuecomment-4006091822 및 https://github.com/linhanwang/Drive-JEPA/issues/12#issuecomment-4231849356 . 저자는영상생성/학습script정리미공개사유를직접설명했다. config공개여부를미확인이라고했던이전판단을정정; 정확CSV내용미확보는유지. OpenScene분할출처 https://github.com/linhanwang/Drive-JEPA/issues/10#issuecomment-4120088355 .


### 12:51 전체 사전학습 ETA

**12:51 KST 전체 Stage1 ETA(조건부 추정):** 현재16particle/FP32/GPU0·1 처리량을공개Drive-JEPA e50의15,300×512=7,833,600clip노출에환산하면GPU연산17.57일,현재wall19.95일,검증주기포함21.85일. 데이터수집/전처리4–6일+학습18–23일+최종검증0.5–1일로총23–30일(10/31–11/7KST) 계획범위. Stage2/PDMS미포함. 정확referenceCSV미확보/본batch512미실측이므로확정종료일아님. 현재로컬1epoch는255/655로별도이며약13:18완료예상. [계산근거](results/lpwm_driving_video_512x256_v1/drive_jepa_matching_audit/full_pretraining_eta_20261008.json).

## 2026-10-08 12:54 KST 현재 작업 상태조회

**2026-10-08 12:54 KST — 사용자 상태조회.** GPU0·1 로컬준비학습 300/655update(16particle/front1/512×256/유효16) 진행. 카드당약43.04GB, loss유한·43source불변·최신modulegradient확인. 네수집process도살아있음. OpenScene8/200archive검증,CoVLA252영상/2.10h,Dojo71clip변환/첫archive미검증,ZIP35약1.89/35.28GB수신(12:53조회). 정확330h비교본학습과Stage2미시작. 실행/설정변경없음. [조회근거](results/lpwm_driving_video_512x256_v1/local_stage1_distributed/status_20261008_1254.json).


## 2026-10-08 12:57 KST 학습 가중치 범위 확인

12:57미세조정범위확인:encoder/context/dynamics/RGBdecoder native전부requires_grad=True,Adam(model.parameters(),lr8e-5). LPIPS VGG만고정;module별300update양수gradient확인. 현재SSL/명령·planningloss없음. 모든조건부parameter가매step비영gradient라는주장은하지않음. 실행변경없음.

## 2026-10-08 사전학습 축소 추천

사용자가 약 30일 사전학습은 불가하다고 하여 학습량 축소를 검토했다. 현재 입력·목적(front1/512×256/2Hz/8frame, 16FG+1BG, native LPWM SSL)은 유지하고 10만 clip에서 검증 후 최대 30만 clip을 권고한다. 유효 batch16·검증 포함 3.8565초/update 기준 각각 6.70/20.09시간, 60만은 40.17시간이다. 외부 loader·공유 부하·다운로드·Stage 2는 이 추정에 포함되지 않는다.

30일 추정에는 330h 코퍼스뿐 아니라 공개 Drive-JEPA checkpoint의 약 783만 clip 노출량까지 맞추는 가정이 있었다. 입력·코퍼스 일치와 학습량 일치는 별개 조건이다. 수집 대기가 가능하면 330h 코퍼스에 축소 노출 예산을 적용하고, 전체 완료 시간이 짧아야 하면 세 source 공통 30–50h 부분집합을 사용한다. 고유 시간·반복 노출·무작위 샘플링의 커버리지를 구분한다. 시간 예산 선택 질문을 보냈으며 기록 시 답변 없음.

이 추천의 하위 연구 질문은 짧은 주행 SSL 적응으로 얻은 particle이 이후 ego 의도·planning gradient 적응에 유용한가이다. 같은 SSL checkpoint에서 frozen LPWM+planner와 joint LPWM+동일 planner를 비교하면 planning gradient의 추가 효과를 검사할 수 있다. Stage 1 자체 기여는 public 미적응 초기화 대조가 별도 필요하다. 공개330h Drive-JEPA checkpoint는 축소 LPWM과 동일 예산 대조가 아니며, 엄밀한 비교는 공통 코퍼스·입력·학습 예산·planner를 양쪽에 적용하고 서로 다른 upstream 사전학습도 밝힌다.

LoRA 전환·RGB decoder 제거·추가 particle 감소를 동시에 적용하지 않도록 권고한다. BF16은 향후 KL/분산 등 수치 안정성과 실제 처리량을 분리 검증할 최적화 후보다. 현재 source/config/queue/다운로드는 변경하지 않았으며 13:03 로컬 준비 439/655 update를 확인했다. 새 긴 본학습이나 Stage 2를 시작하지 않았다. 근거: `results/lpwm_driving_video_512x256_v1/drive_jepa_matching_audit/reduced_pretraining_recommendation_20261008.json`.

## 2026-10-08 JEPA와 LPWM의 공통 작은 클립셋 비교 제안

사용자가 Drive-JEPA 쪽도 작은 클립셋으로 학습해 가능성을 판단하는 안을 제안했다. 현재 고정 OpenScene10,480클립/약11.64h를 공통으로 쓰고 1/5/10회 코퍼스 노출을 기록하는 설계를 추천한다. 총104,800노출의 LPWM Stage1은 현재 속도 약7.02시간, JEPA 및 새고해상도 Stage2 시간은 미측정이다. 전방1/512×256/2Hz/8frame, 같은 clip identity/timestamp/preprocessing·sample order·effective batch, 모델별 짧은 LR schedule을 등록한다. SSL 검증용 recording은 양쪽 모두 제외한다.

Drive-JEPA 논문·코드와 V-JEPA2 공식 공개 가중치를 재확인했다. JEPA 시작점은 330h 주행 적응 전 일반 영상 V-JEPA2 ViT-L/16이며 LPWM은 공개 Sketchy다. e50 주행 모델이나 V-JEPA2.1 distilled 모델을 조용히 대체하지 않는다. `app/vjepa/utils.py:90`는 optimizer와 epoch까지 load하고 train.py가 해당 epoch만큼 스케줄을 넘기므로, 신규 적응은 가중치 초기화 경로와 resume를 분리해야 한다. 실행 코드 수정은 하지 않았다.

최소 후속 실험은 동일 DrivoR 스타일 planner·loss·현재+직전 전방 입력·ego status·navtrain 노출 예산으로 두 백본을 미세조정하는 비교다. 특징 입력 투영의 필요 차이와 규모는 명시한다. 작은 데이터의 절대 PDMS만으로 구조 효과나330h 확장 우열을 입증하지 않으며, 사전학습 전 원래 데이터 차이도 남는다. Frozen LPWM 대 joint 및 미래 개입은 후속으로 planning 적응 기여를 분리한다. 새 학습은 시작하지 않았고 기존 로컬 준비633/655를 확인했다. 상세: `results/lpwm_driving_video_512x256_v1/drive_jepa_matching_audit/common_small_clipset_comparison_proposal_20261008.json`.

## 2026-10-08 네 모델의 축소 비교 제안

사용자가①Drive-JEPA ②DrivoR ③현재의2stage LPWM ④LPWM과planner를처음부터공동학습(front1/512×256)의네조건을제안했다. 공통SSL10480클립×5회, 공통planning약1만장면×5회, 독립recording개발1024장면의초안을작성했다. 이는등록·실행완료수치가아니며기존SSL clip과planning scene을같은단위로취급하지않는다. LPWM SSL52400노출만현속도약3.51시간,JEPA및새고해상도Stage2/네모델전체시간미측정.

연구목적에맞춰모두동일DrivoR planner인표현비교를우선추천하되①를공식planner로유지하는시스템비교도가능하므로async선택질문을보냈다. 답변전새학습기동없음. ①시작은주행330h적응전VJEPA2,②는NAVSIM적응전DINOv2,③④는같은공개Sketchy와새planner초기화다. ③④의미세조정범위·ego명령·particle수·planning노출뿐아니라SSL노출도맞춰야순차대joint해석이가능하다. ④는SSL+planning동시학습을추천했으며이전planning-only목적과같다고부르지않는다. loss유무까지바뀌면두요인의영향이섞인다.

코드상DrivoR는현재영상1frame,patch14,기존LPWM plannerbridge는4camera/128assert다. 따라서모든모델front1/512×256/과거현재2frame으로맞추려면DrivoR시간인터페이스와patch경계padding,LPWM직사각형bridge를명시적으로구현·검사해야한다. 인터페이스변경된baseline은원논문설정과구분한다. 원래백본의사전학습데이터와모델규모차이는남으며공통planner라도pureparticle인과효과라고주장하지않는다.

현재localqueue는local_epoch_and_validation_complete/Stage2false확인했다. 검증완료를품질통과로단정하지않는다. 새학습/설정/다운로드변경없음. 설계:`results/lpwm_driving_video_512x256_v1/drive_jepa_matching_audit/four_model_small_corpus_comparison_proposal_20261008.json`.


## 2026-10-08 네 모델 축소 실험 실행

**2026-10-08 14:09 KST — 사용자 승인으로 네 모델 축소 비교 학습을 시작했다.**
별도 planner 선택 답변이 없어 알린 추천안인 **공통 DrivoR planner**를 사용한다. ① Drive-JEPA 방식 백본+공통 planner ② DINOv2 register+공통 planner ③ LPWM SSL 후 planning ④ LPWM SSL+planning 처음부터 joint. 원 논문 전체 설정 재현이 아니다.
Root `outputs/four_model_small_corpus_v1/`, queue PID136859. LPWM Stage1은 공개 Sketchy에서 새로 시작해219/3275 fullstate 저장 후 호환성 검사를 위해 잠시 양보했고, 검사가 끝나 현재 queue가 같은 상태를 재개한다. 기존 준비655update/기존 LoRA·Adapter는 합산·재개하지 않는다.
OpenScene 고정10480 SSL clips(8frame/2Hz/실제11.644h)×5회=52400노출. 공통 planning10240/dev1024장면, recording 중복0, front1·512×256·과거현재2frame·effective16·5회/3200update. DINO만 patch14 정렬용 우6/하10px padding. 모두 같은241개 planner 초기 tensor를 사용한다.
LPWM16FG+1BG native전체가중치 학습, planning 단계 encoder/context/dynamics/geometry/command FiLM gradient 확인. 순차Stage2는planning loss, joint는planning+0.1SSL이며 SSL총노출도52400으로 맞춘다. 순차Stage1 LR8e-5, planning native1e-5/planner1e-4. DrivoR는공식q/v LoRA rank32.
GPU0·1만, 카드48decimalGB상한. Stage1micro4×누적2×2GPU, 나머지micro2×누적4×2GPU. 최대 joint부하43.63GB, 모든경로2update 및 inference/validation 검사통과; profile가중치는본학습에사용하지않는다. 공식PDMS scorer1장면 호환성확인(성능결과아님).
대기열: LPWM SSL검증 → JEPA SSL검증 → DrivoR/LPWM순차/LPWMjoint/JEPA planning와각pass1·3·5 dev PDMS. Stage1품질실패시해당종속planner는보류하고독립조건은계속한다; 실행오류는queue_failed로중단. 실제진행은queue_state와각progress를확인한다.
1seed·작은devsubset비교이며 전체navtest/330h학습/SOTA/pureparticle인과효과로부르지않는다. 공개초기화데이터·백본규모·순차대joint의LR경로차이가남는다. 설정 `configs/four_model_small_corpus/experiment.json`. 아래 미기동/제안 문장은 이전 이력이다.

14:10 KST 재개검증: 양rank235/3275,43.70GB,121source불변;launch_verification.json에기록.

## 2026-10-08 대기열 병렬 배치 최적화

**2026-10-08 14:36 KST — 사용자 시간 단축 지시로 대기열만 overlap_v2로 교체했다.**
새 controller294935가 기존 train136861을 PID/start_ticks 그대로 인계했다. 기존 controller136859만 종료했으며 학습 재시작·추가노출 없음. 현재 LPWM SSL655/3275(첫epoch후검증경계), root `outputs/four_model_small_corpus_v1/scheduling_v2/`.
원래121개 scientific source/config hash불변. 새 script `scripts/queue_four_model_small_corpus_overlap.py`, 설정 `configs/four_model_small_corpus/scheduling_overlap_v2.json`. 배치/누적/GPU2 topology/LR/loss/seed/데이터/5epoch 불변.
순서: 현재LPWM SSL전용 → JEPA SSL+DrivoR 병행 후보 → LPWM순차planner+JEPA planner 병행 후보 → LPWMjoint전용. 각pair는8update독립/병렬 profile의속도1.05배이상·실제training중첩60%이상·양rank loss일치·전체카드44GB이하를통과해야병행하고아니면자동순차. Profile가중치는본학습에미사용. 실제병렬속도검사는LPWM SSL완료후이므로단축률미확정.
CPU PDMS는pass1/3/5예측+1024count메타데이터완료즉시GPU학습과병행한다. CPU평가동시1개/worker4/nice5. 학습용10240 scene cache모두기존존재확인,추가생성안함. LPWM SSL·joint는43GB급이라GPU독점,48decimalGB제한유지.
8개스케줄검사통과. Sourcehash·PID연속성·새state 근거 `scheduling_v2/handover_verified.json`. 이전serial controller/queue를중복실행하지않는다. 전체pause는studyroot/pause.requested,실패는scheduling_v2/failed.json을먼저확인한다. Stage1gate와개별종속planner보류규칙은보존한다.

## 2026-10-08 최근 미사용 다운로드 삭제

**2026-10-08 14:54 KST — 사용자 지시로 최근 다운로드만 삭제 완료.**
범위는이번에새로받은CoVLA·DrivingDojo및미사용OpenScene보충이미지뿐이다. 기존공용NAVSIM/OpenScene·다른데이터·checkpoint·학습/검증cache는보존했다.
삭제파일할당량51,062,308,864bytes(51.06decimalGB): CoVLA변환5.445GB,DrivingDojo변환2.377GB+ZIP35.283GB,새OpenScene보충7.958GB,빈전송버퍼. 작업공간7.822GB/전용raw저장소43.240GB.
원다운로드3352751/3499336종료,CoVLA3352752는pause marker후종료확인,ZIP3594914는이미완료였다. 관련5control에pause.requested와cleanup_completed.json,root downloads_retired_for_small_corpus.json을기록했다. 이전active.pid/status/converted metadata는역사적출처기록이며파일존재근거가아니다. 이데이터수집을자동재개하지않는다.
소유자·오늘생성범위·현재136021참조파일존재·삭제대상중복0·열린파일0검사후.trash격리→재검사→실삭제했다. 원121science source/config불변,train136861/queue294935유지,양rank950/3275로계속학습. 근거 outputs/recent_driving_download_cleanup_20261008/cleanup_complete.json 및final_verification.json.

## 2026-10-08 현재 축소 실험 데이터셋 확인

현재데이터실사: SSL기존OpenScene10480clips/101recording,검증32recording각1clip. Planning navtrain10240/101recording,navval1024/61recording. SSLtrain↔planningdev recording중복0,SSLval32recording은planningdev61에포함된다. NAVSIMv1 scorer사용,navtest/navhard미평가. navval은DrivoR공식training/default_train_val_test_log_split.yaml의val_logs에서navtrain필터token을구분한것이다.

근거 outputs/four_model_small_corpus_v1/dataset_usage_20261008.json. 실행변경없음.

## 2026-10-08 Planning에서 두 관측 프레임을 쓰는 이유

현재 전방 512×256 입력은 현재 + 약 0.5초 전 이미지다. 시간 변화 단서를 제공하고 네 모델의 관측 범위를 맞추려는 설정이며 1프레임 대비 성능 우위는 미검증이다. JEPA는 tubelet_size=2로 인코딩, LPWM은 두 관측의 particle/context를 native dynamics에 넣어 미래 8단계를 예측하고 현재+미래 속성을 planner에 전달한다. 원 DrivoR는 현재 이미지뿐이며 이번 baseline에는 프레임별 DINO 인코딩과 temporal register fusion을 추가했다. 따라서 원 논문 전체 설정 재현으로 부르지 않는다.

근거: src/planning_aware_future_prediction/object_centric/small_corpus_models.py:27,69,93,131; scripts/prepare_four_model_small_corpus.py:40; reference_repositories/DrivoR/navsim/planning/script/config/common/agent/drivoR.yaml:32. 코드·설정·실행 변경 없이 설명만 수행했다. 새 1프레임 대조 실험은 등록하지 않았다.

## 2026-10-08 DrivoR 원논문 전환 요청 취소 및 원래 축소 비교 유지

사용자가 직전 원논문 방식 유지 요청을 취소하고 원래대로 복구하도록 지시했다. 미실행·미커밋 초안 configs/four_model_small_corpus/drivor_official_revision.json, scripts/queue_four_model_official_drivor.py, scripts/train_small_corpus_official_drivor.py, src/planning_aware_future_prediction/object_centric/small_corpus_official_drivor.py를 제거했다. 새 대기열은 한 번도 실행하지 않았다.

기존 registered scientific source/config121개와 overlap_v2 scheduler source/config 해시 모두 일치. 기존 queue294935/train136861 호스트 실행과 양rank1462/3275를 확인했다. 학습 중단·재시작·가중치 롤백 없음. 전방1·512×256·과거현재2프레임 및 기존5epoch 비교/검증 대기열을 유지한다. 근거 outputs/four_model_small_corpus_v1/cancelled_official_drivor_revision.json.

## 2026-10-08 세 백본의 출력과 공통 planner 계산 그래프 확인

현재small_corpus_models.py 기준 세백본은모두장면당16×256 memory를준다. LPWM sequential/joint는같은inference구조:16FG마다현재1+미래8상태의28차원속성(geometry/appearance/background/context)을연결해252→256으로투영한다. Background는각FG에공유되는feature/context이며별도17번째memory는아니다. JEPA는2frame tubelet2/patch16에서512patch×1024를16개평균pooling→256으로투영하고사전학습predictor는planning에서제거한다. DrivoR는두frame각각16개의추가scene register를DINOpatch/CLS/원reg4와self-attention시키고384→256후같은register index의두시점을concat해512→256으로fusion한다.

공통DrivoRModel:현재ego11→256+학습query64→generator4block(self-attention64↔64,cross-attention64↔scene16,FFN)→각head가전체궤적8×3을출력,최종64후보사용. 좌표누적residual correction은없다. 좌표flatten24를detach후MLP24→1024→256→별도scorer4block의동일attention→ego추가→6subscorehead. NC/DAC/DDC로그+TTC/EP/comfort가중합로그로순위값을만들어argmax후보선택. NAVSIMv1 DDC선택가중치는0이지만해당head와감독은존재한다. Scoringloss는proposal좌표경로에gradient를주지않고sharedscene경로로backbone에전달된다. CurrentLPWM만명령FiLM/명시적future rollout을사용하므로token수일치만으로particle단독효과라고해석하지않는다. 원논문전체pipeline비교나새학습을실행한것이아니며코드조회만수행했다.

근거: src/planning_aware_future_prediction/object_centric/small_corpus_models.py:59,78,102,166; reference_repositories/DrivoR/navsim/agents/drivoR/drivor_model.py:107; transformer_decoder.py:32; layers/image_encoder/dinov2_lora.py:22; layers/losses/drivor_loss.py:242; scripts/train_small_corpus_common_planner.py:54.

## 2026-10-08 주행 명령 생성과 입력 출처 확인

현재planning코드는frame['driving_command']를원본NAVSIM/OpenScene메타데이터에서읽어ego의마지막4차원(7:11)에그대로넣는다. 명령은(left,forward,right,unknown)one-hot. 전체cache11264개가유효one-hot이고각좌/직/우대표원본과동일,unknown현재없음. OpenScene/DriveEngine/process_data/create_openscene_metadata.py:127은get_driving_command(ego_pose,map_api,roadblock_ids)를호출한다. helpers/driving_command.py:40은route보정/차선검색/중심선구성후현재중심선의진행방향에대해20m앞경로점의횡오프셋이+2m이상이면left,-2m이하면right,그사이면forward로기록한다. Route보정불가시unknown. Distance실제default20m이며docstring10m를따르지않는다. 우리학습코드에서미래GT궤적을이용해새명령을만들지는않는다.

명령은planning학습·추론의외부입력. 공통planner의ego embedding에포함되며LPWM에서는MLP4→64→64후scale/shift로분리해attribute CNN conv_in출력을FiLM조건화한다. Stage1 SSL에는명령을넣지않는다. 코드·데이터·학습·대기열변경없음.

## 2026-10-08 Ego status와공식DrivoR입력동일성검사

현재11차원은pose(x,y,heading)3+velocity2+acceleration2+command4이며공식과순서가같다. 공식full_history_status=false이므로마지막ego상태만Linear11→256후학습query64에더하고scorer attention이후에도더한다. 현재cache는현재자차상대pose를명시적으로0으로저장하며공식상대좌표변환의현재pose와동일하다. 이미지2frame과ego현재1frame을구분한다. LPWM은command[7:11]을추가FiLM에넣는차이가있다.

현재train/dev각좌/직/우대표1개씩6개를공식AgentInput과DrivoRFeatureBuilder로재구성해cache와비교:maxabs0전부일치. 전체11264cache의pose3이0인것도확인. 산출물 outputs/four_model_small_corpus_v1/ego_status_official_parity.json. 첫GPU학습환경에서는nuplan경로/rasterio가없어검사import가실패했으며,설치·환경변경없이기존CPU평가환경에서검사완료했다. 학습및대기열은변경하지않았다.

## 2026-10-08 Scorer attention 이후 ego 표현 덧셈 설명

drivor_model.py:177–180은후보좌표detach→pos_embed→scorer_attention→ego_token추가→6head를실행한다. Ego는최종숫자점수이전에256차원feature에더해진다. Generatorhidden query를직접전달하지않고좌표에서scorerquery를다시만들기때문에현재속도/가속도/명령을채점head에직접주입하는역할로해석된다. Detach는ego정보삭제가아니라좌표경로의gradient차단이다. 같은ego vector가64후보각각에더해지며비선형MLP는후보별다른응답을학습할수있다.

직접lateaddition은그전에계산한scorerattention을바꾸지않는다. 후보좌표및LPWM명령조건scene이ego에따라바뀌면간접적으로attention도바뀔수있다. Attention앞주입보다lateaddition이낫다는실험근거는이번코드조회에서확인하지않았으며새변형을실행하지않았다. 학습·대기열그대로유지.

## 2026-10-08 공식 DrivoR generator/scorer ego 이중 주입 재확인

공식clone commit fc6e5aa144bbcb5a046e22c18f1bd5cf3af8634a의navsim/agents/drivoR/drivor_model.py가git무수정임을확인했다. 115–117행의generator query+ego_token과178–180행의scorer attention output+ego_token둘다원공식코드다. 동일한11→256 ego embedding을두곳에서재사용한다. 현재공통planner도동일하며LPWM추가FiLM은별개다. 설정·학습·대기열변경없음.


## 2026-10-08 16:39 KST — 네 모델 축소 비교 중간 진행 조회

2026-10-08 16:39 KST 진행 조회: LPWM SSL 2646/3275 update(80.79%), 4/5 epoch 검증 완료 후 5번째 epoch 진행. GPU0·1 각43.70 decimalGB, 조회 시 utilization100%. Queue294935/train136861/양rank138319·138320 생존. 이번 턴 실행·배치·설정 변경 없음.

2026-10-08 16:39 KST: 고정 held-out32clip 미래 MSE 초기0.068846→epoch1 0.025333→epoch2 0.024397→epoch3 0.024375→epoch4 0.023159. 미래 LPIPS 0.879012→0.524589→0.511373→0.506619→0.503160. 마지막 프레임 반복 MSE0.032324 대비 epoch4 28.35% 낮음. 양rank 비유한loss0, 2600update encoder/context/dynamics/decoder gradient 유한양수. 등록121 source/config 및 scheduler2개 해시 일치. 근거 `results/four_model_small_corpus_v1/progress_20261008_1639.json`.

2026-10-08 16:39 KST: 기존 5epoch LPWM SSL 마무리→최종검증·gate→overlap_v2의 병렬 후보 profile/다음 작업을 유지한다. 최근100update wall 약3.82초 기준 Stage1 학습 잔여 약40.1분이며 최종검증 추가 필요. 나머지 세 조건 본학습 및 새 PDMS는 아직 없음.

2026-10-08 16:39 KST: 현재 약80.8%는 LPWM Stage1만의 진행률이며 네 실험 전체 진행률이 아니다. 4epoch 영상 복원/미래예측 개선은 확인됐지만 주행 관련 객체 보존·16particle 적정성·planning 이득은 미검증. 이번 축소 비교의 새 PDMS가 나오기 전 기존82/83점과 혼합하지 않는다.

등록 science hash 불일치: []; scheduler 검사: {'scripts/queue_four_model_small_corpus_overlap.py': True, 'configs/four_model_small_corpus/scheduling_overlap_v2.json': True}. 최종검증을 제외한 선형 ETA는 2026-10-08T17:19:59.996798+09:00. 미래정답을 관측 입력으로 넣지 않고 2관측→6미래로 검증한다. 평균 presence0.92851/position std0.55859/appearance std0.36747이나 의미론적 객체 구분 성능을 뜻하지 않는다.


## 2026-10-08 16:46 KST — LPWM Stage1 RGB 복원 및 미래예측 시각화

2026-10-08 16:46 KST RGB 시각화 요청: 저장된 before_training/update2620 배열만 CPU로 읽어 3개 비교 PNG를 생성했다. 학습·대기열·원121 science source/config는 변경하지 않았다.

2026-10-08 16:46 KST LPWM epoch4 RGB 정성검증: 고정 검증목록 첫3clip의 현재원본/학습전후복원 및 +3초정답/학습전후예측을 같은512×256 크기로 비교. 초기 Sketchy 가중치보다 도로·건물의 큰 형태가 개선됐지만 차량·보행자와 세부경계가 흐리거나 누락되며 미래예측은 현재복원과 비슷하게 남는 예가 보인다. 32clip 평균MSE 개선을 객체보존/정확한동역학의 증명으로 해석하지 않는다. 결과 `results/four_model_small_corpus_v1/rgb_validation_epoch4/`.

2026-10-08 16:46 KST 기존등록 학습·검증 대기열을 유지한다. RGB질문에는 observed reconstruction과 2관측→6미래 causal forecast를 구분하고, 정적배경의 큰 형태 개선과 작은객체/움직임 미보존을 함께 보고한다. 새학습/튜닝변형이나gate변경은 수행하지 않았다.

2026-10-08 16:46 KST RGB패널은 저장순서 첫3clip이며 성능으로선별하지 않았다. 현재복원은 영상인코딩/디코딩 결과이며 미래예측성공검사가 아니다. 미래패널은 t=-0.5,0만관측하고 +3s의출력을 표시한다. 전체32clip 집계MSE는 6개미래시점 평균으로, 표시한 +3s 단일프레임MSE와 다르다. RGB에서누락된객체정보가latent에도없는지는추가검증없이는단정하지않는다.

산출물: reconstruction_before_vs_epoch4.png, future_prediction_before_vs_epoch4.png, observed_reconstruction_future_overview_epoch4.png. 전처리 후 원본과 저장출력을 직접배치하고 sharpening/보간확대 없음. 비교 원본·배열 SHA와 실제단일프레임 MSE는 visualization_manifest.json에 보존. 새 script scripts/visualize_small_corpus_lpwm_rgb.py. 모델추론과GPU추가사용없음.


## 2026-10-08 16:49 KST — Drive-JEPA 표현 단위 설명

2026-10-08 16:49 KST 표현단위 설명: Drive-JEPA 논문3.1/3.2와 현재 JEPAFrontEncoder를 읽기 전용 확인. 학습·설정·대기열 변경 없음.

2026-10-08 16:49 KST Drive-JEPA backbone 출력은 시공간 patch latent token. V-JEPA SSL은 마스킹된 시공간패치의 EMA target feature를 예측한다. 현재 비교 구현은 patch16/tubelet2/ViT-L1024, planning 512×256의2frame→512patch token→adaptive_avg_pool1d16→Linear1024→256. Planning에서는 predictor를 제거한다. 16token pooling은 공통planner용 우리인터페이스이며 원논문 고유구조로부르지 않는다.

2026-10-08 16:49 KST 설명 시 register=학습된요약token, particle=위치/크기/presence/외관등구조화표현, JEPA=시공간patch feature를 구분한다. 모든token은문맥을담을수있으며 particle1개=의미론적객체1개를보장하지않는다. 기존대기열유지.

2026-10-08 16:49 KST JEPA의예측학습된관측feature와 LPWM의명시적미래particle rollout은다르다. 현재JEPA공통planner입력은16개 pooled patch token이며16개register/particle로해석하지않는다. 임의feature를RGB복원품질로만비교하지않는다.

근거: https://arxiv.org/html/2601.22032v1 §3.1/3.2 및 src/planning_aware_future_prediction/object_centric/small_corpus_models.py:29,59. 논문 버전의 데이터시간수치 등 질문외항목을 현행실험설정으로갱신하지 않았다.


## 2026-10-08 17:03 KST — 현재 RGB 뭉개짐 점검 및 과거 LPWM 동일장면 비교

2026-10-08 17:03 KST 과거 LPWM 동일장면 RGB 비교 완료: CPU2thread로 기존128 Stage1 20epoch 및 Adapter1epoch를 exact3scene 재추론. 현재epoch4 및 이전512/64particle200update는기존배열사용. 원121science source/config불변, GPU학습·대기열 변경 없음.

2026-10-08 17:03 KST 현재첨부PNG의3개복원패널은saved float출력의uint8변환과픽셀완전일치: 그림생성단계가뭉개짐원인이아님. 과거128/64encoder(30decoder) 모델은두번째흰차등을더잘보존하나보행자/세부손실은여전함. 과거Stage1대Adapter RGB차이는작음. 같은512/64particle200update도반복질감/흐림이있어16particle만을원인으로단정불가. 결과 `results/four_model_small_corpus_v1/historical_rgb_same_scenes/`.

2026-10-08 17:03 KST 시각화요청범위에서기존실행유지. 후속원인분리는직사각형encoder/decoder확장·particle수·학습량/노출·loss차이를통제해야한다. 평균RGB오차개선이나학습수치정상을충분한객체보존/주행적응성공으로취급하지않는다.

2026-10-08 17:03 KST 핵심비교제약: 첫2장면은이전Stage1의실제training frame이며현재는heldout. 세번째만양쪽Stage1미학습recording이다. 그림에명시했고일반화우열주장금지. 과거128출력은최근접픽셀로512×256표시크기만확대;정보추가없음. 이번공통8frame clip의두번째영상복원진단은원래12frame학습/4관측planning평가와구별한다.

새script `scripts/compare_lpwm_historical_rgb_same_scenes.py`. Native128 stage1 checkpoint hash71478ee548376bec21a4a22bc219929955aa0bfdd876f704676ade169fb4831f; Adapter는보존epoch01.pt. 실제command는동일이미지camera파일명을rawlog에서대응시켜추출. Encoded64/decoded30의공식filter_key를적용한manualdecode와canonicalforward 최대차0,full/slicedcontext최대차0,normalize_rgb=false. 초기CPU시도는비연속tensor와공식filter누락을잡는assertion에서실패했고입력contiguous·공식selection을새진단script에만반영후완료. 원학습/eval코드변경없음. 비교script는sampling이나모델학습을수행하지않았다. Sourcearrays/체크포인트hash와해상도별MSE,학습중복근거는comparison_report.json에보존.


## 2026-10-08 17:12 KST — 네 모델 축소 비교 진행률

2026-10-08 17:12 KST 진행조회: LPWM SSL 3179/3275(97.07%), 마지막5번째epoch. 기존queue294935/train136861의fresh heartbeat 유지. GPU0·1 각43.70GB/util100·98%. 실행·배치·대기열변경없음.

2026-10-08 17:12 KST: 양rank비유한loss0,3150update encoder/context/dynamics/decoder gradient유한양수. 최근100update wall3.68초,Stage1순수학습잔여약5.9분;저장/최종검증포함17:20–17:25KST조건부예상. 최신완료검증은4epoch이며새PDMS없음. 근거 `results/four_model_small_corpus_v1/progress_20261008_1712.json`.

2026-10-08 17:12 KST: 기존LPWM5epoch최종검증/gate후JEPA SSL+DrivoR 병행후보profile을진행하는등록queue유지. 조건통과시병행,아니면순차. 새로운실험설정변경없음.

2026-10-08 17:12 KST: 약97.1%는LPWM사전학습만의진행률이다. 나머지3조건본학습및각planner평가는아직남아있다. 수치안정성과표현품질/주행성능을구분하며RGB정성한계는보존한다.


## 2026-10-08 17:18 KST — Register/particle/patch token 예산 공정성

2026-10-08 17:18 KST token수 공정성조회: 현재구현/등록config와DrivoR논문Table4(c)읽기전용확인. 학습·대기열·기존science source121개변경없음. 추가16/32/64조건은제안만했으며미등록/미기동.

2026-10-08 17:18 KST: 공통planner입력모두16×256. DrivoR는703patch/frame×384와추가scene register16/frame를ViT에서처리후2framefusion16개;원DINOreg4/CLS는별도. JEPA는2frame tubelet의512patch×1024→평균pool16→256. LPWM은16FG+1BG,각FG의현재+8미래28차원속성→252→256. Shape동일성은표현용량동일성이아님. 근거 `results/four_model_small_corpus_v1/token_budget_fairness_audit.json`.

2026-10-08 17:18 KST: 기존16조건완주후LPWM내부particle16/32/64와공통집약방식/출력16×256을분리하는추가대조를권고. 동일공개초기화방식·노출·학습률·planner·검증목록으로각조건학습하고PDMS/latency/memory를함께비교한다. 단순중간증설은동일조건대조가아니다. 아직새실험실행승인은추가로가정하지않음.

2026-10-08 17:18 KST: Register16선택은원논문에서개수ablation후결정됐지만현재LPWM16의planning충분성은미검증. LPWM내부particle감소와ViT출력요약token수는역할이달라같은개수만으로공정성확정불가. 현재결과는공통planner·작은인터페이스의시스템비교이며LPWM/particle방식일반의우열로확장하지않는다. JEPA평균pool불이익가능성도같이명시.

논문 https://arxiv.org/html/2601.05083v2 Table4(c)/Sec4.2.1;현재코드 small_corpus_models.py:59/78/103. DIM수나특정외관4차원만으로총정보량을계산하지않으며LPWM에도기하/context/future정보가있음을명시한다.


## 2026-10-08 17:34 KST — 128→512 복원 전환 원인 진단 및 대기열 launch 복구

2026-10-08 17:34 KST 해상도 원인 진단 완료: CPU2thread, 기존128 Stage1 checkpoint/64encoder·30decoder/고정3clip로 decoder-only·encoder-only·full512 즉시 전환을 비교했다. 본학습 가중치·설정은 변경하지 않았다.

조회 중 기존queue294935가17:19에다음profile실행기록의중복command키로실패한것을발견했다. LPWM SSL은3275/3275(5epoch)완료·기존Stage1gate통과. 원scheduler보존후복구실행기 `scripts/queue_four_model_small_corpus_overlap_launch_fix.py`/PID1084697로이어갔다. 원121science source불변, 완료LPWM미재학습·완료JEPA8update profile재사용. 현재JEPA/DrivoR병렬profile실행중(조건통과후본학습);추가설정변경없음. 과거실패는scheduling_v2/launch_metadata_failure_20261008.json에보존,stale failed.json만복구확인후제거.

2026-10-08 17:34 KST 동일가중치/particle 즉시해상도전환: 공통128target MSE 평균 native128 .00928192 / decoder-only512 .02199972(2.370배) / encoder-only512 .02717889(2.928배) / full512 .03708157(3.995배). 첫2oldtrain·셋째heldout의고정3예시,새학습0. Decoder-only는동일latent/선택까지고정: 확장경로자체의복원교란근거이며현재16particle재학습품질의주원인확정은아님. 결과 `results/four_model_small_corpus_v1/resolution_transfer_diagnosis/`.

공간변경: CNN출력원래크기로adaptive pooling, BGdecoderseed8×8→16×32bilinear후원conv,glimpse32×32→64×128. beta_rec1→.125는공식loss C*H*W의8배확대를상쇄하므로단순감독약화로부르지않음. 기존동일512/200update/유효4 screen의16대64 reconstructionMSE+39.18%,forecastLPIPS비단조. 현재최종5epoch reconMSE.01212462/future.02285789/LPIPS.50129264. gate통과는완전적응·planning효용증명아님.

2026-10-08 17:34 KST: 새launch_fix실행기1084697의fresh queue_state를기준으로기존4조건대기열을이어간다. 기존294935/원queue를중복재기동하지않는다. 과거registration·source·실패보존;새실행기등록은scheduling_v2/registration_launch_fix.json. 진단상해상도extension을우선점검하되새128/512학습대조는미등록이다. 동일particle·클립노출·loss정규화로짧은재학습대조후원인기여를판정하는것을권고하며,기존비교실험은중간해상도/구조변경없이유지한다.

2026-10-08 17:34 KST: decoder-only2.37배는128NAVSIM가중치를재학습없이확장한3장면복원검사의값이다. 현재512에서5epoch학습한모델의악화율/전체검증/PDMS로보고하지않는다. 공통128평가가고해상도세부이득을반영하지못함,encoder-only는pool/anchors/mask/input보간등의통합효과임을명시. 첫2과거학습장면중복이있으나동일장면쌍의전환검사이며일반화검증아님. 복구한queue는metadata기록만수정했고9개검사통과·실제양profilelaunch확인;원모델/학습설정불변.

새진단scripts/diagnose_lpwm_resolution_transfer.py: native128 parameter tensor와rectangular parameter tensor전수일치·native복원과이전보존값maxabs<1e-6확인. Native128로양쪽출력을area평균한동일target에서MSE비교;이미지는128만nearest표시확대,512출력원본보존. decoder-only는z/scale/feature/presence/depth/background/filterkey동일. Native해상도NPZ와sourcecheckpoint SHA보존. 원서버GPU0·1idle와관련host PID종료확인후복구;새metadata는process_command문자열과command목록을구분. 새로운실행등록만별도파일이며기존scheduler/config무변경. 기존8개admission검사+실제CPUsubprocess를통한launch회귀검사(일반/CPUslot)통과. 근거results/four_model_small_corpus_v1/scheduler_launch_recovery_20261008.json.

17:35 KST 후속확인: JEPA/DrivoR 병렬profile은속도·메모리·중첩을통과했으나등록loss일치기준을실패해기존규칙대로순차선택했다. 비교profile가중치는폐기되며현재JEPA SSL 본학습train1099274부터시작,DrivoR는이후자동실행. 기준완화·학습조건변경없음.

JEPA 본학습 실제26update/416clip·encoder/predictor gradient유한양수확인. 18MB 진단원배열은git외outputs/four_model_small_corpus_v1/diagnostics/resolution_transfer/에저장하고보고서에경로/SHA보존.


## 2026-10-08 17:46 KST — 축소 비교 학습 진행 보고

2026-10-08 17:46 KST 진행조회: queue1084697/train1099274 fresh heartbeat, JEPA SSL267/3275(8.15%,첫epoch40.76%). LPWM SSL3275/3275·5epoch 학습/최종검증완료. 네 planner조건은아직대기, PDMS없음. GPU0·1각9.97/9.90decimalGB(17:45 snapshot), util31/54%. 이번턴조회만수행, 학습·설정·대기열변경없음.

2026-10-08 17:46 KST: JEPA양rank267개기록loss비유한0,최근encoder/predictor gradient유한양수. LPWM최종32clip reconstructionMSE.01212462/futureMSE.02285789/futureLPIPS.50129264;초기대비각81.52%/66.80%/42.97%감소,현재반복futureMSE대비29.28%감소. 기존planning진입gate통과이며객체보존/해상도문제해소/PDMS성공은미확인. 원121science source와새scheduler/config해시불변·active failure0. 근거 `results/four_model_small_corpus_v1/progress_20261008_1746.json`.

2026-10-08 17:46 KST: 기존queue유지: JEPA SSL→DrivoR planning→LPWM순차/JEPA planning(실측병렬admission)→LPWMjoint, pass1/3/5 CPU PDMS. 현재JEPA최근50–200update wall2.62–2.83초/update로첫epoch18:03–18:05,전체SSL학습19:58–20:08KST조건부외삽;앞으로epoch검증/부하변화추가. 전체4조건종료시간이아님. 첫pair는loss일치기준실패로기존규칙대로순차이며해당기준을완화하지않음.

2026-10-08 17:46 KST: 8.15%는JEPA Stage1진행률이며전체실험진행률이아님. 학습수치안정성만확인,JEPA첫epoch검증은아직없음. LPWM복원/해상도진단한계보존;현재512×256·16particle조건은그대로이며새해상도재학습대조미등록. 이번축소실험PDMS미산출,기존81/82점과혼합금지.

JEPA micro2×누적4×GPU2=유효16,worker4/rank,bf16. Training loss와before_training latent validation loss는서로다른표본이므로차이를검증개선으로주장하지않는다. 현재PID는fresh queue파일기반,host PID추가조회없음.


## 2026-10-08 19:20 KST — GPU48GB 제한 내 세 작업 병행 적용

2026-10-08 19:20 KST 사용자 GPU48GB이내 추가병행 지시로 세 작업 본학습을 동시에 실행했다. 새controller1554349 / `scripts/queue_small_corpus_three_jobs.py`, root `outputs/four_model_small_corpus_v1/scheduling_v4_three_jobs/`. JEPA SSL1210957(2210/3275),DrivoR1567970(54/3200),LPWM순차Stage2 1580030(5/3200). 세작업각각GPU0·1 micro2×누적4=유효16,기존worker/LR/데이터/seed/loss/5epoch보존. 원121science source불변.

초기반복검사를위해JEPA451에model/AdamW443state/scheduler/2rankRNG를저장후같은trainer로재개했다. 1084697은정상종료,중간1210946은이후새controller가검증된해당PID만종료하면서train1210957을그대로인계했다. 원본학습update누락/중복0. old1084697/1210946/294935/원queue 재기동금지.

2026-10-08 19:20 KST 실행재현성검사: JEPA 단독재실행loss최대상대.12546%,기존병렬차이.05002%;DrivoR단독.02465%/병렬.01988%. DrivoR3장면궤적차이평균단독3.44mm/병렬2.64mm. 최초1e-4 gate와v3 .1%cap에serial도묶은실패기록보존;v4는병렬간차이에.1%cap·실측단독변동이내·고정검증/출력cap을적용한다고명시해첫pair채택. exactPDMS동등성증명아님.

LPWM추가8update비교(JEPA와2작업 vs JEPA/DrivoR와3작업)는loss상대1.50e-6/gradient상대1.15e-4,3장면궤적평균차이.0826mm/최대.1640mm. LPWM9.0767→9.2536초/update,동등update블록조건부약1.268배,최대카드17.7996GB,전체gate통과. 세본학습양rankloss비유한0·encoder/context/dynamics/geometry/commandgradient확인. 공통planner초기241tensor동일. 17개검사통과. 근거 `results/four_model_small_corpus_v1/parallel_execution_20261008/execution_report.json`.

2026-10-08 19:20 KST v4 대기열유지: JEPA SSL·DrivoR planning·LPWM순차planning 동시진행. JEPA SSL완료/gate후JEPA planner를빈자리에넣어최대3작업. LPWMjoint는기존최대실측43.6GB급이므로다른GPU학습이모두완료된뒤단독실행. CPU공식PDMS pass1/3/5 예측완료즉시1작업/4worker로병행. profile카드44GB/실학습46.5GB저장중단/사용자48GB상한보존. 상태는root queue_state와v4의failed/paused를확인하며과거v2/v3 paused/failed는현재정지근거가아님.

2026-10-08 19:20 KST 변경은실행순서·동시작업수·수치재현성admission해석뿐이다. 원훈련코드·배치·optimizer·데이터·loss·가중치초기화·등록science해시불변,모든profile가중치폐기. v3단독재실행까지묶은엄격cap실패를지우거나통과로덮어쓰지않고v4별도규칙수정근거저장. 병렬차이가단독변동보다작다는8update검사이며최종PDMS동일성보장은아님. 약1.268배는짧은동등update블록대비추정으로전체실험종료시간단축률로보고하지않는다. 신규PDMS아직없음. 512해상도/16particle조건변경없음.

실행파일: reassess_small_corpus_parallel_execution.py / queue_small_corpus_calibrated_parallel.py(과거중간controller) / queue_small_corpus_three_jobs.py(현재). 신규실행환경은원래kjs-lpwm-drivor-joint그대로. CurrentGPU PID와source hash·profilegate·resume443AdamW/scheduler451·양rankRNG2·sampler누락중복검사는위JSON에보존. v4 controller인계는SIGTERM handler가trainer를pause시키므로oldcontroller1210946의UID/cmdline/activejobs확인후해당PID만SIGKILL,분리session의torchrun1210957 start_ticks/생존확인후root queue lock획득했다. 기존모델은재시작하지않았으며실제추가학습행확인. 새controller와원래새train만사용하며기존controller중복기동금지.


## 2026-10-08 20:04 KST — 세 병행 학습 점검 및 첫 DrivoR PDMS

2026-10-08 20:04 KST 진행 점검: v4 controller1554349와 세 학습 JEPA SSL1210957 / DrivoR1567970 / LPWM순차1580030 정상 실행. JEPA3129/3275(95.54%), DrivoR887/3200(27.72%), LPWM Stage2 270/3200(8.44%). LPWM Stage1은5epoch완료, JEPA planner와LPWMjoint는대기다. GPU0·1 전체카드17.80/17.67decimalGB, 양rank비유한loss0·최신modulegradient양수. 원121science 및v4실행6source불변, active failure없음. 이번에는조회만했고학습/설정/대기열변경없음.

2026-10-08 20:04 KST 새 결과: DrivoR pass1(640update) 공식PDMS66.9414 / ADE4.9712m / FDE10.8544m, 독립planning dev1024장면·실패0·train/dev recording중복0. 무과실충돌98.78%,도로준수85.25%,진행률.4623. 최신887update의점수가아니며전체navtest/원논문재현도아님. `validation/pass1.json`의pending은정적이력이며실제완료근거는`drivor/validation/pass1.pdms.json`이다.

JEPA 고정32held-out기록 masked latentL1 초기.651997→pass1 .514094→pass2 .504583→pass3 .498893→pass4 .495580(약24%감소), feature std2.253→2.355. 최종gate는아직대기. LPWM Stage1 futureMSE.0228579/persistence.0323237(29.28%감소),기존gate통과유지. Snapshot `results/four_model_small_corpus_v1/progress_20261008_2004.json`.

2026-10-08 20:04 KST 현재부하의최근50–200update wall속도기준: JEPA Stage1학습경계20:11경,최종검증포함약20:15이후gate통과시JEPA planner자동시작. DrivoR5epoch학습22:06–22:12, LPWM Stage2 첫epoch21:05경/5epoch10월9일04:05–04:15조건부추정. 앞으로epoch검증·PDMS·동시작업교체의부하변화는별도이므로전체4실험완료시각으로해석하지않는다. 기존v4대기열/48GB상한/LPWMjoint독점규칙유지.

2026-10-08 20:04 KST 첫DrivoR PDMS만산출됐고다른세조건학습후PDMS미산출. 서로같은pass/1024dev장면결과가나오기전모델우열·particle이득을주장하지않는다. 과거81/82점의다른실험과직접비교금지. Loss유한/gradient양수는실행정상근거이며객체정보보존·planning개선증명아님. LPWM512×256/16particle RGB복원한계는보존한다.


## 2026-10-08 21:04 KST — 네 번째 LPWMjoint 병행 시작

2026-10-08 21:04 KST 사용자 추가병행 지시 적용: 새 v5 controller2114751 / `scripts/queue_small_corpus_four_planners.py`, root `outputs/four_model_small_corpus_v1/scheduling_v5_joint_overlap/`. 기존DrivoR1567970·LPWM순차1580030·JEPA1891906은 PID/start_ticks 유지로 인계했고, LPWMjoint2114768을 공개초기화부터 새로 시작했다. 현재 drivor 2066/3200, jepa 659/3200, lpwm_sequential 640/3200, lpwm_joint 8/3200. 네 본학습 양rank loss유한·필요modulegradient양수. 이전controller1554349만 종료했으며 기존학습 재시작없음.

2026-10-08 21:04 KST 메모리/수치검사: 같은 실제2scene+2SSL clip/같은RNG에서 native 대 decoder+LPIPS activation checkpointing의 loss차이0,936개 gradient tensor상대L2 .000121019(0.0121%),GPU allocated20.10→12.43GB. RNG/gradient/원래모드복원안전조건 CPU검사2개통과. 별도DDP8update(마지막2배SSL) 네작업동시시험완료,whole-card최대43.1269GB,일반단계약15–16초/update,가중평균16.29초. 기존세조건도각75/43/23update진행. 공통planner초기241tensor동일. 실제본학습GPU0/1약31.91/31.95GB·util100/100%. 원science121 및v5등록6source불변. 근거 `results/four_model_small_corpus_v1/four_jobs_execution_20261008/report.json`.

CPU saved-tensor offload시험은180초이상 update미완료로기각·폐기. 첫checkpoint시험은allocator제한OOM으로종료(본학습이아님);그때남은offloadworker2045695를명시적으로정리했다. 실패로그보존후expandable_segments:True로재시험성공. GPU전체48GB상한·기존세학습은보존됐다.

2026-10-08 21:04 KST 네planner를병행하고pass1/3/5예측완료즉시기존CPU공식PDMS(동시1/worker4)계속. 새joint는기존micro2×누적4×GPU2=유효16,worker2/rank,loss/LR/seed/데이터/총노출그대로. RGB decoder와LPIPS만비재진입activation checkpointing;그외연산과원trainer불변. 동시실행중새allocator27GB,전체카드46.5GB초과시저장중단/사용자상한48GB. 다른세GPU학습완료후v5가native_joint_allowed.json을발행하고각rank에서외부+context2GB이하확인시재시작없이기존activation저장/allocator44GB로전환해재계산비용제거. 기존v4및과거controllers재기동금지.

2026-10-08 21:04 KST 변경은추가병행및activation저장/재계산방식뿐이다. 기존모델/학습조건/실험수변경없음,profile가중치본학습미사용.8update메모리·gradient수치범위확인이최종PDMS동등성증명은아니다. 짧은2단계시간모형은26.91→17.67시간/1.52배를산출했으나과거native2update startup이포함되고향후eval/부하변화가있어확정ETA/보장단축률로보고하지않는다. 최대43.13GB는실측profile고점이며미래모든부하의상한을보장하지않음;기존pause/allocator보호를유지한다.


## 2026-10-08 21:09 KST — JEPA/DrivoR 학습 단계 설명

2026-10-08 21:09 KST 단계구성질문에현재config/model/JEPA complete+gate를읽기전용확인. 기존v5 네planner병행유지,실행·배치·loss변경없음.

2026-10-08 21:09 KST 현재Drive-JEPA방식은공개일반영상V-JEPA2→고정10480주행clip×5의maskedlatent SSL→완료/gate후encoder체크포인트+공통DrivoR planner학습이다.3275update Stage1완료. Stage2에서는SSL predictor/EMA target을제외하고encoder native1e-5/planner1e-4로planning loss학습. DrivoR조건은공개DINOv2 ViT-S reg4→곧바로planning이며추가주행SSL Stage1없음(config stage1:null); q/v LoRA rank32와scene register/입력interface/planner학습. 공개DINO사전학습과이번자체주행Stage1을구분한다.

2026-10-08 21:09 KST 기존네조건학습/검증대기열을유지한다. 단계설명에따른새Stage1추가나학습방법변경은없음.

2026-10-08 21:09 KST JEPA/LPWM순차는같은주행SSL clip을쓰지만DrivoR에는그추가노출이없다. 공통planning데이터/모듈비교이지네조건의총SSL노출·상류사전학습데이터가동일한실험은아님. 현재Drive-JEPA명칭은공통DrivoR planner를쓴변형조건이며원논문전체재현으로설명하지않는다.


## 2026-10-08 21:58 KST — 보호 메모리 중단 진단 및 세 작업 복구

2026-10-08 21:58 KST 메모리 보호 정지에서 복구: v6 controller2415190, DrivoR2415910 / JEPA2415917 / LPWM순차2415934가 GPU0·1에서 실행 중이다. 실제 update는 각각 2616/1025/798로 저장점2453/911/743 이후 증가했다. LPWMjoint는87 fullstate에서 메모리 admission 대기이며 자동 재개한다. 새 root scheduling_v6_memory_recovery, 기존v5는21:27 failed 이력이며 재기동금지. 21:56 GPU0/1은14957/14958MiB. root queue_state가 현재 상태다.

2026-10-08 21:58 KST 중단 원인: 전체카드47,542,435,840bytes가46.5decimalGB 저장/중단 기준을 넘어 DrivoR가 저장 중단했고, v5가 이를 전체 실패로 해석해 나머지도 중단시켰다. 본 실패의 OOM/NaN 근거없음. joint86의2회SSL부하와 겹쳤으나 정확한 프로세스별 peak 원인은 historical NVML 부재로 미확정. 8update profile43.13GB를 장기 peak 보장으로 사용한 여유 판단이 부족했다. 네 fullstate의 Adam step/scheduler/RNG2개 보존 확인, 신규3작업은 loss/gradient 유한·실제 진행. science121 및 old/new execution source hash 불변. CPU 검사3개·compile 통과. 근거 results/four_model_small_corpus_v1/memory_recovery_20261008/report.json.

2026-10-08 21:58 KST v6는 최대3작업, 모델별 보수적 예약량+실측카드사용량으로44GB admission을 적용한다. DrivoR→JEPA→LPWM순차 우선, joint는87부터 메모리가 충분할 때 자동 재개. 보호중단은 해당 작업 재대기·동시수 감소로 처리하며 정상 동반작업을 유지한다. 사용자 pause는 보존하고 미확인 오류는 기존 저장중단 처리. batch2/rank×accum4×GPU2=16, LR/loss/data/5pass 불변. pass1/3/5 CPU PDMS 1작업/4worker 및 최종비교 자동 연결. joint 단독 시 기존v5 native marker 경로를 사용해 adaptive wrapper를 유지한다. 새 재개 wrapper는 loss 초기화 후 첫 train() 시 양rank RNG를 복원하며 resume_records에 증거를 저장한다. 최신 checkpoint 원본은 before_recovery에 hardlink로 보존했다.

2026-10-08 21:58 KST 이번 변경은 보호정지 복구와 병행 스케줄 수정이며 과학적 조건 변경이 아니다. 48decimalGB 사용자 상한/46.5GB trainer guard 유지,44GB는 예약 admission 기준이지 모든 미래 순간사용량 보장이 아니다. short profile이 장기 peak를 충분히 대표하지 못한 사실과 기존 실패 기록은 보존한다. GPU 수치연산의 bitwise 동일성·최종PDMS 동일성을 보장하지 않는다. joint는 아직87에서대기이며 네학습모두동시실행이라고 보고하지 않는다. 새 queue main은 failed v5 상태에서 최초 복구용이므로 활성 v6 위에 중복기동하지 않는다.


## 2026-10-08 22:02 KST — 다음 학습·평가 순서 확인

2026-10-08 22:02 KST 다음 작업 계획 조회: v6 복구 대기열 유지, DrivoR 2692/3200·JEPA 1077/3200·LPWM순차 823/3200 진행, LPWMjoint87에서메모리대기. LPWM/JEPA SSL은각5pass완료. 이번턴실행·조건변경없음.

2026-10-08 22:02 KST 현재완료dev PDMS: DrivoR pass1 66.9414/pass3 78.3977, JEPA pass1 66.9890, LPWM순차 pass1 64.9743. 모두동일독립1024scene이며서로다른pass를최종우열비교하지않음. 근거 results/four_model_small_corpus_v1/next_work_plan_20261008.json.

2026-10-08 22:02 KST 다음순서: 현재세planner 계속→예약메모리충족시joint87자동재개→각1/3/5pass 독립dev1024의예측/CPU공식PDMS→고정pass5네조건비교. 모든planning목표는5pass/3200update이며25epoch아님. 비교시PDMS·ADE/FDE·명령별세부지표와학습량/연산비용을함께정리. 표현분석은고정장면의particle전후/겹침및미래표현개입을검토할계획이며현재자동대기열에새진단이나학습을추가하지않았다.

2026-10-08 22:02 KST 이번요청은다음계획설명이며새실험기동아님. 현대기열은학습/1·3·5pass평가/최종JSON까지만자동화됨. 결과해석은공통planner상태에서의시스템비교와LPWM순차vsjoint의경향성에한정;초기사전학습/백본규모차이·1seed·부분dev의제약유지. 위치이동만으로planning효용판정금지,표현개입등추가진단은필요시다음분석으로명시.


## 2026-10-08 22:05 KST — DrivoR 종료시간 추정

2026-10-08 22:05 KST DrivoR 종료시간 조회: 2741/3200(85.66%),잔여459update. 기존v6 controller2415190과3학습계속,LPWMjoint87대기. 실행변경없음.

2026-10-08 22:05 KST DrivoR 최근50/100/200update의누적wall차이기준 3.173–3.275초/update. 이전pass1/3 추론+저장약28/23초,CPU공식PDMS24/22초. 근거 results/four_model_small_corpus_v1/drivor_completion_eta_20261008.json.

2026-10-08 22:05 KST 현재부하유지시DrivoR 학습종료10/8 22:30–22:35KST,독립dev1024 최종PDMS포함22:32–22:38추정. 학습3200→GPU추론/저장→CPU채점은이미자동연결. 이번시간추정으로대기열수정없음.

2026-10-08 22:05 KST ETA는현재3작업부하와최근wall처리량기준이며추가보호중단·평가대기·서버부하변화시달라진다. 해당시간은DrivoR만의종료이며전체네실험완료시간이아니다. 결과는전체navtest가아닌독립dev1024.


## 2026-10-08 22:13 KST — DrivoR 완료 직후 LPWM 공동학습 우선 전환 등록

2026-10-08 22:13 KST 사용자 DrivoR 직후joint우선 지시 적용: v7 controller2633960, scheduling_v7_joint_after_drivor. 기존DrivoR2415910/JEPA2415917/LPWM순차2415934를PID·start_ticks유지로인계했으며학습재시작없음. 현재update 2889/1204/888, joint87대기. 기존v6 controller2415190만종료했고재기동금지. phase wait_drivor_final_evaluation.

2026-10-08 22:13 KST 전환검사3개통과: DrivoR3200+pass5 PDMS1024/실패0이둘다완료되어야trigger;계획된JEPA yield는checkpoint보존하며memory실패로분류하지않음;joint+LPWM순차예약43GB허용/joint+JEPA+LPWM순차거부/joint완료후JEPA보류해제. 원science121/기존실행6hash불변,신규실행5hash등록. 인계후세학습실제진행확인. 결과 results/four_model_small_corpus_v1/joint_after_drivor_20261008.json.

2026-10-08 22:13 KST 최신순서: DrivoR 최종학습·평가완료→JEPA에소유marker로저장중단요청→종료/paused/latest확인및marker이력보존→joint87 fullstate재개. LPWM순차는메모리허용시계속병행. 사용자48GB상한·46.5GB보호·44GB예약기준유지,추가memory중단시동시수축소하며1작업만가능하면순차도저장대기. joint학습완료후JEPA전체상태자동재개,기존pass1/3/5 CPU평가와최종비교계속. 따라서DrivoR이끝난후JEPA학습완료까지joint가대기하던v6순서는폐기.

2026-10-08 22:13 KST 변경은사용자지시에따른작업우선순위뿐이다. 기존micro2/rank×accum4×2GPU=16·LR/loss/data/5pass유지,새학습조건없음. 지금joint는87에서대기이고전환미발생;실제시작은DrivoR완료와JEPA체크포인트저장/메모리확보후다. 사용자root/per-job pause는보존한다. v7최초인계runner를활성queue위에중복실행하지않는다.


## 2026-10-08 22:34 KST — DrivoR 완료 및 LPWM 공동학습 자동 재개 확인

2026-10-08 22:34 KST DrivoR5pass/3200학습22:30:39완료,최종PDMS22:31:32완료. v7자동전환성공: JEPA1428에fullstate저장대기,LPWMjoint2755586이22:32:13시작해87→93실제진행. LPWM순차2415934는1014/3200으로계속병행. controller2633960유지,phase joint_priority_after_drivor. 이번턴조회만수행.

2026-10-08 22:34 KST DrivoR 독립dev1024 최종PDMS81.2535/ADE1.83710m/FDE4.46461m/실패0. pass1 66.9414→pass3 78.3977→pass5 81.2535. train/dev recording중복0. 네조건양rank최근100row(또는전체87이하) 비유한loss/gradient0,원science121/실행5hash불변. joint양rank RNG복원proof완료,실제88이상update확인. 근거 results/four_model_small_corpus_v1/progress_20261008_2233.json.

2026-10-08 22:34 KST LPWMjoint와LPWM순차학습계속. JEPA1428은의도된메모리양보상태이며joint학습완료후자동재개. pass1/3/5의동일1024dev평가·고정pass5최종비교대기열유지. DrivoR완료실험재기동금지;동일학습량의나머지결과가나오면비교한다.

2026-10-08 22:34 KST DrivoR81.25는이번축소실험dev1024의5pass결과이며전체navtest/공식원논문재현아님. JEPA/LPWM의기존pass1과최종우열직접비교금지. joint는재개직후이므로장기처리량/종료ETA를첫update로추정하지않는다. 현재joint+순차메모리는최신양rank학습로그기준약25.48decimalGB,48GB상한유지.


## 2026-10-08 22:38 KST — 네 조건의 planning/SSL 데이터 동일성 확인

2026-10-08 22:38 KST 네조건데이터동일성조회: 기존v7대기열/학습설정유지,데이터추가·재학습·실행순서변경없음.

2026-10-08 22:38 KST 네planning등록의manifest SHA d08760b560235353da8f297d7beee6807e772f36c7d42c6c1b2ae73548896394가실제manifest와모두일치. navtrain10240/101recording,navval1024/61recording,기록중복0. planning5pass/51200노출/3200update/유효16/4800+epoch순서공통. JEPA·LPWM순차·LPWMjoint의SSL은동일OpenScene10480clip×5=52400노출목표,실제prepare_records함수로목록·4700+pass순서를재구성해LPWMStage1저장hash16535cf4…와일치확인. DrivoR는이추가SSL없음. SSL훈련과planningdev기록중복0,원science121불변. 근거 results/four_model_small_corpus_v1/dataset_comparison_audit_20261008.json.

2026-10-08 22:38 KST 공통5pass planning학습·평가대기열유지. 최종결과에는같은planning데이터/순서/목표노출과서로다른SSL노출·외부초기사전학습을구분해서보고한다. 이번질문에따라DrivoR Stage1을추가하거나데이터를변경하지않았다.

2026-10-08 22:38 KST 데이터공정성정리: planning학습/평가데이터는네조건정확히동일하지만전체학습데이터노출은동일하지않다. DrivoR만추가주행SSL없고DINOv2/V-JEPA2/LPWMSketchy의공개초기사전학습도다름. LPWM순차vsjoint는SSL및planning최종노출목표를같게하고학습시점/목적구성을바꾼비교. 현축소SSL은OpenScene만사용하며CoVLA·DrivingDojo·330h전체아님. 목표노출과현재까지소비량은구분한다.
