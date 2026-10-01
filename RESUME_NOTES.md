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
