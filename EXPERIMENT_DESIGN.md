# 실험 설계: 어떤 미래 정보가, 어떤 상황에서, 얼마나 필요한가

작성 2026-09-20. 연구 명제는 **"planning 에 필요한 미래 정보는 상황마다 다르고,
사람이 미리 정하면 안 된다"** 이다.

## 왜 이 형태인가 — 선행연구가 남긴 빈틈

가장 가까운 선행은 PerceptDrive (arXiv 2607.20175, NAVSIM v1 90.4 PDMS / v2 90.2 EPDMS)
이고, scene-conditioned router 로 geometric·semantic·dynamic 세 prior 를 맥락별로
가중한다. 그러나 네 가지를 하지 않는다.

| 항목 | PerceptDrive | 이 연구 |
|---|---|---|
| 인과 검증 | 추론 시 token masking. 저자 자인 *"probes reliance rather than isolated causal effects"* | **재학습 기반 ablation** |
| 예산 스윕 | 없음 (항상 3개 전부 soft gating) | **K=1..N 정보 요구 곡선** |
| 맥락 분해 | navigation command 3종 | **난이도 계층 × 시나리오 유형** |
| 라우팅 대상 | **현재 프레임** prior | **미래 정보의 종류** |

DA-WAM (2608.19085) 은 후보 궤적마다 다른 future latent 을 만들지만 **종류는 하나**이고
맥락별 적응이 없다. DriveFuture (2605.09701, navhard 1위 55.5 EPDMS) 도 단일 latent 이다.

즉 **"미래 정보의 종류를 상황에 따라 고른다" + "얼마나 필요한지 인과적으로 측정한다"**
가 비어 있다.

## SafeDrive 에 이미 있는 미래 정보 4종

새 데이터 파이프라인이 필요 없다. GT 가 전부 캐시에 들어 있다.

| # | 표현 | GT 키 | 성격 | 현재 planner 도달 |
|---|---|---|---|---|
| **F1** | 미래 BEV semantic (4 프레임) | `all_future_frame_bev_semantic_map` | dense / occupancy | **없음** (loss 전용) |
| **F2** | agent 미래 궤적 | `motion_traj`, `motion_mask` | object-centric / vector | 있음 |
| **F3** | pairwise 미래 충돌 (agent×시점) | `pair_collision_gt` | 상호작용 | 있음 (`pwnc_heads`) |
| **F4** | time-wise 주행영역 준수 | `twdac_gt` | 정적 / 도로 | 있음 (`twdac_heads`) |

F1 이 특히 중요하다. `safedrive_model.py:585-586` 에서 `fut_bev_map` 을 계산해 `output`
에 넣고 끝이다 — **사람이 "미래 dense world 가 필요하다"고 가정해 넣었지만 구조적으로
planning 에 기여할 경로가 없다.** 연구 명제의 교과서적 사례이며, 첫 실험 대상이다.

추가 arm 으로 **F5 = 무감독 미래 latent** 을 둔다. 재구성 loss 없이 planning loss 로만
학습되는 latent 이며, DA-WAM 계열과의 비교군이자 질문 1("사람이 정하지 않은 표현")의
직접 구현이다.

## 실험 행렬 — 전수 대신 최소 설계

N=4 의 전체 부분집합은 15개이고 각 재학습이 12~15시간이라 비현실적이다
(GPU 4장에 2 run 병렬 기준 약 4일). 다음 9 run 으로 줄인다.

| 그룹 | run 수 | 구성 | 무엇을 답하나 |
|---|---:|---|---|
| **All** | 1 | F1+F2+F3+F4 | 상한 기준선 |
| **Leave-one-out** | 4 | 하나씩 제거 | 각 표현의 **한계 기여** |
| **Only-one** | 4 | 하나만 사용 | 각 표현의 **단독 가치** |

두 그룹을 함께 보면 상호 대체성(substitutability)까지 드러난다 — 단독 가치는 높은데
한계 기여가 0 이면 다른 표현이 같은 정보를 담고 있다는 뜻이다.

**맥락 분해는 재학습 없이 공짜로 나온다.** 평가 CSV 를 시나리오 라벨과 join 하면 되므로,
9 run 이 곧 "어떤 상황에서 어떤 표현이 필요한가" 전체 지도를 준다.

### 비용

```
1 run  = 5 epoch × 약 2.4~2.9 h = 12~15 h   (batch 24, GPU 2장)
9 run  ÷ 2 병렬 × 13 h           = 약 58 h ≈ 2.5 일
평가   = run 당 navtest 45분 + 채점 2분
```

F5 arm 을 추가하면 +2 run (All+F5, Only-F5) 약 13시간.

## 평가 — 집계 점수를 쓰지 않는다

navtest 집계 PDMS 는 만점 장면이 30.1% 라 신호를 덮는다. 측정 결과 난이도별 여유가
20배 차이난다.

| 구간 | n | Phase 3 실제 | oracle | 여유 |
|---|---:|---:|---:|---:|
| 쉬움 | 510 | 92.66 | 93.00 | +0.34 |
| 보통 | 958 | 90.61 | 93.19 | +2.57 |
| 어려움 | 532 | 89.11 | 96.04 | **+6.93** |

따라서 **난이도 계층별 PDMS 를 주 지표로 쓴다.** 계층 정의는 잠정적으로
(20 m 내 agent 수 ≥ 15) 또는 (4초 요구 전방변위 > 45 m) 이며, 논문용으로는 더 원칙 있는
정의가 필요하다.

### 벤치마크

- **navtest 난이도 계층** — 주 반복 지표. 인프라가 이미 있고 어려움 구간에 6.93 여유
- **navhard (NAVSIM v2)** — 최종 주장. SOTA 55.5~56.6 EPDMS 로 전혀 포화되지 않았고
  two-stage pseudo-simulation 이라 오차 누적이 드러난다. **이 fork 에는 v2 코드가 없어
  공식 devkit 이전 필요** (`traffic_agents: non_reactive` 문자열만 있고 구현 없음)
- **Bench2Drive** — 폐루프 일반화 주장이 필요할 때. multi-ability 평가가 맥락 분해 축과
  일치하지만 CARLA 스택이라 수 주 규모 작업

## 2단계 — 측정 다음에 오는 method

1단계 측정이 "상황마다 필요한 표현이 다르다"를 보이면, 2단계는 그 선택을 학습시킨다.
gate 는 planning loss 로만 학습하고 예산 K 를 제약한다. 1단계의 정보 요구 곡선이
**gate 가 학습해야 할 정답의 상한**을 제공하므로, gate 가 oracle 선택에 얼마나
근접하는지로 평가할 수 있다.

MoE 의 gate collapse 가 주된 기술 리스크이며 load-balancing 이나 명시적 sparsity budget
이 필요하다.

## 지금까지 확보된 근거 (2026-09-18~20)

재학습 기반 인과 ablation 을 이미 두 건 수행 중이며, 방법론이 검증된 셈이다.

| 실험 | 결과 |
|---|---|
| α 보행자를 월드에 포함 | +0.09, CI [−0.17,+0.36] — 효과 없음 |
| E1 실패 귀인 | 충돌 agent 의 98.1% 가 이미 월드 안 |
| E4 월드 범위 64 m | 초과 장면 2.1% — 범위 아님. 대신 89.7% 장면에서 9.21 m 덜 감 |
| E5 EP 가중치 | Phase3 +0.46 / Phase2 +1.22 (유의) — 단 하이퍼파라미터 튜닝이라 기여 아님 |
| E6 proposal oracle | 256 anchor 중 최선 93.90, 중앙값 3.95 — 모델은 이미 2~4등을 고름 |
| **E2 perception 진짜 동결** | 진행 중 (epoch 4/5). epoch 0 시점 −0.38 (114M→19.2M) |
| **E3 월드 25→5** | 진행 중 (epoch 3/5) |
