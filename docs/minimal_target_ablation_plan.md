# 최소 target 비교 학습 계획 — 실행 전 고정안 v1

**이 문서는 `09913b5` 당시의 계획 기록이다.** 이후 사용자 승인으로 구현·A–E 200-update 실행을
완료했다. 현재 상태는 [실측 결과](target_supervision_exploration_results.md), 실행 설정은
`configs/exploration/target_supervision_run_v1.json`을 따른다. 아래 ‘미구현/미실행’ 문구는 당시 상태이며,
§6의 초기 shortlist threshold는 자동 적용하지 않았다. 최신 지시대로200-update 순위로 target을 확정/탈락시키지 않는다.

2026-10-01, 기준 `1231767` 이후 조사. **아직 학습하지 않았다.** Training CLI, fixed-rule forward,
feature cache, train-only normalization, evaluator는 다음 구현 작업이다. 이 문서/JSON만으로 학습이
실행 가능하다고 보고하지 않는다. GPU 연결 smoke를 반복하거나 동적 K/horizon을 구현하지 않는다.

질문: **동일한 현재 정보·객체 선택·예산·branch에서 어떤 미래 보조 감독이 ego 경로 예측에 유용한가?**
이 탐색은 target 선정용이며 최종 planning 성능/H1/H2/novelty 검증이 아니다.
연구 결정은 [한 페이지 요약](research_question_and_target_decision.md), 입력 유효율은
[조사 결과](navsim_visual_target_coverage.md), 설정은 [target_ablation_plan.json](../configs/exploration/target_ablation_plan.json).

## 1. 먼저 바꾸는 것과 고정하는 것

Frozen 공식 Drive-JEPA encoder와 현재 GT ROI/state, 새 predictor/planner scaffold를 재사용한다.
Selector는 **학습하지 않는다**. 현재 ego 거리로 front-valid 차량·보행자 중 최근접 K4를 고르고,
동점은 기존 stable track ID 순서로 처리한다. 미래 target 유무를 선택에 사용하지 않는다.
4개 미만이면 padding,0개라도 현재 정보로 planning한다. Nominal K와 active count를 함께 기록한다.

현재 입력은 두 frame front image grid, 모든 현재 front-valid entity feature, ego 상태/command다.
K4는 예측 대상에만 적용하며 현재 관측을 가리지 않는다. GT perception/track의 특권성은 계속 명시한다.
미래는8step/약4s 고정. 현재 entity feature1034dim(visual1024+state/class10),
future 출력은visual1024+spatial6dim, ego 출력은8×(x,y,yaw)다. Target/mask는forward 인자가 아니다.

| 조건 / result 이름 | 미래 branch | Visual aux 계수 | Spatial aux 계수 | 비교 목적 |
|---|---|---:|---:|---|
| A `current_context_no_future_branch` | 없음 | 0 | 0 | 현재 정보만의 경로 예측 기준 |
| B `future_branch_planning_supervision_only` | 있음 | 0 | 0 | 추가 모듈/side-channel 효과 |
| C `future_branch_visual_auxiliary` | 있음 | 0.10 | 0 | 영상 미래 감독의 추가 효과 |
| D `future_branch_spatial_auxiliary` | 있음 | 0 | 0.10 | 객체 공간 미래 감독의 추가 효과 |
| E `future_branch_mixed_auxiliary` | 있음 | 0.05 | 0.05 | 혼합 감독의 균형된 탐색 |

B–E의 predictor/planner 폭·입력 크기·두 head는 같다. **Visual-only/spatial-only는 보조 감독의 종류**다.
C에서도 spatial head, D에서도 visual head를 유지하며 planning gradient를 받는다. 따라서 C/D를
‘영상만/상태만 planner에 제공’하는 실험으로 부르지 않는다. B는 미래 정보를 전혀 담지 않는다는 보장도 없다.
두 head 제거/동일 용량 대체는 target 후보를 좁힌 뒤 별도 설계한다.

A는 미래 memory 생성·주입을 생략하는 **활성 구조가 다른** 기준선이다. Parameter-matched 대조가 아니며,
B>A만으로 미래 감독의 효과를 주장하지 않는다. 같은 이름의 중복 run은 만들지 않는다.
현재 target 반복 감독과 현재 feature 직접 전달 adapter도 서로 다른 대조다.

## 2. Loss와 gradient 계약

Planning loss는 현재 ego frame의 waypoint GT를 쓴 imitation proxy다:

```text
Lplan = mean SmoothL1((predicted_xy - target_xy) / 10 meters)
        + 0.1 × mean(1 - cos(predicted_yaw - target_yaw))
Ltotal = Lplan + visual_weight × Lvisual + spatial_weight × Lspatial
```

공식 NAVSIM/Drive-JEPA loss 또는 미분 가능한 PDMS라고 부르지 않는다. XY 두 채널·8시점에 평균을 쓴다.
Angle wrap에 무관한 heading loss를 사용한다. 모든 조건에서 동일하다.

- Target 채널별 mean/population std는 **train split의 선택된 common-valid slot/time**에서만 계산한다.
  Std floor0.05로 whitening한다. Spatial 원표현은이미 x/y÷40m, vx/vy÷10m/s와sin/cos yaw다.
  Target 통계·basis는 B–E에 동일하게 적용하고 checkpoint에 고정한다. Dev 통계는 사용하지 않는다.
- Lvisual/Lspatial은 정규화된 target의 masked MSE다. 먼저 channel을 평균한 뒤 유효slot/time을 평균한다.
  Visual1024와spatial6의 차원 수 자체가loss 합계를1024/6배로 만들지 않게 한다.
- **C/D/E 모두 현재 선택 valid ∩ 미래 visual-valid ∩ 미래 spatial-valid인 공통 mask**를 사용한다.
  조사에서spatial 감독이 더 많이 남았으므로 첫 비교에서는 유효표본 수를 맞춘다.
  Mask는target loss에만 적용하고 현재 후보·planner memory·샘플 선정에는 적용하지 않는다.
  공통valid가0개인window도planning 학습에 남기며 auxiliary는0, 분모/skip count를 기록한다.
- C/D 단일 계수 합과 E 혼합 계수 합은0.1로 맞춘다. **동일 gradient 크기/동일 난도 보장은 아니다.**
  E–C/E–D는 단일 감독 계수도 달라지므로 순수한 한 종류 감독의 추가 효과가 아니라
  같은 총 계수에서의 trade-off 탐색이다. 최종 혼합 기여를 주장할 때는 matched-weight 대조가 필요하다.
  Raw/normalized loss, 유효 감독 수, weighted predictor gradient norm을 함께 기록한다.
  Train 전용 numerical profile에서 극단적 scale/NaN이 있으면 학습 전에 config를 새 버전으로 고친다.
  Dev에서 좋은 결과를 본 뒤 계수·normalization을 바꾸지 않는다.
- B–E: Lplan→predictor/planner. Aux→predictor only. Encoder frozen, fixed rule selector gradient 없음.
  기존 detached hard auxiliary 호출 계약을 유지한다. Zero-weight auxiliary는 호출/계산을 생략해 그 비용도 기록한다.
- A: Lplan→current planner only. Frozen encoder를 학습시키지 않으므로 이 탐색으로 **encoder 표현 학습이
  개선됐다**거나 full JEPA pretraining이 성공했다고 주장할 수 없다.

Primary common mask는front에서 미래까지 남은 객체에 한정된 조건부 질문이다. Native visual/state
mask로 평가하는 예측 오차와 supervision 수를 따로 보고하되, 이번 표에native-spatial 학습 run을 추가하지 않는다.

## 3. 데이터·학습량·평가 선택 고정

[Split manifest](../results/data_surveys/navsim_recording_split_manifest_20261001.json): train12recording/277window,
development4recording/96window. Nonoverlap·same-scene/native-log·cadence 기준은 조사 문서와 같다.
Navmini는 이미 본 개발용 자료이며 최종 독립 benchmark 평가가 아니다. Trainval/navtest/navhard는 이번에 쓰지 않는다.

첫 seed29, 조건당200optimizer update, 계획 batch8, AdamW(lr1e-4,weight_decay1e-4), global grad clip1.
Recording을 균등 선택한 뒤 해당 train recording의 window를 균등 선택하는 sampler를 쓰고
**초기 shared module weights, batch sequence, normalization, 데이터 cache를 A–E 사이에 고정**한다.
조건을 켜거나 끄면서 RNG consumption이 달라지지 않도록 shared initialization state를 별도로 저장한다.
Common planner weight까지 같은 상태에서 시작한다. A의 unused future module optimizer 등록은 생략한다.
Effective sample draw는 조건당1,600(재사용 포함)이고 총1,000update/8,000draw다. Epoch 수라고 부르지 않는다.
실측 profile로 batch를 줄여야 하면 모든 조건을 함께 바꾸고 update/draw 변경을 기록한 새 config를 공유한다.

검증 중간 값을 보고checkpoint를 고르지 않는다. **마지막200update checkpoint**를 동일하게 평가한다.
Shortlist만 seed11로 확인하며,5조건 전체의 불필요한 재실행/seed tuning은 하지 않는다.
이번 단계에서 두seed는 안정성 단서일 뿐 최종 다중seed 검증을 대체하지 않는다.

## 4. 결과 기록과 평가 범위

주 metric은 scene별window XY ADE를 먼저 평균한 **scene-macro ADE(m)**다. 동일window를 조건 사이에
paired 비교하며 FDE(m), circular heading error(rad), recording-macro ADE, 전체window 평균도 함께 보고한다.
4개dev recording의 결과를 각각 남기고 paired recording-cluster bootstrap1,000회(seed29)는 설명용 CI로만 쓴다.
Cluster가4개뿐이므로 유의성/일반화 주장으로 사용하지 않는다.

Performance를 보기 전에 고정할 상황 축:

- 현재front-valid count ≤4 / >4 및0개window.
- 현재ego speed <2 / 2–8 / ≥8m/s.
- 해석 전 command raw index0–3.
- 미래heading proxy left/low/right는 **사후 분석 전용**이고 online 입력/GT 의미label이 아니다.

각bucket의window와recording 수를 같이 보고한다. Window8개/recording2개 미만은 표본 부족 표시를 유지하고
패턴을 일반화하지 않는다. 현재manifest에도 right/merge coverage 부족이 있으므로 억지 결과를 만들지 않는다.

예측/의존도 항목:

- Visual normalized MSE/cosine, spatial 위치·속도 RMSE와heading error를 common/native mask별로 기록한다.
  Spatial RMSE는 whitening 및40m/10m/s scale을 되돌려 물리 단위로 표시한다.
- Current ROI/state를 미래에 반복하는 **persistence reference**와 비교한다. 학습 run이 아닌 예측 기준이다.
- 학습 후 예측 미래를0으로 만들기 / 다른scene의 미래로 교환하기 / detach를 구분한다.
  Zero/swap은의존도 검사이고 재학습 대조의 대체물이 아니다. Detach는 forward를 바꾸지 않는다.
- Nominal/active K, predictor slot/time, mask 분모, parameter 수, peak allocated/reserved/process VRAM,
  encoder/cache/train/eval 시간, 1step·sample·full current pipeline latency를 기록한다.
- Current ROI appearance가 충분한지/미래 branch가 현재정보의 우회 경로인지 아직 모른다.
  보조 감독의 단서가 생긴shortlist에서 current-target 반복 및 capacity-matched current-feature adapter를
  별도 단계로 비교한다. 이름만 바꿔 같은 실험을 두 번 만들지 않는다.

NC/DAC/TTC/EP/comfort/PDMS(or EPDMS)는 현재 작은 scaffold evaluator에 없다.
공식 evaluator와benchmark 버전을 붙이기 전에는 **자율주행 planning 개선**이라고 결론 내리지 않는다.
Scaffold는공식encoder만 재사용한 새 모델이며 official Drive-JEPA baseline 재현이라고 하지 않는다.

## 5. 계산 비용과 실행 한도 — 미실측과 한도를 구분

기존 source의 parameter breakdown은 S264,705/P1,327,536/D895,747이며 공식 encoder303,885,312는 freeze다.
이번 fixed rule에는 S를 optimizer에 넣지 않는다. B–E P+D는2,223,283parameter이며 A는 P 전체와
future 전용 planner module이 비활성이다. 실제 trainable/active parameter 수는 새 runner에서 출력해 확인한다.
A의 작은 active 구조를 B–E와 parameter-matched라 부르지 않는다.

Encoder feature는 **manifest373window만** 한번 만들어 조건 간 공유하는 작은 cache를 계획한다.
FP16 저장 시 current grid+current entity+future ROI의 dense 상한 추산은 약0.61GB(geometry/manifest 부가자료 제외),
실제 byte 수는 생성 시 측정하며2GiB hard cap을 넘으면 중단한다. Future full grid는 ROI를 만든 뒤 버린다.
9clip/window 기준 최대3,357encoder clip 처리라는 작업량 산술이지 실행 시간/VRAM 실측이 아니다.
Encoder caching 비용을 누락한 training 속도를 end-to-end 절감으로 주장하지 않는다.

먼저 train8window만 profile해서 load/encode/각 조건 forward-backward의 시간·메모리·loss scale을 측정한다.
이는 실행 규모 측정이지 추가 성능/연결 검증 성과가 아니다. GPU0을 기본으로 하되 **승인0·1만**, 점유를 다시 읽고
기존 타인 작업을 종료하지 않는다. Profile이 좋다고 전체 dataset cache를 생성하지 않는다.
조건별 학습30분/전체 학습2시간 cap은 보호 한도이며 예상 시간이 아니다. Encoder cache/eval 비용도 별도 보고한다.
현재 GPU-hour, batch8 VRAM, convergence 속도는**미측정**이다.

## 6. 진행·중단 기준과 다음 연구 gate

실행 전 engineering 우선순위 기준(통계적 연구 검증 기준 아님):

1. **즉시 중단/수정**: 미래 GT/mask가 forward에 들어감, train/dev log overlap, shared dataset 쓰기,
   NaN/Inf, normalization 누출, 예산 초과. Dev를 본 후 조용히 조건을 바꾸지 않는다.
2. **후보 우선 확인**: C/D/E 중 A와 B 모두 대비 scene-macro ADE가3% 이상 낮고 FDE가3% 넘게 악화되지 않으며
   4개 dev recording 중3개 이상에서 B보다 낮으면 A/B/해당 target을 seed11로 확인한다.
   3%는 작은 탐색의 우선순위 threshold이지 minimum publishable effect/유의성 보장이 아니다.
3. **애매한 경우**: C/D/E가 B를 못 넘거나 future zero/swap 변화가 미미하면 loss scale, target 품질,
   현재 side-channel, 학습량·표본 부족을 먼저 진단한다. 작은200update 실패만으로 H1/H2 전체를 기각하지 않는다.
   B>A만이면 추가 module의 효과로 보고 target 미래 감독을 기여로 주장하지 않는다.
4. **Target shortlist 이후**: 같은 target/branch/학습량에서 random·거리/TTC 규칙·ego-attention·ST 비교,
   고정 K2/4/8의 성능/active slot/비용과 상황별 marginal benefit을 측정한다. 강한 고정 규칙도 대조에 포함한다.
5. **Method 확대는 아직 보류**: 맥락별 추가 예산의 효과가 안정적이고 문헌상 delta가 남으면 동적 budget을 설계한다.
   Context 교차의 부재만으로 모든 적응형 모델 가능성을 부정하지 않는다. All-entity의 우위를 필수 gate로 쓰지 않는다.

Fixed K selection도 미래 conditioning도 직접 선행연구와 겹친다. 연구 기여는 [문헌 감사](egofsd_foredrive_evidence_audit.md)와
최종 공식 평가를 거쳐 확정한다. 이번 보고 후 먼저 fixed-rule runner/cache/metrics를 구현하고 작은 학습을 수행하는 순서이며,
**현재 커밋에는 학습 결과나 새로운 학습 프로세스가 없다.**
