# 선택적 미래 예측 계산 그래프 — v1 CPU 진단 구현

상태: 2026-10-01. 기준 협업 commit은 `95015df`이다. 사용자/ChatGPT 피드백을 반영해
K-slot과 detach 검사를 구체화한 **synthetic CPU fixture**다. 공식 NAVSIM agent 구현은 아니다.
추가로 frozen official video encoder+GT ROI visual/spatial pilot을 구현하고 실제 영상 batch를 검사했다.
최종 baseline과 target는 미확정이다. 최신 확장은 아래8절 및8.1절을 읽는다.
`1231767` 이후의 고정 K는 개발 기반이고 최종 기여가 아니다. 첫 비교는 선택기를 고정해 target 효과부터 분리한다.
코드: `src/planning_aware_future_prediction/models/selective_entity_future_prediction.py`.
검사: `tests/test_future_prediction_graph.py` / `scripts/validate_future_prediction_graph.py`.
실행 결과와 재현 명령은 `synthetic_validation_results.md`에 기록한다.

## 1. 무엇을 선택하는가

현재 보이는 차량·보행자 instance 중 **미래 latent를 예측할 대상**을 선택한다.
현재·과거 관측을 가리지 않는다. 도로·신호·ego 상태는 공통 맥락으로 유지한다.
정차 차량도 candidate다. 첫 버전에서는 K와 예측 horizon을 고정한다.

다음 세 연산을 구분한다.

| 연산 | 의미 | 첫 버전에서의 위치 |
|---|---|---|
| A: 관측 마스킹 | 현재·과거 정보를 가리고 복원 | 별도 ablation 후보; 주 경로 아님 |
| B: 미래 target 선택 | 현재 관측을 유지하고 일부 미래만 예측 | **주 설계 경로** |
| C: planner 입력 선택 | 모든 미래를 예측한 뒤 일부만 전달 | bottleneck 비교군; sparse prediction 절감 주장 불가 |

## 2. 제안하는 forward 및 학습 경계

```text
추론 시 사용 가능                                  학습 전용
과거·현재 관측 X                                   미래 관측 / GT association Y
      │                                                  │
      ▼                                                  ▼
현재 encoder Eθ → 공통 맥락 C, entity H          고정/EMA target encoder Eξ
                         │                               │
현재 ego 상태·command u ─┼→ selector Sφ                   Z [stop-gradient]
                         │          │                    │
                         │     fixed-K 선택 W            │
                         └──────────┼───────────────┐    │
                              선택 query Q=WH      │    │
                                   │               │    │
                          predictor Pψ(Q,C,u)      │    │
                                   │               │    │
                            예측 미래 latent F̂     │    │
                                   │               │    │
                                   └→ planner Dω(C,u,F̂) │
                                             │     │    │
                                          ego 경로 τ̂│    │
                                             │     │    │
                                   Lplan(τ̂, ego GT) │    │
                                             │     │    │
        backward: Dω → Pψ → W의 surrogate → Sφ│     │    │
                                                   │    │
                     JEPA 보조 경로: W를 detach해 같은 대상의
                     prediction/target loss 계산 → Eθ/Pψ만 갱신
                     Sφ로의 직접 auxiliary gradient는 차단
```

미래 label/관측은 selector·predictor·planner의 입력이 아니다. 학습과 추론의 forward 인터페이스는
같게 하고 loss 함수만 별도 targets를 받는다. Target encoder는 초기에는 고정하는 안을 우선한다.
EMA를 쓰면 latent collapse와 target drift를 추가 점검해야 한다.

## 3. tensor 계약 후보

| 이름 | shape | 출처 / 제한 |
|---|---|---|
| H | B×N×D | 현재 또는 현재까지의 entity feature; 안정적인 객체 식별 필요 |
| V | B×N | 현재 시점의 관측 valid mask; 미래 존재 여부로 만들지 않음 |
| C | B×M×D | 전체 현재 장면의 공통 맥락 |
| u | B×U | 현재 ego 상태 및 실제 제공되는 driving command |
| scores | B×N | Sφ(H,C,u); token index가 아닌 공유 entity scoring |
| W | B×K×N | hard-forward 선택; invalid padding slot은 0 |
| Q | B×K×D | W H; 선택 entity의 현재 feature/위치가 query identity를 제공 |
| F̂ | B×K×T×D | 선택 query만 미래 예측; T와 horizon 고정 |
| Z | B×N×T×D | 학습 전용 aligned future latent; target branch stop-gradient |
| Vfuture | B×N×T | 학습 전용 target 존재/관측 mask; 선택에는 사용하지 않음 |
| τ̂ | B×L×3 | ego (x,y,yaw) 경로 |

N이 달라질 때 실제 예측 수는 `min(K, 현재 valid 객체 수)`다. 부족한 slot을 중복 객체로
채우지 않는다. N=0, 전부 invalid, 사라지는 객체에서 NaN 없이 동작해야 한다.
GT track 대응과 current-ego 좌표 변환은 state/visual pilot에 구현됐다. 검출 query↔track 대응과
추론 시 새로 등장한 객체의 perception/association 처리는 아직 없다.
GT association 실험은 privileged diagnostic이며 perception-based 완성 모델이라고 부르지 않는다.

## 4. hard top-K의 gradient 문제와 초기 후보

단순한 `topk(scores).indices` → `gather(H)`는 scores/selector에 선택 gradient를 주지 않는다.
기존 SafeDrive의 거리 선택을 이름만 selector로 바꾸는 것으로는 H2를 학습할 수 없다.

v1은 hard-forward / soft-backward **순차 조건부 ST**를 구현한다.
하나의 softmax를 K번 복제하지 않는다. 슬롯 j마다 이미 선택된 hard index를 제거한다.

```text
Aj = 아직 선택되지 않은 현재 valid entity 집합
pj = softmax(scores / temperature, support=Aj)
ij = argmax(scores, support=Aj); score 동점은 stable entity ID 오름차순
hj = one_hot(ij), Aj가 비어 있으면 hj=pj=0, slot valid=False
wj = hj + (pj - stopgrad(pj))
Aj+1 = Aj \ {ij}   [hard exclusion, gradient 없음]
Wst = stack(w1, ..., wK)
Qplan = Wst H
Fplan = Pψ(Qplan,C,u)
τ̂ = Dω(C,u,Fplan)
```

- 각 슬롯은 서로 다른 객체를 hard-forward로 고른다. 부족하면 0 query / -1 index로 padding한다.
- temperature는 CPU 진단에서 **0.7**로 고정했다. 자율주행 실험의 최적값으로 주장하지 않는다.
- all-invalid 행은 softmax 전에 유한한 placeholder로 대체하고, 확률·query를 0으로 만든다.
  미래 target의 invalid 여부는 selection에 입력하지 않는다.
- entity ID는 현재 객체의 안정적인 식별자다. selector의 입력에는 넣지 않고 동점 처리에만 쓴다.
  valid ID 중복은 거부한다. 실제 detector의 stable ID 확보는 adapter 단계의 미결이다.
- 공유 entity scoring에 index/slot embedding이 없고, ID 기준 재정렬하므로 entity 순서 변경에
  대해 선택 identity와 gradient가 일관돼야 한다. fixture planner는 slot pooling으로 순서 불변이다.
- 전체 valid 객체 수가 K 이하이면 선택할 부분집합이 없으므로 selector surrogate를 차단하는
  경계도 검사한다. 전체 entity reference는 별도 selector 없는 경로로 평가한다.

이는 **정수 선택의 정확한 gradient가 아닌 편향된 추정**이다. 다음 슬롯의 softmax가 이전
hard 선택에 조건부이며, exclusion의 변화 자체는 미분하지 않는다. 완전한 differentiable
without-replacement 분포나 이론적으로 불편한 policy-gradient estimator라고 부르지 않는다.
작은 학습 과제에서 실제 선택이 개선되는지를 gradient 존재 검사와 별도로 확인한다.

선택을 먼저 하고 predictor가 K query만 처리하는 구조다. 다만 현재 fixture는 padded K slot도
계산하고, auxiliary용 predictor를 재호출한다. sparse wall-clock/FLOPs 절감은 측정하지 않았다.
Encoder가 전체 장면을 처리하는 비용과 target encoder의 학습 비용도 별도로 기록해야 한다.
전체 N개 미래를 dense로 예측한 뒤 gate를 적용하면 C 비교군으로 표시한다.

## 5. loss별 직접 gradient 계약

초기 selector 학습 신호는 미분 가능한 ego imitation loss다. 공식 PDMS simulator score가
그대로 미분 가능하다고 가정하지 않는다. safety surrogate를 넣으면 별도로 정의/검증한다.

| loss | Sφ selector | Eθ current encoder | Pψ predictor | Dω planner | Eξ target encoder |
|---|---|---|---|---|---|
| Lplan | ST 경로로 받음 | freeze 여부에 따름 | 받음 | 받음 | 받지 않음 |
| LJEPA | **직접 받지 않음** | freeze 여부에 따름 | 받음 | 받지 않음 | 받지 않음 |

이 표의 encoder 항목은 실제 모델의 목표 계약이다. v1 CPU fixture에는 현재/target encoder가
없고 H/C/Z를 외부 tensor로 제공한다. Sφ/Pψ/Dω 직접 경로만 구현·측정했다.

보조 loss 계산의 선택 query는 `Qaux=stopgrad(Wst)H`로 만든다. 같은 forward 값을 가지더라도
planning과 auxiliary의 그래프 경계를 분리해야 한다. target 또한 detached hard index로
gather한다. soft target 혼합의 편법으로 target을 쉽게 만드는 gradient를 허용하지 않는다.

v1 fixture는 두 predictor 호출로 분리해 경로를 명백히 한다. 이를 쓰면 추가
학습 연산을 기록해야 하며 추론 sparse 구조와 학습 비용을 섞지 않는다. 이후 단일 호출로
최적화해도 loss별 gradient 계약이 같은지 시험한다. 공유 encoder가 바뀌면서 selector의
입력 분포가 변하는 **간접 영향**까지 차단한다는 뜻은 아니다.

JEPA loss만으로 대상을 고르면 예측하기 쉬운 객체를 택할 수 있다. 따라서 **이 초기 실험 설계에서는**
selector parameter에 들어가는 auxiliary gradient를 의도적으로 막아 planning 신호를 분리한다.
이는 모든 JEPA 방법에 필요한 보편 원칙이 아니다. 후속 ablation에서 허용 여부를 비교할 수 있다.

## 6. 기존 코드와의 대조 및 baseline 판단 상태

| 경로 | 재사용할 수 있는 것 | 주 그래프에 없는 것 / 위험 |
|---|---|---|
| Drive-JEPA perception-free | pretrained image encoder, 현재/과거 입력, ego command, waypoint decoder | entity token/association, 선택적 미래 predictor, target adapter가 없음 |
| Drive-JEPA perception-based | BEV/proposal/scorer 기반 공식 구현 존재; 내부 감사 완료 | 미래 state 출력은 train-only auxiliary, planner 입력 아님; 신규 entity adapter 필요 |
| SafeDrive | instance query, joint motion/planning decoder, 미래 track alignment, 평가 자산 | hard 거리 선택; JEPA latent predictor 구조 아님; 기존 cache와 환경 미이전 |

판단: **Drive-JEPA는 재현/encoder 후보이지, B 경로가 이미 구현된 확정 baseline이 아니다.**
SafeDrive는 코드 참고와 기존 motivation 분석 자산으로 유지한다. 주 플랫폼을 SafeDrive로
바꾸는 결정을 이번 조사만으로 내리지 않는다. perception-based 감사와 adapter 비용을 보고
가장 작은 동일-backbone 실험 기반을 선택한다.

## 7. v1 검증 단계와 대조의 정확한 목적

첫 단계는 실제 자율주행 성능 실험이 아니라 synthetic CPU autograd 검사다.
fixture의 공통 맥락 C는 pooled B×C vector, ego 출력은 B×3이다. 원래 BEV/경로 tensor 계약을
실제 sensor adapter에 구현한 것은 아니다.

1. Lplan만 backward했을 때 Sφ/Pψ/Dω에 유한한 gradient가 존재하는지 확인한다.
2. hard-index-only 대조와 선택 detach 대조에서 selector gradient가 끊기는지 확인한다.
3. LJEPA만 backward했을 때 Sφ에는 gradient가 없고 Pψ에는 있는지 확인한다.
4. 현재 입력·모델 상태를 고정하고 미래 target을 교체해도 선택/예측/ego 경로가 같은지 확인한다.
5. padding, N<K, N=0, 모든 미래 target invalid, permutation에 안전한지 확인한다.
6. **Detach**: 동일 가중치·결정론적 입력에서 forward 값이 정확히 유지되며,
   Lplan→predictor/selector backward만 끊기고 planner gradient는 남는지 검사한다.
7. **제거/교환/교란**: latent를 0으로 만들거나 다른 샘플의 예측과 교환해 출력 민감도를 검사한다.
   permutation-invariant planner에서 같은 K 슬롯을 단순 재정렬하는 것은 올바른 교환 검사가 아니다.
8. **실제 학습 검사**: 다음 합성 과제에서 planning loss만으로 중요한 entity 선택이 개선되는지 측정한다.

합성 과제의 N=6, K=2, 합성 시간 범위=2다. 실제 데이터의 초/프레임 단위가 아니다.
각 entity에는 semantic key와 현재 위치 x·속도 v가 있고,
intent가 서로 다른 두 key를 요청한다. entity 순서는 샘플마다 무작위로 섞는다.
고정 analytic predictor는 선택 query에서 `x+2v`를 계산하고, 고정 planner는 그 둘의 합/√2를
출력한다. label은 요청된 두 entity의 미래값으로 생성하되, selector에는 선택 정답 loss를 주지 않는다.
학습되는 것은 selector뿐이며, 미래 예측/decoder의 동시 학습 난이도는 이 과제에 포함되지 않는다.

3개 seed, 각 1,000 step, batch 128, 별도 생성 seed의 holdout 4,096개를 사용한다.
온전한 command, shuffled command, entity-only no-intent learned, random, motion, fixed-semantic을
비교한다. `input_exact_match`는 현재 key·intent 내적만 쓰는 규칙이고, `relevance_oracle`은
정답 relevance reference다. 이 과제에서는 동일하므로 과거 hindsight/배포 불가 해석을 정정한다.
실행 전 코드에 둔 진단 기준은 recall≥0.8 및 random 대비 MSE≤25%다.
해당 기준은 자율주행 H1/H2의 성공 기준이 아니다.

이 검사가 통과해도 H1/H2, 실제 미래 정보 사용, 효율, novelty가 입증된 것이 아니다.
실제 학습에서는 현재 정보가 predictor를 거쳐 전달되는 단순 side-channel과 시간적 미래
정보의 기여를 구분하는 대조도 필요하다.
본 실험표에 **동일 용량의 현재 entity feature 전달 adapter→planner** 대조를 추가한다.
K·차원·시간축 복제·parameter·학습 step을 맞추고, 미래 보조 target의 유무를 분리해
prediction이 단순 현재 정보 side-channel인지 확인한다. 이 대조의 실제 성능 평가는 미수행이다.

이후 최소 성능 비교 후보: 동일 backbone / entity 표현 / predictor / planner / K / horizon에서
random K, 강한 거리·TTC 규칙 K, context-conditioned K. 규칙의 계산 가능성과 동작을 검증하고,
추가 학습 parameter 및 학습량을 통제한다. global learned / motion / no-planning-gradient는
순차 추가하고 전체 entity 예측은 다른 예산의 reference로만 표시한다.

공통 데이터 split, log-level holdout, paired/cluster bootstrap 및 집계+사전 정의한 상황별
지표를 사용한다. 미래 reference 기반 분석 축을 online selector 입력으로 넣지 않는다.

## 8. 실제 영상 pilot 확장 — 35fbdcf 이후 구현

상세 실행·shape·provenance·gradient는 [영상 pilot 검증](visual_future_prediction_pilot_validation.md).
공식 encoder만 재사용한 신규 scaffold이며 SafeDrive 재개나 공식 Drive-JEPA 전체 재현이 아니다.

```text
현재/직전 front 영상 ── frozen official encoder ─┬─ 현재 image grid ──────────────┐
현재 GT box/track ── projection/ROIAlign ─────────┴─ 현재 ROI + GT state ─┐       │
현재 command/ego 상태 ───────────────────────────── selector(ST, K4) ───┤       │
                                             selected current entity  │       │
                                                      predictor ──────┴──┐    │
                                                   predicted visual+state ─ planner ─ ego8 ─ Lplan
미래 영상+미래 동일GT track/box ─ frozen teacher/ROIAlign ─ Zvisual ─┐
미래 GT state ─ global→현재 ego ─ Zspatial ────────────────────────┴─ Laux
```

Planner는 현재 image/모든 현재 valid entity/ego와 **예측된** 미래만 받는다.
미래 GT/ROI/valid는 target loss builder 전용이며 forward signature에 없다.
Visual target은 `[B,N,8,1024]`; metric target은 `[B,N,8,6]`이다. 현재 ego 기준 위치·속도·heading을
명시적으로 예측하여 ROI 위치 제거 문제를 회피하지만, **별도 GT-state 감독**이므로 순수 JEPA가 아니다.
Target encoder는 현재와 같은 frozen pretrained module이며 EMA 학습은 아직 하지 않는다.

Lplan은 S/P/D에 전달된다. Visual/spatial Laux는 detached hard selection으로 P를 다시 호출하므로
P만 직접 학습한다. K4·8steps·front-only·GT association은 pilot 범위다.
최종 perception·multiview·target choice, 공동 학습 안정성/효율은 미확인이다.

다음 대조에서 **미래 감독 없음(동일 branch)**, **현재 target(동일 capacity)**,
**미래 branch 없음(현재 입력만, S/P 호출 생략)**을 구분한다.
두 보조 감독을 동시에 바꾸면 개별 효과를 분리하기 어렵다. 아래8.1절의 첫 target 비교는
공통mask에서 C visual/D spatial/E mixed를 나눠 탐색하며 필요 시 shortlist의 matched-weight 대조를 추가한다.
모든 미래 예측은 다른 예산의 참고이며 우위를 필수 gate로 삼지 않는다.
해당 대조의 학습 성능 비교는 아직 하지 않았다.
Pilot은 K4·future8·visual1024dim으로 검사했다. 본 학습의 표본/학습량/예산은 별도로 고정하고
GPU-hour를 단일 batch 실행 시간으로 추정 확정하지 않는다.

### 8.1. 1231767 이후 target 효과부터 보는 계획 — 아직 학습 미실행

[연구 결정](research_question_and_target_decision.md), [target 비교 계획](minimal_target_ablation_plan.md),
[여러-log 실제 조사](navsim_visual_target_coverage.md)가 최신 다음 실행 기준이다.

```text
현재 image/entity/ego ───────────────────────────────────────── planner ─ Lplan
현재 front-valid entity ─ fixed nearest-current-distance K4 ─ predictor ─┘ (B–E)
미래 ROI/spatial GT ─ train-only common mask + normalization ─ Laux (C/D/E)
```

A는predictor/future memory를생략, B는같은branch에planning만, C/D/E는보조감독만바꾼다.
B–E에서두head와planner 입력은동일하다. Visual-only/spatial-only는감독구분이며정보채널제거가아니다.
Encoder freeze, selector 비학습, planning→P/D, aux→P only. Future-valid는선택/forward에쓰지않는다.
Train12/dev4recording, train277/dev96window. 공통target-mask/학습량/초기값을맞춘다.
Native mask 잔존율을별도보고하고,current-target 반복과current-feature 직접adapter는혼동하지않는다.

EgoFSD의선택/joint planning과ForeDrive의미래latent conditioning에직접중복한다.
현ST나fixedK 자체를novelty로전제하지않는다. Target을좁힌뒤동일예산선택비교와고정K별
맥락적marginal benefit부터측정하며동적K/horizon은구현하지않는다.
Common-mask model학습/유효target의planning효과/최종baseline은아직미확인이다.

## 9. 진행 / 재검토 기준

- 그래프·label 경계 검사 실패 → 성능 학습 전에 수정.
- planner가 future latent를 무시함 → predictor/decoder 접속과 목적함수를 재검토.
- 동일 예산의 강한 비교군을 넘지 못함 → H2를 축소/재검토; 큰 학습으로 바로 확대하지 않음.
- 개선이 parameter·연산·추가 supervision으로 설명됨 → 선택 정책의 기여라고 주장하지 않음.
- 특정 상황 축·seed에만 효과 → 일반적 맥락 적응 주장 유보.
- selector가 한 종류/객체로 collapse → 원인과 예산·gradient를 점검하고 regularizer의 효과를 분리.

## 10. 구현 동작의 1차 문서 근거

Detach는 graph에서 분리하지만 storage를 공유하므로 in-place 조작하지 않는다.
[PyTorch 2.8 detach 문서](https://docs.pytorch.org/docs/2.8/generated/torch.Tensor.detach.html).
Stable argsort의 동점 순서 보존을 사용하되 먼저 entity ID를 정렬한다.
[PyTorch 2.8 argsort 문서](https://docs.pytorch.org/docs/2.8/generated/torch.argsort.html).


## 2026-10-03 구현 보충: 위치 연결·희소 관계·관측 이력 마스킹

고정된공식Drive-JEPA planner위region K8/4시점extension의실제계산그래프는[세논문착안구현명세](drive_jepa_region_research.md#구현검증)에있다.
현재memory전체는유지하고예측미래residual을선택위치에scatter한다. SPARTAN착안hard관계는bridge한층에적용하며predictorcontext는dense다.
C-JEPA착안마스킹은보조forward에서만관측최신region을이전anchor로대체하고현재/미래latent를복원한다. 정답gradient는predictor만받는다.
IA착안선택은현재까지4frame의움직임점수top8이며futureGT/validmask/충돌label을읽지않는다.
27run/개발PDM평가완료:추가planning이득과미래예측필요성은확인하지못했다. 객체instance계산그래프의완성/일반화검증으로해석하지않는다.
