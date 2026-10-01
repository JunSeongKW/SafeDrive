# 선택적 미래 예측 계산 그래프 — 설계 초안 v0

상태: 2026-10-01 중간 조사. **계산 그래프 구현·gradient smoke test 미수행**.
코드 조사 근거와 자산은 `RESEARCH_STATUS.md`에 있다. 이 문서의 구조는 제안이며
baseline 및 target 구현을 최종 확정한 것이 아니다.

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
검출 query ↔ track-token 대응, ego frame 변환 및 새로 등장한 객체 처리는 아직 adapter가 없다.
GT association 실험은 privileged diagnostic이며 perception-based 완성 모델이라고 부르지 않는다.

## 4. hard top-K의 gradient 문제와 초기 후보

단순한 `topk(scores).indices` → `gather(H)`는 scores/selector에 선택 gradient를 주지 않는다.
기존 SafeDrive의 거리 선택을 이름만 selector로 바꾸는 것으로는 H2를 학습할 수 없다.

초기 구현 후보는 hard-forward / soft-backward straight-through(ST)다.

```text
Whard = valid 객체에서 중복 없이 K개를 고르는 one-hot assignment
Wsoft = 현재 scores로 만드는 미분 가능한 assignment surrogate
Wst   = Whard + Wsoft - stopgrad(Wsoft)
Qplan = Wst H
Fplan = Pψ(Qplan,C,u)
τ̂     = Dω(C,u,Fplan)
```

forward는 hard 선택, backward는 연속 완화의 기울기다. 이는 **정수 선택의 정확한 gradient가
아닌 편향된 추정**이다. Wsoft의 K-slot 생성, temperature, valid masking 및 중복 방지 처리는
아직 확정/구현하지 않았다. NaN·collapse 검사를 먼저 한다.
REINFORCE 등 대안은 첫 후보가 작동하지 않을 때 비용과 분산을 비교한다.

선택을 먼저 하고 predictor가 K query만 처리한다는 구조를 유지한다. 전체 N개 미래를 dense로
예측한 뒤 gate를 적용하는 우회 구현은 C 비교군으로 별도 표시한다.

## 5. loss별 직접 gradient 계약

초기 selector 학습 신호는 미분 가능한 ego imitation loss다. 공식 PDMS simulator score가
그대로 미분 가능하다고 가정하지 않는다. safety surrogate를 넣으면 별도로 정의/검증한다.

| loss | Sφ selector | Eθ current encoder | Pψ predictor | Dω planner | Eξ target encoder |
|---|---|---|---|---|---|
| Lplan | ST 경로로 받음 | freeze 여부에 따름 | 받음 | 받음 | 받지 않음 |
| LJEPA | **직접 받지 않음** | freeze 여부에 따름 | 받음 | 받지 않음 | 받지 않음 |

보조 loss 계산의 선택 query는 `Qaux=stopgrad(Wst)H`로 만든다. 같은 forward 값을 가지더라도
planning과 auxiliary의 그래프 경계를 분리해야 한다. target 또한 detached hard index로
gather한다. soft target 혼합의 편법으로 target을 쉽게 만드는 gradient를 허용하지 않는다.

첫 명세는 두 predictor 호출로 분리해 경로를 명백히 하는 안을 고려한다. 이를 쓰면 추가
학습 연산을 기록해야 하며 추론 sparse 구조와 학습 비용을 섞지 않는다. 이후 단일 호출로
최적화해도 loss별 gradient 계약이 같은지 시험한다. 공유 encoder가 바뀌면서 selector의
입력 분포가 변하는 **간접 영향**까지 차단한다는 뜻은 아니다.

JEPA loss만으로 대상을 고르면 예측하기 쉬운 객체를 택할 수 있다. 따라서 초기에는
selector parameter에 들어가는 auxiliary gradient를 의도적으로 막아 planning 신호를 분리한다.

## 6. 기존 코드와의 대조 및 baseline 판단 상태

| 경로 | 재사용할 수 있는 것 | 주 그래프에 없는 것 / 위험 |
|---|---|---|
| Drive-JEPA perception-free | pretrained image encoder, 현재/과거 입력, ego command, waypoint decoder | entity token/association, 선택적 미래 predictor, target adapter가 없음 |
| Drive-JEPA perception-based | BEV/proposal/scorer 기반 공식 구현 존재 | 내부 entity 표현과 future path 감사 미완료; 아직 선정 근거 불충분 |
| SafeDrive | instance query, joint motion/planning decoder, 미래 track alignment, 평가 자산 | hard 거리 선택; JEPA latent predictor 구조 아님; 기존 cache와 환경 미이전 |

판단: **Drive-JEPA는 재현/encoder 후보이지, B 경로가 이미 구현된 확정 baseline이 아니다.**
SafeDrive는 코드 참고와 기존 motivation 분석 자산으로 유지한다. 주 플랫폼을 SafeDrive로
바꾸는 결정을 이번 조사만으로 내리지 않는다. perception-based 감사와 adapter 비용을 보고
가장 작은 동일-backbone 실험 기반을 선택한다.

## 7. 다음 최소 검증 — 아직 실행하지 않음

첫 단계는 실제 자율주행 성능 실험이 아니라 synthetic CPU autograd 검사다.

1. Lplan만 backward했을 때 Sφ/Pψ/Dω에 유한한 gradient가 존재하는지 확인한다.
2. hard-index-only 대조와 선택 detach 대조에서 selector gradient가 끊기는지 확인한다.
3. LJEPA만 backward했을 때 Sφ에는 gradient가 없고 Pψ에는 있는지 확인한다.
4. 현재 입력·모델 상태를 고정하고 미래 target을 교체해도 선택/예측/ego 경로가 같은지 확인한다.
5. padding, N<K, N=0, 모든 미래 target invalid, permutation에 안전한지 확인한다.
6. 예측 미래 latent 제거·교환·detach 대조를 만들고 planner 경로가 바뀌는지 확인한다.

이 검사가 통과해도 H1/H2, 실제 미래 정보 사용, 효율, novelty가 입증된 것이 아니다.
실제 학습에서는 현재 정보가 predictor를 거쳐 전달되는 단순 side-channel과 시간적 미래
정보의 기여를 구분하는 대조도 필요하다.

이후 최소 성능 비교 후보: 동일 backbone / entity 표현 / predictor / planner / K / horizon에서
random K, 강한 거리·TTC 규칙 K, context-conditioned K. 규칙의 계산 가능성과 동작을 검증하고,
추가 학습 parameter 및 학습량을 통제한다. global learned / motion / no-planning-gradient는
순차 추가하고 전체 entity 예측은 다른 예산의 reference로만 표시한다.

공통 데이터 split, log-level holdout, paired/cluster bootstrap 및 집계+사전 정의한 상황별
지표를 사용한다. 미래 reference 기반 분석 축을 online selector 입력으로 넣지 않는다.
첫 K·T·차원·학습량은 실제 loader와 자원 실측 후 결정하며 GPU-hour를 미리 확정하지 않는다.

## 8. 진행 / 재검토 기준

- 그래프·label 경계 검사 실패 → 성능 학습 전에 수정.
- planner가 future latent를 무시함 → predictor/decoder 접속과 목적함수를 재검토.
- 동일 예산의 강한 비교군을 넘지 못함 → H2를 축소/재검토; 큰 학습으로 바로 확대하지 않음.
- 개선이 parameter·연산·추가 supervision으로 설명됨 → 선택 정책의 기여라고 주장하지 않음.
- 특정 상황 축·seed에만 효과 → 일반적 맥락 적응 주장 유보.
- selector가 한 종류/객체로 collapse → 원인과 예산·gradient를 점검하고 regularizer의 효과를 분리.
