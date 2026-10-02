# Drive-JEPA 미래 경로의 구조별 추가 학습

## 실행 전 고정한 질문과 범위

사용자의 추가 학습 승인에 따라, 기존 planner를 보존하면서 미래 경로의 표현력과
ego 의도 반영을 강화하면 작은 개발 split에서 planning 오차를 줄일 수 있는지 확인한다.
본 연구의 선택·예산 가설을 입증하는 실험이 아니라, 다음 선택 실험을 위한 구조 비교다.

기준 HEAD는 `2185ce5`이며 이후 완료한 선택 비교의 미커밋 코드·결과를 먼저 보존한다.
기존 비교에서 learned K4는 200 update의 dev scene-macro XY ADE가 0.242582m로
원본 0.220644m보다 나빴다. 초반 개선 뒤 dev 오차가 증가했고 미래 MSE도 persistence보다
나빴다. 이를 데이터 부족·과적합 하나로 확정하지 않으며 구조와 최적화 모두 후보로 남긴다.

이번에는 기존 128 train/64 dev, recording 16/8, frozen cache를 그대로 쓴다.
Held-out/navtest/navhard를 읽거나 평가하지 않으며 기존 GT ROI pilot과 WA를 재개하지 않는다.

## 구조 비교

| 조건 | Predictor | Selector | Encoder 변화 | 보조 target |
|---|---|---|---|---|
| mlp_recipe_control | 기존 절대 feature MLP | 기존 additive ego conditioning | 없음 | 미래 |
| contextual_residual | 전체 현재 patch를 cross-attention으로 참조, 현재+변화량 | 기존 | 없음 | 미래 |
| ego_query_residual | 위와 같음 | ego query와 patch key 내적 | 없음 | 미래 |
| ego_query_residual_lora | 위와 같음 | ego query | 미래 branch에 복제한 마지막 4 block의 QKV LoRA | 미래 |
| lora_current_target_control | 위와 같음 | ego query | 위와 같음 | frozen 현재 feature를 4시점에 반복 |

미래 prediction query는 K4×4시점이다. Current context는 전체 512 patch를 유지하므로
encoder 전체의 연산 절감을 주장하지 않는다. Residual predictor는 128차원, 4-head,
2-layer cross-attention이며 출력 delta head를 0으로 초기화한다.
Ego 8차원은 기존에도 들어갔다. 새 구조는 입력 추가가 아니라 conditioning 방식 변경이다.

LoRA는 rank4/alpha4/dropout0, 마지막 block20–23의 QKV에만 적용한다.
원본 planner 입력과 원본 encoder 가중치는 고정한다. 별도 future branch encoder tail만 적응한다.
Teacher는 full planning checkpoint의 encoder로 고정하며 EMA를 도입하지 않는다.
이는 미래 latent 회귀를 쓰는 JEPA형 확장이지 공식 JEPA 사전학습 전체 재현이 아니다.

## Gradient와 관측 경계

Planning loss는 bridge, predictor, selector와 활성 LoRA에 전달한다.
보조 loss는 detached hard selection으로 predictor를 다시 호출하여 selector와 bridge에는
전달하지 않고 predictor와 LoRA만 갱신한다. Teacher feature와 미래 유효 mask는 loss에만 쓴다.
미래 GT는 online forward, candidate 선택, 현재 encoder prefix에 들어가지 않는다.

Bridge zero-init은 시작 시 원본 trajectory를 보존한다. Delta zero-init은 persistence를
구조적으로 구현한다. 초기 개선과 학습으로 얻은 미래 변화량 개선을 구분해야 한다.
LoRA B=0일 때 A의 초기 gradient가 0인 것은 예상된 현상이며 그래프 단절이 아니다.

## 공통 학습 절차와 상한

설정은 `configs/drive_jepa_selective_future/architecture_followup_v1.json`에 실행 전에 고정한다.
각 조건은 seed29/47/83, random hard K4의 aux-only warmup100회 후 learned joint200회다.
모든 조건에서 같은 batch 순서와 대응되는 공통 모듈의 초기화를 사용한다.
단계 경계에서 optimizer를 새로 만들며, 기존 실험의 정확한 resume이라고 부르지 않는다.
기존 MLP도 같은 새 절차로 재학습하므로 구조 변경과 새 학습 절차의 영향을 분리할 수 있다.

AdamW, batch8, weight decay0.01, gradient clip1, predictor/bridge LR1e-4,
selector/LoRA LR2e-5, 각 단계 cosine decay로 초기 LR의 0.1까지 낮춘다.
공식 length-normalized L1 planning loss와 raw latent MSE를 쓴다.
보조 가중치는 warmup1, joint0.01이며 sweep하지 않는다.
Joint0/50/100/200에서 평가하고 마지막 고정 checkpoint를 보고한다. Best dev 선택은 하지 않는다.

전체 15run/4500update, 총 wall-time90분, 조건당15분, allocated GPU12GiB 상한이다.
시작 free16GiB, 실행 reserve6GiB. 실제 여유를 확인한 GPU1 한 프로세스만 사용한다.
이 수치는 실행 상한이지 예상 사용량 실측값이 아니다.

LoRA에는 frozen 첫20 block 출력이 필요하므로 같은192개 현재 관측만 추가로 읽는다.
새 prefix cache는 약384MiB 예상/1GiB 상한, 기존 cache와 다른 경로다.
공식 전처리와 고정 encoder 출력을 재대조한다. FP32 batch/kernel 차이 허용값은
atol=rtol=1e-5로 사전 설정한다. 원본 planner 출력 보존은 같은 실행 모드에서 bitwise로 검사한다.
단순 CPU autograd-on/off 경로의 다른 attention kernel은 수치 동일성과 혼동하지 않는다.

## 평가와 해석

Scene-macro XY ADE와 recording별/command별 오차, seed별 차이, seed 평균·표준편차를 보고한다.
같은 selected patch의 미래 MSE와 persistence를 비교하고 horizon별 오차를 보존한다.
정책에 따라 target patch가 달라지므로 조건 간 raw MSE만으로 predictor 우열을 판정하지 않는다.
Current-target 조건은 현재 feature를 반복 감독하는 대조이며 direct feature adapter가 아니다.
Zero/swap은 입력 변화량과 trajectory 변화량을 함께 기록하는 의존도 진단이다.
이는 재학습 no-future 대조나 안전성 평가를 대신하지 않는다.

성공하더라도 반복 사용한 작은 dev split의 탐색 결과다. 대응 seed·recording 불확실성을
분리하며 공식 PDMS 향상 또는 학습된 최적 선택을 주장하지 않는다.
상한에서 종료하고 결과가 좋지 않아도 조건·데이터·loss를 추가 탐색하지 않는다.

## 실행과 상태

구현 및 CPU 계약 검사 후 실제 공식 모듈의 초기 출력·tail 동등성 검사를 통과해야 학습한다.
Raw 출력은 `outputs/drive_jepa_selective_future/architecture_followup_v1_20261002/` 아래 새로 생성한다.
학습 결과와 실행 비용은 완료 후 이 문서에 추가한다. 현재 이 절의 계획은 실측 결과가 아니다.
