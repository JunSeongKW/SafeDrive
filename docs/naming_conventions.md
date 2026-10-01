# 협업 가독성을 위한 명명 규칙

사용자 확정 규칙: 이후 연구자가 폴더·코드만 보고 대상과 역할을 알아볼 수 있어야 한다.
이는 디렉토리명뿐 아니라 코드의 모든 주요 이름과 결과 metadata에 적용한다.

## 기본 원칙

1. 프로젝트명은 연구 목적을 설명한다. `PlanningAwareFuturePrediction`은 작업명이지
   확정된 논문/방법명이나 특정 baseline 채택 선언이 아니다.
2. Python package / file / function / variable은 `lower_snake_case`, class는 `PascalCase`,
   상수는 의미가 드러나는 `UPPER_SNAKE_CASE`를 쓴다.
3. 역할 없는 `new`, `temp`, `final`, `test1`, `data`, `toy`를 주요 이름으로 쓰지 않는다.
4. 함수는 동작+대상을 표현한다. `generate_synthetic_entity_batch`,
   `predict_selected_entity_future_latents`, `decode_ego_plan`, `compute_training_losses`처럼 쓴다.
5. 변수는 출처·시간·의미를 구분한다. 현재 `entity_features`, 예측 `predicted_future_latents`,
   학습 전용 `future_latent_targets`를 같은 이름으로 뭉뚱그리지 않는다.
6. `entity_valid_mask`(현재 후보), `selected_entity_valid_mask`(선택 slot),
   `future_target_valid_mask`(미래 GT 관측 가능 여부)를 구분한다.
7. 시간 범위와 token 개수를 구분한다. `num_future_steps`와 합성 시간의
   `SYNTHETIC_PREDICTION_TIME_OFFSET`는 같은 개념이 아니다. 실데이터에는 `_seconds` 등 단위를 붙인다.
8. `detach_future_latents`처럼 boolean은 동작/조건을 드러낸다. 결과 key도 정책을 정확히 설명한다.
9. 설명용 수식의 H/C/u/N/K/T와 shape 기호는 허용한다. 코드 인터페이스에서는 역할 있는 이름을 쓴다.
10. `torch.optim`, `Tensor.grad`, `nn`, `F` 등 표준 외부 API는 유지한다. 공식 참고 코드나
    역사적 실험의 이름을 현재 취향대로 바꾸지 않는다.

## 현재 코드의 대응 예

| 이전 이름 | 현재 이름 |
|---|---|
| `Selection` | `EntitySelectionResult` |
| `ContextSelector` | `ContextConditionedEntityScorer` |
| `SequentialSTTopK` | `SequentialStraightThroughEntitySelection` |
| `SelectiveFutureGraph` | `SelectiveEntityFuturePredictionGraph` |
| `h`, `c`, `u` | `entity_features`, `scene_context`, `ego_intent` |
| `batch()` | `generate_synthetic_entity_batch()` |
| `toy()` | `run_synthetic_selection_learning()` |
| `gradient_norm()` | `compute_parameter_gradient_norm()` |
| `global_learned_no_intent` | `entity_only_selection_without_intent` |

Scorer는 score를 학습하고 selection operator는 score에서 K개를 고른다. 역할이 다르므로 이름도
구분한다. 미래 target은 loss에서만 사용하며 forward 입력과 명칭을 분리한다.

## 변경 시 확인할 것

- 상대 import, 문서 링크, 실행 명령, 환경 경로, 설정과 metadata key를 함께 갱신한다.
- Virtualenv는 절대 shebang 경로를 포함하므로 디렉토리만 옮겨 재사용하지 않고 새 경로에서 만든다.
- 체크포인트의 state-dict key를 변경하는 실제 모델 refactor에는 변환/호환 계획이 필요하다.
  현재는 외부 checkpoint를 쓰지 않는 synthetic fixture만 재명명했다.
- 수치·source hash·timestamp가 포함된 과거 실행 기록은 보존하고 새 실행의 결과를 추가한다.
- 명명 변경 전후 같은 설정/seed에서 동작과 결과가 유지되는지 검사한다.
