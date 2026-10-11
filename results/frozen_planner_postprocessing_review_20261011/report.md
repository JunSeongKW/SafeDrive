# 고정 planner의 추론 단계 경로 보정: 연구 방향 검토

조사일: 2026-10-11. 조사일 이전 공개 원문·공식 코드와 저장된 실제 실험을 확인했다. 이번 작업은 연구 방향 검토이며 새 학습·CEM 성능 비교·GPU 작업은 실행하지 않았다.

## 결론

**백본 재학습을 피하려는 자원 조건에 맞는 방향이다. 다만 고정 모델에 탐색·보정기를 붙이는 것만으로는 차별화가 부족하다.** 우선 연구 질문은 다음으로 좁힌다.

> 고정 scorer의 판단 안정성으로 경로 보정의 실제 이득과 악화를 구분하고, 제한된 추론 예산에서 보정 적용 여부를 결정할 수 있는가?

이는 기존의 상황별 정보 선택 문제에서 출발한 상황별 추론 자원 배분 질문이다. 미래 표현의 선택적 학습 자체를 검증하는 실험과는 범위가 다르다. 사용자의 이번 방향 제안을 기존 LPWM 학습 중단 요청으로 해석하지 않는다.

## 가까운 방법과 겹침

| 방법 | 고정/학습 범위 | 추론 중 동작 | 이번 방향과의 관계 |
|---|---|---|---|
| TOAD | 학습된 planner·scorer 고정, 추가 학습 없음 | 제어 공간 Gaussian CEM, bicycle rollout, scorer·anchor·comfort 평가 | 직접 재사용할 최적화 baseline |
| DriveVer | 시각 encoder 고정, confidence·보정 decoder 학습 | 보정 필요도에 따라 방향 보정 적용 | 별도 작은 모듈 학습을 허용할 때의 가까운 비교군 |

TOAD는 backbone feature를 한 번 추출하고 반복 채점에 재사용한다. 탐색 범위 제한, comfort 비용, 원래 경로와 최종 비교는 이미 방법에 포함된다. 이를 새 기여로 주장하지 않는다. [원문 §3.2·4.3·4.4](https://arxiv.org/html/2606.07170v1), [공식 코드](https://github.com/valeoai/TOAD).

DriveVer는 보정 방향을 GT 경로 방향으로 감독하고 보정 필요도를 BCE로 학습한다. **§III-C의 DrivoR/DINOv2 LoRA 설명과 §IV-A.3의 verifier 학습 중 visual encoder 고정을 구분한다.** 따라서 새 verifier 학습이 필요한 방법이지만, 이 단계에서 백본을 다시 학습한다고 설명하면 부정확하다. 공식 코드·checkpoint는 이번 검색에서 확인하지 못했다. [원문 §III-C·III-D·IV-A.3](https://arxiv.org/html/2607.00399v1).

단순 confidence threshold로 보정 적용 여부를 결정하는 방식은 DriveVer와 겹친다. scorer 주변에서만 탐색하는 방식은 TOAD와 겹친다. 아래 판단 안정성의 추가 가치가 입증되기 전에는 novelty 확보로 결론 내리지 않는다.

## 현재 코드와 실제 근거

- TOAD clone: `reference_repositories/TOAD/`, commit `cfa88e008080d0799b2859b5935c257048ec0adf`.
- 기존 CEM·bicycle·comfort: `reference_repositories/TOAD/drivor_cem_portable/navsim/agents/utils/cem_utils.py`. `cem_optimize`는 `torch.no_grad`의 샘플링 기반 탐색이다. 미분 가능한 scorer가 필수는 아니다.
- 임의 경로를 받는 고정 scorer: `src/planning_aware_future_prediction/models/residual_control_sampler.py:85`, `FrozenDrivorTrajectoryScorer`. 경로 입력 gradient를 보존한 진단용 wrapper이며 현재 기본 planner forward를 변경하지 않았다.
- 기존 실제 cache는 해당 소규모 DrivoR의 `16×256 FP32` scorer scene feature다. Drive-JEPA의 `128×1024` cache와 교환 가능한 표현이 아니다.
- DrivoR scorer의 candidate self-attention 때문에 같은 경로의 점수가 함께 들어가는 후보 집합에 의존한다. 후보 순서만 바꾸는 것은 이 의존성을 검사하는 적절한 개입이 아니다.

### 기존 공식 NAVVAL 64장면 검증 재확인

| 기존 후보 조건 | 선택 PDMS | 후보 내 실제 최고 PDMS | 선택 손실 |
|---|---:|---:|---:|
| Native64 | 76.8109 | 98.8412 | 22.0303 |
| Nearby64 | 78.8271 | 84.6712 | 5.8441 |
| Nearby16 | 80.1637 | 84.5054 | 4.3417 |

이는 `pass5.pt/3200update`로 수행한 기존 측정이다. 논문 저자가 공개한 완성 DrivoR/TOAD checkpoint 성능이 아니다. Nearby16은 같은 nearby pool의 앞16개를 다른 후보 context에서 재채점한 진단으로, 정식 M16 TOAD 실험이 아니다.

기존 검사에서 64/64장면의 실제 경로 gradient는 유한·양수였고 cache 단계 backbone 호출은 0회였다. Neural score 증가 방향의 실제 PDMS 변화는 세 반경 모두 recording-bootstrap 구간이 0을 포함했다. 71.9% 장면의 작은 감소가 관측됐지만 1점 초과 하락은 0개였으므로 큰 안전 악화로 과장하지 않는다. 동일 경로의 후보 context 변화에 따른 최대 logscore 차는 0.026462다. **이 안정성이 실제 보정 실패를 예측한다는 증거는 아직 없다.**

또한 경로→제어→bicycle 경로의 평균 ADE가 0.4112m였으므로, 보정 효과를 분석할 때 원본 경로와 BM roundtrip 경로를 별도로 비교해야 한다. 자세한 근거는 [기존 검증 보고서](../frozen_drivor_scorer_calibration_20261011/report.md)와 [실측 JSON](../frozen_drivor_scorer_calibration_20261011/results.json)에 있다.

작은 scorer·NC/DAC reader를 추가한 최근 pilot도 확실한 해결책을 보여주지 못했다. [직접 visual reader 결과](../direct_metric_visual_probe_v1/report.md)의 PDMS는 기존78.48/shared79.55/specific79.55였지만 NC 실패3→5/192로 사전 안전 기준을 통과하지 못했다. 시각 feature를 추가하는 것만으로 scorer 문제가 해결된다고 가정하지 않는다.

## 사용할 기반: 공개된 NAVSIM v1 scorer 확인이 먼저

2026-08-27 공개된 [TOAD Nav1 release](https://github.com/valeoai/TOAD/releases/tag/Nav1)의 실제 asset을 확인했다.

- `nav1_30epochs_with_134k_simscale_103ktrainval_random_bouquet.pth`
- 공개 SHA256: `257fa93d0145b6a67542f41eb1b02258649639ecee1b5a9a1f45e463594a26c8`
- 원 학습에 103k trainval와 134k Simscale, random bouquet가 사용됐다. 기존 소규모 checkpoint와 학습 조건이 다르다.
- [공식 nav1 실행 설정](https://github.com/valeoai/TOAD/blob/nav1/README.md)과 matching checkpoint를 함께 사용해야 한다. 기존 cache·모델 설정에 바로 대입하지 않는다.

이번에는 asset 공개를 확인했으며 다운로드·strict loading·공식 scorer 성능 재현은 아직 하지 않았다. 기존 검토에서 prescribed local checkpoint가 없었던 사실을 공개 weight 부재로 해석하지 않는다. 이 checkpoint를 쓰는 연구에서는 원본/TOAD/제안 방식 모두 같은 공개 weight로 비교한다.

## 최소 변경의 차별화 후보

**후보 context에 대한 판단 안정성을 이용한 선택적 최적화**를 우선 검토한다.

1. Backbone·base planner·scorer를 고정하고 현재·과거 관측만 사용한다.
2. 동일 경로를 동일 크기의 서로 다른 후보 context에 넣어 채점한다. 다른 context는 native/nearby 후보에서 생성하되 미래 GT는 입력하지 않는다.
3. 경로 간 상대 순위·NC/DAC/TTC 판단의 변동을 측정한다. 단순 높은 score를 신뢰도로 간주하지 않는다.
4. 이 변동이 실제 개선/악화를 구분할 때만 보정 허용·중단 신호로 사용한다. 작은 calibration module이 필요하면 cached feature와 후보/지표만으로 학습하고 모든 원 가중치는 고정한다.

이 신호는 서로 다른 후보 집합에 대한 기능적 안정성이다. 측정만으로 인식 불확실성이나 실제 안전 보장을 주장하지 않는다. 여러 context 채점은 비용을 증가시키므로 공식 TOAD보다 많은 scorer 호출을 사용해 얻은 개선과 구분해야 한다.

## 가장 작은 후속 실험 순서 — 아직 실행하지 않음

### 먼저 수행할 진단 하나

공식 Nav1 checkpoint를 정확히 로드해 개발64–128장면에서 기존 순위·off-proposal 채점 검사를 반복한다. 새로운 연구용 holdout은 그동안 반복 사용한 개발 패널과 recording 단위로 분리한다. Cache 전처리·sample ID·관측 시점 검증 후 원본과 cached 예측을 대조한다.

고정 후보 수에서 후보 context만 바꾸어 얻은 안정성이 실제 PDMS 순위 오류 또는 작은 보정의 성능 악화를 예측하는지 검사한다. 여기서 신호가 없으면 안정성 기반 적용 판별 연구는 보류한다.

### 진단 근거가 있을 때의 비교

| 비교군 | 동작 |
|---|---|
| A | 같은 공개 checkpoint의 원본 planner |
| B | 같은 checkpoint + 수정 없는 공식 TOAD |
| C | 같은 checkpoint + 판단 안정성에 따른 선택적 TOAD |

- 별도의 BM roundtrip-only 진단으로 경로 표현 변환 효과를 분리한다.
- 데이터·후보/노이즈 seed·제어 범위·comfort·scorer를 공통으로 유지한다. C의 진단 호출도 총 예산에 포함한다.
- 동일 최대 연산 예산과 동일 평균 연산 예산에서 각각 비교한다. 지연시간 중앙값·95백분위와 상황별 호출 수도 기록한다.
- PDMS, NC/DAC/TTC/EP/comfort/DDC, 실제 후보 내 최고점과 최종 선택의 차이, 악화율·개선율·fallback률을 함께 보고한다.
- PDMS evaluator와 미래 GT는 오프라인 평가/학습 라벨에만 사용한다. 추론 loop에서 공식 PDMS를 계산해 선택하지 않는다.
- 반복 개발 패널로 threshold를 선택한 뒤 그곳에서 개선을 확정하지 않는다. 최종 NAVTEST는 조건 확정 후 별도로 평가한다.

**진행 근거:** 공식 scorer가 off-proposal에서도 쓸 수 있고 안정성 신호가 보정 위험을 구분함. **보류 근거:** 신호가 무관하거나 비용을 맞추면 이득이 사라짐. **방향 전환 근거:** 안정적으로 점수를 높여도 실제 지표 악화가 지속되면 최적화 확대 대신 scorer의 일반화 문제를 먼저 다룸. 성공 기준의 수치는 후속 비교 실행 전에 별도 등록한다.

## 이번 작업의 보존 및 비용

기존 측정·공식 원문·코드 검토와 등록 파일 hash 확인만 수행했다. 현재 LPWM 보호 대상 source/config/input 293개 SHA 불일치0. 새 optimizer step0·GPU0초·실험 queue 추가0. 기존 LPWM 학습을 보존했다.

조사 링크·검색어·측정 출처·보존 검사 결과는 `sources_and_repository_evidence.json`에 남겼다.
