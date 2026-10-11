# 공식 TOAD NAVSIM v1 scorer 진단

## 결론

**가중치 로딩·캐시 동등성·실제 경로 입력 gradient 검사는 통과했다. 평균 선택손실 기준도 통과했지만, 큰 gradient 보정은 실제 PDMS를 악화시켰다.**
후처리 연구를 위한 베이스로 사용할 기술적 조건은 확보됐다. 안전한 최적화나 판단 안정성 기반 적용 여부의 효과는 아직 입증되지 않았다. 추가 학습 및 전체 CEM 성능 비교는 시작하지 않았다.

## 평가 범위

- NAVSIM v1의 NAVTEST **64장면 / 38 recording**, 관측 명령 기준 좌회전 21·직진 22·우회전 21. 모델의 예측을 보기 전에 seed와 장면을 고정했다.
- 공개 가중치는 trainval + Simscale로 학습했으므로 NAVVAL은 제외했다. NAVTEST1024 개발 패널에 이미 노출됐으므로 독립 최종 테스트가 아니며, 명령별 균형 subset 점수다.
- 원 패널의 `scene_type`은 미래 GT heading으로 만든 라벨이다. 이번 subset 선정과 상황별 집계에는 그것을 사용하지 않고 원본 현재 driving command를 사용했다.
- 모든 가중치를 고정했다. optimizer **0 step**, encoder 재학습 없음. CEM 전체 반복은 실행하지 않았다.

## 실제 점수

선택손실은 같은 후보 집합에서 공식 PDMS가 가장 높은 경로의 점수와 scorer가 선택한 경로 점수의 차이다. Oracle 경로는 오프라인 진단에만 사용했다.

| 조건 | 후보 수 | 선택 PDMS | 후보 내 실제 최고 | 평균 선택손실 | NC 또는 DAC 위반 선택 |
|---|---:|---:|---:|---:|---:|
| 학습된 후보만 별도 채점 | 64 | 95.8391 | 99.8511 | 4.0120 | 1.56% |
| 공식 기존 후보 설정, CEM OFF | 128 | 96.1419 | 99.9470 | 3.8052 | 1.56% |
| 주변 후보 | 64 | 96.1052 | 99.5385 | 3.4333 | 1.56% |
| 동일 주변 후보 중 앞 16개 | 16 | 96.0235 | 98.5681 | 2.5446 | 1.56% |

공식 128후보는 학습된 64후보와 random bouquet 64후보다. 공식 eval 명령도 이 설정을 사용한다. 코드의 training-only 주석과 달리 실제 augmentation에 training 조건문은 없다.
Nearby64는 원본 fallback 1개 + bicycle roundtrip center 1개 + Gaussian 62개다. 표준편차는 공식 CEM 초기식(전체 proposal 제어 std × 0.5, acceleration/yaw-rate floor 0.1/0.025)을 사용한다. Nearby16은 같은 pool의 앞 16개를 별도 채점하므로 후보 개수와 candidate context가 모두 달라진다.

| 관측 명령 | 장면 수 | 공식 128후보 선택 PDMS | 주변 64후보 선택 PDMS |
|---|---:|---:|---:|
| left_turn | 21 | 94.7644 | 94.5961 |
| straight | 22 | 98.1017 | 98.1036 |
| right_turn | 21 | 95.4662 | 95.5208 |

NC/DAC/TTC/progress/comfort/DDC 하위지표, 장면별 결과와 recording bootstrap 구간은 results.json 및 scene_results.csv에 있다. DDC의 PDMS 가중치는 0이므로 최고 PDMS가 모든 주행 조건의 충족을 뜻하지 않는다.

## Gradient가 실제 성능을 개선하는가?

다른 63후보를 고정하고 center 후보의 neural score에 대해 제어 입력 gradient를 계산했다. 보정 반경은 초기 std로 정규화한 gradient 방향의 길이다. **Neural-only 목적 검사이며 공식 TOAD의 comfort·anchor를 포함한 전체 목적/CEM의 결과가 아니다.**

| 보정 반경 | Neural score 상승 장면 | 실제 PDMS 평균 변화 | 중앙값 | Recording bootstrap 95% 구간 | 1점 초과 하락 장면 |
|---|---:|---:|---:|---|---:|
| 0.05 | 95.3% | +1.5347 | +0.0000 | [-0.0026, +4.8939] | 0 / 64 |
| 0.1 | 87.5% | +1.5389 | +0.0000 | [-0.0084, +4.8988] | 0 / 64 |
| 0.25 | 70.3% | -6.2057 | +0.0000 | [-13.0045, -1.4334] | 5 / 64 |

작은 보정의 평균 증가 약 1.53점은 1점 초과 개선 장면 하나에 크게 좌우되며 중앙값은 0이고 신뢰구간은 0을 포함한다. 큰 보정에서 나타난 악화 때문에 scorer gradient를 그대로 믿고 움직이는 방식은 지지하지 않는다.
원본 경로를 제어로 역변환해 bicycle로 재생성한 center는 원본과 평균 ADE **0.4424m** 다르다. 위 변화는 이 center 대비이며 원본 planner 경로 대비 개선이 아니다.

## 후보 구성에 따른 점수 안정성

모든 context의 후보 수를 64개로 유지하고 공통 16개 경로 외의 48개 경로만 바꿨다. 공통 경로 중 최고 순위가 달라진 장면은 **25 / 64**다.
상대 점수 불안정성과 실제 선택손실의 Spearman 상관은 **-0.0622**였다. 현재 검사에서는 안정성 지표가 오류를 식별한다는 근거를 얻지 못했다. 안정성→위험도를 바로 해석하거나 이 값으로 보정 여부를 결정하면 안 된다.

## 기술적 검증

- 공개 asset SHA256 일치, 공식 nav1 commit **0110647142888dd56644565588510812bdafaa27**, **467 tensor strict load**. 모든 40,787,582 parameter 고정.
- 입력: 현재 4카메라 F0/B0/L0/R0, 카메라당 **1148×672**, PIL resize + ImageNet 정규화. 현재 시점 뒤 영상 입력 0. 이전/현재 ego를 저장하고 planner는 현재 11항목을 사용한다.
- 공식 `DrivoRFeatureBuilder.compute_features/_get_camera_feature`의 원본 AST 함수 본문을 재사용했다. 해당 환경에서 불필요한 map imports를 생략했고, 실제 이미지 3장면의 독립 전처리/ego 대조 오차는 0이었다.
- 캐시: image_backbone 출력 **64×256 FP32 register token** 및 ego **1×256**. 모든 token valid이며 register가 원본 픽셀과 직접 대응하지는 않는다. 기존 Drive-JEPA/소형 DrivoR cache와 혼용하지 않았다.
- 64장면 원본 forward와 cached scorer 점수 최대 오차 **0**, cached backbone 호출 **0**, 고정 weight digest 불변·weight gradient 누적 없음. 64장면의 경로 입력 gradient norm이 유한하며 0보다 크다.
- FP32 finite difference에는 작은 score 변화의 양자화 오차가 있었다. 독립 CPU FP64 실제 scorer 검사 **3장면 × 2방향 통과**, 최대 절대 오차 약 **6.4e-11**. 이는 성능 개선 근거가 아닌 미분 경로 검증이다.
- 공식 NAVSIM v1 평가와 경로별 progress 정규화를 사용했다. 일괄 계산을 변경 없는 reference+single-candidate evaluator와 대조한 최대 오차 **0**.

## 비용과 보존

완료된 GPU 단계: profile **10.18초** + 본 검사 **53.56초** = **63.74초**. 모델 로딩과 I/O를 포함한 wall time이다. 최초 입력 구성 오류는 forward 전에 발생했고 소요시간을 별도로 계측하지 못했으므로 제한 관리에서 120초를 예약했다.
본 검사 cache 추출 **19.00초**, cached scorer 진단 **5.95초**. GPU 0 전체 카드 관측 최대 **42.025GB**, 추가 PyTorch peak reserved **0.602GB**. GPU 1에 새 작업 없음.
공식 CPU 평가: **61.70초 / 12,672후보 / 2 worker**. 독립 CPU derivative 검사 8.38초. 기존 LPWM source/config/input **293파일 hash 불일치 0**, 현재 학습·대기열 보존.

## 사전 기준과 다음 실험 하나

평균 선택손실 ≤5점 및 원본 대비 NC/DAC 위반 선택률 악화 ≤1pp 기준은 official128/nearby64/nearby16에서 통과했다. 다만 선택손실의 bootstrap 상한은 5점을 넘는 조건이 있으며 이는 작은 subset의 point-estimate gate다. scorer 전반의 안전성을 검증했다고 해석하지 않는다.
**다음 우선순위: 같은 고정 checkpoint에서 CEM OFF와 공식 TOAD의 전체 comfort·anchor 포함 CEM을 같은 장면/공통 noise로 비교해 실제 순증가를 먼저 확인한다.** 성공하면 후보 구성 안정성이 보정 실패를 구분하는지 별도 평가한다. 현재 상관만으로 안정성 기반 selector를 설계하거나 sampler 학습을 시작하지 않는다.
기존 축소 실험과는 입력·학습 데이터/량·후보 수·평가 장면이 다르므로 이 점수 차이를 scorer만의 학습 효과로 해석하지 않는다. 전체 NAVTEST/EPDMS, 최적화 성능 향상·novelty는 미확인이다.

## 경로 및 재실행

- 결과: `results/official_toad_navsim_v1_scorer_calibration_20261011/results.json`, `comparison.csv`, `scene_results.csv`, `cpu_gradient_validation.json`, `execution_provenance.json`.
- Cache/공식 metric/phase 로그: `outputs/official_toad_navsim_v1_scorer_calibration_20261011/`.
- 시각화: `scorer_calibration_summary.png`.

```bash
cd /rhome/junseong/PlanningAwareFuturePrediction
bash scripts/run_official_toad_navsim_v1_scorer_calibration.sh
```

완료된 phase는 재측정하지 않는다. 가중치/원본 데이터/참고 repo는 Git에 포함하지 않으며 로컬 경로와 SHA는 provenance에 기록했다.

[TOAD 원문](https://arxiv.org/abs/2606.07170) · [공식 NAV1 코드 및 설정](https://github.com/valeoai/TOAD/tree/nav1) · [공개 가중치](https://github.com/valeoai/TOAD/releases/tag/Nav1)
