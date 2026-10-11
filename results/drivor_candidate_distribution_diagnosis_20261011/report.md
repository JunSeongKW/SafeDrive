# 원본 DrivoR scorer의 후보 분포별 성능 진단

## 질문과 범위

연구의 하위 질문: 상황에 따라 탐색 후보를 조절하기 전에, 어떤 후보 분포에서 frozen scorer의 예측과 선택을 신뢰할 수 있는가?
**모든 행의 채점기는 동일한 원본 DrivoR scorer다. TOAD generated는 후보를 만든 모델을 뜻하며 TOAD scorer를 사용한 행이 아니다.**

공식 원 DrivoR Scaling checkpoint(29epoch 완료/53911update), 동일 NAVTEST64/38recording, 기존의 16,384개 후보와 공식 NAVSIM v1 PDMS를 재사용했다. 새로운 학습·trajectory 생성·GPU·공식 평가기 호출은 0이다. 이전에 사용한 개발 패널이며 전체 NAVTEST/독립 최종 평가가 아니다.
선택 지표는 각 분포를 같은 64후보 예산으로 따로 채점했다. 점수 예측과 geometry 분석에는 256경로를 한 번에 채점한 결과도 따로 사용해 후보 간 self-attention context를 고정했다.

## 후보 분포의 정의

- DrivoR generated64: 원 DrivoR trajectory generator가 낸 64경로. 이번 모델의 자기 후보 분포이며 학습 데이터의 확률 밀도라고 가정하지 않는다.
- TOAD generated64: TOAD에서 배포한 DrivoR generator가 낸 64경로. 별도 공개 가중치에 의한 후보 분포다.
- Random bouquet64: 공식 generator의 constant22/smooth21/freeform21 제어를 bicycle rollout한 경로. 물리 범위로 제한했지만 객체·지도와의 안전을 보장하지 않는다.
- Nearby64: 이전 TOAD 선택 경로 1 + bicycle 역변환·재전개 중심 1 + TOAD 초기 Gaussian 주변 후보 62. 원 DrivoR 후보를 중심으로 생성한 분포가 아니므로 이 한계를 유지한다.

## 같은 후보 예산에서의 실제 성능

| 후보 분포 | 후보 수 | 예측 MAE ↓ | 순위 상관 ↑ | Oracle 최고 | Scorer 선택 | 선택손실 ↓ | NC<1 선택 | DAC<1 선택 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| DrivoR generated | 4096 | 17.405 | 0.437 | 99.875 | 93.609 | 6.266 | 1/64 | 0/64 |
| TOAD generated | 4096 | 17.901 | 0.415 | 99.851 | 93.093 | 6.758 | 1/64 | 0/64 |
| Random bouquet | 4096 | 15.236 | 0.584 | 93.843 | 71.407 | 22.436 | 5/64 | 6/64 |
| Nearby proposals | 4096 | 19.926 | 0.569 | 99.539 | 90.661 | 8.877 | 1/64 | 3/64 |

MAE는 모든 후보의 예측 점수 절대오차, 순위 상관은 같은 장면 안에서 후보 순서를 얼마나 맞추는지, 선택손실은 같은 후보 집합의 최고 실제 PDMS−scorer가 선택한 실제 PDMS다. Oracle은 후보 내 상한이며 항상 100 또는 인간 GT 경로가 아니다.

## 핵심 관측과 해석

1. **선택 기준으로는 generator 후보에서 상대적으로 안정적이고, 광범위한 random 후보에서 취약했다.** 자기64후보의 선택손실6.27점 대비 random64는22.44점이었다. Nearby64도8.88점이며 자기 후보보다 선택손실이 컸다. 상대적 비교이며 자기 후보를 정확하게 채점한다는 결론은 아니다.
2. **전체 MAE와 평균 순위 상관만으로 탐색 분포의 안전성을 판단하면 안 된다.** Random의 MAE15.24는 native17.40보다 작고 순위상관도 더 높지만, 실제 best 후보를 놓치는 손실과 안전 실패가 더 컸다. Random은 후보 평균 실제PDMS28.27, native는77.45로 실제 점수 구성이 크게 다르다.
3. **Random 안의 좋은 후보를 과소평가했다.** 같은256context에서 실제90–100점 random 후보232개의 MAE41.38점/평균 signed error−41.32점이었다. Native의 같은 구간2508개는 MAE14.53점이었다. 분포 전체가 어려운 것보다, 유용한 random 후보를 찾아내는 문제가 두드러졌다.
4. **Native에도 위험한 예외를 과대평가하는 문제가 남았다.** 실제NC/DAC미준수 native 후보655개의 실제평균은0점인데 환산예측은평균40.95점이었다. 정상적인 generator 경로와 비슷한 모양만으로 안전을 보장할 수 없다.
5. **자기 후보에서 멀어질수록 오류가 커지는 단순 패턴은 없었다.** RMS0.5–2m 후보 MAE22.65,10m이상5.75점이었다. 후자는 실제평균4.36점으로 대부분 나쁜 후보라 비교가 쉬울 수 있다. 이 설명은 구성 차이를 근거로 한 추론이며 학습 밀도/OOD 원인 측정이 아니다.
6. Constant/smooth/freeform의 MAE는15.66/14.62/15.17점으로, 제어 family만으로 큰 우열을 정할 근거는 부족하다. 서로의CI가 겹치며 causal training effect 비교가 아니다.

이전 보고의 DrivoR93.70은 두generator64개씩을합친128후보였고 이번93.61은자기64개만단독채점한값이다. 가중치나공식정답을바꾼것이아니라후보구성과scorer context가다른별도평가다.

**MAE가 낮아도 가장 좋은 후보를 잘 선택한다고 볼 수 없다.** 분포마다 실제 0점/고득점/위험 경로 비율이 달라, 낮은 점수의 쉬운 후보가 많으면 전체 MAE가 낮아질 수 있다. 선택손실·순위·조건부 오류를 함께 본다.

## 같은 256 후보 context에서 예측 정확도

| 분포 | 예측 MAE | 오차 ±5점 이내 비율 | 실제 NC/DAC 미준수 후보 비율 |
|---|---:|---:|---:|
| DrivoR generated | 18.478 | 43.16% | 15.99% |
| TOAD generated | 19.241 | 42.29% | 15.60% |
| Random bouquet | 15.155 | 54.64% | 60.30% |
| Nearby proposals | 20.160 | 41.63% | 36.52% |

## 무작위 제어 패턴별 차이

| 공식 생성 family | 후보 수 | 예측 MAE | 순위 상관 | 평균 signed error |
|---|---:|---:|---:|---:|
| Constant controls | 1408 | 15.658 | 0.582 | -7.177 |
| Smooth controls | 1344 | 14.616 | 0.568 | -3.752 |
| Freeform controls | 1344 | 15.167 | 0.610 | -5.067 |

상수 제어는 시간 전체 같은 가속도/yaw rate, smooth는 시작→끝 제어를 선형 변화, freeform은 시간별 noise를 더한 제어다. 원 공식 코드의 `family_mix` 연결 순서로 라벨을 붙였다. 서로 22/21/21개여서 family별 oracle 선택손실 우열을 비교하지 않았다.

## 기하학적 조건에 따른 예측오차

| 자기 후보와의 RMS 거리 | 후보 수 | 장면 수 | recording 수 | MAE | 평균 signed error |
|---|---:|---:|---:|---:|---:|
| [0.0, 0.5)m | 10074 | 64 | 38 | 18.830 | -2.746 |
| [0.5, 2.0)m | 3077 | 64 | 38 | 22.651 | -5.912 |
| [2.0, 5.0)m | 1547 | 64 | 38 | 14.260 | -3.174 |
| [5.0, 10.0)m | 1249 | 60 | 37 | 12.150 | -4.874 |
| [10.0, inf)m | 437 | 48 | 31 | 5.755 | +1.236 |

거리는 현재 장면의 DrivoR 생성64경로 중 가장 가까운 경로와 8개 waypoint의 평균 제곱 XY거리 제곱근이다. 현재의 geometry 유사성을 측정하며 실제 학습 밀도 또는 엄밀한 OOD 판정을 의미하지 않는다. 끝 heading·jerk proxy와 결합한 heatmap의 각 셀에 후보 수를 표시했다. 속도/가속도/jerk는 계획 waypoint로부터 역산한 proxy이며 실제 simulator의 comfort 지표와 다르다.

## 후보 context의 영향

| 분포 | 64→256 context로 같은 분포의 선택 변경 | 선택 PDMS 변화 | 동일 후보 예측값 최대 변화 |
|---|---:|---:|---:|
| DrivoR generated | 34/64 | +0.310 | 39.014 |
| TOAD generated | 34/64 | +1.027 | 43.739 |
| Random bouquet | 24/64 | +0.268 | 76.630 |
| Nearby proposals | 27/64 | +1.444 | 58.644 |

64와256은 candidate 수까지 다르므로 이 검사는 같은크기 context 안정성 검사가 아니다. Context 변화가 존재한다는 기능 진단이며 오류의 원인이라는 증명은 아니다.

## 그림과 산출물

- `scorer_performance_by_distribution.png`: 분포별 선택 점수/선택손실·MAE·순위·안전 실패·실제 점수 구성.
- `predicted_vs_actual_by_distribution.png`: 오른쪽 아래는 과대평가한 나쁜 경로, 왼쪽 위는 과소평가한 좋은 경로. 색 농도는 후보 수.
- `trajectory_candidate_distributions.png`: 같은 현재 영상에서 서로 다른 64후보의 실제 경로 분포. 보라색은 낮은 실제 PDMS, 노란색은 높은 실제 PDMS. 직진/좌회전 명령의 저장 순서 첫 장면을 사용해 오류가 큰 장면만 고르지 않았다.
- `trajectory_geometry_and_error_slices.png`: geometry·실제 점수 구간·random family별 오류. 실제 점수 구간은 retrospective GT 분석이므로 inference-time 구분기로 사용하지 않는다.
- `results.json`: 명령별 결과, score-band와 안전 조건부 오류, recording bootstrap CI 포함.
- `candidate_diagnostics.csv`: 16,384개 개별 후보의 실제·예측·기하학 proxy와 sample/recording ID.

## 확인한 것과 아직 모르는 것

Frozen weights 불변/gradient0/backbone 호출0, GPU↔CPU cached score replay를 검증했다. 이번 고정된 후보 분포에서 scorer 오류를 분해했으며, 더 넓은 NAVTEST와 새로운 random seeds에서도 같은 특성이 유지되는지는 아직 검증하지 않았다. Random/near는 TOAD에서 유래했으며 현재 분포 차이로 학습 과정의 원인을 확정하지 않는다. Bootstrap은 recording 표본 변동만 반영하며 학습 seed 변동은 아니다. 위험 label은 공식 NC<1 또는 DAC<1로 정의했으며 NC 부분감점도 포함한다.

## 다음 확인 하나, 미실행

이번에관측된특성은자기후보와가까운거리의안전경계/유용한random후보에서score를다르게보정해야할가능성을시사한다. 우선같은점수구간과같은후보예산에서작은위치·속도변화가NC/DAC판단을어떻게바꾸는지paired검사를권고한다. 현재진단만으로geometry threshold기반필터나새학습의개선을확정하지않으며자동학습/후속queue는시작하지않았다.

## 비용·재실행

CPU rescoring wall 52.56초, CPU 2thread/microbatch 2scene. 새GPU0초·학습0·공식평가0. 기존공식16,384후보 측정 재사용. 새CPU scorer scene-forward 192회.

```bash
cd /rhome/junseong/PlanningAwareFuturePrediction
bash scripts/run_drivor_candidate_distribution_diagnosis.sh
```

## 추가 검증 근거

CPU singleton과microbatch2의실제3scene×3context동등성검사통과: log score최대차 2.86e-06,확률최대차 6.56e-07. 학습/encoder호출/GPU0. Plot행간격·범례·좌표방향표시수정전후여섯측정함수AST및모든results.json수치가완전히같음을확인했다. 추가CPU검사 8.23초. `presentation_and_batch_validation.json`과`provenance.json`에원측정sourceSHA·수정후SHA·원source snapshot경로를보존했다.
