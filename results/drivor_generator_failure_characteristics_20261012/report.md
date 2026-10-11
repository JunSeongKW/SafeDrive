# DrivoR generator 후보 집합·장면별 scorer 실패 특성

## 결론

현재 확인된 가장 큰 취약점은 회전 장면의 점수 예측오차와, 위험한 후보의 안전 점수를 과대평가하는 선택 오류다. 후보의 전체 퍼짐 정도·속도·주변 객체 수만으로 선택 실패를 예측하는 일관된 관계는 확인되지 않았다.

연구의 하위 질문: 장면·후보의 어떤 정보가 안전과 진행 판단에 필요하며, 기존 표현/채점이 그 정보를 놓치는가?

## 측정 조건

- 이미 사용한 NAVSIM v1 NAVTEST 개발용 부분집합 64장면 / 38 recording. 전체 NAVTEST나 독립 최종평가가 아니다.
- 원본 공개 DrivoR Scaling 체크포인트의 generator64와 같은 모델의 scorer가 주 분석이다. 실제 평가 대상은 4초/8 waypoint의 동일한 4,096개 후보다.
- 추가 확인은 TOAD 배포 모델이 생성한 후보64를 원본 DrivoR scorer로 평가한다. TOAD의 CEM 출력이나 별도 scorer 방법 비교가 아니다.
- scorer의 후보 간 self-attention 문맥을 유지하기 위해 각 generator의 완전한 64후보 문맥 점수를 사용했다. 이전 공통128/256 문맥 결과와 직접 섞지 않았다.
- 후보 좌표/공식 PDMS/관측 feature/점수 캐시 재사용. GPU·학습·새 encoder/scorer/oracle 실행 모두 0.
- 현재 GT 객체 라벨은 사후 장면 분석용이며 모델 입력으로 추가하지 않았다. vehicle / pedestrian / bicycle만 actor 집계에 포함한다.
- 점수 추정은 100*exp(log_score)/12, 실제 점수는 기존 공식 NAVSIM v1 PDMS. 예상 NC/DAC/TTC sigmoid 출력은 정답 안전 확률로 보장되지 않는다.

## 후보 생성과 선택 구분

| 후보 출처 / 같은 원본 scorer | 후보 중 oracle 최고 | 최종 선택 | 선택손실 | 후보 점수 MAE | 선택손실 >10 장면 |
|---|---:|---:|---:|---:|---:|
| drivor_generated64 | 99.87 | 93.61 | 6.27 | 17.40 | 8/64 |
| toad_generated64 | 99.85 | 93.09 | 6.76 | 17.90 | 9/64 |

선택손실은 후보 중 실제 최고 PDMS − scorer 선택 경로 실제 PDMS이다. MAE가 커도 순서만 맞으면 선택은 성공할 수 있다. Oracle는 사람 GT 경로가 아니라 이 후보 집합의 실제 최고다.

원본 후보에서 선택 NC 실패 1/64, DAC 실패 0/64, TTC 실패 5/64. TTC 실패를 포함한 장면이 전체 선택손실의 67.5%를 차지한다. 최악 1장면이 24.9%, 최악 3장면이 47.0%를 차지해 소수 사례의 영향이 크다.

## 현재 주행 지시·주변 객체별 분해

| 관측으로 정의한 구간 | 장면 수 | 후보 점수 MAE | 선택손실 | 선택손실 >10 |
|---|---:|---:|---:|---:|
| command_name: left_turn | 21 | 22.82 | 11.45 | 4 |
| command_name: right_turn | 21 | 20.10 | 5.10 | 3 |
| command_name: straight | 22 | 9.67 | 2.43 | 1 |
| pairwise_rms_distance_median_m: [0.0, 2.0) | 30 | 16.29 | 4.50 | 4 |
| pairwise_rms_distance_median_m: [2.0, 5.0) | 34 | 18.39 | 7.82 | 4 |
| median_endpoint_heading_abs_deg: [0.0, 10.0) | 25 | 11.18 | 6.74 | 3 |
| median_endpoint_heading_abs_deg: [10.0, 30.0) | 16 | 21.48 | 3.95 | 2 |
| median_endpoint_heading_abs_deg: [30.0, 180.0001) | 23 | 21.33 | 7.36 | 3 |
| ego_speed_m_per_s: [0.0, 2.0) | 16 | 15.77 | 7.57 | 1 |
| ego_speed_m_per_s: [2.0, 6.0) | 25 | 21.50 | 7.98 | 5 |
| ego_speed_m_per_s: [6.0, 1000000.0) | 23 | 14.10 | 3.50 | 2 |
| vehicle_count_within30m: [0, 1) | 7 | 15.32 | 0.58 | 0 |
| vehicle_count_within30m: [1, 3) | 14 | 19.12 | 4.64 | 2 |
| vehicle_count_within30m: [3, 1000000) | 43 | 17.19 | 7.72 | 6 |
| vulnerable_actor_present_within30m: False | 29 | 19.15 | 7.37 | 3 |
| vulnerable_actor_present_within30m: True | 35 | 15.95 | 5.36 | 5 |

회전 명령에서 MAE가 더 크다. 30m 내 차량 3대 이상 구간에서 선택손실 평균이 더 높지만, 객체 수와 선택손실의 연속 상관 95% recording bootstrap 구간은 0을 포함한다. 차량 많은 장면이 항상 실패하거나 보행자 존재가 악화 원인이라는 결론은 낼 수 없다.

## 후보 집합의 형태

후보 분포는 8 waypoint XY의 후보 간 RMS 거리 중앙값, 끝점 횡방향 표준편차, 끝 heading의 원형 표준편차, 속도 분산 등으로 계산했다. 속도/jerk는 waypoint 간격에서 계산한 proxy이며 실제 시뮬레이터 comfort나 제어 출력이 아니다.

| 분포 특징 → 후보 MAE | Spearman | recording bootstrap 95% CI | 명령·현재 속도 고려 후 partial 상관 | partial 95% CI |
|---|---:|---|---:|---|
| endpoint_lateral_standard_deviation_m | 0.346 | [0.09288839197573256, 0.550173100617733] | 0.117 | [-0.1500760888575689, 0.3738318683042441] |
| endpoint_heading_circular_standard_deviation_deg | 0.335 | [0.11653552197410803, 0.5229891930723178] | 0.093 | [-0.2249774469052984, 0.39708589747670775] |
| median_endpoint_heading_abs_deg | 0.365 | [0.14934804528154424, 0.5421430615937958] | 0.156 | [-0.07406995350283077, 0.3873000957708276] |

횡방향/heading spread 및 회전 크기는 MAE와 양의 연관이 있지만, 주행 명령·속도를 고려하면 약해진다. 따라서 회전 장면 효과를 분포 효과로 단정할 수 없다. 이 세 특징의 선택손실 상관 구간은 모두 0을 포함한다.
전체 경로 RMS 퍼짐, 끝 속도 분산, waypoint jerk와 scorer 실패의 일관된 관계도 확인되지 않았다. 평균 점수 예측과 후보 순위 실패를 반드시 구분해야 한다.

## 유사한 후보 사이의 위험 경계

| 두 후보 XY RMS 거리 | 후보 쌍 수 | 실제 PDMS 차이 >5 쌍 | 순위 반전률 | 실제 차이 >50 쌍 | 그중 예측 차이 <5 쌍 |
|---|---:|---:|---:|---:|---:|
| within1m | 32383 | 5781 | 9.86% | 3120 | 363 |
| over1m | 96641 | 55255 | 22.88% | 19001 | 1897 |

1m 이하로 닮은 후보에서도 실제 점수가 50점 넘게 달라지는 경우가 있다. 하지만 가까운 후보의 순위 반전률은 먼 후보보다 낮아, 후보가 닮았다는 이유만으로 scorer가 더 실패한다고 해석하면 안 된다. 쌍들은 같은 장면·후보를 공유하므로 독립 표본 129,024개로 취급하지 않았다.

## 주요 오류 사례

| token | 현재 명령 | 30m 차량 수 | 선택 경로 예측/실제 | 실제 NC/TTC | predicted NC/TTC |
|---|---|---:|---|---|---|
| 075154c90a0a5d7c | left_turn | 8 | 98.88/0.00 | 0/0 | 0.997/0.991 |
| e61adc75d7da50c3 | left_turn | 8 | 84.60/53.12 | 1/0 | 0.979/0.908 |
| 625a015c1f7c5170 | left_turn | 9 | 98.18/58.33 | 1/0 | 0.994/0.987 |

모든 오류를 '회전 궤적이 어려움'으로 환원할 수 없다. 최악의 좌회전 명령 장면은 후보 endpoint heading 중앙값이 약 1.5°로, 후보가 실제로 크게 회전하지 않았는데도 충돌 위험을 놓쳤다. 명령과 경로 형태는 별도 변수다.

## 시각화 해석

- [특징별 산점도](scene_characteristics_vs_scorer_error.png): 점 1개 = 한 장면의 후보 집합. 위 행은 선택손실, 아래 행은 후보 MAE. 색은 실제 관측 주행 지시.
- [특징별 연관성과 CI](feature_associations_recording_bootstrap.png): CI가 0을 포함하는 관계는 부호가 안정됐다고 주장하지 않는다. 15특징×2결과의 탐색적 비교로, CI에 다중비교 보정은 하지 않았다. CSV의 naive p/FDR는 장면 독립 가정이며 주장 근거로 쓰지 않는다.
- NAVSIM 사례: 왼쪽 현재 전방 영상, 가운데 64개 경로를 실제 PDMS로 색칠, 오른쪽 같은 경로를 scorer 추정치로 색칠. 보라=낮음/노랑=높음. 빨강=선택 경로/파랑 점선=후보 내 oracle. 객체 박스는 현재 GT.
- [오류·같은 명령 비교 사례 1](generator_error_vs_comparison_case1_navsim.png). 사례는 손실을 확인한 뒤 선택한 설명용 예시이며 대조군 실험이 아니다.
- [오류·같은 명령 비교 사례 2](generator_error_vs_comparison_case2_navsim.png). 사례는 손실을 확인한 뒤 선택한 설명용 예시이며 대조군 실험이 아니다.
- [오류·같은 명령 비교 사례 3](generator_error_vs_comparison_case3_navsim.png). 사례는 손실을 확인한 뒤 선택한 설명용 예시이며 대조군 실험이 아니다.

## 사실 / 추정 / 아직 알 수 없는 것

**측정 사실**: 회전 명령의 MAE 증가, 좋은 후보가 있음에도 위험 후보 선택, 소수 TTC/충돌 오류의 손실 집중, 단순 후보 RMS 다양성과 선택손실의 불안정한 관계.
**추정**: 현재 scorer의 표현/채점이 충돌·TTC 경계의 후보 간 차이나 회전 장면의 객체 미래를 충분히 구분하지 못할 수 있다. 출력 오차만으로 표현 손실·학습 데이터 부족·GT 속도/가림 원인 중 무엇인지 확정할 수 없다.
**미확인**: 교차로·가림·신호등별 실패율. 신뢰할 별도 라벨을 만들지 않아 이 분류를 주장하지 않는다. 4카메라 scorer이므로 전방 영상 하나에 보이지 않는 객체도 영향을 줄 수 있다.

## 추천하는 다음 실험 하나

공개 DrivoR의 generator와 native64 후보를 고정한 채, 후보의 공식 NC/TTC 라벨을 이용한 scorer 안전 순위 보정의 유효성부터 검증한다. 학습·평가에서 동일한64 candidate context를 유지하고, 후보가 닮았지만 실제 안전 상태가 다른 쌍도 포함한다. 전체 후보에서 선택손실·NC/TTC 실패·진행률을 함께 평가한다. 목표는 후보 수 확대보다 현재 좋은 후보를 위험 후보보다 높게 고르는지다. 본 작업에서는 학습하지 않았다.
이 아이디어 자체는 새 성공 근거가 아니다. 이전 scorer_selection_learning_v1은 축소 학습된 다른 모델에서 BCE+pairwise ranking 보정을 시험했으나 개선하지 못했다(기존78.4753→조건부77.5871). 따라서 같은 보정기의 규모를 키우거나 sampler를 덧붙이기보다, 충분히 학습된 공개 모델·native64 문맥에서 가장 작은 보정만 통과하는지 먼저 확인해야 한다.
처음 학습한다면 공식 학습 split에서 라벨을 만들고 recording을 분리한다. 여기 사용한 NAVTEST 패널은 이미 여러 차례 노출된 개발 진단 자료로만 유지하며, 최종 주장은 독립 recording 평가에서 확인한다.

## 검증·실행 비용·재현

원본 캐시 4개 hash 불변, 이전 동일64 문맥 MAE/선택 PDMS 차이 <1e-10, 현재 token/ego/명령 대응 검사 통과. LPWM 등록 293개 파일 불일치0, 최신 13549update. 학습·queue 변경 없음.
CPU 분석 35.40초, 후속 sensitivity/report 6.24초(시각화 비용 별도). 새 GPU 시간·학습 step·encoder/scorer/oracle 호출 모두0. NAVSIM 시각화 환경/실측 시간은 visualization_manifest.json에 기록한다.

```bash
bash scripts/run_drivor_generator_failure_characteristics.sh
```

설정은 configuration.json, subset/원캐시 hash는 subset_manifest.json, 장면별 전체 값은 scene_characteristics.csv, 모든 연관성은 feature_associations.csv, 후보 쌍 집계는 candidate_pair_diagnostics.csv, sensitivity/검증은 sensitivity_and_validation.json에 있다.
