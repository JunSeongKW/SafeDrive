# 동일 후보에서 Oracle·DrivoR·TOAD 배포 scorer 비교

## 결론과 범위

공식 NAVSIM v1 oracle은 미래 GT와 지도 등을 이용하는 실제 평가기다. Oracle 행은 동일 후보 집합 중 실제 PDMS가 가장 높은 경로를 선택하는 오프라인 상한이며, PDMS를 항상 100으로 놓은 값이 아니다.
TOAD에 별도의 새 scorer 구조는 없다. 비교 대상은 DrivoR 공개 Scaling 가중치와 TOAD에서 배포한 random-bouquet DrivoR 가중치다. 실제 scorer 및 decoder 코드가 같음을 확인했다. 모든 가중치 고정, 추가 학습/CEM 0.
NAVTEST64 / 38 recording / 현재 명령 좌21·직22·우21의 이전 노출 개발 패널이다. 전체 NAVTEST·독립 최종 test가 아니다. 두 scorer 모두 동일한 전체 pool을 채점해 candidate self-attention context를 통제했다.

## 실제 선택 결과

| 동일 후보 pool | 후보 수 | Oracle 최고 PDMS | DrivoR 선택 | TOAD 배포본 선택 | DrivoR 선택손실 | TOAD 선택손실 |
|---|---:|---:|---:|---:|---:|---:|
| balanced_native128 | 128 | 99.9470 | 93.6972 | 95.9485 | 6.2498 | 3.9985 |
| random_bouquet64 | 64 | 93.8428 | 71.4072 | 84.7829 | 22.4356 | 9.0599 |
| nearby64 | 64 | 99.5385 | 90.6614 | 96.1052 | 8.8771 | 3.4333 |

Balanced native128는 각 모델이 생성한 64개 경로를 합친 공통 pool이다. Random64와 nearby64는 이전 공식 TOAD 진단의 동일 경로를 재사용했으므로 TOAD 기반 분포라는 한계가 있다. 후보 생성 품질과 scorer 선택 품질을 분리하기 위해 같은 pool의 최고 점수와 선택 점수 차이를 계산했다.

## 예측값과 실제 값

| 후보 pool | scorer | PDMS 환산 예측 MAE | 장면 내 Spearman | 선택 NC 위반 | 선택 DAC 위반 |
|---|---|---:|---:|---:|---:|
| balanced_native128 | drivor_public | 17.6633 | 0.4304 | 1/64 | 0/64 |
| balanced_native128 | toad_released_drivor | 15.8864 | 0.4953 | 1/64 | 0/64 |
| random_bouquet64 | drivor_public | 15.2358 | 0.5836 | 5/64 | 6/64 |
| random_bouquet64 | toad_released_drivor | 8.2607 | 0.6465 | 4/64 | 0/64 |
| nearby64 | drivor_public | 19.9264 | 0.5695 | 1/64 | 3/64 |
| nearby64 | toad_released_drivor | 16.8640 | 0.6511 | 1/64 | 0/64 |

환산 예측 = 100×NC확률×DAC확률×(5×progress+5×TTC+2×comfort)/12. Raw log score를 공식 PDMS와 같은 단위라고 취급하지 않았다. Soft submetric 조합은 실제 oracle의 시뮬레이션/threshold 처리와 다르며 안전 확률 보장이 아니다. NC·DAC·TTC·comfort 및 progress 예측오차, 0.9 안전 confidence에서 false-safe 비율, 명령별 점수와 recording-bootstrap CI는 results.json에 기록했다.

## 실제 결과의 해석

1. 공통 native128의 선택손실은 DrivoR 6.25점, TOAD 배포본 4.00점이다. 같은 pool 안에 더 좋은 경로가 있으므로 이 차이는 후보 부족보다 **scorer 선택의 오차**를 측정한다. 후보 내 oracle도 전체 가능한 경로의 최적값은 아니다.
2. TOAD 배포본은 무작위64에서 +13.38점, 주변64에서 +5.44점 높았다. TOAD에서 유래한 두 후보 분포에서 원 DrivoR보다 더 잘 선택했다는 관측이다. Random-bouquet 학습 한 요소의 효과나 전체 CEM 성능을 입증하는 비교는 아니다.
3. 두 모델 모두 안전 판단 오류가 남는다. Native128에서 두 모델은 각각 NC<1인 경로를 1/64회 선택했다. Random64에서는 NC<1 선택이 5→4회, DAC<1 선택이 6→0회였다. Nearby64에서는 DAC<1 선택이 3→0회였다. NC<1은 공식 지표의 부분 감점도 포함하며 모든 경우를 완전 충돌로 부르지 않는다.
4. Native128 전체 후보의 환산 예측 MAE는 17.66→15.89점, 장면 내 순위 상관은 .430→.495였다. 순위 개선과 절대 점수 보정은 서로 다른 문제이며, 95.95의 선택 성능이 모든 경로 점수를 정확히 예측했다는 의미는 아니다.
5. Native128에서 실제 NC<1 또는 DAC<1인 후보 1,294개 중, 두 안전 head가 모두 .9 이상인 후보는 DrivoR 298개(23.03%), TOAD 배포본 247개(19.09%)였다. 후보들이 서로 독립이라는 가정이나 이 비율을 실제 배포 사고 확률로 해석하지 않는다.

### 같은 장면에 대한 대응 비교

| 후보 pool | TOAD 배포본−DrivoR PDMS | Recording bootstrap 95% CI |
|---|---:|---|
| balanced_native128 | +2.2513 | [+0.3639, +4.6766] |
| random_bouquet64 | +13.3757 | [+5.8999, +21.5252] |
| nearby64 | +5.4438 | [+0.3851, +12.4535] |

2000회 recording 단위 bootstrap이다. 이 패널의 표본 불확실성을 나타내며 학습 seed 변동이나 독립 최종 평가의 불확실성을 대신하지 않는다.

### 현재 주행 명령별 native128 선택

| 현재 명령 | 장면 수 | Oracle | DrivoR | TOAD 배포본 |
|---|---:|---:|---:|---:|
| left_turn | 21 | 100.0000 | 88.5786 | 94.6636 |
| straight | 22 | 100.0000 | 97.7253 | 97.9932 |
| right_turn | 21 | 99.8385 | 94.5959 | 95.0913 |

현재 ego status에 저장된 명령을 이용했다. 결과를 보고 후보 성능으로 상황을 분류하지 않았다. 각 21–22장면으로 특정 상황의 일반적 우열을 단정하지 않는다.

## 그림 해석

- `oracle_vs_public_scorers.png`: 위쪽 초록은 같은 후보 중 실제 최고, 파랑은 DrivoR가 고른 경로, 주황은 TOAD 배포본이 고른 경로의 **실제 PDMS**다. 초록과 다른 막대 사이가 선택손실이다.
- 아래 scatter는 x=모델 환산 예측, y=공식 실제 PDMS다. 대각선에서 멀수록 예측오차가 크다. 오른쪽 아래에 있는 점은 모델이 좋다고 예측했으나 실제 점수가 낮은 경로다.
- `case_*_navsim.png`는 결과를 보고 선택한 오류 사례다. 초록은 인간 GT가 아니라 **후보 내 oracle 최고 경로**, 파랑/주황은 두 scorer 선택이다. 극단 사례를 일반적인 장면처럼 해석하지 않는다.

## 통제·가중치·해석 한계

- 같은 이미지4camera/현재1frame/1148×672/PILresize/ImageNet, 같은 현재ego11. 각 checkpoint 자신의 backbone feature64×256FP32를 사용한다. 서로 다른 모델의 feature를 임의 교환하지 않았다.
- DrivoR asset은 파일명30epoch이나 metadata는 epoch28(29완료)/53911update; TOAD 배포본은 epoch29(30완료)/55770update다. 둘 다 배포 명시 trainval103k+Simscale134k지만 학습량·학습 recipe와 backbone 적응이 완전히 같은 통제는 아니다. 점수 차이를 random-bouquet 한 요소의 인과효과로 해석하지 않는다.
- 기존 local 축소 DrivoR와 공개 모델을 구분한다. 이번 공식weights의 점수로 과거 local scorer의 수치를 대체하지 않는다. CEM ON/OFF 성능 비교는 수행하지 않았다.
- 467tensor strictload, 원 DrivoR forward와 compatibility forward 최대오차 0, cache score 매칭/encoder 호출0, frozenweights 불변. 공식 oracle pair evaluator 검증 최대오차 0.0.

## 실제 비용

GPU0 phase 266.74초(load/I/O/cache포함), 전체 카드 관측 최대 42.010GB. 새 GPU1 작업 없음. 공식 CPU평가 86.24초/16,384후보/2worker. 보호293파일 hash 불일치0·LPWM 학습/queue 유지.

## 재실행

```bash
cd /rhome/junseong/PlanningAwareFuturePrediction
bash scripts/run_common_candidate_scorer_comparison.sh
```

[DrivoR 가중치](https://github.com/valeoai/DrivoR/releases/tag/Scaling) · [TOAD 배포 가중치](https://github.com/valeoai/TOAD/releases/tag/Nav1) · [TOAD method](https://arxiv.org/html/2606.07170v1#S3.SS2)

## NAVSIM 오류 사례

- [balanced_native128 / 075154c90a0a5d7c](case_1_balanced_native128_navsim.png): 실제 PDMS(DrivoR/TOAD배포/Oracle) 0.00 / 0.00 / 100.00.
- [balanced_native128 / e61adc75d7da50c3](case_2_balanced_native128_navsim.png): 실제 PDMS(DrivoR/TOAD배포/Oracle) 53.12 / 98.29 / 100.00.
- [nearby64 / 131ab0263b07507c](case_3_nearby64_navsim.png): 실제 PDMS(DrivoR/TOAD배포/Oracle) 0.00 / 100.00 / 100.00.

## 다음 확인 하나

같은 TOAD 공개 가중치·같은 장면·공통 random noise에서 CEM OFF와 전체 공식 CEM ON(comfort·anchor·fallback 포함)의 실제 PDMS와 oracle 선택손실을 비교한다. 현재 결과는 학습된 scorer의 경로 선택 오류가 남는다는 근거이며, 최적화 루프가 이를 줄이거나 악화시키는지는 아직 평가하지 않았다. 이 후속 작업은 자동 기동하지 않았다.
