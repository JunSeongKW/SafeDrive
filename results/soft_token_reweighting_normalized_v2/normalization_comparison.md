# 중요도 출력 정규화만 추가한 비교 실험

연구 질문: 장면·ego 조건부 중요도의 작은 bias를 정규화로 키우면 실제 attention과 planning에 기여하는가?

## 변경과 통제

B/C 중요도 출력에 valid-token population 표준편차 정규화(분모 std+1e-6)만 추가했다. 초기 beta=0.1, MLP 및 출력층 초기화는 동일하다. 추가 파라미터·loss·학습률 변경은 없다. 정규화로 비균일한 중요도가 만들어지는 것 자체는 학습 효과나 객체 이해의 증거가 아니다.
navtrain10,000 / navval1,000, seed0, batch32, 각1,000update, 동일 초기 planner·캐시·batch schedule의 SHA 일치를 확인했다. A도 재학습했으며 기존 A 대비 예측 최대 차이는 1.13755417이다.

## 실제 결과

| 조건 | 이전 PDMS | 정규화 실험 PDMS | 이전 대비 | ADE m | FDE m |
|---|---:|---:|---:|---:|---:|
| baseline | 87.2890 | 85.6691 | -1.6199 | 0.5805 | 1.4396 |
| unconditioned | 85.1933 | 86.6783 | +1.4850 | 0.5804 | 1.4458 |
| conditioned | 87.2759 | 85.4146 | -1.8613 | 0.5803 | 1.4432 |

**재실행 변동:** 변경 없는 A도 이전 대비 -1.6199점 달라졌다. 첫 loss 차이는 update3에서 2.23517418e-08였으며 이후 증폭됐다. seed·초기 상태·배치 순서와 초기 검증 예측은 동일하다. 엄격한 연산 결정성은 설정되지 않았고 정확한 kernel 원인은 분리하지 못했다. 따라서 이전→이번 점수 차이를 정규화만의 효과로 귀속할 수 없다. 이번 A/B/C를 주 비교로 쓰되 아래 구간도 학습 재실행/seed 변동은 반영하지 않는다.

| 이번 실험 내 비교 | PDMS 차이 | 기록 단위 bootstrap 95% 구간 |
|---|---:|---|
| B_minus_A | +1.009260 | [0.4083701250244172, 1.709814069072292] |
| C_minus_A | -0.254498 | [-1.124838709839886, 0.7627395023628153] |
| C_minus_B | -1.263758 | [-2.0174206003595754, -0.48795589017867735] |
| C_beta_zero_minus_C | -0.000178 | [-0.0002652663841508937, -8.984847311470156e-05] |
| C_shuffle_minus_C | -0.000144 | [-0.00026344008726079834, -2.7223176895767437e-05] |

## 기능적 사용과 성능을 구분

C의 추가 bias 표준편차: 이전 0.00057546 → 0.09818598. 기존 QK 대비 표준편차 비율 평균: 0.02515% → 4.28702%.
C에서 bias를 끌 때 평균 waypoint 변화: 이전 0.000001592m → 0.000269002m.
C의 정상·bias0·shuffle PDMS는 위 표로 판단한다. 기존 planner가 이미 ego를 받으므로 C가 ego를 쓰도록 보장되지는 않는다. 한 모델의 개입은 기능적 의존성 검사이며, 재학습 대조나 객체의 인과적 중요도 검증과 다르다.
이번 C의 명령 변경에 대한 중요도 변화는 장면 내 중요도 std의 약 0.212%다. 강한 주행 명령 의존을 배웠다는 근거는 없다.
이번 bias OFF/shuffle의 점수 차이가 작다면, bias를 실제로 계산하고 attention을 바꾼다는 사실과 planning에서 유의미하게 활용한다는 주장은 구분해야 한다. 나머지 decoder 경로가 첫 층 bias의 영향을 약화할 가능성은 추정이며 이 실험으로 원인을 확정하지 않는다.
C−A의 기록 단위 구간에0이 포함된다. planning 개선은 불확실하다. 다음 실험 하나는 연산 재현성 설정을 먼저 검증한 뒤 동일 A/B/C를 추가 seed에서 반복하는 것이다(이번에 실행하지 않음).

## 비용·검증·한계

학습 합계 0.048351 GPU h. 기존 실제 feature 캐시를 재사용하여 이번 encoder/cache 생성0초; 재사용 검증 4.36초. profile/평가/CPU 시각화 비용은 학습 비용에 포함하지 않는다.
기존6종 검증과 실제 training sample의 정규화 std·bias ON/OFF attention/경로 영향·단일 valid-token finite gradient 검사를 통과했다. CPU 검증 상세는 verification.json. 상황별 점수·하위 지표·파라미터·메모리·latency·gradient는 report.md/results.json에 있다.
단일 seed, 과거 사용된 부분 navval 개발셋이다. 독립 test나 전체 NAVTEST 점수, 객체 이해/계산량 절감으로 해석하지 않는다. Bootstrap은 기록 간 변동만 다루며 학습 seed 변동은 다루지 않는다. beta는 자유롭게 학습되므로 정규화 후에도 필요하면 bias를 줄일 수 있다.

## 시각화

![Bias and attention intervention](normalized_bias_and_attention_difference.png)

이전 C와 정규화 C의 bias는 같은 색상 범위다. 마지막 세 열은 동일한 정규화 C에서 OFF/ON/차이를 비교한다. Grid는 공간 anchor이며 객체 detection/segmentation이 아니다.

![Same scene command](same_scene_command_bias_difference.png)

동일 scene·동일 C에서 중요도 입력의 명령만 변경했다. native planner ego는 그대로다. 각 패널은 직진 명령 대비 bias 차이다. 영상과 일치하지 않는 명령도 있어 성능/인과 검증으로 해석하지 않는다.

## 재실행

```bash
cd /rhome/junseong/PlanningAwareFuturePrediction
/rhome/junseong/envs/kjs-drive-jepa-extension/bin/python scripts/run_soft_token_reweighting_suite.py --config configs/soft_token_reweighting/pilot_normalized_v2.json --replay-id rerun_001
```
