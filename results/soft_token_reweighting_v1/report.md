# Frozen-token soft reweighting pilot

단일 seed 0, 공식 navtrain 10,000장면 / navval 1,000장면(61개 recording)의 부분 개발셋 실험.
이 실험은 고정된 시각 표현에 상황·주행 지시 조건부 중요도를 주는 것이 기존 decoder attention보다 planning에 도움이 되는지 묻는다.

## 실제 결과

| 모델 | PDMS | ADE m | FDE m | 검증 loss | 학습 GPU h | 추가 파라미터 |
|---|---:|---:|---:|---:|---:|---:|
| baseline | 87.2890 | 0.5911 | 1.4578 | 0.23222 | 0.0157 | 0 |
| unconditioned | 85.1933 | 0.5882 | 1.4514 | 0.23234 | 0.0161 | 269,346 |
| conditioned | 87.2759 | 0.5832 | 1.4447 | 0.23053 | 0.0158 | 269,346 |

| 대응 비교 | PDMS 차이 | recording bootstrap 95% 구간 |
|---|---:|---|
| B_minus_A | -2.0957 | [-3.7046401283344936, -0.41563913740073605] |
| C_minus_A | -0.0131 | [-0.7792832590874638, 0.702554766370099] |
| C_minus_B | +2.0826 | [0.3767055955095482, 3.6986901370163423] |
| C_beta_zero_minus_C | -0.0000 | [-1.3716921906953283e-06, -4.1980505253390996e-07] |
| C_shuffle_minus_C | -0.0000 | [-1.488325491853613e-06, -5.399375042477575e-07] |

## 입력·구현·통제

공개 Drive-JEPA perception-free ViT-L의 full planning checkpoint(encoder와 planner)를 초기값으로 재사용했다. proposal-centric 모델이 아니다. Encoder는 eval/frozen이며, 관측 전방카메라 과거·현재(-0.5,0초) 2프레임만 쓴다. 상하28px crop, OpenCV linear512×256, ImageNet 정규화를 유지했다.
Encoder의 B×512×1024 출력에 원래 AvgPool2d2를 적용한 B×128×1024 FP32 feature를 저장했다. 이번 방법이 추가로 token을 줄이지 않는다. 메모리는128개 시각 grid와1개 ego token이다. 원래 image_fc, 위치 embedding, planner Transformer encoder3층/decoder3층, trajectory head 전체를 A/B/C 공통으로 학습한다. 이 planner 내부 encoder는 동결한 시각 ViT와 다른 모듈이다.
중요도는 LN(feature), valid-token 장면 평균, ego8→32 embedding, 좌표xy/camera/time을128hidden MLP에 넣어 계산한다. 첫 decoder cross-attention의 float mask(B×8,8,129)에 beta*r를 더한다. 모든 head/query에 공유하고 ego token bias는0, padding은-inf다. K/V, residual, 나머지 attention은 유지한다. 초기 beta=.1, 출력층 std=.001. Valid-token 평균을 빼서 상수 offset을 제거한다.
B는 중요도 모듈에만 ego/명령0을 주며 native decoder에는 실제 ego를 준다. B/C 구조·파라미터 수와 초기 상태가 같다. A/B/C 모두 동일 pretrained planner와 seed0 batch schedule, Adam lr1e-4/기본 betas/weight_decay0, 공식 length-normalized L1(alpha5), dropout0, FP32를 쓴다. 추가 loss·regularizer·future module은 없다.
공통 batch32 × 1000update. Profile 가중치는 버리고 원 초기값에서 재시작했다. 학습 총 0.0476 GPU h, 3h 상한 이내.
전체 split의 scene/log/timestamp를 확인했고 train101개 recording과 val61개 recording 중복0이다. 공식 log split을 유지한다. 데이터 규모 축소와 과거 navval 사용 이력, 공개 checkpoint의 validation 선택 가능성 때문에 독립 최종 test로 부르지 않는다.

## 검증

verification.json의 자동 gate가 통과했다. beta0 출력/loss 동등성, finite/nonzero 중요도·planner gradient, encoder frozen/no-gradient와 cached step의0 encoder call, padding확률0/내용불변, 실제MHA 합성 bias 단조증가, raw공식 feature builder/ego/GT와 캐시 일치가 포함된다.

## 비용 및 중요도 진단

| 모델 | step초 | decoder batch ms | peak allocated GB | beta | 중요도 entropy/log128 | command 변경 상대효과 |
|---|---:|---:|---:|---:|---:|---:|
| baseline | 0.0412 | 4.652 | 0.349 | 0.000000 | 1.000000 | None |
| unconditioned | 0.0500 | 7.054 | 0.394 | 0.096578 | 1.000000 | 0.0 |
| conditioned | 0.0459 | 6.801 | 0.394 | 0.095010 | 1.000000 | 0.0031047973316162825 |

캐시 생성 671.8초, 데이터/manifest 준비 56.9초. 공식 채점 총 127.9초, 예측/진단 총 28.2초. 학습 GPU 시간에서 분리했다. GPU0 공유 상태 실측으로 간섭이 있는 latency이며 속도 향상으로 해석하지 않는다. Token 수가 그대로여서 연산량 감소를 주장하지 않는다.

## 상황별 결과

| 상황 | 표본 | A | B | C |
|---|---:|---:|---:|---:|
| left | 286 | 88.8209 | 90.0733 | 90.2819 |
| straight | 546 | 88.1592 | 83.8548 | 87.4812 |
| right | 168 | 81.8530 | 81.2360 | 81.4913 |
| speed_below_0.5_mps | 55 | 91.6081 | 90.1065 | 92.0114 |
| speed_at_least_0.5_mps | 945 | 87.0376 | 84.9074 | 87.0003 |

좌/직진/우는 실제 ego command를 사용했다. 정지는 현재속도<0.5m/s라는 관측 기준으로만 표시했다. 신뢰 가능한 교차로 라벨은 없어 생성하지 않았다. 하위 NC/DAC/EP/TTC/comfort/DDC는 results.json 및 모델별 scores CSV에 보존했다.

## PDMS 하위 지표 (0–100 환산)

| 지표 | A | B | C |
|---|---:|---:|---:|
| no_at_fault_collisions | 99.1500 | 98.7000 | 99.1500 |
| drivable_area_compliance | 95.4000 | 93.5000 | 95.5000 |
| ego_progress | 79.2536 | 77.1440 | 79.1821 |
| time_to_collision_within_bound | 96.7000 | 96.4000 | 96.6000 |
| comfort | 100.0000 | 100.0000 | 100.0000 |
| driving_direction_compliance | 97.0500 | 97.3000 | 97.2000 |

## 해석과 다음 실험

C−A의 기록단위 구간이0을 포함한다. 이 pilot으로 조건부 reweighting의 성능 향상을 확정할 수 없다. 평균차이와 seed간 불확실성을 구분해야 한다.
C−B는 ego조건의 추가 이득을, C의 beta0/valid-token shuffle은 학습된 bias 사용 여부를 검사한다. 이 개입은 재학습 대조가 아니며 분포 변화 효과를 포함한다. 높은 entropy는 균일한 bias, 작은 command상대효과는 약한 명령 의존을 시사하며 객체 이해의 증거가 아니다.
확인된 사실: C의 실제 qk logit 표준편차 평균은 2.354835, 추가 bias는 0.00057546다. 장면별 비율 평균은 0.02515%다. 중요도 확률의 변동계수는 0.05755%로 거의 균일하다. FP64 entropy=0.999999964359; 기존 FP32 값1은 반올림이다.
C의 beta를0으로 만들 때 waypoint의 평균 이동은 0.000001592m, 최대 0.000097043m였다. PDMS 차이는 약 -0.000000891점으로 실질적 차이가 없다. 이 수준의 bootstrap 구간이0을 제외해도 물리적으로 유의미한 효과라고 주장하지 않는다.
명령만 바꿀 때 중요도 변화는 원 분포 표준편차의 약 0.310%, ego8개 값을 장면 간 섞을 때 장면별 상대변화 평균은 1.264%였다. 입력 의존성이 약하게 측정됐으나, 완전히 무시한다고 단정할 수는 없다. 섞은 ego는 해당 영상과 맞지 않을 수 있어 인과적 성능 검사로 해석하지 않는다.
추정: 현재 bias의 크기가 작아 기존 content attention을 실질적으로 바꾸지 못한 것이 직접적인 제한 요인일 수 있다. 작은 초기 출력층, 이미 학습된 planner와의 공동 최적화, 시각 feature의 전역 혼합 중 무엇이 원인인지는 이번 실험만으로 구분하지 못했다. B와 C는 학습 후 planner 가중치도 다르므로 점수 차이를 추론 중 중요도 배치의 기여로 설명할 수 없다. 조건부 객체 중요도를 배웠다는 증거는 확인하지 못했다.
다음 실험 하나: valid token 사이에서 중요도 r의 표준편차를 정규화해 beta=0.1이 실제 약0.1 logit 규모가 되도록 한 A/B/C 통제 pilot. 데이터·seed·초기 planner·batch·step·loss를 유지하고 bias의 기능적 사용과 PDMS를 다시 검사한다. 추가 정규화 loss나 hard selection을 도입하는 제안은 아니며, 이번 작업에서는 실행하지 않았다.

![Learning and PDMS](learning_and_pdms.png)

![Importance and attention](importance_and_attention.png)

시각화는 미리 고정된 명령별 첫 validation 장면이다. 위치는8×16 grid anchor이며 ViT와 planner encoder의 전역 정보 혼합 때문에 해당 픽셀만의 인과적 중요도로 해석하지 않는다. Token은 과거·현재 두 프레임을 함께 인코딩하므로 현재 프레임의 단일 객체에 대응한다고 가정하지 않는다.

## 재실행

--replay-id가 별도의 config/output/results 경로를 생성한다. 같은 replay-id를 재사용해 결과를 덮어쓰지 않는다.

```bash
cd /rhome/junseong/PlanningAwareFuturePrediction
/rhome/junseong/envs/kjs-drive-jepa-extension/bin/python scripts/run_soft_token_reweighting_suite.py --config configs/soft_token_reweighting/pilot_v1.json --replay-id rerun_001
```

캐시 및 checkpoint는 outputs/soft_token_reweighting_v1, 공유 수치/검증/manifest/그림은 results/soft_token_reweighting_v1에 있다.

## 측정 기록 정정

Pilot 완료 후 profiling 전체시간 변수 재사용 오류를 수정했다. raw execution.json의 profiling_seconds=1.748은 마지막 profile step 이후 시간이며 전체 profiling 시간으로 사용하지 않는다. 검증 완료 기록→profile 완료 기록의 관측 창은 16.272초(전체 profiling 실행의 상한 창)다. 개별 step 시간·학습 시간·캐시 시간·PDMS는 영향 없다. 실제 학습 source는 pilot_execution_source.py(SHA 66797bb57474bdb75ae8751a904c0fd7cc718a8029ab0af95ae3135a6396c9ef)에 보존했고, A/B/C 모두 이 동일 source를 사용했다. 이후 수정은 이 타이머와 진단/보고서/시각화에 한정되며 모델·학습 함수는 그대로다.
