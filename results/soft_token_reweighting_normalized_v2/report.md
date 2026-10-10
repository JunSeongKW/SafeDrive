# Frozen-token soft reweighting pilot

단일 seed 0, 공식 navtrain 10,000장면 / navval 1,000장면(61개 recording)의 부분 개발셋 실험.
이 실험은 고정된 시각 표현에 상황·주행 지시 조건부 중요도를 주는 것이 기존 decoder attention보다 planning에 도움이 되는지 묻는다.

## 실제 결과

| 모델 | PDMS | ADE m | FDE m | 검증 loss | 학습 GPU h | 추가 파라미터 |
|---|---:|---:|---:|---:|---:|---:|
| baseline | 85.6691 | 0.5805 | 1.4396 | 0.22969 | 0.0156 | 0 |
| unconditioned | 86.6783 | 0.5804 | 1.4458 | 0.22995 | 0.0166 | 269,346 |
| conditioned | 85.4146 | 0.5803 | 1.4432 | 0.23062 | 0.0162 | 269,346 |

| 대응 비교 | PDMS 차이 | recording bootstrap 95% 구간 |
|---|---:|---|
| B_minus_A | +1.0093 | [0.4083701250244172, 1.709814069072292] |
| C_minus_A | -0.2545 | [-1.124838709839886, 0.7627395023628153] |
| C_minus_B | -1.2638 | [-2.0174206003595754, -0.48795589017867735] |
| C_beta_zero_minus_C | -0.0002 | [-0.0002652663841508937, -8.984847311470156e-05] |
| C_shuffle_minus_C | -0.0001 | [-0.00026344008726079834, -2.7223176895767437e-05] |

## 입력·구현·통제

공개 Drive-JEPA perception-free ViT-L의 full planning checkpoint(encoder와 planner)를 초기값으로 재사용했다. proposal-centric 모델이 아니다. Encoder는 eval/frozen이며, 관측 전방카메라 과거·현재(-0.5,0초) 2프레임만 쓴다. 상하28px crop, OpenCV linear512×256, ImageNet 정규화를 유지했다.
Encoder의 B×512×1024 출력에 원래 AvgPool2d2를 적용한 B×128×1024 FP32 feature를 저장했다. 이번 방법이 추가로 token을 줄이지 않는다. 메모리는128개 시각 grid와1개 ego token이다. 원래 image_fc, 위치 embedding, planner Transformer encoder3층/decoder3층, trajectory head 전체를 A/B/C 공통으로 학습한다. 이 planner 내부 encoder는 동결한 시각 ViT와 다른 모듈이다.
중요도는 LN(feature), valid-token 장면 평균, ego8→32 embedding, 좌표xy/camera/time을128hidden MLP에 넣어 계산한다. 첫 decoder cross-attention의 float mask(B×8,8,129)에 beta*r를 더한다. 모든 head/query에 공유하고 ego token bias는0, padding은-inf다. K/V, residual, 나머지 attention은 유지한다. 초기 beta=.1, 출력층 std=.001. Valid-token 평균을 빼서 상수 offset을 제거한다.
B는 중요도 모듈에만 ego/명령0을 주며 native decoder에는 실제 ego를 준다. B/C 구조·파라미터 수와 초기 상태가 같다. A/B/C 모두 동일 pretrained planner와 seed0 batch schedule, Adam lr1e-4/기본 betas/weight_decay0, 공식 length-normalized L1(alpha5), dropout0, FP32를 쓴다. 추가 loss·regularizer·future module은 없다.
공통 batch32 × 1000update. Profile 가중치는 버리고 원 초기값에서 재시작했다. 학습 총 0.0484 GPU h, 3h 상한 이내.
전체 split의 scene/log/timestamp를 확인했고 train101개 recording과 val61개 recording 중복0이다. 공식 log split을 유지한다. 데이터 규모 축소와 과거 navval 사용 이력, 공개 checkpoint의 validation 선택 가능성 때문에 독립 최종 test로 부르지 않는다.

## 검증

verification.json의 자동 gate가 통과했다. beta0 출력/loss 동등성, finite/nonzero 중요도·planner gradient, encoder frozen/no-gradient와 cached step의0 encoder call, padding확률0/내용불변, 실제MHA 합성 bias 단조증가, raw공식 feature builder/ego/GT와 캐시 일치가 포함된다.

## 비용 및 중요도 진단

| 모델 | step초 | decoder batch ms | peak allocated GB | beta | 중요도 entropy/log128 | command 변경 상대효과 |
|---|---:|---:|---:|---:|---:|---:|
| baseline | 0.0402 | 5.002 | 0.349 | 0.000000 | 1.000000 | None |
| unconditioned | 0.0462 | 7.389 | 0.394 | 0.099888 | 0.998970 | 0.0 |
| conditioned | 0.0459 | 7.313 | 0.394 | 0.098200 | 0.999018 | 0.00212091370485723 |

캐시 생성 671.8초, 데이터/manifest 준비 56.9초. 공식 채점 총 130.6초, 예측/진단 총 24.2초. 학습 GPU 시간에서 분리했다. GPU0 공유 상태 실측으로 간섭이 있는 latency이며 속도 향상으로 해석하지 않는다. Token 수가 그대로여서 연산량 감소를 주장하지 않는다.

## 상황별 결과

| 상황 | 표본 | A | B | C |
|---|---:|---:|---:|---:|
| left | 286 | 88.6070 | 90.3418 | 88.1176 |
| straight | 546 | 85.3705 | 86.5183 | 85.1829 |
| right | 168 | 81.6379 | 80.9616 | 81.5659 |
| speed_below_0.5_mps | 55 | 90.2460 | 92.1042 | 91.9318 |
| speed_at_least_0.5_mps | 945 | 85.4027 | 86.3625 | 85.0353 |

좌/직진/우는 실제 ego command를 사용했다. 정지는 현재속도<0.5m/s라는 관측 기준으로만 표시했다. 신뢰 가능한 교차로 라벨은 없어 생성하지 않았다. 하위 NC/DAC/EP/TTC/comfort/DDC는 results.json 및 모델별 scores CSV에 보존했다.

## PDMS 하위 지표 (0–100 환산)

| 지표 | A | B | C |
|---|---:|---:|---:|
| no_at_fault_collisions | 99.0500 | 99.1500 | 99.1500 |
| drivable_area_compliance | 94.3000 | 94.8000 | 93.8000 |
| ego_progress | 77.5457 | 78.6280 | 77.6950 |
| time_to_collision_within_bound | 96.1000 | 96.7000 | 96.0000 |
| comfort | 100.0000 | 100.0000 | 100.0000 |
| driving_direction_compliance | 96.9500 | 97.1500 | 97.0000 |

## 해석과 다음 실험

C−A의 기록단위 구간이0을 포함한다. 이 pilot으로 조건부 reweighting의 성능 향상을 확정할 수 없다. 평균차이와 seed간 불확실성을 구분해야 한다.
C−B는 ego조건의 추가 이득을, C의 beta0/valid-token shuffle은 학습된 bias 사용 여부를 검사한다. 이 개입은 재학습 대조가 아니며 분포 변화 효과를 포함한다. 높은 entropy는 균일한 bias, 작은 command상대효과는 약한 명령 의존을 시사하며 객체 이해의 증거가 아니다.
이번 조건은 기존 valid-token 평균 제거 후 population 표준편차+1e-6으로 나누는 정규화만 추가했다. 정규화 이전 pilot과의 비교·개입·해석은 normalization_comparison.md 및 normalization_comparison.json을 참조한다. 정규화로 분포가 비균일해지는 것 자체는 학습된 의미의 증거가 아니다.

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

## 이번 정규화 조건의 실행·비용

입력 캐시와 subset은 기존 v1에서 읽기 전용 재사용했다. 위 캐시 생성 시간은 이전 실험 비용이며 이번 생성 비용은0초다. 실제 해시 검증 비용은 input_reuse.json에 기록한다.

```bash
python scripts/run_soft_token_reweighting_suite.py --config configs/soft_token_reweighting/pilot_normalized_v2.json --replay-id rerun_001
```

이전 v1 실행 명령은 정규화 없는 원 비교군의 재현용이다.
