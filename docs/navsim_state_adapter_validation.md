# 실제 NAVSIM GT-state adapter — CPU 연결 진단

2026-10-01. **실제 mini 로그를 읽은 engineering smoke test**다. 시각 JEPA·공식 Drive-JEPA
재현·H1/H2 성능 검증·공동 학습 안정성 검증이 아니다. 현재 입력에 GT 객체 기하정보를 사용하므로
특권 입력 pilot이다. SafeDrive 학습을 재개하지 않았고 공용 데이터·환경·GPU를 변경하지 않았다.

## 실제 실행 범위

- `dataset/navsim_logs/mini/2021.05.12.22.00.38_veh-35_01008_01518.pkl`의 window start0/12.
  동일 log·동일 scene `165060762e765a5a`의 두 구간이며 독립 성능 표본 둘이 아니다.
- History4, future8, 예상 간격0.5s, timestamp 허용 오차0.05s. 실제 마지막 offset은
  4.001300 / 4.001175s로 기록했다. 시각은 index 기반으로 정확히4s라고 가정하지 않는다.
- 현재 반경40m의 GT 차량·보행자, 거리순 candidate cap32, selected K4.
  H `[1,32,10]`, ego-history/dynamics C `[1,16]`, command `[1,4]`.
  future **state** target `[1,32,8,6]`, 예측 `[1,4,8,6]`, ego `[1,8,3]`.
- Valid future labels255/256와249/256. Absent/invalid track은 loss mask로 처리하고
  미래 유효성으로 현재 후보/선택을 바꾸지 않는다.
- 작은 learnable S/P/D, 총6,601 parameter. 각 구간에서 공동 backward와 Adam lr0.001의
  **1회 update**만 수행했다. Loss 감소/수렴을 성공 기준으로 사용하지 않았다.
- Front image1920×1080의 header/무결성만 확인했다. Image tensor/encoder forward는 미수행이다.

## 검사 결과

**28/28 unittest 통과**: 기존 graph13개 + state adapter13개 + 합성 rule/회귀2개.
추가 합성 선택 학습 또는 accuracy tuning은 하지 않았다. 기존 고정 fixture의 gradient norm15개도
historical readability report와 정확히 같다.

첫 실제 구간의 gradient norm:

| loss/대조 | selector | predictor | planner |
|---|---:|---:|---:|
| Planning | 0.00350536 | 0.15334630 | 0.94902224 |
| Future-state auxiliary | 0 | 0.32624449 | 0 |
| Selection detach | 0 | 0.15334630 | 0.94902224 |
| Future detach | 0 | 0 | 0.94902224 |

두 번째 구간도 같은 zero/nonzero 경계를 통과했다. Detach는 fixed-weight forward 출력이
정확히 같았고, state prediction을 0으로 바꾸면 untrained fixture 출력이 변했다.
이것은 **학습된 모델이 미래 정보를 유용하게 쓴다**는 성능 증거가 아니다.
Target label 교체 후에도 fixed-weight selection/ego output은 동일했다.

Association 검사: annotation 순서 변경, 새 track/소멸, 먼 미래 target, invalid NaN, 빈 현재 집합,
중복 track, timestamp gap, log 경계, lidar extrinsic translation을 검사했다.
실제 데이터의 annotation을 key로 사용했을 뿐 detector association을 검증하지 않았다.

처음 실제 로그 검증에서는 표준 `atan2(R[1,0],R[0,0])` yaw와 공식 NAVSIM의
`pyquaternion.yaw_pitch_roll` convention이 roll/pitch가0이 아닐 때 약간 달라 ego XY target 대조가
실패했다(첫 구간 lateral 최대 약0.00021m 차이). 허용 오차를 완화하지 않고 공식 yaw convention으로
수정한 뒤 검사를 다시 통과했다. Nonzero roll/pitch regression test를 추가했다.

공용 log 및 실제 읽은 두 image의 SHA256은 전후 같았다. 모든 공유 원본을 전수 hash한 것은 아니며,
코드상 입력 파일은 읽기만 했다. Output은 프로젝트 `outputs/adapter_diagnostics/`로만 기록한다.
Ruff 검사와 `git diff --check`도 통과했다.

## 재현과 공유 결과

```bash
cd /rhome/junseong/PlanningAwareFuturePrediction
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src \
  runtime/environments/future_prediction_cpu/bin/python -m unittest discover -s tests -v
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 \
  runtime/environments/future_prediction_cpu/bin/python scripts/validate_navsim_state_adapter.py
```

기존 환경을 upgrade하지 않은 CPU venv(Python3.12.13, torch2.8.0+cu128, 기존 torch 읽기 전용 참조).
실행된 파일·config·source hash·parent commit·elapsed time·실측 shape/gradient는
`results/adapter_diagnostics/navsim_tracked_state_validation_20261001.json`에 있다.
이 JSON의 parent commit은 **실행 전 기준**이며 새 코드의 commit이라고 주장하지 않는다.
State adapter / graph / 실행 script의 source hash로 실제 코드 내용도 식별한다.

다음은 [baseline·visual target 감사](baseline_and_target_adapter_audit.md)의 official encoder
weight gate와 실제 image ROI/teacher 연결이다. GT-state smoke를 이 gate 대신 사용하지 않는다.
실측 GPU-hour·VRAM·PDMS·visual latent 개선 결과는 아직 없다.
