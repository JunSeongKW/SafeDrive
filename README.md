# Planning-Aware Future Prediction

현재 맥락과 ego 주행 의도에 따라, 같은 예산에서 planning에 유용한 객체의 미래를 선택적으로
예측하도록 학습하는 연구 작업공간이다. **SafeDrive baseline 연구는 잠정 중단 상태다.**
이 이름은 연구 목적을 설명하는 작업명이며 최종 논문명·방법명·baseline은 아직 미확정이다.

## 처음 보는 사람/에이전트의 시작 순서

1. `AGENTS.md`: 명명 규칙·GPU·환경·공용 데이터 안전.
2. `HANDOFF.md`: 현재 상태와 다음 작업.
3. [연구 상태](docs/research_status.md), [계산 그래프](docs/selective_entity_future_prediction_graph.md),
   [합성 CPU 검증](docs/synthetic_validation_results.md).
   다음 단계 조사: [baseline·visual target 명세](docs/baseline_and_target_adapter_audit.md),
   [실제 NAVSIM 상태 adapter 진단](docs/navsim_state_adapter_validation.md),
   최신 [실제 영상 GT ROI pilot 검증](docs/visual_future_prediction_pilot_validation.md).
   현재 다음 실행 기준: [연구 질문·target 결정](docs/research_question_and_target_decision.md),
   [여러-log 데이터 유효율](docs/navsim_visual_target_coverage.md), [최소 학습 계획](docs/minimal_target_ablation_plan.md),
   **최신 [200-update 미래 감독 비교 결과](docs/target_supervision_exploration_results.md)**.
4. [명명 규칙](docs/naming_conventions.md), [경로 이전과 호환성](docs/directory_migration.md).

## 현재 구현과 과거 자산의 구분

```text
PlanningAwareFuturePrediction/
├── src/planning_aware_future_prediction/models/   CPU graph·frozen video encoder·새 visual pilot
├── src/planning_aware_future_prediction/adapters/ NAVSIM GT-state / GT visual ROI adapter
├── tests/test_future_prediction_graph.py          gradient·경계 계약 검사
├── scripts/validate_future_prediction_graph.py    CPU 검증과 합성 선택 학습 실행
├── scripts/validate_navsim_state_adapter.py       실제 mini 로그·track·상태 gradient 진단
├── scripts/validate_visual_future_prediction_pilot.py  실제 영상·ROI·gradient 진단
├── configs/visual_pilot/                         official encoder pin·overlay dependency
├── configs/exploration/                          고정 split·A–E 실행 설정
├── scripts/cache_target_supervision_features.py  제한된 frozen feature cache 생성
├── scripts/train_target_supervision_ablation.py  거리규칙 K4·동일조건 A–E 학습
├── scripts/summarize_target_supervision_run.py    실측 기록·원본 무결성·공유 결과
├── scripts/survey_navsim_visual_target_coverage.py  CPU target 유효율/side projection 조사
├── docs/                                        연구·계산 그래프·검증·명명 문서
├── results/synthetic_diagnostics/                작은 공유용 검증 결과와 과거 raw 기록
├── results/adapter_diagnostics/                  실제 상태 연결 진단 (시각 JEPA 아님)
├── results/visual_diagnostics/                   영상+GT ROI+spatial 혼합 감독 진단
├── results/data_surveys/                         여러-log 유효율 summary·탐색 train/dev manifest
├── results/target_supervision_exploration/       200-update 비교 실측·곡선·한계
├── reference_repositories/Drive-JEPA/            공식 코드 참고 clone (git 제외)
├── runtime/environments/future_prediction_cpu/   CPU 실행 환경 (git 제외)
├── outputs/synthetic_diagnostics/               실행 산출물 (git 제외)
├── navsim/, exp/safedrive/, analysis/*.csv        중단된 SafeDrive 참고 자산
└── README_SAFEDRIVE_ARCHIVE.md                    기존 SafeDrive 안내 보존
```

현재 package는 legacy SafeDrive/NAVSIM을 import하지 않는다. Synthetic CPU fixture, GT-state adapter,
official frozen video encoder+GT ROI visual/spatial target pilot을 구현했다. GPU0 실제 영상 batch의
gradient·단일 update와33tests를 통과했다. 이는 혼합 감독 연결 진단이며 순수 visual JEPA,
detector 기반 inference, 공식 baseline score, H1/H2 성능 검증은 아니다.
최신CPU조사는16recording/128window의target 유효율과train12/dev4recording split을고정했다.
이후 명시적 in-memory rectification,373window 약616MB frozen cache와 fixed-rule runner를 구현하고
A–E 각각200update(seed29,batch8)를 실행했다. 38tests 통과. C의 낮은 예측 분산/branch 무시 경고가 있으며
아직 곡선이 내려가는 초기 탐색이다. Target/최종 성능/novelty를 확정하지 않았다.
고정K 선택은개발기반이고EgoFSD/ForeDrive 대비최종기여로확정하지않는다.
`navsim/`, 기존 `setup.py`, 기존 학습·cache script는 역사적 SafeDrive pipeline이다.
새 연구를 실행하려고 과거 학습/캐시 명령이나 root의 legacy setup을 실행하지 않는다.

## 현재 CPU 검증 실행

다른 머신에서 받을 때는 checkout 폴더명을 명시한다:

```bash
git clone -b junseong/main https://github.com/JunSeongKW/SafeDrive.git PlanningAwareFuturePrediction
```

```bash
cd /rhome/junseong/PlanningAwareFuturePrediction
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 \
  runtime/environments/future_prediction_cpu/bin/python scripts/validate_future_prediction_graph.py
```

실행 환경은 기존 torch를 읽기 전용 참조한 venv이며, 완전 독립 baseline 재현 환경은 아니다.
이전 checkout 경로의 venv를 재사용하지 않고 새 경로에 만들었다. 실제 baseline에는 별도 환경을 만든다.
단위 검사만 실행하려면:

```bash
CUDA_VISIBLE_DEVICES='' PYTHONPATH=src \
  runtime/environments/future_prediction_cpu/bin/python -m unittest discover \
  -s tests -v
```

실제 GT-state 진단 (공용 원본 읽기만, CPU):

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 \
  runtime/environments/future_prediction_cpu/bin/python scripts/validate_navsim_state_adapter.py
```

## 데이터·GPU·협업

- `/home/user/data/Dataset/`은 연구실 공용 원본: **절대 직접 수정하지 않는다**.
- 작업·변환·cache는 `/rhome/junseong/`; 새 원본 다운로드만 `/home/user/data/processed_dataset/`에 총 1 TB 한도.
- 승인 GPU는 0·1이며 기존 프로세스를 중지할 권한은 아니다. 현재 CPU 검증에서는 GPU를 쓰지 않는다.
  영상 pilot과200-update 비교는 점유 재확인 후 GPU0만 사용하고 종료했다. 현재 백그라운드 학습은 없다.
- Git 원격은 협업 이력 보존을 위해 `JunSeongKW/SafeDrive`, branch `junseong/main`을 유지한다.
  저장소 주소가 현재 연구의 baseline을 의미하지 않는다. 원격 저장소명 자체는 변경하지 않았다.
- 협업 시작 commit은 `95015df`; 코드·결과·문서를 commit 단위로 공유한다.
