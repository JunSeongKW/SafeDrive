# 연구 작업공간 이름 변경과 호환성

날짜: 2026-10-01. 목적은 **이름만 보고 현재 연구와 과거 baseline을 구분**하고,
폴더·함수·변수·결과의 역할을 협업자가 바로 이해하게 하는 것이다. 연구 질문/알고리즘 변경은 아니다.

## 실제 경로 대응

| 이전 | 현재 |
|---|---|
| `/rhome/junseong/SafeDrive/` | `/rhome/junseong/PlanningAwareFuturePrediction/` |
| `scripts/research/selective_future_graph.py` | `src/planning_aware_future_prediction/models/selective_entity_future_prediction.py` |
| `scripts/research/test_selective_future_graph.py` | `tests/test_future_prediction_graph.py` |
| `scripts/research/run_graph_validation.py` | `scripts/validate_future_prediction_graph.py` |
| `docs/RESEARCH_STATUS.md` | `docs/research_status.md` |
| `docs/SELECTIVE_FUTURE_GRAPH.md` | `docs/selective_entity_future_prediction_graph.md` |
| `docs/GRAPH_V1_VALIDATION.md` | `docs/synthetic_validation_results.md` |
| `/rhome/junseong/research_sources/Drive-JEPA/` | `reference_repositories/Drive-JEPA/` |
| `exp/graph_cpu_env/` (이전 경로 shebang) | `runtime/environments/future_prediction_cpu/` (새로 생성) |
| `analysis/research/graph_v1_validation.json` | `results/synthetic_diagnostics/future_prediction_graph_v1_before_readability_refactor_20261001.json` |
| 기존 SafeDrive `README.md` | `README_SAFEDRIVE_ARCHIVE.md` |

루트 `README.md`는 현재 연구 안내로 교체했다. 이름을 늘린 소스는 formatter로 줄바꿈을 정리했다.
현재 코드의 함수/변수 대응과 표준 외부 API 예외는 `naming_conventions.md`에 있다.
CLI는 `--training-steps`, `--batch-size`, `--holdout-samples`, `--seeds`, `--output`을 사용한다.

## 보존한 것과 하지 않은 것

- Git 이력, `junseong/main` branch, 원격 `JunSeongKW/SafeDrive`는 유지했다. 원격 repo 이름을
  변경하거나 새 원격을 생성하지 않았다. 주소는 역사이지 현재 baseline 선언이 아니다.
- 기존 SafeDrive 모델·loss·CSV·checkpoint와 공식 Drive-JEPA 소스는 변경하지 않았다.
- 공식 참고 repository의 함수/변수나 외부 `torch.optim`, `Tensor.grad`, `torch.autograd.grad`는
  명명 대상이 아니다. Refactor 중 외부 API 참조 변경을 검사에서 발견해 수정한 후 다시 검증했다.
- 과거 raw report는 바이트 내용·metric·timestamp·source hash를 그대로 보존해 이동했다.
  fe8c930 링크의 옛 코드/문서는 Git 이력에서 계속 확인할 수 있다.
- 예전 `exp/graph_cpu_env`는 역사적 로컬 환경으로 남아 있고 현재 실행에 쓰지 않는다.
  Venv의 shebang은 절대 경로라 새 경로의 venv를 만들었다. 외부 dependency upgrade/install은 없다.
- 공용 dataset은 기존 symlink 그대로이고 원본에는 쓰지 않았다. GPU를 사용하지 않았다.
- 사용자/타 프로젝트 디렉토리와 기존 `.claude` 세션/메모리 저장 폴더는 재명명하지 않았다.
  Claude를 새 프로젝트 경로에서 시작할 때는 최신 `CLAUDE.md`→`AGENTS.md`와 HANDOFF를 읽힌다.
  이전 project slug의 auto-memory가 새 경로에서 자동 로드된다고 가정하지 않는다.

## 실행 전후 동작 비교

- Refactor 후 13개 graph/경계 검사가 모두 통과했다.
- 동일 seed 0/1/2, step 1,000, batch 128, holdout 4,096의 합성 선택 학습을 재실행했다.
- 3개 seed×7개 정책×3개 metric = **63개 수치가 이전 실행과 정확히 같았다**.
- 초기 selector의 9개 metric과 15개 gradient norm도 정확히 같았다.
- 새로운 metadata key, 경로, source hash, 환경 경로와 실행 시간만 다르며 알고리즘을 변경하지 않았다.
- 새 raw report: `results/synthetic_diagnostics/readability_refactor_validation_20261001.json`.

이는 기존 synthetic 진단의 동작 유지 검사다. 연구 가설의 새 성능 증거가 아니다.

## 다른 사람/에이전트가 시작할 때

```bash
git clone -b junseong/main https://github.com/JunSeongKW/SafeDrive.git PlanningAwareFuturePrediction
cd PlanningAwareFuturePrediction
```

현재 머신에서는 VSCode/터미널의 이전 작업 폴더 대신 새 폴더를 연다. 옛 디렉토리 이름으로
alias/symlink를 만들지 않아, 현재 연구를 SafeDrive로 다시 오해하게 하는 진입점을 남기지 않았다.
공통 명명 규칙은 프로젝트 `AGENTS.md`와 `docs/naming_conventions.md`에 commit되며,
로컬 home의 `AGENTS.md`/`WORKSPACE_GUIDE.md`에도 현재 작업 진입점을 기록했다.
