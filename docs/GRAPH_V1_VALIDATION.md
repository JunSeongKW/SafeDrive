# 계산 그래프 v1 — CPU 검증 결과와 재현

날짜: 2026-10-01. 협업 시작 기준: `95015dfadbf4595520bb111b4f74d5793aef9fb3`.
이번 결과는 **synthetic graph 및 selector 학습 가능성 진단**이다. NAVSIM 성능, JEPA의
유효성, 미래 정보 필요성 또는 H1/H2/novelty를 증명하는 실험이 아니다.

## 1. 변경한 것

- `scripts/research/selective_future_graph.py`: 순차 조건부 ST 선택, 최소 predictor/planner,
  미래 target 분리, planning/auxiliary 직접 gradient 경계.
- `test_selective_future_graph.py`: stdlib unittest로 13개 graph/선택 계약 검사.
- `run_graph_validation.py`: 검사 실행, 3-seed 합성 선택 학습, 비교군, 수치/환경/source hash 기록.
- `SELECTIVE_FUTURE_GRAPH.md`: v1 명세. Detach의 forward 불변과 제거/교환의 출력 개입을 구분.
- 기존 SafeDrive/Drive-JEPA 모델이나 공용 데이터는 변경하지 않았다.

## 2. K-slot 선택 명세

각 slot에서 **남아 있는 valid 후보**에만 softmax를 적용하고, 최고 score의 한 객체를 hard로
선택한다. 이전 slot의 hard index는 다음 후보에서 제외한다. score 동점은 entity ID 오름차순으로
결정한다. Entity ID는 scorer 입력이 아니고 동점 처리용이다.

Forward는 서로 다른 K개 객체를 선택하고, backward는 해당 단계의 조건부 softmax를 쓴다.
이는 hard exclusion을 미분하지 않는 **편향된 surrogate**다. 단일 softmax K회 복제나
without-replacement 분포의 정확한 gradient라고 주장하지 않는다.

전체 valid 수≤K이면 모든 객체가 예산 안에 들어가므로 부분집합 선택의 surrogate gradient를
차단한다. 첫 12개 검사 이후 이 경계를 추가 검토해 13번째 검사와 구현을 보완했다.
이 처리와 fixture의 pooling planner는 **순서 불변의 집합 소비자**를 전제한다.
실제 entity adapter의 안정적 ID 확보 및 rank-aware decoder는 아직 검증하지 않았다.

## 3. Graph 검사 — 13/13 통과

| 검사 목적 | 결과 |
|---|---|
| Planning→selector/predictor/planner | 각각 유한한 nonzero gradient |
| Hard-index-only 및 선택 detach | 출력 정확히 동일; selector gradient 0 |
| Future detach | 출력 정확히 동일; predictor/selector gradient 0, planner gradient 유지 |
| JEPA auxiliary만 backward | selector/planner/target 차단; predictor gradient 존재 |
| 미래 target 교체 | loss는 바뀌지만 fixed-weight selection/future/ego 출력은 그대로 |
| Future 제거 / 다른 샘플 예측과 교환 | fixture 출력 변화; 학습된 NAVSIM 성능 변화 검증 아님 |
| Entity 순서 변경 | 선택 identity·출력·selector gradient 일관 |
| Invalid target의 NaN / 전부 invalid | 안전하게 마스킹, auxiliary loss 0 |
| N=0 / 현재 entity 전부 invalid | padding 및 유한한 fallback 출력 |
| K>N / 중복 방지 / tie / invalid gradient | padding·ID tie 규칙과 invalid gradient 0 |
| 전체 후보가 예산에 들어감 | 의미 없는 부분집합 gradient 0 |
| 비정상 설정 / valid ID 중복 / valid score NaN | 명시적으로 거부 |

서로 다른 항목이 같은 test에 묶여 있어 표 행 수와 unittest 수가 반드시 일치하지 않는다.
Gradient norm의 실제 고정 fixture 측정:

| backward loss/대조 | selector | predictor | planner |
|---|---:|---:|---:|
| Planning | 0.009558 | 0.275470 | 1.282172 |
| Auxiliary | 0 | 0.554858 | 0 |
| Hard indices | 0 | 0.275470 | 1.282172 |
| 선택 detach | 0 | 0.275470 | 1.282172 |
| Future detach | 0 | 0 | 1.282172 |

현재 encoder와 target encoder는 fixture에서 구현하지 않았다. H/C와 detached future latent를
외부 tensor로 제공한다. 이 표는 실제 visual encoder까지의 gradient를 검사한 결과가 아니다.

## 4. 작은 합성 선택 학습

질문: **planning loss의 ST gradient가 존재하는 것을 넘어 실제로 중요한 객체를 고르도록
학습할 수 있는가?** 설계된 작은 과제에서만 확인한다.

- Entity 6개, K=2. 각 객체에는 semantic key와 현재 위치 x·속도 v가 있다.
- Intent는 필요한 서로 다른 두 semantic key를 명시한다. 샘플마다 entity 순서를 무작위로 섞는다.
- 고정 analytic predictor가 선택 객체의 `x+2v`를 계산하고, 고정 planner가 둘의 합/√2를 출력한다.
  여기서 2는 합성 시간 범위이며 실제 데이터의 2초나 2개 future token을 뜻하지 않는다.
- Selector만 planning MSE로 학습한다. 정답 entity 선택 label에 대한 별도 loss는 없다.
- Seed 0/1/2, 각각 1,000 step×batch 128. Global no-intent 정책도 같은 구조/학습량으로 학습한다.
- Train/holdout 생성 seed를 분리했다. Holdout은 seed마다 4,096개이며 학습 중 선택/튜닝에 쓰지 않았다.
- 실행 전 코드에 둔 기준: relevant recall≥0.8, random 대비 MSE≤25%. 3/3 seed 통과.

| 정책 | Holdout planning MSE, 3-seed 평균 | 중요한 객체 recall | 두 객체 exact-set 정확도 |
|---|---:|---:|---:|
| Context/intent 학습 선택 | 0.001660 | 99.996% | 99.992% |
| No-intent entity-only 학습 | 7.167637 | 33.276% | 6.714% |
| 학습 선택기의 command 교란 | 6.643663 | 33.468% | 6.942% |
| Random K | 6.734545 | 33.272% | 6.868% |
| Motion K | 8.444285 | 33.248% | 6.657% |
| Fixed semantic key K | 6.667207 | 33.049% | 6.551% |
| Hindsight oracle | 0 | 100% | 100% |

학습 전 context selector recall은 seed별 33.35/35.89/34.23%였다.
학습 후 exact-set 정확도는 seed별 100/99.9756/100%다.
No-intent 정책도 entity key·현재 x/v는 본다. JSON 이름 `global_learned_no_intent`는
파라미터 공유 규칙을 뜻하며, 완전한 입력 독립 정책이라는 뜻이 아니다.
Fixed semantic은 key 0/1을 고르는 정책이지 불안정한 token index 고정이 아니다.
Oracle은 relevant label을 보는 hindsight reference이며 배포 가능한 비교군이 아니다.

**해석 한계**: intent가 중요한 semantic key를 직접 지정하는 매우 쉬운 과제이며,
predictor/planner를 학습하지 않는다. 미래값도 현재 x/v의 알려진 결정론적 함수다.
따라서 미래 예측이 현재 정보 전달보다 유리하다거나 자율주행에서 중요도가 학습된다고
결론 내리지 않는다. 3-seed 평균은 기술적 진단 요약이며 독립 성능 주장/유의성 검정이 아니다.

## 5. 환경과 재현 명령

실측: CPU 1 thread, Python 3.12.13, torch 2.8.0+cu128. GPU는 사용하지 않았다.
최종 실행 전체 wall time은 약 **38.82초**였다. 현재 호스트에서만 측정한 값이다.

기존 alpasim 환경을 업그레이드하지 않고 `exp/graph_cpu_env` venv를 만들었다.
`--system-site-packages`로 기존 torch를 **읽기 전용 참조**하므로 완전 독립 dependency 환경은 아니다.
그래프 진단용 임시 실행 환경이며 실제 baseline에는 별도 dependency-pinned 환경이 필요하다.
공용 데이터·외부 원본 다운로드·cache 생성·GPU 점유는 없었다.

```bash
cd /rhome/junseong/SafeDrive
/rhome/junseong/miniconda3/envs/alpasim-cuda128/bin/python -m venv --system-site-packages exp/graph_cpu_env
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 \
  exp/graph_cpu_env/bin/python scripts/research/run_graph_validation.py \
  --output exp/research/graph_v1/results_v1_final.json
```

Full report: `analysis/research/graph_v1_validation.json`.
원본 output: `exp/research/graph_v1/results_v1_final.json` (git 제외).
Report는 기준 commit, 3개 실행 source의 SHA256, config, seed, 환경, gradient norm 및 raw metric을 포함한다.
구현 파일은 이 commit에서 추가되므로 기준 commit은 협업 출발점이지 실행 코드 commit이 아니다.
Source hash로 실행 당시 코드를 식별한다.

## 6. 아직 미확인 / 다음 결정

- 실제 entity 생성·track association·미래 latent target·ego-motion 정렬.
- Drive-JEPA perception-based 내부 감사와 최종 baseline 선정.
- 공동 학습되는 predictor/planner에서 selector의 안정성, 미래 latent 무시·collapse.
- 동일 용량의 **현재 entity feature 전달 adapter** 대비 미래 예측의 추가 기여.
- 같은 K·학습량의 실제 규칙/random 비교, 상황별 성능, 독립 holdout, wall-clock/FLOPs.
- ST surrogate의 다른 temperature/seed/실제 데이터 안정성 및 대안 estimator와의 차이.

ChatGPT에 검토받을 결정: 순차 조건부 ST를 첫 실제 prototype의 임시 선택 방식으로 유지할지,
어떤 현재-feature 대조와 미래 target adapter가 최소 비용으로 미래 예측의 기여를 분리할지.
다음 단계는 **실제 adapter 및 perception-based 코드 감사 후 baseline 결정**이다.
