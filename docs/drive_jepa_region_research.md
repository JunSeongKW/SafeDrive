# SPARTAN / C-JEPA / IA-JEPA 착안 통제 실험

## 결과 확인 전 등록 — 2026-10-03

사용자 승인: 세 방법을 우선순위에 따라 적용하고 planning 성능 개선을 확인한다.
하위 질문: 같은 K8 예산에서 위치에 연결된 미래 정보, 희소 의존 관계, 관측 이력 마스킹,
움직임 기반 선택이 planning과 학습형 선택의 효용을 개선하는가?
기준 코드 `5cac8aa`; 설정 `configs/drive_jepa_selective_future/region_research_v1.json`.

## 적용의 범위

1. [SPARTAN v3](https://arxiv.org/html/2411.06890v3): 선택된 미래 메시지를 해당 128개
   planner 공간 cell에 scatter한다. 선택된 8 region 사이에 현재 특징으로 결정하는 한 층의
   directed attention을 두고 dense와 hard sparse를 비교한다. 자기 연결은 유지한다.
   Sigmoid threshold + straight-through와 고정 1e-4 sparsity penalty를 사용한다.
   논문의 Gumbel sampling / 제약식 Lagrangian / 여러 층 경로 regularizer를 재현하지 않는다.
   희소화 대상은 미래 정보를 planner에 전달하는 연결이며 기존 predictor의 context는 dense다.
2. [C-JEPA v2](https://arxiv.org/html/2602.11389v2): 선택 8개 중 무작위 4 region의 최신
   관측 latent를 가리고 이전 2-frame anchor로 대체한다. 선택 shortcut와 decoder context 양쪽에
   같은 개입을 한다. 해당 현재 latent와 전체 선택 미래 latent를 복원한다.
   같은 query/target/학습량의 관측을 가리지 않은 대조군을 둔다. 추론에는 완전한 현재 관측을 쓴다.
3. [IA-JEPA](https://arxiv.org/html/2605.15466v1): 관측된 4 RGB frame의 absolute change를
   다시 차분하고 평균해 8×16 region별 움직임 점수를 만든다. 상위 8개를 선택한다.
   미래 영상·collision annotation·future valid mask는 선택에 사용하지 않는다.
   카메라 운동 보정은 이번 비교에 포함하지 않는다.

모두 **논문에서 착안한 현재 구조의 변형**이다. Region은 객체 slot이 아니고, encoder 특징은
이미 전역 맥락을 포함하므로 정보 차단/인과 발견을 주장할 수 없다. C-JEPA earliest anchor도
고정 공간 위치의 과거 특징이며 움직이는 객체의 identity anchor와 다르다.

## 사전 고정한 실험

| 순서 | 조건 | 분리할 효과 |
|---|---|---|
| 1 | located dense learned / random | 위치에 연결된 미래 전달, 선택 효용 |
| 2 | located sparse learned / random | 같은 위치 전달 구조의 희소 연결 |
| 3 | located sparse current only | 미래 없이 추가 adapter만 쓴 대조군 |
| 4 | sparse history unmasked / masked | 관측 마스킹 자체의 효과 |
| 5 | sparse history masked random / motion | 마스킹 구조에서 학습형·무작위·움직임 선택 |

9조건 × 3 seed(29/47/83) × 800 update. 원본 frozen planner와 기존 공통 100-step
predictor warmup을 재사용한다. Train512/dev192 window, recording64/24, batch8,
같은 batch schedule, LR2e-5/selector4e-6, future loss0.01, current reconstruction0.01.
K8/2×2 native-cell region/4 future tubelet, 32 future output query로 고정한다.
마스킹 조건은 학습 보조 경로에서 현재 복원 query8개를 추가하므로 전체 훈련 계산량은 다르다.
현재 특징 대조군은 predictor를 호출하지 않으므로 더 싸다. 비용을 따로 보고한다.

새 과거 영상 cache는 기존 동일 704 window의 관측 frame0–3만 읽어 workspace에 쓴다.
사전 파일 검사에서 704개 모두 관측 영상 4장이 존재했다. 원본 cache와 데이터는 보존한다.
공용 GPU0/1 중 여유 확인한 한 장만 사용: 입장16GiB / reserve6GiB / 자체 allocated8GiB,
조건20분 / 전체3시간 상한. 실패나 음성 결과 뒤의 자동 sweep는 하지 않는다.

## 판단 기준

- 평가 checkpoint는 마지막800으로 고정. Update0는 원본 동일성 검사다.
- 모든 조건을 보고하며 dev 결과로 다음 조건의 구조·hyperparameter를 변경하지 않는다.
- ADE와 공식 IL loss를 계산하고 3 seed 대응 차이 및 recording 단위 bootstrap 95% CI를 보고한다.
  CI는 seed 평균 후 recording resampling이며 seed 불확실성/다중 비교를 모두 보장하지 않는다.
- 명령·현재 속도 계층으로 분해한다. 원본 점수로 난이도를 나누지 않는다.
- 기존 global K8 learned/random 결과를 보존된 비교 대상으로 쓴다. 같은 초기화·batch hash 확인.
- 같은 dev192 window의 공식 PDM scorer 평가도 준비한다. 새 cache는 workspace에서 공식 processor로
  만들고 현재 frame token / 5초 state 범위를 확인한다. Navtest/held-out은 사용하지 않는다.
- 원본보다 개선과 같은 예산 random보다 개선을 구분한다. 개발셋의 예비 결과이며 독립 test의
  개선이나 최종 논문 기여를 확정하지 않는다.

## 구현·검증

- `models/region_future_research.py`: 위치 연결 bridge, 희소 연결, history mask loss, motion 점수.
- `scripts/train_drive_jepa_region_research.py`: 관측 cache / 대응 실험 / 보존 checkpoint.
- `tests/test_region_future_research.py`: 원본 동일성, 위치 scatter, slot 교란, 희소 edge,
  관측 마스킹 범위, 미래 label gradient 차단, 움직임 국소화, selection detach.

결과는 실행 완료 후 아래에 덧붙인다.
