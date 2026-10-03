# LPWM 표현 학습과 플래너의 개발 PDMS 비교

## 검증할 하위 질문

Ego intent를 encoder 내부에 넣고 planning과 관련 객체의 미래 위치를 공동 감독하면,
planning만으로 encoder를 학습할 때보다 유용한 미래 정보를 보존하고 PDMS가 개선되는가?

사용자의 2026-10-03 명시적 학습·플래너 연결·PDMS 평가 요청에 따른 새 실험이다.
완료된 LPWM 영상 복원 파일럿과 별개이며, 기존 WA/encoder 실험을 재개하지 않는다.

## 구현과 통제

- 시작점: 이전 raw seed29 LPWM checkpoint를 모든 조건/seed에 공통 사용한다.
- 입력: 현재와 0.5초 전 전방 RGB 두 장(128×128), 현재 command/속도/가속도 8차원.
- LPWM의 공식 particle encoder에서 64개 입자의 위치·크기·transparency·appearance·배경 정보를 추출한다.
- Ego FiLM을 encoder의 particle attribute CNN 내부에 삽입해 encoder 출력 자체가 조건화되도록 한다.
- 관측 입자 attention → 현재 객체 상태/4초 미래 위치 → trajectory decoder → 8개 ego waypoint.
- 기존 RGB decoder와 latent-action context/dynamics는 제거한다. 공식 LPWM 전체 알고리즘 재현이 아닌
  **LPWM 사전학습 encoder 기반 supervised task model**이다. 현재 입자 번호는 객체 track ID가 아니다.
- GT 투영 박스와 입자의 현재 geometry를 Hungarian 대응한다. 현재 geometry·metric 위치·class와
  동일 GT track의 미래 위치를 감독한다. 미래 위치는 현재 ego 좌표로 변환해 camera motion과 분리한다.
- 위험 가중치는 정답 ego 경로와 객체 미래 경로의 최소 거리로 정의한 bounded proxy다.
  학습 목표의 가중치에만 쓰며 실제 causal importance/safety 정답이라고 주장하지 않는다.
- GT 객체·track·미래 영상·미래 ego pose는 모델 입력에 없다. 미래 head 출력은 planner 입력에 있다.
- LoRA 없이 encoder 가중치를 직접 갱신한다. Planning loss와 객체 미래 loss의 encoder gradient를 검사한다.

| 조건 | LPWM encoder 갱신 | 내부 intent | 객체 미래 감독 |
|---|---|---|---|
| frozen_particles | 아니오 | 아니오 | 없음 |
| planning_joint | 예 | 예 | 없음 |
| object_future_uniform | 예 | 예 | 균일 |
| object_future_risk | 예 | 예 | 경로 근접도 가중 |
| object_future_risk_no_intent | 예 | 아니오 | 경로 근접도 가중 |
| ego_only | 사용하지 않음 | 해당 없음 | 없음 |

모든 LPWM 조건은 같은 planner/future head 구조이며, planning-only에서도 future head는 planning
gradient로 학습된다. Frozen과 joint 차이는 encoder 갱신과 내부 intent의 묶음 효과다.
위험 가중/균일 비교 및 내부 intent 유무 비교로 추가 효과를 분리한다.

## 사전 등록 범위

`configs/lpwm_planning/controlled_v1.json`: 6조건 ×3seed(29/47/71) ×1,000update, batch8.
512 train /192 development, recording 분리, 기존 공식 개발 metric cache 재사용.
모든 704개 window에서 두 관측 영상이 존재함을 확인했다. 미래 영상 가용성으로 표본을 고르지 않는다.
전체 train 22직진/65회전/390가림후보/35기타, 개발 12/25/147/8이다.
가림후보는 투영 겹침 proxy이고 motion 유형과 독립적인 label이 아니다.
Primary condition은 결과를 보기 전에 `object_future_risk`로 고정한다. 마지막 checkpoint만 평가한다.
개발 점수로 조건/학습량을 재선정하지 않으며 독립 test 결과로 부르지 않는다.

공식 Drive-JEPA는 같은192장면의 보존된 궤적/점수를 참조한다. 사전학습량/해상도가 다르므로
LPWM의 엄격한 matched baseline은 frozen/planning-only이며 Drive 비교로 architecture 우월성을 분리할 수 없다.

GPU0·1 한 worker씩, 여유6GiB/allocated20GiB/각run60분/worker6시간 상한.
학습 완료된 모델부터 별도 CPU 프로세스가 공식 PDM을 평가한다. 공용 원본은 읽기만 한다.

## 평가

PDMS와 collision/drivable/progress/TTC/comfort, ADE, 현재 객체 박스 대응, 미래 객체 위치 오차,
동일 선형 probe, scene/미래 branch 교란을 보고한다. Recording cluster bootstrap은 paired seed 평균 차이에
적용하며 전체 seed 불확실성/다중 비교를 해결하지 않는 탐색적 구간이다.
직진·회전·가림후보 외 command/현재 speed 분해도 보고해 단일 상황 정의에 의존하지 않는다.

## 근거와 범위

[LPWM](https://arxiv.org/html/2603.04553v1)은 particle 표현을 imitation learning에 연결하는 기반을 제공한다.
이번 구현은 사전학습된 encoder를 task supervision에 맞게 바꾸는 후속 변형이다.
[SAVi++](https://arxiv.org/abs/2206.07764)는 실제 주행 영상의 객체 표현 학습에서 감독 신호 설계의
필요성을 뒷받침하는 선행연구다. 기하/객체 감독 추가만으로 novelty를 주장하지 않는다.
