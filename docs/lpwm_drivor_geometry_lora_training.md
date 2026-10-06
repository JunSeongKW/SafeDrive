# Particle 위치·크기·presence head까지 planning LoRA를 적용하는 조건

**최신 사용자 지시: 본학습 보류.** 사용자가 [계층별 설명](lpwm_lora_layer_catalog.md)을 읽고 적용할 대상을
선택할 때까지 새 조건을 시작하지 않는다. 아래는 후보 구현·검사용 설정이며 새 본학습은 실행되지 않았다.
기존 첫 epoch checkpoint는 보존·중단 상태다. 새 root의 `pause.requested`와 `pending_user_layer_selection.json`을 존중한다.

## 사용자 지시와 검증할 질문

2026-10-06 사용자 지시: DrivoR 비교는 25 epoch 학습 후로 미룬다. 첫 epoch에는 particle 표현을
시각화하고, 위치·크기 변화가 없으면 기존 학습을 중단한 뒤 head LoRA를 추가해 다시 학습한다.

검증할 하위 질문은 **현재 particle geometry까지 planning gradient로 직접 적응시키면,
기존 attention/future LoRA보다 주행에 필요한 정보를 보존하고 사용할 수 있는가**다.
재배치 자체가 객체 이해 또는 planning 성능 향상을 보장하지 않는다.

## 별도 조건의 구현

기존 조건과 source/checkpoint는 보존한다. 새 진입점:

- 모델: `src/planning_aware_future_prediction/object_centric/lpwm_drivor_geometry_lora.py`.
- 학습: `scripts/train_lpwm_drivor_geometry_lora.py`.
- 설정: `configs/lpwm_drivor_geometry_lora/navsim_v1.json` 및 `navsim_v2.json`.
- 큐: `scripts/queue_lpwm_drivor_geometry_lora.py`.
- 표현 진단: `scripts/monitor_lpwm_drivor_geometry_representations.py`.

기존 Q/V rank32 LoRA는 유지하고 다음 6개 Linear map에 adapter를 추가한다.

| Head | 첫 Linear | 마지막 Linear | 역할 |
|---|---|---|---|
| `xy_head` | 2048→256, rank8 | 256→4, rank4 | 현재 위치 평균·분산 출력; deterministic 학습에서는 평균을 사용 |
| `scale_xy_head` | 2048→256, rank8 | 256→4, rank4 | 현재 크기 평균·분산 출력; deterministic 학습에서는 평균을 사용 |
| `obj_on_head` | 2048→256, rank8 | 256→1, rank1 | presence 분포를 결정하는 출력 |

추가 파라미터는 **57,633개**. 모든 adapter는 `alpha=rank`, scale1이며 출력 projection을0으로 초기화한다.
원래 LPWM 파라미터109,545,263개와 buffer는 고정한다. Head LoRA를 포함한 LPWM adapter 총1,401,121개,
새 FiLM/projection/공식 planner를 포함한 전체 학습 파라미터18,471,199개다.

이미지 CNN은 우선 고정한다. Head만으로 현재 geometry에 gradient가 연결되고 출력이 변할 수 있는지를
실제 입력·planning loss로 검사한다. Conv-LoRA는 이 경로의 연결 실패나 후속 근거가 있을 때 별도 조건으로 검토한다.

## 학습 조건과 초기화

새 조건으로 실행할 경우 public LPWM과 동일 seed2의 새 planner에서 시작한다. 이전 epoch1의 LoRA,
planner, optimizer 상태를 이어받지 않는다. 따라서 이전 조건의 나머지24epoch라고 부르지 않는다.

- v1: navtrain+navval103,288장면, 25epoch40,350update.
- v2: 별도 public 초기화, navtrain85,109장면, 10epoch13,300update.
- GPU0·1, GPU당 microbatch16×누적2×2GPU=effective64, loader2/oracle4 per rank.
- 카드 전체 사용량48 decimal GB 제한. 본학습 전 동일 DDP/microbatch 실측을 통과해야 한다.
- 원래 DrivoR planner/loss/ego11D 경로·기본 LR2e-4·AdamW·25epoch scheduler를 유지한다.
- 기존 LPWM command4D FiLM 유지. RGB reconstruction 및 직접 객체 GT 보조 loss는 추가하지 않는다.
- 공식 scorer의 proposal stop-gradient는 유지한다. Geometry adapter는 visual memory를 통한
  trajectory loss와 scorer loss의 gradient를 받는다.

## 실행 전 검사

`scripts/audit_lpwm_drivor_geometry_lora.py`는 실제 영상2장면×4cam과 공식 online oracle/loss로 검사한다.

- 초기 zero-LoRA 출력과 이전 공개 모델 출력의 완전 동일성.
- 이전 조건과 planner 초기 state의 완전 동일성.
- 현재 위치·크기·presence에서 해당 head adapter까지의 autograd 연결.
- 실제 planning loss에서 head별 gradient가 유한한 양수인지.
- 두 번의 검사 업데이트 후 geometry adapter만 끌 때 위치·크기·presence가 달라지는지.
- 원래 가중치·buffer SHA 보존, 공식 scorer 좌표 stop-gradient 보존, 카드 메모리 제한.

검사용 가중치는 폐기하며 본학습에 사용하지 않는다. 첫 검사에서 결과 JSON의 NumPy 정수 타입 직렬화
문제를 발견해 inventory를 Python 정수로 수정했다. 수정 후 검사 전체를 다시 실행해 통과했다.
실제 DDP2rank/유효64 검사도 별도 실행한다. 학습 정상성과 주행 성능 개선을 구분한다.

## 중간·최종 평가

기존96장면 panel을 유지해 위치·크기·presence, 객체/미래 readout, 명령 경로 분해,
particle·미래 개입을 같은 방식으로 추적한다. 이 panel과 navval은 본학습 분포다.
새 모델의 초기 표현도 별도로 저장하며, 과거 진단 결과를 새 모델의 결과로 덮어쓰지 않는다.

첫 epoch 원래 조건 시각화: `scripts/visualize_lpwm_epoch_particle_geometry.py`.
같은 particle ID를 비교하고, 점은 중심, 사각형은 실제 glimpse 크기, 점 반경은 presence를 뜻한다.
그림에 표시하는16개 사각형은 학습 전 presence 상위16개로 고정하며 전체64개 중심은 모두 표시한다.
분포 통계는 선택된16개가 아닌 96×4×64=24,576개 전부로 계산한다.

DrivoR와의 성능 비교는25epoch 이후로 미룬다. 기존 최종 full navtest와 독립 v2/warmup/navhard 큐를 유지한다.
해상도·사전학습·encoder 명령·adapter 예산 차이는 남아 있으므로, 점수 차이를 순수 register 대 particle 효과로 단정하지 않는다.
