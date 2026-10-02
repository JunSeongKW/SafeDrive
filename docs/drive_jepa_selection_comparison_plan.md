# Drive-JEPA 선택 비교 v1 — 실행 전 고정 명세

기준 `2185ce5`. 사용자의 다음 단계 진행 요청에 따라 연결 검사를 반복하지 않고,
공식 frozen planner를 유지한 새 patch-future branch의 작은 학습 비교로 넘어간다.
WA/pilot/확대 cache/동적 예산은 재개하지 않는다.

질문: 같은 K4·미래 4 tubelet에서 선택 정책을 바꾸면 작은 개발 표본의
planning 학습 경향이 달라지는가? 이는 객체 선택이나 최종 논문 성능 검증이 아니다.

- 기존 native-recording train/development 배정에서 각각 16/8 recording,
  recording당 최대 8개 비중첩 window. Recording 및 window 순서는 SHA256로 고정한다.
  현재 두 front 이미지 존재만 입장 조건이다. 미래 이미지 누락은 auxiliary mask만 바꾸며
  window/현재 후보를 제거하지 않는다. Held-out 및 과거 노출된 두 recording은 제외한다.
- 현재 encoder/공식 planner 모두 frozen. 현재512 patch와 미래4×512 latent를 FP32로
  작은 별도 cache에 저장한다(상한3GiB). 원본 공식 crop/resize/정규화만 사용한다.
  같은 image-grid 미래 target이며 object tracking/ego-motion compensation은 없다.
- 동일 K4: 고정2×2격자 IDs `[136,151,360,375]`, seeded uniform random,
  planning-conditioned ST. 별도 한 대조는 ST와 같은 구조에서 auxiliary weight만0.
  규칙 선택은 image-grid 규칙이며 거리/TTC 규칙으로 부르지 않는다.
- 공통 predictor/bridge/selector 초기화, batch 순서, 3 seed(29/47/83), 각200 update.
  AdamW lr1e-4·weight_decay.01, constant schedule, batch8, gradclip1.
  공식 planning IL loss + .01×raw future latent MSE. 가중치는 dev를 보기 전 고정하며 튜닝하지 않는다.
  Planning→selector/predictor/bridge, auxiliary→predictor만, 원본 parameter 불변.
  Fixed/random은 selector parameter를 보유하지만 사용/학습하지 않는다(활성 용량 차이 명시).
- 평가0/50/100/200; 최종고정200 checkpoint만 주표에 보고. Best-dev 선택 없음.
  원본 no-branch는 재학습 없는 기준선. Train/dev window 및 recording macro XY ADE,
  공식 IL loss, 미래 MSE/persistence, 선택 ID 변화·분포·gradient/parameter update,
  peak memory/time를 기록한다. Future zero/swap은 의존도 진단일 뿐 인과 증명이 아니다.
- 최대 총3600초/조건240초/allocated6GiB. 이 범위에서 실패하면 조건/데이터/모델을
  바꾸어 자동 재시도하지 않는다. 현재 shared GPU0·1만 사용한다.

200 update에서 우열이나 future의 기여를 확정하지 않는다. 특히 auxiliary 없는 모듈도
현재 feature를 planner에 전달할 수 있다. 현재-feature 직접전달 대조와 공식 safety/progress
평가는 다음 별도 판단 사항이다. Navtest는 학습·선택·개발에 사용하지 않는다.

이 표본은 기존 pretrained checkpoint의 학습 데이터와 중복될 수 있는 navtrain 부분집합이다.
여기서 train/dev 분리는 **새 extension 학습**의 recording 분리이지 foundation model이
전혀 보지 않은 데이터라는 뜻이 아니다.
