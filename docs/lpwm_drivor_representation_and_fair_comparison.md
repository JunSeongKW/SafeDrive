# 학습 중 particle 유용성 검사와 DrivoR 공정 비교

## 2026-10-06 제안 검토: 첫 epoch 진단 이후 1148×672 조건

사용자는 현재 첫 epoch를 이전 조건과 비교해 개선이 보이면 DrivoR와 같은 전처리·1148×672로 본학습하는 방향을 제안했다.
**권고: 해상도 차이를 줄이는 타당한 후속 방향이다. 다만 현재 코드의 resize 설정만 바꿀 수 없고, 고해상도 구조·가중치 호환성과 실제 GPU 비용을 먼저 검증해야 한다.**
이번 턴은 코드 조사와 설계 검토이며 고해상도 구현·실측·학습·예약 변경은 수행하지 않았다.

### 첫 epoch가 답할 수 있는 범위

이전 Q/V-only와 새 세 경로 LoRA의 정확한 1614 update를 같은 장면·명령에서 비교한다.
위치/크기/presence 변화와 collapse를 검사하고, 객체 상태 readout·미래 정보·명령 반응·particle 개입에 따른 planning 반응을 함께 본다.
점이 차량이나 도로로 옮겨갔다는 사실만으로 planning 개선을 선언하지 않는다. 현재 96장면은 trainval 진단 분포다.
최종 navtest를 해상도·설계 선택의 반복 튜닝에 사용하지 않으며, 독립 성능을 말하려면 별도 미학습 개발 평가 설계가 필요하다.
첫 epoch 결과는 초기 학습 경향이며 25 epoch 성능이나 고해상도 효과의 증거가 아니다. 개선 미확인도 저해상도 정보 손실과 학습량의 영향을 배제하지 못한다.
조회한 비교 watcher snapshot은 285/1614 update, waiting_for_exact_epoch1_representations였다.

### 코드에서 확인한 변경 지점

- 현재 ParticleSceneEncoder.forward는 (4,3,128,128)을 assert하며 cache도128²다.
- LPWM models.py는 image_size를 정수·정사각형으로 사용하고 num_patches=(image_size//patch_size)^2로 계산한다.
- modules/modules.py ImagePatcher는 H/W에 같은 image_size를 쓰고 patch 위치와 unfold shape를 정사각형에서 생성한다.
- ParticleAttributeEncoder의 geometry Linear 입력 차원은 crop_size와 CNN 출력 공간 크기에 의존한다.
- ParticleFeaturesEncoder, BgEncoder, ParticleInteractionEncoder에도 공간 크기에 의존하는 flatten/projection이 있다.
- 실제 공개 hparams의 patch_size=16이며 1148은16의 배수가 아니다. 가장자리 padding/valid mask 및 원래 좌표 복원을 명시해야 한다.

따라서 H/W 분리, patch 좌표 및 경계 처리, glimpse 좌표·크기 정의, feature projection을 함께 조정해야 한다.
기존 CNN/heads를 재사용하는 후보는 고해상도 feature에서 고정 크기 glimpse 또는 pooling을 만드는 방법이다.
그 과정에서 전체 영상이 다시128²로 축소돼 작은 객체 정보가 사라지는지 검사해야 한다. 이것은 후보 설계이며 호환성 검증 결과가 아니다.
처음에는 현재64particle/카메라·16planner token/카메라를 유지해 particle 예산 변경 효과를 함께 섞지 않는 것을 권한다.
Context/dynamics는 latent particle 인터페이스를 유지하면 재사용할 후보지만 전체 checkpoint strict 호환이 자동 보장되지는 않는다.

### 전처리·실행·비교 권고

같은4카메라·현재시점, 원본에서1148×672 resize, 보간 및 GridMask 정책을 맞춘다. 공식 DrivoR은 ImageNet mean/std를 쓴다.
공개 LPWM은 RGB[0,1]을 사용하므로 동일 정규화값을 그대로 넣으면 입력분포가 달라진다.
Backbone 경계의 고정 역정규화 또는 정규화 적응 여부를 따로 검증·명시해야 하며, 서로 다른 처리를 완전히 동일하다고 부르지 않는다.
공식 코드: <https://github.com/valeoai/DrivoR/blob/main/navsim/agents/drivoR/drivor_features.py>.
공식 설정: <https://github.com/valeoai/DrivoR/blob/main/navsim/planning/script/config/common/agent/drivoR.yaml>.

입력 pixel은47.0859배지만 총VRAM·시간이같은배수라고 추정하지 않는다. 실제forward/backward·finite gradient·checkpoint 재사용·고정가중치 보존과 두 GPU 카드전체48GB 이하를 실측한다.
작은 microbatch부터 프로파일하고 gradient accumulation으로 effective batch64를 유지한다. 현재batch16의 유지 가능성은 미확인이다.
작은 실측은 실행 호환성 검사이며 최종 본학습은 동일한 전체 split·학습량·planner/loss로 수행하는 별도 조건을 권한다.

공정 비교용 기본안은 공개 가중치에서 새 고해상도 run을 시작하는 것이다. 현재128²의1epoch를 이어받으면 추가 학습 이력이 생긴다.
128² warmup→고해상도 전환은 별도 curriculum으로 기록하고 DrivoR에도 대응 학습량을 맞춰야 한다.
고해상도용 구조 변경 후 같은 구조의128² 대조가 있어야 해상도 효과와 구조 변경 효과를 분리할 수 있다.
해상도를 맞춰도 DINO/LPWM 사전학습, encoder 명령 FiLM, 미래 계산, LoRA 범위 차이는 남는다. 시스템 비교와 register–particle 단독 인과 비교를 구분한다.


## 2026-10-06 재확인: 카메라 입력과 현재 세 경로 LoRA의 비교 범위

공식DrivoR NAVSIM recipe와현runtime을대조했다. 카메라는F0/B0/L0/R0의4개와현재1frame로같다.
해상도는 **가로×세로 DrivoR1148×672, LPWM128×128**로다르며카메라당pixel수는771,456대16,384(47.0859배)다.
이는pixel수비율이며실제정보량/성능/연산량비율이아니다. 양쪽모두전체이미지를resize하지만LPWM은정사각형으로변환한다.
LPWM원체크포인트와연결구현을유지하려고128을사용한선택이지DrivoR과동일입력조건을맞춘것이아니다.

공식DrivoR은ImageNet정규화와학습GridMask를쓰며LPWM은RGB/255,normalize_rgb=False,GridMask없음이다.
현재LPWM은Q/V-only단계에서확대되어Linear229+Conv44/LoRA4,683,650개를학습한다.
**아래기존표의LPWM Q/V42개·1,343,488개수치는이전attention-only조건이다.**

같은공식NAVSIM evaluator·split의시스템성능비교는가능하지만설정/해상도/사전학습/계산량을함께보고해야한다.
사용자연구명제인register대particle의구조효과를분리하는엄밀한통제비교는현재조건만으로충족되지않는다.
차이가나면해상도·사전학습·명령FiLM·추가future연산·LoRA예산·증강의효과가섞인다.

후속대조설계는(1)공식DrivoR설정의성능기준,(2)동일camera/raw정보·해상도·종횡비·증강·planner초기값/
loss/update수·평가를맞춘조건,(3)동일pretrained visual features에서register/particle표현을비교하는기전대조로구분한다.
공통해상도에서도DINO/LPWM의사전학습차이는남는다. LPWM고해상도적응은checkpoint/feature head/particle수호환검사가필요하다.
128영상의단순upsampling은원래detail을복구하지않는다. DrivoR저해상도대조는해상도효과분리용이며공식DrivoR재현점수로부르지않는다.

현재학습은128²조건에서planning에따른LPWM표현적응을확인하는실험으로유지한다.
본학습설정을중간에바꾸거나DrivoR대조학습을추가기동하지않았다. 사용자의25epoch후DrivoR비교지시는유지한다.
근거: `results/lpwm_drivor_planning_path_lora_v1/camera_input_fairness_audit.json`.
공식설정: <https://github.com/valeoai/DrivoR/blob/main/navsim/planning/script/config/common/agent/drivoR.yaml>.

## 판단 요약

현재 실행은 **공식 DrivoR planner와 loss를 사용하는 LPWM perception 시스템**이다.
Planner backend, 관측 시점·카메라, 공식 split 및 epoch 수는 맞췄다. 그러나 현재 한 조건만으로
“구조화된 particle의 의도별 미래 정보가 register보다 유용하다”는 주장을 입증할 수는 없다.
해상도, 사전학습, encoder 명령 입력, 미래 연산량 및 학습 세부 설정을 함께 바꾼 상태이기 때문이다.

학습은 유지한다. 등록된 training source/config/optimizer에 손대지 않고, 저장 checkpoint를 읽는 별도 진단을 추가한다.

## 최초 중간 결과: 공개 초기값 → 100 updates

**후속 원인 확인:** [LoRA 위치 경로 감사](lpwm_drivor_lora_geometry_audit.md).
실제100update 체크포인트에서 LoRA를 꺼도 현재 좌표는 bitwise 동일했고, FiLM을 끄면 공개 초기 좌표로 복귀했다.
현재 attention LoRA는 좌표 생성 뒤에 있으므로, 공간 재배치가 작은 이유를 초기 warmup만으로 설명하면 안 된다.

96장면 모두 진단했다. 100/40,350 updates로 전체 예정량의 약0.25%이며 아직 warmup이다.
학습 전/후 같은 장면·동일 probe 규칙을 사용했다.

| 지표 | 초기값 → update100 | 현재 해석 |
|---|---|---|
| 현재 중심 이동 | 평균0.00305px, 최대0.05446px / 128px 영상 | 배치는 사실상 유지 |
| 차량 영역 presence 비율 | 9.4928% → 9.4923% | 집중 증가 미확인 |
| 보행자 영역 presence 비율 | 1.3347% → 1.3344% | 집중 증가 미확인 |
| 도로 proxy 영역 presence 비율 | 18.4076% → 18.4203% | 아주 작은 변화; semantic road GT가 아님 |
| 명령 변경 시 encoder-only 궤적 변화 | 0 → 평균0.2834m | 표현을 통한 명령 반응은 생김; 올바른 반응인지는 별도 |
| 객체 종류 readout macro F1, 전체14D | 0.3799 → 0.3778 | 정보 판독 개선 미확인 |
| 예측2초 feature로 GT 이동 판독 L2 | 3.8933m → 3.9021m | 개선 미확인 |
| 예측4초 feature로 GT 이동 판독 L2 | 8.0206m → 8.0322m | 개선 미확인 |

2초/4초 모두 예측 feature와 현재 feature의 readout 오차 차이 CI가0을 포함한다.
미래 순서를 뒤집거나 현재 반복으로 대체한 planning 점수 변화도 CI가0을 포함한다.
관련 particle 개입은 고정후보 선택 점수 감소를 보였으나, 유효 표본이5장면뿐이고 대조 개입과의
안정적인 차이를 확정하기에는 부족하다. 이를 planning 유용성 입증으로 보고하지 않는다.

**현재 결론:** encoder 의도 입력이 planner에 영향을 주는 경로는 작동한다. 반면 공간 재배치,
객체·미래 정보 보존 및 planning 이득이 함께 좋아졌다는 증거는 아직 없다. 초기100update로
방법 실패를 판정하지 않고 같은 기준으로500/1000/이후 checkpoint를 추적한다.

초기/100update 진단은288.7/257.0초, 최대 reserved0.654GB였고 원래 LPWM freeze SHA는 동일했다.
본학습의 GPU당batch16/누적2·유효64 및 기존 평가 큐는 유지했다.
GT 박스·미래 객체 상태·지도는 이 진단에서만 사용하며 LPWM 학습 입력이나 loss에 추가하지 않는다.

## 1. Particle이 driving에 유용한 정보를 배우는가

### 고정 진단 장면과 범위

공식 학습 장면에서 모델 성능을 보지 않고 seed71 해시와 사전 정의 회전 기준으로 골랐다.
24 recording group × 4장면 = 96장면: 직진42, 좌회전30, 우회전24.
회전 기준은 expert 4초 경로의 마지막 heading ±0.25rad다. 전체 모집단의 비율을 나타내는 표본은 아니다.
차량1,646·보행자1,304·자전거24개 **객체–카메라 관측**이 있다. 중복 view/근접 시점이 있으므로
2,974개 독립 객체라고 부르지 않는다. 자전거 분석의 표본은 특히 작다.

이 장면들은 upstream planner가 학습하는 데이터 안에 있다. 모니터링 결과를 독립 validation이나
full navtest PDMS로 보고하지 않는다. Probe만 recording 단위 fit18/evaluation6으로 분리한다.
전체 독립 planning 결과는 기존 큐의 full navtest 및 별도 v2 EPDMS에서 확인한다.

### A. 공간 분포: 위치·크기·presence를 함께 확인

**시각화 해석:** particle 위치는 LPWM이 위치와 scale로 주변 RGB glimpse를 추출하는 중심이다
(`reference_repositories/LPWM/modules/modules.py:3011`의 feature encoder). ‘그 부근에 시각 표현을 배치했다’는
해석은 가능하지만, planner의 attention weight나 주행 중요도 자체를 나타내지는 않는다.
Glimpse는 유한한 면적을 갖고 interaction/dynamics 및 공유 background feature가 정보를 섞는다.
또한 현재 planner 입력은 current+future attributes를 투영한 뒤 particle4개씩 평균하므로 점과 planner token이1:1이 아니다.
최신 monitor 그림의 노란 점은 현재 중심, 점 크기는presence, 색 사각형은 평가용GT투영박스다.
색 사각형을 particle glimpse 범위나 예측 detection으로 읽지 않는다. Presence도 주행 중요도 확률은 아니다.
하늘에 점이 있다고 하늘 정보만 쓴다고 단정할 수 없고, 보행자 위에 점이 있다고 정지 판단에 활용했다고 단정할 수도 없다.
이를 구분하려면 위치·scale·feature와 해당 particle 개입 시 planning 변화를 함께 확인한다.

- 현재 4카메라의 **64개 particle 전부**를 동일 장면에서 비교한다. Top16만 골라 성공/실패를 판단하지 않는다.
- 차량/보행자/자전거 GT 투영 영역 안의 중심 비율과 presence 가중 비율을 측정한다.
- 영역 면적 비율로 나눈 enrichment를 함께 기록해 큰 객체·큰 도로 영역이 유리한 효과를 분리한다.
- 학습 전 대비 좌표 이동량과 foreground feature 변화량을 별도로 기록한다.
- 주행 궤적 근처 객체를 정의하는 평가용 corridor proxy는 아래 particle 개입 대상 선정에 사용한다.

GT box는 3D cuboid의 pinhole 투영이다. Crop 없이 실제 full-image 128×128 resize와 맞췄다.
박스 투영만으로 실제 가림·가시성을 확정하지 않는다. 겹치는 박스는 가림 위험 proxy다.

도로는 픽셀 segmentation GT가 없으므로 **지도 drivable area + 평면 지면에 대한 카메라 ray 교점**으로 만든 proxy다.
현재 근거리 객체의 바닥 높이로 지면을 추정하고 ±0.5m도 함께 계산한다. 투영 객체 박스 영역은 제외한다.
언덕·고가·지도 오차·미라벨 가림 물체 때문에 정확한 도로 mask가 아니다. 96장면 중95개는 기존 map cache가 있다.
상단 1/3의 particle 비율은 보조 화면 위치 통계이며 ‘하늘 분할 정확도’라고 부르지 않는다.

**현재 LoRA에서는 원래 xy/scale/presence CNN/head가 고정**되어 있다. 명령 FiLM이 입력 feature를 바꾸어
현재 배치를 바꿀 수 있지만, Q/V LoRA는 interaction/context/dynamics의 feature와 미래 상태를 주로 바꾼다.
실제 autograd 검사에서 현재 xy는 이 Q/V LoRA 파라미터84개와 직접 연결되지 않았고, FiLM과는 연결됐다.
따라서 점이 크게 움직이지 않았다는 사실만으로 LoRA 학습 실패를 결론 내리지 않는다.

### B. 실제 객체·미래 정보 접근성: readout

현재 GT 박스와 particle glimpse의 겹침·presence로 전체 particle을 가중 pooling한다.
현재에 계산한 연결 가중치를 2초/4초 예측 particle에도 그대로 사용한다. 미래 GT로 particle을 다시 고르지 않는다.
지원 particle이 없는 객체도 zero feature로 남기며 조용히 제외하지 않는다.

같은 recording split과 고정 ridge 정규화0.1로 다음 feature를 비교한다.

| Readout 입력 | 검사 목적 |
|---|---|
| GT ROI 좌표·크기 | privileged geometry만으로 가능한 판독 기준선 |
| 현재 particle foreground 4D | 외형 feature에 객체 종류·미래 단서가 있는가 |
| 현재 particle 전체14D | 현재 구조화 표현에 보존된 정보 |
| 예측2초/4초 particle 전체14D | 현재 표현보다 미래 상태를 읽기 쉬워졌는가 |

종류 판독은 macro F1/balanced accuracy/클래스별 수를 보고한다. 미래는 현재 ego 좌표계에서의
GT track displacement L2를 측정하고, zero displacement 및 **GT 현재 속도를 쓰는 특권 CV 기준선**도 제시한다.
CV는 추가 GT 정보를 쓰므로 배포 가능한 동일 입력 planner 기준선으로 해석하지 않는다.
Current 대비 predicted feature 오차 차이에 recording bootstrap CI를 붙인다.

이 검사는 학습된 detector나 tracking 정확도가 아니다. GT가 객체 위치와 대응을 제공하는 readout이다.
LPWM particle ID를 객체 track ID와 같다고 가정하지 않는다. 높은 readout 점수만으로 planner가 해당 정보를
사용한다고 결론 내리지도 않는다.

### C. 의도에 따라 encoder 자체가 바뀌는가

고정된 12개 개입 장면에서 같은 영상에 command index0/1/2를 넣고 다음 경로를 나눈다.

1. **Encoder만 변경:** 대체 명령으로 LPWM 표현을 계산하되 planner ego 입력은 원래 명령 유지.
2. **Planner만 변경:** LPWM 표현은 고정하고 공식 planner의 ego 명령만 변경.
3. **둘 다 변경:** 실제 전체 조건화 경로.

현재 중심 이동량, foreground feature 변화, 미래 attribute 변화 및 궤적 변화를 각각 기록한다.
공개 초기화 한 장면에서는 encoder 쪽 변화0, planner-only 변화는 양수임을 검증했다.
이는 zero-init FiLM과 일치하며, 단순히 궤적이 바뀌는 것으로 encoder 의도 학습을 주장하는 오류를 막는다.

다른 명령의 실제 정답 경로는 주어지지 않는다. 따라서 counterfactual 명령 결과를 원래 human GT와
비교해 옳고 그름을 판단하지 않는다. ‘의도 반응’과 ‘그 의도에 적합한 표현’은 별도 주장이다.
후자는 실제 명령별 관련 객체 분석, 미래 readout 및 planning 개입의 일관된 개선이 필요하다.

### D. 그 정보가 planning에 기여하는가

같은 checkpoint·같은 장면에서 다음 inference intervention을 한다.

- 미래8단계를 현재 표현 반복으로 대체.
- 미래 단계 순서를 뒤집어 시간 순서 의존성 확인.
- Expert corridor에 가까운 GT 객체와 겹치는 최대8개 particle을 카메라 내 평균으로 대체.
- 동일 카메라 수·presence·glimpse 면적을 맞춘 다른 particle도 같은 수만큼 대체.

변경된 표현으로 generator와 scorer를 다시 실행한다. 동시에 **baseline의 64개 후보는 고정한 채 재채점**하여
후보 생성 변화와 scorer의 정보 활용을 구분한다. 공식 training oracle로 동일 후보 집합을 채점하고,
선택 궤적 점수·고정 후보 선택 점수/최선 후보와의 차이·ADE 변화 및 recording CI를 기록한다.
이 작은 학습 패널의 oracle score는 full navtest PDMS가 아니다.

개입 전 particle→memory 재구성과 기존 forward의 proposal/score가 bitwise 같은지 검사했다.
평균 대체와 미래 반복은 학습 분포 밖 입력일 수 있다. 단순 민감도를 ‘안전성의 인과적 증명’으로
해석하지 않고 matched control 및 재학습 ablation으로 보완해야 한다.

### 무엇을 성공 증거로 볼 것인가

객체 쪽 중심 이동은 보조 증거다. 강한 주장을 위해서는 다음이 함께 필요하다.

1. 객체/미래 정보 readout이 사전학습 상태와 적절한 현재 표현 기준선보다 개선된다.
2. Encoder-only 명령 변경 경로가 실제 표현과 planning에 영향을 준다.
3. 관련 particle/미래 정보를 바꿨을 때 matched control보다 planning 점수가 일관되게 나빠진다.
4. 독립 벤치마크의 동일 조건 재학습 ablation에서도 이득이 재현된다.

현재 planning loss만으로 latent rollout을 학습하므로, 8개 latent 단계가 실제0.5초 간격 미래 상태로
정렬된다는 보장은 없다. 미래 ablation에 민감하더라도 추가 비선형 연산의 효과일 수 있다.
GT 미래 상태 접근성과 시간 순서 검사를 함께 보고하며, 계산량을 맞춘 current-only 대조도 필요하다.

## 2. DrivoR와 공정하게 비교할 수 있는가

### 지금 맞춘 것과 남은 차이

공식 논문은 v1 train+val25epoch, v2 train10epoch, 4GPU×batch16, AdamW base LR2e-4를 사용한다.
[DrivoR 논문 Appendix D](https://arxiv.org/html/2601.05083v2#A4).
아래의 세부 구현 차이는 내려받은 `fc6e5aa` 코드와 현재 실행 source를 대조한 결과다.

| 항목 | 현재 상태 | 해석 |
|---|---|---|
| Generator/scorer/원래 loss/stop-gradient | 공식 코드를 직접 호출 | 동일 |
| 카메라·관측 프레임 | F0/B0/L0/R0, 현재1장 | 동일 |
| Planner ego 입력·주입 | 11D→256D, generator/scorer에 추가 | 동일;16장면 입력 오차0 |
| Planner memory | 64 tokens×256D | 동일 |
| 후보/시간 범위 | 64후보×8poses,4초 | 동일 |
| 공식 학습 split·epoch | v1 trainval25 / v2 train10 | 동일 |
| Optimizer·기본LR·유효batch | AdamW/2e-4/64 | 동일 |
| Q/V LoRA | rank32, scale1 | 방식 동일, 파라미터 예산 다름 |
| 이미지 | DrivoR672×1148 / LPWM128×128 | 큰 정보량 차이 |
| 사전학습 | DINOv2 LVD-142M / LPWM Sketchy | 서로 다른 데이터·목표·backbone |
| Encoder 명령 조건화 | DrivoR 없음 / LPWM FiLM | Intent 효과의 혼입 |
| Augmentation/정규화 | DrivoR GridMask0.7+ImageNet / LPWM 원래 정규화 | 추가 차이 |
| LoRA 학습량 | DINO589,824 / LPWM1,343,488 parameter | 같은 rank가 같은 적응 예산은 아님 |
| 내부 압축 | learned camera register / particle4개씩 고정 평균 | 효과가 서로 다른 압축 연산까지 포함 |
| 미래 연산 | DrivoR 없음 / LPWM8-step prior | 추가 연산·시간 정렬 가정 차이 |
| 정밀도 | 공식기본FP16 / 현재BF16+encoderFP32 | 구현 차이 |
| GPU 분할 | 공식4×16/accum1 / 현재2×16/accum2 | 유효batch는 같지만 난수/연산 순서 다름 |
| 마지막 배치 | 공식drop_last=True / 현재False | update 수가 정확히 같지 않음 |
| Backend 초기값 | 현재는 backbone 생성 전에 planner 생성 | seed만 같아도 native DrivoR 초기값은 다를 수 있음 |

공식 recipe와 표준 distributed sampler를 적용해 계산하면 v1은40325 대 현재40350 updates,
v2는13290 대13300이다. 이는 저자의 실제 로그를 읽은 수치가 아니라 코드에서 계산한 값이다.
현재 등록 실험 중간에 drop_last·precision·augmentation을 조용히 바꾸지 않는다.
기존 문서의 ‘동일 프로토콜’은 split/epoch/기본 optimizer 수준으로 한정하며, 완전한 재현이라는 해석을 정정한다.

### 권장 비교를 세 층으로 나눈다

**① 공식 benchmark 비교:** 공개 DrivoR 원래 설정과 현재 LPWM 시스템을 같은 pinned evaluator,
동일 full token 목록에서 평가한다. 실제 시스템 경쟁력을 비교하는 표다. DINO/LPWM 전체 차이의 결과이며
particle 구조만의 효과라는 인과 주장을 붙이지 않는다. 특히 v2의 human-error filtering 수정 전후 점수를 섞지 않는다.

**② 현재 연구의 통제 비교:** 동일 planner 초기 state, minibatch token 순서, effective batch,
precision, dropout/augmentation 정책, 학습 update 수, 평가 코드를 고정한 DrivoR 재학습을 따로 둔다.
Native resolution 결과와 공통 영상 정보량 조건을 함께 보고한다. DINO patch14에128이 나누어떨어지지 않으므로
공통128 RGB에 결정적 padding을 해140 입력으로 받는 등의 규칙을 명시해야 하며, 이를 native DrivoR 재현이라고
부르지 않는다. padding·위치 embedding 처리는 실행 전 동등성/shape 검사를 거친다.

Encoder 명령 입력은 `perception family × encoder intent off/on`으로 나눈다.
Backend ego 입력은 모두 유지한다. Intent 있는 LPWM만 원래 intent 없는 DINO와 비교하면 두 효과가 섞인다.
동일 rank 비교와 학습 파라미터/latency 예산을 맞춘 비교도 구분한다.
이렇게 해도 서로 다른 사전학습/backbone의 차이가 남는다. 구조 자체를 엄밀히 분리하려면
동일 pretrained visual features 위에서 particle화와 register화를 비교하는 추가 설계가 필요하다.

**③ LPWM 내부의 기전 ablation:** 동일 공개 초기화·동일 planner로 intent off/on × future off/on을
재학습하여 미래 정보와 의도 조건화의 효과를 분리한다. 현재 checkpoint에서 꺼보는 개입만으로 재학습 대조를
대신하지 않는다. Future-off는 token/channel budget을 유지하고, 미래 연산량 효과를 분리할 current-token mixer
대조도 고려한다. 현재 표현을 고정한 planner-only 조건은 동일 새 backend·학습량으로 다시 맞춰야 한다.
과거82.52의 작은 별도 planner 결과를 이번 대조군 점수로 재사용하지 않는다.

최종 비교는 복수 seed(예:2/17/41), paired recording CI, PDMS/EPDMS와 NC/DAC/TTC/진행/comfort,
사전 정의한 직진·회전·겹침/작은 객체 하위 그룹을 함께 보고한다. Test 점수로 epoch나 loss를 고르지 않는다.
이 문서는 검토·설계이며 새 DrivoR 대조군 장기 학습을 자동 큐에 추가하지 않았다.

## 진입점

- 고정 패널: `outputs/lpwm_drivor_representation_monitor_v1/panel.json`.
- 상태·결과: 같은 root의 `watch_status.json`, `update_*/{complete,summary,distribution,readouts,interventions,intent_sensitivity}.json`.
- 그림 갤러리: 같은 root의 `index.html`.
- 진단: `scripts/monitor_lpwm_drivor_representations.py --watch`.
- 공정성 감사: `results/lpwm_drivor_representation_monitor_v1/comparison_protocol_audit.json`.

초기값→update100/500/1000→이후2000단위 및 각 epoch 부근→최종 checkpoint를 순차 확인한다.
Trainer가100update마다 저장하므로 epoch 기준 검사는 그 시점 이후 처음 확보한 저장본을 쓸 수 있다.
실제로 검사한 update 번호와 checkpoint SHA를 항상 저장한다. 작은 진단은 GPU0 최대4GiB allocator,
카드 전체48GB 제한 아래에서 실행하며 메모리가 부족하면 학습을 우선한다.
