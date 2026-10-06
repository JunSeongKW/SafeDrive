# Geometry·appearance·future 경로 LoRA 공동학습

## 이번 실험의 질문

현재 particle 배치, 같은 위치의 시각 정보, 미래 particle 예측을 함께 planning loss로 적응시키면
주행 상황과 ego 의도에 유용한 표현을 배우는가? 사용자가 세 경로를 직접 선택하고 본학습을 승인했다.
이 승인은 이전의 **계층 선택 전 새 학습 보류**를 이 새 조건에 한해 대체한다.

기존 attention-only 1614update checkpoint와 실제 중단1615update, geometry-head-only 준비 조건은 보존한다.
새 조건은 공개 Sketchy LPWM과 seed2의 동일한 초기 planner에서 시작한다. 기존1epoch나 연결 검사 가중치를 이어받지 않는다.

## 정확한 적응 대상

| 목적 | LoRA 대상 | 방식 |
|---|---|---|
| 현재 위치·크기·presence | prior CNN, attribute CNN, `xy_head`, `scale_xy_head`, `obj_on_head` | CNN은 Conv-LoRA, head의 두 Linear 모두 LoRA |
| 같은 위치의 정보 보존 | foreground appearance CNN/feature head, background appearance CNN/head, interaction의 영상 CNN·projection·attention·FFN·출력 head | Conv-LoRA + Linear LoRA |
| 주행에 필요한 미래 정보 | 공유 context prior의 projection·attention·FFN·prior 출력, dynamics projection·attention·FFN, 미래 위치·크기·presence·깊이·feature 출력 | Linear LoRA |

- **Linear229개 + Conv2d44개**, 추가 LPWM 학습 parameter **4,683,650개**.
- Attention Q/K/V/output rank32, 다른 Linear와 Conv는 rank8을 기본으로 입출력 차원에 맞게 제한한다.
  따라서 현재 위치·크기 최종 출력은 rank4, presence 최종 출력은 rank1이다. 모든 alpha/rank=1.
- 원래 LPWM **109,545,263개 parameter와 모든 buffer 고정**. BatchNorm 통계도 고정한다.
- RGB decoder와 planning에서 사용하지 않는 context posterior 출력에는 LoRA를 붙이지 않는다.
- Context의 encoder/dynamics/top-level alias는 같은 module이며 중복 adapter를 만들지 않는다.
- Conv-LoRA는 `W + B@A`로 펼친 커널을 적응시키고 기존 stride/padding/dilation/bias를 유지한다.
  B는0 초기화한다. 기존4D command FiLM hook은 **원래 convolution과 LoRA를 합친 출력**에 적용한다.
- 새 FiLM·particle trajectory projection·camera embedding·공식 DrivoR planner는 전체 학습한다.
  총 학습 parameter는 **21,753,728개**다.

구현: `src/planning_aware_future_prediction/object_centric/lpwm_drivor_planning_path_lora.py`.
전체 계층별 rank·dimension·대상 목록은 `results/lpwm_drivor_planning_path_lora_v1/registration.json`에 있다.

## 입력·loss·gradient

현재4카메라128×128 → 현재64particle/카메라 → context prior와 dynamics의8step 미래 →
현재+미래 속성126D를256D로 projection → 고정4particle 평균으로16token/카메라 → 공식 DrivoR planner.
동일 공식 ego11D planner 입력 및 encoder의 추가 command4D FiLM을 유지한다.

Loss는 기존 공식 DrivoR trajectory imitation + online oracle 후보 점수 학습을 그대로 사용한다.
별도 RGB reconstruction loss=0, 직접 객체 detection/box 보조 loss=0이다.
Online oracle은 지도·객체 정보를 사용하므로 **객체 GT 보조 loss 없음이 GT-free 학습이라는 뜻은 아니다**.

Trajectory loss → generator → 공유 scene memory → 미래/현재 particle → 모든 활성 LoRA 경로.
Score loss → scorer → 공유 scene memory → 같은 LoRA 경로.
공식 scorer의 proposal 좌표 stop-gradient와 비미분 oracle은 유지한다.
따라서 score loss가 candidate 좌표를 직접 안전 방향으로 미는 refiner를 새로 구현한 것은 아니다.
LPWM native parameter를 고정해도 activation gradient는 지나가며, 이 경로에 `no_grad`를 추가하지 않는다.

결정적 mean 기반 경로이므로 각 Gaussian head의 logvar 출력 row 전체에 학습 신호가 있다고 주장하지 않는다.
현재 confidence/score의 native detach도 유지한다. 검사에서 말하는 gradient 양수는 각 adapter의 출력 행렬 norm이다.

## 데이터와 실행

| 항목 | 설정 |
|---|---|
| V1 본학습 | 공식 navtrain85,109 + navval18,179 =103,288장면 |
| 학습량 | 25epoch, 1,614update/epoch, 총40,350update |
| GPU | 0·1만, 다른 프로세스 포함 카드전체48decimalGB 이하 |
| 배치 | GPU당16 × 누적2 × GPU2 =유효64 |
| Worker | rank당 loader2 + online oracle4 |
| Optimizer | AdamW, weight decay0.01 |
| Peak LR | LPWM LoRA2e-4, planner/새 projection2e-4 |
| Schedule | 기존 공식85,000/64×25 convention, warmup3,322update 후 cosine |
| 초기화 | 공개 LPWM + 새 planner seed2; 모든 추가 LoRA의 B=0 |
| Checkpoint | 100update·epoch마다, optimizer/scheduler/rank별RNG 저장 |

공식 dataset-size convention을 유지해 schedule은33,225update이고 실제 학습은40,350update이다.
이로 인한 마지막 cosine 반등도 기존 조건과 같으며 새로 바꾸지 않았다.
V1의 navval도 학습에 포함하므로 여기서의 고정96장면 검사를 독립 validation 성능이라고 부르지 않는다.

## 본학습 전 확인한 사실

- 실제2장면×4카메라·online oracle loss·optimizer2update 검사 통과.
- LoRA 초기 출력이 native 출력과 동일하고 초기 planner도 이전 seed2와 동일함을 확인.
- **273개 adapter 모두 planning gradient 연결 및 출력 행렬 norm>0 확인**.
- 현재 xy/scale/presence head와 prior/attribute CNN에서 현재 좌표까지 직접 연결됨을 확인.
- Geometry adapters만 꺼도 현재 위치·크기·presence가 달라짐을 확인. 이것은 연결 증거이며 성능 개선 증거가 아니다.
- Original LPWM parameter/buffer digest는 전후 `a5dd234560e97c6033391d059088fab0fb56ed4a595e33d7f664e379ae8ccd5d`로 동일.
- 두 GPU에서 실제 microbatch16/누적2/유효64로2update 완료, 최대 카드전체 **40,161,509,376bytes**.
  양 rank gradient가 유한·양수이고 메모리 한도 이내. Engineering 가중치는 본학습에서 사용하지 않는다.

증거: `results/lpwm_drivor_planning_path_lora_v1/engineering_audit.json`, `adapter_gradients.json`, `ddp_engineering.json`.

## 학습 중·후 검증

- 학습 전/100/500/1000/각epoch/2000간격/final의 고정96 trainval 장면,4카메라,직진·좌회전·우회전·가림 proxy를 관찰.
- 점의 위치·glimpse 크기·presence를 따로 비교하고 처음 고른 같은 particle ID를 추적해 시각화.
- 객체 종류 readout·미래 객체 상태 readout·주행 명령 민감도·particle/미래 표현 개입을 같은 진단으로 유지.
  지도 기반 road proxy 및 박스 대응은 보조 지표다. Particle이 차량/도로로 이동했다고 planning 효용을 단정하지 않는다.
- Monitor는 parameter를 바꾸지 않는다. GPU0 총42GB 미만에서만 진입하고 allocator4GiB, 검사 중46.5GB guard.
  본학습 전체48GB guard 및 초과 시 checkpoint 중단은 유지한다.
- 첫 epoch는 정확한 `epoch_01.pt`를 우선 사용한다. 시각화는 해당 update의 `geometry/`에 저장한다.
- V1 최종25epoch → full navtest 검증 → 별도 공개 초기화 V2 navtrain-only10epoch → warmup/navhard EPDMS 순서로 queue.
  단계가 실패·중단·불완전하면 다음 종속 단계를 진행하지 않는다. DrivoR baseline의1epoch 비교 학습은 실행하지 않는다.

DrivoR planner·loss·유효배치가 같아도 LPWM의 공개 사전학습 데이터,128² 해상도,미래 rollout,명령 FiLM,
particle pooling,이번에 확대한 LoRA 범위·parameter budget은 DINO register 조건과 다르다.
최종 benchmark는 시스템 비교이며 **register 대 particle 구조만의 인과 효과**를 입증하려면 이 차이들을 통제한 별도 비교가 필요하다.

## 운영 진입점

- Config: `configs/lpwm_drivor_planning_path_lora/navsim_v1.json`
- Train: `scripts/train_lpwm_drivor_planning_path_lora.py`
- Root: `outputs/lpwm_drivor_planning_path_lora_v1/`
- 진행: root의 `navsim_v1/progress.json`, `queue_status.json`
- Gallery: `outputs/lpwm_drivor_planning_path_representation_monitor_v1/index.html`
- 중단: 새 root의 `pause.requested`. 기존 조건의 pause 파일은 그대로 보존한다.
- Source/config279개를 등록했다. 실행 중 수정하지 않고, 후속 변경은 별도 조건/등록으로 처리한다.

## 기동 확인

본학습 PID3186133, 최종평가 queue3186134, 표현 monitor3186135가 실행 중이다.
첫3update에서 양 rank의 모든 경로군 gradient 유한·양수, 전체103,288scene/40,350update 목표 확인.
Monitor 동시 실행 시 GPU0 카드전체 약41.18GB/GPU1 약40.16GB.
증거: `results/lpwm_drivor_planning_path_lora_v1/main_training_start.json`.
학습 중 loss의 몇 step 변화로 planning 개선을 판단하지 않으며 최종 성능은 아직 없다.


## 첫 epoch에서 이전 LoRA 조건과 직접 시각화 비교

사용자 후속 요청에 따라 별도 CPU watcher **3317230**을 추가했다. 본학습·기존 monitor279개 등록 source는 변경하지 않았다.
이전 attention Q/V LoRA의 **정확한1epoch/1614update** 표현과 새 세경로LoRA의 **정확한1epoch/1614update** 표현을 비교한다.
실제process중단시의1615update나나중checkpoint로대체하지않는다.
두조건의공통학습전96×4×64 current particle attributes가 **bitwise 동일**함을 사전검사했다.

- 입력·주행명령·장면·카메라·native particle 번호를고정한다. 같은번호는같은객체identity를보장하지않는다.
- 대표그림: 원영상 / 공통초기 / 이전1epoch / 새1epoch. 별도겹침그림에서 cyan=이전,orange=새모델.
- 점=전체64particle중심, 사각형=초기presence상위16개의고정ID glimpse크기, 점반지름=presence.
- 위치변화는128×128입력pixel단위, 크기는축별절대·상대변화, presence는절대변화량으로집계한다.
- 전체96장면·4카메라·직진/좌회전/우회전/투영박스겹침proxy별통계를저장한다.
- 그림선택은새epoch1결과를보기전에고정: 상황별대표와가림proxy, 상세12장면×4카메라.
  모든96장면·64particle의초기/이전/새조건좌표·크기·presence는CSV에보존한다.
- 이동량이커졌다는사실만으로주행정보보존/PDMS개선을주장하지않는다.

설정: `configs/lpwm_drivor_review/epoch1_lora_scope_comparison.json`.
진입점: `scripts/compare_lpwm_lora_scope_epoch1.py --config configs/lpwm_drivor_review/epoch1_lora_scope_comparison.json --watch`.
상태: `outputs/lpwm_drivor_epoch1_lora_scope_comparison_v1/status.json`.
생성예정: 같은폴더의 `index.html`, `epoch1_comparison.png`, `epoch1_overlays.png`, `summary.json`, `particle_geometry.csv`.
새모델첫epoch가아직끝나지않아최종비교그림은현재없다. 완료된engineeringrenderer이미지는보존하거나실제학습결과로제공하지않았다.
초기동일성·기존평균이동0.3023136787px재현·동일입력비교0·4열렌더링검사통과.
증거: `results/lpwm_drivor_epoch1_lora_scope_comparison_v1/preflight.json`.

## 적용 계층 수와 역할 요약

| 모듈 | LoRA 수 | 역할 |
|---|---:|---|
| prior CNN | Conv7 | 영상patch에서particle의초기기준위치를제안 |
| attribute CNN | Conv7 | 기준위치주변영상으로현재geometry head의입력feature생성;command FiLM적용 |
| xy/scale/presence heads | Linear6 | 현재위치offset·glimpse크기·활성정도출력 |
| foreground appearance CNN/feature head | Conv7+Linear2 | 현재glimpse의외관feature보존 |
| background appearance CNN/head | Conv11+Linear2 | 장면의배경·전역외관정보보존 |
| particle interaction | Conv12+Linear29 | 전역영상context와particle간관계를반영해feature·깊이관련latent등수정 |
| context prior | Linear75 | particle상태·관계에서미래예측에쓰는context분포생성 |
| dynamics projection/attention/FFN | Linear103 | context와현재상태를조건으로미래particle상태를계산하는내부변환 |
| 미래particle 출력heads | Linear12 | 미래위치·크기·presence·깊이관련latent·foreground/background feature출력 |

합계Conv44+Linear229=273adapter,4,683,650학습parameter다. Native CNN/Linear/normalization/embedding/RGBdecoder원래값은고정.
새commandFiLM·temporalprojection·cameraembedding·planner는LoRA가아닌추가모듈전체학습이다.


## 같은 영상에서 명령을 바꿨을 때 — update100 확인

동일영상에서도command4D→FiLM→attribute CNN→xy/scale/presence head 경로로현재geometry가달라질수있다.
LoRA가명령별다른가중치를선택하는것은아니며동일LoRA가명령에따라달라진feature를처리한다.
완료100update진단12장면/24명령교체에서현재중심이동평균0.002942px,최대0.026271px(128²입력)로반응확인이나매우작음.
초기0update명령교체위치변화는0이었다. 크기/presence도구조적으로영향경로가있지만기존intent JSON에는별도변화량집계없음.
실제좌회전에유용한객체배치·정보보존개선은미확정. 이번턴GPU추가실험/학습설정변경없음.

근거: `results/lpwm_drivor_planning_path_lora_v1/intent_sensitivity_update100_review.json`.


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
