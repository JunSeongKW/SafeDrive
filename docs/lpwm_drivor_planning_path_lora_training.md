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
| Worker | rank당 loader2 + online oracle8; update3159까지 oracle4 |
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


## 실제 camera 입력 시각화

현재학습입력시각화완료: `outputs/lpwm_camera_input_visualization_v1/four_camera_original_vs_input.png`.
전/후/좌/우원본1920×1080·실제cache128×128·3배최근접확대를함께표시. 장면c94ee8ade05a5b12/navtrain.
원본4개는바이트그대로복사,입력4개는실제cache에서읽고공식현재BICUBIC전처리와pixel완전일치확인.
각원본/128PNG와`index.html`,`manifest.json`도같은폴더에있음. CPU작업만수행,학습/279source등록불변.

## 2026-10-06 — 첫 epoch 전 300 update의 현재 particle 시각화

사용자 요청으로 latest.pt의 정확한300 update를 별도 snapshot으로 복사해 고정96장면·4카메라에서 현재 particle을 추론했다.
본학습은 계속 진행하며, 이 진단은 optimizer/학습조건/queue를 변경하지 않는다.

| 초기 대비 변화 | 100 update | 300 update |
|---|---:|---:|
| 중심 평균 이동 (128² 입력 pixel) | 0.00986 | 0.03150 |
| 중심 최대 이동 (pixel) | 0.31135 | 0.74734 |
| 가로·세로 축 크기 절대 변화율 평균 | 0.0820% | 0.2369% |
| Presence 절대 변화 평균 | 0.00173 | 0.00599 |

총24,576particle 중 중심이1pixel 이상 이동한 비율은0, 어느 크기 축이5% 이상 바뀐 비율도0이다.
현재geometry는bitwise고정이 아니지만, 아직 육안으로 뚜렷한 재배치라고 보기 어렵다. Planning 성능 개선의 증거로 해석하지 않는다.
직진·좌회전·우회전 각기존첫장면(index2/0/3)을 사용해 결과크기에따른장면선택을피했다.
시각화 열은 입력/초기/300update/겹쳐보기이며, 초기presence상위16개의같은번호박스와전체64중심을표시한다.
사각형은glimpse범위이고GT/detection bbox가아니다. 같은번호는semantic tracking이아니다.128입력3배표시,이동과장없음.

- 이미지: `outputs/lpwm_drivor_intermediate_geometry_update300_v1/before_after_overview.png`.
- 갤러리/각장면4카메라/전체particle CSV: 같은폴더의`index.html` 및 `particle_geometry_changes.csv`.
- 공유수치: `results/lpwm_drivor_planning_path_lora_v1/intermediate_geometry_update300/geometry_report.json`.
- 새읽기전용스크립트: `scripts/visualize_lpwm_intermediate_particle_geometry.py`.
- Checkpoint SHA256: `4c3f082092a9c22d0ec7e4500b3885b8b3831902e981178cde85bb700fee50d6`.

검증: 초기저장표현재현maxerror0, full모델경로와현재encoder추출표현maxerror0, nativefreeze digest동일,
279등록source/config전후동일. 현재표현추론96장면약61초, GPU0최대표본전체41.138GB,진단reserved0.638GB.
최대3GiB진단allocator/진입42GB미만/46.5GB중단guard로48GB카드상한을보호했다.
본학습firstepoch1614자동시각화와이전조건비교watcher는유지한다. 자동채팅push는사용가능한예약도구가없어설정되지않았다.

## 2026-10-06 — 작은 geometry 변화의 원인: warmup과 LoRA 실제 보정량

사용자 질문: 사전학습 가중치가 강해서 LoRA가 힘을 못 쓰는가, 학습량 부족인가?
확인된 사실은 **아직 warmup 초기이며, LoRA는 실제로 업데이트되지만 geometry 보정이 작다**는 것이다.
원인별 인과 기여를 분리한 실험이 아니므로 학습량 부족을 확정하거나 더 학습하면 재배치/성능이 반드시 좋아진다고 주장하지 않는다.

- 공식 convention을 따르는 25epoch recipe의 scheduler horizon은33,225, warmup은3,322update(실제2.058epoch)다.
- 시각화한300update의 다음step LR은1.80616e-5, peak2e-4의9.03%다. 첫300step에 실제사용한LR평균은9.00079e-6(peak4.50%)이다.
- 첫epoch1614 종료시에도LR은peak48.59%로 warmup중이다. 첫epoch는학습경향·연결검사이며LoRA수렴/실패의최종판정시점이아니다.
- 첫epoch설계설명에warmup이2epoch이상지속한다는점을충분히연결하지못했다. 이후판정에서는이조건을명시한다.
- 300update main log의xy/scale/presence LoRA gradient norm은각0.0698354/0.0158284/0.0008330로양수다.
  그룹별parameter수와parameterization이다르므로planner norm과단순비율로영향력/신호부족을판정하지않는다.

별도GPU없이300checkpoint를CPU에서검사했다. shared alias를storage로중복제거한273adapter모두유효DeltaW=B@A가0이아니다.
alpha/rank=1, B=0초기화이므로현재값은실제학습으로생긴보정이다.
전체adapter의100*||DeltaW||F/||W0||F 중앙값0.054164%,범위0.001655~0.467947%.
현재geometry최종Linear의비율: xy0.002749%, scale0.011352%,presence0.001668%.
이는**행렬크기비율**이고activation/output기여비율이아니다. 특히xy/scale최종4행에는mean뿐아니라logvar도들어가며
현재deterministic경로에서logvar행의B는0이다. 작은비율만으로LoRA기능효과가없다고결론내리지않는다.

현재RGBreconstruction loss=0,직접objectGT보조loss=0이다. 복원gradient와planninggradient의경쟁은현재학습의원인이아니다.
사전학습의초기표현/고정base가남아있으며LoRA가그위에보정을배운다. LoRA는rank를제한하지만보정크기를원래가중치의작은비율로강제하지않는다.
참고: LoRA원논문4.1 <https://arxiv.org/html/2106.09685#S4.SS1>.

Planning loss는planner/feature/future를바꾸어도줄어들수있으며particle을특정객체로이동시키는직접목표는아니다.
따라서동일위치에서정보만개선되는가능성과geometry개선이부족한가능성을함께검사해야한다.
LoRA geometry 최종head는출력4에rank4, presence출력1에rank1이어서그출력층에더큰rank를주는해결책은없다.
앞단hidden Linear/CNN과전체경로의rank/최적화효과는별도문제다.

권고: 기존첫epoch진단은유지하고,작은변화만으로실패판정하지않는다. warmup후기존4,000update(2.48epoch)진단까지
geometry/feature/readout/개입추세를함께본다. 그때도geometry가필요한정보를놓치면짧은warmup또는geometry경로LR등을
독립조건으로검토한다. 위치를많이움직이는것자체를최적화하지않는다. 이번조사는학습·LR·loss·queue를변경하지않았다.

실행스크립트: `scripts/audit_lpwm_lora_update_strength.py`.
근거: `results/lpwm_drivor_planning_path_lora_v1/update300_lora_strength_audit.json`.

## 2026-10-06 — 매500 update 변화 추세 자동 진단

사용자 요청에 따라 별도 controller `scripts/monitor_lpwm_particle_trends_every500.py`와
`configs/lpwm_drivor_review/particle_trends_every500.json`을 등록했다.
새 PID568996이 이전 읽기 전용 monitor3186135를 대체한다. 본학습3186133, queue3186134,
첫 epoch 조건 비교3317230 및 기존279개 source/config hash는 유지했다.

- 일정: 500/1000/1500/2000… + 기존100/각epoch경계/최종40350, 중복 제거106시점.
- atomic `latest.pt`의 정확한 update를 hard link로 보존한다. 진단이 메모리를 기다려도 checkpoint를 먼저 확보한다.
  다음 update를 이전 시점으로 잘못 표기하지 않는다. Epoch checkpoint fallback은 쓰기가 끝난 뒤 사용한다.
- 같은96장면×4카메라, 같은 초기particle 번호·top16박스로 초기 대비 및 직전 진단 대비 geometry를 비교한다.
  카메라별/직진·좌·우회전별 분포, 표현 readout·미래·명령 반응·개입 진단은 기존 evaluator를 재사용한다.
- output: `outputs/lpwm_drivor_particle_trends_every500_v1/index.html`, `trend.png`, `trend.csv`,
  `latest_report.txt`, `reports/update_XXXXXX.json`. Gallery는60초새로고침; 자동채팅push는 연결되지 않았다.
- 원래 output의 `update_001614/complete.json`을 그대로 사용하므로 첫epoch Q/V-only vs 세경로 비교와 호환된다.
- GPU0 카드전체42GB미만 진입, allocator4GiB,46.5GB중단guard. 학습48decimalGB 상한 유지.
  진단 실패는 status에 기록하고 학습은 건드리지 않는다. 현재 시점 소요약5분 + 로딩/시각화 여유.

검증: CPU4개 검사 통과(atomic교체 후 원본 snapshot보존,이전checkpoint거절,미래checkpoint오표기거절,
epoch/final일정유지). 기존0/100/500결과로보고서·이미지생성을확인했고새controller생존/다음1000대기확인.
이번500결과는 교체 전 evaluator가 완료한 정확500진단을 재사용했으며 새 학습 결과로 다시 계산하지 않았다.

| 초기 대비 변화 | 100 update | 300 update(현재 geometry만) | 500 update |
|---|---:|---:|---:|
| 중심 평균 이동,128²입력px | 0.00986 | 0.03150 | 0.07453 |
| 크기 축 평균 절대변화율 | 0.0820% | 0.2369% | 0.5592% |
| Presence 평균 절대변화 | 0.001728 | 0.005987 | 0.015460 |

500의 최대 중심 이동은1.60377px. 차량/보행자 영역의 중심비율은초기와같고,
도로proxy는20.0329→20.0164%로 뚜렷한재배치가없다. 전체현재particle정보readout F1은초기.379859→.363157,
appearance-only는.292610→.304012로혼재한다. 같은장면의명령교체시중심평균변화는.018508px이다.
결론: 변화 크기는증가했으나주행효용개선은아직확인되지않았다. 이panel은학습분포진단이며독립navtest가아니다.

15:05KST학습582update,최근100속도26.49초/update. 다음1000도달18:09KST,보고는추가10~20분여유추정.
500간격은현재속도로약3.68시간이다. 공유부하/진단메모리대기에따라변동한다.
등록·상태·500보고/이미지공유본: `results/lpwm_drivor_planning_path_lora_v1/particle_trends_every500/`.

## 2026-10-06 19:20 KST — 1000 update 중간 결과

본학습1090/40350,첫epoch67.53%. 정확1000checkpoint 진단이96장면에서완료됐고다음1500을대기한다.
Train3186133/queue3186134/monitor568996/첫epoch비교3317230생존확인. GPU0·1전체약41.14GB.
등록source/config·학습조건·후속순서는변경하지않았다.

| 초기 대비 변화 | 100 | 500 | 1000 |
|---|---:|---:|---:|
| 중심 평균 이동(px,128²입력) | 0.00986 | 0.07453 | 0.66717 |
| 크기 축 평균 절대변화율(%) | 0.08202 | 0.55918 | 5.83110 |
| Presence 평균 절대변화 | 0.001728 | 0.015460 | 0.185218 |
| 전체 현재particle 판독F1 | 0.368943 | 0.363157 | 0.400721 |

1000에서위치1px초과변화18.71%,한축이상크기5%초과변화52.27%,최대위치6.914px.
직진/좌회전/우회전의평균위치변화는각.707/.630/.645px이다. 같은장면명령교체중심변화도
500의.01851px→1000의.14182px로증가했다. 명령에반응하는경로가있다는증거이며대안명령의정답/유용성검증은아니다.
원래native digest는초기와같고xy/scale/presence LoRA gradient norm은1000에서.44938/.05089/.02789로양수다.

**유용성은혼재한다.** 전체현재particle 판독F1은초기.379859보다.020862높지만,
appearance-only는초기.292610→.247818로낮아졌다. 전체판독의보행자F1은.44395→.52878,
차량F1은.67480→.65187이다. GT ROI geometry-only control .41481보다전체판독도낮다.
따라서전체F1상승을외관의미개선/객체이해개선으로단정하지않는다.
Readout은recording분리된2286fit/688eval object-view이며,LPWM본학습분포에속한다. 독립navtest가아니다.

차량영역중심비율8.88265→8.93148%,보행자1.20036→1.20036%,도로proxy20.03289→19.75329%.
도로presence가중비중은18.40762→8.18433%로크게감소했다. 같은초기도로중심particle집합을CPU분해하면
평균presence가.593817→.195958로줄었다. 전체presence평균도.632211→.474806이다.
Presence는plannerattention이아니며도로proxy는semanticsegmentation이아니다. 변화가도로정보손실인지,
불필요한중복표현을줄이는것인지현재진단으로구분하지못한다. 도로로집중하는현상이라고보고하지않는다.

미래객체변위판독오차는1000에서2초현재표현3.8203m/미래표현3.8403m,
4초현재7.8223m/미래7.8584m이다. 현재대비미래오차차이95%CI는각[-.00592,.06359],[-.00982,.09341]로0포함.
초기미래표현오차3.8933/8.0206m보다낮아졌지만미래표현이현재보다추가정보를주는증거는없다.
0변위baseline3.3339/6.7270m보다도높아,이readout에서미래상태예측성능이충분하다고볼수없다.
12장면에서미래를현재반복으로대체하면ADE가평균.08524m감소하며선택경로oracle점수차이CI는0을포함한다.
교란의분포이탈/소표본한계가있고최종planning기여/모듈제거타당성을확정하지않는다.

양rank100update창평균trainingloss:1~100=28.0283,401~500=14.5374,901~1000=7.7566.
이중trajectoryloss는24.3410→12.7072→5.3521로감소,scoreloss는3.6873→1.8303→2.4045로비단조다.
서로다른minibatch에서LPWM+planner를함께학습한결과이며독립PDMS/LPWM단독미세조정효과가아니다.
1000의LR은peak30.10%,warmup3322가남아있다. 효과부족/성공을지금확정하지않고기존진단일정을유지한다.

최근50/100/200wall속도29.42/31.84/32.07초/update.1500은10/6 22:42~23:00,
첫epoch1614는23:38~10/7 00:01,4000은10/7 19:08~21:16학습도달예상;진단처리10~20분여유별도.
이는현재공유부하외삽이며보장/통계적예측구간이아니다.
시각화/근거: `results/lpwm_drivor_planning_path_lora_v1/intermediate_update1000_20261006/`.

## 2026-10-06 — 중간 공식 PDMS 평가

사용자가PDMS도확인하도록요청해 `scripts/evaluate_lpwm_intermediate_panel_pdms.py`를추가했다.
고정96패널중standardMetricCache가있는동일95장면/24recording을초기·500·1000에서비교했다.
Metric cache없는token `20c5f1c678e7548a`는결과를보기전모든시점에서같이제외했다.
공식NAVSIM v1 `pdm_score`와standardMetricCache,40×.1s simulator를사용한다.
기존12장면의DrivoR trainingoracle값은progress정규화/scorer구현이다르므로이공식95점수와합치지않는다.

| 시점 | PDMS | GT경로 ADE(m) | PDMS 0점 장면 |
|---|---:|---:|---:|
| 학습 전 | 30.4299 | 9.5512 | 46/95 |
| 500 update | 63.4057 | 4.5412 | 16/95 |
| 1000 update | 67.8315 | 3.1777 | 11/95 |

| 상황 | 500 PDMS | 1000 PDMS | 장면수 |
|---|---:|---:|---:|
| 직진 | 63.2425 | 70.7622 | 41 |
| 좌회전 | 63.2932 | 74.3446 | 30 |
| 우회전 | 63.8250 | 54.6835 | 24 |

500→1000+4.42584점,24recording pairedbootstrap5000회/seed71의95%CI[-2.81579,11.01668].
평균은높아졌으나추가개선확정은아니다. 우회전은낮아졌고작은학습패널의장면구성에영향받는다.
초기→1000+37.40157점CI[29.95645,45.10453]은무작위planner부터의공동학습효과이며LPWM단독효과분리는아니다.
1000mean no-at-fault collision.96842/drivable compliance.90526/TTC.90526/comfort.98947/progress.45427.

기존저장particle을모델planner로재생하므로영상encoder재추론을줄였다. 기존12baseline×3의GT ADE일치36검사
최대오차0,정확checkpointSHA/nativefreezehash/원래monitor source등록검사를통과했다.
GPU0진입42GB미만/allocator4GiB/46.5GBguard,실제전체최대41.98079GB. CPU2worker로285건채점성공/실패0.
최초sandbox NVML실패후host에서실행했으며평가2617416은완료됐다. 본학습/queue/500표현monitor는유지한다.

Output `outputs/lpwm_drivor_intermediate_pdms_v1/evaluation_complete.json`,공유결과
`results/lpwm_drivor_planning_path_lora_v1/intermediate_pdms_95_v1/summary.json`,CSV,`pdms_progress.png`.
이것은학습분포중간진단이며fullnavtest가아니다. 최종25epoch후fullnavtest평가일정은기존queue에있다.
이번95PDMS는일회평가다. 이후확인시별도output에 `--updates 1500` 등으로실행할수있다.

## 2026-10-06 23:29 KST — 첫epoch완료 중간결과

첫epoch의exact1614 checkpoint가23:00에저장됐고현재1682update/epoch2다.
CPU예약비교가완료돼old Q/V-only와새geometry·appearance·future LoRA를같은96장면/1614update에서비교했다.

| 초기 대비 변화 | 이전 Q/V LoRA | 새 세경로 LoRA |
|---|---:|---:|
| 평균 중심 이동(px) | 0.3023 | 1.2361 |
| 평균 크기축 절대변화율(%) | 2.0203 | 7.7142 |
| Presence 평균 절대변화 | 0.04613 | 0.22253 |

새particle의42.12%는1px넘게이동했고67.43%는한축이상크기가5%넘게변했다.
Native original digest는같다.전체current판독F1은초기.37986→.42052, old.37299보다높으나
appearance-only는초기.29261/old.31081보다낮은.26691이다.Geometry변화량으로효용을판정하지않는다.
같은장면명령교체중심평균변화.25580px, 도로presence가중비중은초기18.4076→7.2731%다.

미래변위readout은2초현재3.90490/미래3.83489m, 4초현재8.01687/미래7.83373m.
현재대비미래오차차이95%CI가2초[-.09651,-.03744], 4초[-.34250,-.05745]로0을포함하지않는다.
6recording/622·554object-view평가에서의예비신호이며navtest가아니다.0변위baseline3.33391/6.72696m보다오차가높다.
미래표현의추가정보가일부읽히기시작했지만실제planning기여/충분한미래예측성능은아직미검증이다.

새체크포인트의공식PDMS도동일95장면/24recording에서추가평가했다.

| Update | PDMS | Expert ADE(m) | Zero score장면 |
|---|---:|---:|---:|
| 500 | 63.4057 | 4.5412 | 16 |
| 1000 | 67.8315 | 3.1777 | 11 |
| 1500 | 63.5197 | 2.0007 | 24 |
| 1614(epoch1) | 67.9864 | 2.1082 | 17 |

1000→1614의차이는+0.15491점, recordingpairedbootstrap5000회/seed71 CI[-7.14543,7.16457]로거의같은수준이다.
도로준수는.90526→.84211, 좌회전PDMS74.3446→63.7964로낮아졌다.
직진70.7622→74.5582, 우회전54.6835→61.9971은높아졌다.Planning개선은상황별/시점별로비단조다.
PDMS는학습분포진단이며LPWM+planner전체효과, LPWM단독효과분리/최종navtest결과가아니다.

실행 `outputs/lpwm_drivor_intermediate_pdms_epoch1_v1/`, 190건성공/실패0, 원래12baseline×2재생ADE24검사오차0.
평가GPU전체최대41.72913GB, 등록source/config/nativefreeze불변.기존학습/queue/정기monitor유지.
2000은10/7 01:41전후, 3322warmup은10:49전후, 4000은15:31전후도달예상/진단여유별도.
첫epoch도warmup중이므로기존계획대로추세를본다.고해상도/새loss/DrivoR대조실험은추가등록하지않았다.
결과 `results/lpwm_drivor_planning_path_lora_v1/epoch1_intermediate_report_v1/report.json`, PDMS그래프, 동일epoch비교PNG.

## 2026-10-07 09:18 KST — 3,000 update 중간 결과·25 epoch ETA·포화 판단

본학습3,020/40,350update(7.4858%), 1.8711epoch. Train3186133/queue3186134/monitor568996 host생존.
양rank3,020행 비유한loss0, native/실행digest각1개. Update3,000의14LoRA경로와plannergradient유한·양수.
본학습batch16/누적2/effective64·loader2/oracle4·source/config/loss/LR/queue변경없음.
추가평가2301619: exact2,000/2,500/3,000 × 동일95장면 공식NAVSIM v1 PDMS 완료.
285성공/실패0, 기존baseline12×3 ADE재생검사36개최대오차0, GPU전체최대41.72913GB/nativefreeze유지.
PDMS70.16709/72.92701/74.90627. 첫epoch대비3,000 +6.91986점CI[1.02676,12.38134].
1000대비+7.07477 CI[-.78508,14.36931], 2500대비+1.97926 CI[-5.66113,9.02419]; 비교시점별차이숨기지않음.
직진epoch1 74.5582→84.4096/좌63.7964→68.2930/우61.9971→66.9380. 도로준수.84211→.88421.
공식채점방법이지만학습분포패널이고navtest아님; 공동학습효과이며LPWM단독미세조정이득분리아님.

Geometry초기대비2,000 1.385px/10.879% →2,500 2.163px/13.527% →3,000 1.813px/10.957%.
3,000의63.29%가1px초과이동,86.19%는한축크기5%초과. 명령교체중심.399664px.
WholecurrentF1 .419269/.386249/.393593, appearance .273758/.352573/.356225. 초기whole.379859/appearance.292610.
차량중심8.88265→8.79720%,보행자1.20036→1.21257%,roadproxy중심20.03289→20.68257%.
Roadpresence가중비중18.40762→5.34835%지만presence는plannerattention이아니고feature는그대로입력된다.
미래readout첫epoch의예비이득지속안됨:3,000 2초current3.97808/predicted4.01345m(차이+.03536,CI[.00270,.10121]),
4초8.25388/8.21429m(차이-.03959,CI[-.26095,.23276]). 절대오차0변위baseline보다높음.
Future를current반복으로교체시12장면trainingoracle선택점수+.007808CI[.002938,.014411]; 공식PDMS나인과효과로해석안함.

사용자후속질문25epoch잔여와saturation방법답변:23.13epoch/약11.6–12.1일,10월19일전후학습종료(평가별도).
최근50/100/200 wall27.93/27.32/26.90초. Epoch2경계10:51–10:55, warmup3,322 11:33–11:39,4,000 16:37–16:55예상.
Warmup전이면서PDMS상승이므로현재포화근거없음. 독립개발/실질최소개선delta/3–5epoch patience/LR감소구간과안전·표현진단권고.
현재navtrain+navval모두학습으로독립validation없음. 95훈련진단을일반화포화근거로쓰지않고navtest반복tuning도하지않음.
자동조기중단/새학습조건미등록. DrivoR25epoch비교는같은학습예산유지;조기종료실험은동일budget/중단규칙필요.
결과/생성스크립트/CSV/PDMS·future·particle PNG: results/lpwm_drivor_planning_path_lora_v1/intermediate_update3000_20261007/.

### Saturation 판단 제안 — 현재 실행에는 적용하지 않음

학습loss 감소의 정체만으로 중단하지 않는다. 미리 분리한 recording 개발집합에서 같은 evaluator로 epoch별평가한다.
Warmup종료후 학습률감소구간을 포함해 관찰하고, 예를들어3–5epoch 동안 best대비실질개선delta=.5 PDMS점 미만이며
paired recording CI도큰추가이득을지지하지않는지를검토한다. CI가0포함하는것은단지정밀도부족일수있다.
엄밀한비열등성/포화판단에는추가이득CI상한이등록delta보다작은지확인하는등표본불확실성을반영해야한다.
평균PDMS정체에도좌회전/가림/충돌/도로준수나미래정보보존이개선되면가설에대한추가정보가있을수있다.
반대로전체PDMS상승은미래표현의실질기여를보장하지않는다. Frozen/current-only 등의동일planner통제가후속으로필요하다.
현재모든navtrain+navval이학습됐으므로이run에독립개발검증이있다고주장하지않는다.
25epochDrivoR비교는현재계획대로동일학습예산을유지하고, 중단규칙을바꾸려면별도통제조건으로등록한다.

## 2026-10-07 09:30 KST — Particle 겹침·이동 화살표와 향후 자동 출력

사용자가3,000 비교그림에학습전후particle을겹친그림도항상함께요청했다.
새CPU전용 scripts/publish_lpwm_particle_geometry_overlays.py를추가했다. 기존학습/monitor/source/config는불변.
입력은완료diagnostic의particle_attributes.npy/panel/images/geometry_report; 새학습·GPU모델추론없음.
직진2/좌회전0/우회전3 front카메라에서현재64particle을같은번호로대응했다.
청록빈점·점선박스가전,주황점·실선박스가후. 흰화살표의끝점은실제변경된중심이다.
Glimpse박스는초기presence상위16개를고정. 겹침dot반경은고정하여위치변화를비교하며presence표시와구분.
512tile또는384tile로영상·좌표를같이확대했으며displacement자체추가증폭1.0.
기존3열PNG를복사하고4열겹침을추가해사용자가본기존그림과직접대응되도록했다.
3,000처음PNG둘을시각적으로검사한후CPUwatchpublisher2507743 실행,100/500/1000/1500/1614/2000/2500/3000 자동생성확인.
Sandboxbackground시도PID5는작업종료와함께끝났으므로host에서재실행했다. GlobalPID5를signal하지않음.
초기preview/source도별도보존,실제publisher source SHA b97ab198b03057788f51108d77de94c6123dc75c4b67a2469d6047f405c57f85.
향후매500/epoch기존diagnostic완료마다새시각화가자동추가된다. Watcher final40,350처리후종료/자체stop.requested지원.
Root outputs/lpwm_drivor_particle_geometry_overlays_v1/의index.html에기존비교·겹침·4열링크.
현재PNG/overlay metadata는results/lpwm_drivor_planning_path_lora_v1/intermediate_update3000_20261007/에추가했고
launch/status는results/lpwm_drivor_planning_path_lora_v1/particle_geometry_overlay_visualization/에보존했다.
Geometry변화는planningbenefit/attention이나semantic tracking의증거가아니다. 앞으로전후와겹침을함께보고한다.

실제PNG검사: 통합그림의header아래기존3열pixel이원본과완전히일치. 8시점출력과CPUheartbeat, source/launch/metadata hash일치를확인했다.

## 2026-10-07 09:52 KST — 차량 밀집 장면3개의 전후 particle 비교·겹침

사용자가기존scene보다주변차량이많은다른이미지도요청했다.기존96고정패널에서front투영vehicle GT수로screening,
차량박스최소4×3input pixel로큰객체수도계산했고rawcontactsheet를육안검토했다.
실제차량밀집교통으로scene41(c3ba85bd489a5e47)/21(2516f0fa67f9535f)/27(2b2c80d7c63e5ea4)를선정.
각기다른recording의도심버스·승용차교통/고가도로아래대기/근거리차량대기. 모두straight유형이며유형대표분해는아님.
투영차량GT31/15/13은가림포함수로완전히보이는차량수·모델검출수로해석하지않는다.
선택기준은GT밀도와영상의traffic내용이며particle변화량/PDMS로선정하지않음.
새script scripts/visualize_lpwm_vehicle_rich_particle_changes.py로exact3,000savedattributes를읽어CPU렌더.
청록전/주황후/동일64particle/초기presence상위16glimpse박스/실제이동화살표·고정overlaydotradius를유지.
이미지encoder/planner재추론·학습·GTloss추가없음. 기존source/config/queue/monitor/overlaypublisher변경없음.
Front평균중심이동scene41 2.212736px /21 2.484261 /27 1.126128;사례효용증거로해석안함.
원본1920×1080 RGB를BICUBIC128²로resize한값이3장면모두실제cache input과bitwise동일함을검사했다.
초기BILINEAR가정에서일치검사가실패했고실제prepare_lpwm_drivor_joint_data.py의BICUBIC으로정정해통과.
완료results/lpwm_drivor_planning_path_lora_v1/vehicle_rich_particle_visualization_update3000/에는
vehicle_rich_particle_overlays.png, vehicle_rich_before_after_with_overlay.png, vehicle_rich_original_camera_images.png,
scene041/021/027_four_camera_overlays.png, selection_and_geometry_report.json(source/checkpoint/panel/attributes hash).
출력PNG3개를육안검토했다. 향후같은3장면은 --updates와새 --output을주어재생성하고전후+겹침을함께제공한다.

## 2026-10-07 10:01 KST — 현재 particle 시각화 해석

이 그림은 같은 현재 이미지에서 학습 전후 모델이 추정한 particle의 2D 중심, glimpse 크기, 일부 presence를 보여준다.
현재프레임만 표시하며 dynamics의 미래particle이나 실제차량의시간상움직임을그린것은아니다.

| 표시 | 실제 코드의 의미 | 해석 범위 |
|---|---|---|
| 점 위치 | particle의학습된영상좌표 / glimpse중심 | 객체 중심정답이나semantic object ID로확정하지않음 |
| 전후별그림의초기상위16개점크기 | 표시반경2+4×presence, presence는0–1 | obj_on활성도이며planner중요도/검출confidence아님 |
| 전후별그림의나머지48개점크기 | 고정반경2px | presence가낮다는뜻아님 |
| 겹침그림점크기 | 청록빈점반경4px /주황실점반경2px로고정 | 크기차이에서presence/중요도변화를읽지않음 |
| 박스위치·폭·높이 | 중심±학습된scale/2의glimpse범위 | particle이외관정보를얻는영상crop범위; 객체GT/detection bbox아님 |
| 박스개수16개 | 각scene/camera별초기presence상위16개번호고정 | 학습후새로운top16을선택한그림아님;나머지48개도내부scale존재 |
| 색·점선·실선 | 청록/점선은학습전,주황/실선은학습후 | 객체종류/안전도/attention을표시하지않음 |
| 흰화살표 | 같은particle번호의학습전후중심차이 | 실제좌표끝점/이동추가증폭없음.물체의물리적속도나미래궤적이아님 |

Presence는LPWM의obj_on latent로원래foreground활성도/alpha compositing에관여한다.
공식RGBdecoder는obj_on으로particle alpha를곱하고depth로혼합하며,appearanceencoder도obj_on을받으면CNNfeature를곱한다.
현재planner입력은64particle의현재+미래14Dattributes전체를projection/pooling하므로작은점/낮은presence만으로삭제되지는않는다.
같은차에여러particle이걸칠수있고한glimpse에차량·도로·배경이함께있을수있다.
큰박스는더넓은범위,작은박스는더좁은범위를뜻할뿐정보량/중요도/정확도를보장하지않는다.
Particleinteraction과context로다른particle정보가추가되므로박스가표현의유일한정보출처도아니다.
차량쪽중심이동은기하학적변화의증거이며객체정보보존·미래예측·planning기여는readout/개입/PDMS로따로검증한다.
위치·크기가비슷해도appearance/context/dynamics feature는달라질수있다. 현재이미지에서미래예측성능을판정하지않는다.

확인코드: scripts/visualize_lpwm_planning_path_geometry.py:45(draw_particles),
scripts/publish_lpwm_particle_geometry_overlays.py:64(draw_geometry_overlay),
reference_repositories/LPWM/modules/modules.py:3010(appearancecrop), :5344(alpha obj_on),
src/planning_aware_future_prediction/object_centric/lpwm_drivor_joint.py:108(전체particle projection/pooling).
이번작업은코드확인·해석설명과문서기록이며학습/시각화source/config/실행중publisher변경없음.


## 2026-10-07 CPU oracle 병렬성 변경

원래 모델·loss·config·279개 등록 소스는 유지하고 실행 override만 새로 등록했다.
`configs/lpwm_drivor_planning_path_lora/execution_batch16_loader2_oracle8.json`은
**GPU당 microbatch16 / 누적2 / 유효64 / loader2**를 유지하며 oracle만 rank당4→8로 변경한다.
새 실행 source283개는 `outputs/lpwm_drivor_planning_path_lora_v1/oracle8_execution_registration.json`에 등록했다.

CPU-only 측정은 현재 두 rank의 각16장면·실제 생성 후보를 고정하고 두 서비스를 동시에 실행했다.
4→8→6→4 순서, 시작 요청 제외 각5회이며 모든7개 subscore가 exact equality를 통과했다.
이는 고정 요청과 warm metric cache에서의 비교로 전체 학습 속도 측정을 대신하지 않는다.
[Oracle 실측](../results/lpwm_drivor_planning_path_lora_v1/throughput_20261007/oracle_parallelism_report.json).

3,159update 경계에서 full model/AdamW/scheduler/두 rank RNG를 저장하고 재개했다.
796개 optimizer state의 step은 전부3,159이며 native digest는 기존과 동일하다.
중단전 `latest.pt`의 inode는 별도 hard link로 보존했고 초기화나 학습 update를 되돌리지 않았다.
재개 시각의 loader iterator 생성은 bitwise 동일한 dropout 난수열을 보장하지 않는다.
DataLoader/sampler의 같은 epoch·같은 next_update를 복원해 이전 장면에서 optimizer update를 다시 하지 않는다.

본학습PID3144180, 후속queue3144181, 매500update monitor3144182로 교체했다.
겹침 publisher2507743은 그대로 유지한다. Queue는 기존 v1/fullnavtest/v2/EPDMS 순서이며 v2에도 oracle8 실행을 적용한다.
모든 GPU 입장 예산과 매update의 카드전체48decimalGB 검사를 유지한다.

준비 controller의 첫 경로표기 검사 오류는 pause 전에 발생해 기존 학습에 영향이 없었다.
실패한 소스·registration·이유를 `outputs/lpwm_drivor_planning_path_lora_v1/oracle8_resume_20261007/`에 보존하고
수정 이유와 이전hash를 실행 registration amendment에 기록한 후 별도checked 출력에서 재개했다.
기존 실행중 연구 소스·설정·등록은 변경하지 않았다.

### 실제 본학습 속도 확인

| 측정 | 변경 전 oracle4 | 변경 후 oracle8 |
|---|---:|---:|
| Update 구간 / rank별 표본 | 3038–3137 /100개 | 3161–3170 /10개 |
| Update 간 wall time | 26.318초 | 23.502초 |
| 두 rank 중 느린 update 시간 평균 | 26.257초 | 23.733초 |
| rank0 oracle 시간 평균 | 4.783초 | 3.245초 |
| 카드 전체 최대 사용량 | 40.325GB | 40.162GB |

첫 재개 update3160의 CUDA/oracle 초기화와 이전 loader 건너뛰기53초는 안정 속도 비교에서 제외했다.
짧은 wall time 창에서 **10.699% 시간 감소**를 관찰했다. 같은 속도가 유지되면 update3170 이후
잔여37,180update는 약10.11일, 변경 전 속도 대비29.08시간 단축으로 추정된다.
이는 V1 학습만의 외삽이며 10월17일 오후 전후로 계산되지만 이후 검증·V2 학습을 포함하지 않는다.
변경 후 표본10개와 공유 서버·장면 비용 차이 때문에 장기 속도 개선량을 확정하지 않는다.
GPU 역전파 시간이 함께 변했으므로 관측한 전체 개선량을 전부 worker 변경의 인과 효과로 간주하지 않는다.

[학습 속도·재개·ETA 보고](../results/lpwm_drivor_planning_path_lora_v1/throughput_20261007/training_speed_comparison.json).
