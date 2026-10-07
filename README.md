# Planning-Aware Future Prediction

**2026-10-07 10:42 KST 중간점검 — 학습3,213/40,350, 새 검증은 아직3000까지.**
Epoch2 종료까지15update, 약10:48KST 학습경계3228 도달/진단처리시간별도.
Oracle8 재개후53steadyupdate wall23.350초, 최대40.162GB, 양rank전체로그NaNloss0/3200모든gradient그룹유한·양수.
기존279/new283sourcehash불변·queue3144181/500monitor3144182 fresh상태. 학습/queue/monitor 변경없음.
완료된최신95학습장면 PDMS3000=74.9063, geometry1.813px/10.957%, 미래표현추가효용미확인;현재3213성능값으로부르지않음.
상태보고 results/lpwm_drivor_planning_path_lora_v1/intermediate_status_20261007_1040.json.

**2026-10-07 10:25 KST — 현재 학습 CPU oracle 병렬성 증설.**
GPU0·1 microbatch16×누적2×2=유효64와 loader2 유지, oracle만 rank당4→8로 변경했다.
3159 fullstate 저장·복원, 현재3170. 본학습 안정10update wall26.32→23.50초(-10.7%), 최대40.16GB.
짧은 순차 측정이며 장기 속도 보장은 아니다. 남은 V1 학습 약10.1일/10월17–18일, 후속평가·V2별도.
Train3144180/queue3144181/500monitor3144182, overlay2507743. 기존279 연구소스 불변, 새실행283source등록.
[실측·재개 보고](results/lpwm_drivor_planning_path_lora_v1/throughput_20261007/training_speed_comparison.json).

**2026-10-07 09:52 KST — 차량이 밀집한 장면의 particle 시각화 추가.**
도심 교통·고가도로 아래·가까운 차량 대기 장면3개, 같은3,000update.
[겹침·이동 화살표](results/lpwm_drivor_planning_path_lora_v1/vehicle_rich_particle_visualization_update3000/vehicle_rich_particle_overlays.png) ·
[입력·학습 전·후·겹침](results/lpwm_drivor_planning_path_lora_v1/vehicle_rich_particle_visualization_update3000/vehicle_rich_before_after_with_overlay.png) ·
[원본 카메라](results/lpwm_drivor_planning_path_lora_v1/vehicle_rich_particle_visualization_update3000/vehicle_rich_original_camera_images.png).
차량 투영 GT 수와 육안 확인으로 장면41/21/27을 선택했으며, particle 변화나 PDMS를 선택 기준으로 사용하지 않았다.


**2026-10-07 09:30 KST — Particle 겹침 시각화 추가.**
[학습 전·3,000 update 겹침](results/lpwm_drivor_planning_path_lora_v1/intermediate_update3000_20261007/particles_overlay_before_vs_update3000.png) ·
[원본·전·후·겹침 4열](results/lpwm_drivor_planning_path_lora_v1/intermediate_update3000_20261007/particles_before_after_with_overlay_update3000.png).
청록 점선/빈 점은 학습 전, 주황 실선/점은 학습 후, 흰 화살표는 같은 particle 번호의 실제 이동.
[자동 갤러리](outputs/lpwm_drivor_particle_geometry_overlays_v1/index.html), CPU publisher2507743.
앞으로 particle 이미지는 **기존 전·후 비교와 겹침 이미지를 함께** 제공한다.


**2026-10-07 09:18 KST — 3,000 update 중간 점검 완료.**
본학습 3,020/40,350 update, epoch 2. 정확한 2,000·2,500·3,000 update 표현·공식 PDMS 진단 완료.
동일 95학습장면 PDMS **70.17 → 72.93 → 74.91**; 첫 epoch 67.99 대비 +6.92점(CI[1.03,12.38]).
Particle 중심 평균 이동 1.813px / 크기 변화 10.957%, current 전체 판독 F1 .3936 / appearance .3562.
미래 정보 판독의 첫 epoch 이득은 유지되지 않았다. 독립 navtest / LPWM 단독 효과는 미검증.
[누적 보고서·그래프](results/lpwm_drivor_planning_path_lora_v1/intermediate_update3000_20261007/report.json).
25 epoch까지 약 23.13 epoch / 11.6–12.1일 남음, 10월 19일 전후 학습 종료 예상; 평가 시간 별도.
기존 본학습·대기열·매500 표현 진단 유지. 아래 날짜가 이전인 문장은 실행 이력이다.


**2026-10-06 23:29 KST — 첫 epoch 완료·이전 조건 비교 및 중간 PDMS 완료.**
현재1682update/epoch2, 첫epoch checkpoint23:00저장. [동일1614update particle 비교](outputs/lpwm_drivor_epoch1_lora_scope_comparison_v1/epoch1_comparison.png).
95학습장면 PDMS:1000=67.83 /1500=63.52 /1614=67.99. 최신보고 `results/lpwm_drivor_planning_path_lora_v1/epoch1_intermediate_report_v1/report.json`.
새geometry평균이동1.236px/크기7.714%,old Q/V는.302px/2.020%. Nativefreeze유지,본학습과정기진단계속.

**2026-10-06 15:05 KST — 500 update 간격 자동 표현 진단 실행 중.**
[누적 이미지·보고서](outputs/lpwm_drivor_particle_trends_every500_v1/index.html), 새 monitor PID568996.
기존 monitor3186135만 교체했으며 본학습3186133·queue3186134·첫 epoch 비교3317230은 유지한다.
정확한500진단 완료: 중심 평균0.07453px, 크기 평균0.5592% 변화. 주행 효용 개선은 아직 미확인.
이후1000/1500/2000… 및 기존 epoch 경계 진단. 자동 채팅 푸시는 연결되지 않았으며 로컬 결과가 갱신된다.

**실제 입력 이미지:** [원본1920×1080 / 학습입력128×128 비교](outputs/lpwm_camera_input_visualization_v1/four_camera_original_vs_input.png) · [4카메라 원본 갤러리](outputs/lpwm_camera_input_visualization_v1/index.html).

**2026-10-06 후속 요청: 첫 epoch에서 이전 LoRA 조건과 직접 시각화 비교를 예약했다.**
별도CPU watcher3317230, `outputs/lpwm_drivor_epoch1_lora_scope_comparison_v1/status.json`.
공통초기 / 이전Q·V LoRA1614 / 새세경로LoRA1614를 같은96장면·카메라·명령·particle번호로비교한다.
현재새epoch1을기다리는중이며최종비교PNG는아직없다. 기존25epoch학습과monitor는지속한다.

**2026-10-06 최신 사용자 승인: geometry·appearance·future 세 경로 LoRA 본학습 시작.**
사용자가 대상 계층을 선택하고 실행을 요청했으므로, 이전 계층 선택 대기는 이 새 조건에 한해 해제됐다.
[설계·gradient 검사·실행 설정](docs/lpwm_drivor_planning_path_lora_training.md).
Linear229 + Conv44 / LPWM LoRA4,683,650개, native109,545,263개·buffer고정, 공식 DrivoR planner 전체학습.
새 공개 초기화 / navtrain+navval103,288 /25epoch40,350update / GPU0·1 /batch16×누적2×2=유효64.
Loader2·oracle4/rank, 총48decimalGB 상한. 실제DDP2update 최대40.16GB·모든273adapter planning gradient 검사통과.
본학습3186133 /후속queue3186134 /monitor3186135. Root `outputs/lpwm_drivor_planning_path_lora_v1/`.
고정96장면 전·중·후 geometry/readout/intent/개입검사; DrivoR 비교는25epoch 이후.
새source/config279개 등록불변. 이전attention1614/1615와geometry-only hold는 보존하고 재개하지 않는다.
아래 새학습보류/과거실행중 문장은 이전 이력이다.

**최신: 첫 epoch 진단 완료·학습 대기.** [LoRA 적용 후보 전체 설명](docs/lpwm_lora_layer_catalog.md).
사용자가 적용 계층을 선택할 때까지 새 본학습을 시작하지 않는다. 기존1614update 상태와 학습 전후 시각화를 보존했다.
Head LoRA는 코드·연결·DDP 검사만 준비됐으며 본학습 미실행이다. DrivoR 비교는25epoch 후로 미룬다.

**현재 particle 위치와 LoRA:** [원인 감사 결과](docs/lpwm_drivor_lora_geometry_audit.md).
Q/V LoRA는 학습되지만 현재 좌표 생성 뒤에 있다. 현재 위치 변화는 command FiLM에 의존하며,
현재 좌표를 직접 적응시키는 geometry 경로 LoRA는 아직 적용하지 않았다.

**2026-10-06 학습 중 표현 진단·공정 비교 점검:**
[검증 방법과 비교 한계](docs/lpwm_drivor_representation_and_fair_comparison.md).
기존 LoRA 본 학습은 유지하고, 별도 읽기 전용 monitor가 고정96장면의 분포·정보 readout·의도 경로·planning 개입을 기록한다.
갤러리 `outputs/lpwm_drivor_representation_monitor_v1/index.html`; 학습 분포 진단이며 독립 navtest 성능이 아니다.


**최신 실행(2026-10-06): 공개 LPWM 원래 가중치 고정 + DrivoR 방식 Q/V LoRA.**
Native full run은 35 update에서 보존·중단했고 공개 LPWM으로 새 LoRA 학습을 시작했다.
LoRA rank32/scale1/42 Q·V projection, planner 전체 학습. 상세 [LoRA·ego 입력·실행 실측](docs/lpwm_drivor_lora_training.md).
현재 `outputs/lpwm_drivor_lora_v1/`, train2994997/queue2994998. 배치16×누적2×GPU2=유효64,
loader2/oracle4 per rank, 카드전체48decimalGB. Batch24 및 worker증설까지 실측 후 배치16을선택했다.
공식planner ego11D 경로는동일, LPWM command4D FiLM은추가경로다. 아래 native-weight 실행은이전이력이다.


**최신 실행(2026-10-06): 공개 LPWM + 공식 DrivoR planner E2E 본 학습 시작.**
사용자가 Stage1/2를 합친 joint planning 학습 및 DrivoR와 같은 공식 PDMS/EPDMS 조건을 새로 승인했다.
GPU0·1/batch8×누적4×2=유효64/seed2. v1 공식 navtrain85,109+navval18,179=103,288장면/25epoch40,350update.
진입점 [설계·검증·비교 한계](docs/lpwm_drivor_joint_training.md), `configs/lpwm_drivor_joint/`.
본 학습 PID2788260, 후속queue2839064. v1완료→full navtest→별도 v2 train-only10epoch→warmup/navhard EPDMS.
Backend/원본loss는 DrivoR, perception은 공개LPWM의활성원래가중치를업데이트; DINOv2+LoRA와미세조정/해상도는같지않다.
아래 DrivoR/추가epoch/navtest 보류 및 모든작업종료 문장은 과거 승인·완료 이력이다.

**최신 완료(2026-10-05 23:11 KST): NAVSIM Stage1의 planning 효과 확인.**
공개LPWM고정+planner78.9161 → NAVSIM적응LPWM고정+동일planner82.5238, PDMS+3.6077점(CI[+1.5065,+5.8539]).
동일seed47/planner1epoch4707/GPU당batch8×2/75,297train; 내부개발1024중1021유효PDM/40recording,navtest아님.
최종평가·paired비교완료,등록작업모두종료. 상세 `docs/lpwm_planning_experiment.md` 최상단과
`results/lpwm_stage1_effect_v1/queue/stage1_effect_summary.json`. 아래 실행중표시는과거이력이다.

**최신(2026-10-05): NAVSIM Stage1 효과를 공개 LPWM 고정 대조군 하나로 검증한다.**
완료된 Stage1-LPWM 고정+planner 결과82.5238을 재사용하고 공개 Sketchy LPWM+동일planner만 새로 학습한다.
사용자 정정에 따라 batch8/GPU×2·누적1=유효16,seed47/75,297장면/1epoch4707update를 유지한다.
두 조건 재학습·배치 확대 안은 실행하지 않았다. 기존 다섯 실험도 유효16/4707은 유지했으며
microbatch/dropout/SSL sampling 및 full schedule 차이는 [배치 감사와 해석](docs/lpwm_planning_experiment.md)에 기록했다.
진입점 `configs/lpwm_planning/stage1_effect_v1/queue.json`, `scripts/queue_lpwm_stage1_effect.py`.
상태 `outputs/lpwm_stage1_effect_v1/queue/queue_state.json`. 현재 상태는 HANDOFF 최신절을 우선한다.
아래 기존 네방법·frozen queue 실행중 문장은 완료 전 보존 이력이다.

**최신(2026-10-05): 현재 네 방법 비교 뒤 frozen-LPWM planner 대조군 자동 실행을 추가했다.**
기존 v4 queue421603/full 학습은 유지하며, 새 CPU queue871940이 전체 학습·검증·집계를 기다린다.
진입점 `configs/lpwm_planning/frozen_control_v1/queue.json`, 상태 `outputs/lpwm_frozen_control_v1/queue/queue_state.json`.
Stage1 LPWM 전체 가중치·buffer·encoder 명령 FiLM 고정, 기존 planner만 seed47/75,297장면/1epoch 학습한다.
GPU0·1/카드전체48GB, batch8우선·유효planning16/world8, 동일1024planning/256world검증.
실제 체크포인트 CPU freeze/gradient/초기동일성검사와7개실행조건검사통과. GPU감사·profile은선행실행완료후자동수행한다.
새source7개/config4개도hash등록됐으므로기동중수정금지. [설계와한계](docs/lpwm_planning_experiment.md).

**최신(2026-10-05): 사용자 요청으로 모든 후속 학습에 batch8 우선 정책 적용.**
LoRA8 실제profile최대47.60GB통과,4422→4448 본학습재개확인(약3.29s/update).
현재 진입점은 `configs/lpwm_planning/card_budget_measured_v4/queue.json`과
`scripts/queue_lpwm_measured_card_budget.py`(queue421603). GPU0·1 각각 전체48decimalGB(다른 사용자 포함)를 감시한다.
이전 고정allocator23.2GiB/allocated22.75GiB/free3GiB는 이번 승인으로 대체됐다.
현재 NVML점유에서 우리allocator를 제외한 사용량과 workspace192MiB/반올림64MiB를 뺀 잔여예산을 적용한다.
LoRA·Adapter·full 모두batch8 먼저실측,48GB내통과하면8채택; 메모리실패시에만4/2fallback.
LoRA4422 model+AdamW226state를 별도보존하고4423부터재개. Partial완료결과재사용,full원본2095재개유지.
유효planning16/SSL8/worker0/기존loss·LR·데이터·4707목표유지. Profile학습결과는본학습에사용하지않는다.
새queue등록후source/config변경금지, 이전queue1902774는superseded이며재기동금지.
이전 48GB v1의 고정프로세스예산/배치선택 설명은 보존이력이다.
CUDA expandable_segments를 사용한다. v2/v3 profile 실패와원본4422를보존; 이전시험queue408762/413191은종료됐다.


**최신(2026-10-05): 사용자 요청으로48GB 실행 설정을 별도 등록했다.**
`configs/lpwm_planning/execution_48gb_v1/queue.json`에 네 가지 미세조정 순서를 유지했다.
부분학습1576 checkpoint+AdamW를 보존하고 배치4/8 실측 후1577부터 이어간다.
GPU별 전체48decimalGB/여유3GiB, 유효batch16/SSL8,worker0. 실측 결과와 실행 상태는
`outputs/lpwm_48gb_planning_v1/queue/queue_state.json` 및 방법별`*_execution_selection.json`.
이전 두 queue는 종료·대체됐으며 아래 PID/46GB는 이전 이력이다.


**최신(2026-10-04 23:53 KST): 네 가지 미세조정 순차 비교를 등록했고 부분 학습은 계속 실행 중이다.**
일부 계층 → 검증 → LoRA → 검증 → Adapter → 검증 → 이전 전체 미세조정 재개 → 검증.
직접 객체 GT 보조 loss는 후순위이며 자동 실행하지 않는다. 전체 조건은 기존 model+AdamW update2095를
이어받아 총4707에서1epoch 경향 검증하며 원본 보존. 이전20epoch LR 스케줄/conv_in 명령 입력도 유지한다.
현재 torchrun1602577, Partial/LoRA queue1675463, 후속 queue1709131(waiting).
진입점 `configs/lpwm_planning/four_method_sequence_v1.json`; 상태 `outputs/lpwm_four_method_queue_v1/queue_state.json`.
[정확한 모듈·gradient·재개·평가 범위](docs/lpwm_planning_experiment.md).
아래 과거 두 조건/새 full 초기화 계획은 이 최신 지시로 대체됐다.

**최신(2026-10-04): 객체GT보조학습을후순위로옮기고 일부계층 미세조정 대 LoRA를 먼저 비교.**
현재partial GPU학습은이어받고, 두방법모두GT보조OFF/전체navtrain학습분할75,297개/1epoch.
새queue `scripts/queue_lpwm_adaptation_methods.py`, 설정 `adaptation_method_comparison_v1.json`,
상태 `outputs/lpwm_adaptation_method_comparison_v1/queue_state.json`.
Partial학습→검증→LoRA batch4/안전할경우8 profile→학습→검증·paired비교.
LoRA rank16/alpha32/84attention projection, LPWM adapter1.343M+planner2.212M학습.
Native LPWM가중치는고정, 기존SSL·명령입력·학습률·평가장면유지. 이전GT ON자동실행은해제됐다.

**최신 사용자 승인(2026-10-04 22:51 KST): Stage2 일부 계층 미세조정으로 빠른 경향 확인.**
`partial_output_layers_v1.json`: LPWM native 출력계층5.56M와 planner2.21M 학습,
전체75,297개 navtrain/조건별1epoch/4,707update, 객체GT 보조loss off/on 비교.
GPU0·1 batch4×accum2=유효16, SSL유지, LoRA없음. 초기/512update마다128dev모니터,
조건후1,024planning+256world dev검증, 전·중·후particle시각화 자동연결.
아래 full run의2095update/pause는 그대로 보존하며 새 실험만 승인됐다.
실행상태 `outputs/lpwm_partial_planning_v1/queue_state.json`,
[구성·속도·검증 범위](docs/lpwm_planning_experiment.md#2026-10-04-stage2-native-출력-계층-부분-미세조정).

**현재 상태(2026-10-04 22:23 KST): 사용자 요청으로 Stage2 일시중단. 자동 재개 금지.**
첫 조건 `metric_plus_world`의 **2,095 update / 0.4451 epoch**에서 정상 signal checkpoint를 저장했다.
모델과 optimizer746개 state의 step2,095 일치, source/config hash 일치, GPU0·1 worker 종료를 확인했다.
재개 파일은 `outputs/lpwm_object_future_planning_v3/metric_plus_world/latest.pt`이며,
별도 보존본 `outputs/lpwm_object_future_planning_v3/user_pause_20261004/checkpoint_update002095.pt`도 있다.
명시적 재개 요청 후 같은 설정으로 update2,096부터 이어간다. `pause.requested`를 임의 제거하거나
다음 조건을 시작하지 않는다. 상태 `user_pause_status.json`, 근거
[중단 검증](results/lpwm_object_future_planning_v3/user_pause_20261004.json), 재개 절차는 HANDOFF4절.
아래 실행 중 문구와 ETA는 중단 이전 이력이다.

**현재 실행(2026-10-04): GPU0·1에서 Stage2 전체 navtrain 어블레이션 시작.**
사용자는 객체 GT 감독의 채택을 확정하지 않았으며 **객체 보조 loss 없음/있음 비교 후 결정**을 요청했다.
`configs/lpwm_planning/object_future_joint_v3.json`: `metric_plus_world` → 전체 개발 검증 →
`metric_object_future_plus_world` → 전체 개발 검증·paired 비교. LPWM 전체 low LR1e-6,
planner3e-4, 원래 SSL world loss 유지, 각75,297개 train 장면/20epoch/94,140update.
GPU당batch2×누적4×2GPU=16. 프로세스 `kjs-lpwm-stage2`, queue PID1131167.
GPU별 **전체 VRAM46GB 이하**(다른 사용자 점유 포함), CPU RAM에는46GB 제한 없음.
현재 첫 조건 본학습 update16 확인, 전체VRAM각약35.9GB/학습 peak allocated11.87GiB.
진행 `outputs/lpwm_object_future_planning_v3/queue_state.json`,
`outputs/lpwm_object_future_planning_v3/metric_plus_world/progress.json`.
기존Stage1 실패gate는 수정하지 않았다. Top16 박스proxy의타당성정정과나머지검증결과에근거한
별도 [진입 amendment](results/lpwm_object_future_planning_v3/stage1_admission_amendment.json)로 실험 진입을 등록했다.
미래 객체 상태/native mask 검증은 미완료이며 완전한 적응이나 PDMS 개선을 이미 입증했다는 뜻이 아니다.
과거 v2 queue를 재개하지 않는다. 아래 Stage2 미시작/GT loss 미구현 문장은 이번 실행 이전 이력이다.

**완료(2026-10-04): 객체 정보 보존 검증의 첫 단계인 frozen current-state readout.** 공개/적응
LPWM의 전체 train23,126/dev7,745 clip 표현 추출과 CPU linear readout 학습·평가를 완료했다.
종류 판독 점수는 개선됐으나 위치·속도 결과는 혼재하며 GT-ROI geometry 대조보다 약하다.
결과 `results/lpwm_object_readout_validation_v1/summary.json`; 미래상태/mask/planning 개입 검사는 후속이다.
설정 `configs/lpwm_navsim_adaptation/object_readout_validation_v1.json`, 상태
`outputs/lpwm_object_readout_validation_v1/queue_state.json`은 complete. LPWM 재학습/Stage2 실행은 아니다.
[수정된 학습 원칙·검증 범위](docs/lpwm_planning_experiment.md): **Stage1은 객체 GT 없는 SSL로 유지한다.**
GT 객체/미래 보조 loss는 Stage2 planning+world 공동학습에 한정하며 라벨 의존성과 정보 유지도 비교한다.
이 Stage2 보조 loss는 설계 단계이며 아직 구현/기동하지 않았다.

**현재 상태(2026-10-04 12:29 KST): Stage1 학습·전체 개발7,745clip 평가 완료. 객체 박스 대응 비열등성 gate 실패로 Stage2 진입 차단.**
복원·미래예측 등 나머지 gate는 통과했지만 top16 box recall@IoU0.1의 paired CI 하한이 등록 기준에 미달했다.
**해석 정정(13:30 KST):** 이 항목은 자체 particle–GT 박스 기하 proxy이며 LPWM의 detection/segmentation
성능이 아니다. 단독 필수 gate로 쓸 타당성은 미확립이다. [지표 정의와 한계](docs/lpwm_planning_experiment.md).
결과 `results/lpwm_navsim_full_posttraining_v2/summary.json`; 기준 완화나 추가 학습 없이 원인 진단이 다음 단계다.
**Planner 연구 설계 제안(2026-10-04):** [문제의식·선행연구·비교 방법](docs/lpwm_planning_experiment.md).
같은 후보 집합에서 객체·시간별 미래 정보가 경로 선택에 주는 이득을 검증하는 안이며, 아직 runtime/queue에 적용하지 않았다.
아래는 등록된 학습·검증 계획과 완료 이력이다.
공식 encoder·context·dynamics·RGB decoder 전체 109.55M parameter와 temporal ELBO를 유지한다.
학습 23,126 clip/122 recording, development 7,745 clip/40 recording; 12프레임 전체 posterior 복원+11개 전이 KL.
과거4장→미래8장 자율 예측은 별도 평가/planner 프로토콜이며, Stage1 학습의 RGB rollout loss가 아니다.
20 epoch / 28,920 update. GPU0·1 DDP, GPU당 batch4×누적2, 유효batch16, FP32, 데이터 worker0.
실측 batch2 1.73s → batch4 1.61s/update; worker0·2·4·8은 1.61~1.63s로 추가 이득 미확인.
`kjs-lpwm-stage1` 본학습은2026-10-04 11:19KST경 완료. 공개/적응모델 전체개발평가도 완료했다.
설정 `configs/lpwm_navsim_adaptation/full_posttraining_v2.json` + `execution/batch4_accumulation2_workers0.json`.
시각화 `outputs/lpwm_navsim_full_posttraining_v2/visualization/index.html` (같은 개발 장면 학습 전·중·후).
Stage1 적응 gate 후 LPWM 낮은LR + planner 전체학습: 후보 imitation / metric 증류 / 미래particle 후보보정 3조건 비교.
현재 queue `scripts/queue_lpwm_validated_training.py`, 설정 `metric_distillation_v2.json`; CPU teacher준비완료(train/dev coverage99.8%).
각조건 학습 후 검증 gate를 통과해야 다음 작업/독립test로 진행한다. 설계 `docs/lpwm_planning_experiment.md` 최신절.
과거 `full_joint_training_v1.json` 호출은 `active_pipeline.json`을 통해 새 queue에 join한다. 이전 단일경로 planner를 별도로 실행하지 않는다.
Stage2 train75,297/dev27,076; 적응gate 실패로 GPU 학습은 미시작이며 PDMS 개선도 미검증이다.
이전 cap8,192/4epoch 실행안은 대체됐고, v1 파일과 과거 결과는 그대로 보존한다.

**완료: LPWM의 NAVSIM 객체 표현 적응 실험.**
공식 main `4cf53c4`와 49쪽 논문을 조사하고 Sketchy checkpoint를 strict loading했다.
원영상·회전 보정 × 3 seed × 300 update, 90 train/30 development clip의 학습·평가를 완료했다.
원영상 적응의 복원 MSE는 0.05680→0.01640이지만, 객체 박스 대응률은 17.71→18.47%로 추가 개선 미확정이다.
과거만 사용하는 미래 MSE는 0.03015, 마지막 영상 유지 0.03032로 차이 CI가 0을 포함한다.
회전 보정은 시야 손실이 커 채택하지 않는다. Planning/PDMS 이득은 평가하지 않았다.
보고서 `docs/lpwm_navsim_adaptation_results.md`, 논문 검토 `docs/lpwm_paper_and_driving_assessment.md`.
실제 이미지·GIF `outputs/lpwm_navsim_adaptation_v1/visualization/`, 공유 PDF/JSON `results/lpwm_navsim_adaptation_v1/`.
좌표 검사 3개, 미래 입력 교란 검사 8개 모델 통과. 등록 작업 종료, 기존 Drive/WA/공용데이터 보존.


**완료(2026-10-03): encoder 자체 미래 표현 학습과 개발 평가.**
LoRA 없이 마지막 2개 또는 6개 encoder block을 직접 학습하고,
내부 ego FiLM·미래 감독·target 선택·입력 마스킹·미래 loss 강도를 16조건 × 3 seed로 비교했다.
총 48회 / 24,576 update, 공식 개발 PDM과 공통 future probe, 48개 raw 영상 추론 검증을 완료했다.
원본 ADE/PDM은 0.352210 m / 87.119134%, planning-only 2블록은 0.347629 m / 88.396320%,
6블록은 0.346061 m / 88.718100%다. ADE 감소는 관측됐지만 PDM 개선 구간은 0을 포함한다.
**미래 감독의 실질적인 추가 planning 이득은 확인하지 못했다.** Intent에 따른 encoder 출력 변화는 검증했다.
CPU 167개 검사 통과, 원본 가중치 보존, 우리 학습·평가 종료. 독립 test나 전체 encoder 사전학습 결과가 아니다.
[전체 결과와 한계](docs/encoder_future_learning_results.md), [등록 설계](docs/encoder_future_learning.md)

**사용자용 이미지:** 실제 전방 영상·원본/재학습 궤적·장면별 오차·미래 감독의 추가 효과를 3장으로 정리했다.
[이미지 설명과 PDF](docs/encoder_future_learning_results.md#실험-결과-이미지). 새 학습이나 GPU 추론은 하지 않았다.


현재 맥락과 ego 주행 의도에 따라, 같은 예산에서 planning에 유용한 미래 예측 대상을 선택하도록
학습하는 연구 작업공간이다. 현재 구현 단위는 camera patch이며 객체 instance와 구분한다.
**SafeDrive baseline 연구는 잠정 중단 상태다.**
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
   [200-update 미래 감독 비교](docs/target_supervision_exploration_results.md),
   [저분산 진단·제한1000-update 결과](docs/future_prediction_variance_followup_results.md),
   **최신 [물리·ridge 기준선 / 3-seed residual 기반 판단](docs/pilot_foundation_decision_results.md)**,
   [공개 future-planning 기반 감사 / 선택·예산 계획](docs/public_future_planning_foundation_audit.md),
   [공식 Drive-JEPA full planning checkpoint 전체 평가](docs/official_drive_jepa_reproduction.md),
   **현재 최신 [선택적 미래 예측 기반 추천·통제 실험 명세](docs/future_prediction_foundation_decision.md)**,
   [저장 navtest의 현재 상황별 현황](docs/official_navtest_current_context_summary.md).
4. [명명 규칙](docs/naming_conventions.md), [경로 이전과 호환성](docs/directory_migration.md).

## 현재 구현과 과거 자산의 구분

**이전 완료(2026-10-03): 세 논문 착안 적용·평가.** SPARTAN 희소 연결 → C-JEPA 관측 마스킹 →
IA-JEPA 움직임 선택을 9조건 × 3 seed × 800 update로 비교했다.
같은 개발 192개 window의 PDM은 기존 global 88.826, 새 sparse 87.742, 마스킹 87.742,
움직임 선택 87.692다. 현재 특징만 쓰는 대조군도 87.742로, 추가 미래 예측의 이득은 확인하지 못했다.
CPU 162개 검사 통과, 원본 가중치 보존, GPU 작업 종료. [전체 표·대응 CI·적용 범위](docs/drive_jepa_region_research.md).

**이전(2026-10-03): 선택 위치 학습·파이프라인 진단 완료.**
등록15run 후 사용자 지시에 따라 개수·크기 탐색을 추가하지 않고 동일K8/같은크기로진단했다.
위치교체는관측되지만planning상중요요소선택은미입증이다. Gradient단절보다위치대응활용약함과
보조목표방향불일치가의심된다. 현재GPU작업없음. [최신결과·원인·재현](docs/research_status.md).
새갤러리 `outputs/drive_jepa_selective_future/spatial_region_selection_visualization_20261003/index.html`.

**최신 사용자용 시각화(2026-10-03)**: 학습된 MLP/ego-query selector와predictor의
전후 선택위치·미래latent 오차·실제1024-D출력을 확인하는 CPU 갤러리를 생성했다.
`outputs/drive_jepa_selective_future/selector_predictor_visualization_20261003/index.html`.
진입점 `scripts/visualize_drive_jepa_learned_modules.py`, config `learned_modules_visualization_v1.json`.
새학습/GPU사용 없이 dev192개 저장결과 대조. 모든 사진은 실제참조영상이며 생성예측영상이 아니다.
밤샘학습은87run/61,200update 전부종료(commit8182f6c). 아래69run/18run진행안내는과거이력이다.

**최신 작업(2026-10-03)**: [원인 분리·동일 예산 선택 비교](docs/drive_jepa_overnight_causal_followup.md).
69run/46,800update와 실제 모델6,144번 patch교체 진단을 완료했다. 추가dev 원본ADE0.352210m,
낮은LR ego0.346338m이나 원본대비 cluster CI는0포함, 선택 우월성/PDMS 개선은 아직 미확인이다.
다음은 같은 낮은LR의 fixed/random/aux-off 대조18run을 별도 등록해 실행한다. 완료모델은재학습하지않는다.
GPU1단일작업/09:00KST상한/공용데이터와원본weights보존. 상태·재개는HANDOFF를우선한다.

**이전 완료(2026-10-03)**: [구조별 추가 학습 결과](docs/drive_jepa_architecture_followup.md).
기존 공식 planner를 보존한 채 contextual residual predictor, ego-query selector,
미래 branch last4 encoder QKV LoRA를15run/4500update로 비교했다.
MLP+새절차 dev ADE0.209975m, 원본0.220644m, LoRA0.216459m.
작은 개발 표본의 개선 경향이며 공식 PDMS 향상이나 학습형 선택 우월성은 미검증이다.
원본 hash/branch-off 출력 보존, OOM0, 현재GPU작업없음. 추가 확대/sweep은 자동 실행하지 않는다.
공유 [결과 JSON](results/drive_jepa_selective_future/architecture_followup_v1_20261003/summary.json).

**최신 사용자 지시(2026-10-02)**: [공식 Drive-JEPA planner 재사용 + 선택적 patch 미래 경로](docs/drive_jepa_selective_future_connection.md).
WA-JEPA는9253/12146(76.18%)에서중단/보존했고자동재개하지않는다.
[부분 결과](results/official_wa_jepa_reproduction/partial_navtest_at_drive_extension_20261002.json): PDMS91.102506,
동일9253scene Drive89.019762; 전체평가수치가아니다. 원본Drive 전체PDMS89.224320은그대로보존한다.
새경로는learnable K4 front patch선택→경량future latent→원본planner memory의zero-init residual이다.
CPU전체103tests/실제project-train2recording의출력보존·gradient·비용gate를통과했고현재GPU작업없다.
성능향상/최적선택/새논문기여를주장하지않는다. [연결결과JSON](results/drive_jepa_selective_future/connection_v1_20261002.json).

**아래는 보존된 WA 작업 이력(fd5fc5f 이후)**: [WA-JEPA 공식 재현·sparse interface](docs/official_wa_jepa_reproduction.md).
**이전상태: 2026-10-02 21:03KST 사용자요청으로 GPU0은5개/GPU1은2개, 총7worker로증설.**
20:40 재개시각2개였고GPU0대기열만추가했다. 최소6GiB reserve/타인작업보호/기존14-waypartition 유지.
8686/12146(71.5%)완료scene를 재사용하고 남은3460을 같은14-way partition에서 순차처리한다.
모델·12-step·seed·scorer·checkpoint 불변. 12GiB launch admission/6GiB running reserve/우리worker만SIGINT.
타인작업은 건드리지 않는다. [재개기록](results/official_wa_jepa_reproduction/shared_gpu_resume_state.json),
[이전중단기록](results/official_wa_jepa_reproduction/paused_evaluation_state.json).
[모듈별추론병목측정](docs/wa_jepa_inference_module_timing.md): shared denseencoder0.545s/
predictor24.105s/model24.698s, predictor약98%. 이전단독GPU측정은6.413/6.694s(95.8%).
12-step 공식 source/published checkpoint를 고정하고 전용 Conda에서 원본 agent/scorer를 실행한다.
Dense 전체 평가와 canonical all-ID/fixed/random 소수 scene 검증만 승인됐으며 학습은 하지 않는다.
아래는 이전 기반 결정·보존 이력이다. 최신 범위는 HANDOFF와 WA-JEPA 보고서를 따른다.

최신 사용자 결정: **578be6e의 공식 평가 결과를 보존하고 선택 연구의 기반·첫 통제 실험을 결정**한다.
공식 source/config/scorer/checkpoint는 `configs/official_drive_jepa/reproduction_v1.json` 그대로다.
Drive-JEPA PB의 train-only future head가 planner에 전달되지 않음을 코드로 재확인했다.
WA-JEPA는 joint future/trajectory 경로가 있어 다음 기반으로 추천하지만, native spatial patch-tube라는
범위 축소 승인과 full weight/config 호환 검증이 남았다. **아직 이전/selector 구현/학습을 시작하지 않았다.**
기존 CSV+현재 speed/command만으로12146scene 상황 통계를CPU16.29초에집계했고metadata누락0/hash불변이다.
자체 pilot·확대·residual은 보류하며 새 환경/대용량 다운로드/추가 전체 평가는 이번 범위 밖이다.
**공식 전체 평가 완료**: navtest12,146scene 전부성공/실패·누락·중복0, PFViT-L PDMS89.224320.
논문v2 Table2 PF89.0 대비+0.224320점이며정확원인/공식허용오차미확정. GPU0·1평가약12분21초,
총worker4, 종료후카드해제. 결과 `results/official_drive_jepa_reproduction/`, 보고서에재현명령/근거를기록했다.
학습/selector 구현은하지않았고이평가를반복하거나다른작업을자동재개하지않는다.
공식 encoder만 사용했던 pilot은 공식 planning checkpoint 재현이 아니며 모든 기존 자산을 보존한다.
아래 WA-JEPA 계획은 이전 판단 이력이며 최신 명세를 먼저 읽고 현재 자동 실행하지 않는다.
현재 pilot은 연결·gradient 진단 자산으로 보존하고 predictor 튜닝/확대 학습은 보류한다.
WA-JEPA 공식 코드의 joint future/planning 경로와 공개 checkpoint 메타데이터를 확인했다. 다음은
공식 기반의 가중치 로딩·추론 재현 후 같은 예산의 상황별 선택→학습형 선택→평균 예산 배분 비교다.
WA-JEPA full model 재현이나 새 selector 구현이 완료된 상태는 아니다. 등록된 pilot 확대 설정을 자동 실행하지 않는다.

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
├── scripts/diagnose_visual_future_prediction_variance.py  기존checkpoint mean/persistence·분산·교란 진단
├── scripts/train_bounded_future_prediction_followup.py   상한고정 추가seed/continuation·E/F detach 비교
├── scripts/survey_navsim_visual_target_coverage.py  CPU target 유효율/side projection 조사
├── docs/                                        연구·계산 그래프·검증·명명 문서
├── results/synthetic_diagnostics/                작은 공유용 검증 결과와 과거 raw 기록
├── results/adapter_diagnostics/                  실제 상태 연결 진단 (시각 JEPA 아님)
├── results/visual_diagnostics/                   영상+GT ROI+spatial 혼합 감독 진단
├── results/data_surveys/                         여러-log 유효율 summary·탐색 train/dev manifest
├── results/target_supervision_exploration/       200-update 비교 실측·곡선·한계
├── results/future_prediction_diagnostics/        저분산/horizon·추가seed·E/F 및 새checkpoint 검증
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
아직 곡선이 내려가는 초기 탐색이었다. `9353acf` 후에는 기존cache/200학습을 반복하지 않고
추가7400update로A seed11/47, A–E seed29총1000, E/F 대응seed29/11을 실행했다.
C저분산이 .012→.214로 개선됐지만 visual persistence를 모든horizon에서 못넘었고,
E/F 차이는 작다. Target/최종 성능/novelty를 확정하지 않았다. 현재 추가학습은 종료했다.
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
  공식 Drive-JEPA 전체평가는GPU0·1병렬로완료했고현재우리백그라운드평가도없다.
- Git 원격은 협업 이력 보존을 위해 `JunSeongKW/SafeDrive`, branch `junseong/main`을 유지한다.
  저장소 주소가 현재 연구의 baseline을 의미하지 않는다. 원격 저장소명 자체는 변경하지 않았다.
- 협업 시작 commit은 `95015df`; 코드·결과·문서를 commit 단위로 공유한다.
