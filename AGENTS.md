# Planning-Aware Future Prediction — 에이전트 작업 규칙

**2026-10-06 15:05 KST 최신: 사용자 요청으로 매500update 표현 진단을 등록했다.**
`scripts/monitor_lpwm_particle_trends_every500.py`, config `configs/lpwm_drivor_review/particle_trends_every500.json`.
새 monitor568996이 기존 monitor3186135를 대체했다. 이전 monitor를 중복 실행하지 않는다.
별도 root `outputs/lpwm_drivor_particle_trends_every500_v1/`의 registration/status/index 확인.
기존 monitor의 watch.lock과 진단 output을 재사용해 exact1614 CPU 비교3317230과 호환한다.
1000/1500/2000…500간격 + 기존epoch/final106시점. 정확checkpoint hardlink 보존, 미래update 대체 금지.
학습3186133/queue3186134 및 기존279source/config·원래monitor registration은 불변이다.
500진단 완료,582학습 진행 확인. 로컬보고/PNG 자동갱신이며 채팅push 기능은 연결되지 않았다.

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

**2026-10-06 최신 사용자 지시: LoRA 적용 계층 설명 후 사용자가 선택한다. 새 학습 금지.**
`docs/lpwm_lora_layer_catalog.md`에 모든 Linear234/Conv2d70개와 역할별 후보를 정리했다.
기존 Q/V LoRA는 epoch1/1614 정확한 checkpoint 보존·중단; 실제 process의1615는 별도 보존된 in-flight update다.
96scene 진단과 시각화 완료: `results/lpwm_drivor_epoch1_particle_review_v1/`.
Geometry head LoRA 후보 코드와 실제loss2update/DDP2update 검사는 준비했지만 본학습·새queue·새monitor는 미기동이다.
`outputs/lpwm_drivor_geometry_lora_v1/pause.requested`와 `pending_user_layer_selection.json`을 존중한다.
사용자 선택 전 prepared config를 실행하지 않는다. DrivoR 비교는25epoch 이후로 미룬다는 지시는 확정됐다.
아래 첫epoch DrivoR비교선택대기/geometry미구현/학습진행중 문장은 이전 이력이다.

**2026-10-06 최신: 첫 epoch 검증·DrivoR 비교 후 나머지24epoch 검토 제어 등록.**
진입점 `docs/lpwm_drivor_epoch1_review.md`, `configs/lpwm_drivor_review/epoch1.json`, 제어PID2949030.
1614update의정확한model/AdamW/scheduler/RNG를먼저보존하고기존pause신호로본학습·후속queue·monitor를대기시킨다.
Root `outputs/lpwm_drivor_epoch1_review_v1`의status/resume_state/training_held확인. 등록source/config변경금지.
경계외1update가진행됐으면별도보존;재개기준은epoch_01_resume.pt(1614)이며실중단latest를그대로쓰지않는다.
25epoch LR스케줄유지/epochs1변경금지. GeometryLoRA미적용. DrivoR동일1epoch vs공개최종참고비교선택을사용자에게요청중.
Navval도본학습포함이므로독립검증이라고부르지않고navtest를자동튜닝용으로전환하지않는다.
검토전24epoch자동재개/새DrivoR학습자동실행금지. 사용자요청의중간검토완료후문서의재개순서를따른다.

**2026-10-06 위치 변화 원인 감사:** `docs/lpwm_drivor_lora_geometry_audit.md`.
100update main checkpoint의 LoRA84tensor 갱신/양수planning gradient 확인. Q/V LoRA OFF 시 current xy bitwise동일,
FiLM OFF 시 initial xy bitwise복원, current xy→LoRA84개 autograd 연결없음/FiLM4개는연결됨.
현재 LoRA는 좌표생성 이후 interaction/context/dynamics에 있다. 초기warmup만을 위치불변원인으로설명하지않는다.
Geometry head/CNN LoRA는 수정제안이며 미적용. 이번요청은원인검사였고기존본학습·queue·등록source/config유지.

**2026-10-06 최신 사용자 요청: 학습 중 particle의 driving 유용성 및 DrivoR 공정 비교 검토.**
기존 LoRA train2994997/queue2994998을 유지하고 읽기 전용 checkpoint monitor3351138을 추가했다.
진입점 `scripts/monitor_lpwm_drivor_representations.py --watch`, root `outputs/lpwm_drivor_representation_monitor_v1`.
고정96 trainval장면/24recording의 전후 분포·readout·의도 경로 분해·particle/미래 개입을 기록한다.
Monitor registration sealed; source/config/panel hash 변경 금지. GPU0/allocator4GiB 상한/카드40GB 미만 진입,46.5GB 중단.
현재0/100update 진단 완료; 이후500/1000/2000간격·epoch·final 저장 checkpoint를 감시한다.
이 panel은 본학습 분포이고 도로는 평면지도 proxy다. 점의 이동·궤적 명령 반응만으로 이해·미래 효용을 주장하지 않는다.
공정비교 차이 및 별도 통제실험 설계는 `docs/lpwm_drivor_representation_and_fair_comparison.md`.
이번 검토로 DrivoR baseline/추가 학습을 새로 큐에 넣지 않았다. 기존 본학습과 평가 후속 순서는 유지한다.


**2026-10-06 최신 사용자 요청: DrivoR 방식 LoRA 및 여유 VRAM 활용.**
공개 LPWM native109.55M 고정, Q/V LoRA32·scale1·42projection1.343M와 새FiLM/projection/공식planner를학습한다.
기존 native-full35update와 pause는보존하고새LoRA는public-init. LoRA1update를저장후배치증설하여이어간다.
현재진입점 `configs/lpwm_drivor_lora/navsim_v1.json` + `execution_batch16_loader2_oracle4.json`,
`scripts/train_lpwm_drivor_lora_parallel.py`, `scripts/queue_lpwm_drivor_lora_parallel.py`.
기본설정은이력으로batch8이나실제execution override는batch16/accum2/GPU2/effective64,loader2/oracle4이다.
Batch24와loader4/oracle8실측후속도상batch16선택. GPU0·1총48decimalGB. 효과비교시microbatch변경이력을밝힌다.
`outputs/lpwm_drivor_lora_v1/`의 registration/queue_registration/parallelism_registration 소스·설정hash수정금지.
공식planner ego11D 주입은동일하나encoder에명령4D FiLM 추가;완전히같은ego-conditioning이라고부르지않는다.
후속v1fullnavtest→독립publicv2train→warmup/navhard평가를같은실행override로진행한다.
아래native-weight실행및PID는이전이력이며중복실행하지않는다.


**2026-10-06 최신 사용자 승인: 공개 LPWM + DrivoR joint E2E 학습 및 공식 PDMS/EPDMS 평가.**
과거 DrivoR/추가epoch/navtest 보류를 이번 명시적 요청 범위에서 갱신한다.
진입점 `docs/lpwm_drivor_joint_training.md`, `configs/lpwm_drivor_joint/`.
v1 trainval103,288/25epoch→fullnavtest, 별도 public-init v2 train85,109/10epoch→warmup/navhard EPDMS.
GPU0·1, 카드전체48decimalGB, GPU당batch8×누적4×2=유효64를유지한다.
`outputs/lpwm_drivor_joint_v1/registration.json` 및 `queue_registration.json`에등록된source/config수정금지.
기존full/frozen/Stage1효과실험은완료보존; 새학습에그weights/개발분할을넘기지않는다.
Root의pause.requested는사용자중단요청이며queue와trainer가존중한다. 타인process/공용원본은변경하지않는다.

**2026-10-05 최신 사용자 승인: 현재 학습·검증 뒤 fixed-LPWM planner 대조군을 자동 실행한다.**
기존queue421603은그대로두고후속CPUqueue871940이완료marker와네방법검증artifact를기다린다.
`configs/lpwm_planning/frozen_control_v1/queue.json` / `scripts/queue_lpwm_frozen_control.py`가추가진입점이다.
Stage1 LPWM의모든가중치·buffer와encoder command FiLM까지고정/eval/no_grad,동일planner만학습한다.
Planner에는ego명령입력을유지하며seed47/75,297navtrain장면/1epoch4707update/동일평가panel을쓴다.
SSL0.02항은detach된모니터이며어떤가중치도갱신하지않는다. 직접객체GT와navtest자동실행은없다.
GPU0·1총48decimalGB/card/8우선후메모리실패때4·2/유효planning16/world8/worker0을유지한다.
선행실행오류·불완전검증·freeze위반은차단하며,성능가설실패자체로대조군을선택적으로생략하지않는다.
새queue도source/config등록후수정금지. 전체일시중단시새root또는선행v4root의pause.requested를확인한다.
기존원본full2095pause자산은과거보존이므로새queue의자동중단조건으로추가하지않는다.

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


Claude Code 는 `CLAUDE.md`(= `@AGENTS.md`)로, Codex 는 이 파일로 같은 내용을 읽는다.
**현재 상태**는 `HANDOFF.md`, 상세 실험 기록은 `RESUME_NOTES.md`(시간순 실험 일지)와
`EXPERIMENT_DESIGN.md`(실험 설계·근거) 에 있다. 이 파일에는 바뀌지 않는 규칙만 둔다.

현재 루트는 `/rhome/junseong/PlanningAwareFuturePrediction/`이다. **SafeDrive baseline 연구는
잠정 중단**됐다. 이 저장소의 과거 `navsim/`·SafeDrive 학습 script·CSV·checkpoint는 보존된
참고 자산이다. 현재 코드의 진입점은 `src/planning_aware_future_prediction/`, `tests/`,
`scripts/validate_future_prediction_graph.py`다. 원격 저장소명은 협업 이력이지 baseline 선택이 아니다.
실제 데이터 진단은 `src/planning_aware_future_prediction/adapters/navsim_tracked_state.py`와
`scripts/validate_navsim_state_adapter.py`다.
이는 **현재 GT 상태를 사용하는 특권 입력 진단**이고 visual JEPA나 공식 baseline 재현이 아니다.
권고 visual adapter·baseline·남은 gate는 `docs/baseline_and_target_adapter_audit.md`를 읽는다.
현재 실제 영상 진입점은 `scripts/validate_visual_future_prediction_pilot.py`다.
Official frozen encoder+GT ROI의 visual/spatial 혼합 감독 pilot을 구현했고
`docs/visual_future_prediction_pilot_validation.md`에 검증·한계가 있다. 공식 full-stack baseline 재현,
pure visual JEPA, deployment perception 또는 학습 후 성능으로 해석하지 않는다.
`1231767` 이후 최신 다음 실행 기준은 `docs/research_question_and_target_decision.md`,
`docs/navsim_visual_target_coverage.md`, `docs/minimal_target_ablation_plan.md`다.
고정 K는 개발 기반이며 최종 novelty가 아니다. Target 비교 전에 선택 방식을 동시에 바꾸지 않는다.
최신 실행은 `docs/target_supervision_exploration_results.md`: 거리규칙 K4/seed29/A–E 각200update 완료.
이는 선택기 학습이 아닌 미래 감독 비교다. 초기 순위로 target을 탈락시키지 않으며 C/E의 낮은 예측 분산을
검토한 뒤 다음 학습 규모·비교군을 결정한다. 실행 설정은 `configs/exploration/target_supervision_run_v1.json`이다.
`9353acf` 이후에는 `docs/future_prediction_variance_followup_results.md`가 최신이다.
완료cache/200학습을 반복하지 않고 추가7400update와mean/persistence/분산·E/F 대응seed진단을 완료했다.
1000에도visual persistence보다나쁘며 JPEG보정은조건부. 등록상한에서멈췄고 다음구조ablation은미실행이다.
설정 `configs/exploration/future_prediction_diagnostic_followup_v1.json`, 공유 `results/future_prediction_diagnostics/`.

`607da52` 이후에는 `docs/pilot_foundation_decision_results.md`와
`docs/public_future_planning_foundation_audit.md`가 최신이다. 물리/ridge·실단위 미래 오차와 하나의
visual-only residual 대응3seed 비교를 완료했다. Pilot은 ridge보다 약하며 추가 튜닝/확대는 최신 사용자
지시에 따라 보류한다. **현재 predictor 개선을 연구의 선행 필수 과제로 삼지 않는다.**
다음은 미래 예측과planning 연결이 있는 공개 기반의 공식 동작 재현→상황별 동일예산 선택→학습형 선택→
동일 평균예산 배분이다. WA-JEPA의 source/metadata와 tiny attention autograd만 확인했으며 full 모델은
미재현이다. Native 공간·시간 token과 객체를 같은 선택 단위로 부르지 않는다. 기존 확대 config를 자동 재개하지 않는다.

보존된 재현 작업: **공식 Drive-JEPA full planning checkpoint 추론·전체 navtest 평가**.
진입점 `docs/official_drive_jepa_reproduction.md`, `configs/official_drive_jepa/reproduction_v1.json`.
독립 Conda/worktree와 공식 전처리·planner·scorer만 사용. Pilot encoder-only 결과와 구분한다.
이 작업에서는 학습·fine-tuning·selector·WA-JEPA 실행 금지. 재현 보고 후 pilot을 자동 재개하지 않는다.
공식full PFViT-L/navtest 전체평가가완료됐다:12146성공/실패·누락·중복0,PDMS89.224320.
`results/official_drive_jepa_reproduction/full_navtest_results.json`과report를먼저읽고평가를반복하지않는다.
논문과의차이0.224320점의정확원인/허용오차는미확정이며선택·예산가설의검증결과가아니다.

**578be6e 이후 최신 지시**: 공식 평가 자산을 보존하고 코드 기반 후보·현재 상황별 현황·최소 통제 실험을 정리한다.
진입점 `docs/future_prediction_foundation_decision.md`, `docs/official_navtest_current_context_summary.md`.
Drive PB future auxiliary head는 train-only이며 planner 입력이 아니다. WA-JEPA native spatial patch-tube
선택을 추천했으나 객체 instance와 다르며 사용자 범위 승인/full checkpoint 호환 gate가 남았다.
현재 분석·설계 단계에서 **새 환경/대용량 다운로드/학습/selector·동적K 구현/추가 전체 평가 금지**.
저장navtest 통계는 coverage용이지 H1 증거나 tuning용이 아니다. 완료한PF 전체평가를 반복하지 않는다.

**fd5fc5f 이후 최신 사용자 승인**: WA-JEPA 공식 checkpoint 재현과 학습 없는 sparse interface 검사.
진입점 `configs/official_wa_jepa/reproduction_v1.json`, `scripts/evaluate_official_wa_jepa.py`,
`docs/official_wa_jepa_reproduction.md`. 과거 조사 단계의 환경/download/평가 금지는 이번 승인 범위에서 해제됐다.
공식 source 별도 worktree/전용 Conda/strict loading/소수 scene smoke 후 gate 통과 시 dense 전체 평가.
Canonical spatial patch-tube는 객체 instance가 아니다. All-ID 동일성 확인 전 sparse 제거 금지.
Fixed/random sparse는 소수 scene만, 학습형 selector와 새 학습은 금지. Drive/pilot/확대/residual은 보존/보류.

**2026-10-02 19:12KST 최신 사용자 지시: GPU0·1반환을 위해 WA 평가를 중단했다.**
완료8686/12146scene보존/3460남음. 사용자재개요청과GPU재배정확인전GPU작업자동재개금지.
`outputs/official_wa_jepa_reproduction/evaluation_pause.json`은명시적user pause이므로임의삭제하지않는다.
재개검증/명령은HANDOFF와WA재현보고서를따른다. CPU보존검사는가능하지만worker/full/smoke기동은금지다.

**2026-10-02 20:40KST 후속 사용자 지시가 위 pause를 해제**: 공유 GPU0·1에서 워커를 줄여 재개.
각GPU 최대2/총4 worker, 기존14-way partition/source/weight/12step/seed/scorer 유지.
`shared_gpu_resume_v1.json`의 12GiB launch admission과6GiB reserve, CPUguard와bounded queue를 사용한다.
메모리압력으로 중단된 worker는 자동재시작하지 않으며 타인process는 절대중단하지 않는다.
새사용자pause는 항상 우선하고 queue도pause marker를 확인한다. Pilot·학습·추가benchmark 실행은 여전히보류.

**2026-10-02 21:03KST 사용자후속요청**: GPU0만최대5개로증설, GPU1은2개유지.
`shared_gpu_resume_v2.json`의GPU별상한과동일12GiB입장/6GiBreserve를사용한다.
CPUqueue 교체는기존worker를adopt하며GPUworker를중단/중복기동하지않는다.
모듈별비용진단의짧은단일model process는종료됐고공식dense평가는그대로실행중이다.

## 세션 시작 루틴

**2026-10-05 최신 사용자 실행 승인: GPU별 전체 VRAM48decimalGB로 올리고 배치 증설 실측.**
`configs/lpwm_planning/execution_48gb_v1/queue.json` / `scripts/queue_lpwm_48gb_planning.py`가
기존 두 CPU queue를 대체했다(새 queue1902774). 이전46GB/6GiB 및 현재학습중단금지 규칙은 이번 승인에 한해 갱신된다.
총48GB/물리free3GiB/allocator23.2GiB/allocated22.75GiB, GPU0·1만 사용한다.
Partial1576 model+AdamW125state를 별도 보존 후 batch4/8을 실측하고 안전한 빠른 설정으로1577부터 재개한다.
LoRA4/8 → Adapter2/4/8 → full2/4도 독립 profile 후 본학습·검증. 유효planning16/SSL8,worker0,
기존 모델/loss/LR/navtrain75,297/4707updates/검증panel 유지. Profile weights는 본학습에 사용하지 않는다.
Source/config는 새queue 등록hash를 따르며 실행 중 수정금지. 원본full2095/pause도 보존한다.
전체 중단 요청은 새queue root의pause.requested로 전달하며 중복queue를 기동하지 않는다.


**2026-10-04 23:53KST 최신 지시: 일부 계층 → LoRA → Adapter → 전체 low-LR 재개 순서.**
`four_method_sequence_v1.json`과 `queue_lpwm_followup_methods.py`가 기존 partial/LoRA queue 완료 뒤
Adapter와 full을 순차 실행한다. 각 조건 뒤 검증, 직접 객체 GT는 후순위. 현재 training1602577와
queue1675463/1709131을 중복 실행하지 않는다. 실행 중 source34개/config10개 hash를 변경하지 않는다.
사용자가 마지막 full은 이전checkpoint 재사용을 요청했으므로 이 새 후속 조건에 한해 재개를 승인했다.
기존20epoch 원본의 pause marker/2095checkpoint/config/source는 보존하고, 별도 출력에서2096→4707로 이어간다.
원래 conv_in 명령 위치/20epoch LR 스케줄/AdamW746state를 유지한다. 다른 세 조건과의 차이를 보고한다.
GPU0·1/전체46decimalGB/6GiB 여유 유지. 현재 partial을 멈추는 추가 GPU benchmark를 실행하지 않는다.


**2026-10-04 최신 사용자 우선순위 변경: 객체 GT 조건 후순위, partial 대 LoRA 먼저.**
신규진입점 `configs/lpwm_planning/adaptation_method_comparison_v1.json`과
`scripts/queue_lpwm_adaptation_methods.py`다. 현재partial OFF GPU학습은보존·인계하고,
기존CPUqueue만교체해GT ON자동실행을제거한다. Partial학습/검증→LoRA메모리·속도profile/
gradient검증→LoRA학습/검증→paired방법비교순서다. 두방법모두직접객체GT보조OFF,
같은Stage1/planner/명령위치/loss/LR/전체train1epoch. 새LoRA는rank16/alpha32,
particle interaction/context/dynamics attention84projection, native LPWM전체가중치고정.
GPU0·1/전체46decimalGB제한유지. 배치8은작은배치실측으로안전여유가예측될때만시험한다.
기존source/config/20epoch fullpause는그대로보존하고추가감독실험을자동실행하지않는다.

**2026-10-04 22:51 KST 최신 사용자 승인: 빠른 Stage2 부분 미세조정 별도 실험.**
사용자가 LoRA 또는 일부 계층 학습으로 경향을 먼저 보도록 요청했다. 신규 실행은
`configs/lpwm_planning/partial_output_layers_v1.json` / `scripts/queue_lpwm_partial_planning.py`다.
LPWM native 출력 계층5.56M만 갱신, planner2.21M 전체학습; LoRA는 사용하지 않는다.
전체 navtrain75,297개를 조건별1epoch, 객체GT 보조loss off/on 비교하고 각조건후 고정dev panel검증.
GPU0·1, batch4×accum2×2=16, checkpointing유지, 각GPU전체VRAM46decimalGB/6GiBreserve.
이 승인은 새 부분학습에만 적용한다. 이전 full run2095update와pause marker/source/config는 보존하며
`lpwm_object_future_planning_v3`나 WA를 재개하지 않는다. 신규queue source/config도 등록후변경금지.

**2026-10-04 22:23 KST 최신 사용자 요청: Stage2 일시중단.**
`outputs/lpwm_object_future_planning_v3/pause.requested`와 `user_pause_status.json`을 따른다.
첫 조건2,095 update의 model+optimizer checkpoint를 보존했고 queue/torchrun/GPU worker는 종료했다.
명시적 사용자 재개 요청 전 GPU 학습·추론·다음 조건/queue를 자동 재시작하지 않는다.
이번 사용자 pause는 아래 이전 실행 승인보다 우선한다. `queue_failed.json`의 pause-marker 예외는
현재 queue 구현이 남기는 중단 기록이며 학습 발산으로 해석하지 않는다. 원본 기록을 지우지 않는다.
재개 절차와 checkpoint SHA는 HANDOFF4절 및 `results/lpwm_object_future_planning_v3/user_pause_20261004.json`.

**2026-10-04 최신 사용자 승인: GPU0·1 Stage2 시작, 객체 GT 보조 감독은 on/off 어블레이션 후 채택 결정.**
최신 진입점은 `configs/lpwm_planning/object_future_joint_v3.json`과 HANDOFF 1–5절이다.
Queue1131167, `outputs/lpwm_object_future_planning_v3/queue_state.json`; 처음 `metric_plus_world` 본학습 시작.
기존 top16 box gate 실패와 source/수치는 보존하고 명시적 `stage1_admission_amendment.json`으로
해당 proxy만 단독 진입 조건에서 제외했다. 아래 과거 ‘Stage2 차단’ 상태를 현재 승인 범위에 적용하지 않는다.
Stage1 SSL 유지, Stage2 직접 객체 GT loss 유무 비교. GPU마다 다른 사용자를 포함한 전체VRAM46GB 이하,
CPU RAM46GB 제한은 사용자가 아니라고 정정했다. 본학습 source/config hash등록됨; 실행 중 변경/중복기동 금지.
과거 metric_distillation_v2 실패 queue 자동재개 금지. 조건별 학습 후 검증하며 성능가설 미충족은 숨기지
않고 다른 등록조건의 비교를 계속하되, 실행오류/누출/불완전학습/메모리한계는 의존 작업을 차단한다.

**현재 상태(2026-10-04 12:29 KST): Stage1 학습·전체 개발평가 완료, 객체 박스 대응 비열등성 gate 실패로 Stage2 차단.**
Top16 box recall@IoU0.1 paired CI[-0.04152,-0.02364]의 하한이 등록기준-0.02에 미달했다.
나머지 원래gate와 추가causal/noncollapse/coverage는 통과. Checkpoint/summary 보존 후 원인 진단하며 기준완화/강제Stage2 금지.
공식 encoder·context·dynamics·RGB decoder 전체 109.55M parameter와 temporal ELBO를 유지한다.
학습 23,126 clip/122 recording, development 7,745 clip/40 recording; 12프레임(4 observed+8 future).
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

아래 과거 승인·결과는 보존 이력이다. 현재 실행은 HANDOFF 최신 절을 우선한다.

**2026-10-03 최신 후속 승인: LPWM을 planning 객체·미래 감독으로 재학습하고 planner 연결 후 PDMS 평가.**
`docs/lpwm_planning_experiment.md`의 6조건×3seed×1000update 및 dev192 공식 PDM 실행이다.
이 명시적 요청이 이전 완료 파일럿의 추가 학습 보류를 해당 범위에서 갱신한다. 기존 WA/Drive 실험은 재개하지 않는다.


**최신 완료: 사용자 요청 LPWM 객체 중심 표현 파일럿.**
공식 main 4cf53c4·Sketchy checkpoint·전용 환경에서 raw/rotation_stabilized ×3seed×300update를 완료했다.
진입점 `docs/lpwm_navsim_adaptation_results.md`, `docs/lpwm_paper_and_driving_assessment.md`.
복원은 개선되나 주요 객체 대응·과거만의 미래 예측 추가 효용은 미확정이다.
이전 학습 금지는 이번 명시적 LPWM 파일럿 승인 범위에서 해제됐고 등록 6회는 종료됐다.
완료 파일럿·WA·기존 encoder 재실행이나 추가 sweep을 자동 시작하지 않는다.

**2026-10-03 최신 완료: encoder 자체 미래 표현 학습.**
사용자의 encoder 재학습 요청으로 마지막 2개/6개 block 직접 갱신, 내부 intent FiLM,
미래 감독·입력 마스킹·target 선택·강한 보조 loss 비교를 16조건 × 3 seed × 512 update로 완료했다.
진입점 `docs/encoder_future_learning_results.md`, `results/encoder_future_learning_v1/combined_summary.json`.
LoRA 없음. 기존 selector/future bridge는 추론에서 제거했고 미래 head는 학습 전용이다.
두 블록으로도 ADE 감소는 관측됐으나 선택적 미래 감독의 실질적 추가 이득은 확인하지 못했다.
CPU 167개 검사와 48개 encoder raw 영상 추론 검증 통과, 원본 가중치 보존, 모든 작업 종료.
전체 사전학습·독립 test가 아니며 완료 48회, WA, 기존 navtest 평가를 자동 재시작하지 않는다.
아래의 과거 실행 범위는 해당 시점의 이력이다. 현재 상태는 HANDOFF와 이 결과 보고서를 우선한다.


**이전 완료:** 사용자승인SPARTAN/C-JEPA/IA-JEPA착안9조건×3seed×800update 및dev192공식PDM비교를완료했다.
`docs/drive_jepa_region_research.md`, `results/drive_jepa_region_research_v1/summary.json`이최신이다.
새변형은기존global비교군을넘지못했고current-only도동일성능으로미래예측추가효용미확인이다.
CPU162통과/원본hash보존/학습·평가종료. 완료실험재실행/추가sweep/WA·navtest자동재개없음.

**최신 "아키텍처 개선 방향으로 추가 학습" 승인**: 기존 선택 비교는 완료했고
`configs/drive_jepa_selective_future/architecture_followup_v1.json`의5조건×3seed
warmup100+joint200만 실행한다. 원본planner/teacher 고정, future branch contextual predictor/
ego-query/last4 QKV LoRA 비교. 같은192window 재사용, 전체90분/condition15분/allocated12GiB 상한.
GPU0·1 중 점유 재확인 후 사용. 이전 200update 승인 제한은 이 새 범위에서 갱신됐다.
WA/pilot/데이터 확대/동적K/공식benchmark 반복은 계속 보류. 계획/상태는
`docs/drive_jepa_architecture_followup.md`를 우선한다.
2026-10-03 위5조건×3seed는 완료됐다. 같은 작업을 반복하지 말고 공유결과를 읽는다.
원본planner/teacher hash보존/branch-off동일, 현재GPU학습없음. LoRA의추가이득은확인하지못했다.
새학습·확대·sweep·benchmark는 자동시작하지 않는다.

**`2185ce5` 이후 최신 사용자 지시: 다음 단계 진행 승인.**
공식 frozen Drive planner 위의 작은 동일 K4 선택 비교를 진행한다.
`configs/drive_jepa_selective_future/selection_comparison_v1.json`과
`docs/drive_jepa_selection_comparison_plan.md`: train128/dev64, recording16/8,
fixed/random/learned/learned-no-aux, 대응3seed 각200update만. 과거 1-step 제한은 이 범위에서 해제됐다.
원본 no-branch는 평가만, 새 extension만 학습; WA/pilot/대규모 확대/동적K는 계속보류.
과거 노출 두 recording 제외, held-out/navtest 학습·평가 금지, 원본 모델/환경/공용데이터 보존.

**2026-10-02 최신 사용자 지시가 위 WA 실행 계획에 우선**: WA를9253/12146에서중단하고
공식Drive-JEPA planner를재사용하는선택적patch 미래extension을설계·구현·연결검사한다.
진입점 `docs/drive_jepa_selective_future_connection.md`, `configs/drive_jepa_selective_future/connection_v1.json`.
WA raw/partialCSV·JSON/비용/가중치는보존, pause marker유지/자동재개금지.
새경로의learnable selector/경량predictor/residual memory adapter는승인됐다. 원본planner교체금지.
작은train-only CPU/GPU forward/gradient/비용과1diagnostic step만수행; 대규모학습·pilot확대·동적K/horizon은미승인.
현재선택단위는front camera patch이며객체instance로부르지않는다. 공식baseline checkpoint/config는수정하지않는다.

```bash
git status --short --branch && git log --oneline -10
cat HANDOFF.md
rg --files src tests docs
ls -lt results/synthetic_diagnostics
```

## 세션 종료 루틴

1. `HANDOFF.md` 의 1(실행 중)·2(결과)·3(변경)·4(다음)·5(미결) 절을 갱신하고, 실험 결과의
   상세는 `RESUME_NOTES.md` 맨 아래에 날짜 절로 덧붙인다(기존 절은 고치지 않는다).
2. 커밋한다: `tools/handoff-commit.sh "<에이전트 이름>" "[exp] 제목 한 줄"`. 그리고 `git push mine` (원격 `mine` = GitHub JunSeongKW, 다른 서버가 같은 상태를 보도록). HANDOFF.md 3절이
   커밋 본문이 된다. 접두어는 `[exp]` `[code]` `[infra]`. `junseong/*` 브랜치에만 커밋한다(스크립트가 거부).
3. 장시간 학습·평가는 nohup 으로 띄우고, 끝나면 다음 작업이 자동으로 이어지도록 체이닝한다
   (`until grep -q EVAL_DONE <로그>` 방식; 프로세스 이름 grep 은 자기 명령줄과 일치해 무한 대기한다).

## 바뀌지 않는 규칙

- **협업 명명(사용자 확정)**: `docs/naming_conventions.md`를 따른다. 폴더·파일·class·function·
  인자·변수·설정·result key는 대상과 역할이 드러나게 짓는다. 주요 인터페이스의 h/c/u, N/K/T,
  batch/toy/temp/final 같은 축약·임시명은 피한다. 현재 관측/예측/미래 GT와 각 valid mask를
  명칭으로 구분한다. 수식 기호와 표준 외부 API(torch.optim, Tensor.grad 등)는 임의로 바꾸지 않는다.
  재명명 시 import·문서·CLI·환경·result schema도 갱신하고 전후 동작을 검사한다.

- **연구 명제(검증할 가설)**: "planning 에 필요한 미래 정보는 상황마다 다를 수 있고,
  현재 맥락·주행 의도와 planning objective 로 예측 대상을 선택하도록 학습할 수 있는가?"
  모든 실험 제안은 이 명제의 어느 하위 질문에 답하는지 한 줄로 밝힌다. 가설이 기각되면 다음
  가설은 원래 명제의 하위 질문 중에서 고른다. 효과가 큰 곁가지보다 원래 질문을 겨냥한 실험이 우선.
- **평가는 집계와 상황별 지표를 함께 쓴다**: 난이도 계층·시나리오 유형별 분해로 보고한다. 난이도 정의를
  여러 개 두고 정의를 바꿔도 유지되는 것만 주장한다. baseline 점수로 계층을 나눈 뒤 비교하는 것은
  평균 회귀 때문에 무효(RESUME_NOTES "난이도 정의가 결론을 바꾼다").
- **결과 보고**: 성능 향상 크기가 아니라 논문 기여가 되는지를 기준으로 말한다. 여러 실험은 한 표로 모은다.
- **GPU(2026-10-01 사용자 승인)**: 현재 서버에서는 **0·1만 사용**한다. util 0% 는 비어 있다는
  뜻이 아니며 기존 타인 프로세스를 건드리지 않는다. 두 카드는 RTX A6000 약 48 GB로 확인했다.
  이전 서버의 4~7 할당·H100 batch 설정은 현재 서버에 적용하지 않는다. 본 학습 batch/메모리는
  실측하고, 과거 실험과 비교할 때 유효 배치·학습량 차이를 기록한다.
- **프로세스 이름**: 연구 목적·실험이 드러나는 환경명을 쓰고 python을 절대경로로 호출한다.
  사용자 이니셜 `kjs`를 붙여 `kjs-<연구/모델>-<역할>` 형식을 사용한다(예: `kjs-wa-jepa-eval`).
  실행 중인 환경/prefix는 이동하지 않으며 필요하면 실행 별칭 symlink로 다음 실행부터 적용한다.
  현재 CPU 환경은 `runtime/environments/future_prediction_cpu/`다. 과거 SafeDrive env 명칭을
  새 연구에 재사용하지 않는다. GPU 실행 전 호스트 점유·정책을 다시 확인한다.
- **공유 머신**: 다른 연구원(junhyeok, hanbin, dogun, uisung)의 프로세스·컨테이너·폴더는 건드리지
  않는다. `pkill -f` 금지, PID 를 먼저 확인하고 죽인다. 대량 삭제는 `.trash-*/` 로 옮겼다가 실행 중인
  작업이 없을 때 지운다.
- **환경**: 새 의존성은 새 env 에. 이전 서버의 `safedrive` 환경
  (py3.10 / torch 2.1.0+cu121 / mmcv 2.1.0 sm_90 / spconv-cu120 / mmdet 3.2.0)은
  현재 서버에 존재하는 환경이 아니다. 기존 base·타 프로젝트 env를 업그레이드하지 않는다.
- **연구 협업 상태**: `docs/research_status.md`는 확인 사실·실행 이력·미확인 항목,
  `docs/selective_entity_future_prediction_graph.md`는 계산 그래프 명세다. 설계 초안과 구현·실험 결과를 구분하고,
  Codex/ChatGPT 간 인수인계에 commit·경로·근거를 남긴다. ChatGPT 제안을 검증 결과로 취급하지 않는다.
- **데이터 경로·안전(AXE-080)**: `/rhome/junseong/`이 코드·변환 결과·metric/feature cache를 포함한
  작업공간이다. `/home/user/data/Dataset/` 전체는 연구실 공용 원본이므로, 그 안의 파일·디렉터리를
  직접 생성·수정·이동·이름 변경·삭제하지 않고 작업공간의 프로젝트에 심볼릭 링크로만 연결한다.
  기존 `dataset -> /home/user/data/Dataset/navsim` 링크는 그대로 읽기 전용 사용한다. 새 원본만
  `/home/user/data/processed_dataset/`에 총 1 TB 한도 안에서 다운로드한다.
- **실행 방식**: 확인 질문으로 멈추지 말고 합리적 기본값으로 진행한 뒤 가정을 결과와 함께 보고한다.
  되돌릴 수 없는 삭제만 예외.
- **현재 경로**: 코드 `src/planning_aware_future_prediction/`, 검사 `tests/`, 실행 `scripts/`,
  공유 결과 `results/`, 로컬 산출물 `outputs/`, 실행 환경 `runtime/environments/`,
  공식 참고 clone `reference_repositories/`(마지막 세 항목은 git 제외).
- **과거 SafeDrive 자산**: `navsim/`, `scripts/run/`, `scripts/analysis/`, `analysis/*.csv`,
  `exp/safedrive/`, `ckpts/` 등은 중단된 pipeline의 코드·결과다. 현재 명명 작업에서 수정/재학습하지 않는다.
