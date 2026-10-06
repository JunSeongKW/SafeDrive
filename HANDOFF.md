# HANDOFF — 이 파일 하나로 다음 에이전트가 이어받는다

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

**2026-10-06 최신: 사용자가 LoRA 적용 계층을 선택할 때까지 새 본학습 보류.**
첫epoch1614checkpoint 보존·96scene진단·직진/좌회전/우회전 시각화 완료. 기존run은1615에서정상중단됐고
추가1update는별도보존됐다. 위치평균0.302px/96.36%가1px이내,크기평균변화2.020%.
새headLoRA코드/실제loss2update/DDP2update검사는준비했지만본학습·새queue·새monitor미실행.
진입점 `docs/lpwm_lora_layer_catalog.md`, 후보설정 `configs/lpwm_drivor_geometry_lora/`.
새root `outputs/lpwm_drivor_geometry_lora_v1/pause.requested`와`pending_user_layer_selection.json`유지.
사용자가계층설명을보고직접대상을정하겠다고명시했다. 기존run재개/새조건시작금지.
DrivoR비교는25epoch완료후로미룸. 아래실행중/첫epoch비교선택대기는이전이력이다.

**2026-10-06 09:40 KST 최신: 첫epoch 검토를 위한 대기 제어 등록.**
사용자제안에따라제어PID2949030이1614update/epoch1 정확한재개상태를보존한뒤기존본학습·queue·monitor를pause한다.
현재1541update/73남음,예상10:04KST. 아직pause는아니며학습진행중. 이후96scene진단을실행하고DrivoR비교·검토대기.
상세 `docs/lpwm_drivor_epoch1_review.md`, root `outputs/lpwm_drivor_epoch1_review_v1/status.json`.
사용자에게DrivoR동일1epoch학습 vs공개최종모델참고비교를선택요청중. 새DrivoR학습은미실행.
나머지24epoch의자동진행은이검토를위해보류된다. 아래10월15일ETA는중간검토대기시간미포함이전추정.

**2026-10-06 08:55 KST 중간점검:** LoRA v1 1,400/40,350update(3.47%), epoch1의86.74% 완료.
Train2994997/queue2994998/monitor3351138 host 생존·진행 정상. 최근300 wall속도20.707초/update,
현재v1학습 ETA10월15일17시KST(부하변동가능),epoch1은오늘10:09쯤. V2학습·benchmark평가 시간은별도.
표현monitor는1000까지완료, 다음1614부근. 객체readout F1 .3799→.3475로효용개선아직미확인.
Snapshot `results/lpwm_drivor_lora_v1/status_20261006_0855.json`. 설정·학습·queue변경없음.

**2026-10-06 최신 원인 검사:** [현재 좌표와 LoRA 경로 감사](docs/lpwm_drivor_lora_geometry_audit.md).
실제100update LoRA를 꺼도 current xy는같고 FiLM을끄면초기좌표로돌아간다. LoRA84tensor는모두갱신됐다.
좌표생성뒤의attention만LoRA여서 current위치 직접학습경로가빠져있다. Geometry LoRA 제안은미적용이며기존학습유지.

**2026-10-06 01:25 KST: 기존 LoRA 학습 유지 + 표현 진단 monitor 추가.**
Monitor3351138, `outputs/lpwm_drivor_representation_monitor_v1/watch_status.json` 확인.
고정96장면 학습 전 및 update100 진단 완료, 이후500부터 자동 추적. 보고서 `docs/lpwm_drivor_representation_and_fair_comparison.md`.
학습 분포 진단이며 공식 성능/독립 검증이 아니다. 순수 Register–Particle 인과 비교에는 통제 차이가 남았다.

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

마지막 갱신: 2026-10-06 10:36 KST (Codex)

**최신 완료(2026-10-05 23:11 KST): NAVSIM Stage1의 planning 효과 확인.**
공개LPWM고정+planner78.9161 → NAVSIM적응LPWM고정+동일planner82.5238, PDMS+3.6077점(CI[+1.5065,+5.8539]).
동일seed47/planner1epoch4707/GPU당batch8×2/75,297train; 내부개발1024중1021유효PDM/40recording,navtest아님.
최종평가·paired비교완료,등록작업모두종료. 상세 `docs/lpwm_planning_experiment.md` 최상단과
`results/lpwm_stage1_effect_v1/queue/stage1_effect_summary.json`. 아래 실행중표시는과거이력이다.

**이전 실행(2026-10-05): 공개 LPWM 고정 + 동일 planner 학습으로 Stage1 효과 검증.**
사용자 정정: 완료된 적응 고정군82.5238을 재사용하고 새 조건 하나만 학습한다.
GPU당8×누적1×GPU2=유효16/75,297장면/seed47/1epoch4707update를 동일하게 유지한다.
진입점 `configs/lpwm_planning/stage1_effect_v1/queue.json`, `scripts/queue_lpwm_stage1_effect.py`.
Queue1869615: 공개고정학습4707/4707,22:53:37정상종료. 22:53:38부터public_control_evaluation실행중.
DrivoR·객체GT·navtest자동실행금지유지. 등록source/config는변경하지않는다.
아래 모든작업종료는 이전 다섯조건 완료시점이며 새 공개 고정 조건과 구분한다.

**최신 완료:** 네 LPWM 미세조정 및 Frozen LPWM 동일 planner 대조군의 학습·검증·paired 비교 완료.
Frozen PDMS82.5238, Adapter82.4852, LoRA81.9141, partial81.6341, full81.2282.
네 미세조정-minus-frozen PDMS 신뢰구간 모두 0 포함: 이번 설정에서 표현 미세조정의 추가 이득 미확인.
20:56:16 모든 등록 작업 종료. 상세 `docs/lpwm_planning_experiment.md` 최상단 및
`results/lpwm_frozen_control_v1/completed_comparison_20261005/summary.json`.
**DrivoR/추가 seed·epoch/navtest를 자동 실행하지 않는다.** 아래 과거 실행 중 문장은 이력이다.

세션 시작: 이 파일 + `git log -10` + `AGENTS.md`.

**완료: LPWM의 NAVSIM 객체 표현 적응 실험.**
공식 main `4cf53c4`와 49쪽 논문을 조사하고 Sketchy checkpoint를 strict loading했다.
원영상·회전 보정 × 3 seed × 300 update, 90 train/30 development clip의 학습·평가를 완료했다.
원영상 적응의 복원 MSE는 0.05680→0.01640이지만, 객체 박스 대응률은 17.71→18.47%로 추가 개선 미확정이다.
과거만 사용하는 미래 MSE는 0.03015, 마지막 영상 유지 0.03032로 차이 CI가 0을 포함한다.
회전 보정은 시야 손실이 커 채택하지 않는다. Planning/PDMS 이득은 평가하지 않았다.
보고서 `docs/lpwm_navsim_adaptation_results.md`, 논문 검토 `docs/lpwm_paper_and_driving_assessment.md`.
실제 이미지·GIF `outputs/lpwm_navsim_adaptation_v1/visualization/`, 공유 PDF/JSON `results/lpwm_navsim_adaptation_v1/`.
좌표 검사 3개, 미래 입력 교란 검사 8개 모델 통과. 등록 작업 종료, 기존 Drive/WA/공용데이터 보존.

세션 끝: 상태 문서 갱신 + `tools/handoff-commit.sh` + `git push mine`.
**완료(2026-10-03): encoder 자체 미래 표현 학습과 개발 평가.**
LoRA 없이 마지막 2개 또는 6개 encoder block을 직접 학습하고,
내부 ego FiLM·미래 감독·target 선택·입력 마스킹·미래 loss 강도를 16조건 × 3 seed로 비교했다.
총 48회 / 24,576 update, 공식 개발 PDM과 공통 future probe, 48개 raw 영상 추론 검증을 완료했다.
원본 ADE/PDM은 0.352210 m / 87.119134%, planning-only 2블록은 0.347629 m / 88.396320%,
6블록은 0.346061 m / 88.718100%다. ADE 감소는 관측됐지만 PDM 개선 구간은 0을 포함한다.
**미래 감독의 실질적인 추가 planning 이득은 확인하지 못했다.** Intent에 따른 encoder 출력 변화는 검증했다.
CPU 167개 검사 통과, 원본 가중치 보존, 우리 학습·평가 종료. 독립 test나 전체 encoder 사전학습 결과가 아니다.
보고서 `docs/encoder_future_learning_results.md`, 통합 `results/encoder_future_learning_v1/combined_summary.json`.
본 학습 source `ae5c2da`, 추가 대조군 `e7d57b1`. 아래 이전 작업은 완료 이력이다.

**이전 작업 완료:** SPARTAN/C-JEPA/IA-JEPA 착안9조건×3seed×800update 및개발PDM비교를완료했다.
등록commit5b85a01. 원본0.352210m/87.119134, 기존global0.347129m/88.826031,
새sparse0.349238m/87.742199, +mask0.349230m/87.742244, +motion0.348680m/87.691769.
현재특징만0.349207m/87.742499로미래예측추가효용미확인. 기법별추가효과CI모두0포함.
CPU162통과/원본hash보존/학습56.39분/peak1.719GiB/개발6528score완료. 우리GPU작업없음.
`docs/drive_jepa_region_research.md`, `results/drive_jepa_region_research_v1/summary.json`이최신이다.
새방법채택/추가sweep/WA/navtest자동재개하지않는다. 아래이전최신표시는이력이다.

**최신 사용자 요청(2026-10-03): 중요 요소 선택을 개선하는 크기·개수·학습 신호 비교.**
**후속최신지시: 개수·크기보다위치이동과학습파이프라인진단이우선.** 확인시등록15run은이미완료됐고
PID2624673은종료돼중단할process없었다. 새크기/개수/학습sweep는시작하지않는다.
같은K8/2×2region의planning vsretention 6모델의읽기전용진단도완료했다.
설정 `location_learning_diagnosis_v1.json`, train16recording을결과무관hash고정,
scoregradient·hard교체·미래slot교란·fusion attention/원본의존도; optimizerupdate0/15분/4GiB상한.
실측62.91초/1.260GiB, 768hard교체. Gradient단절없음/위치대응활용약함/proxy방향불일치관측.
현재실행중인우리학습·진단없음. 최신결론·재현은 `docs/research_status.md` 맨위절을본다.
이전15run결과: retention은위치74–89%교체/현재teacher유지오차0.824→0.457m이나
devADE0.347129→0.346951m으로선택효용미확정. 장면중앙으로쏠리는다른편향이관측된다.
기준122e885. `spatial_region_selection_v1.json`에 5조건×3seed×800update를 결과 확인 전 등록했다.
현재704cache/공식frozen planner/공통warmup·batch순서 재사용. 작은16patch, 큰4region,
큰8region random/planning/planner-retention 비교. Region은2×2native token 평균이며 객체가 아니다.
Retention은현재planner의선택정보의존도proxy로 selector만학습;미래중요도의정답으로해석하지않는다.
2시간전체/20분조건/allocated8GiB/여유6GiB, 단일GPU0또는1만. 기존결과덮어쓰기없음.
진입점 `scripts/train_drive_jepa_spatial_regions.py`, 15run/33분29초/peak1.665GiB완료. CPU전체154검사통과.
이번명시적재학습요청이아래시각화전용·09:00마감의과거범위를갱신한다.

**이전 사용자 요청: 학습된 selector/predictor 시각화.** 밤샘 87run/61,200update는
03:45에 모두 종료됐고 결과 commit은 `8182f6c`다. 아래 18run 등록/실행 문장은 과거 이력이다.
이번에는 새 학습 없이 기존 seed29의 MLP/ego-query 낮은LR final800을 CPU에서 복원했다.
192dev window/24recording 모두 저장 GPU selected IDs와 일치, window MSE 차이 최대9.54e-7.
사용자용 갤러리: `outputs/drive_jepa_selective_future/selector_predictor_visualization_20261003/index.html`.
명령 ID별 2개·서로 다른6recording을 결과/미래유효성과 무관하게 hash선정했다.
현재영상 선택 전후/확률·빈도/미래정답영상과crop/예측오차/실제1024-D latent 비교를 제공한다.
사진은 모두 실제 참조 영상이며 생성 예측 영상이 아니다. 미래target은고정격자·2frame tubelet이다.
이 요청은 시각화·읽기전용 검증이며 GPU/학습/WA/pilot 재개는 하지 않는다.

**현재 우선 작업(2026-10-03 밤샘 승인)**: [원인 분리·재학습 계획](docs/drive_jepa_overnight_causal_followup.md).
사용자가 추가 질문 없이 다음 오전09:00KST까지 명제를 유지한 실험·문헌 조사·기록을 승인했다.
기준8159aad, 원인진단 및7조건×3seed×400update를완료했다. 모두고정400update dev평균은원본보다나쁘다.
첫 단계69run/46,800update와 실제 모델6,144번patch교체 진단까지 모두완료했다.
추가dev 원본0.352210/MLPlearned0.361504/random0.351212m; learned 우월성 없음.
낮은LR MLP0.347694/ego0.346338m은 개선경향이나 원본대비cluster CI가0포함한다.
다음 별도등록18run은 낮은LR fixed/random/jointaux-off/currentfeature대조다. 완료learned모델재사용.
GPU1단일process/16GiB입장·6GiBreserve·allocated8GiB상한, 원본/공용데이터/WA중단상태 보존.
아래 '학습 없음/새학습 자동금지'는 이전 완료 시점 이력이며 이번 명시적 승인 범위에는 적용하지 않는다.

**이전 완료 작업**: [Drive-JEPA 구조별 추가 학습](docs/drive_jepa_architecture_followup.md).
기존 선택 비교 4조건×3seed×200update는 완료/보존했다. Learned dev ADE0.242582m은 원본0.220644m보다 나쁘다.
같은192window에서 contextual residual predictor → ego-query selector → future-branch encoder LoRA를
분리 비교하여 5조건×3seed, aux warmup100+joint200을 모두 완료했다(2026-10-03).
Dev ADE는 원본0.220644m, MLP+새절차0.209975m, contextual0.223808m,
ego-query0.216938m, LoRA0.216459m. 작은 dev 개선 경향이지 공식 PDMS/선택 가설 검증은 아니다.
최대 allocated2.674GiB, 전체21분40초, OOM0. 현재 실행 중인 우리 학습/검증은 없다.
실행 진입점 `scripts/run_drive_jepa_architecture_followup.py`, 설정 `architecture_followup_v1.json`.
아래 이전 진행 문장은 이력이며 최신 상태는 이 절과 report를 우선한다.

**이전 연결 작업**: [Drive-JEPA 선택적patch 미래경로 연결](docs/drive_jepa_selective_future_connection.md).
WA-JEPA는9253/12146(76.181%)에서7개우리worker만SIGINT/CPUhelper종료, 실패·중복0/해시검증완료.
부분PDMS91.102506 vs같은9253scene Drive89.019762; 전체결과가아니다. 남은2893은자동재개금지.
원본Drive 전체PDMS89.224320/weights/환경/source를보존하고공식planner에K4 patchfuture residual을추가한다.
`2185ce5` 이후 "다음 단계 진행" 승인으로 작은 동일K 선택 비교를 실행한다.
계획 `docs/drive_jepa_selection_comparison_plan.md`, 16train/8dev recording·128/64window,
4조건×3seed×200update. 원본planner frozen, WA/pilot/대규모 확대는 보류한다.

**아래는 이전 WA 실행 이력**: [WA-JEPA 공식 재현과 sparse 검증](docs/official_wa_jepa_reproduction.md).
이전(2026-10-02 21:03KST): **사용자요청으로GPU0 5개/GPU1 2개/총7worker로증설했다.**
20:40 각2개/총4worker로재개한뒤GPU0여유확인후대기구간4·6·8을추가했다. 기존GPUworker중단없음.
8686/12146scene(71.513%)보존/남은3460/실패·중복0을 출발점으로 기존14shard를 queue처리한다.
원본model/source/checkpoint/12step/seed/scorer/config 불변; 새변경은동시성·메모리보호뿐이다.
12GiB launch admission/6GiB running reserve, 우리worker만SIGINT, pressure-stop 자동재시도금지.
이전19:12 중단기록/2.89MB백업은그대로보존. 타인작업/공용데이터를변경하지않았다.
Strict6scene/all-ID6scene/72trial비용검사는완료/보존. **FullPDMS는아직미완료**.
설정 `configs/official_wa_jepa/reproduction_v1.json`, 전용 Conda/worktree. 학습 없음.
사용자명명/명시적재개승인: 이니셜 `kjs`, `/rhome/junseong/envs/kjs-wa-jepa-eval/bin/python`.
10:32KST 우리14worker만SIGINT 정상중단/562완료scene보존 후동일14shard/config/seed로재개했다.
실제Conda prefix는이동하지않고symlink만추가했다. 저장완료scene를skip하며in-flightscene만재계산한다.
Latest4-step와논문/checkpoint12-step 불일치를 찾아 직전 공식404d8af/12-step를 결과 확인 전 고정했다.
이전 조사 단계의 환경/다운로드/평가 금지는 이번 승인 범위에는 적용하지 않는다.
공통 조사 상태: [docs/research_status.md](docs/research_status.md).
계산 그래프 초안: [docs/selective_entity_future_prediction_graph.md](docs/selective_entity_future_prediction_graph.md).
시간순 이력: `RESUME_NOTES.md`. 과거 설계: `EXPERIMENT_DESIGN.md`.
CPU v1 결과: [docs/synthetic_validation_results.md](docs/synthetic_validation_results.md).
실제 데이터 상태 진단: [docs/navsim_state_adapter_validation.md](docs/navsim_state_adapter_validation.md).
Baseline·visual adapter 감사: [docs/baseline_and_target_adapter_audit.md](docs/baseline_and_target_adapter_audit.md).
최신 실제 영상 pilot: [docs/visual_future_prediction_pilot_validation.md](docs/visual_future_prediction_pilot_validation.md).
최신 연구 결정: [docs/research_question_and_target_decision.md](docs/research_question_and_target_decision.md).
여러-log CPU 조사: [docs/navsim_visual_target_coverage.md](docs/navsim_visual_target_coverage.md).
당시 최소 학습 계획: [docs/minimal_target_ablation_plan.md](docs/minimal_target_ablation_plan.md).
**최신 실행 결과**: [docs/target_supervision_exploration_results.md](docs/target_supervision_exploration_results.md).
**9353acf 이후 최신**: [저분산 후속 결과](docs/future_prediction_variance_followup_results.md),
[전처리·문헌·실행 범위](docs/future_prediction_diagnostic_scope_and_evidence.md).
직접 선행연구: [docs/egofsd_foredrive_evidence_audit.md](docs/egofsd_foredrive_evidence_audit.md).
**607da52 이후 최신 결과**: [pilot 기반 판단](docs/pilot_foundation_decision_results.md).
**최신 완료 작업(기준578be6e)**: 공식 평가 자산 보존 + 선택적 미래 예측 기반/통제실험 설계.
보고서: [기반추천·계산그래프](docs/future_prediction_foundation_decision.md),
[저장navtest 현재상황현황](docs/official_navtest_current_context_summary.md).
공식 source/checkpoint/scorer 고정 설정: `configs/official_drive_jepa/reproduction_v1.json`.
WA-JEPA native spatial-tube 기반을 추천했으나 범위 승인/full strict compatibility gate는 남았다.
모델 이전/selector 구현/학습/새 환경/대용량 다운로드/추가 전체 평가를 실행하지 않았다.
현재 작업 루트: `/rhome/junseong/PlanningAwareFuturePrediction/`.
명명 규칙: [docs/naming_conventions.md](docs/naming_conventions.md).
경로 이전: [docs/directory_migration.md](docs/directory_migration.md).

## 0. 현재 연구 의도 — 최신 사용자 프롬프트가 우선

**현재 주행 맥락·ego 의도·planning 목적에 따라 같은 예측 예산에서 유용한 객체의 미래를
선택하도록 학습할 수 있는가?** JEPA 채택 자체가 핵심 기여는 아니다.

- H1: 상황에 따라 유리한 미래 정보 구성이 다를 수 있다. 아직 일반적 사실로 확립되지 않았다.
- H2: 맥락/planning-conditioned 선택이 같은 예산의 강한 비교군을 넘는가.
- H3: 같은 평균 예산에서 상황별 예측량 배분이 고정 예산보다 나은가. 공식 기반 재현 후 검증한다.
  동적 K/horizon은 이번 단계에서 미구현이며 현재 pilot은 **고정 K·고정 horizon의 진단 자산**.
  EgoFSD 중복 때문에 fixed-K 객체 선택을 최종 novelty로 전제하지 않는다.
  Pilot target 튜닝을 선행 필수 과제로 삼지 않고 공개 future-planning 기반 위에서 직접 검증한다.
- 관측 마스킹 / 미래 target 선택 / planner 입력 선택을 구분한다. 주 초안은 미래 target 선택.
- SafeDrive는 기존 motivation 자산과 코드 참고다. 주 baseline으로 임의 회귀하지 않는다.
- Drive-JEPA의 공식 PF planning 전체 평가는 완료됐다. **직접 연결형 future predictor의 확정 baseline은 아니다.**
- 과거 “모든 기존 JEPA 마스크는 입력과 무관” / “두 비교 열이 아니오면 novelty 확보” 주장은 철회한다.

## 1. 실행 중인 작업

최신: geometry·appearance·future LoRA 본학습3186133/queue3186134/monitor3186135 실행 중.
`outputs/lpwm_drivor_planning_path_lora_v1/navsim_v1/progress.json` 첫3update 양rank에서103,288scene/40,350목표와
모든14개 LPWM 경로군·planner gradient 유한·양수 확인. 실제micro16/누적2/유효64·loader2/oracle4.
이 새 root의pause만 새실행을제어하며 기존hold를지우지않는다. 처음부터공개LPWM+동일seed2planner.


현재 본학습 없음. Old train2994997/queue2994998/monitor3351138은 정상 종료했고 review2949030도진단완료.
새geometry조건은단일GPU2update와GPU0·1 DDP2update검사만완료,검사용가중치본학습사용금지.
새본학습/새queue/새monitor는아직기동하지않았다. 명시적인사용자계층선택을기다린다.

첫 epoch 검증의 타당성 질문에 공식 코드로 답변했다. 이번 설명에서 학습·hold·queue 설정을 변경하지 않았다.
Review status 최신 조회: training_update=1561, waiting_for_epoch_boundary, target=1614.

09:40KST: 본학습1541update,새review제어2949030 waiting_for_epoch_boundary. target1614/첫epoch 예상10:04.
기존trainer source/config무변경. Epoch fullresume checkpoint확보후localpause를써trainer·oldqueue·watcher종료대기,
이후1614checkpoint로96scene표현진단을수행한다. 끝나면held_for_drivor_comparison_and_review이며24epoch자동재개안함.

08:55KST 확인: v1 1400/40350, 총약8시간8분경과. GPU0/1 각29.59decimalGB,조회util82/100%.
Batch16×누적2×2=64/loader2·oracle4 유지. 최신1400checkpoint08:54:43저장,active pause/완료marker없음.
최근300 wall20.7067초/update→현재v1종료10/15 16:57KST,epoch1 10/6 10:09KST 추정.
후속v2 13300update는같은속도라면약3.19일추가;navtest/warmup/navhard평가시간별도라전체queue종료시간은미확정.

위치 원인 진단은 optimizer0/3장면의별도실행으로완료했다. GPU reserved0.736GB/카드총30.683GB,본학습그대로진행.
새진단script는 `scripts/audit_lpwm_drivor_lora_geometry.py`; 완료결과를덮어쓰거나재실행하지않는다.

후속 시각화 의미 질문은 코드 읽기와 설명만 수행했다. 본학습·monitor·등록 source/config 변경 없음.

읽기 전용 representation monitor3351138이 본학습 checkpoint를 자동 진단한다. 본학습/queue는 유지한다.
고정96장면의 초기0/100update 완료(288.7/257.0초, peak reserved0.654GB), 다음500update 대기.
GPU0/별도 allocator4GiB 상한/카드40GB 미만 진입·46.5GB 중단. Main training 중단 요청도 존중한다.
별도monitor root의 registration은 sealed이며 변경금지. 학습source/config는 이번 턴 수정하지 않았다.


2026-10-06 최신: LoRA본학습 parent2994997,후속queue2994998. Root `outputs/lpwm_drivor_lora_v1`.
현재baseconfig는기존batch8이지만 `active_execution.json`/`execution_batch16_loader2_oracle4.json`이우선이다.
배치16/누적2/GPU2=유효64/loader2·oracle4 per rank/48decimalGB. 1update model+AdamW+schedule+RNG재개.
원래full35update,첫LoRA1update사본과pause/launch는각oldroot와`before_parallelism_upgrade/`에보존.
`training_parallel.log`, `progress.json`, `queue_status.json`에서실제진행확인. 진단weights는main미사용.
아래2788260/2839064는종료·보존된full실행이다.

2026-10-06 00:13 KST: v1 joint 본 학습 24/40,350 updates, parent2788260, queue2839064 정상.
GPU별총약45.26GB/batch8/acc4/effective64. 최근10update평균37.3초→v1잔여약17.4일(공유자원변동).
`outputs/lpwm_drivor_joint_v1/navsim_v1/progress.json`와`queue_status.json`확인; 현재 stage=navsim_v1_training.
Navtest12,146 및warmup220입력완료,navhard5,912입력CPU준비중(session47116);입력완료누락은queue가평가전재준비.
Epoch별particle그림,100update마다resume checkpoint. 본학습첫100update까지latest.pt가아직없을수있다.

2026-10-05 23:11 KST: 공개고정대조군학습·검증·paired비교모두정상완료(queue1869615 complete).
검증23:11:21,비교23:11:28. 현재등록학습/평가없음. 아래실행중/ETA는이전조회이력이다.

2026-10-05 22:54 KST: 학습22:53:37 returncode0/summary저장완료,22:53:38 최종개발검증자동시작(PID2514642).
결과예상23:05–23:15KST, 아래22:53저장대기문장은직전조회이력이다.

2026-10-05 22:53 KST: 공개고정본학습4707/4707 update와마지막128scene monitor완료.
Queue heartbeat정상이며최종저장/검사/종료후동일개발평가가자동진행된다. 정확snapshot은results/lpwm_stage1_effect_v1/training_eta_20261005_2253.json.

2026-10-05 22:44 KST: 공개고정본학습4192/4707, queue1869615 public_control_training·heartbeat정상.
DrivoR 논문/공식source 학습방법 조사만 수행했고 GPU/현재 등록source/config/대기열은 변경하지 않았다.

2026-10-05 22:27 KST: 공개고정본학습3184/4707·queueheartbeat정상확인. 사용자 planning신호강화제안은
기존로그감사/설계검토만진행했으며새backward/학습/대기열등록/현재설정변경없음.

2026-10-05 22:19 KST: 공개고정 queue1869615 public_control_training2784/4707,heartbeat정상.
사용자요청 일부계층/전체particle시각화는 기존snapshot/체크포인트만 CPU로 읽어 완료했다. 본학습/GPU조건변경없음.

2026-10-05 22:04 KST: 공개고정학습 queue1869615 public_control_training heartbeat정상(update1904확인).
이번 사용자 질문은 Adapter 효과 해석이며 코드·기존 결과만 조사했다. 새 GPU진단/학습/queue변경 없음.

2026-10-05 21:56 KST: 공개 LPWM 고정 대조군 queue1869615는 public_control_training 정상 진행 중.
Adapter 대 고정군 particle 시각화는 완료된 두 조건의 48개 snapshot을 CPU로 처리했으며 GPU/학습설정은 변경하지 않았다.

2026-10-05 21:44 KST: 공개고정본학습768/4707(16.3%),queue1869615 heartbeat정상.
최근1.054–1.127s/update,학습22:55–23:10/검증결과23:05–23:25KST추정. 가속진단은종료했다.
Batch8/effective16/기존monitor/source/config유지. 결과완료후자동Stage1효과비교예정.

2026-10-05 21:35 KST: Stage1 효과용 공개 LPWM 고정 대조군 본학습304/4707,유효배치16,
LPWM각gradient0/planner양수,최근약0.93s/update. Queue1869615=public_control_training.
CPU실제학습2step감사60.45초/GPU감사26.15초,7음성검사통과. Profile전체카드최대26.36GB/48GB.
일반도구sandbox의detach자식은종료돼본학습미실행,권한승인후host에서등록queue정상기동했다.
추가읽기전용진단 `scripts/benchmark_lpwm_frozen_representation_batching.py`로표현계산batch8/16/32출력동일성·속도,
고정world SSL monitor 생략시planner gradient동일성을확인중이다. 본학습/등록source/config는변경하지않았다.

2026-10-05 20:56 KST: Frozen 검증20:56:12·네방법대비집계20:56:16 완료, queue_state/completion 모두 complete.
Main421603/frozen871940 및 최종 평가1766544/학습1597075 종료 확인. 현재 등록 LPWM GPU 작업 없음.
CPU 원자료 재집계·hash/장면/공통누락/CI 일치 확인과 비교 PNG/PDF/JSON 생성 완료. 아래는 이전 상태 기록이다.

2026-10-05 20:46 KST: Frozen planner 대조군 학습4707/4707완료(20:45:37),현재frozen_control_evaluation. GPU0·1배치8/유효16/전체75297장면1epoch를마쳤고다음최종개발검증진행. DrivoR미등록유지.

2026-10-05 20:39 KST: 사용자 요청한 네방법 결과비교의 CPU재집계·원집계동일성·그래프·보고서완료. Frozen queue는frozen_control_training, 최신4368/4707. GPU새실행/설정변경없음.

2026-10-05 20:27 KST: 네 가지 미세조정 학습·검증 완료. Full 학습19:06:33/검증19:16:44/집계19:16:48 완료. Frozen 대조군은19:18:08 자동 시작, 현재3616/4707 (76.8%) planner만 학습 중. Queue871940/frozen_control_training heartbeat 정상; main421603은complete다. DrivoR 미등록 유지.

2026-10-05 18:35 KST: 사용자에게 frozen 대조군 자동 연결을 재확인했다. Main queue는 full_low_learning_rate_training, frozen queue는 waiting_for_full_training_and_validation이며 양쪽 heartbeat 정상. 기존 대기열 유지.

2026-10-05 17:02 KST: 사용자 지시로 현재 LPWM 내부 미세조정 vs frozen 대조군 학습·검증·비교까지 완료한다. Main full 학습 3280/4707, frozen CPU queue 대기 정상. DrivoR 비교는 후속 과제로 보관하며 대기열에 넣지 않는다.

2026-10-05 16:55 KST: 사용자 지적에 따라 register/particle 후속 비교의 공통 DrivoR planner 설계 원칙을 기록했다. 기존 full 학습 3184/4707, frozen 대조군 대기 유지. 공통 DrivoR 구현·새 학습은 미실행이다.

2026-10-05 16:47 KST: DrivoR register와 LPWM particle의 논문·공식/현재 코드 비교를 완료했다. Full 학습은 3104/4707 진행, frozen 대조군 대기 유지. 이번 턴은 조사·설명이며 GPU 실행 변경 없음.

2026-10-05 16:31 KST: full_low_learning_rate_training 유지, 최신 2928/4707. 현재 조건은 LPWM native 전체와 planner/command 입력을 함께 학습하는 full fine-tuning이다. 후속 frozen 대조군은 대기 중이다.

2026-10-05 16:25 KST: full low-LR 학습 2864/4707 (60.85%), 남은 1843 update. 최근 128/256/512 update의 경과시간 기준 약 5.61–5.63s/update; 학습 종료 19:15–19:40, 최종 개발 검증 종료 19:30–20:10 KST 예상. Main queue는 full_low_learning_rate_training, frozen 대조군 queue는 waiting_for_full_training_and_validation이며 양쪽 heartbeat 정상.

2026-10-05 16:17 KST: 사용자승인 frozen-LPWM planner대조군 후속CPUqueue871940 기동, waiting_for_full_training_and_validation/heartbeat정상/GPU자식0. 기존v4queue421603은full_low_learning_rate_training 유지. 새진입점configs/lpwm_planning/frozen_control_v1/queue.json.

2026-10-05 15:29 KST: 사용자 Adapter 결과보고에 저장된최종평가CPU집계·CI·시각화를완료. 현재queue는full_low_learning_rate_training;이번턴새GPU실행/학습·설정변경없음.

2026-10-05 15:22 KST: 마지막 full_low_learning_rate 본학습 실행 중. 15:12:12KST 시작, 원본2095 model+AdamW746state 재개, 최신2192/4707. Batch8 profile은할당예산OOM,4×누적2×GPU2=유효16선택. Adapter학습·검증완료. Runtime변경없음.

2026-10-05 14:58 KST: Adapter 본학습4707/4707 완료, v4 queue421603은residual_adapter_evaluation 진행. 배치 확대 점검 도중 pjh-wamvla-v2가 GPU0·1에복귀(PID290180/290181,각19522MiB). GPU속도profile미실행·runtime변경없음.

2026-10-05 14:42 KST: 사용자 pjh- 단서로 과거 자체 GPU 조회 원본을 재확인. 09:35:49 GPU0·1 pjh-wamvla-v2 PID290957/290958, 각20168MiB 확인. 14:42 현재는 우리866456/866457만 해당GPU사용. 실행변경없음.

2026-10-05 14:37 KST: 사용자junheok GPU0·1이탈시간질문으로호스트GPU/소유자/accounting조회. 현재866456/866457 junseong kjs-lpwm-stage2만GPU0·1사용. Adapter4176/4707(14:35:32 snapshot). 실행변경없음.

2026-10-05 14:31 KST: 공개LPWM 해상도가능범위질문. 공식modelzoo·64/128config·해상도의존head코드재확인. 활성학습/queue변경없음.

2026-10-05 14:24 KST: 타E2E 입력해상도/축소성능질문을공식문헌·코드로확인했다. 현재학습상태재조회·queue변경없음.

2026-10-05 14:16 KST: 전처리의구조적이유질문에대해공식LPWM소스·체크포인트설정감사. 학습·queue변경없음.

2026-10-05 14:10 KST: 이미지해상도질문에대해공개hparams·전처리·Stage1/2입력과실제cache헤더읽기전용확인. 학습/queue/설정변경없음.

2026-10-05 11:19 KST: 학습충분성질문에따라완료Stage1/LoRA 저장곡선CPU집계·시각화. GPU/queue변경및현재Adapter진행률재조회없음.

2026-10-05 11:14 KST: Stage1/2 모두1epoch인지에대한후속질문. 저장완료summary재확인만실시했고학습·queue변경없음.

2026-10-05 11:09 KST: 사용자1seed/1epoch질문에대해저장summary/config와DiffusionDrive·DrivoR원문을확인했다. Stage2 seed47/1epoch75297장면4707update,Stage1 20epoch28920update를구분했다. 활성학습/queue변경및진척재조회없음.

2026-10-05 11:01 KST: 완료LoRA 결과요청에응답하여저장된최종평가1024planning/256world를Partial과CPU paired집계·시각화했다. Adapter기존학습/queue변경없음. 최신진행률재조회는이번턴실시하지않았다.

2026-10-05 10:52 KST: Adapter본학습10:46:36KST시작,update80/4707. GPU당batch8/accum1×2=16,전체VRAM각45.89GB. Queue421603/torchrun866051. LoRA검증10:43:53완료후profile/gradient/engineering검사통과해자동진입. 이번턴runtime변경없음.

2026-10-05 10:39 KST: Adapter는 아직미시작. 현재queue attention_lora_evaluation, LoRA학습완료후 world_retention 97/256검증중. Adapter node/실행선택파일없음. Heartbeat정상갱신. 근거results/lpwm_card_budget_measured_v4/adapter_start_check_20261005.json.

2026-10-05 10:35 KST 사용자 분할 질문: 현재 Stage1/2 manifest와 평가 panel을 읽기전용 감사했다. 본턴 학습상태 재조회/학습·queue·config 변경 없음. 최신 실행 상태는 직전 LoRA 완료 기록 참고.

2026-10-05 10:31 KST 확인: LoRA 본학습은10:15:22KST 4707/4707update,1epoch 정상종료(returncode0). checkpoint/latest/epoch01 보존. 마지막128개발monitor PDMS83.6380/ADE1.14796/FDE2.72267. Queue421603/evaluation457092 실행,학습427197 종료. 현재 검증 단계 {'stage': 'persistent_future', 'completed': 132, 'total': 1024}. Adapter/full은 이후 등록 대기. 읽기전용점검으로runtime변경없음.

최신 질문: navval 정의와 논문 공정 비교 프로토콜. DrivoR 학습 코드/분할 설정을 읽고 권고안을 기록했다. 활성 학습·queue·source/config 변경 및 새 평가 실행 없음.

최신 사용자 질문은 E2E 논문의 학습·평가 분할 관행 설명이다. DiffusionDrive/DrivoR/DriveSuprim/PARA-Drive 및 NAVSIM 공식 문헌을 확인했다. 활성 GPU 학습·queue·등록 source/config 변경이나 추가 평가 실행은 없다.

2026-10-05 사용자평가split질문: 읽기전용공식token·자체manifest대조및설명정정. 현재학습/queue변경없음. LPWM81.63/84.56는navtest일부가아닌navtrain내부개발평가다. 이번턴현재학습진행률재점검은하지않음.

2026-10-05 10:06 KST 사용자 중간결과 요청: 저장로그·개발검증을 읽고 같은128monitor/동일4096update 비교 및곡선PNG 작성. Partial1epoch4707 최종1024개발 PDMS81.6341/ADE1.28250/FDE3.04333. LoRA4528/4707(10:04snapshot),최신공통4096 monitorPDMS partial80.6169/LoRA84.5584(+3.9415점),ADE1.23115/1.11915. Adapter미시작/full2095보존재개대기. Runtime/queue/source/config변경없음.

최신 2026-10-05 09:59 KST: v4 queue421603 / torchrun427197 / LoRA batch8×accum1×GPU2=16 본학습재개확인. 4422보존상태에서4448/4707까지진행,loss유한. 카드전체약47.60GB/각GPU100%,8profile·gradient audit·engineering평가통과. 근거 `results/lpwm_card_budget_measured_v4/resumed_batch8_status_20261005.json`. 아래v2/v3시험중표시는과거이력.

최신: 실측workspace 여유192MiB+rounding64MiB/expandable_segments를 쓰는 v4 queue421603으로교체,진입점 `configs/lpwm_planning/card_budget_measured_v4/queue.json`. v2 queue408762/v3 queue413191은시험후종료. 두시험은47.449/47.456GB에서기존추가여유512+128MiB에걸렸으며reserved-unallocated584→252MiB. 현재v4에서8본학습가능여부시험중. LoRA4422 보존원본불변.

최신 2026-10-05 09:45 KST: 사용자 모든 후속작업batch8/전체48GB 요청으로 새queue408762 시작. 진입점 `card_budget_batch8_v2/queue.json`, 상태 `outputs/lpwm_card_budget_planning_v2/queue/queue_state.json`. 이전queue1902774 CPU만종료, LoRA worker정상SIGINT 저장4422 완료. 현재새LoRA8profile→4423재개→검증→Adapter8우선→검증→full8우선(2095재개)→검증. 완료Partial결과재사용. 등록기존source38/config20불변.

최신 2026-10-05 09:36 KST: LoRA GPU당 batch4×누적2×2GPU=유효16, 4352/4707 update. Queue1902774/torchrun3292771/worker3293934·3293935 정상. Partial batch8 학습·최종검증 완료 후 방법별 profile에서 LoRA batch8이 allocator OOM으로 탈락하여 batch4 자동선택. 현재도 batch8이라는 해석은 틀림. 이번에는 읽기전용 실행점검 및 문서기록만 수행, runtime 변경 없음. 근거 `results/lpwm_48gb_planning_v1/batch_verification_20261005_0935.json`.

최신질문은PDMS SOTA문헌조회. 이번턴학습상태재점검/추가GPU작업/queue변경없음. 기존4방법학습대기열유지.

최신10/5 01:59KST: Partial3184/4707(67.64%)까지정상진행. Queue1902774/torchrun1978684/worker1979560·1979561유지. 읽기전용상태조회이며runtime변경없음. `results/lpwm_48gb_planning_v1/status_20261005_0159.json`.

최신추가질문은LPWM대planner효과분리방법설명. 기존4방법queue/source/config를변경하지않았고새대조군학습미실행. 마지막상태점검은아래00:47KST기록.

최신10/5 00:47KST 읽기전용점검: Partial1712/4707(36.4%), 기존queue1902774/torchrun1978684 정상. Batch8본학습계속/후속LoRA·Adapter·full대기. Runtime설정변경없음. 근거 `results/lpwm_48gb_planning_v1/health_review_20261005_0045.json`.

재개 후 확인: 1616/4707update, loss유한/OOM0, 최근새로그평균2.749s/update. `results/lpwm_48gb_planning_v1/post_resume_status.json`.

최신10/5 00:40KST: 사용자48GB 실행요청에 따라 **queue1902774 / torchrun1978684**로 전환. Partial1576 model+AdamW를 보존하고 batch8×accum1×2GPU=16으로1577부터 본학습재개. 새상태 `outputs/lpwm_48gb_planning_v1/queue/queue_state.json`. 기존queue1675463/1709131과training1602577은종료·대체. 아래46GB/기존PID문장은이전이력이다.

최신48GB 가능성 질문은 계산/상태조회만 수행. Partial1312update 확인, 기존 두 queue와46GB상한은 유지.

최신10/5 00:14KST: 두 queue와partial 학습 계속 실행,1232/4707update. 이번 워커/배치 검토는 CPU 입력 비교와 읽기 전용GPU조회이며 현재 학습/queue/config는 변경하지 않았다.

**최신 2026-10-04 23:56 KST**: Partial torchrun1602577를 재시작 없이 queue1675463이 인계했다.
GT ON 자동 실행은 제거됐고 partial→검증→LoRA→검증→paired비교가 실행된다.
후속 queue1709131은 `four_method_sequence_v1.json`으로 앞 queue를 기다리며 Adapter→검증→full재개→검증을 실행한다.
Full은 사용자의 최신 지시로 기존2095 model+AdamW를 별도 출력에서4707까지 이어간다. 기존 pause 자산은 보존한다.
23:53KST partial912/4707, 최근3.66초/update, 오류/NaN 없음. GPU0·1 전체약36.44/36.31GB.
LoRA/Adapter/full GPU 본학습은 아직 대기 중이며 속도/결과가 확인되지 않았다.


- **2026-10-04 23:10 KST:** 사용자의부분학습대LoRA차이질문에현재코드/원논문을대조해설명했다. 기존부분queue running/첫조건240update·source20개/config불변을파일로확인. LoRA추가/학습재시작없음.

- **2026-10-04 23:07 KST 상태점검:** 부분미세조정queue1601034/torchrun1602577 유지, 첫조건metric_plus_world192/4707update(4.08%) 진행. 새학습/추론/설정변경없이기존로그·프로세스·GPU·등록hash를확인했다. 최신근거 `results/lpwm_partial_planning_v1/status_review_20261004_2307.json`.

- **2026-10-04 22:54 KST 최신: 사용자 승인 Stage2 부분 미세조정 별도queue1601034 시작.** 설정 `partial_output_layers_v1.json`, 상태 `outputs/lpwm_partial_planning_v1/queue_state.json`. 첫 조건gradient/freeze/causal감사 통과, torchrun1602577에서 본학습48/4707update확인(22:58KST). LPWM출력계층5.56M+planner2.21M, 전체75,297navtrain/조건당1epoch/4,707update, GPU0·1 batch4×accum2=16, 객체GT보조loss off→검증→on→검증 순서. 기존 full2095update와pause/source/config는 보존하며 이전실행을재개한것이아니다. 최신사용자승인은새부분실험에만적용된다.

- **2026-10-04 22:23 KST 사용자 요청으로 Stage2 일시중단. 자동 재개 금지.** `metric_plus_world`2,095/94,140 update(0.4451epoch), model+optimizer 저장 완료. Queue1131167/torchrun1135635/worker1135737·1135738 모두 host ps에서 종료 확인, GPU compute 목록에서 우리 worker 없음. `outputs/lpwm_object_future_planning_v3/user_pause_status.json`이 현재 상태다. 아래 진행 중 문구는 중단 전 이력이며 사용자 재개 요청 전 신규 GPU 작업/후속 조건/queue를 시작하지 않는다.

- **2026-10-04 21:57 KST 실행 명세 확인:** 첫 조건1,888/94,140 update, 0.4011epoch. Queue1131167 유지, source14개/config hash 일치, 오류 marker 없음. 사용자 요청은 현재 학습 방식의 상세 설명이며 이번 턴 새 GPU 진단/학습/추론/설정 변경 없음. 최신 근거 `results/lpwm_object_future_planning_v3/training_execution_review_20261004_2200.json`.

- **2026-10-04 21:32–21:44 중간점검:** 기존queue1131167/torchrun1135635의첫조건본학습계속,1,744update시점0.3705epoch/94,140. Failure/stopped없음·source14개/config불변. 별도고정128dev/31recording 진단은완료됐고GPU반환,학습은중단·재시작하지않음. 최신시점은 `results/lpwm_object_future_planning_v3/health_check_20261004_2132/post_diagnostic_training_status.json`.

- **2026-10-04 최신: Stage2 본학습 시작, queue1131167 / GPU0·1 / `kjs-lpwm-stage2`.** `object_future_joint_v3.json`, `outputs/lpwm_object_future_planning_v3/queue_state.json`; `metric_plus_world` update16/94,140 본학습확인. 이어 검증→`metric_object_future_plus_world`→검증·paired어블레이션. GT채택은미확정. Batch2/GPU×누적4×2GPU=16,20epoch/75,297train. LPWM1e-6/planner3e-4/SSL0.02. 사용자46GB제한은**각GPU전체VRAM**(다른사용자포함),CPU RAM제한아님. 과거실패v2queue재개금지/현재queue중복기동금지/source수정금지.

- 2026-10-04 14:55 KST: 완료된 frozen readout의 전체/종류별/크기·거리·장면별 결과 해석 완료. 저장 summary만 읽었으며 추출·학습 재실행 없음. 현재 상태 판독의 일부 개선, 위치·희소 클래스 병목, GT 위치 조건부 범위를 연구 문서 첫 절에 정리. Stage2 미시작 유지.

- 2026-10-04: 설명 중 frozen current-state readout queue complete 확인. 두모델각30,871clip/312,611객체관측 추출 및CPU판독완료,LPWMupdate0/Stage2미시작. 새자동학습없음. `completion_review_20261004.json` 및shared summary확인; 완료queue재실행금지.

- 2026-10-04 14:41 KST: 검증절차/주석범위확인. 공개29,828/적응30,340 of30,871clip,오류없음/LPWMupdate0. 상태근거 `results/lpwm_object_readout_validation_v1/status_and_annotation_scope_20261004.json`. Runtime/queue변경없음.

- 2026-10-04 14:33 KST: 최신 사용자 방향은 Stage1 SSL 유지, GT 객체 감독은 Stage2 planning+world 보조 loss에 한정. 직전 Stage1 GT 보정 제안 철회. 기존 frozen readout queue 계속 실행(공개18,692/적응19,076 of30,871, 오류없음/LPWMupdate0), runtime/config 변경 없음.

- 2026-10-04 14:20 KST: 사용자 검증 방법 승인에 따라 전체 frozen object readout 진단 실행 중. Queue504595 / 공개GPU0 worker504614 / 적응GPU1 worker504615, `kjs-lpwm-object-validation`. 30,871clip/312,611객체 관측; 추출 후 CPU 판독·paired report 자동 실행. 현재적응4,356clip/223초/peak2.69GiB, LPWM update0. Stage2는 계속 미시작. 등록 source/config를 실행 도중 변경하지 않는다.

- 2026-10-04: 사용자 요청 객체 구분 검증 설계 완료. Frozen state probe / 실제 alpha 기반 instance 분리 / temporal association / causal future / 학습된 planner 개입의 다섯 검사를 기존 연구 문서 첫 절에 제안. 새 학습·feature 추출·mask 주석·gate 변경 없음. Stage2 미시작 유지.

- 2026-10-04 13:30 KST: 사용자지적에 따라 객체검증의정의를확인/정정. LPWM detection/semantic segmentation 단계는없고,실패는자체top16 particle–GT박스기하proxy. 등록값/실패상태보존하되필수gate타당성미확립을문서에명시. 분해작업은저장NPZ/GT구조와GPU여유조회까지;추가추론/학습/Stage2없음.

- 2026-10-04 13:20 KST: 사용자요청 Stage1 최종결과 재확인 완료. 저장된 공개/적응 각7,745고유clip 누락·중복0 및 checkpoint metadata/gate SHA 일치 확인. 원래15/16통과·객체box gate실패로 Stage2차단 유지. 재학습·GPU평가·기준변경 없음.

- 2026-10-04 13:00 KST: 사용자요청 planner 문제의식/방법론 재정리. 원문 문헌과 현재 코드/등록설계 비교 후 `docs/lpwm_planning_experiment.md` 첫 절에 제안 저장. 후보 선택 오류를 기준으로 미래 표현 효용을 검증하는 방향이며 runtime/config/queue 변경·새학습 없음. Stage1 실패 gate에 의한 Stage2 차단 유지.

- 2026-10-04 12:42 KST: 사용자요청 E2E 안전/refinement 선행연구 감사 완료. 공식 소스 pin+VAD/UniAD 핵심loss CPU검사만 수행. Stage2 gate 차단 유지, LPWM runtime/학습/queue 변경 없음.

- 2026-10-04 12:29 KST: Stage1 전체평가 완료. 원래 적응gate의 object_box_recall_noninferiority 실패로 queue의 stage1_gate가 exit2, 의존Stage2 차단. 추가causal/noncollapse/coverage 전부통과. Checkpoint보존, 강제진입/기준완화/새학습 없음. 이번 요청은 안전BCE/충돌좌표loss 의미 설명.

- 2026-10-04 12:06 KST: planner loss/SG/문헌 대조는 읽기 전용 코드 감사. Queue는 Stage1 전체학습·검증 대기 상태이며 Stage2 본학습 미시작. Runtime/config/평가/queue 변경 없음.

- 2026-10-04 11:53 KST: 배경particle/Stage2재배치가능성에대해현재소스와손실을읽기검토. 실행중학습·평가·queue·시각화코드변경없음.

- 2026-10-04 11:46 KST: 사용자 particle/사각형 의미 질문은 저장시각화와코드읽기로답변. 실행중평가/queue/source변경없음. 아래진행률은해당시점의기록이다.

**2026-10-04 11:42 KST 진행/ETA 점검:** Stage1 전체개발평가 공개/적응 각3248/3208/7,745clip. GPU0 PID3569927/GPU1 PID3569928 정상, 기존supervisor919150/queue1481082 유지. 학습20epoch는완료, Stage2는gate대기.

**2026-10-04 11:21 KST 최신:** Stage1 20epoch/28,920update 완료, training_summary.json 및 최종checkpoint 확인. 기존supervisor가 공개GPU0/적응GPU1 전체7,745개 개발평가를자동시작했다. Stage2는적응gate대기. CPUteacher준비완료. 이전학습중상태는당시기록.

**2026-10-04 11:17 KST 재점검:** Stage1 28,912/28,920(99.97%) 계속진행. GPU0/1 모두100%,약36.1GiB사용/11.3GiB여유. queue1481082 정상. CPUteacher준비완료102,199유효장면(train75,165/dev27,034); parent종료정상. 아직전체Stage1 gate미완료.

**2026-10-04 01:05 KST 상태 점검:** Stage1 7,280/28,920 update, epoch5/20 개발평가 완료. GPU0/1 PID920132/920133 정상, utilization87/100%, 카드당약36.1GiB 사용/11.3GiB 여유. queue1481082 및 CPUteacher1481089 유지.

**2026-10-03 23:00 KST Stage1 설명 감사:** 본학습 update2,832/28,920 유지, 새 queue는 `waiting_for_stage1_full_training_and_validation`. 실행 source hash 모두 등록과 일치. 최신 설명은 `docs/lpwm_planning_experiment.md` 첫 절.

**2026-10-03 22:49 KST 최신: Stage1을 유지하면서 planner/검증 queue와 CPU teacher 준비를 실행했다.**
- Stage1 supervisor919150 / torchrun920080 / ranks920132,920133, `kjs-lpwm-stage1` 그대로 유지.
- 마지막 확인 update2,432/28,920, loss21.9190, 1.619s/update. 본 GPU학습 중단·추가 GPU실험 없음.
- Stage1 batch4/GPU×누적2×2GPU=16, FP32, worker0. train23,126/dev7,745,20epoch.
- 새 queue PID1481082, CPU teacher PID1481089, CPU16worker. `outputs/lpwm_metric_planning_v2/queue_state.json` 및 `candidate_teacher.log` 확인.
- 현재 queue는 기존 Stage1 완료·전체개발평가를 기다린다. CPU는 전체102,373 planning 장면의512후보 공식채점 정답/metric cache 준비 중.
- `configs/lpwm_planning/metric_distillation_v2.json`: metric_plus_world / imitation_plus_world / metric_refinement_plus_world, 각20epoch/94,140update.
- 원래 Stage1 supervisor는 종료 후 `launch_lpwm_full_planning.py`를 부르며, active_pipeline.json dispatcher가 새 queue에 join. 중복 GPU학습을 시작하지 않는다.
- Stage1 적응gate+8장면 causal검사/coverage/noncollapse →teacher gate→planning gradient/GPU profile→조건별 학습/개발/world검증→독립navtest.
- **Stage2 본학습/PDMS는 아직 없음.** 후보oracle ADE 및 CPU1update 연결검사를 성능으로 해석하지 않는다.
- queue_registration.json에 실행source hash 고정. 시작 후 runtime코드 변경은 queue source 검사를 실패시키므로 임의 수정하지 않는다.
- Gate 실패 시 queue_failed.json 진단 저장/의존작업 차단. Stage1 학습은 새 queue가 소유하지 않으며 중단하지 않는다.

### 아래는 이전 실행 이력 (위 최신 상태와 구분)


**최신 방향 수정: LPWM encoder·context·dynamics·planner 공동 학습.** 사용자가 원본 context/dynamics를 제외한 선택을 지적하고 공동 학습을 요청했다.
기존 encoder-only는 완료7run(frozen3/planning3/uniform seed29)만 보존하고 우리 worker3425264/3425265/scorer3425266을 SIGINT 종료했다.
`outputs/lpwm_planning_v1/superseded_by_joint_world_model.json`을 따른다. 이전18run queue/finalizer를 자동 재개하지 않는다.
새 코드 `src/planning_aware_future_prediction/object_centric/lpwm_joint_world_planner.py`: 공식 encoder6.035M/context39.389M/dynamics59.869M+planner0.821M, RGB decoder만 제외.
과거2영상→관측transition posterior→미래8step은policy prior만으로 autoregressive rollout, activation checkpointing으로 gradient 보존.
GPU0 batch1 3update에서 planning/future loss 각각 세모듈gradient>0, 미래label변경시예측동일을 통과했다. bf16 batch4 profile 진행/새 데이터·규모 준비 중이다.
기존512train 중13개/192dev 중2개가 공식navtrain token필터밖임을 발견했다(로그는전부navtrain). 새 학습은 공식token까지엄격필터한다.

**LPWM 등록 학습·평가 종료.** 6개 모델/총1800update, 8모델×30clip 평가를 완료했다.
PID2583673/2583803 queue는 PILOT_VARIANT_DONE으로 종료, 추가 읽기 전용 2개 재평가도 종료했다.
GPU 학습 peak10.145GiB, 합계 학습807.39초. 현재 새 학습을 실행하지 않는다.
파일럿 source/config/checkpoint/metrics는 outputs/lpwm_navsim_adaptation_v1/와 results/에 보존한다.

아래는 이전 완료 작업의 이력이다.

**최신 시각화 완료:** `scripts/visualize_encoder_planning_results.py`로 저장 결과만 CPU에서 렌더링했다.
PNG `outputs/encoder_future_learning_v1/visualization/`, 공유 PDF/감사 JSON `results/encoder_future_learning_v1/visualization/`.
새 학습·모델 추론·PDM 재계산·GPU 사용은 없다. 그림 3장 모두 시각적으로 확인했다.

**현재 실행 중인 우리 학습·평가는 없다.**
본 36회와 추가 12회 모두 512 update에서 종료했고, 공식 개발 PDM·future probe·raw 영상 재추론을 완료했다.
`outputs/encoder_future_learning_v1/`, `outputs/encoder_future_additional_controls_v1/`에 체크포인트와 로그가 있다.
각 train worker와 `results/`의 completion.json, trained_inference_audit.json은 완료 상태다.
추가 평가의 마지막 validation 경로 오류는 공용 prefix cache 위치를 참조하도록 수정하고 저장 점수로 복구했다.
학습이나 PDM 재실행 없이 종료했으며 완료 launcher를 다시 실행하지 않는다.

**아래는 종료된 이전 실행 이력이다.**

**현재 실행중인 작업 없음.** 등록27run과CPU PDM192장면×34조건모두종료했다.
출력 `outputs/drive_jepa_selective_future/region_research_v1_20261003/`,
`region_research_pdm_v1_20261003/`, `region_research_preserved_controls_v1_20261003/`.
학습PID3270611/평가PID3289792/보존대조PID3381536은완료됐다. 완료작업자동재개없음.
집계보고서와검증은 `results/drive_jepa_region_research_v1/`에공유한다.

새region실험15run과후속위치진단모두완료/GPU작업없음. 새출력 `outputs/drive_jepa_selective_future/spatial_region_selection_v1_20261003/`.
중단시 같은config/output에 `train_drive_jepa_spatial_regions.py --resume`; optimizer/scheduler/RNG/완료update복원.

현재 우리 학습 작업은 없다. 87run 종료 후 사용자 요청으로 CPU 시각화를 생성했다.
진입점 `scripts/visualize_drive_jepa_learned_modules.py`, config `learned_modules_visualization_v1.json`.
재현은 새 output-directory를 지정한다. 기존 cache/checkpoint/결과는 덮어쓰지 않는다.

최신: 최초69run과 선택진단은 모두 종료. 별도 matched-conservative-selection18run을 등록했다.
진입점 `configs/drive_jepa_selective_future/matched_conservative_selection_v1.json`,
`run_drive_jepa_overnight_sequence.py --series matched_selection --detach`.
상태 `outputs/drive_jepa_selective_future/matched_conservative_selection_sequence_20261003/active_stage.json`.
기존 낮은LR learned6run을 재사용하고 같은800update의18개 대조만 새로 실행한다.
09:00 또는2시간/메모리guard/실패stop; 새 cache·heldout/navtest·WA/pilot 없음.
아래는 이 밤샘 작업의 앞선 실행 이력이며 실제 실행 여부는 각 completion/stopped marker를 우선한다.
기존 checkpoint 진단은 완료(149초/1.571GiB). 단순 과적합으로 확정하지 않고 gradient결합,
fusion 강도, 선택 이동, 목적함수를 분리한다. `latest.pt`에25update마다복구상태저장/--resume제공.
Raw경로 `outputs/drive_jepa_selective_future/overnight_causal_followup_v1_20261003/`.
후속표본704window(train512/64recording, 추가dev192/24recording)를CPU에서사전고정했다.
`overnight_recording_coverage_v1.json`/`overnight_coverage_training_v1.json`과
`scripts/run_drive_jepa_overnight_sequence.py`가첫실험완료후단일GPU1에서cache→27run학습→export를이어간다.
09:00마감/실패·pressure자동retry없음; old128cache 재사용,9GiBcache상한,heldout/navtest불사용.
실제상태는 `outputs/drive_jepa_selective_future/overnight_sequence_20261003/active_stage.json`을본다.
01:14KST queue PID2134994, cache child2135322를확인했다. PID는재사용가능하므로추후실제명령/상태재확인.
원격push는credential socket오류로실패. 로컬commit과공유결과디렉토리는보존한다.
후속cache704개완료:301.14초/신규6,042,394,944bytes/reused128/allocated1.207GiB/원본hash불변.
추가futureprojection3조건×3seed와보수적LR/메모리정규화4조건×3seed를별도등록했다.
현재27run후에GPU1순차실행하며no-new-cache/동일800update/pairedbatch/09:00상한을지킨다.
`overnight_projection_sequence_20261003/`는coveragequeue완료를기다린후실행하며동시GPU점유하지않는다.
전체등록69run/46800jointupdate이며시간·메모리·실패stop우선.끝나면알려진결과파일만scope한commit/push시도.
01:43KST 두 번째 queue PID2296713의 coverage 완료 대기를 확인했다. 등록 커밋4355bf2.
현재 실행 상태는 각 queue의 active_stage.json/completion.json/stopped.json을 우선한다.
기본격리환경CUDA접근실패는업데이트전이며호스트실행으로전환했다. CPU130tests통과.
새 projection identity 테스트는 처음에 no-grad fast path와 grad-enabled path를 비교해 실패했다.
동일 no-grad 조건으로 바로잡은 뒤 bitwise 통과했으며 gradient 검사는 별도로 통과했다.
01:52KST coverage queue 완료, 두 번째 queue가 projection child2389185를 시작했다.
첫48run의 window별 집계/paired batch/원본값/aux gradient 경계를 별도 CPU 검사로 확인했다.
실험 감사 진입점 `scripts/audit_drive_jepa_overnight_evidence.py`; 전체69run 종료 후 한 JSON으로 기록한다.
학습 종료 뒤 train32recording의 단일 patch 교체/선택 W gradient 근사를 비교하는 읽기 전용
진단을 등록했다. `run_drive_jepa_overnight_sequence.py --series surrogate --detach`.
학습 횟수는 늘리지 않으며 해당 queue는 projection/conservative queue 완료 뒤만 GPU1을 쓴다.
아래 완료/진행 문장은 해당 과거 실행 이력이다.

추가 구조 비교와 학습 후 실제 영상 checkpoint 검증을 완료했고 GPU 프로세스는 종료했다.
GPU1만 사용, 원본 planner/teacher 고정, LoRA는 별도 future branch tail만 학습했다.
Raw `outputs/drive_jepa_selective_future/architecture_followup_v1_20261002/`,
공유 `results/drive_jepa_selective_future/architecture_followup_v1_20261003/`.
기존 cache192개를 재사용하고 current-prefix만402.9MB 추가했다. 공용 데이터는 읽기만 했다.
학습 후 off 출력은 동결·warmup을 맞추면 원본 bitwise 동일, 원본 weight hash 불변.
각 run의 stage별 checkpoint에 optimizer/scheduler/RNG/완료 update를 보존했다.
Runner는 새 출력만 허용하고 resume CLI는 아직 없다. 완료 학습을 반복하지 않는다.

**최신 선택 비교 진행 중**: `scripts/run_drive_jepa_selection_comparison.py` GPU0 단일 process.
Raw `outputs/drive_jepa_selective_future/selection_comparison_v1c_20261002/`.
FP32 cache192window/2,014,131,648bytes 완료. 각조건 결과·optimizer/RNG/checkpoint를 별도 저장한다.
V1/v1b는 학습 전 bitwise gate에서 중단되어 학습·cache0; 파일 보존.
Singleton batch grid stride를 공식 reshape→permute와 맞춘 v1c는 bitwise gate통과.
아래 실행 현황은 과거 이력이며 최신 완료 여부는 위 raw 결과에서 확인한다.

**현재 GPU작업 없음**: WA 평가/queue/guard/aggregate는중단됐다. Drive selective-future 실제연결진단도완료/프로세스종료했다.
WA pause marker/16JSONL/새압축snapshot/부분CSV·JSON를보존했다. snapshot SHA256
`6aed87ec3feea7ecb7c9813ab0855af3fea84c3036e21f5b894eadb14f71249b`.
원본Drive planner와encoder를freeze/eval하며runtimeinput gradient는유지한다. 새패키지설치/환경수정없다.
Conda 실행별칭 `/rhome/junseong/envs/kjs-drive-jepa-extension/bin/python`은기존Drive전용prefix의symlink다.
실행 `scripts/validate_drive_jepa_selective_future_connection.py`, 새local결과 `outputs/drive_jepa_selective_future/`.

**아래는 중단 전 이력**: bounded dense평가. 20:40KST 사용자권한 확인/16JSONL SHA256검증 후pause marker를acknowledged로보존.
공유 `results/official_wa_jepa_reproduction/shared_gpu_resume_state.json`; 과거pause/snapshotJSON는 역사적기록으로유지.
Dedicated tmux `planning-aware-wa-jepa`: memory_guard/bounded_scheduler/aggregate와동시에최대7worker(0:5/1:2).
`scripts/schedule_official_wa_jepa_workers.py`는 14-waypartition을변경하지않고같은GPU의다음shard를순차기동.
현재구간0·2·4·6·8→GPU0,1·3→GPU1; 끝나면남은짝수/홀수구간을같은모델로처리.
Profile v2/CPUqueue만교체하여기존4worker를adopt했고guard/aggregate는유지했다.
Runtime현황 `outputs/official_wa_jepa_reproduction/bounded_scheduler_status.json`,
현재queue로그 `bounded_scheduler_gpu0_increase.log`, guard `memory_guard_shared_gpu.log`, 집계 `aggregate_shared_gpu.log`.
Pressure marker `memory_pressure_stop.json`이생기면추가기동/자동재시도없음; 원인검토후사용자와재개범위를정한다.
새사용자pause는우선이며기존pause script가marker를만들고등록된우리worker만정상중단한다.
모델·공식score설정·Conda prefix/원본데이터/pilot/Drive결과는보존. 학습은없다.

추론병목진단완료: [module timing](docs/wa_jepa_inference_module_timing.md), 공유module_timing_summary.json.
Shareddense한scene encoder0.545s/predictor24.105s/12step당2.009s/model24.698s(97.6%predictor).
Fixedsparse ID생성9.918ms/packing27.957ms, shared조건이라purekernel이나동일점유speedup으로해석금지.
이전단독6scene측정predictor6.413s/전체6.694s=95.8%. Diagnosticprocess종료/모델·평가결과수정없음.

2026-10-02 WA preparation: 전용Conda clone/공식stage2·Meta encoder 다운로드 완료.
Checkpoint SHA/구조, 공식navtest12146·4-view71488files/cache completeness와호환성CPU검사통과. 5tests통과.
원본agent key/shape검사일치확인; GPUsmoke진행중. LD_LIBRARY_PATH에전용prefix/lib를프로세스별로지정한다.
원본reference bec2966 보존, 실행worktree404d8af. GPU0의 타인CARLA는건드리지않고 GPU1우선 사용.
재개: `runtime/environments/wa_jepa_official_evaluation/bin/python scripts/evaluate_official_wa_jepa.py preflight`.
GPU는CUDA_VISIBLE_DEVICES=1/CPU스레드1로 `smoke`, gate 통과 후 `full`. 우리 학습은없다.
공유결과 `results/official_wa_jepa_reproduction/`, 재개scene기록 `outputs/official_wa_jepa_reproduction/`.

**완료 이력상 우리 학습/평가는 종료됐고 이번 작업에서 새로 띄우지 않았다.** 시작 HEAD578be6e/작업트리clean을확인했다.
완료 DrivePF 결과·source·checkpoint·config·scorer·원본CSV는그대로보존했다.
공식 DrivePB/WA reference소스와학습·추론경계를읽고작은공개metadata만조회했다.
저장navtest12146scene current speed/command 현황을CPU16.29초에집계, metadata누락0/원본CSVhash불변.
진입점 `results/foundation_selection/`. 전체평가반복/새cache생성/학습/GPU사용없음.
Pilot·확대·residual·SafeDrive는계속보류. 다음구현은사용자범위결정후별도승인된단계로진행한다.

아래는 보존된 **이전 공식 평가 실행/안전정리 이력**이다. 당시 baseline HEAD와시간을바꾸지않는다.

**최신 지시 적용 / 안전 정리**: 기준 HEAD `5c6e6d5`, 시작 시 미커밋 변경 없음.
호스트의 junseong 프로세스를 확인했고 자체 pilot 학습/cache/평가 실행은 없었다.
따라서 종료한 프로세스 없음. 다른 사용자의 GPU4–7 작업 및 대화 세션을 건드리지 않았다.
완료 pilot 결과·373 cache·checkpoint·3620-window manifest는 보존한다.
200-window profile/확대 cache/확대 학습/held-out 평가 및 WA-JEPA full inference는 미실행·보류.
Pilot 재개는 별도 사용자 승인 후 기존 config·checkpoint·optimizer/RNG 상태로만 검토한다.
**이번 작업 종료점은 공식 Drive-JEPA 재현 보고이며 pilot/WA-JEPA를 자동 재개하지 않는다.**
공식 source `548bb82`의 독립 Conda/worktree, full PF ViT-L planning checkpoint·공식 metric cache 준비 완료.
세 파일 공식SHA256일치, `pip check`와공식scorer import통과, 3scene full planner/encoder strict·finite·scorer smoke통과.
공식split12146token/log136/front14247image/cache12146, 누락0. Old split도동일12146이며 oldCSV의12147행은average행 포함이다.
**GPU0·1 전체 평가 완료 / 실행 중인 우리 작업 없음**: 각각6075/6071scene,68log,worker2개(총4개).
21:14:44→21:27:05 KST, 약12분21초. 공식12146scene 전부valid/finite, 실패·누락·중복0.
PDMS89.224320 vs 논문v2 Table2 PF89.0, +0.224320점. 임의 허용오차로정확재현성공을단정하지않는다.
두raw CSV의average행을제외한원본scene mean을사용했다. 실제Hydra model/scorer/센서/split이pinned설정과일치.
Raw `outputs/official_drive_jepa_reproduction/full_navtest_gpu{0,1}_v1/`와`logs/`보존.
Shared `results/official_drive_jepa_reproduction/full_navtest_results.json`, `official_scene_scores.csv`, `execution_cost.json`.
보고서 `docs/official_drive_jepa_reproduction.md`에명령/hash/하위metric/차이감사/검수링크를정리했다.
완료후전용tmux/gpu0·1우리compute process없음(25MiB baseline). Worker증가·재시작·추가평가/학습없음.
Pilot·WA-JEPA 자동재개금지. 다음활용방향은사용자와결과검토후결정한다.

**실행 중인 학습 없음.** 기준 `607da52`의 373-window cache와 1000-update checkpoint를 재사용했다.
CPU 기준선·실단위 미래 오차, 한 가지 visual-only residual 대조를 대응 3-seed/새 5000-update로 완료했다.
Recording 3-way manifest, 4표본 JPEG 민감도, 공식 WA-JEPA attention의 CPU gradient 검사도 완료했다.
Raw: `outputs/pilot_foundation_decision/`, 공유: `results/pilot_foundation_decision/`.
사용자 후속 지시에 따라 **predictor 튜닝·pilot 데이터 확대·확대 학습은 보류**한다.
전처리 gate는 미해소. 200-window encoder profile/확대 cache/확대 A/B/E/F/held-out 모델 평가는 미실행이다.
**다음 작업은 공개 future-planning 기반 재현 후 선택·예산 가설로 직접 복귀**다.
WA-JEPA는 source/weight 메타데이터/tiny attention만 확인했고, full weight 로딩·공식 추론은 아직 안 했다.

현재 호스트는 `user-ESC8000A-E11`. 승인 GPU **0·1**, RTX A6000 약48GB.
조사 초반 타인 CARLA 점유가 있었지만 종료 직전 확인은 각각25MiB/compute process 없음이다.
위 점유 문장은 이전 pilot 조사 당시 이력이다. 이번 공식 평가에서는 점유 재확인 후 GPU0·1을 사용하고 해제했다.
다음 실행 시 다시 확인하고 타인프로세스/환경은 건드리지 않는다.

이전 중단 작업: O0 epoch 1 / F3 epoch 0 checkpoint라는 인수인계가 있다.
실제 checkpoint 내부 epoch와 resume 적합성은 이번에 검증하지 않았다.
**O0·O1×4를 재개하지 않고, F3/F4도 현재 실행하지 않는다.**
큰 cache 재생성이나 SafeDrive 재학습은 새 연구 방향을 확인한 다음 별도 결정한다.

## 2. 최근 결과와 조사 사실

최신: `results/lpwm_drivor_planning_path_lora_v1/`에 실제 loss2update와DDP2update 증거를보존했다.
273adapter모두 output-gradient>0/unused0,head3개와prior·attributeCNN에서currentgeometry연결확인.
초기 native 출력/초기planner동일, 원래가중치·buffer digest a5dd2345…5d 유지.
Linear229/Conv44 LoRA4,683,650/전체학습21,753,728개; DDP카드최대40,161,509,376bytes.
이는연결·실행검증이지객체이해/주행성능향상결과가아니다. 본학습공식PDMS/EPDMS아직없음.


Epoch1 96scene×4cam×64particle: 중심이동평균.302314px/중앙값.207698/p90 .658991/max3.975736,
1px초과3.6377%;크기축평균변화2.02025%/중앙값1.35067%;presence절대변화평균.046125.
Bitwise동일하지않지만대다수geometry시각적변화작음. Roadproxy presence18.4076→17.1646%,차량9.4928→9.6553%.
Readout전체F1 .379859→.372994;appearance F1 .292610→.310808. 미래2/4s current대비CI0포함.
12scene미래반복/순서역전개입은선택궤적oracle score를낮추는초기결과;학습분포/OOD개입/작은표본이며전체실패판정금지.
자료 `results/lpwm_drivor_epoch1_particle_review_v1/`: 실제전후PNG2개,geometry/readout/summary/개입결과.
Geometry adapter 후보는head당2Linear/rank8→4·4·1/57,633추가param,LPWM총LoRA1,401,121.
실제공식loss에서세headgradient양수/현재출력직접연결/원본SHA보존/zero-init동일/원래planner초기동일검사통과.
GPU0·1batch16×accum2×2 DDP2update통과,카드실측최대30.6163GB/29.6149GB. 본학습아님.
모델객체순회Linear234/Conv2d70개전체경로CSV/JSON을 `results/lpwm_drivor_geometry_lora_v1/`에보존.

공식 DrivoR는 매 epoch validation 설정과 fit의 val loader 연결이 있으며, 최종 v1/v2 예산은25/10epoch다.
DiffusionDrive 공개 recipe100epoch, VAD base48+12epoch/매epoch저장/각단계끝평가 확인.
1epoch는 초기 진단·제한 예산 비교에 유효하나 최종 방법 순위·수렴 판단 근거는 아니다. 상세 epoch1 review 문서에 출처 추가.

Epoch대기제어CPU검사3개통과: latestatomic교체시열린inode보존,기존사용자pause보호,epoch불일치resume거부.
1614state에model/AdamW/scheduler/rankRNG2개/다음epoch위치모두있어나머지24epoch재개가능.
25epoch schedule warmup3322update여서첫epoch은아직warmup. Epochs1로바꾸면동일스케줄비교가아님.
현재navval도학습포함이므로진단96scene/DrivoRtraining분포비교를독립validation으로보고하면안됨.

1400개동기화update 로그의loss/trajectory/score 값NaN·Inf0. 두rank모든학습영역gradient양수,1400LoRA .14196/.08853/.20810.
두rank평균first100→last100: total28.0402→4.9325,trajectory24.3438→2.4956,scoreloss3.6963→2.4368.
서로다른training배치평균이므로validation성능/PDMS개선으로해석하지않는다.
표현1000: 96장면현재중심초기대비평균.2508px/최대3.1373px이동(FiLM경로),객체전체14D F1 .3475(initial.3799).
미래2s/4s current대비readout오차차이CI는여전히0포함. 본학습진행정상과표현효용확인은구분한다.
Main83/queue250/parallelism4sourcehash불변확인. 500/1000진단summary/readout/complete/등록사본을results에보존.

실제100update LoRA84/84tensor변경. Main gradient interaction/context/dynamics .03895/.04668/.06041, 두rank동일.
LoRA OFF current xy bitwise동일;FiLM OFF current xy initial과bitwise동일;둘OFF 전체attrs initial과동일.
Current xy의84LoRA gradient는모두None,FiLM4개연결. 위치생성뒤LoRA는current xy를직접학습하지못함.
LoRA OFF future xy RMS0.492px변화(3장면)는future 계산의존성이지정확도개선이아님.
실제planningloss양수gradient와좌표projection autograd검사를혼동하지않는다. Native freeze 및checkpoint SHA보존.

Particle 중심/scale은 appearance RGB glimpse 추출에 사용됨을 native feature encoder 코드로 확인했다.
최신 monitor 노란점=현재중심,점크기=presence,색box=GT투영;planner중요도/attention그림이아니다.
Current+future14D(background포함) projection→4particle평균→planner이므로 점과planner token은1:1이아니다.

새 진단96장면/24recording, straight42/left30/right24, object-camera views2974.
100update: 현재중심 평균0.00305px/최대0.05446px 이동, 분포 거의 동일. Encoder-only 명령변경 궤적0→0.2834m.
전체14D 객체readout F1 .3799→.3778, 미래2s L2 3.8933→3.9021m/4s8.0206→8.0322m. 아직 개선 미확인.
실제 명령 경로는 작동하나 적합성/미래정보/기여는 별개; 초기0.25%warmup만으로 실패로 판정하지 않는다.
Readout fit18recording/2286views, eval6recording/688views; 양쪽 모두 upstream trainval에 속한다.
공개 초기값: encoder-only 명령변화0, planner-only 평균궤적변화0.3523m; zero-init FiLM으로 예상된 기준선이다.
초기2s/4s 미래 readout의 current 대비 차이는 CI0 포함. 아직 학습된 미래정보 유용성을 입증하지 않았다.
4개 CPU 기하/연결검사 및 96scene 실제 planner replay/고정후보 score 동일성 통과, main native SHA 보존.
Fairness audit: DrivoR LoRA589824 vsLPWM1343488, defaultFP16 vsBF16+encoderFP32,
drop_lastTrue/False 및 backend 생성순서 차이 추가확인. 공식recipe 유도v1 updates40325 vs현40350;
v2 13290 vs13300. 저자 실제로그 숫자가 아닌 source/standard sampler 기준 계산이다.


LoRA초기출력/native출력bitwise동일,초기planner동일,실제optimizer2update후원래LPWM가중치+buffer SHA불변.
LoRA각영역/FiLM/plannergradient양수및실제변경,DDP검사통과. Native109,545,263고정,LoRA1,343,488/전체trainable18,413,566.
Ego navtrain8+navval8장면을공식FeatureBuilder와비교:11차원오차0. 공식Linear11→256+generator/scorer더하기그대로.
추가command4D FiLM은LPWM attribute CNN conv_in에적용;공식DINO에는없음. 비교시통제할차이.
실측GPU8/16/24 각각16.30/29.59/43.00GB,steady31.56/18.29/19.35초. 16+loader4/oracle8평균19.93초.
단기실측으로전체최적화나성능향상증거아님. 근거 `results/lpwm_drivor_lora_v1/`와새LoRA보고서.

공개LPWM+공식DrivoR joint 연결검사통과: batch8단일GPU2update/2GPU누적2update,
imitation/BCE각각native위치·scale·presence·영상encoder·prior·dynamicsgradient확인.
BCE→generator좌표는공식DrivoR의detach로차단. GT/longer원본builder오차<3.1e-7,
변환trainingcachevsfresh공식cache7subscore차이0(2장면),testTrue/False차이0.
공식v2.2warmup220캐시·두단계score집계연결검사통과(0궤적engineering검사,LPWM성능아님).
새LPWM모델PDMS/EPDMS아직없음. DrivoR는DINOv2공개pretrained+Q/VLoRA32,planner신규학습이다.

Stage1없음PDMS78.9161/ADE1.37286/FDE3.34235,Stage1적용82.5238/1.16347/2.78246.
Stage1PDMS+3.6077점CI[+1.5065,+5.8539],ADE−.20939m/FDE−.55988m. 복원LPIPS.76725→.30433/미래.80444→.39225.
이번동일조건개발비교에서Stage1효과확인,Stage2미세조정추가효과미확인이라는이전결론은유지한다.

Stage1효과공개고정조건 optimizer update는4707/4707완료. 직전동일범위frozen검증634.823초를기준으로
최종1024planning/256world검증·paired비교완료23:05–23:15KST예상(공유부하·저장/검사에따라변동).

DrivoR paper v2/공식fc6e5aa 확인: 외부DINOv2 pretrained→NAVSIM 단일공동학습.
기본DINOv2고정/Q·V LoRA32+새camera register+두decoder학습; WTA L1+6BCE, 복원SSL/직접객체loss없음.
후보→scorer는detach지만scoreloss→공유perception은연결된다. 논문Stage1/2는NAVSIM-v2평가단계다.
최종v1 navtrain+navval25epoch/v2 navtrain10epoch, batch16×4/AdamW2e-4. 상세docs/lpwm_planning_experiment.md최상단.

Loss-balance로그감사:현재L_plan+.02SSL/globalclip5. Gradient실측기록만집계해partial37/37,Adapter31/37,
full재개20/20에서합산norm>5. Partial실측norm중앙87.96. Loss크기와gradient기여는구분하며SSL지배는미확정.
근거results/lpwm_planning_loss_balance_review_20261005/existing_training_audit.json. Loss10배가실제update10배라는보장없음.

일부계층·전체에서 위치/크기/presence head가 trainable로등록되고 Stage1대비 실제weight변경도 확인됐다.
동일8장면512particle최종중심이동 partial평균.55063/최대1.71375px,full평균.46071/최대2.52902px.
초기geometry/입력RGB정확일치,고정군3시점불변. 전중후/고정비교/확대갤러리/weight감사:
results/lpwm_geometry_finetuning_visualization_20261005/. Full0은원래v3저장snapshot,중후는2095에서이어받은실행이다.
Planning+SSL공동학습변화이며planning단독원인/유용객체재배치/PDMS추가이득으로해석하지않는다.

Adapter현재좌표/크기/presence nativehead고정,interaction출력은현재좌표/크기를덮어쓰지않음(공식encoder확인).
현재geometry변화는주로명령FiLM경로;interaction/context/dynamics adapter의변화를현재점이동만으로판정불가.
Planning-only감사gradient는encoder그룹(interaction포함)/context/dynamics각.030906/.020353/.225220으로연결확인.
Adapter−frozen PDMS−.03859점CI[−1.16421,+1.00706],ADE−.00834mCI[−.04080,+.02423]:추가이득미확인.

Adapter 대 Stage1 적응 고정군의 동일 개발 8장면·512 particle 비교 완료: 최종 중심 이동 평균0.08466px,
최대0.34762px(128×128기준),크기 평균절대차0.27184px,presence 평균절대차0.00492.
입력RGB·초기geometry 정확일치,고정군 중심/크기/presence 0·2353·4707update 정확불변.
대표PNG/전중후/변화량/PDF·8장면 확대 갤러리: results/lpwm_adapter_vs_frozen_particles_20261005/.
뚜렷한 객체방향 재배치는 미관측. Feature/미래정보 변화는 이 geometry 시각화만으로 판정할 수 없다.

동일seed반복표현계산batch8/16/32의32scene시간3.199/2.942/2.785초. 큰배치출력이batch8과달라cache가속미채택.
자체반복출력차이0,혼합particle최대차이16=1.464/32=1.216으로physicalobject오차와구분.
Detached SSL monitor생략planninglogits동일,gradient최대차이0.000488은동일조건반복에서도발생.
짧은공유GPU진단으로장기훈련동일성을주장하지않고현재monitor도유지. 진단optimizerupdate0/본학습변경없음.
근거results/lpwm_stage1_effect_v1/frozen_batching_repeatability_benchmark.json및training_eta_20261005_2144.json.

과거다섯조건의summary·global_batch_size로그·resume provenance감사:모두planning16/world8/4707update유지.
Partial4×2→8×1(update1576경계),LoRA4×2→8×1(4422경계),Full2×4→4×2(2095경계),Adapter/Frozen전구간8×1;
GPU모두2개다. 물리배치변경으로효과배치가커진실험은아니다. Dropout/SSL샘플링/반올림차이와full스케줄차이는남는다.
근거 `results/lpwm_card_budget_measured_v4/batch_history_audit_20261005.json`.
공개고정조건CPU/GPU실검사:initialplanner기존고정군과정확일치,공개checkpoint weights/buffers정확일치,
2optimizerstep후world/FiLM/particle불변,미래보조입력독립,planner명령반응양수. 공개조건planning성능은아직없다.

최종 Frozen PDMS82.5238/ADE1.163467/FDE2.782463. Partial/LoRA/Adapter/full 대비 차이는
미세조정-minus-frozen 기준 -0.8897[-2.2018,+0.2507] / -0.6097[-1.4261,+0.1113] /
-0.0386[-1.1642,+1.0071] / -1.2956[-3.2034,+0.3145] PDMS점. 모두0포함하므로 추가이득미확인,
동등성·무효 일반화·대조군 우월성의 입증도 아님. Adapter ADE -0.00834m[-0.04080,+0.02423]도 불확실.
Frozen 최종artifact에서 LPWM·FiLM weights/buffers hash 불변 검증, world LPIPS는 Stage1과 약1e-7차이.
Frozen 미래→관측반복 교체 PDMS82.5238→79.7709,차이+2.7529[1.7547,3.8131]점은 기존 미래표현 활용진단.
Stage1 자체 우월성/객체정보보존/표현미세조정추가효과와 구분한다. 1024동일장면/유효PDMS1021/40recording,
world256 동일성 확인. 근거 `results/lpwm_frozen_control_v1/completed_comparison_20261005/summary.json`.

Frozen 최종학습summary:5226.38초(약1h27m),trainable2,207,495/LPWM0,peakallocated3.906GiB,LPWM·encoderFiLM가중치/buffer hash불변true. Full peakallocated20.735GiB대비감소는world전체no_grad로역전파activation/gradient/AdamWstate가필요없어진설정에부합한다. 같은GPU당batch8/유효16학습량유지. 근거results/lpwm_frozen_control_v1/batch8/metric_plus_world_training_summary.json.

Adapter평균PDMS82.4852/ADE1.15513/FDE2.75672최상,LoRA81.9141/1.18220/2.84057;partial81.6341/1.28250/3.04333;full81.2282/1.29596/3.01611. 여섯PDMS쌍CI모두0포함. 미래LPIPS변화partial+11.764%기준실패/LoRA-0.103%/Adapter+0.385%/full+0.802%. 근거completed_four_method_review_20261005/summary.json.

완료 내부개발 PDMS: partial81.6341/LoRA81.9141/Adapter82.4852/full81.2282. Full의 등록5개검사 통과, partial은미래LPIPS유지기준미달. Frozen GPU감사/배치8/평가구동검사통과, LPWM trainable0/FiLM고정/planner2,207,495개; 로그LPWM각gradient0/planner양수. GPU0·1각총약26.4GB, allocated약3.91GiB. 근거 results/lpwm_frozen_control_v1/status_20261005_2025.json.

등록 sequence는 현재 학습→최종개발검증/네방법집계→freeze GPU 감사/배치실측/평가구동확인→frozen-LPWM 동일planner 학습→검증/paired비교다. 대조군은 Stage1 LPWM 체크포인트와 seed47의 초기 planner로 시작하고 world/encoder FiLM 전체를 eval/no_grad로 고정한다.

사용자 확정 연구 순서: (1) 같은 LPWM에서 planning 표현 적응의 추가 효과 검증 완료, (2) 공통 DrivoR planner 아래 구조화된 particle 미래 표현 vs 일반 압축 feature 비교. 두 번째는 목적과 통제 원칙만 기록했고 실행을 등록하지 않았다.

표현 비교의 planner 고정은 구조·초기값·학습 규칙을 맞추고 조건별로 planner를 학습한다는 뜻이다. 센서·해상도·관측 이력·memory 길이/차원·loss/gradient까지 맞춰야 한다. DrivoR backbone vs LPWM은 사전학습 차이가 남으므로 particle 속성의 단독 효과와 구분한다. 연구 문서 최신절에 비교별 해석 범위를 정정했다.

DrivoR register도 planning loss로 중요 영역에 특화된다. LPWM의 차이는 위치/크기/외형 속성 구조·명시적 future rollout·SSL 목표다. Native particle ID는 patch ID이며 객체 ID 아님. 현재 명령 FiLM은 particle attribute encoder에 적용하고64×12=768memory를 쓴다. 상세 docs/lpwm_planning_experiment.md 최신절.

전체 111,757,238개 = 학습 가능 111,757,238개. LPWM109,545,263 / planner+command2,211,975. 128 update마다 측정하는 gradient 검사에서 encoder/context/dynamics/RGB decoder/planner 모두 양수. 기본LR1e-6/3e-4, loss=imitation+metric BCE+0.02SSL. RGB decoder는 SSL 경로에서 갱신된다. 근거 results/lpwm_card_budget_measured_v4/full_parameter_scope_20261005.json.

ETA 근거: results/lpwm_card_budget_measured_v4/full_training_eta_20261005_1624.json. 최근 경과시간 차분으로 중간 저장·검증 시간을 포함해 추정했고, 이전 최종 평가 실측은 11.8–28.4분이다. 현재 loss는 유한하며 GPU0·1 점유 각 약43.45GB, batch4×누적2×2=유효16 유지.

실제Stage1 CPU감사통과: Adapter와초기planner/출력차이0,optimizer2step후world·encoder-FiLM가중치/buffer/hash·particle표현불변,world모듈gradient0/planner6.94398,planner명령반응/미래보조입력독립성통과. 실행조건음성검사7개통과. 근거results/lpwm_frozen_control_v1/queue/cpu_audit.json및registration_status.json.

Adapter 최종 개발1024/유효PDMS1021·world256: PDMS82.4852/ADE1.15513/FDE2.75672. LoRA대비PDMS+0.5711점CI[-0.3703,1.5020]로우월성미확정. 미래LPIPS0.393758(Stage1+0.385%) 유지기준통과. 예측미래→관측반복시PDMS79.6509,차이+2.8342점[1.1428,4.5169]. 근거results/lpwm_card_budget_measured_v4/completed_adapter_review_20261005/summary.json.

2026-10-05 15:22 KST: full 재개검사·gradient·engineering 통과, 본학습loss유한. LPWM1e-6/planner3e-4 기본LR,전모듈학습+SSL유지/직접객체GT없음. GPU전체각43.45GB. 상세results/lpwm_card_budget_measured_v4/full_resume_status_20261005_1522.json.

2026-10-05 배치 확대 검토: 현재기준GPU당8×누적1×2=유효16이며12/16증설은유효24/32로바뀐다. Adapter 완료 직전약2.1s/update,입력준비수ms. 여유GPU1실측용script를준비했으나sandbox NVML조회실패후host조회에서pjh복귀를확인해추가GPU실행하지않음. 근거results/lpwm_available_vram_throughput_20261005/capacity_review.json.

2026-10-05 pjh 이력 확인: 실제 경로 /rhome/junhyeok/miniconda3/envs/pjh-wamvla-v2/bin/python. 09:35:49 GPU별 프로세스/UUID 대응 확인. 이후 OOM로그에도 같은PID가 있으나 로그조회시각을 생존시각으로 해석하지 않는다. 총점유10:52→14:34 감소는 기존근거이며 정확 종료/부재시간은 여전히미확정. 근거 results/lpwm_card_budget_measured_v4/pjh_process_history_review_20261005.json.

14:34카드각23591MiB,현재compute각23558MiB우리계정만. 저장10:52공동점유각45.89decimalGB 대비감소. GPUaccountingDisabled/queue최신resource덮어쓰기로정확종료시간미확인. 과거snapshot에소유자정보없어특정인의종료시각으로단정금지.

공식64와128config존재;확인한modelzoo공개weights는모두128. 현재Sketchy무수정경로128. 256은가중치이식+적응검증대상,직사각형은정사각형가정수정추가. 이번턴고해상도실행검증없음.

확인해상도(W×H):Drive-JEPAfront512×256/3viewstitch1024×256,DiffusionDrive3viewstitch1024×256,VADtiny640×360/base1280×720(padding전),DrivoR1148×672×4. LPWM128은더강한축소. DrivoR dev feature압축16k→64token PDMS90.2→90.0은pixel축소실험아님.

공개128설정은patch수/glimpse/head차원/bg투영/decoder/buffer에연결되어입력resize만수정불가. LPWM개념의128상한은아님. 상하28crop·종횡비왜곡은우리전처리선택이며LPWM필수요구아님.

입력은CAM_F0 1920×1080→상하28px crop→1920×1024→INTER_AREA128×128. 종횡비비보존. 캐시152495×128×128×3 uint8,모델[0,1]. Stage1/SSL12프레임,Stage2 planning4프레임동일128. hparams image_size128/normalize_rgbfalse.

Stage1 devSSL loss epoch15→20:19.88686→19.69678(-0.956%),후반둔화. LoRA128monitor는3072/4096/4707 PDMS81.4273/84.5584/83.6380,ADE1.12896/1.11915/1.14796. 후반학습loss소폭감소·개발변동으로추가학습효과/수렴미확정. 근거convergence_review_20261005/summary.json.

재확인: Stage1 epochs20/completed_updates28920/seen_train_clips23126. 완료Stage2 LoRA epochs1/completed_updates4707/train_clips75297. 두단계모두1epoch라는해석은틀림.

1seed는독립학습1회이며1epoch는현재학습분할1회순회다. 현재Stage2는초기방법선별,최종수렴증거아님. DiffusionDrive100epoch/DrivoR v1 25·v2 10epoch확인;반복seed수미확인. 기존LoRA CI는recording변동이며학습seed변동미포함.

최종LoRA PDMS81.9141/ADE1.18220/FDE2.84057 vsPartial81.6341/1.28250/3.04333. PDMS차+0.2799점 CI[-0.7341,1.3308]로우월성미확정;ADE차-0.10030 CI[-0.17094,-0.02787],FDE차-0.20276 CI[-0.37366,-0.02727]. 미래LPIPS Stage1.392249→LoRA.391846(-0.103%,유지통과),Partial.438393(+11.764%,유지실패). 근거results/lpwm_card_budget_measured_v4/completed_lora_review_20261005/summary.json 및comparison.png/pdf.

Adapter는Stage1에서독립초기화,LoRA완료가중치사용안함. Particle interaction1/context4/dynamics6 총11block 뒤 LN→Linear(64)→GELU→zero-init Linear 잔차추가. NativeLPWM전체고정/Adapter704960+planner·command2211975=2916935학습. 초기particle/metric출력차0,진단update후frozen해시동일,planning·SSL gradient검증통과. 근거 results/lpwm_card_budget_measured_v4/adapter_training_and_architecture_20261005.json.

LoRA검증의initial/trained/persistent_future JSON은생성됐고world/최종summary는아직없다. Queue실패marker없음. 이번턴은상태확인만수행했다.

navtrain103288 중 Stage2 train75297(72.900%)/dev27076(26.214%)/timestamp제외915(0.886%). Recording122/40개,train/dev 중복0. Stage1 complete12RGB train23126/dev7745는각 Stage2동일분할의부분집합이며recording분할동일. 현재Stage2검증128monitor/1024planning/256world로dev전체27076을평가하는것아님. 근거 results/lpwm_card_budget_measured_v4/navtrain_usage_breakdown_20261005.json.

LoRA completed_updates4707/epochs1/train75297/effectivebatch16. 최종checkpoint SHA54b893143b62e5dc35cc5552109c1622a68c11c55ed617ad77436025ab9c98c6. 최종loss6.27299 유한. 재개후본학습batch8, 카드최대47.598GB/guard없음. 최종개발검증 미완료이므로 monitor83.64를1024최종점수로해석하지않는다. 근거 results/lpwm_card_budget_measured_v4/lora_training_completion_20261005.json.

DrivoR fc6e5aa: navtrain filter103288token/1192log segment. 기본 train_logs978/val_logs214개로 중복0; competition train1192개로 val214도 포함. 비캐시 full training은 같은 navtrain filter에서 train_logs|val_logs 합집합을 사용. navval은 여기서 학습용 풀 안의 개발 검증 부분이며 별도 데이터셋 아님.

문헌 확인: DiffusionDrive navtrain100epoch→navtest; DrivoR navval 어블레이션 후 v1 competition split(navtrain+navval)25epoch→navtest. NuScenes planning은 공개 val 비교도 사용한다. NAVSIM v1 navtest/PDMS와 v2 navhard_two_stage/EPDMS를 구분한다. 상세와 원문 링크는 docs/lpwm_planning_experiment.md 최신절.

공식navtrain103288/navtest12146token대조: 평가panel1024모두navtrain,navtest0. Stage2manifest train75297/dev27076,train/dev token·recording중복각0. Monitor128/최종개발1024 유효PDMS127/1021. Current automatic_navtest=false 및queue공식test단계없음확인. 근거results/lpwm_card_budget_measured_v4/evaluation_split_clarification_20261005.json.

Partial world256 같은장면 LPIPS 복원0.304335→0.305230(+0.294%,유지통과),미래0.392249→0.438393(+11.764%,등록10%유지gate실패). 회전/작은객체/박스겹침상황군에서도미래LPIPS악화. Partial미래particle을현재표현반복으로교체하면PDMS81.6341→76.2361,paired차이5.398점/95%CI4.139–6.680;추론입력분포교란검사이므로미래예측학습·LPWM미세조정의독립효과단정금지. Stage1전체7745의LPIPS는복원0.770675→0.300016/미래0.807938→0.382994,SSL영상적응확인이나객체/완전적응증거는별개. JSON/PNG `results/lpwm_card_budget_measured_v4/interim_review_20261005/`.

최종v4 batch8 profile8update통과,최대카드47,598,010,368bytes(<48GB),peak allocated23.8352GiB,마지막6회평균3.48327s. 본학습4448 update_seconds=3.293,loss=6.57218. Source42/config29 hash일치,CPU9검사통과,원본4422 model+AdamW226state보존. 현diagnostic module_gradients 빈dict는재개후다음128배수gradient진단전상태이며gradient0이아님;profile8의세world영역+planner norm모두유한양수.

두실측에서allocator를제외한추가CUDA사용량은약122MiB였으므로192MiB+64MiB로여유를측정기반조정. v4에서도물리48GB전체감시/같은scientific설정유지. 이전실패/중단로그보존하며v2/v3자동재기동금지.

LoRA4422 checkpoint SHA04efad377910390cb4ba947eb5b410f5972438d10199eb2b071b1bac7e104f61, AdamW226state 모두step4422. 새card budget이 다른사용자와CUDAcontext를 고려해allocator예산을산출하며48decimalGB카드상한0.5초감시. 고정23.2GiB·22.75GiB·free3GiB삭제; 계산용workspace512MiB+rounding128MiB만유지. CPU6검사통과(예산,8우선,후속모든방법설정,기존optimizerfork동일update). 실제8profile/재개수치는다음상태기록확인.

최신 2026-10-05 09:35 KST: LoRA8 profile은 물리free4.63GiB/카드전체45.935decimalGB 상태에서 프로세스별23.20GiB allocator allowance에 도달하여20MiB allocation 실패. 카드전체48GB 초과나 물리메모리 고갈이 아님. LoRA4 본학습은 우리process GPU당14452/14456MiB, 카드전체34658/34663MiB, GPUutil100/84%. Partial 최종1024dev(유효PDMS1021)/256world 검증은 PDMS81.6341, ADE1.28250m; reconstruction 유지통과/forecast LPIPS 유지실패. 전체검증통과로 보고하지 않는다. Queue는 사전등록된 독립 방법비교 지속 설정에 따라 LoRA로 진행했으며 Partial checkpoint를 LoRA에 이어받지 않는다.

NAVSIM-v1 논문최고확인치95.1: DrivoR+SimScale+Traffic-Element Awareness(arXiv2608.18035Table2). iDriveVLA는94.95/공개리더보드1위를저자보고(2609.30818). TOAD94.9/ChainFlow94.85/DriveSuprim93.5도원문확인. Live HF순위행은접근확인불가로논문보고치와실시간순위구분.

최신3072monitor PDMS81.8502/ADE1.28197m/FDE2.97124m;2560의82.3708/1.39076/3.28219대비PDMS0.52점하락·ADE/FDE개선. 고정dev128/유효PDM127.2353중간시각화저장. Loss/gradient유한/OOM0/해시38source20config일치/VRAM총45.6GB.

효과분리설계: optimizer의planner그룹에는particle생성에영향주는ego-FiLM도포함됨. A(world+FiLM고정,planner학습),B(world고정,FiLM+planner학습),C현재부분학습을구분. C-B가native LPWM적응의조건부추가효과. C와동일한학습계층을SSL만으로갱신하는D와비교해야planning gradient자체의효용검증가능. 모두제안이며실험결과없음.

최신1536dev128/유효PDM127: PDMS79.1126(1024의76.6794대비상승), ADE1.5267m/FDE3.5224m(직전1.4031/3.2788대비악화). 최신monitor는batch변경1576이전, 변경후첫검증2048대기.1664실제gradient유한·양수/frozenRGB0. 평균trainloss8.7177→6.9469, world학습loss안정이나held-out보존검증전.

최신48GB 실측: 동일저장상태에서batch4/8 각8update, 뒤6회평균4.4078→2.9921s(32.12%시간감소). Batch8 카드전체최대관측45.5921GB/물리free최소4.9521GiB/tensorpeak22.0778GiB,OOM0. Worker0유지(기존0/2/4/8CPU비교에서증설이득없음). Checkpoint/AdamW보존CPU검사3개및4planning/2world engineering평가통과. `results/lpwm_48gb_planning_v1/execution_review.json`참조.

48GB 상한 검토: batch8 중심 추정47.07–47.20GB는 수치상 들어가지만 상한 여유0.80–0.93GB뿐이며, 추가 workspace1GiB를 포함하면48.14–48.28GB다. 예상 free3.45–3.58GiB로 현재6GiB guard도 충족하지 못한다.6GiB는 우리가 정한 보수적 운용 여유이며 물리적 불가능을 뜻하지 않는다. 현재코드 GB는10진(48GB=44.70GiB). 질문에대한계산검토만수행했고 실제batch8/제한변경없음.

최신워커 실측0/2/4/8: CPU 로딩0.746/1.946/1.699/1.582ms/update(2회평균, GPU전송/연산 제외). 실제입력 준비0.01408/전체3.742초=0.376%. Batch8 추정22.091GiB allocated, 현재카드총47.20/47.07GB로46GB/6GiB reserve조건초과. 현재batch4/accum2/worker0 유지.1024monitor PDMS76.6794%, ADE1.40314m. 상세results/lpwm_throughput_review_20261005/ 및연구문서 최신절.

최신: partial512update 고정128dev 모니터PDMS70.8503%, ADE1.86759m, FDE4.20983m(유효PDM127).
초기random planner2.41294%/8.45013m 대비 학습 진전이며 LPWM 적응만의 이득은 미분리.
Partial trainable world5,558,389/planner2,211,975. LoRA world1,343,488, Adapter704,960, full109,545,263.
LoRA/Adapter/full CPU causal/gradient/SSL 검사와 관련7개unit test 통과. GPU검사는 각 조건 직전 대기 중.
Full2095 checkpoint와746 AdamW state 복원/원모델 출력 차이0 검증. `to_logvar`는 현재Identity로 파라미터0.
근거 `results/lpwm_four_method_queue_v1/launch_and_health_check.json` 및 연구문서 최신절.


- 부분계층선택과LoRA갱신방식은별도축이다. 현재는native출력12모듈직접학습이고중간attention LoRA와의직접비교없음. W고정이어도LoRA유효W+BA로encoder표현변경가능;activation/backward비용은적용위치에좌우된다. 예시1024²행렬의rank8은학습파라미터1,048,576→16,384이며64배속/총메모리64배절감주장은아님. 원논문/공식구현과현계산그래프해석을연구문서에구분해기록했다.

- 23:07점검: 최근128update의wall평균3.843초/update, 초반loss11.309→최근7.540(서로다른train batch의로그). Update128기준encoder/context/dynamics/planner gradient유한·양수, RGBdecoder0. Failure/stopped없음/등록source20개·config불변. GPU전체36.444/36.311GB/사용률83·84%,우리peakallocated12.073GiB. 첫학습후개발모니터는아직미실행(초기0만완료)이므로PDMS개선판정없음.

- 본학습48update/유효768scene입력확인. 초기128dev 모니터완료(PDMS2.41294, 유효127/128; 무작위planner기준). 기록된정상update평균3.496초, peakallocated12.073GiB, GPU전체36.444/36.311GB<46GB. NaN/OOM/중단marker없음/source20개불변. 첫512update모니터예상23:25–23:30KST, 조건당학습약4.6시간+검증/공유부하여유. 근거 `results/lpwm_partial_planning_v1/launch_verification_20261004.json`.

- 부분학습OFF총학습7,770,364/111,757,238개(6.95%), LPWM5,558,389/109,545,263개(5.07%). Nativeattribute/feature/interaction/context/dynamics출력head만갱신; RGBdecoder와Transformerblocks고정. FiLM을attribute CNN conv_out뒤로이동해고정CNN앞부분backward를줄였다. LoRA없음. LPWMLR1e-5/planner3e-4, SSL0.02유지.
- 동일유효batch16/SSL8 profile: batch2누적4 6.36–7.07초/peak7.04GiB; batch4checkpoint해제는1update후allocated21.5GiBguard초과로중단보존; batch4누적2/checkpoint유지선택은5update완료·마지막3회평균3.755초·peak12.06GiB. 이전full7.75초와시점이다른짧은비용비교다. 초기화depth_head=None오류(0update)도보존. 근거 `results/lpwm_partial_planning_v1/engineering_review.json`.
- CPU개발panel격리/균형/순서불변·teacher결측paired처리2검사통과. Profile가중치로4planning/2world평가경로실행완료(성능결과아님). 새queue첫감사: planninggradient encoder1.26456/context.09443/dynamics2.67883/planner19.17081/decoder0, futureGT교란prediction/logit차이0, 명령particle변화6.98e-5. 1진단optimizer후모든frozenparameter SHA동일/grad없음/optimizer중복없음확인. 진단weight는미저장·본학습미사용.
- 이전등록source14개와pausedcheckpoint SHA aef7ab37…4b9bac 불변확인. 새학습전config/source20개 및개발planning1024/world256각40recording패널등록. 본학습실제결과와GT효용은아직미확정. 이전full결과/조건과새부분실험결과를합치지않는다.

- 일시중단 체크포인트는22:22:56 atomic 저장, reason=`signal`, update2,095, optimizer746개 state 전부step2,095, source14개/config hash 일치. SHA `aef7ab37bfdfcf3a7c032774b0aaf4f4bbd76e8f84849a24a5b5fac2664b9bac`,1,342,277,703bytes. 학습 elapsed16,275.53초. `latest.pt`와 `user_pause_20261004/checkpoint_update002095.pt` hardlink 보존, 원본Stage1/기존결과불변. 마지막 progress 로그2,080보다 signal checkpoint가 최신이다.
- Pause marker를 queue가 확인해 정상 signal 저장을 요청했다. `queue_failed.json`의 `RuntimeError('Queue paused by marker')`는 이 구현의 의도된 중단 기록이며 수치/OOM 실패가 아니다. Training stopped marker와 완료 sentinel은 없다. 자동 후속 작업 차단, 실제 GPU 재개 시험은 실행하지 않음. CPU에서 checkpoint 로드/optimizer 일치 검증 완료.

- 실제 구성 재확인: 관측4장/ego8D→명령 FiLM particle encoder→causal 미래8step→768개14D particle token→2층256D 후보 scorer. 고정512개 train medoid, soft L1-distance imitation+6metric BCE+0.02공식ELBO. 첫 조건 object/refiner 비활성, 두 번째만 object GT auxiliary. World branch는 별도12장 posterior 복원+전이KL이며 past-only RGB rollout loss가 아니다. 조건당 전체20epoch, planning16/SSL8clip 샘플 per update. 새 수치 평가 없이 기존 실행 코드를 대조한 설명이다.

- 중간점검: 초기16–256vs최근1504–1744로그평균total9.6775→7.0208/imitation5.9555→4.5907/metric3.3568→2.0600/world18.2615→18.5083. 128update단위gradient모든모듈유한/양수;첫update큰gradient와clip5유지. Saved1,536의746optimizerstate와실제모든모듈weight변경확인(LPWM상대L2약.05–.07%,planner16.8%).
- 같은저장본1,536의고정128dev/31recording 간이평가: 미학습planner→학습후ADE8.4098→1.5844m/FDE17.6252→3.7796m/PDMS1.9397→78.5124. 초기무작위planner대비진전이며강한baseline/LPWM표현/GTaux효용증거아님. 전체dev/world유지검증미완료. 본학습VRAM각35.2GB,진단중최대36.489GB<46GB,진단287.9초/.661GiB완료. 최근7.75秒/update,첫epoch정기ADE/FDE예상10/5 04시전후.

- Stage2진입은새명시적 `results/lpwm_object_future_planning_v3/stage1_admission_amendment.json`. 원래top16박스proxy실패는보존,나머지15기준과causal/noncollapse통과 및readout부분개선/사용자승인에근거한통제실험진입이다. 완전적응·미래객체이해·PDMS개선의입증아님.
- CPU14검사통과. 미래objectloss단독gradient encoder.2705/context.0666/dynamics1.1311/planner·aux.4065/RGBdecoder0. 미래GT교란trajectory/logit차이0/명령particle차이>0. GT보조head는loss전용;추론GT입력없음. 전체target102,373record/1,216,641현재객체관측;기존판독3,223객체와현재box차이0/상태최대3.82e-6.
- Batch4는공유GPU용20GiB allocatorcap에걸려OOM(당시free6.91GiB),실패보존. Batch2누적4의5update검사통과/peak11.88GiB/steady6.9–7.6초. 본학습update16검증전체VRAM각35.9GB(46GB미만),입력준비0.03–0.11초로worker0사용. 본학습초기예상조건당7.5–8.5일+검증,2조건순차. 시점별 `launch_verification_20261004.json` 참조.

- 2026-10-04 결과 상세: combined 차량 F1 67.13→73.26/보행자47.08→50.72/자전거1.15→1.32(×100). 자전거133/359 정답이나 예측19,807건으로 오탐 다수; encoder·class-balanced linear probe·불균형 원인 분리 필요. 작은 box macro-F1 33.64→36.18,40m이상32.97→34.77. 회전은분류상승에도위치오차악화;겹침우선분류에따른그룹편향과CI없음 명시. 미래·native instance·PDMS 검증 미완료.

- Frozen readout완료: appearance macro-F1 .32018→.38669,geometry+appearance .38453→.41766(분류CI없음),GT-ROI-only .55473로더강함. Combined y MAE+0.044m(CI양수),vx−0.053m/s(CI음수),x/depth/vy차이CI0포함. 조건부현재정보의일부개선이며native객체분리·미래·PDMS개선아님. 로컬/공유summary동일/전체추출완료를확인.

- 현재probe는 vehicle/pedestrian/bicycle 3종. 원본sample log에는traffic_cone/barrier/czone_sign/generic_object도확인. 주석은3D oriented bbox(x,y,z,length,width,height,heading),class,3Dvelocity,instance/trackID. 2D ROI는카메라투영이며pixelmask아님. 전체클래스빈도조사결과로일반화하지않음.

- Stage1 객체 GT 감독을 추가하면 label taxonomy/누락/수에 편중될 위험이 있다는 사용자 지적을 반영했다. SSL 사전학습의 장점이 GT fine-tuning으로 반드시 전부 사라지는 것은 아니지만, Stage2에서도 의존성과 forgetting 위험은 남는다. World SSL 유지·unannotated unknown·직접box/개수/presence강제 없음·GT auxiliary 유무/라벨량 대조를 새 원칙으로 문서화했다.

- GT box를 encoder 필수입력으로 사용하는 안보다 train-only 객체/미래 감독으로 사용하는 안을 권고했다. 현재정보가 유지되면 Stage2 planning+world+object/future 보조감독 비교, 현재정보가 부족하면 Stage1 추가적응 우선. 아직 이 학습 변경은 제안이며 실행되지 않았다. VAD§3.4의 중간감독+planning 공동학습을 원문 대조했다.
- 신규 검증: 23,126train/7,745dev clip, 객체 관측train222,646/dev89,965. Train recording98fit/24validation, dev40. 선형ridge 7입력대조, state5항목/class3, 정규화강도train내선택/이후전체train적합, 상태오차recording CI. CPU5검사와GPU4clip실행검사통과. 최종probe점수는 아직 없음.
- Decoder 감사: 공식 encoder64/decoder선택30이며 새 판독기는 full64를 decode한 alpha로 GT-localized pooling한다. Native30 복원mask와 구분하는 amendment를 결과 확인 전 공유 JSON에 기록. 현재 검사를 native instance segmentation으로 해석하지 않는다.

- LPWM 원문은 비지도 keypoint/box/mask 발견을 제안하며 explicit tracking을 제거한다. 따라서 semantic head 부재를 객체 분해 불가능으로 해석하지 않고, particle index를 track ID로 해석하지 않는다. Dittadi ICML2022의 segmentation/object-property prediction 분리 평가와 공식 linear/MLP probe를 참고했다.
- 후속 검사는 GT-localized 정보 판독과 자동 instance 발견을 분리한다. 전체64 feature/alpha 신규 추출이 필요하며, 현재 확인한 cache의 projected box는 pixel instance mask GT가 아니다. 수동 검수 mask 평가셋은 제안 단계다. Probe/새 객체 통계/PDMS 결과 없음.

- 공식decoder RGBA/alpha합성 확인: mask는복원기여이지semantic/GTinstance segmentation보장아님. 평가box는particle position/scale로만든glimpse영역;presence상위16→GTvehicle/pedestrian/bicycle 투영box와class-agnostic Hungarian IoU≥.1. Category는모델출력이아닌GT평가분류.
- Top16절단/presence선택/객체부분표현/일대일matching때문에proxy하락만으로객체정보손실·도메인적응실패를입증하지못함. 이지표의필수gate사용근거부족을인정하고현재상태해석정정. 원인분해미완료.

- 최종검증 재확인 `results/lpwm_navsim_full_posttraining_v2/validation_review_20261004.json`: causal forecast LPIPS .382994 vs persistence .399487, 객체ROI MSE .029812 vs .042921. 현재box IoU.1 recall은32.20→28.93%이나 IoU.3은13.01→17.77%,4초미래IoU.1은19.39→26.74%. 모든객체표현악화로일반화하지않음.
- Scenario LPIPS: 등록직진126개는 .40014 vs 유지.37988로악화(CI양수),회전1289개는 .40524 vs .47328로개선,overlap5958개는 .37575 vs .38354이나차이CI가0포함. Overlap우선배정이므로직진전체의대표분석아님. 위험별gate통과는비열등성이지전계층우월성아님.

- 문헌 추가: WorldDrive는 미래latent 증류+preference ranking, ResWorld는 ego정렬 temporal residual+미래BEV refinement, CAPO는 prediction 교체에 따른 control 변화로 중요도 학습. SafeDrive/EgoFSD/ForeDrive까지 고려하면 particle/intent/future ranking/encoder joint만으로 novelty 확보 불가.
- 제안 중심질문: 같은 후보·관측·예산에서 어떤 객체/시간의 미래 정보가 경로 선택 손실을 줄이는가. DriveSuprim 계열 선택형 planner를 통제된 기준으로, fixed-bank oracle와 ranking regret로 후보/표현 병목을 분리. Future/frozen-joint/intent/utility-weighted 감독 비교는 새 제안이고 실행결과가 아님.

- SafeDrive 공식 Phase3는 TwDAC 좌표 reference를 detach하지만 shared motion/plan query는 FRNet에 연결돼 safety BCE가 SWNet decoder를 학습함. 우리 scorer는 coarse feature를 읽으므로 고유 future_refinement_decoder에 이 경로가 없는 차이를 확인. 직접 회피 좌표비용과 구분.
- VAD 공식 kernel CPU검사: collision grad norm0.790569/loss1.3→1.26875, boundary1.104161/.363616→.302604. UniAD공개532fc33 CollisionLoss는 torch.tensor(bbox[:2])로끊겨 loss2.85의requires_grad=False/수치미분2.00009. 해당공개함수의문제이며논문전체재현아님. `results/e2e_planner_safety_audit_20261004/` 참조.

- Stage1 top16 object box recall@IoU0.1 공개0.3219947→적응0.2893251, paired 차이-0.0326696/CI[-0.0415200,-0.0236445], 등록 CI하한>=-0.02 미충족. `results/lpwm_navsim_full_posttraining_v2/summary.json` 자동생성본을보존. 복원/미래LPIPS·객체ROI 및나머지위험gate 통과. Stage2 미시작.
- 안전 BCE의 unsafe정답0은 안전확률을 낮추는 감독이므로 detach 제거만으로 충돌회피 목적이 되지 않음. 직접 좌표 회피 목적에는 미래GT footprint clearance/안전pseudo경로 회귀 등의 별도 objective가 필요하며 이번에는 설명만 기록.

- Planner 감사: 512후보 soft CE+6metric BCE, 보정32개 WTA regression+6metric BCE+0.5 temporal BCE+0.01 comfort, 전체+0.02 world ELBO. LPWM/planner 경계 SG 없음. Refined pose→채점 embedding 및 CPU oracle에 SG; 공유 coarse feature는 채점기로 gradient 유지. Temporal head는 보조감독, 최종선택식에 직접 미사용. DrivoR의 완전한 생성/채점 분리를 그대로 구현한 것은 아님.

- Stage2의14D particle attribute(position/scale/presence/depth/features/bg)는detach없이planner에연결됨. LPWMLR1e-6/planner3e-4/worldELBO.02. 위치재배치는가능하나명시적객체중심배치loss/정지선감독없음. 기존aggregate gradient/command차이검사는위치별gradient나의미있는이동증거아님.

- 시각화확인:64개중심점, presence상위16개 learned-scale 사각형,색은particle index,점크기/alpha는presence. GT/검출box/불확실성/위험도표시아님. Presence sum은객체수아님,16개선택은시각화만.

**2026-10-04 11:42 KST:** 11:21의각242clip 기록과현재카운트로평가처리율/ETA계산. 공개2.381clip/s,적응2.350clip/s. 각GPU48/51%,VRAM2.6/2.9GiB는추론측정. `results/lpwm_navsim_full_posttraining_v2/evaluation_progress_20261004_1142.json`에기준시각/완료예정시각보존.

**Stage1 최종 학습결과:** dev512 loss19.696784,PSNR21.076768,dynKL4221.361,contextKL697.720. 고정8장면미래MSE .0210534로epoch15 .0230964에서추가개선; persistence .0403294. 앞선10→15정체가최종20epoch까지지속된것은아님. `results/lpwm_navsim_full_posttraining_v2/training_completion_check_20261004.json`; 전체적응성공판정전.

**2026-10-04 11:17 KST:** dev512 epoch19 loss19.6928/PSNR21.1001dB. DynamicsKL epoch5 3840.44→epoch19 4209.58(+9.61%). 고정8장면 미래MSE epoch5 .026577→10 .023120→15 .023096으로후반개선정체. 실행정상과미래학습성공구분; 최종전체평가필요. `results/lpwm_navsim_full_posttraining_v2/health_check_20261004_1117.json` 참조.

**2026-10-04 01:05 KST:** 고정dev512 epoch5 loss21.1328(초기64.8380), PSNR20.8349dB(초기13.1449), dynamicsKL3840.44, contextKL710.35. 모든 기록loss/gradient finite, 56회 module검사 양수. 최근평균1.586s/update. `results/lpwm_navsim_full_posttraining_v2/health_check_20261004_0104.json` 참조.

**Stage1 실제 목적 확인:** 12장 posterior 복원+11전이 particle/context KL이며, 4장→8장 free rollout RGB loss는 없다. 첫 epoch dev512 loss64.838→23.399/PSNR13.145→20.461dB. 고정8장면 causal forecast MSE0.053663→0.029535, persistence0.040329; 전체 적응 통과 결과 아님. `results/lpwm_navsim_full_posttraining_v2/stage1_diagnostic_snapshot_20261003.json`에 원시값/hash 보존.

**2026-10-03 22:49 KST: LPWM planner 설계 및 CPU 연결검사 (본학습 결과 아님).**
- DrivoR/DriveSuprim 공식 source clone 및 논문, Hydra-MDP/Drive-JEPA 논문, 보존 SafeDrive 코드 읽기검토.
- train-only512 trajectory medoid: train oracleADE0.35923m/dev0.34544m, devp95 0.72152m. 실제planner 성능아님.
- 512후보 일괄채점의 progress를 PDM reference와 개별 정규화, 공식개별score와 train/dev 오차≤2.14e-8. 전체준비에서도segment별4후보 parity검사.
- SafeDrive의 future BEV head는 auxiliary이며 planner입력으로 오해하지 않음. 우리refiner는 predicted LPWM future particles를 직접 읽음.
- 실제공개LPWM/학습영상 CPU1update: planning gradient encoder23.0228/context5.0042/dynamics15.0257, RGBdecoder0 (worldloss 없는진단).
- future_refiner/offset/refined_metric/temporal_safety gradient 모두>0, 미래GT교란 trajectory/logit차이0, command변경 particle평균차이0.02015.
- CPU6개 검사통과; 별도 source-hashed report `results/lpwm_metric_planning_v2/`. 진단weight저장없음.
- CPU공식cache save_buffer가 sandbox에서정체돼검증한 CPU PID3개만종료, sandbox밖재실행정상. GPU학습미중단.


**이번 세션의 확인 결과:**
- 공식 navtrain log AND token 필터, 기존40개 development recording 유지. 나머지122개 recording 모두 학습 사용.
- 이전 navtrain heldout recording도 이번 사용자 전체학습 승인으로 train에 포함됐으므로 독립평가로 부르지 않는다. navtest 학습 사용 없음.
- 공유 RGB cache152,495장/약7.0GiB. Stage1 train23,126/dev7,745; Stage2 train75,297/dev27,076.
- Stage2 ego 상태·미래 경로 생성 시 기존 공식 cached target과 max차이0.0.
- 공개 원본 Sketchy SHA6d62bf5a...부터 시작. 전체4모듈 gradient>0/weight change 검사 통과.
- batch2/accum4/worker0:1.729s; batch4/accum2/worker0:1.612s. worker2:1.627s,4:1.634s,8:1.612s,0재확인:1.624s. 각8update 중 초반2개 제외; 짧은 공유GPU 측정이며 전역최적 증거가 아니다.
- 본 학습 batch4 peakallocated29.10GiB, nvidia process약36.1GiB. 6개 처리량 측정에서 OOM0, worker증가 추가이득 미확인.
- planning gradient 독립 CPU audit: encoder4.015/context0.590/dynamics2.472/planner157.929, 미래GT교란 출력차이0, 1진단update 후 intent particle차이0.01828. 진단weight미저장/Stage2성능결과아님.
- 프로토콜/카메라기하/분산데이터 재개·RNG 검사8개 통과. Stage2 GPU경로는 gate 후 검증 예정.
- 공유 기록 `results/lpwm_navsim_full_posttraining_v2/`; 아직 전체 학습 완료나 planning 개선 결과 없음.

### 이전 완료 결과 보존


**최신 방향 수정: LPWM encoder·context·dynamics·planner 공동 학습.** 사용자가 원본 context/dynamics를 제외한 선택을 지적하고 공동 학습을 요청했다.
기존 encoder-only는 완료7run(frozen3/planning3/uniform seed29)만 보존하고 우리 worker3425264/3425265/scorer3425266을 SIGINT 종료했다.
`outputs/lpwm_planning_v1/superseded_by_joint_world_model.json`을 따른다. 이전18run queue/finalizer를 자동 재개하지 않는다.
새 코드 `src/planning_aware_future_prediction/object_centric/lpwm_joint_world_planner.py`: 공식 encoder6.035M/context39.389M/dynamics59.869M+planner0.821M, RGB decoder만 제외.
과거2영상→관측transition posterior→미래8step은policy prior만으로 autoregressive rollout, activation checkpointing으로 gradient 보존.
GPU0 batch1 3update에서 planning/future loss 각각 세모듈gradient>0, 미래label변경시예측동일을 통과했다. bf16 batch4 profile 진행/새 데이터·규모 준비 중이다.
기존512train 중13개/192dev 중2개가 공식navtrain token필터밖임을 발견했다(로그는전부navtrain). 새 학습은 공식token까지엄격필터한다.

LPWM 6run 완료: 원영상 적응 복원 MSE0.01640, 객체 점 포함31.76%, 박스IoU≥0.3 대응18.47%, 미래MSE0.03015.
공식무적응은0.05680/40.73%/17.71%/0.06934, 마지막영상유지 미래MSE0.03032다.
객체 박스 대응 개선+0.76pp의recording CI[-5.80,+6.43]로0포함. 회전보정 미래MSE0.08633이나
공통유효영역에서는raw0.03025/rotation0.03319로 시야손실이 악화의 상당부분을 차지한다.
과거4→미래4(2초), future GT/pose 누수없음. 객체 지표는투영GT있는24clip,영상지표30clip.
3개좌표검사/8모델미래교란0/두재평가기존지표차이0,6개checkpoint보존.
4PNG/3GIF/공유PDF, 논문및코드검토와결과보고서를작성했다.

**시각화에서 확인한 거리 변화:** 6블록 planning의 192개 구간에서 seed 평균 ADE 기준
113개 개선·4개 ±1mm·75개 악화다. ±1mm는 표시용이며 통계적 동등성 기준이 아니다.
4초 예측 끝점의 모델 간 이동량 중앙값은 원본→6블록 planning 6.263cm,
6블록 intent planning→강한 미래 감독 1.790mm. 이동량과 GT 오차 감소는 다른 지표다.
실제 영상/BEV/시간별 GT 오차 3사례는 seed29의 ADE 개선 최대·중앙·악화 최대를 의도적으로 선택했다.
원본 사진 token과 cache hash, cached GT의 ADE 대조를 통과했다.

**최신 결과: 48회 / 24,576 update 완료.**
원본 ADE/PDM 0.352210 m / 87.119134%; planning-only 2블록 0.347629 / 88.396320,
6블록 0.346061 / 88.718100. 2블록 ADE 차이 −4.581 mm의 recording 95% CI는 [−6.595, −2.570].
6블록 대 2블록의 ADE/PDM 비교 구간은 0을 포함해 깊이 증가의 우월성은 미확정이다.
6블록 intent planning PDM 88.437488, 균등 미래 λ0.05 88.438069, λ0.5 88.438153으로 미래 감독 추가 이득은 매우 작다.
Raw 영상 명령 개입에서 intent 모델 42개는 encoder 표현이 바뀌고 비조건부 6개는 불변이었다.
귀책 충돌 없음 점수와 우회전 명령·정지 근처 구간의 악화가 있어 평균 PDM 상승을 안전성 개선으로 해석하지 않는다.
CPU 167개 통과, 원본 보존, 48개 raw 추론 오차 최대 1.145e-5. 상세 CI·상황별·비용은 최신 보고서 참조.

**아래는 보존된 이전 결과다.**

**최신 결과:** 27run완료. SPARTAN추가연결희소화/C-JEPA마스킹/IA움직임선택의추가planning이득은확인하지못했다. 원본대비일부ADE감소는있지만current-only도동일하며PDM개선CI는0포함. 전체수치/상황별/구성요소/비용은새report참조.

CPU전체154검사 통과. K8 plain→retention은위치교체26–37%→74–89%, ADE0.347129→0.346951m이나CI0포함.
진단: planning→score연결정상; future slot교환후trajectory0.24–0.27mm변화; proxygradient4–10배/방향거의직교.
전체현재입력bypass+위치정보를섞는future fusion+현재teacher proxy불일치가원인후보. 의미GT평가아님.

2026-10-03 CPU 시각화 검증: seed29 warmup100→joint800에서 selector 선택 교체율은
ego-query43.1%, MLP26.6%. Parameter 상대변화는 selector .759%/.915%, predictor3.11%/10.47%.
같은 final-selected 위치의 유효 patch-time 평균 MSE(before/after/current-copy):
ego2.3792/2.2716/2.3964, MLP4.3746/2.8763/2.1647. 두 모듈 업데이트·유한출력 확인이나
선택 효용/공식planning 개선은 별도 미확정이다. 전체dev192개·24recording/seed29만의 진단이다.
미래유효감독 patch-time수는 horizon1/2/3/4별600/468/344/300; 없는GT는회색/곡선공백.
Source source commit8182f6c + 실행 runner SHA, checkpoint/cache/image hash를 결과에 보존한다.

- Coverage27run/21,600update 완료: 추가dev192window/24recording의 원본ADE0.352210m.
  MLPlearned0.361504/fixed0.355226/random0.351212/ego-query0.371554m.
  Random-learned 차이−0.010292m, cluster95%CI[−0.018142,−0.003216]; 탐색적 다중비교다.
  Random-original 차이의 CI는0포함. 선택학습의 우월성/원본대비 개선을 확정하지 않는다.
  자세한 수치와 이전64window 결과와의 구분은 밤샘 보고서를 본다.

첫21run/8400update/1096.94초완료. 고정400step dev ADE MLP .232000, ego .239397,
aux-only .236099, gradientprojection .239264, halfbridge .233743, frozenS .234913, uniformADE .242933.
원본 .220644보다모두평균이높다. Train은 .1294~.1874로줄어일반화우려가있으나원인단정금지.
비교군의ego대비recordingCI는모두0을포함. 작은표본의부정적결과도공유JSON/CSV에그대로보존.

새진단: ego-query residual의200update dev ADE .216938 → half-gain개입 .209855m.
MLP .209975 → current-feature대체 .205721m. 개입/OOD결과이지 별도학습승자나 미래무용성증거아님.
P planning/aux gradient의음수cosine비율 MLP62.5%/ego54.2%(각24trainbatch), 인과원인미확정.
ForeDrive의outputdetach를그대로쓰면S학습이끊기므로parameter-onlyfreeze대조를 별도명세했다.

선택 비교 완료: 원본 dev ADE0.220644, fixed0.215851, random0.217723,
learned0.242582, learned-no-aux0.238094m(대응3seed 평균). 공식 planning 성능 검증이 아니다.
Shared `results/drive_jepa_selective_future/selection_comparison_v1_20261002/`.
추가 학습은 사전 커밋d3bbced에서15run/4500update를 완료했다.
MLP dev0.209975±0.008071m, 원본 대비 차이-0.010669m이나 recording CI[-0.025015,+0.000101]로
일반적 개선을 확정하지 않는다. Contextual/ego-query/LoRA는 같은 patch persistence MSE를 넘었지만
planning 개선은 일관적이지 않다. LoRA 추가 ADE 차이-0.000479m/CI에0포함.
실제planning·aux→LoRA 전달, aux→selector/bridge 차단 확인. GPU1/최대allocated2.674GiB/OOM0.
첫 postflight 초기 참조의9.5367e-7 FP32차이는별도보존; 동일동결·warmup검사에서bitwise통과.
보고서 `docs/drive_jepa_architecture_followup.md`, 원본/과거결과/학습조건과미확인해석을분리한다.

**최신WA중단결과**:9253/12146성공/실패·중복0, partialPDMS91.102506/같은scene Drive89.019762.
CSV/JSON `results/official_wa_jepa_reproduction/partial_navtest_at_drive_extension_20261002.*`.
Sparse/all-ID/모듈시간 결과는그대로보존. 전체WA점수/학습형선택성능/가설검증결과로해석금지.
**Drive extension**: 설계·모듈·CPU계약검사 구현. 실제모델검사결과는새report에서확인하고기존pilot과혼동하지않는다.
공식strict full loading/2project-train navtrain recording/11신규계약+2split검사·전체103tests통과.
off 전후/init-on bitwise원본동일, 원본309981955params hash불변; 신규1271489params.
zero-initialbridge1step후planning→selector9.39481e-6/predictor1.05006e-5/bridge.101464,
aux→selector0/predictor1.598178/bridge0, futuredetach→S/P0/bridge는양수. 원본planner gradients0/frozen.
Future teacher5/8tubelets valid/selected20target; +2/+3/+4s없는window도유지/현재선택에미래validity미사용.
2window batch timing 원본.126048/off.110916/on.120065s, sharedGPU차이를순수overhead/속도향상으로해석금지.
Peakallocated 약1.261GiB/최종진단45.176s/OOM없음/현재GPU작업없음. JSON `results/drive_jepa_selective_future/connection_v1_20261002.json`.
이는연결검사이며selector/predictor최적화·미래정확도·planning향상은미검증; optimizer1step은bridge출력projection만갱신했다.

**578be6e 이후 최신 조사**:

- DrivePB `Scorer.forward` 미래collision/areahead는train-only. pred_agents_states를trajectory에소비하는경로없음.
- WA jointscene/trajectory inference와trajectoryloss→scenehidden 확인. single-forward의finalscene_out은직접planninggradient없음.
- WA currentstoredcommand누락시futuretrajectory추론fallback발견;향후selectortrain입력에사용금지명세.
- WA native4view patch-tube선택추천, 객체와구분. Camera당128tube/총2048futuretoken vs8192dense를설계만작성.
  Shape/indexpacking과positionST/auxpolicydetach미구현,fullweight/config/VRAM/latency미검증.
- Saved scene분석12146/136recording/metadata누락0. Forward8070/left2501/right1575scene,
  command별PDMS90.153576/88.806043/85.127191. H1증거나정책tuning데이터아님.
- 새7context+기존5official집계CPUtests통과/Ruff통과. 기존reproduction디렉토리/config원본보존.

**최신 공식 Drive-JEPA 실측**: 전체12146scene 성공, NC99.082002/DAC96.558538/EP83.034487/
Comfort99.983534/TTC96.023382/DDC98.196937/PDMS89.224320. 논문PF89.0 대비+0.224320점의정확원인은미확정.
Fullplanning checkpoint·독립Conda·공식전처리/scorer이며pilotencoder-only/ADE가아니다.
실행약741초/총4worker/샘플VRAM GPU0·1 7,810/7,811MiB, 모든scene finite.
최종JSON/CSV/hash·설정대조/비용과5집계tests/Ruff를공유한다. 아래pilot결과는보존된이력이다.

**최신 실제 결과**: Dev scene-macro ADE 정지9.1239/CV1.0951/CA0.9260/train-fitted ridge0.7677m.
기존 seed29 A–F/1000은 1.55–1.75m. 대응 3-seed C absolute1.7066±0.0554 vs visual residual1.7545±0.1180m,
차이+0.0478m; seed별 −0.0297/+0.0978/+0.0753m. Recording cluster CI[+0.0137,+0.0864]는
4 dev group/고정된 3-seed에 조건부다. Residual visual MSE 약0.381은 persistence0.395보다 조금 낮지만
큰 0.79→0.38 변화는 구조적 skip 효과다. 공간 +4초 위치오차 CV2.0197m vs D/E/F6.3530/5.5511/5.6630m.
G1/G2 약세는 pilot 문제이지 연구 가설 기각이 아니다.
Trainval/navtrain1192segment→162recording, cap24: train1857/82group, dev865/40, held898/40.
Held에는 mini 없음/모델 평가 없음. 목표12000은 cap 이론상3888로 불가능하여 강제로 맞추지 않았다.
전체54tests/Ruff/새 final5 checkpoint strict·finite·sampler·RNG presence 확인.
JPEG export 원본 근거는 여전히 없다. 아래부터는 보존한 이전 결과/역사적 감사다.

아래는 **이전 서버에서 보고된 결과**다. 이번 세션에서 재계산하거나 학습 provenance를 검증하지 않았다.
이전 보고 조건: navtest 12,147, Phase 2 5 epoch, batch 24 × 2 GPU.

| 실험 | 보고 PDMS | 보고 Δ vs P2 | 해석 유의점 |
|---|---:|---:|---|
| phase3 논문 재현 | 90.96 | +1.96 | 독립적으로 재평가하지 않음 |
| P2 baseline | 89.00 | — | 기준 |
| α 보행자 추가 | 89.09 | +0.09 | 객체 클래스 변경 |
| F1 미래 BEV 감독 제거 | 89.25 | +0.25 | 보조 감독 변경; future latent 입력 제거와 구분 |
| F2 motion 감독 변경 | 88.67 | −0.33 | 당시 pair_Disp 경로/설정 감사 필요 |
| E3 월드 25→5 | 88.62 | −0.38 | 객체 수 변경 |
| E2 perception freeze | 87.98 | −1.02 | 학습 용량 변경 |

DAC 상승만으로 planning 과적합이 원인이라고 확정하지 않는다.
상이한 감독·용량·객체 수·클래스 변경을 동일한 미래 정보 선택 ablation으로 묶지 않는다.
기존 맥락별 부호 역전은 cross-fitting·log 상관·다중검정을 거쳐 재검증할 관찰이다.

**2026-10-01 코드에서 확인한 것**:

- SafeDrive `select_topk`는 후보 ego 경로와 현재 객체의 거리를 사용하고 정수 top-K index를 gather한다.
  선택 feature 내용의 gradient와 선택 정책의 gradient는 다르다. 현재 선택기는 학습된 selector가 아니다.
- Drive-JEPA perception-free downstream은 pretrained encoder→ego trajectory decoder다.
  감사한 분기에는 별도 entity 미래 latent predictor 호출이 없다.
- `context_axes.py`의 fwd/lat/dheading/bow는 미래 PDM reference 기반이다.
  “전부 현재 입력 기반”이라는 과거 설명은 잘못됐다. online selector 입력으로 재사용하지 않는다.
- target track 정렬과 미래 ego-frame 변환 참고 코드는 있지만 entity latent adapter는 아직 없다.

실제 파일: CSV 22개, O0/F3 last.ckpt 각각 1,382,312,902 / 1,382,302,278 bytes.
크기는 이전 전송 기록과 일치한다. checksum·내용 검증은 미수행이다.

**v1 CPU 결과**: 13/13 graph 검사 통과. Planning만 backward하면 S/P/D gradient가 있고,
auxiliary는 S/D를 직접 차단한다. Detach는 출력이 정확히 같고 해당 gradient만 차단한다.
3-seed 합성 선택 학습에서 exact-set 정확도 99.9756~100%; command 교란 시 성능 저하.
이는 고정 analytic predictor/planner를 쓴 쉬운 과제다. 미래 정보 필요성·NAVSIM 성능·H1/H2를
증명한 것이 아니다. 원본 수치·source hash는 `results/synthetic_diagnostics/future_prediction_graph_v1_before_readability_refactor_20261001.json`에 있다.

**이름 변경 후 재검증**: 13/13 검사, 같은 3개 seed의 7개 정책×3개 metric=63개 값,
초기 metric 9개 및 gradient norm 15개가 이전 실행과 정확히 같다.
새 source hash·경로·환경을 담은 report는
`results/synthetic_diagnostics/readability_refactor_validation_20261001.json`에 있다.

**후속 실제 데이터 진단**: 28/28 unittest. 같은 mini log/scene의 두 구간에서 GT-state H
`[1,32,10]`→K4 예측 `[1,4,8,6]`→ego `[1,8,3]`의 gradient 계약·공동 backward·1회 optimizer
update가 통과했다. Current annotations는 privileged input; visual JEPA/NAVSIM 성능 아님.
Log/image hash는 전후 동일. 기존 synthetic gradient norm15개도 변하지 않았다.
합성 relevance는 input key·intent로 계산 가능하여 `input_exact_match`와 `relevance_oracle`을
구분하고 이전 hindsight/배포 불가 해석을 정정했다. 추가 synthetic training은 하지 않았다.

**감사 결과**: Drive-JEPA v1 perception-based는 proposal query를 정제하고 score로 선택한다.
미래 collision-object state head는 train-only auxiliary이며 planner/score head의 입력이 아니다.
권고 scaffold는 공식 front encoder+단순 trajectory decoder+새 image ROI/미래 memory adapter다.
SafeDrive를 주 baseline으로 되돌리지 않는다. Weight/visual batch gate 후 최종 baseline을 결정한다.

**최신 영상 pilot 결과**: official weight5,127,748,765bytes의 pinned revision/SHA256 확인,
target_encoder292tensor strict load. Frozen encoder303,885,312params + 신규 scaffold2,487,988params.
실제 mini window0에서32현재 후보 중 front ROI13, K4, future8의 visual1024dim+spatial6dim 예측.
Planning S/P/D norm0.029646/0.466658/25.917812; visual aux0/3.903199/0; spatial aux0/6.385962/0.
Future detach는 forward 동일·S/P gradient0, no-future-branch는 S/P 호출 생략·gradient0.
공동 backward와1회 selector update,33/33tests 통과. 최종 영상 재검사20.24s, CUDA peak allocated
1,324,247,552bytes. Epoch 비용·학습 성능·전체 process VRAM 측정이 아니다.
GT geometry/association과 front-only 제한, future ROI의 visibility 편향 및 rectification 미확인이 남는다.
순수 visual JEPA/공식 Drive-JEPA 전체 모델 재현/H1·H2 성과로 부르지 않는다.
Shared log1/image10 hashes 전후 동일. 결과: `results/visual_diagnostics/visual_future_pilot_verified_20261001.json`.

**최신1231767 이후 조사/결정**:

- EgoFSD v6의 intention/attention 객체 선택→joint motion/planning, ForeDrive v2의 latent conditioning과
  직접 중복한다. EgoFSD official tree `23fec8aba3e828ef228939e30e3020240d8b0cae`는 README/assets only;
  selection autograd는 코드 미공개로 미확인. ForeDrive Eq.(7)은 planning↛predictor를 명시한다.
- 현재 pilot은 frozen visual+explicit spatial 감독이고 객체 대상 선택이지 정보 종류 선택이 아니다.
  공식 encoder 재사용·신규 작은 planner라는 사실을 유지한다. Final novelty/target은 미확정.
- Mini64segment=52recording group proxy. 고정 sampling16group, train12/dev4. 과거 smoke group은 dev only.
  비중첩 train277/dev96window를 고정하고 native log token overlap 없음을 확인했다.
- Survey128window: front-valid>K4 58/128=45.3%, front0은7/128. 현재 front621개 중+4s visual406=65.4%,
  spatial581=93.6%. Left future-heading proxy11window는 visual36.4%; right0, merge label 미확인.
- Current candidate 중 side-only projection26.9%. 4contact sheet를 직접 읽었고 occlusion/rectification은 미해결.
  미래 target 유효성으로 현재 candidate/window를 필터하지 않았다. CPU약8.08s, log16/image16 hash 전후 일치.
- 첫 target 비교는 fixed-nearest K4, A branch 없음/B planning만/C visual/D spatial/E mixed.
  B–E 같은2head/branch, C/D/E 공통mask와 train-only normalization, 200update/조건으로 계획했다.
  이는09913b5 당시계획이었으며 아래실행에서구현했다.

**최신 09913b5 이후 실행**:

- Camera K/D/1920×1080가 원본 nuPlan DB와 4표본 일치; in-memory stored-K/D rectification 후
  crop28/resize512×256, pinhole pixel-center ROI 규약. JPEG export 이력의 byte 검증은 미완료다.
- Manifest train277/dev96window만 616,266,713bytes cache, current input/future target 분리.
  같은 초기 state/batch 순서/common-supervision-count sequence로 A–E 각200update 완료.
- Dev scene-macro ADE(m): A5.921/B5.881/C5.921/D5.749/E5.872; 아직 곡선 하락 중, 단일 seed/4dev recording.
  D를 최종 target으로 선정하지 않았다. C visual 분산 비율0.012/swap ΔADE0.0006m로 평균 회귀·branch 무시 경고.
  E visual 분산 비율0.120, 모든 forecast는 현재 persistence보다 아직 MSE가 높다.
- Train-only common4785관측으로 normalization. Availability27659slot/time/zero-common133draw가 모든 조건 동일.
  A 활성762627/B–E2223283params; aux→planner gradient0, scorer 변화0, P/D update 확인.
- 38tests/Ruff 통과, shared log16/image3730 pre/post SHA 동일. GPU0 종료/환경 보존.
  JSON: `results/target_supervision_exploration/seed29_updates200_20261001.json`.
  Cache/last200checkpoint/curve: `outputs/feature_caches/target_supervision_rectified_v1b/`,
  `outputs/target_supervision_exploration/seed29_updates200_v1/`.

**9353acf 이후 최신 측정**: C200은train-mean(.950)수준, persistence .395.
373구간 track/time/spatial/gather 검사 일치, normalization/loss 분모/currentROI 경로 정상.
1000의dev ADE seed29 A/B/C/D/E/F=1.672/1.745/1.649/1.699/1.575/1.554m.
C visual MSE.794/분산비.214, E.834/.203으로 개선됐지만persistence를 못넘었다.
A200 3seed std.334m(A만); E/F 대응29·11의F ADE이득.021/.051m는 작은탐색결과다.
Swap donor120slot의availability confound 및 JPEG export미확인을 명시했다.
새3test+관련5test/Ruff/새final10checkpoint optimizer·sampler·CPU RNG검증 통과.
원본cache/구결과/기존전체recovery는반복하지않았다.

## 3. 마지막 커밋 이후 바뀐 것

- 사용자 선택에 따라 current geometry·appearance/interaction·context/dynamics/future 세 경로의 Linear/Conv LoRA 조건을 별도 구현했다.
- Conv-LoRA 커널 업데이트, FiLM hook 합성 위치, 공유 context 중복 방지, native digest를 검증했다. 273adapter planning gradient 모두 양수.
- 공개 LPWM/동일 seed2 planner 초기 출력 동일성·native parameter/buffer 불변·geometry adapter 직접효과·DDP batch16/유효64 실제검사 통과.
- 새 설정/source279개 hash등록, full103,288scene25epoch 본학습3186133와 최종평가queue3186134 및96장면monitor3186135 시작.
- 전후 location/size/presence 시각화, exact epoch checkpoint 우선진단, 이후 fullnavtest→별도v2→EPDMS 자동연결을 유지했다.
- 기존 attention-only1614/1615·geometry-only준비조건과hold는보존. LoRA범위확대와공정비교한계/성능미확정을 연구문서에기록.

## 4. 다음 단계 — 기반 추천 검토 후 (최신 사용자 지시가 아래 과거 계획에 우선)

최신: 현재세경로LoRA25epoch를지속하고read-only고정96panel진단을확인한다. 새DrivoR1epoch학습없음.
첫epoch정확한epoch_01.pt의geometry크기·presence시각화와readout/개입검사를보고한다.
최종v1→fullnavtest→별도공개initv2/navtrain10epoch→warmup/navhardEPDMS가queue에등록됐다.
설정/소스변경은279개seal을깨므로기동중수정금지. Pause/오류시다음종속작업중단.


**최우선: 사용자에게계층설명후적용대상선택을기다린다.** 아래자동재개안보다이명시적지시가우선한다.
선택이오면기존후보와차이를반영한새config/registration을만들고필요한gradient/메모리검사후본학습한다.
선택전geometry config/queue/monitor나old24epoch를실행하지않는다. 기존pause를제거하지않는다.
DrivoR비교는25epoch후진행한다. 첫epoch DrivoR추가학습선택질문은최신지시로대체됐다.

첫 epoch review에서는 학습 연결과 표현 사용 여부를 우선 판단하고, 낮은 점수만으로 연구 가설을 탈락시키지 않는다.
Warmup 이후 고정 checkpoint 비교는 권고이며 아직 새 실행 계획으로 등록하지 않았다.

가장먼저review root status확인. 첫epoch를기다리는중이면제어/학습중복기동금지.
DrivoR비교방식사용자응답을확인하고해당비교를준비한다. 동일1epoch이면25epoch scheduler/backend초기state/data/effectivebatch를맞춘다.
1614경계검토완료전원래25epochqueue를자동재개하지않는다. Epoch1체크포인트검토후재개시epoch_01_resume.pt를복원한다.
실중단시1615까지갔으면overshoot state를보존하고재개로그에서중복시도와정식학습경로를구분한다.

다음epoch1(1614update) 완료와자동표현진단을확인한다. ETA는최근window로재계산하고fullv1학습과전체queue를구분한다.
현재loss감소만으로유용한particle학습/PDMS개선을단정하지않는다. Geometry-LoRA제안은아직미적용이다.

현재조건은attention feature/future LoRA 조건으로해석한다. 직접공간재배치수정은 xy_head/scale_xy_head/obj_on_head Linear LoRA,
필요시attribute CNN ConvLoRA를포함하는별도설계가필요. 기존registered실행을중간변경하지않는다.
좌표head의작은출력차원에맞는rank와공통planner/학습량비교를먼저정하고,gradient/위치반응확인뒤효용검증한다.

시각화 해석은 위치·glimpse scale·presence와 planning 개입 결과를 구분하여 보고한다. 기존 자동 monitor 유지.

새monitor 중복기동금지. watch_status/monitor.log/각update complete+summary/readouts/intent/interventions를 확인한다.
0→100→500→1000→이후2000간격·epoch·final 비교. 현재v1만 감시하며 후속v2 감시는 별도등록 필요.
점의 이동뿐 아니라 readout·관련particle 대 matched-control 개입·고정후보 regret를 함께 판단한다.
모든 probe는 train-distribution 진단이며 독립 성능은 기존 navtest/EPDMS 큐에서 확인한다.
본학습 loss를 중간 변경하거나 결과를 보고 panel을 바꾸지 않는다. DrivoR 통제군은 문서 설계만 있고 미실행.


현재 LoRA parallel 실행/queue를중복기동하지않는다. 진행/VRAM/registered hash/완료coverage를점검한다.
현재기동은`train_lpwm_drivor_lora_parallel.py --config configs/lpwm_drivor_lora/navsim_v1.json --execution configs/lpwm_drivor_lora/execution_batch16_loader2_oracle4.json`.
V1최종25epoch40,350update→fullnavtest→별도public-initv2 10epoch→warmup/navhard EPDMS순서를새queue가자동수행한다.
재개시기본trainer로돌아가지않는다(기본batch8이적용됨). 이전native-fullpause는해제하지않는다.
입력navhard미완성시queue가평가전재준비한다. 이사실을평가완료로보고하지않는다.

실행중인등록joint학습을중복실행하지않는다. v1완료→fullnavtest검증→별도public-initv2학습→공식warmup/navhardEPDMS자동.
수치/coverage/정상checkpoint완료후다음stage로이행;실패시queue중단. test결과로epoch선택하지않는다.
초기업데이트는warmup중이며수렴/성능개선을단정하지않는다. 다음조회는progress/메모리/queue부터확인.
순수register-vs-particle주장을위한DINObaseline동일해상도/적응예산통제는별도실험이며현재미실행.

완료한Stage1유무결과를보고한다. 모든등록작업종료,추가학습/DrivoR/객체GT/navtest자동실행없음.
후속planning신호강화·공통DrivoR planner비교는별도설계로남는다.

공개고정4707업데이트완료후최종저장·검사→동일개발평가→적응고정82.5238과Stage1효과paired비교를자동이어간다.
현재예상결과완료23:05–23:15KST. 등록외실험을추가하지않는다.

DrivoR의joint driving학습/LoRA/score detach설명을사용자에게보고한다. 현재공개고정4707→검증→Stage1효과비교유지.
DrivoR학습이나동일planner대조는이번조사만으로자동등록하지않는다.

Planning신호강화는제안단계:같은checkpoint·trainingbatch의loss별gradient분리후partial구조고정으로SSL계수.02/.02÷3/.002비교.
후속학습시같은partial4707+AdamW에서분기하고동일추가1epoch대조군도실행해야한다. 원래완료partial점수를대신쓰지않는다.
현재학습은유지하며새학습/대기열은등록하지않았다. 실행시공통LRschedule·source/config/분기SHA를별도고정한다.

일부계층/전체particle비교이미지와전중후갤러리를사용자에게제공한다. 해당head가학습됐고실제위치가변했음을설명한다.
Planning단독원인·객체중요도개선은미분리로남기고현재공개고정학습→평가→Stage1비교를유지한다.

현재실험의추가이득미확인을보고한다. 위치만이아닌현재/미래feature변화와동일모델개입/gradient경쟁을먼저진단하는안을제시했다.
새진단/geometry학습해제/epoch증가는미실행이며현재공개고정학습→동일검증→Stage1효과집계를유지한다.

사용자에게 Adapter 대 고정군 실제 비교 이미지·갤러리를 제공한다. 현재 공개고정 학습→검증→Stage1효과 비교는 유지한다.
위치변화가 작다는 관측을 객체정보 손실/학습실패로 단정하지 않는다. 새 semantic probe·재학습을 자동 실행하지 않는다.

가속진단완료:큰배치cache/monitor생략을본학습에채택하지않았다. 기존동일설정으로4707→최종평가→비교유지.
현재ETA23:05–23:25KST는공유부하에따른추정이다. 추가속도실험/변형재학습을자동으로확대하지않는다.

현재public_control_training4707완료→동일1024planning/256world검증→적응고정군82.5238과paired Stage1기여집계.
공개조건하나만실행한다. 배치확대한두조건재학습안은사용자정정으로실행하지않았다.
큰배치표현계산·정기world monitor 등속도개선은gradient/출력/실측이득확인후판단하며,
현재등록source/config를실행중에수정하지않는다. DrivoR/추가epoch/GT/navtest자동실행없음.

현재 사용자 요청한 Frozen 대조군 검증과 네미세조정대비 결과 보고까지 완료했다.
표현미세조정추가이득미확인을 기준으로 후속 연구를 결정하되, 고정 대조군을 이후 비교의 기준으로 유지한다.
DrivoR 공통planner 비교는 기록된 후속과제이며 사용자 지시 없이 대기열 등록/자동실행하지 않는다.
추가seed·epoch·객체GT·navtest도 자동시작하지 않는다. 아래 '완료 대기'는 과거 단계 기록이다.

Frozen 최종1024planning/256world검증과paired비교완료를기다리고,초기frozenhash재검증·미세조정대비PDMS/경로오차/영상유지를보고한다. 학습종료되었으므로이번VRAM관찰로배치변경/재학습하지않는다. DrivoR추가기동없음.

Frozen 대조군의4707학습·최종검증·paired비교를마친뒤표현적응추가효과를판정한다. 현시점Adapter/LoRA가후속검토후보이나PDMS우열·수렴미확정이다. 추가seed/epoch/DrivoR을자동등록하지않는다.

Frozen 본학습4707까지 완료→동일1024planning/256world검증→기존네방법과paired비교를 자동 실행한다. 최근약1s/update로학습20:45–20:55,검증·비교21:00–21:20KST예상. 결과수치와freeze hash확인을 함께 보고한다. 이후DrivoR자동실행없음.

현재 학습과 검증/집계 완료를 기다리는 frozen CPU queue가 자동으로 후속 검사를 거쳐 planner 학습을 기동한다. Stage1 표현과 가중치/buffer 불변 검사를 유지하며 새 DrivoR 대기열은 추가하지 않는다.

우선 기존 full 학습→개발검증/집계→frozen LPWM·encoder FiLM 고정 대조군 학습→검증/미세조정 대비 결과 보고를 완료한다. 이후 DrivoR 비교는 해야 할 후속 과제로 기억한다. **현재 완료를 계기로 DrivoR 작업을 자동 실행하거나 대기열에 추가하지 않는다: 사용자의 명시적 지시다.**

기존 full→개발검증→frozen 대조군은 계속한다. 후속 표현 비교는 공통 DrivoR trajectory/scoring planner와 동일 학습 규칙을 기준으로 설계하고, 미래/encoder명령/속성구조의 효과를 분리한다. 현재 공통planner 구현·실행 설정은 미등록이다.

기존 full→검증→frozen 대조군 순서를 유지한다. Register/particle 공정 비교 및 미래/명령 ablation은 이번 설명의 연구 제안이며 새 실행으로 등록하지 않았다.

Full native LPWM+planner 학습과 검증을 유지한 뒤, 기존 등록 순서대로 LPWM 전체 및 encoder command FiLM을 고정한 동일 planner 대조군을 자동 실행한다. 이번 확인으로 학습 범위를 변경하지 않았다.

현재 full 학습 종료와 최종 개발 검증·네 방법 집계 후, 등록된 frozen 대조군 GPU 감사/profile/학습/검증을 자동 실행한다. 대조군 전체 종료 시각은 자체 속도 실측 뒤 갱신한다.

v4네방법학습·검증·four_method_summary완료marker→새queue GPU감사→8우선profile(메모리실패4/2)→engineering→고정LPWM+동일planner75297장면/1epoch4707→개발1024planning/256world→adaptation_vs_frozen_summary 자동실행. 초기Stage1+seed47planner사용,profile가중치폐기. 추가seed/epoch/GT/navtest자동실행없음.

기존 full low-LR4707학습→동일개발검증→네방법집계유지. Adapter는추가world0.705M로planning평균양호,LoRA는미래영상유지소폭우세. 최종선별은full결과함께검토하며추가seed/epoch/frozen대조는이번보고로등록하지않음.

현재 full2095→4707 본학습 완료 후 동일1024planning/256world개발검증과네방법집계자동실행. 새로운batch실험·학습연장·공식navtest추가없음.

2026-10-05 최신: v4 Adapter검증→기존full batch8우선profile·메모리초과시4/2fallback→2095checkpoint재개→검증유지. 준비한추가실측script는자동대기열에등록하지않음. 공유점유가다시사라질때는GPU상태와유효배치변경여부를재확인한다.

2026-10-05 pjh 확인 후: 기존 학습·검증 queue 유지. 정확한 타 사용자 종료 시각을 추정값으로 보고하지 않는다. 추가 GPU작업·설정변경 없음.

기존Adapter→검증→fullqueue유지. 점유감소질문으로배치/allocator/source/config변경이나새GPU작업없음.

기존queue유지. 고해상도확장은별도구현·호환·VRAM·적응검증을거치는후속방향이며이번질문으로자동시작하지않았다.

기존queue유지. 추가학습량비교와별도로해상도/종횡비/particle예산을통제하는비교를향후검토하되이번에는자동등록없음.

현재queue유지. 향후해상도확장은원본재처리·달라지는모듈가중치이식·particle/좌표/decoder호환검증후별도실험대상이다. 이번에는미실행.

현재해상도128과기존queue유지. 고해상도비교는별도원본재처리/모델호환·VRAM검증을필요로하며이번설명으로실행등록하지않았다.

학습연장권고만기록:Stage1 20epoch고정후유망Stage2+frozen대조를총3→5epoch,동일1024/256개발검증·LR스케줄명시. 이번질문으로자동등록하지않았고기존Adapter/fullqueue유지.

현재대기열유지. 이번질문은단계별학습량확인이며추가epoch·seed학습을등록하지않았다.

이번권고:현재방법선별후유망조건+frozen-LPWM학습planner대조를수렴까지비교하고핵심조건3seed평균·편차확인. 추가epoch/seed는이번개념질문으로자동등록하지않았고기존Adapter→검증→full재개queue유지.

현재Adapter→검증→full재개대기열유지. LoRA는world유지·궤적오차가양호하지만PDMS우월성미확정. 연구인과주장에는동일표현고정planner대조/seed반복/공식navtest가필요하며이번보고로자동추가하지않았다. 초기추론후보62개불일치원인은후속감사대상;현재queue의최종집계는초기동일성assert를요구하지않음을확인했다.

Adapter4707update/1epoch까지학습→같은개발1024planning/256world검증→보존full2095재개순서. 현재LoRA검증은완료됐고별도summary생성됨. 원래명령대로batch8우선/전체48GB감시유지. 추가학습·소스변경없음.

LoRA world 유지검증·최종집계 완료→기존queue의Adapter batch8우선메모리profile·학습·검증→full재개순서. 학습/queue/source/config를이번상태질문으로변경하지않았다.

현재 자체 개발분할과 공식 navval을 혼용하지 않는다. 단계별 장면/clip수와 실제 평가panel수를 구분해서 보고한다. 완료후 공식평가는 여전히별도이며이번설명으로데이터재분할/재학습을하지않았다.

LoRA 후속1024planning/256world 및 미래표현개입 검증 완료 후 같은장면 Partial과 비교. 기존queue는 검증 후 Adapter→full재개 순서다. 새로운학습/대기열/공식test를이번상태질문으로추가하지않았다.

권고: Stage1/2 공통 고정 개발 분할에서 설정 결정→동일 공식 navtrain 전체 풀로 최종 비교 모델 학습→고정 checkpoint로 v1 navtest 전체 평가. v2 navhard는 별도 확장 표. 외부 LPWM pretraining과 센서/감독/학습량 차이 명시. 아직 이 권고를 실행 queue에 등록하지 않았다.

권고안: 내부 개발에서 방법·학습량을 정하고 frozen-LPWM 대조군과 제안 방법을 동일 조건으로 준비한 후 고정 checkpoint로 전체 navtest를 평가한다. 추가 navhard는 v2 프로토콜로 별도 검증. 이번 설명만으로 신규 학습·벤치마크 queue를 등록하지 않았다.

공식성능주장에는고정checkpoint로navtest전체12146및공식v1 scorer평가가추가로필요하다. navhard는v2/EPDMS별도프로토콜. 현재queue가공식test까지자동실행한다고설명하지않는다. 이번명확화질문에서는활성학습·등록queue를변경하지않았다.

현재batch8 LoRA를끝낸뒤같은1024planning/256world최종검증으로partial과비교. 그후등록Adapter→full재개순서유지. 128monitor와1024최종개발점수를직접비교하지않는다. Frozen-LPWM+학습planner대조군은계속제안단계로현재queue자동추가없음.

LoRA8본학습4707→최종1024planning/256world검증→Adapter8우선profile/학습/검증→full8우선profile(2095optimizer계속)/학습/검증→4방법paired보고. 현재user전체48GB/batch8공통정책은v4queue에완료적용됐으므로다음턴다시승인을묻지않는다. 실행중새source/config수정·중복queue없음.

현재관찰대상은v4 queue421603이다. outputs/lpwm_card_budget_measured_v4/queue 의profile/selection/progress를본다. v2/v3루트pause는실행교체기록으로보존;원래사용자fullpause도보존.

최신: 새queue408762만관찰. 모든미완료방법8profile을우선하고48GB내성공이면8본학습. 메모리실패시4/2로유효batch16유지; 다른오류는중단. LoRA재개checkpoint보존4422,full보존2095사용. 이전대기열/원본full20epoch재기동금지. 종료후같은1024planning/256world개발검증과4방법paired보고계속.

최신: 현재 LoRA batch4 학습을 유지하고4707 완료 후1024planning/256world 검증→Adapter profile/학습/검증→원래full2095 재개/검증 순서를 관찰한다. LoRA8 확대는 기존 allocator 제한에 걸렸으므로 카드여유만으로 가능하다고 단정하지 않는다. 이번 배치확인 요청으로 메모리guard/source/config를 수정하거나 학습을 재시작하지 않았다.

SOTA비교시전체navtest/입력센서/학습데이터/SimScale·외부감독유무를맞춘다. 현재dev128 PDMS를최고보고95.1과직접순위비교하지않는다. 이번질문으로새논문방법이나추가학습을자동적용하지않았다.

최신ETA: 현재partial학습10/5 03:15–03:30KST예상(현재wall2.913s/update,공유부하변동·종료후검증별도). 다음3584monitor와종료후1024planning/256world평가확인. 기존LoRA→Adapter→full대기열유지.

효과분리대조는연구문서최신절에제안으로기록. 현재4방법queue에자동추가하지않았다. 추가시native고정+FiLM학습B를우선비교하고batch4→8일정/seed/SSL/학습량을맞춘다.

최신점검후계속실행: 다음2048monitor에서batch8전환후개발지표확인. 현재partial학습완료10/5 03:10–03:40KST예상(종료후1024planning/256world검증시간별도). 현재queue중복기동/추가설정변경없음.

최신: 새queue1902774만 관찰하며 partial→검증→LoRA→검증→Adapter→검증→full2095이어받기→검증 순서를 유지. 각방법실측으로안전·속도선택하고최종 `results/lpwm_48gb_planning_v1/queue/four_method_summary.json`확인. `configs/lpwm_planning/execution_48gb_v1/queue.json`이현재진입점. 새queue root의pause.requested로전체중단. 기존queue/source/config를수정·재기동하지않는다. 아래48GB조회만했다는기록은후속실행승인으로대체됐다.

48GB로상한만바꿔도현재free6GiB/allocated21.5GiB guard는통과하지못함. 가능성질문을실제guard변경·학습재시작승인으로해석하지않았다.

현재학습을계속하고 기존후속방법별batch profile 선택결과를 확인한다. 입력worker 증설/현재batch8 실행/activation checkpointing 해제를 자동적용하지 않는다. 추가GPU프로파일은현재partial와겹쳐기동하지않는다.

현재 두 queue 상태를 읽고 완료될 때마다 고정 평가 결과와 비용을 확인한다.
`outputs/lpwm_adaptation_method_comparison_v1/queue_state.json`과 `outputs/lpwm_four_method_queue_v1/queue_state.json`이 최신이다.
새 중복 queue/기존 GT ON queue/원본20epoch 전체 queue를 실행하지 않는다. Source/config 변경이 필요하면 별도 amendment를 등록한다.
전체 순서를 중단하려면 현재 활성 queue와 후속 queue에 pause marker를 기록한다. 과거 full root의 pause는 보존용이며 새 후속queue의 pause와 다르다.
현재 partial 학습은10/5 약04시 전후 완료 예상, 검증 시간·공유GPU 부하로 변동. 나머지 전체 ETA는 각 profile 실측 후 갱신한다.
최종 `results/lpwm_four_method_queue_v1/four_method_summary.json`을 확인한다. 현재 아직 없다.


- 현재부분GT두조건queue를유지한다. LoRA는후속설계후보이며이번설명만으로추가실행하지않는다. 비교시적용계층/intent위치/학습량/GT·SSL조건을맞추고실제step시간·VRAM·PDMS·world유지를함께측정한다.

- 최신ETA(23:07KST): 첫512update개발모니터10/4 23:30경, 첫조건학습10/5 04:00–04:30, 두조건학습+검증10/5 10:00–11:00. 최근3.843초/update 기준학습만의합산은09:00경이며검증등여유를포함했다. GT ON조건속도/전체panel검증시간은미실측이므로다음로그로갱신한다.

- **최우선은새부분학습queue1601034확인이다.** `outputs/lpwm_partial_planning_v1/queue_state.json`, 조건별progress/validation_log/visualization, `queue_failed.json`을읽는다. 실행중source/config20개hash를바꾸거나중복기동하지않는다. 아래기존fullrun재개절차는이번승인범위가아니다.
- 초기/512update마다128dev의실제후보PDMS·ADE/FDE, final4707update후고정1024planning/256world검증. 첫조건검증→GT보조ON감사/학습/검증→paired report자동연결. GT효용미확인이면선택사항유지. 런타임오류/누출/메모리위반은queue중단; 과학적개선미확인은기록하고대조조건진행.
- 새queuepause는 `outputs/lpwm_partial_planning_v1/pause.requested`로요청한다. 기존fullpause를건드리지않는다. 조건당약5–6시간은짧은profile추정이며본학습속도로갱신한다. 최종성능·독립표현효용은아직없다. 완료작업자동반복/추가epoch·navtest금지.

- **최우선: 사용자 재개 요청을 기다린다.** 아래 queue 유지/검증 예정은 중단 이전 계획이다. 자동 재개·pause marker 제거 금지. 현재 task는 일시중단 및 재개 가능성 확인까지 완료했다.
- **명시적 재개 요청을 받은 뒤의 절차:** (1) host 프로세스/queue lock/GPU0·1 점유 확인, (2) checkpoint SHA·update2,095·optimizer/config/source hash 확인, (3) `queue_failed.json`이 정확히 사용자 pause 예외인지 확인하고 `pause.requested`/`queue_failed.json`/`user_pause_status.json`을 날짜별 이력 디렉터리로 이동 보존, (4) 같은 config로 아래 queue를 `--detach` 실행, (5)2,096 이후 진행/메모리/다음작업차단해제 확인. 완전학습 sentinel은 없으므로 queue는 기존 준비·감사/profile 완료를 재사용하고 본학습에 `--resume`을 전달한다. 기존 failure를 삭제하거나 gate를 변경할 필요가 없다.

```bash
/rhome/junseong/PlanningAwareFuturePrediction/runtime/environments/kjs-lpwm-stage2/bin/python scripts/queue_lpwm_validated_training.py --config configs/lpwm_planning/object_future_joint_v3.json --detach
```

- 위 명령은 지금 실행하지 않는다. 현재 user pause 해제 승인 후 프로젝트 루트에서 사용한다. LR/epoch permutation/rank-update Torch seed/world NumPy sampling은 저장 update/config로 복원된다. 동일설정 resume와 LoRA/freeze로 바꾸는 새 실험은 구분하며, 구조/optimizer를 바꿀 경우 기존 full-finetune 결과를 이어진 동일실험으로 보고하지 않는다.

- 현재 구현 상세는 `docs/lpwm_planning_experiment.md`의 2026-10-04 실행 명세를 우선한다. 이번 요청에 따라 학습 전략을 바꾼 것은 아니다. 고정64particle/8future/한 장면 공통미래→후보 채점 구조이며 상황별 예측 예산 선택, 후보 action별 세계 rollout은 미구현이다. 독립 표현 효용 검증과 GT 채택을 이미 완료했다고 하지 않는다.

- 완료중간진단을반복하지말고health_check_20261004_2132/summary.json을읽는다. 첫epoch4707update의512dev ADE/FDE정기평가가다음확인시점(초기예상10/5 04시). 최종전체dev/PDM/world유지및GTaux on/off는등록queue에서계속. Update1,536의78.51PDMS를전체개발·최종·독립test성능으로인용하지않는다.

- **현재queue유지:** `outputs/lpwm_object_future_planning_v3/queue_state.json`, `metric_plus_world/progress.json`, `queue_failed.json`유무를확인한다. 시작후source/config변경금지. Source14개hash는queue_registration에고정,GT추가조건검증까지자동연결. 진행중작업을과거gate차단상태로오해해중복재실행하지않는다.
- 학습후각27,076dev의초기/학습후/미래persistence비교·공식후보PDM/7,745world유지검증. 두조건후 `ablation_summary.json`에paired PDMS와world비교. GT채택은개발증거에근거한잠정판정이며독립futureprobe/native mask/planner객체개입은아직후속. 실제PDMS결과를이전prototype와혼동하지않는다.
- VRAM상한46GB/GPU는다른사용자포함. 현재우리할당11.9GiB이지만외부점유변동감시;초과시우리child만signal checkpoint정지. CPU RAM은관측만. Checkpoint복구는등록config/source그대로하며실패원인확인전자동반복금지.

- 완료 결과 상세는 연구 문서 첫 절. 현재 정보의 일부 판독 개선을 보존하되 과거만의 미래 상태 판독을 persistence/ego-motion 대조와 비교하는 후속 검증이 필요하다. 자전거 점수의 낮음은 encoder뿐 아니라 balanced ridge/불균형을 분리해야 한다. Stage2 GT auxiliary 구현·진행 기준 amendment는 아직 미실행이며 기존 queue를 임의 재개하지 않는다.

- Frozen current-state readout은완료됐다. Sharedsummary/completion_review를읽고완료추출·판독을반복하지않는다. 미래상태판독/native mask/planner개입은후속이며GT-ROI판독성공과자율객체발견을구분한다. Stage1SSL유지/GT보조감독Stage2원칙유지.

- 현재3종의조건부판독과원본주석7종확인,2D투영ROI와native3Dbox를구분해설명한다. 차량/보행자만주석된데이터라고하지않는다. 미래판독·instance mask·PDMS개입은현재queue의후속이다.

- 최우선 학습 원칙: Stage1 SSL 유지. Frozen probe가 약하다고 Stage1에 GT loss를 자동 추가하지 않는다. 해상도/가림/판독기/SSL 목적을 진단한다. GT state/future 보조감독은 Stage2에만 구현하고 planning+world 대조와 비교한다. 아래 Stage1 GT 보정을 제안한 문장은 철회된 이력이다.

- 최우선: `outputs/lpwm_object_readout_validation_v1/queue_state.json`, 각모델progress, 완료 후 `results/lpwm_object_readout_validation_v1/summary.json` 확인. 신규queue를 중복실행하지 않는다. Source 변경/worker 실패 시 queue_failed 기록 후 의존 작업 차단. 원본 Stage1 실패queue 자동재개 없음.
- Readout 결과는 GT observed ROI/track으로 위치를 알려준 조건부 정보검사다. GT-ROI/geometry/background 대조와 support/종류별 표본을 함께 보고 native detection/tracking으로 확대해석하지 않는다. Native30 alpha 검사·검수mask·causal future readout·학습된 planner 개입은 여전히 후속 작업이다.
- 검증 결과를 확인하고 Stage2 GT object/future 보조감독의 새 학습config/source/gate amendment를 명시한다. 현재정보/미래정보의 병목은 분리하되 Stage1 추가적응은 SSL 범위로 유지한다. 객체loss훈련head를 그대로 독립probe로 평가하지 않는다.

- 최신 제안은 `docs/lpwm_planning_experiment.md`의 ‘객체 구분·정보 보존의 검증 설계’ 절. 공개/적응 checkpoint frozen probe + full64 alpha 진단 우선, mask/temporal/future 진단 후 학습된 planner 개입 순서다. 판독기는 각 표현에 동일 예산으로 따로 학습하고 geometry/background/GT-ROI 대조를 둔다. 새 평가 코드·cache는 아직 없음.
- 기존 box proxy는 보조 지표로 해석한다. 새로운 필수 gate를 만든다면 결과를 보기 전에 목적·비교군·허용 열화량을 등록하고 기준 수정 근거와 기존 실패를 보존한다. Stage2 유용성을 Stage1 진입 조건으로 요구하는 순환 검증은 피한다.

- 최우선해석정정: box proxy의하락을객체이해실패로간주하지않음. 후속분해는full64/top16·presence순위·scale/center·alpha support·GT객체종류와표현정보보존을구별해metric적합성부터검토. 기존gate결과는보존하고후속판정기준은명시적amendment로구분해야함. 현재진행된원인분해결과는아직없음.

- Stage1 보고는 `docs/lpwm_planning_experiment.md` 첫 절 및 validation_review JSON 기준. Scale/presence/top16선택/객체종류로 box proxy 실패원인을 분해하고, 직진 및0.5초LPIPS 악화를 별도로 진단할 필요가 있다. 이번에 실행한 것은 저장결과 검토이며 원인 진단·추가학습 완료로 보고하지 않음.

- 최신 연구 제안은 `docs/lpwm_planning_experiment.md`의 2026-10-04 첫 절을 우선 읽는다. Stage1 실패원인 진단 → 기존 candidate teacher로 oracle coverage/위험 분포 확인 → 동일후보 current/persistence/future·frozen/joint 비교 → 객체×시간 utility 감독 → 근거가 있을 때 실제 selective rollout 예산 순서. 이 설계는 기존 queue의 자동 실행 변경을 의미하지 않는다.
- 기존 3조건 설계의 refiner 확대보다 원래 표현 가설의 식별이 우선이라는 권고다. 후속 실행은 gate 해결 및 명시적 config/source 재등록을 거쳐야 하며, 실패한 queue를 자동 재기동하지 않는다.

- 선행연구 감사 후 후속비교안은 shared refined-feature safety 감독, 안전 다중pseudo-target 회귀, footprint/road 직접비용을 분리하는 것. 현재는 설계 제안이며 등록runtime에 적용되지 않았음. Stage1 실패원인 진단/gate가 우선한다.

- 최우선 상태 갱신: Stage1 gate 실패 원인인 top16 box recall 감소를 원시표본/scale/presence/객체종류로 진단한 뒤 다음 학습을 정한다. Queue 실패를 무시하거나 임계값을 완화하지 않는다. 아래 Stage1 평가진행/gate후 자동실행은 실패 전 계획. 충돌좌표loss는 제안이며 구현됐다고 보고하지 않는다.

- Planner 설명은 `docs/lpwm_planning_experiment.md`의 Loss와 gradient 상세절 기준. 기존 합산 loss 연결검사를 loss별 기여나 안전 좌표 최적화의 실증으로 표현하지 않는다. 등록 queue를 유지하며 Stage1 gate 후 Stage2 실제 GPU audit/profile/본학습으로 진행한다.

- Stage2재배치가설확인은같은scene/command의geometry·presence·feature/future·planning개입을분리한다. 객체/지도coverage의면적·visibility보정과frozenLPWM대조는추가진단설계이며현재queue에등록됐다고보고하지않는다.

- 시각화해석시점/박스/색/presence설명은 `docs/lpwm_planning_experiment.md`의시각화범례절참조. 기존학습·평가queue를이설명작업때문에수정/재시작하지않는다.

- 최신ETA는전체개발평가추론에한정한다. 이후통계집계/Stage1추가검사/teacher gate/GPUprofile가이어진다. Stage2조건별94,140update×3의전체완료시각은실제GPUthroughput측정전확정하지않는다.

- 최우선: 이미완료한Stage1을재학습하지않고 공개/적응전체개발평가와gate를확인한다. 기존supervisor/새queue유지. 과거pipeline_stopped(10/03 21:34)는현재실패로오해하지않는다.

- 2026-10-04 11:17 KST: 등록20epoch 최종checkpoint를유지하고 완료후공개/적응전체7,745clip 평가→기존gate판정. 개발dynamicsKL반등과미래MSE정체는진단신호이며threshold완화/학습중단/epoch추가/최종checkpoint교체근거로즉시사용하지않는다.

- 2026-10-04 01:05 KST 점검에서 재시작·설정변경 필요 없음. 기존 full Stage1→개발검증→gate→Stage2 queue 유지. 학습만 ETA10/04 10:40~11:30KST 수준이며 최종전체개발평가는 별도.

- Stage1 복원/ELBO 개선을 곧바로 미래/객체/planning 성공으로 보고하지 않는다. 전체7,745개 개발 평가와 supplement gate를 따른다. 고정 공개/적응LPWM 대조, 고정/공동학습 대조, oracle future 및 loss별 gradient 분해는 필요시 추가할 진단이며 현재 queue에 실행 등록됐다고 하지 않는다.

**최우선: Stage1과 검증 queue를 유지한다.**
1. Stage1 progress/PID와 새queue state, CPUteacher worker_progress/segments를 확인한다. 추가GPU profile로Stage1중단금지.
2. 활성 Stage2 설정은 `metric_distillation_v2.json`이고, source hash는 `outputs/lpwm_metric_planning_v2/queue_registration.json`이다.
3. Gate통과전 Stage2를직접실행하지않는다. 실제적응checkpoint의GPUgradient/DDP누적profile이queue에등록됐다.
4. 실패시 `queue_failed.json`과해당validation_gate.json의원인을진단한다. 문턱을낮추거나학습완료를가정하지않는다.
5. 새queue가종료돼도현재Stage1은별도기존supervisor가계속관리한다. 실패분석없이 old 단일경로planner를켜지않는다.
6. Source버그수정이필요하면기존queue등록/완료node를보존하고명시적amendment/재등록한다. 고정source검사를우회하지않는다.
7. 모든조건검증후fullnavtest12,146과pairedrecording CI. 이전Drive-JEPAfullbaseline을재추론하지않고보존결과참조.
8. 원격push는과거 VSCode Gitcredential socket 문제로실패했다. 이번push상태는로그/마지막note확인.

### 과거 다음단계 (자동 실행 금지)


**최신 방향 수정: LPWM encoder·context·dynamics·planner 공동 학습.** 사용자가 원본 context/dynamics를 제외한 선택을 지적하고 공동 학습을 요청했다.
기존 encoder-only는 완료7run(frozen3/planning3/uniform seed29)만 보존하고 우리 worker3425264/3425265/scorer3425266을 SIGINT 종료했다.
`outputs/lpwm_planning_v1/superseded_by_joint_world_model.json`을 따른다. 이전18run queue/finalizer를 자동 재개하지 않는다.
새 코드 `src/planning_aware_future_prediction/object_centric/lpwm_joint_world_planner.py`: 공식 encoder6.035M/context39.389M/dynamics59.869M+planner0.821M, RGB decoder만 제외.
과거2영상→관측transition posterior→미래8step은policy prior만으로 autoregressive rollout, activation checkpointing으로 gradient 보존.
GPU0 batch1 3update에서 planning/future loss 각각 세모듈gradient>0, 미래label변경시예측동일을 통과했다. bf16 batch4 profile 진행/새 데이터·규모 준비 중이다.
기존512train 중13개/192dev 중2개가 공식navtrain token필터밖임을 발견했다(로그는전부navtrain). 새 학습은 공식token까지엄격필터한다.

**LPWM 파일럿 완료 후 제안:** 결과·실제 particle 시각화를 검토한다.
다음 후보는 RGB 복원 추가 확대보다 의미 feature/depth 감독, particle 상태의 ego motion 분리,
가림 belief, encoder 내부 ego intent와 planning loss의 통제 비교다. 후속 구조는 아직 실행하지 않았다.
완료6run/전체navtest/WA를 자동재개하지 않는다. 최신결론은 LPWM 결과보고서를 우선한다.
아래는 이전 단계의 이력이다.

**최신 사용자 요청 처리:** 생성한 세 이미지를 사용자에게 전달한다. 상세 설명은 결과 보고서의
`실험 결과 이미지` 절이다. 이전에 질문한 다음 방향은 시각화 요청으로 전환됐으며 새 학습을 시작하지 않았다.

**이번 등록 실험은 모두 완료했다.** 결과 보고서를 기준으로 다음 encoder 학습 목적을 검토한다.
Planning-only 기준선을 유지하고 미래 변화 정보를 같은 용량의 readout으로 더 잘 꺼낼 수 있는지를 우선 검증할 것을 제안한다.
이는 후속 제안이며 새 학습 구성은 실행하지 않았다. 단순 block 수·loss weight 추가 sweep은 이번에 더 하지 않는다.
원격 인증이 복구되면 `git push mine junseong/main`으로 로컬 결과를 공유한다.

**아래는 이전 단계의 다음 작업 이력이며 현재 자동 실행 지시가 아니다.**

**현재 다음 판단:** 기존global비교군유지. 현재특징전달과미래변화정보의planning기여를분리하는후속설계가우선이며구현/학습미실행. 완료27run/PDM을반복하지않고새sweep는자동시작하지않는다.

현재다음: K/크기고정, global future fusion 대신선택위치에대응하는spatial-memory연결을검토.
아직구현/추가학습하지않았다. 개수·크기/proxyweight추가sweep금지. 현재15run을반복하지않는다.

최신: 사용자에게 CPU 시각화 갤러리와 ZIP을 전달해 실제선택위치/미래latent를 검토한다.
가중치·선택변화와 예측오차 감소는 좋은 선택정책/안전성향상의 증거와 구분한다.
추가 학습을 자동 재개하지 않는다. 아래는 과거 다음단계 이력이다.

최신: 등록된 matched-conservative-selection18run을 완료하고 별도CPU감사를 수행한다.
기존69run감사와32train recording patch교체진단은 이미완료했으므로반복하지않는다.
이번 고정 마지막 checkpoint 결과와 원본/고정/무작위 대조를 구분해 보고하며,
더 좋은 중간 step을 골라 최종 성능으로 바꾸지 않는다. Held-out/navtest tuning은 하지 않는다.
일상적선택은묻지않고기록. 마감/메모리/안전조건을넘으면중단·보존, 실패무한재시작금지.

추가 학습은 완료됐다. 더 복잡한 구조를 기본 모델로 승격하지 않고, MLP+새절차를 저비용 개발 참조로 보존한다.
다음은 새 절차의 fixed/random/learned 선택 비교 여부를 검토한다. 이번에는 추가 sweep을 실행하지 않았다.
공식 baseline/pilot/WA/held-out는 보존하며 새학습/전체benchmark/동적K를 자동 시작하지 않는다.
Git 인증이 복구되면 `git push mine junseong/main`으로 로컬 커밋을 공유한다.

**최신**: 승인된4조건×3seed×200update를등록상한에서종료하고, fixed/random/learned와aux없는
대조의대응seed·recording별개발경향을공유한다. Best-devcheckpoint선택이나실패후튜닝은하지않는다.
현재-feature직접전달 대조/공식planning평가/더큰학습은이번소규모결과검토후별도결정한다.
WA전체/기존pilot/residual/확대학습을자동재개하지않는다.

**아래는 WA 중단 전 계획이며 현재 실행 금지**: 공유GPU0:5/GPU1:2. 기존7worker/GPU launcher명령을추가실행하지않는다.
Queue/status/guard log→남은scene증가/메모리/OOM확인→14shard완료시자동aggregate를확인한다.
새pause/메모리pressure가발생하면추가기동없이보존하고보고. 상세명령은WA보고서/새safety profile.

지금은densefull완료를확인한다. 이미끝난smoke/all-ID/72trial를반복하지않는다.
전체완료후 `scripts/evaluate_official_wa_jepa.py aggregate`(CPU)→Table3하위지표/complete대조→commit/push.
Currentno-grad진단은학습형selectorutility의증거가아니다. 새학습은별도승인전금지.
최신 승인 순서: preflight/strictsmoke→공식dense전체→별도sparse all-ID동등성→fixed/random소수scene.
학습형selector·동적K/horizon·pilot·WA학습 금지. Full은smoke/cost/호환gate통과조건부.

1. 최신 `docs/future_prediction_foundation_decision.md`를commit기준으로ChatGPT·Claude에검수공유한다.
2. 준성이WA native spatial-tube라는범위축소를승인할지객체instance유지안을선택할지결정한다.
3. 승인후전용Conda/worktree/officialweight+Metaencoder/config strict loading/소수train-dev full동작·비용gate부터.
   현재미실행이며기존Drive환경upgrade/전처리섞기/전체모델재현성공주장금지.
4. all-ID 원본동등성→prediction전packing/selectorgradient/누출/비용→실행상한등록 순서.
5. fivecondition 원본/random/fixed-rule/planning-conditioned/all-future 참조를고정K/horizon으로통제.
   native recording train/dev/독립holdout,paired≥3seed. Dynamicbudget는후속.
6. 공식DrivePF전체평가를반복하지않고+0.224320원인을미확정으로유지한다. Pilot자동재개/추가학습없음.

아래는 `5c6e6d5`까지의 보존된 후속 연구 계획이며 지금 실행하지 않는다.

1. 최신 두 report/JSON을 commit 기준으로 ChatGPT·Claude에 검수 공유한다. Pilot 추가 튜닝/확대는 하지 않는다.
2. WA-JEPA 공식 source/weights/config/environment를 pin하고 strict loading→실제 현재/과거 NAVSIM batch의
   추론·gradient·VRAM/latency를 확인한다. 아직 full 모델 실행 없음. 공식 평가 설정을 먼저 고정한다.
3. 동일 예측 예산에서 future 대상·범위 구성이 맥락별 planning 결과를 바꾸는지 비교한다.
4. 현재 관측·ego 의도의 선택 학습, 동일 평균 예산의 상황별 배분을 순서대로 검증한다.
5. WA-JEPA native 공간/시간 token과 객체를 구분하고 최소 adapter/packing을 명세한다.
   Loss mask만 줄여 predictor 계산 절감을 주장하지 않는다. 동적 K/horizon은 현재 미구현.

navtest는 개발·진단용이며 최종 독립 평가가 아니다. navhard 접근·사용 이력·공식 프로토콜을
확인한 뒤 최종 평가 경로를 정한다. 이전 서버의 데이터 크기·GPU-hour를 현재 실측치로 취급하지 않는다.
**현재 대규모 학습·SafeDrive 재학습·전체 cache 생성은 시작하지 않는다.**
등록된 `pilot_foundation_decision_v1.json` 확대 계획은 후속 사용자 지시로 보류됐다. GPU가 비어도 자동 재개 금지.

## 5. 확정 범위 / 미결

최신: 사용자가세경로LoRA를명시선택하여새본학습승인. LPWM native및buffer고정/LoRA만적응,
공식DrivoRplanner전체학습. 직접objectGTaux/RGBreconstruction0,onlineoracle는지도·객체GT활용.
Particle재배치및feature/future효용은검증할가설이다. Registered96panel은trainval분포진단.
해상도128²/사전학습/미래rollout/FiLM/LoRA범위등DrivoR와차이가있어최종system비교만으로
register–particle구조의단독인과효과를주장하지않는다. 아래계층선택보류는이전이력이다.


최신명시적hold:사용자가LoRA대상을정하기전새학습금지. Prepared geometry head조건은최종선택이아니다.
기존첫epoch결과에는작은위치/크기변화가있어'완전동일'이라고부르지않는다. 추가geometry의주행효과는미검증.
Conv-LoRA는설명대상후보이며미구현/미학습. 기존Q/V와geometryhead후보를혼동하지않는다.
Native context는encoder.ctx_enc/ctx_module/dyn.context_decoder 공유;중복삽입주의. RGBdecoder와posteriorhead는현재planning경로미사용.

최신 질문은1epoch검증의방법론에관한설명요청이며, DrivoR비교방식선택이나24epoch자동재개승인으로해석하지않았다.
공식코드의중간validation지원과저자들의실제1epoch후진행여부결정관행은구분한다. 후자의빈도는확인되지않음.

현재사용자요청은1epoch검토후나머지24epoch진행제안이다. 이를위한reversible hold를등록했고비교방식응답은대기중.
비교결과없이무조건24epoch이어가기금지. 모델구조변경도아직없다. 독립검증셋부재와공정비교confound는해결되지않음.
Control이의도적으로pause하면oldqueue의stopped표시는학습실패와구분한다. Resume모델뿐아닌optimizer/scheduler/RNG보존필수.

08:55조회ETA10/15 17시KST는현재v1 25epoch만의추정이다. 장기간공유부하·epoch저장/진단비용으로변동가능.
후속v2는독립public-init10epoch이며v1 fullnavtest후실행된다. 전체queue의확정완료시간은아직없음.

‘LoRA가학습됨’과‘현재particle재배치를직접학습함’은다르다. 이번Q/V적용범위는후자를충족하지않으며FiLM경로만있다.
관측좌표에대한Q/VLoRA미분0은구조적문제여서attention rank나epoch증가만으로경로가생기지않는다.
Geometry LoRA는미구현/미학습. 재배치자체가유용성보장은아니며readout·개입·동일조건benchmark검증이필요.

Particle 위치만으로 planner가 어디를 중요하게 보는지 확정할 수 없다. 동일좌표에서도 feature/미래/사용도가 달라질 수 있다.

사용자 요청 두 검토는 진행 중 학습의 진단/비교감사다. 학습 및 평가 기존 queue는 유지했다.
분포는 차량/보행자/자전거 GT 투영 및 평면 지도 도로 proxy이며 semantic mask/가림 정답이 아니다.
교차명령에는 대안GT가 없으므로 encoder-only 민감도로 의도 반응을 측정하지만 적합성을 단정하지 않는다.
의미적 미래예측 loss 없이 latent8step을 쓰므로 실제0.5초 미래 상태 의미가 유지되는지 readout으로 확인해야 한다.
Baseline register도 planning loss로 학습된다. 일반 feature=planning과 무관한 feature라고 주장하지 않는다.
전체 시스템 비교는 가능하나 순수 particle 구조/미래 보존 우월성은 common-backend/정보·예산/intent/future 통제 재학습이 필요하다.


최신: perception은native FT에서Q/VLoRA32로바뀌었다. CNN/xy/scale/presencehead고정,현재geometry는추가FiLM입력변화로달라질수있다.
공식planner ego11D경로만동일하며encoder FiLM은추가. 해상도·backbone·pretraining·명령조건화차이로순수RegistervsParticle효과미확립.
배치16증설에서도nominal effective64/학습량/모델/loss/LR는같지만dropout난수분할때문에batch8과bitwise동일학습은아니다.
짧은순차profile의속도는공유자원·cache영향을포함한다. 큰배치/워커가더빠르다는가정은실측에서지지되지않음.
공식PDMS/EPDMS·수렴·의미있는미래정보보존은아직미검증. 아래native-FTvsLoRA차이는이전조건의이력이다.

확정: 공식DrivoRbackend/loss/연속64후보oracle,공개LPWM에서joint학습,동일공식split/epoch/effective64/seed2.
차이: LPWM128×128vsDINO672×1148,활성원래weightsFTvsLoRA,perception증강/사전학습/연산량/LR다름.
단순해상도·adaptationconfound없는representation우월성/의미적미래보존/particle이동효용은아직미확인.
기존Stage1효과+3.6077은이전내부개발결과이며새공식benchmark와직접차이를내지않는다.

Stage1이득은동일seed47/planner1epoch/내부개발1024(1021PDM)·40recording범위다. CI는recording bootstrap이며seed변동미포함.
공개Sketchy사전학습대NAVSIM적응비교이지random초기화대SSL비교가아니다. 미래LPIPS개선과semantic객체보존을동일시하지않는다.

4707optimizer update완료와training process정상종료/최종검증완료는구분한다. 22:53의마지막128scene monitor는
최종1024scene평가결과가아니다. ETA는직전동일조건평가시간을활용한추정이며확정마감이아니다.

DrivoR는논문과정적source를확인한것이며새학습재현결과가아니다. Register는LPWM particle과다르고명시적미래world model도없다.
DrivoR의SSL없는task적응결과만으로LPWM Stage1/SSL이불필요하다고판단하지않는다. 직접객체loss없어도oracle은장면GT/지도를사용한다.

배경particle이복원때문에불필요하게유지된다는것은가설이다. 기존loss는미래예측/KL포함SSL이고나무·건물무용도미검증.
Globalclip작동은확인했지만planning대SSLgradient기여/충돌은아직분리실측하지않았다. 계수비율3/10배는update배율이아니다.
후속3조건은설계제안이며실행승인요청/새queue없이문서화만했다. 현재공개고정Stage1효과실험은별도로유지한다.

Partial/full head의실제갱신과입력상particle이동을확인했으나loss별기여율은미분리다. SSL/weightdecay도동시적용됐다.
Full update0은v3원본이고2353/4707은2095재개실행이다. 같은초기geometry가확인됐지만LR/schedule/FiLM위치/배치이력차이는남는다.
8장면전체64particle통계이지전체NAVSIM통계나semantic객체대응성능이아니다. 새학습/추가GPU진단은없었다.

Planninggradient양수와PDMS추가이득은별개다. Adapter의encoder그룹gradient양수를native위치head갱신으로해석하지않는다.
SSL0.02의실제gradient우세/충돌은같은checkpoint·batch에서미검증;기존별도감사norm을우세비율로직접비교하지않는다.
Particlegeometry만으로feature변화/미래정보개선/명령별선택성을판정하지않는다. 1seed1epoch는수렴·일반적무효검증이아니다.

이번 시각화는 완료된 NAVSIM 적응 고정군과 Adapter군 비교이며 현재 공개고정 조건이 아니다.
8장면 FP32 관측4프레임 snapshot은 BF16 배치평가/전체분포/미래particle진단과 구분한다.
공통particle인덱스는 객체track이 아니며 presence는 planning중요도가 아니다. Adapter+명령FiLM+SSL 묶음 조건이다.
공간배치 변화는 작지만 latent/context/dynamics 변화나 planning 개입효과를 여기서 검증하지 않았다.

Batch확대진단에서표현동일성이성립하지않아현재물리batch8을유지한다. 이전다섯실험의유효16보존은확인됐지만,
물리배치이력의정확planning영향은미분리다. 본학습두조건은동일8×1×2이므로이요인을맞춘다.
진단timing은동시GPU부하하의국소측정이며전체학습속도보장아님. ETA는최종평가10–15분을포함한다.

최신사용자범위는공개checkpoint LPWM고정+freshplanner하나이며Stage1고정82.5238을재사용한다.
배치증설로유효배치가커지면재사용비교가교란되므로현재8×1×2유지. 기존5조건은유효배치16유지였고완전히무효가아니다.
단물리배치이력에따른dropout/SSL샘플링과full LR/명령위치차이로방법단독우월성은입증안됨.
Public상태의legacy stage1_checkpoint_sha256 필드는공개초기checkpoint SHA alias; navsim_stage1_performed=false와
initial_lpwm_checkpoint_sha256를함께읽는다. Cache용stage1_config경로를모델이적응가중치를읽었다고오해하지않는다.
가속진단은실제결과/gradient보존확인전채택금지,profile가중치는본학습에사용하지않음.

Frozen 최종 성능/고정검증/paired비교는 완료됐으며, 아래 '아직 미확정/진행중'은 과거 이력이다.
네방법모두PDMS추가이득미확인이나1seed1epoch/내부개발panel로미세조정의일반적무효를단정하지않는다.
PDMS는변형없는512고정후보의cache된공식PDM점수,전체navtest/continuousrefiner평가아님.
Frozen eval/FiLM고정 대 adaptation train/FiLM학습, full명령위치/20epoch LR스케줄 차이가 있다.
Adapter작은ADE/FDE감소도CI0포함,Partial/Full은경로오차증가구간양수. LPIPS는객체상태보존검사아님.
40recording bootstrap은훈련seed분산/다중비교보정을포함하지않는다. 추후공정비교/수렴/객체별인과검증은미결.

3.906GiB는학습로그PyTorchpeakallocated이며nvidia-smi전체점유와다르다. 조회GPU전체연산사용률은타사용자도포함하므로우리작업단독활용도로해석하지않는다. Frozen본학습완료/최종검증진행을구분하며128장면마지막monitor PDMS83.389는최종1024결과가아니다.

Four-method결과는1seed/1epoch/내부개발1024장면이다. Full종료planner LR2.989e-4 대나머지3e-5,명령위치/LR스케줄다름. 따라서full방식열등성이나Adapter확정우승주장불가. 직진world9/회전36/투영겹침197은영상오차진단이지객체보존·상황별PDMS가아니다.

네미세조정완료는등록1epoch/1seed학습과내부개발검증완료이며수렴/독립navtest완료가아니다. Frozen 최종성능과미세조정추가효과는아직미확정이다. Frozen queue의predecessor_status/stage는대기때값이남아있을수있으므로main실제complete와현재frozen_control_training을우선한다. 메모리감소는고정모델no_grad경로에부합한다.

Frozen 대조군의 초기 LPWM은 Stage1 완료 가중치다. 현재 Stage2 미세조정 완료 가중치를 가져와 고정하는 실험으로 바꾸지 않는다. Planner는 같은 초기값에서 새로 학습한다. 후속 실행 전 검증과 freeze/memory 검사가 있으므로 학습 마지막 update와 동시에 본학습이 시작되는 것은 아니다.

사용자는 DrivoR 대조실험의 필요성과 목적을 확정했으나 지금 대기열 등록은 명시적으로 금지했다. 기존 frozen 대조군 실행 승인은 유지된다. DrivoR 구현·profile·학습·평가의 신규 자동 실행은 하지 않는다.

공통 planner만으로 backbone·사전학습·SSL·memory예산 차이가 제거되지는 않는다. 두 frontend의 시스템 비교와 구조화 particle 속성의 인과 비교를 구분한다. 이번 사용자 질문을 기존 작업 취소나 즉시 planner 교체 명령으로 해석하지 않았고 실행 중 hash를 유지했다.

현재 구현은 DrivoR register만 particle로 교체한 통제실험이 아니다. 입력 카메라·해상도·시계열·memory개수·planner 후보생성이 다르다. 미래64particle×8step은 고정이며 상황별 예측 대상/예산 선택은 미구현이다. Planning-aware 표현 자체를 신규 기여로 단정하지 않는다.

모든 파라미터가 학습 가능하다는 것과 모든 scalar가 매 step 변경되거나 모든 모듈이 planning gradient를 직접 받는다는 것은 다르다. RGB decoder는 planning 그래프 밖이며 SSL로 학습한다. 모듈별 양수 gradient는 확인했으나 이번 턴에 각 scalar 업데이트 차분을 새로 측정하지 않았다.

이번 종료 예상은 공유 GPU 부하가 현재 수준으로 유지된다는 조건의 실측 기반 추정이며 보장/통계적 신뢰구간이 아니다. Frozen 대조군 속도는 미실측이다. 이번 턴은 로그 조회·ETA 기록이며 학습 source/config/queue는 변경하지 않았다.

고정대조군은encoder FiLM4480개까지고정/eval/no_grad하여표현변화를차단한다. Planner2207495개학습·ego명령입력유지. Adapter/LoRA와의차이는표현적응+FiLM학습+world train/eval모드를함께포함하고full스케줄차이도남는다. GPU감사·실측속도·학습효과는아직미확인. 새source49/config33등록불변유지;중단은outputs/lpwm_frozen_control_v1/queue/pause.requested또는선행v4pause를사용.

Adapter PDMS가평균최고이나LoRA/partial대비CI에0포함,1seed/1epoch라확정우열아님. 미래반복개입은표현활용진단이며미세조정효과분리/객체정보특화증거아님. 직진world9개로일반화불가,원거리미래객체flag의전체영상오차소폭악화. LoRA와초기후보일치/최종325개차이,partial과초기62개차이유지.

Full조건은native LPWM전체+planner학습이며LoRA/Adapter고정방식이아니다. 기존full의conv_in명령입력/20epochLR스케줄을유지하여다른세방법과의순수기법단독비교가아님. Batch8 profile실패와현재batch4정상본학습을구분한다.

2026-10-05 배치확대·재계산감소속도향상은실측전이며미확정. pjh재등장으로각GPU약20.47GB사용,우리예산은48GB에서해당점유와context를뺀범위다. 새진단script는구문검사만통과했고GPU/gradient동일성미검증.

2026-10-05 추가 확인: 과거 pjh-wamvla-v2의 GPU0·1 점유는 원본 조회로 확정됐다. 프로세스 종료 이력이 없어 정확한 이탈시각/경과시간은 미확정이다.

Junheok현재GPU0·1프로세스없음확인. 몇시간전에종료했는지는기록부재로미확정. 추가공동점유감소관측구간은10:52~14:34이며개인별종료기록이아님.

64설정존재와공개64weight존재를구분. 256dataset링크는256checkpoint증거아님. 고해상도LPWM재사용범위/성능미검증.

축소로작은객체정보손실가능하나우리PDMS손실량미검증. VAD Tiny/Base차이는여러구성이함께달라해상도인과증거아님. 모델간입력/센서/화각차이를명시한다.

128설정의호환성유지와자율주행최적성은별개. Crop28/왜곡resize의독립효과미검증. 고해상도구조확장의효과/비용미확정.

128해상도의작은객체정보손실은가능한제약이지확정된PDMS병목아님. Epoch확장으로공간정보손실이해결된다고가정하지않는다.

Epoch횟수는수렴증거가아님. Stage1 512clip temporal목적함수와past-only미래예측수렴은별개. Stage2 128monitor와1024최종점수혼용금지. 추가epoch성능향상·최적예산미확정.

Stage1 20epoch완료와Stage2 방법별1epoch비교를구분한다. Epoch완료만으로수렴·최적성은판정하지않는다.

2026-10-05 11:09 KST: 1epoch에서방법우열·충분학습여부확정불가. 다seed실험·frozen-LPWM대조·공식navtest는미실행. 문헌epoch예시는검증했으나해당논문반복seed수는확인못함. 현재queue변경없음.

LoRA경향gate는모두통과하나대부분미학습planner대비학습·world유지검사임. Frozen표현대조는아직없다. 현재initial저장출력62/1024후보불일치,최종567/1024후보차이;초기가중치차이/정밀도등원인은이번집계에서미확정. 변경위치·배치일정도달라기법단독인과효과단정금지. PDMS81.91은navtrain개발1024최종이며이전84.56은128monitor중간점수.

Adapter초기학습진행은확인했지만planning효과는미검증. NativeCNN·출력head·RGBdecoder가중치는고정이며중간표현에SG를넣은것이아님. 기존학습가능intentFiLM은attributeCNN conv_out에유지된다. Inventory의planner수는FiLM도포함. Context가dynamics와공유되어region별총parameter단순합산금지.

Adapter실행중이라는해석은현재시점에틀림. LoRA학습은완료됐지만검증은진행중이며Adapter profile도아직시작하지않았다. 모든후속방법batch8우선/각GPU전체48GB기존정책유지.

915개제외사유는 planning_manifest의 discarded_irregular_timestamp_tokens와코드상12frame간격0.5s/atol0.1검사로확인. Stage1수량차이는미래포함12RGB가용성과timestamp조건때문. 이전heldout40recording은현재학습122에포함됐으며현재navtrain에독립최종test가남았다고설명하지않는다.

LoRA 학습완료와 검증완료를 구분: 본학습 정상종료는확인했으나 현재최종검증은진행중. PDMS개선·world유지 최종결론은검증완료후확인. Monitor128 PDMS83.6380은navtrain내부개발이며공식navtest아님.

navtrain이라는 이름은 전체 공식 filter 풀과 그 안의 실제 gradient 학습 부분을 혼용할 수 있으므로 token manifest 기준으로 표기. 최종 학습에 val을 합친 후 그 val은 독립 검증 아님. 우리 기존 자체dev는 DrivoR 기본val과 동일하지 않으며 분할 변경 시 Stage1 노출도 고려해야 한다.

우리 내부 개발 분할은 DrivoR navval과 동일하지 않다. 현재 1epoch 결과는 경향 실험이며 논문 최종 성능 아님. 공식 테스트 단계와 수렴·표현 기여 대조군은 여전히 필요하다. 모든 논문이 val/test를 동일하게 사용한다거나 test를 반드시 한 번만 평가한다고 일반화하지 않는다.

"전체navtest아님"은navtest일부를평가했다는뜻으로오해가능했으므로"navtrain내부개발검증"으로명시정정했다. 현재LPWM의공식navtest/navhard점수는없고내부PDMS를논문숫자와직접비교불가. 이전공식Drive-JEPA navtest완료자산과현재LPWM결과를구분한다.

중간해석: LoRA가유망하나4096시점128scene/1seed결과이며모든시점에서우세하지않다. Partial의미래LPIPS보존실패는명시하며SSL가중치/학습범위최적값을이번질문으로바꾸지않았다. PDMS는고정후보공식cache기반개발점수이며전체navtest/SOTA직접순위비교불가. Planner학습효과와LPWM적응효과는분리대조가없어미확정.

LoRA8은47.60GB에서실제profile및본학습재개성공. 후속Adapter/full도8을먼저시험하도록등록했으나해당방법의8실측성공을이미확인한것은아님. 총48GB감시는유지하며초과/메모리실패시에만축소. 기존부분학습forecast보존실패와원인분리대조군부재등과학적한계는이번실행변경으로해소되지않았다.

현재v4는512MiB보다작은192MiB추가CUDA여유를실측근거로적용하며GPU카드전체48GB를완화하지않았다. nativeallocator상한은shared점유에따라달라진다. 모든후속방법8우선정책은v4에공통등록됐다.

최신사용자승인은모든후속학습에batch8우선/각카드48decimalGB상한공통적용. 실행변경만허용하며loss/모델/학습량/GT사용변경없음. SharedGPU타인점유급증은사전보장불가; 동적allocator와0.5초감시로보호. 안전여유내실측으로8가능여부판정,실패를숨기거나작은배치를8이라고보고하지않는다.

최신: 메모리 감소 원인은 Partial8에서 LoRA4로 방법·microbatch가 바뀐 것. 유효batch16 유지와 처리속도 동일은 별개다. Batch8의 실패는 명시적 프로세스 allocator guard에 의한 OOM이며 물리48GB 용량만의 한계로 단정할 수 없다. Partial forecast LPIPS 유지gate 실패는 보존하며, LPWM 미세조정 단독효과/최종독립test는 여전히 미확정. LoRA 최종결과 대기.

최고치95.1은2026-10-05조사에서확인한NAVSIM-v1논문보고치. 공식실시간리더보드1위는미검증. TE/SimScale조건의효과를우리LPWM의효과로전이해단정하지않는다.

최신개발PDMS는79–82주변변동이며단조상승아님. 이번배치8본학습은정상. 최종world보존/독립test/미세조정단독효과는미확정이고frozen대조군은제안단계. 현재학습이나실험등록변경없음.

현재미학습planner초기값대비상승은LPWM미세조정단독효과를식별하지못한다. 학습된planner에Stage1 LPWM을교체하는것도표현분포불일치때문에단독증거로부족. Frozen대조및SSL-only대조는아직제안단계.

최신중간판단: 정상실행과개발PDMS상승은확인되지만ADE/FDE는직전보다악화. LPWM적응단독planning이득과개발world성능유지는아직미확정.8batch실행후PDMS검증은2048에서처음수행된다. Source38/config20해시일치/전체VRAM45.6GB/OOM0.

최신: GPU0·1카드전체48decimalGB/물리free3GiB/allocator23.2GiB/allocated22.75GiB. 유효planning16/SSL8/worker0/GT OFF/navtrain1epoch4707유지. 원본1576와full2095상태보존. Batch변경에따른dropout·개별SSL추출RNG변경은가능하며bitwise동일재개가아니다.8update실측은장기속도보장이나PDMS개선증거가아니다. 본학습및후속방법성능은검증대기. SharedGPU급증까지절대OOM방지를보장하지않으며상한감시로우리process만중단한다.

48GB검토는추정이며batch8실행성공/실패결과가아니다. 상한과물리free여유,GB/GiB를구분한다. 기존원격인증오류는미해결.

최신속도검토: CPU loader 비교는 end-to-end 훈련속도 측정이아니다. 현재설정의 전역최적이나 새speedup은 미입증. 두batch점메모리외삽은추정이며, GPU전체사용률은우리프로세스사용률과구분한다. 현재 runtime34source/10config불변.

원격 공유: c4cc73e에 코드·설정·보고서를 로컬 commit했다. 이번 git push mine은 기존 VSCode credential socket ECONNREFUSED/GitHub 인증 실패로 끝났다. 원격 반영은 미완료이며 실행 중 학습과 두 queue에는 영향이 없다.

최신 승인 범위는 전체train1epoch의 네 조건 경향 비교다. 객체GT loss는 모두OFF/후순위.
Full은 기존2095update를 이어받으므로 새학습2612update이며, 원래conv_in명령과20epoch스케줄을 유지한다.
다른 방법과 명령 위치/스케줄/적응 위치·자유도가 달라 순수한 PEFT 효과를 분리한 비교는 아니다.
No trained frozen-LPWM control,1seed,노출된dev panel; 독립test/수렴/LPWM만의 인과적 이득은 미확정.
원본2095 checkpoint와pause기록 보존, 새로운자동재개는 별도full후속조건만 승인됨. GPU46GB는 CPU RAM 제한이 아니다.


- LoRA가현재출력계층부분학습보다더빠르거나PDMS가높다는근거는아직없다. 현재particle재구성을목적으로하면영상encoder경로의적용위치가핵심이며context/dynamics전용LoRA와구분한다. 직전b2c48fd원격push도GitHub인증오류로실패했다.

- 이번턴은상태확인·요약·ETA보고다. 초기무작위planner평가만있고학습후PDMS개선미확인. 두조건GT OFF/ON각1epoch와SSL유지/부분갱신설정그대로이며새대조조건/추가epoch를등록하지않았다. 기존fullrun2095updatepause유지. 직전8294fc3원격push는인증오류로실패했다.

- 후속질문에LPWM원문§5.2/A.5의frozen world+mappingL1, Drive-JEPA encoder1e-5/planner1e-4, V-JEPA2-AC frozenencoder, UniAD Stage2고정backbone/BEV와task공동학습, OpenVLA LoRA경로를확인해설명했다. 현재12개출력모듈선택은우리의계산예산설정이다. 표현수정독립효용에는frozen-LPWM+trained-planner대조가후속으로필요하며현재GT두조건queue는변경하지않았다.

- **최신사용자는부분학습으로빠른경향확인을승인했다.** 두조건은미세조정방법2개가아니라객체GT보조OFF/ON이다. 사용자의후속질문에동일Stage1+동일초기planner에서각각1epoch,순차학습검증한다고설명했다. GT채택미정/Stage1SSL/GPU0·1전체46GB제한유지. 기존fullpause는유지하고새실험만시작했다.
- 이번비교는1epoch/1seed/고정소규모dev패널이다. 초기planner대비개선만으로LPWM부분미세조정독립효용을입증하지않는다. Frozen-LPWM+trained-planner 동일예산대조/수렴/다중seed는후속. Fullrun과학습량·LR·FiLM위치·평가패널도달라단순수치순위비교금지. GT입력없는OFF도정답경로/privileged metric감독을사용한다.

- **최신 확정 상태는 사용자 pause다.**2,095 update까지 보존/재개가능성 CPU검증/우리GPU메모리반환확인. 명시적 요청 전 재개하지 않는다. LoRA/부분학습 전환은 이번에 요청·적용되지 않았다. 기존 first-epoch ETA와 종료 ETA는 중단으로 무효이며 재개 시 다시 계산한다. 직전92509d9 원격push는기존인증오류로실패했다.

- 최신 요청은 현재 학습을 상세 설명하는 것이다. 기존 queue를 그대로 유지했다. Config 설명문 일부가 구안이지만 actual conditions와 model 분기는 명확하다. 실행 중 hash를 바꾸지 말고 향후 종료 후 문구 정리를 고려한다. 직전75687c2 push는 기존 GitHub 인증 오류로 실패했다.

- 이번요청은실행중학습중간점검이다. 기존training을보존하며로그/가중치/고정128dev의초기진전을확인했다. LPWM표현미세조정효용·객체GT채택·영상/미래성능유지·최종PDMS는여전히미정. 신규진단은완료/기존queue계속,본학습재시작없음. 직전1eb9471의원격push는기존Git인증오류로실패한상태.

- **최신 사용자 결정:** Stage2실행승인,객체GT보조감독채택은미확정이며없음/있음어블레이션후결정. Stage1SSL유지. GPU0·1각전체VRAM46GB이하/CPU RAM46GB제한아님. 본학습source/config고정. 같은후보teacher를쓰므로‘GT없음’은직접객체보조loss없음이지privileged metric-supervision까지없는조건은아님.
- 새진입amendment는top16 proxy만보조지표로재분류,원래실패수치/기준/source보존. 성능향상판정실패시반대조건의실험은계속하며실행정합성/누출/메모리실패는중단한다. 조건당20epoch고정. 기존3조건(refiner/imitation-only)계획은현재2조건비교에의해보류. Navtest는노출이력이있어독립검증으로자동진행하지않는다.

- 이번 요청은 완료 검증 결과 보고다. 현재 정보 판독의 부분 개선으로 해석하며 객체 자동 발견/정밀 상태·미래/PDMS 향상으로 일반화하지 않는다. 현재 판독만 완료, Stage2/새 gate 변경 없음. 직전 bb3a6be의 push는 기존 credential socket/GitHub 인증 오류로 실패해 원격 동기화가 안 된 상태다.

- 이번요청은검증요약/객체라벨범위설명. GT는평가head감독/association에만쓰고Stage1SSL표현은고정이다. 원본3Dbox를camera로투영하며pixelmaskGT는현재검증cache에없음.

- 최신 사용자 방향 확정: Stage1에는 객체 GT 입력/loss를 넣지 않고 Stage2에서만 GT auxiliary+planning+SSL 유지. 감독 없는 요소도 표현해야 하며 전체 pipeline을label-free로부르지 않는다. Stage2로감독을옮기는것만으로label편중해결을보장하지않음. 이번수정은문서원칙이며진행중frozen진단과기존runtime는불변.

- 최신 사용자는 검증 방법 적용을 승인했다. 이에 frozen readout 진단을 실제 시작했지만 GT보조loss Stage1/Stage2 학습은 아직 제안 단계다. No GT-box deployment input, LPWM update0, GPU0·1만 사용/6GiB reserve/8GiBallocated cap. Shared 원본 데이터 read-only.
- 현재 판독은 full64 decoding alpha를 사용하는 평가 구성이다. Native decoder는variance선택30개이며 native instance mask와 혼동하지 않는다. Pixel mask의수동검수/미래상태판독/학습된planner개입은 이queue가 완료해주지 않는다. 과거4장 GTassociation을 사용하므로 자동tracking 결과도 아니다.

- 이번 요청은 ‘객체 구분을 어떻게 검증할지’에 대한 방법론 제안이다. 작은 판독기는 평가용이고 LPWM을 갱신하지 않는다. 성공해도 자동 검출/planning 활용을 증명하지 않으며, 실패만으로 모든 객체 정보 손실을 입증하지 않는다. 새 학습/전체 feature 추출/수동 mask/Stage2 실행은 이번 턴에 수행하지 않았다.

- LPWM에새detection/semantic segmentation모듈을붙인것이아님. 복원alpha mask와latent 객체/부분분해가능성을‘객체구분불가능’으로일반화하지않으며,particle–instance일대일대응/semanticlabel은보장되지않음. Box gate미통과와LPWM도메인적응실패를동일시하지않음. Proxy만으로Stage2중단을정당화할과학적근거는미확립이나현재runtime차단을임의우회하지않음.

- Stage1 개발검증완료시각12:17KST,단일seed/40recording bootstrap CI이며독립test·seed불확실성아님. 고정gate의객체box대응실패는유효하지만미래box/IoU.3는개선되어실패원인은미확정. Stage2 PDMS결과없음. 이번사용자요청은결과확인으로처리.

- 이번 요청은 planner의 문제의식/방법론 검토다. DriveSuprim 계열 선택형 planner 추천은 공식 DriveSuprim 재현/최종 방법 확정/학습결과가 아니다. 기존 사용자 승인 LPWM low-LR+planner full training은 유지하며 frozen LPWM은 원인 구분용 대조 제안.
- 새로운 중심 가설과 novelty는 미검증. CAPO형 utility는 로그/scorer/후보에 의존하는 정보 손실 proxy이며 reactive causal ground truth가 아니다. 손실 가중과 예측 후 token선택을 실제 rollout 연산절감으로 부르지 않음. 현재3조건 queue에 새 ablation이 이미 추가됐다고 보고하지 않는다.

- 안전BCE가 공유decoder를 학습하는 것과 좌표를 안전방향으로 직접 미분하는 것을 구별한다. SafeDrive와 우리refiner를 같은gradient설계로부르지않음. UniAD공개kernel검사결과를논문저자실험전체에일반화하지않음. 이번8연구 비교는 문헌/코드와 작은CPU검사이며PDMS우월성검증없음.

- Stage1 적응 gate는 전체평가 후 실패했으며 Stage2가 시작되지 않음. Box 대응 proxy 감소와 reconstruction/future 예측 개선이 공존한다. 최신 질문은 BCE/SG의 의미 설명으로 처리했고 runtime/source등록 변경 없음. 추가 안전좌표loss의 방식·가중치·검증은 아직 미등록.

- Refiner 고유 decoder/offset head는 WTA 회귀·comfort로 직접 학습한다. Refined 안전 BCE는 공유 LPWM/coarse feature와 scorer를 학습하지만 detached pose 경로로 offset head를 직접 학습하지 않는다. 추가 collision 좌표 loss/DrivoR식 완전 분리는 현재 실험에 미구현이며 이번 설명 요청으로 runtime을 변경하지 않음.

- 배경집중의원인을reconstruction하나로단정하지않음. RGB면적/무늬/patch기원/해상도도가능원인. 정지선준수독립metric없음,위치이동은planning표현개선의필요·충분조건아님.

- Particle 중심/scale는지역표현의기하이며검출객체정답경계/영속적tracking을보장하지않는다. GIF는동일장면의학습checkpoint변화이지실제시간객체이동영상이아니다.

- 평가진행률/ETA는부분개발결과의성능판정이아니다. 전체Stage1gate/Stage2시작미완료이며GPU소량사용은현재추론단계때문이다. 처리율변동/후속집계시간은ETA에불확실성을준다.

- Stage1학습완료와적응검증완료를구분한다. 개발dynamicsKL후반상승은남아있지만최종8장면자율미래MSE는epoch15대비추가개선. 전체7,745평가/CI/위험별결과로판정한다. 이번원격push도기존Git인증문제로실패(b7a42d3); 로컬기록보존.

- 최신점검에서복원개선지속이나후반개발dynamicsKL상승,소수장면자율미래예측정체관측. 과적합/실패로단정하지않으며 전체개발과거전용평가대기. Teacher coverage통과는별도oracle/vocabulary gate나Stage1성공을의미하지않음.

- 10/04 상태 점검: 학습/개발ELBO 개선과 정상 실행 확인, 전체7,745개 과거만의 미래예측·객체 적응gate 및 Stage2 PDMS는 아직 미확정. 기존원격 Git 인증오류 상태는 이번push 결과와 함께 확인.

- 최신 공유 상태: e968b1a 로컬 commit 완료. 이번 `git push mine`도 sandbox DNS 실패 후 host 재시도에서 기존 VSCode credential socket ECONNREFUSED/GitHub 인증 실패로 끝났다. 원격 반영 미완료, 학습·queue에는 영향 없음.

- Stage1 RGB decoder는 `decode_with_ctx=False`; 복원 gradient는 encoder/decoder에, dynamics/context KL이 encoder/context/dynamics에 전달된다. Stage1은 ego command/객체 GT/경로 loss 없음. Scalar loss에서 복원 비중≈96%는 gradient 비중이 아니다. 현재 module gradient만 기록하며 loss별 norm/방향은 미측정. Stage1/2의 완전한 인과분리를 완료했다고 주장하지 않는다.

**최신 planner 범위:** 3조건 full navtrain20epoch/seed47. world-off ablation은사용자추가refinement요청을우선해후속으로보류.
LPWM은full-low-LR, 새planner는full-LR; LoRA/encoderfreeze아님. 원본체크포인트보존.
카메라128×128/정규화particle depth의한계, 객체ID미보장, action-conditioned worldmodel아님.
적응gate통과가완벽한주행이해나planning개선증명은아니다. Stage2 GPUprofile·성능은미검증.
CPUteacher는16worker/host여유64GiB, 학습중onlineoracle는4/rank. GPU0·1만, DDP입장free44GiB/실행reserve6GiB/allocated상한38GiB.


- 공유 상태: dea246b 로컬 commit 완료. `git push mine`은 DNS sandbox 오류 후 host 재시도했으나 VSCode credential socket 연결거절/GitHub 인증실패. 원격 반영 미완료이며 GPU학습은 영향 없이 계속된다.


**현재 확정/미결:**
- Stage1 공개전체 LPWM world-model 적응 → 성공 시 Stage2 저LR LPWM+전체planner 공동학습. frozen LPWM 및 별도Stage3는 사용자정정으로 대체됨.
- 현 실행설정은 측정한 범위의 합리적선택이며 모든 batch/precision/worker의 전역최적이 아니다. 학습을멈추는 추가탐색은 사용자최신지시와 충돌한다.
- OOM 여유6GiB를 지키며 다른사람프로세스/공용데이터를 수정하지 않는다. CPU RAM은 충분하며 fulltrainer자체 hostRAMcap은 없다(workerbenchmark만32GiB reserve감시).
- Stage1은20epoch의1seed. adaptationgate는dev기반, 독립test/객체identity/planning향상 증거가 아니다.
- Stage2 VRAM·throughput·완료시각은 아직 미측정. 독립CPUaudit 및 구문검사만 완료, gate 이후 GPU/DDP profile을 필수 실행한다.
- Stage1첫시도의 첫update정체 원인 미확정; 계측후 재기동은 정상. source/counters/log를 보존했으며 완료하지않은update를 학습량에 포함하지 않는다.

### 과거 범위 이력


**최신 방향 수정: LPWM encoder·context·dynamics·planner 공동 학습.** 사용자가 원본 context/dynamics를 제외한 선택을 지적하고 공동 학습을 요청했다.
기존 encoder-only는 완료7run(frozen3/planning3/uniform seed29)만 보존하고 우리 worker3425264/3425265/scorer3425266을 SIGINT 종료했다.
`outputs/lpwm_planning_v1/superseded_by_joint_world_model.json`을 따른다. 이전18run queue/finalizer를 자동 재개하지 않는다.
새 코드 `src/planning_aware_future_prediction/object_centric/lpwm_joint_world_planner.py`: 공식 encoder6.035M/context39.389M/dynamics59.869M+planner0.821M, RGB decoder만 제외.
과거2영상→관측transition posterior→미래8step은policy prior만으로 autoregressive rollout, activation checkpointing으로 gradient 보존.
GPU0 batch1 3update에서 planning/future loss 각각 세모듈gradient>0, 미래label변경시예측동일을 통과했다. bf16 batch4 profile 진행/새 데이터·규모 준비 중이다.
기존512train 중13개/192dev 중2개가 공식navtrain token필터밖임을 발견했다(로그는전부navtrain). 새 학습은 공식token까지엄격필터한다.

**최신 LPWM 범위:** 작은 continued adaptation과 개발 평가이며 논문 전체 재현/새 독립test/PDMS 평가는 아니다.
Object-centric decomposition과planning중요도는미확정, particle ID는persistent객체ID가 아니다.
회전warp는병진시차를제거하지않고시야손실이있다. 가림후보는projected-overlap proxy이며semantic/visibility정답이없다.
SAVi++/3D-DLP를고려하면depth/3D추가자체는novelty가아니다. 완료학습은더확대하지않는다.


**시각화 범위:** 저장된 개발 결과와 실제 사진이다. 선택된 3사례는 결과 기반 설명용이며 일반 성능 표본이 아니다.
BEV에 지도/장애물 정보는 없고 closed-loop 재생도 아니다. 미래 특징 중요도나 생성된 미래 영상으로 해석하지 않는다.
공유 PDF에 그림을 보존하고 PNG는 로컬 outputs에 둔다. 결과 원본 파일은 변경하지 않았다.

**확정:** 두 블록 학습으로 encoder 표현과 개발 ADE를 바꿀 수 있다. 내부 intent도 표현에 영향을 준다.
**미확정:** 선택적 미래 감독의 실질적 planning 이득, 6블록의 2블록 대비 우월성, 독립 test 일반화.
이번은 작은 512 train / 192 dev의 부분 encoder continued fine-tuning이며 LoRA·전체 encoder 사전학습은 아니다.
미래 target은 fixed camera region·4개 horizon·32개 region으로 고정했다. Learned adaptive selector·객체 추적·동적 예산은 미구현이다.
Bootstrap은 seed 평균 차이의 recording 불확실성만 나타내고 다중 비교 보정이 없다.
원본과 공용 데이터, 이전 결과는 보존했고 모든 GPU 작업은 종료했다.
결과 commit은 `955e82f`다. `git push mine`은 sandbox DNS 실패 후 제한 밖에서 재시도했으나
VS Code Git 인증 소켓 ECONNREFUSED / No anonymous write access로 실패했다.
실험·평가·로컬 보존은 완료했으며 인증 복구 후 push만 남았다.

**아래는 이전 실험의 미결 사항이다.**

**현재 미결:** 세착안변형의추가이득/미래예측필요성/학습선택우월성미확인. Region변형이며객체기반논문재현/독립test/동적예산개선이아니다. 실험 결과 로컬 commit은 `215cb6b`다. `git push mine`은 sandbox DNS 차단 후 밖에서 재시도했으나, 기존 VS Code Git 인증 소켓 ECONNREFUSED / No anonymous write access로 실패했다. 인증 복구 후 push만 남았으며 실험·평가는 모두 완료됐다.

현재region-retention은frozen current-planner의정보유지proxy이며 semantic/인과/future중요도정답아님.
Globally contextual latent/mean대체분포변화가한계. 큰영역평균은공간상세도를낮추므로픽셀면적과출력token예산을별도보고.
기존K4warmup에서이어진비교이며처음부터최적region학습을증명하지않음. 공식PDMS/heldout/WA재개없음.

최신 시각화: strict submodule loading·CPU/GPU 저장결과 대조 통과. Encoder는 기존 캐시,
planner/scorer는 이번에 다시 실행하지 않았다. 사진은 실제GT, latent heatmap은1024개channel
값이지 공간/RGB reconstruction이 아니다. 선택의 인과적 중요도와 learned정책우월성은미확정.

최신: 현재 학습형 선택의 우월성은 미확인이다. Gradient 경로의 존재와 실제 불연속 선택
효용을 근사하는지를 구분한다. 공간적 선택 집중은 redundancy 가능성이지만 collapse 증거가 아니다.
W-gradient의 실제단일patch교체 loss부호일치율은MLP86–92%,ego74–80%; full score/optimizer유효성 보장은아니다.
같은 낮은LR의 fixed/random과 learned를 비교하기 전 서로다른LR 결과로선택우월성을주장하지않는다.

밤샘연구질문/고정K4/front/원본planner보존은확정. 어느최적화조건이개선되는지는실행전미확정.
동적K/horizon/WA재개/heldout/navtest튜닝/새확률모델은이번범위밖이다.

이번 LoRA는 원본 planner encoder fine-tune이 아니라 미래 branch 복제 tail의 적응이다.
EMA teacher 미도입, frozen future target 유지. Dev 개선 경향은 있으나 일반화/선택 우월성/공식 안전지표는 미검증이다.
학습비용과 제한된 추론비용은 실측했다. Shared GPU timing을 순수 architecture speedup으로 해석하지 않는다.

**현재 작은 비교의 한계**: foundation checkpoint가이미학습했을수있는navtrain의extension용split이다.
Dev64/8recording·200update는최종독립평가아님. ADE/IL는공식안전·진행지표를대체하지않는다.
Fixed/random은selector를호출하지않아activeparameters가learned보다작다.
Aux없는비교도새경로가현재feature변환으로작동할수있으므로미래정보의기여를단독식별하지못한다.

**최신extension미결**: fixedcamera-grid target은객체/ego정렬world state가아니다. ST는current내용과위치의biasedsurrogate이며
hard선택변경의미래target 교체/중복제외 gradient를근사하지않는다. 원본planner재사용/gradient통과가선택유용성의증거는아니다.
최초zeroresidual에서는selector/predictorgradient0이정상; bridge1step후검사와구분한다. 현재baselineweights는freeze/preserve.
Split노출기록 `results/drive_jepa_selective_future/project_split_exposure_audit_20261002.json`을새최종프로토콜에반영해야한다.

**이번재개의미결**: 공유GPU의타인메모리/연산수요는변동가능하며reserve/guard는OOM완전보장이아니다.
전체평가완료시간은현재동시사용속도로실측후판단. Fullmetric은12146완료전확정하지않는다.

**이번 추천의 미결**: WA native spatial-tube단위 승인 vs 객체instance유지. WAfullcheckpoint/config
strict호환/공식score/VRAM/latency, sparse-all 동등성, 실제planning→ST선택학습과camera4coverage.
Position-mediated ST는편향surrogate설계이지fullgradient검증결과가아니다. 추천을최종novelty확정으로부르지않는다.

**확정**: 미래 예측 대상/필요성·예산 배분이 연구 질문; 고정 K·horizon은 개발 기반으로 사용;
첫 target 비교는 selector를 고정하고 감독부터 분리; GPU 0·1만 사용;
공용 원본 직접 수정 금지; 기존 환경/프로세스 보존; 작업공간은 `/rhome/junseong`;
새 데이터셋 원본 다운로드만 `/home/user/data/processed_dataset/`에 총 1 TB 한도.

**미결**: 최종 visual/spatial/mixed target/선택 연구 기반, ST 공동 학습 안정성,
multiview/occlusion/GT 대체association, 미래 활용·동일 예산 효과·novelty delta·독립 holdout.
이번1000update도A–E는단일seed(별도E/F만대응2seed)이고persistence를못넘었다.
이 문장의1000은607da52까지의 이력이다. 이후 C visual-only 대조3seed를 완료했고 공개 기반으로 우선순위가 바뀌었다.
C/E저분산은추가학습으로개선됐지만 capacity/조건부평균/regularization/실제미래활용의분리는남았다.
JPEG original-distorted 취급은 명시적 운영 가정으로 별도 원본 byte 증거는 없다.
Frozen visual teacher와 GT ROI는 구현됐지만 deployment perception/일반화는 검증하지 않았다.
현재 ST는 편향된 임시 추정이다. 작은 연결 검사 성공을 성능·효율로 일반화하지 않는다.
공식Drive-JEPA PF 전체평가는완료됐으나논문과의차이0.224320점의정확원인/허용오차는미확정이다.
PF 추론에는별도futurepredictor가없으므로선택적미래예측기반이최종확정됐다고해석하지않는다.
**명명 규칙 확정**: 프로젝트·파일·class·function·인자·변수·config·result key가 역할을 직접 설명해야 한다.
현재 작업명은 Planning-Aware Future Prediction이며 최종 논문명·방법명은 미확정이다.

AD-E2E-JEPA v1 공개일은 **2026-09-28**이다. 과거 “3주 전” 표기는 잘못됐다.
원문 §3.4의 downstream IL은 patch predictor를 제거하는 경로다.
이 사실을 goal-conditioned zero-shot 경로 또는 모든 JEPA 구현에 일반화하지 않는다.

아래 6절은 **이전 서버 이전 기록**이다. AICA 환경·과거 데이터 크기는 현재 서버의 검증 결과가 아니다.

## 6. 다른 서버에서 재구성 (git 으로 오지 않는 것)

**AXE-080 데이터 규칙(2026-09-30 확인):** `/rhome/junseong/`이 코드·변환·cache를 포함한
작업공간이다. 연구실 공용 원본 `/home/user/data/Dataset/`은 절대 직접 수정하지 않고 작업공간의
프로젝트에 심볼릭 링크로만 연결한다. 새로 필요한 데이터셋은 `/home/user/data/processed_dataset/`에
총 1 TB 한도 안에서 내려받는다. 현재 SafeDrive 링크는 `dataset -> /home/user/data/Dataset/navsim`이다.

| 항목 | 이 서버 위치 | 옮기는 방법 |
|---|---|---|
| conda env (6.7 GB) | /home/kaist5/miniconda3/envs/safedrive (링크 `kjs-SafeDrive-exp2`) | `scripts/run/build_env.sh` + `build_mmcv.sh`. **mmcv 2.1.0 은 대상 GPU arch 로 소스빌드**(sm_90=H100). 사전빌드 wheel 은 sm_90 커널이 없어 deformable attention 이 조용히 전부 0 이 된다 |
| 체크포인트 3.0 GB | ckpts/safedrive_phase{1_90ep,2_5ep,3_10ep}.ckpt | rsync |
| 데이터셋 | `dataset -> /home/user/data/Dataset/navsim` | 공용 원본은 읽기 전용으로 취급하고 심볼릭 링크로만 사용 |
| navtrain feature cache 444 GB / metric cache 22 GB | exp/safedrive_train_cache, exp/train_metric_cache_navtrain | rsync 또는 `scripts/run/cache_navtrain.sh` 재생성(수 시간). **일부만 없으면 `scripts/run/repair_cache.sh` 로 없는 것만 채운다** |
| navtest/navmini 캐시 5 GB | exp/metric_cache_navtest, exp/feat_cache_navmini, exp/metric_cache_navmini | rsync |
| 실험 산출물 | exp/safedrive/*, exp/training/* | 결과 CSV(`eval_*/traj_*.csv`)만 rsync 하면 충분 |
| 학습 체크포인트(진행 중) | exp/safedrive/{o0_nofuture,f3_nopairnc}/lightning_logs/checkpoints/ | 재개하려면 rsync. O0 는 epoch 1, F3 는 epoch 0 까지 |

**git 으로 오는 것**: 코드, 설정 yaml, `scripts/run`·`scripts/analysis`, `analysis/*.csv`(라벨·측정값),
RESUME_NOTES / EXPERIMENT_DESIGN / HANDOFF / AGENTS, trajectory_anchors.

### 서버 이전 실전 메모 (2026-09-30)

이전 서버 `cloud-orO3Hf` 는 사설 IP(192.168.0.2)만 가진 **NAT 뒤 클라우드 VM** 이라
outbound 가 막혀 있다(AXE-28·AXE-080 모두 ping·포트22 도달 불가). 그러나
**외부 IP `61.107.200.100` 으로 들어오는 것은 된다** — 이전에 새 서버에서 당겨오기(pull)로
성공한 이력이 있다.

→ **`tools/pull-from-cloud.sh` 를 새 서버에서 실행한다.** tier 로 나눠 두었다.

| tier | 내용 | 크기 |
|---|---|---|
| 1 | Claude 메모리·대화 + 평가 결과 CSV + 재개용 `last.ckpt` | **2.6 GB** |
| 2 | 사전학습 ckpt + navtest/navmini 캐시 | 8 GB |
| 3 | navtrain metric cache | 22 GB |
| 4 | navtrain feature cache (**권장하지 않음**) | 444 GB |

```bash
bash tools/pull-from-cloud.sh 1          # 포트가 22 가 아니면 SRC_PORT=xxxx 를 앞에 붙인다
```

tier 1 이 `exp/safedrive` 전체(52 GB)를 받지 않는 이유: 중간 epoch 체크포인트가 run 당
7.8 GB 씩 쌓여 있는데 재개에는 `last.ckpt` 하나만 필요하다.

그 외 원칙:

1. **코드·스크립트·분석 CSV 는 git 으로만 옮긴다** (이미 `mine` 원격에 push 됨, 16 MB).
2. 새 서버에서 **`bash tools/check-new-server.sh`** 를 먼저 돌려 GPU·디스크·데이터셋·conda·
   레포 상태를 확인하고, **없는 것만** 옮긴다.
3. **`exp/safedrive_train_cache`(444 GB) 는 옮기지 않는다** — SafeDrive 전용 feature builder
   산출물이라 Drive-JEPA 등 다른 모델이 재사용할 수 없다. JEPA 전환에는 원본 데이터셋만 필요하다.
4. Claude 메모리·대화는 git 으로 오지 않는다. `/home/kaist5/data/junseong/claude-context-0930.tar.gz`
   (2.5 MB, `memory/` 14개 + 이 세션 transcript)를 별도 경로로 옮긴다. **메모리에 pin 된 연구
   명제가 들어 있으므로 이것만은 꼭 옮긴다.**

원본 데이터셋 실제 크기(`/home/kaist5/Dataset/navsim/dataset`, 심볼릭 링크 모음):

| 항목 | 크기 |
|---|---|
| sensor_blobs | 2.5 TB |
| **navhard_two_stage** | **31 GB** ← NAVSIM v2 navhard 데이터가 이미 내려와 있다 |
| navsim_logs | 16 GB |
| private_test_hard_two_stage | 10 GB |
| maps | 1.4 GB |
| warmup_two_stage | 1.2 GB |

### 환경변수

새 서버에서 경로가 다르면 다음만 맞춘다(기본값은 이 서버 경로).

```bash
export SD_SCRIPTS=<레포>/scripts/run      # 스크립트끼리 서로를 찾는 경로
export SD_LOGS=<레포>/exp/logs            # 학습·평가 로그 출력
export SD_ANALYSIS=<레포>/analysis        # 라벨·측정 CSV
```
