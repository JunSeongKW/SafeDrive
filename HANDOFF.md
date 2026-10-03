# HANDOFF — 이 파일 하나로 다음 에이전트가 이어받는다

마지막 갱신: 2026-10-03 09:45 KST (Codex)

세션 시작: 이 파일 + `git log -10` + `AGENTS.md`.
세션 끝: 상태 문서 갱신 + `tools/handoff-commit.sh` + `git push mine`.
**최신 사용자 요청: 학습된 selector/predictor 시각화.** 밤샘 87run/61,200update는
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

- 최신: 학습된 selector/predictor CPU 시각화 script·config·6개 계약검사·공유요약 추가.
- 현재/future GT 사진과 predicted latent를 명확히 구분하고 before는auxwarmup후임을 표기.
- 동일 final-selected 위치로 warmup/final/persistence 비교, 공식front crop/resize 재사용.
- 192dev 전부 저장GPU 선택ID일치와MSE오차tol1e-5 검증, 144개CPU테스트 통과.
- 이미지·NPZ·전체JSON은 로컬갤러리/ZIP, Git에는코드·설정·소형요약·기록만 포함한다.
- 기존 18run 등록 상태를 완료87run/commit8182f6c로 갱신. 추가학습/GPU점유 없음.

- 사용자밤샘자율실험승인 반영/09:00KST마감·공유GPU상한 고정.
- 후속704표본선정/미래validity미사용·노출dev제외·기존train포함검사/9GiB상한을등록했다.
- 복구가능학습runner에고정/무작위선택·현재feature대조·nested128/512조건을추가,야간단일GPUqueue구현.
- 기존공식image_fc복사/동결·rank4LoRA 미래projection대조와낮은LR/relative-memory-L2대조를추가등록.
- 원본projection/encoder/planner는수정하지않음.초기동일성은동일no_grad실행조건에서검사한다.
- 기존모델읽기전용원인진단script 및 gradient-routing모듈/검사/복구가능run/config/계획추가.
- 원본frozenplanner/기존cache/checkpoint보존. 이번결과는실행후별도기록한다.
- 첫48run 완료 결과와 독립 CPU artifact/scene 집계 감사, spatial selection 분산 진단을 추가했다.
- 이후 읽기 전용 실제 patch교체 진단을 등록했고, 실행 source/config hash와 의미가 명확한 JSON 필드를 기록한다.

- d3bbced의 사전 고정 5조건×3seed 추가 학습을 4500update에서 종료하고 결과·비용·곡선을 공유했다.
- MLP+새절차 dev ADE0.209975m, 원본0.220644m; 작은 개발 표본의 CI는0을포함한다.
- Contextual residual의 미래 MSE 개선과 planning 효과를 분리하고, LoRA의 추가 이득 미확인을 기록했다.
- 학습된 실제 영상경로의 off 원본 보존/online-cache 일치/엄격delta복원/추론 비용을 검증했다.
- 최초postflight의미세차이도보존했다. 원본 hash불변/OOM0/모든우리GPU프로세스종료.
- 공유JSON/CSV, 보고서와재현명령, 최신handoff를갱신했다. GitHub push는credential오류로미완료.

**2185ce5 이후**: frozen-encoder cache forward/explicit fixed·random patch ID 입력을추가.
원본planner와weights 불변, smalltrain/dev3seed 선택비교 config·사전계획·runner·exporter추가.
Batch1 stride 차이로~9.54e-7 출력오차를 발견해 공식배치와동일화; 허용오차완화없음.
기존 split에서train16/dev8 recording·192window만캐시, 과거노출2recording제외.
공통초기화/batch순서·K4/future4·200update/3seed, learned-noaux대조와의존도진단을기록한다.
GPU0단일작업/새환경설치없음/공용원본·WA·Drive전체평가·기존pilot보존.
새cached/explicit-ID/split/stride검사포함 전체107CPUtests통과.

**865be79 이후 최신전환**: WA 우리7worker만정상중단, CPUhelper종료/16raw해시확인/9253scene부분JSON·CSV와snapshot보존.
Drive 원본planner의module을재사용하는K4 learnable patch selection/경량futurepredictor/zero-init residual memory 구현.
train-only2recording/1diagnosticstep/원본출력보존/gradient·target경계/비용검사 config·script·CPUtests 추가.
Timestamp의sub-ms jitter를exact0.5s로오인한loader검사를수정하고실제offset을기록; 첫실패로그보존.
원본Drive source/weights/환경·WA자산/공용원본/타인process는변경하지않았다.
최종전체CPU103tests/Ruff/gitdiffcheck통과, 실제official2window연결gate통과/결과JSON·재현명령·한계문서공유.
첫contract진단은officialnavtrain만확인하고내부split필터를누락하여held_out1/development1recording을사용했다.
이전weights는재사용하지않고rejectreport/exposureaudit보존, 기존projecttrain만선택하는필터/회귀검사후새초기화로재검사했다.
원본splitmanifest는수정하지않았고노출된heldout group은향후이extension의미사용독립평가로주장하면안된다.

아래는865be79까지의변경이력이다.

후속GPU0증설: 새v2 GPU별상한5/2, CPUqueue 기존4worker adopt후교체, GPUworker중단0/추가3개확인.
Moduleprofiling script/2CPUtests/공유JSON/보고서 추가. Dense timing완료후packing진단namespace오류가있었고
unwrap으로수정, 완료dense측정은보존·재사용했다. Packed-all최종trajectory bitwise동일/현재profiler종료.
최종 관련CPU tests 24/24통과, 수정/신규scripts·tests의Ruff검사통과. 실행source hash와lint후source hash는별도기록.
현재같은공식설정의full평가는계속중이며원본/Git-public설정/타인process/데이터변경없음.

**5ed8a60 이후 공유GPU재개**: 사용자요청으로각GPU2개/총4worker 상한, 14-waypartition은그대로.
새bounded CPUqueue/메모리입장검사/6GiB reserve profile과7개CPU tests를추가했다.
Launcher는특정기존shard만선택기동; manifest atomic write. Guard는free memory/GPU mapping 검증,
압력marker를남기고우리PID만SIGINT한다. 실패/guard-stop worker 자동재시도없음.
8686완료scene의SHA/config hash확인, pause archive, guard/queue/aggregate복원,4worker 기동확인.
README/AGENTS/HANDOFF/research_status/RESUME/report 갱신. 기존config/model/scorer/seed/12step 불변.

**846b98d 이후 사용자GPU반환요청**: 우리14worker/CPUguard/aggregate만정상종료. GPU0·1각25MiB확인.
8686완료/3460남음/실패·중복0,16JSONL SHA256검사와2.89MB별도snapshot백업완료.
Pause marker/명시적resume gate/CPU보존검사/3pause tests추가. 단순launcher와directfull이기동차단됨확인.
HANDOFF/research_status/RESUME/report와공유상태JSON갱신. 원본데이터/타인process/기존결과변경없음.

아래는 이전커밋까지의완료이력이다.

공식WA strict1162keys/6scene smoke성공. Nativeall-ID6scene×12step bitwise동일, separatepatch보존.
Fixed/randomfuture2048vs8192:72timedtrial완료,전체inference6.69s→약2.07s(약69%감소).
6scene소수PDMS민감도이지논문재현/선택학습성능이아니다. ReservedVRAM은allocatorcarryover로불변.
사용자요청에따라14worker/45decimalGBcap/owned-PIDmemoryguard/재개·자동집계 구현.
전용tmux14worker실행확인/현재각GPU약39.88GB/OOM0. Full평가아직미완료,과거32scene재사용.
명시적사용자요청으로562완료scene보존 후우리14worker만정상중단/재개, `kjs-wa-jepa-eval` 실행별칭적용.
Conda prefix/weights/14-waypartition/seed/scorer불변. 기록 `outputs/official_wa_jepa_reproduction/kjs_process_label_pause.json`.

WA공식source/preset/checkpoint12 vs4 sampling불일치 확인, 결과조회전12-step404d8af고정.
독립Conda/worktree/공식NAVSIMv1 준비, 공개planning/encoder weights 확보, preflight/strictloading/
원본agent·scorer 평가harness와config 추가. 이번은준비/검사이며학습결과가아니다.

**공식 결과 보존 후 기반 결정 (기준578be6e)**:

- DrivePB model/refiner/scorer/targets/agent loss/Lightning training/PB eval를직접재감사;futureheadtrain-only/planner입력아님.
- WA jointinference/teacher/flowloss/positioner/gradient flags/strictloader/evalpreset를읽고single-forward scene_out예외기록.
- 공식HF파일목록만조회,PB v1/v2·WA공개weight의name/size/publishedhash공유. 새weightdownload/fullload없음.
- Score join전에current speed/command bins/minimumsample기록. 저장scene12146전체현황/metadata/CSV/JSON추가.
- CPU16.29초/GPU0/noinference,meta누락0/136native recording,originalscoreCSVhash불변. H1/tuning으로해석금지.
- WA native spatial-tube 기반추천과fixedbudget controlled five-condition 계산그래프작성. 미구현/범위결정대기.
- New7context + existing5official aggregation tests/Ruff통과;기존공식평가자산/referenceclone/pilot환경그대로보존.
- HANDOFF/README/AGENTS/researchstatus갱신,작업기록/근거/미확인구분. Pilot/확대/residual/새selector/동적K/추가평가없음.

## 4. 다음 단계 — 기반 추천 검토 후 (최신 사용자 지시가 아래 과거 계획에 우선)

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
