# HANDOFF — 이 파일 하나로 다음 에이전트가 이어받는다

마지막 갱신: 2026-10-04 21:47 KST (Codex)

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

- 사용자중간점검요청으로진행/로그/gradient/메모리/체크포인트/sourcehash를읽기검토했다. 실행중source/config/queue/모델은수정하지않음.
- 독립read-only `check_lpwm_stage2_training_health.py`추가,update1,536checkpoint를hardlink로고정해optimizer/실제parameter변경검증. 기존epoch-monitor hash첫128개dev를예측전등록해초기planner와paired평가. GPU0진단4GiB상한·전체44GB중단선·10분제한내287.9초완료,전체최대36.489GB. 본학습계속확인.
- 공유JSON에protocol/weightaudit/초기·학습후예측/pairedsummary/로그통계/진단후상태저장. 실제training curvePNG생성·시각확인. 기존연구문서/HANDOFF/RESUME에중간개선과작은개발집합·미학습planner비교의한계기록. 성능결과로학습량·checkpoint선택변경없음.

## 4. 다음 단계 — 기반 추천 검토 후 (최신 사용자 지시가 아래 과거 계획에 우선)

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
