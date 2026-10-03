# Planning-Aware Future Prediction — 에이전트 작업 규칙

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
