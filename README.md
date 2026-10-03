# Planning-Aware Future Prediction

**현재 진행: 전체 navtrain 가용 영상으로 공개 LPWM의 Stage 1 post-training.**
공식 encoder·context·dynamics·RGB decoder 전체 109.55M parameter와 temporal ELBO를 유지한다.
학습 23,126 clip/122 recording, development 7,745 clip/40 recording; 12프레임(4 observed+8 future).
20 epoch / 28,920 update. GPU0·1 DDP, GPU당 batch4×누적2, 유효batch16, FP32, 데이터 worker0.
실측 batch2 1.73s → batch4 1.61s/update; worker0·2·4·8은 1.61~1.63s로 추가 이득 미확인.
현재 `kjs-lpwm-stage1` 이름으로 본 학습 중; 추가 속도 실험을 위해 중단하지 않는다.
설정 `configs/lpwm_navsim_adaptation/full_posttraining_v2.json` + `execution/batch4_accumulation2_workers0.json`.
시각화 `outputs/lpwm_navsim_full_posttraining_v2/visualization/index.html` (같은 개발 장면 학습 전·중·후).
Stage1 적응 gate 통과 후 LPWM 낮은LR + planner 전체 학습의 통합Stage2, planning-only/영상목표유지 비교.
Stage2 train75,297/dev27,076; GPU 실행은 적응 gate 후. 현재 성능 개선이나 학습 완료를 주장하지 않는다.
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
