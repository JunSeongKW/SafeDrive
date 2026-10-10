# 연구 상태 — Codex / ChatGPT 공통 인수인계

## 2026-10-10 18:13 KST — 5 epoch 공통 NAVTEST 비교 완료

동일1,024장면/44recording·공식NAVSIMv1·모두5epoch·실패0: LPWM82.3243/DrivoR83.9662/Drive-JEPA85.3751. LPWM4→5+.7633점(CI−.3996~+1.9229), 직진증가/좌·우감소. 학습scene/노출량·입력·SSL·seed·scheduler차이가남아표현단독효과·공정한논문우위입증아님. 원25epoch학습/조건불변,epoch6계속. 전체NAVTEST평가는아니다. [전체비교](lpwm_planning_experiment.md) · [결과](../results/lpwm_front_history_stage1_lora_v1/epoch5_common_comparison_20261010.json).

## 2026-10-10 17:59 KST — 5 epoch 학습·내부dev 검증 완료

5epoch 5,885update 학습17:51·저장17:51:16·자동내부검증17:53완료. 내부dev1,021 PDMS 83.9882(동일dev4epoch 81.8300 대비+2.1582), ADE/FDE 1.7463/4.2489m. 공통NAVTEST1,024 또는전체NAVTEST점수아님.

현재6epoch,원25epochqueue/조건불변. 5epoch공통NAVTEST비교는미평가. [결과](../results/lpwm_front_history_stage1_lora_v1/epoch5_completion_status_20261010_1759.json).

## 2026-10-10 14:07 KST — LPWM 4 epoch NAVTEST 부분집합 PDMS 81.5610

공통 독립1,024장면·44주행 기록·공식NAVSIMv1·실패0. LPWM3epoch80.2828→4epoch81.5610(+1.2782); 기존 대조군은 DrivoR3epoch81.0005/JEPA3epoch81.9835이다. 대조군4epoch checkpoint가 보존되지 않았으므로 동일4epoch 비교로 부르지 않는다. LPWM의 ADE/FDE는1.6974/4.1122→2.1684/5.3802m로 악화했으며 입력·학습 예산도 서로 다르다. 학습code/config/286sources와25epoch 대기열은 불변, epoch5 계속. 전체NAVTEST·원논문재현·표현의단독효과 입증이 아니다.

[상황별 점수·입력·학습량·한계](lpwm_planning_experiment.md) · [전체 결과](../results/lpwm_front_history_stage1_lora_v1/epoch4_common_comparison_20261010.json)

## 2026-10-10 13:47 KST — 이전 축소 LPWM 저성능 진단 완료

공통 3 epoch 개발셋 PDMS: DrivoR78.40/JEPA74.70/LPWM순차70.09/공동63.97. 별도 고정192장면에서 후보 최고97.8–98.6 대비 LPWM 선택67.79/63.36으로 선택 손실이 큰 것을 확인했다. 현재·미래·background·geometry/context 연결과 planning gradient는 정상 전달된다. 두 CPU 표본의 가중 SSL gradient는 planning의28–32%이며 항상 충돌하거나 SSL이 지배한다는 근거는 없다. Frozen-native adapter 대조는 동일 이전2단계 조건으로 구성하고 CPU 연결·native고정 검사를 완료했으며 본학습·대기열 기동은 하지 않았다. 현재25epoch 학습의286개 source는 불변이다.

[전체 진단과 한계](lpwm_planning_experiment.md) · [결과 JSON](../results/small_corpus_planning_diagnosis_20261010/diagnosis.json)

## 2026-10-06: 세 경로 planning LoRA 본학습

사용자가 선택한 geometry·appearance·future LoRA 조건을 시작했다.
Linear229/Conv44, LPWM native·buffer 고정, 공식 DrivoR planner 전체 학습.
실제273adapter gradient와DDP batch16/유효64/40.16GB 검사 통과.
[설계·증거·검증 범위](lpwm_drivor_planning_path_lora_training.md).
최종 planning 개선은 아직 미확인. 아래는 이전 이력이다.

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
[전체 결과와 한계](encoder_future_learning_results.md), [등록 설계](encoder_future_learning.md)

**사용자용 이미지:** 실제 전방 영상·원본/재학습 궤적·장면별 오차·미래 감독의 추가 효과를 3장으로 정리했다.
[이미지 설명과 PDF](encoder_future_learning_results.md#실험-결과-이미지). 새 학습이나 GPU 추론은 하지 않았다.


## 완료 — 2026-10-03 SPARTAN/C-JEPA/IA-JEPA 착안 통제 비교

등록 commit `5b85a01`: 9조건 × 3 seed × 800 update를 완료했다.
GPU1에서 56.39분, 최대 allocated 1.719GiB. CPU 162개 검사 통과.
같은 train 512 / dev 192 window(개발 182 scene / 24 recording), K8·4초, frozen Drive-JEPA를 유지했다.

| 조건 | 개발 ADE(m) | 개발 PDM(%) |
|---|---:|---:|
| 원본 | 0.352210 | 87.119134 |
| 기존 global learned | 0.347129 | 88.826031 |
| 새 sparse | 0.349238 | 87.742199 |
| + C-JEPA 마스킹 | 0.349230 | 87.742244 |
| + IA-JEPA 움직임 선택 | 0.348680 | 87.691769 |
| 현재 특징만 사용하는 대조군 | 0.349207 | 87.742499 |

각 기법의 추가 효과와 learned 대 random의 대응 CI는 0을 포함한다.
원본 대비 일부 ADE 감소가 관측되지만 current-only도 동일하고, PDM 개선 CI는 0을 포함한다.
**추가 planning 이득·미래 예측의 필요성·선택 가설은 아직 입증하지 못했다.**
새 방법으로 교체하거나 추가 sweep를 하지 않는다.

공식 scorer로 개발 34조건 × 192 = 6,528개 score를 완료했다. Navtest 전체 재평가는 하지 않았다.
우리 GPU 작업은 모두 종료됐다. [범위·표·CI·상황별·해석](drive_jepa_region_research.md),
[공유 수치](../results/drive_jepa_region_research_v1/summary.json).
후속 제안은 현재 특징 전달과 미래 변화 정보의 기여 분리이며, 아직 실행하지 않았다.
원본 모델·공용 데이터·WA pause를 보존했다.

## 최신: 선택 위치 학습·파이프라인 진단 완료 (2026-10-03)

결론: **위치는 학습으로 바뀌지만, planning에 중요한 요소로 이동하는 학습은 아직 입증되지 않았다.**
사용자 후속 지시에 따라 추가 개수·크기 탐색은 하지 않았다. 확인 시 등록된 15run은 이미 완료돼
종료할 프로세스가 없었다. 현재 GPU 학습·진단 작업은 모두 종료됐다.

### 고정 개수·크기에서 확인한 것

동일한 8개 영역/각 2×2 native patch(입력 영상상 32×32 pixel), 3개 seed, 각 800update다.
영역은 객체가 아니며 좌표를 연속 이동시키는 모델이 아니라 격자 후보의 선택 순위를 학습한다.

| 동일 예산 조건 | 학습 전후 선택 집합 교체율 | Dev scene-macro XY ADE, m (평균 ± seed SD) |
|---|---:|---:|
| 무작위 (현재 window별 고정) | 해당 없음 | 0.347460 ± 0.000108 |
| Planning loss로 선택 | 26.0–37.2% | 0.347129 ± 0.000866 |
| 현재 planner 유지 보조 목표 추가 | 74.0–89.1% | 0.346951 ± 0.000446 |

마지막 조건은 선택된 현재 특징만으로 원본 planner의 출력을 보존하는 proxy를 추가했다.
이 proxy의 오차는 0.824→0.457m로 낮아졌지만, planning ADE 차이는 확인하지 못했다.
마지막 조건−planning 조건의 대응 recording-bootstrap 95% CI는 [-0.000943,+0.000625]m,
무작위 대비 [-0.001206,+0.000172]m다. 둘 다 0을 포함한다. 반복 개발 데이터이며 확증 검정이 아니다.
동일한 6개 장면의 그림에서 화면 아래쪽을 벗어나기도 하지만, 중앙 먼 영역으로 집중하면서
가까운 보행자·차량을 놓치는 사례도 보인다. 의미적 중요 대상의 GT 검증으로 해석하지 않는다.

### 실제 코드·checkpoint에서 분리한 원인

Train의 서로 다른 16 recording을 결과와 무관하게 hash로 고정했다. 위 두 학습 방식×3seed,
가중치 업데이트 없이 768회 hard 위치 교체와 score gradient·미래 특징 교란을 검사했다.

1. **Gradient 단절은 관측하지 않았다.** 최종 모델의 planning→선택 점수 gradient norm은
   0.000395–0.000792다. 점수 instrumentation/명시 ID 경로 모두 원래 출력과 max 차이 0이었다.
   CPU에서는 instrumentation 전후 parameter gradient도 bitwise 동일했다.
   Hard 교체 손실 변화와 score 기반 근사의 부호 일치는 81.6–93.7%다.
   이는 국소적인 근사 점검이지 ST가 전역 최적 선택을 보장한다는 뜻은 아니다.
2. **위치-미래 특징 대응을 활용하는 정도가 약하다.** 보조 목표 모델에서 미래 branch를 끄면
   궤적이 평균 5.6–7.0cm 바뀌지만, 좌표는 유지한 채 미래 특징을 선택 slot 사이에서 섞으면
   0.24–0.27mm만 바뀐다. 이때 feature RMS 변화는 1.12–1.14로 입력 교란 자체는 작지 않다.
   현재 구조는 미래 특징들을 cross-attention으로 전체 current memory에 전달한다.
   위치 대응보다 전체 내용에 의존할 가능성을 지지하지만, 단독으로 구조적 원인을 확정하지는 않는다.
   Attention normalized entropy는 0.82–0.86이므로 '완전히 균등 평균한다'는 해석도 하지 않는다.
3. **새 보조 목표의 방향이 planning과 어긋난다.** 보조 목표 모델에서 weighted proxy의 score
   gradient norm은 planning의 약 4.3–10.5배이며 두 gradient의 평균 cosine은 -0.067~+0.021이다.
   중앙으로 선택을 움직이게 할 수 있어도 미래 정보의 planning 효용을 가르치는 목표는 아니었다.
4. **표현·평가의 한계가 남는다.** Frozen ViT token은 이미 전역 맥락을 담고, predictor도 전체
   current grid를 본다. 그림의 사각형을 '그 안의 물체만 참고한다'고 해석할 수 없다.
   고정 영상 좌표의 미래 tubelet을 예측하며 객체 추적/ego-motion 정렬 기반 중요도는 아니다.

따라서 개수·크기나 proxy 가중치를 더 올리는 것은 다음 우선순위가 아니다. 다음 구현 후보는
**K·크기를 고정하고, 선택된 미래 특징을 대응하는 current spatial memory 위치에 연결하는 방식**이다.
현재의 전역 fusion과 동일 조건으로 비교해 '어디의 미래인가'에 대한 실제 유용성이 증가하는지 먼저
확인해야 한다. 이 변경/추가 학습은 아직 실행하지 않았다. 잘못된 선택을 가려주는 위치 규칙도 추가하지 않았다.

### 보존·재현·검수

- 기준 `122e885` → 학습 등록/실행 코드 `8e8e49a` → 결과/진단 등록 `9dcf80c`.
- 학습: 15run/12,000update, 2,008.58초, peak allocated 1.665GiB, GPU1만 사용, OOM 없음.
- 읽기 전용 진단: 62.91초/1.260GiB, optimizer update 0, 공식 모델 및 각 extension hash 불변.
- CPU 전체 154 tests 및 새 코드 Ruff 통과. 공용 원본/기존 결과/공식 baseline 보존.
- [결과 JSON](../results/drive_jepa_selective_future/spatial_region_selection_v1_20261003/summary.json),
  [진단 JSON](../results/drive_jepa_selective_future/spatial_region_selection_v1_20261003/location_diagnosis/summary.json).
- 시각화: `outputs/drive_jepa_selective_future/spatial_region_selection_visualization_20261003/index.html`.
  현재 실제 사진/선택 위치이며 미래 생성 영상이 아니다. 이전과 동일한 6개 recording, seed29 고정.
- 코드: `scripts/train_drive_jepa_spatial_regions.py`, `scripts/diagnose_drive_jepa_location_learning.py`,
  `scripts/report_drive_jepa_spatial_regions.py`. 각 config와 새 output 경로를 명시해 재현한다.
- 진단 첫 시도는 official import의 cwd 변경에 따른 상대 출력 경로 오류로 측정 전에 중단됐다.
  원본 실패 디렉토리를 보존하고 절대 경로로 v1b에서 완료했다. Raw source_commit이 official
  reference commit을 가리킨 metadata 문제는 원본을 바꾸지 않고
  [provenance clarification](../results/drive_jepa_selective_future/spatial_region_selection_v1_20261003/provenance_clarification.json)에 정정했다.
- GitHub push는 기존 VSCode credential socket 오류로 막혀 있다. 로컬 커밋과 산출물은 보존한다.

## 이전 등록 및 진행 이력 — 선택 영역 확대·planning 보조 신호

사용자가선택시각화를확인한뒤재학습요청. 기준122e885, 새설정 `spatial_region_selection_v1.json`.
K16 native / K4 2×2지역 / K8지역 random·planning·retention의5조건×3seed×800update.
같은704cache, 공통100warmup과기존800batch순서·낮은LR고정. 원본planner/encoder frozen.
Region평균 미래latent를예측하며planner현재입력은512개그대로. 미래mask는aux에만사용.
Retention은선택current정보로원본planner판단을보존하는training-only SmoothL1 proxy이며selector만갱신.
이는contextual features/mean masking의모델의존도측정이지인과적중요도나미래효용의정답이아니다.
동일K8 controls가주비교, K/면적증가조건은자원변경을분리해서해석한다. CPU전체151검사통과.
등록코드commit8e8e49a, GPU1단일학습PID2624673(실제명령확인필수), 시작11:41경.
당시 첫3run(K16native)800update완료. 이후15run전체완료/원본출력동일성과gradient경계통과.
2시간/condition20분/own8GiB/sharedreserve6GiB; 등록상한뒤추가자동튜닝없음.

설계근거: 선택하는면적0.78%와초기zero-bridge의간접planning신호만으로의미있는대상선택을보장할수없다.
Native4→16은개수,4patch→4region은영역풀링,8region 3조건은동일예산선택신호를비교한다.
Teacher행동을보존하는선택학습의참고는 [DynamicViT §3.3의 teacher/distillation losses](https://papers.nips.cc/paper_files/paper/2021/file/747d3443e319a22747fbb873e8b2f9f2-Paper.pdf).
논문의분류token sparsification/attention masking을재현하는것은아니다. 이실험은원본현재경로를그대로두고,
training-only mean-replacement readout으로selector에SmoothL1XY목표를추가한다. 일반적설계참고이지novelty주장아님.
Raw `outputs/drive_jepa_selective_future/spatial_region_selection_v1_20261003/`, 로그는동명`.log`.
원격push는기존VSCodecredential socket거절/anonymous write오류로실패; 로컬commit은보존했다.

## 최신 — 87run 완료 및 실제 selector/predictor 시각화

자동 후속18run까지03:45종료, commit8182f6c. 낮은LR에서도 learned-vs-fixed/random의
recording CI는0포함한다. Current-feature 전달도0.346890m이며 미래 예측의추가효용은미확정.
새사용자요청에따라 기존 seed29 MLP/ego-query checkpoint만 CPU로복원해 시각화했다.
Dev192/24recording 전체의 selected IDs는 저장GPU결과와 같고 predictorMSE차이는최대9.54e-7.
Before=100auxwarmup후, After=고정800jointupdate. 같은최종선택위치로predictor를비교한다.
Ego-query는선택43.1%교체/MSE2.3792→2.2716(copy2.3964), MLP26.6%교체/4.3746→2.8763(copy2.1647).
이는가중치업데이트와예측학습의관측이며 좋은선택/PDMS개선의증명은아니다.
실제사진은정답참조, 별도latent heatmap은비공간1024channel값이며RGB생성이아니다.
갤러리경로: `outputs/drive_jepa_selective_future/selector_predictor_visualization_20261003/index.html`.
새GPU/학습/환경설치/공용데이터변경없음. 전체CPU144tests통과. 아래실행상태는과거이력이다.

## 최신 상태 — 69run 완료, matched low-rate 선택 비교 등록

첫69run/46,800update 종료. 추가dev192window/24recording에서 원본ADE0.352210m,
낮은LR MLP0.347694±0.000392, ego0.346338±0.001650m. 이전LR보다는개선되나
원본대비 cluster CI는0포함하며, 반복개발/다중비교/공식PDMS미평가 한계를유지한다.
같은높은LR에서는 random0.351212<learnedMLP0.361504이므로선택학습성공으로부르지않는다.
실제모델6개/patch교체6144건진단완료, 기준출력차이0, 원본hash보존, optimizerupdate0.
MLP W-gradient부호일치86–92%, ego74–80%는 full selector정책학습이나일반화보장이아니다.
V1 batch/grad-mode혼합검사 실패도보존했고 V2는tol을늘리지않고실행조건을맞췄다.
다음별도18run은같은낮은LR의fixed/random, jointaux-off, currentfeature대조다.
기존learned6run은재사용, 새cache/target/architecture/heldout평가 없음. 사용자밤샘승인·09:00상한 유지.
상세와재현경로: [밤샘실험 보고서](drive_jepa_overnight_causal_followup.md).
기존69run로컬결과commit313a3ab; GitHub push는기존인증오류로실패했다.

## 현재 작업 2026년 10월 3일 원인 분리 재학습

사용자가 다음 오전09:00KST까지 질문 없이 실험·문헌조사를 이어가도록 승인했다.
기준8159aad, 사전계획/코드커밋8f5b18d. [밤샘 비교 계획](drive_jepa_overnight_causal_followup.md).
읽기전용 기존checkpoint진단과 7조건×대응3seed×400update를 완료했다.
현재결과를 단순 과적합으로 단정하지 않고 fusion강도·gradient결합·selector변화·목적함수를 분리한다.
동일192window와auxwarmup을재사용한다. Parameter-onlyfreeze는planning→selector를유지하고
planning→predictorparameter만차단하며, ForeDrive의future-outputdetach그대로가아니다.
CPU121tests통과. 기본격리환경의CUDA차단으로학습전실패했고 가중치/optimizer업데이트는없었다.
호스트GPU1실행으로전환했으며다른사용자process/공용데이터는건드리지않았다.
첫21run/8400update완료: MLP .232000/ego .239397/aux-only .236099/gradientprojection .239264/
halfbridge .233743/frozenS .234913/uniformADE .242933m, 원본 .220644m보다평균이높다.
1096.94초/원본hash불변. Gradient충돌이나목적함수변경하나만의문제로확정하지않는다.
9b28ebc에서추가navtrain cache704표본과27run×800update를사전등록했다.
Train128⊂512, dev192/24group은기존mini/extension개발recording과분리;heldout불사용.
704개cache는301.14초에 완료했다(신규6.04GB, 이전128파일 재사용). GPU1단일queue에서
동일K fixed/random/learned와현재feature대조를 포함한27run도 완료했다.
추가dev192window/24recording: 원본ADE0.352210, MLPlearned0.361504,
fixed0.355226, random0.351212m. Random 대비 학습형 선택의 우월성은 확인되지 않았다.
Random-original 차이의 cluster CI는0포함하며, 다른 dev의 이전0.220644와 직접비교하지 않는다.
4355bf2에 future projection 이전9run과 보수적 LR/메모리 정규화12run을 추가 등록했다.
두 번째 queue는 앞 queue 완료를 기다리며 GPU 작업을 중복 실행하지 않는다.
전체 등록69run/46,800jointupdate, 09:00KST 및 메모리/실패 상한이 우선이다.
전체CPU130tests통과. Push는VSCode credential socket오류로실패/로컬커밋보존.
기존 '새학습금지/실행없음' 문장은아래과거완료시점기록이다.

## 최신 완료 상태 — 2026-10-03 구조별 추가 학습

공식 Drive-JEPA planner 위에서 같은192window의 5조건×3seed 학습을 완료했다.
사전 실행 커밋은 `d3bbced`, 상세는 [구조 비교 결과](drive_jepa_architecture_followup.md)다.
기존 선택 비교와 이번 추가 학습을 구분한다. 원본 planner·가중치·평가 결과는 변하지 않았다.

기존 비교의 learned dev ADE0.242582m에서, 새 공통 절차의 MLP 대조는0.209975m였다.
원본0.220644m보다 낮지만 recording-cluster CI가0을 포함하며 학습 절차와 초기화도 달라졌다.
새 contextual residual0.223808m, ego-query0.216938m, LoRA0.216459m였다.
미래 latent 정확도와 planning 유용성은 별개였고, 더 복잡한 구조의 우월성을 확인하지 못했다.
LoRA는 미래 branch의 복제된 last4 QKV에만 적용했고 원본 planner 입력은 유지했다.
고정 target encoder를 썼으며 EMA/공식 JEPA 사전학습 전체 재현으로 부르지 않는다.

Planning과 aux gradient가 LoRA로 전달되고 aux가 selector/bridge로 가지 않음을 실측했다.
학습 후 실제영상에서 branch-off 원본 bitwise 보존과 엄격 checkpoint 복원을 검증했다.
최초 참조와 후속 호출의9.5367e-7 차이도 보존했고, 동결·warmup을 맞춘 검사에서 통과했다.
원본 hash 불변, OOM0, 최대PyTorch allocated2.674GiB,15run/4500update 약21분40초.
GPU1 학습·검증은 종료했다. 다른 연구원·공용데이터·기존환경을 변경하지 않았다.

공유: [summary JSON](../results/drive_jepa_selective_future/architecture_followup_v1_20261003/summary.json).
추가학습/공식benchmark/held-out/WA/pilot 자동재개는 없다. 다음은 원래 선택 질문에 대한
fixed/random/learned 통제 비교 여부를 검토할 차례다. GitHub push는 기존 인증 오류로 미완료다.

## 이전 연결 검증 — 공식 Drive-JEPA planner의 선택적 미래 extension

2026-10-02 사용자전환승인. WA는9253/12146성공/76.18%에서중단/보존했고자동재개하지않는다.
PartialPDMS91.102506 vs동일9253scene Drive89.019762; 원본Drive 전체89.224320은그대로다.
새설계 [선택적patch 미래 연결](drive_jepa_selective_future_connection.md): current front512patch→learnable K4
→4futuretubelet latent→zero-init residual over original128image memories→원본Transformer/trajectoryhead.
원본planner/encoder freeze, target은동일fullcheckpoint frozenencoder의future camera-grid feature다.
미래GT는별도auxloss에만공급, policyauxgradient는차단/online current-only. ST는최적선택보장이아니다.
연결진단만승인됐으며새성능결과/학습형선택성공/novelty확정은아니다. 자체pilot·확대·WA학습은보류한다.
실제official fullcheckpoint strict load/2navtrain recording에서연결gate통과: off/init-on bitwise동일,
bridge1step후planning→selector/predictor, aux→predictoronly, detachforward동일/해당backward차단.
원본weights hash불변/원본planner modules재사용/신규1271489params. CPU전체103tests통과.
최종project-train진단45.176s/peakallocated1.261GiB/현재GPU프로세스없음. [실측JSON](../results/drive_jepa_selective_future/connection_v1_20261002.json).
단일diagnosticstep은bridge출력projection만갱신했고selector/predictor의유용한학습은아직미실행이다.
이전v1b진단의내부split필터누락(heldout1/development1노출)은[별도audit](../results/drive_jepa_selective_future/project_split_exposure_audit_20261002.json)로공개했다.
그weights/결과는최종진단에재사용하지않았다. 기존split은보존하지만노출recording을독립holdout으로주장하면안된다.

## 보존된 실행 이력 — `fd5fc5f` 이후 WA-JEPA 공식 재현

**이전상태(2026-10-02 21:03KST): 사용자요청으로 GPU0:5/GPU1:2/총7worker.**
20:40각2개로재개후GPU0만증설. CPUqueue교체시기존GPUworker를adopt/중단0; profile v2.
기존14-waypartition을bounded queue로순차처리하고model/12step/seed/scorer/config는불변.
8686/12146(71.513%)완료·3460남음을출발점으로재사용하며학습/추가benchmark는없다.
16개scene파일SHA256검증/별도압축snapshot/이전pause를보존했다. 12GiB입장/6GiB reserve와우리PID만guard.
Pressure-stop은자동retry없음. [재개기록](../results/official_wa_jepa_reproduction/shared_gpu_resume_state.json).
[GPU0증설](../results/official_wa_jepa_reproduction/gpu0_worker_increase.json),
[모듈별시간](wa_jepa_inference_module_timing.md): shareddenseencoder0.545s/predictor24.105s/model24.698s,
predictor97.6%; isolated6scene predictor6.413s/agent6.694s(95.8%). Predictor가주병목이며학습/설정변경없음.
[중단상태](../results/official_wa_jepa_reproduction/paused_evaluation_state.json),
[백업metadata](../results/official_wa_jepa_reproduction/paused_snapshot_backup.json).

**실측진행(2026-10-02)**: officialstrict1162keys누락/shape0,6scene원본smoke성공.
Nativecanonicalall-ID6scene×12step bitwise동일(PE/predictor/denoising/trajectory).
Untrained fixed/randomfuture8192→2048,QKV/FFN 실제생략,72timedtrials6.69s→약2.07s(약69%latency절감).
소수PDMS는민감도자료일뿐최종성능/선택학습증거아님. Reserved메모리절감은입증하지않았다.
공식dense전체는GPU0·1각7총14worker/각~39.88decimalGB<45GB/OOM0으로실행하다중단했다.
Full12146scene결과와논문Table3대조는**미완료**. 中間같은5187scene비교는42.7%시점기술통계이며
최종결과나미래예측의통제된인과효과가아니다. 기존Drive전체결과보존.

Camera별 spatial patch-tube를 잠정 단위로 승인받았다. 객체 instance/최종 기여는 아니다.
공식 원본 agent/scorer→dense 전체 평가→all-ID 동등성→fixed/random 소수 scene 비용 검사 순서다.
학습형 selector/학습/pilot 재개는 금지. 진입점 [WA-JEPA 실행 보고](official_wa_jepa_reproduction.md).
Latest bec2966은4-step, 직전 공식404d8af/published state.pt/논문은12-step이다.
결과를 보기 전404d8af 원본12-step preset으로 고정했다. 기존Drive 재현값은 변경하지 않는다.

## 최신 우선순위 — `578be6e` 이후 기반 결정과 첫 통제 실험 설계

완료한공식PFViT-L/navtest12146전부성공/PDMS89.224320을변경없이보존했다. +0.224320 원인은미확정.
이번에는Drive-JEPA PB/WA-JEPA의source를읽고공식공개파일metadata만조회했다.
PB미래객체head는train-only이고planner에전달되지않는다. WA는jointscene/trajectory추론과
trajectoryloss→scenehidden경로가있지만single-forward의finalscene_out에는직접planninggradient가없다.
WA공식train loader의future-derived command fallback도찾아향후선택실험에서는금지하도록명세했다.

**추천: WA-JEPA native spatial patch-tube 선택**. Current4-view encoder/planning자산을보존하고
prediction전futurequery를pack한다. 객체instance선택과동일한주장은아니다. 범위승인과
strictweight/config/실제fullgradient·비용검증후에만진행한다. Selector/model수정은미구현.
학습·새환경·대용량download·추가평가·pilot자동재개없음.

저장sceneCSV와currentmetadata만CPU16.29초분석:12146scene/136recording,metadata누락0,
속도5구간/command3유효범주/교차구간. OriginalCSV SHA불변. 상황별점수차이는H1증거가아니다.
새7contexttests+기존5집계tests통과. 사실/문헌/설계/미확인과loss별gradient경계를구분해기록했다.

최신진입점: [기반·계산그래프·최소비교군](future_prediction_foundation_decision.md),
[현재상황현황표](official_navtest_current_context_summary.md), `results/foundation_selection/`.
공식소스Drive548bb82/WA bec2966을그대로읽었으며공식reference clone변경없음.

## 보존된 완료 이력 — 공식 Drive-JEPA 재현

사용자 지시로pilot개발·데이터확대·추가학습은보류. WA-JEPA도다음순위다.
`5c6e6d5`를기준으로전용Conda/공식source별도worktree/PFViT-Lfullplanning checkpoint를준비했다.
공식v1 navtest12146token의로그·현재front·공식metric cache completeness통과, 3scene strict/scorer smoke완료.
GPU0·1에겹침없는6075/6071scene execution shard로전체추론·평가완료:12,146scene 전부valid/finite,
실패·누락·중복0. PDMS89.224320 vs 논문89.0(+0.224320점), 전체wall약741초/총worker4.
실제Hydra의모델·scorer·입력·split일치확인, 원본scene별CSV와비용·telemetry·5개결과검사공유.
GPU0·1해제, 자체pilot/WA-JEPA자동재개없음. 허용오차/원논문실행별상세provenance가없어정확재현성공단정금지.
진입점: [공식재현보고](official_drive_jepa_reproduction.md), `configs/official_drive_jepa/reproduction_v1.json`.
종료점인논문Table2 PF89.0과전체공식지표대조까지완료했다. Pilot·WA를자동재개하지않는다.
이번PF forward는별도future predictor 없이encoder→공식trajectory decoder이므로선택연구기반확정으로부르지않는다.

## 최신 결정 — `607da52` 이후

현재 pilot은 연결·gradient 검증 자산으로 보존하고 predictor 튜닝·확대 학습은 보류한다.
물리/train-fitted ridge 재현과 하나의 visual-only residual 대응3seed 비교를 완료했지만,
pilot은 ridge보다 약하고 residual의 planning 이득을 확인하지 못했다.
이는 연구 가설 실패가 아니라 기반 적합성 판단이다.

다음은 미래 예측과 planning 연결이 있는 공개 모델의 공식 재현→동일 예산의 상황별 선택→
현재 관측/ego 의도의 학습형 선택→동일 평균 예산 배분이다. WA-JEPA를 첫 재현 후보로 추천한다.
공식 source/공개 checkpoint 메타데이터/작은 attention backward만 확인했으며 full 모델 재현은 미수행이다.
공간·시간 token과 객체 instance의 주장 범위를 구분한다.

실측/한계: [pilot 기반 판단](pilot_foundation_decision_results.md).
다음 진입점: [공개 기반 감사와 선택·예산 계획](public_future_planning_foundation_audit.md).
공유 JSON: `results/pilot_foundation_decision/`. 등록된 확대 계획은 자동 재개하지 않는다.
아래는 이 결정 이전의 보존된 진행 이력이다.

갱신: 2026-10-01. 협업 출발점은 `95015df`다. 현재 **CPU graph + NAVSIM GT-state 진단 +
공식 frozen encoder를 이용한 실제 front-video/GT ROI visual·spatial target 연결 검사 완료** 상태다.
`09913b5` 이후에는 **명시적 영상 보정/투영 규약,373window 약616MB frozen feature cache,
현재 거리 K4 fixed-rule runner와 A–E 각200update(seed29,batch8) 비교 완료**로 진행했다.
최신 실측·한계는 [target 감독 비교 결과](target_supervision_exploration_results.md)다.
`9353acf` 이후 최신은 [저분산 진단·제한1000update](future_prediction_variance_followup_results.md)다.
기존cache/200학습 재사용, 추가7400update(A추가2seed/seed29 A–E1000/E–F 대응29·11) 완료.
C의분산비.012→.214/visual MSE.951→.794지만persistence.395를전horizon에서못넘었다.
E/F ADE차이는.021/.051m로작고미래정보의인과기여를증명하지않는다.
구checkpoint는RNG없어optimizer-state continuation이지정확한resume아님. 새checkpoint RNG/optimizer를검사했다.
JPEG 보정의실필요성은여전히조건부이며cache재생성·residual/확률적/selector추가는없다.
이는 미래 감독의 학습 가능성/초기 경향이며 selector 학습이나 최종 target·novelty 결정이 아니다.
공식 NAVSIM baseline 재현·성능·순수 visual JEPA·배포 가능한 perception 구현 완료 보고는 아니다.
동적인 상태는 이 파일과 `HANDOFF.md`, 계산 그래프는 `selective_entity_future_prediction_graph.md`,
시간순 이력은 `RESUME_NOTES.md`에서 관리한다.

## 1. 최신 사용자 의도와 범위

연구 질문: **현재 주행 맥락과 ego 의도를 이용하여, 같은 예측 예산에서 planning에 유용한
객체의 미래를 선택적으로 예측하도록 학습할 수 있는가?** JEPA 채택 자체는 기여가 아니다.

- H1: 상황에 따라 유리한 미래 정보 구성이 다를 수 있다. 아직 일반적인 사실로 확립되지 않았다.
- H2: context/planning-conditioned 선택이 동일 예산의 강한 random/규칙/학습 비교군보다
  좋은 planning을 만들 수 있다. 검증할 주 가설이다.
- H3: 정보량 K 또는 horizon의 맥락 적응. 동적 구현은 아직 하지 않는다. Target 결정 후
  고정 K별 맥락적 추가 예산 이득부터 측정하며 최종 기여는 선행연구 검토와 실험 후 결정한다.
- 개발 범위: 고정 K, 고정 horizon, 차량·보행자 instance의 **미래 예측 대상 선택**.
  EgoFSD 중복 때문에 이것 자체를 최종 novelty로 전제하지 않는다.
  정차 차량도 후보에서 배제하지 않는다. 도로·신호 등은 공통 맥락으로 유지하는 후보안이다.
- 관측 마스킹, 미래 예측 대상 선택, 예측 후 planner 입력 선택을 구분한다.
- 사용자 승인 GPU는 **0, 1**뿐이다. 승인됐다고 기존 타인 프로세스를 중지하거나
  카드가 비어 있다고 가정하지 않는다.

최신 상세 사용자 프롬프트가 과거 HANDOFF의 강한 주장보다 우선한다.
연구 질문의 변경, 큰 학습·cache 생성은 대안과 비용을 먼저 보고한다.

## 2. 실제 조사 및 변경 현황

### 95015df까지 실행한 조사

1. SafeDrive 지침, git 상태, HANDOFF, 기존 메모리·분석 자산 및 데이터 경로를 읽었다.
2. SafeDrive의 instance 선택, joint world/planning decoder, loss 및 미래 target 정렬 코드를 추적했다.
3. Drive-JEPA 공식 저장소를 작업공간에 shallow clone하고 아래 commit을 고정해 기록했다.
   perception-free model/agent/feature builder 및 encoder loader를 읽었다.
   perception-based 최상위 모델은 읽었지만 내부 scorer/BEV 모듈 감사는 미완료다.
4. Drive-JEPA / AD-E2E-JEPA 원문에서 pretraining과 downstream 경로를 구분해 확인했다.
   다른 출발 논문의 제한적 원문 조사는 있었으나, 전체 novelty 비교표와 코드 검증은 미완료다.
5. 호스트의 GPU 종류와 기존 프로세스를 읽기 전용으로 확인했다.
6. 이 상태 문서와 계산 그래프 **초안**을 작성하고, 과거 인수인계의 오해 소지를 정정했다.

### 이번 v1에서 추가 실행한 것

- 사용자/ChatGPT 검토를 반영해 순차 조건부 ST의 K-slot, invalid, temperature, ID tie,
  permutation 및 모든 객체가 예산에 들어가는 경우를 구현했다.
- CPU fixture에 selector/predictor/planner와 loss별 gradient 경계를 구현했다.
  기존 SafeDrive/Drive-JEPA model에는 아직 붙이지 않았다.
- **13개 unittest 통과**: planning→selector, auxiliary 차단, detach의 forward 불변과
  backward 차단, target 교체 독립성, permutation, padding/empty/invalid/tie 등을 검사했다.
- **3-seed 합성 선택 학습 통과**: 정해진 analytic predictor/planner를 두고 selector만 planning
  MSE로 학습했다. 별도 holdout에서 두 relevant entity의 exact-set 정확도는 99.9756~100%다.
- Command 교란 / no-intent entity-only 학습 / random / motion / fixed-semantic / relevance reference를
  비교했다. 현재 key·intent만으로 relevance를 계산할 수 있으므로 hindsight/배포 불가 해석은 정정했다.
  쉬운 인위적 과제이므로 자율주행 H1/H2 증거로 사용하지 않는다.
- 결과·환경·config·source SHA256은 `results/synthetic_diagnostics/future_prediction_graph_v1_before_readability_refactor_20261001.json`,
  해석·한계·재현 명령은 `docs/synthetic_validation_results.md`에 남겼다.

### ChatGPT fe8c930 검토 이후 추가한 것

- 합성 `input_exact_match`와 `relevance_oracle`의 동등성 검사. 추가 합성 학습/튜닝 없음.
  No-intent key는 이미 `entity_only_selection_without_intent`로 바뀐 상태를 유지한다.
- Drive-JEPA v1 perception-based backbone/refiner/scorer/target/loss를 함수 경계까지 감사했다.
  agent future-state 출력은 train-only supervision이며 planner/score 입력으로 사용하지 않는다.
- [baseline 및 target adapter 감사](baseline_and_target_adapter_audit.md)에 source commit·함수·shape·
  gradient·특권 입력·target association·비용/미확인을 남겼다.
- 권고 scaffold는 Drive-JEPA frozen front-video encoder+단순 planner+새 visual ROI adapter다.
  SafeDrive 주 baseline/retraining은 여전히 잠정 중단이고 공식 baseline 재현과 scaffold를 구분한다.
- 실제 mini 한 log의 두 구간으로 GT-state adapter+waypoint8 graph의 gradient/공동 backward/
  1회 optimizer update를 검사했다. 시각 encoder는 실행하지 않았고 image header/무결성만 확인했다.
  이 두 구간은 같은 scene/log로 독립 성능 표본이 아니다.
- 실제 데이터 대조에서 ego yaw convention 차이를 발견해 공식 pyquaternion 규약으로 수정하고
  roll/pitch가 0이 아닌 회귀 검사를 추가했다. 공용 log/image hash는 전후 같다.

### ChatGPT 35fbdcf 검토 이후 추가한 것

- 공식 encoder weight 한 개(5.13GB), pinned revision/size/SHA256 및292tensor strict loading 확인.
  `FrozenDrivingVideoEncoder`는 eval/no_grad, current와 future에 같은 frozen teacher 사용.
- 새 venv overlay에 timm1.0.30만 설치했다. 기존 torch환경은 읽기 전용 참조하며 upgrade하지 않았다.
- GT track/box front ROI adapter와 visual+explicit spatial 미래 predictor, 새 작은 trajectory decoder 구현.
  ROI appearance만으로 위치·motion 보존을 가정하지 않고 별도 current-ego state target6dim을 쓴다.
  이는 혼합 감독 pilot이지 pure visual JEPA나 공식 Drive-JEPA 전체 모델이 아니다.
- GPU0에서 실제 mini window0의 현재·미래 clip9개를 실행, 13front 후보에서K4 선택·future8 예측.
  Planning→S/P/D, visual/spatial auxiliary→P only, future detach·no-branch 계약 및1회 update 통과.
- 33/33 unittest(기존28+visual5), Ruff 통과. 추가 synthetic training·GT-state test 확대 없음.
- ROI contact sheet를 직접 확인했다. 대체로 정합하나 occlusion/rectification 미확인, front 제한 유지.
  현재/미래 appearance cosine과 명시적 spatial motion을 설명 통계로만 기록했다.
- 미래 감독 없음 / current-target / no-future-branch / all-entity 참조를 별도 대조로 설계했다.
  실제 수행은 경계 진단이며 공동 학습 비교나 planning 성능 평가는 아직 하지 않았다.
- 공유 log1개/image10개 hash 전후 동일. 새 원본 dataset 다운로드·공유 dataset 쓰기 없음.
  상세: [영상 pilot 검증](visual_future_prediction_pilot_validation.md),
  `results/visual_diagnostics/visual_future_pilot_verified_20261001.json`.

### 아직 하지 않은 것 (현재 기준)

- Official planner weight/evaluator/benchmark score 재현, detector/tracker 기반 inference.
- 충분히 수렴한 여러 seed 공동 학습, 최종 미래 활용·동일-K 공식 성능·효율 평가, novelty 표 완성.
  여러 장면의 작은 fixed-rule200-update 학습/의존도/분산 진단은 아래 최신 실행에서 완료했다.
- 최종 visual-only vs mixed target 및 baseline/독립 holdout 확정.
- 본 학습·전체 cache 재생성·SafeDrive 재학습. 공용 dataset 수정 및 기존 환경 upgrade.

### ChatGPT 1231767 검토 이후 — 최신 조사/결정

- 연결 검사 추가 확대 대신 [연구 질문·target 결정](research_question_and_target_decision.md)을 작성했다.
  Fixed K는 개발 기반이고 현재 visual+explicit spatial은 객체를 선택하지 정보 종류를 선택하지 않는다.
  Official encoder만 재사용했고 planner는 신규 scaffold라는 구분을 유지한다.
- [EgoFSD/ForeDrive 근거 감사](egofsd_foredrive_evidence_audit.md)를 원문 최신 version과 대조했다.
  EgoFSD의 intention/attention 기반 선택 및 joint motion/planning, ForeDrive의 latent future conditioning과
  직접 겹친다. EgoFSD official pinned tree에는 README/assets만 있고 selection autograd는 미확인이다.
  ForeDrive Eq.(7)은 planning gradient가predictor로 가지 않는 경로를 명시하며, 공개 구현은 확인 못했다.
- [여러-log CPU 조사](navsim_visual_target_coverage.md):16recording/128window, current front count>K4는
  45.3%, +4s visual65.4%/spatial93.6% 잔존. Left future-heading proxy11window는visual36.4%, right0,
  merge semantic label 미확인. Projection-valid를 occlusion visibility로 부르지 않는다.
- Mini64segment를52원본recording proxy로 묶고16group을 sampling, train12/dev4로 고정했다.
  이전 smoke recording은dev only. Nonoverlap train277/dev96window manifest, native log token도split 간 불겹침.
  Coverage나future 유효성으로 현재 candidate/window를 선별하지 않았다.
- 4개front/side/future projection contact sheet를 직접 확인했다. Occlusion은 남으며
  공식load/projection 경로에undistort가 없다고 stored JPEG rectification이 확인된 것은 아니다.
- [최소target 학습 계획](minimal_target_ablation_plan.md): 고정현재거리/K4로 A branch없음,
  B planning만, C visual aux, D spatial aux, E mixed aux. B–E는같은두head/branch로감독만바꾼다.
  C/D/E 공통 visual∩spatial mask, train-only normalization,200update/조건 계획을 실행 전 명시했다.
- 실제 수행은CPU metadata/projection 조사뿐. 새training runner/cache/normalization/성능metric 구현 및
  GPU실행/공동 학습은 하지 않았다. 공용log16/reviewimage16 hash 전후동일. 기존model source와결과는보존했다.

### ChatGPT 09913b5 검토 이후 — 최신 실제 비교 학습

- 원본DB와camera K/D/image size 대조, 직진/회전4표본 ROI 직접 검토, crop/resize pixel-center 규약 검사.
  JPEG 가공 이력은 별도 원본 byte 비교로 확정 못했으므로 original-distorted 취급 가정을 남겼다.
  저장K/D rectification을 메모리에서 수행한 뒤 pinhole ROI를 사용한다. Occlusion/시간 차이는 남는다.
- Train277/dev96window만 frozen official encoder로cache,616,266,713bytes. Future teacher clip/target/mask는
  training 전용 dict로 분리하고 현재 candidate/window를 future validity로 거르지 않는다.
- 고정현재거리 K4, 동일 초기화/batch/normalization, A branch없음/B planning만/C visual/D spatial/E mixed,
  각200update. B–E는두head·폭동일, aux→planner gradient0. A활성762,627/B–E2,223,283parameter.
- Train-only common target4,785관측에서whitening; 같은27,659slot/time supervision availability sequence 확인.
  Dev ADE(scene-macro m)는A5.921/B5.881/C5.921/D5.749/E5.872. 아직모든곡선하락중/단일seed/4devrecording.
- C예측visual분산/target비율0.012, swap ΔADE0.0006m로평균회귀/branch무시경고. Encoder는freeze되어
  encoder collapse가아니다. Evisual분산0.120, 모든forecast MSE는current persistence보다아직높다.
  초기순위로target을탈락시키거나최종planning개선·novelty를주장하지않았다.
- 38/38tests/Ruff통과, log16/image3,730pre/posthash동일. GPU0만사용하고종료/메모리해제,
  GPU1기존프로세스보존. Shared원본쓰기/envupgrade/추가download/SafeDrive재학습/동적K구현없음.
- 실제공유JSON: `results/target_supervision_exploration/seed29_updates200_20261001.json`.
  Raw cache/last checkpoint/구간별지표/곡선은`outputs/`에있다. 다음은C/E경고와학습량을검토하고
  충분한동일조건학습/다른seed·current-target 대조를결정하는것이며 자동추가실행은없다.

95015df는 문서·작업 규칙 변경이었다. 이번 v1은 독립 synthetic fixture와 결과 기록을 추가한다.
기존 모델, loss, 실험 결과 CSV는 변경하지 않았다.

## 3. 실제 경로·버전·자산

| 항목 | 조사 결과 / 범위 |
|---|---|
| 현재 연구 checkout (SafeDrive 이력 보존) | `/rhome/junseong/PlanningAwareFuturePrediction`, branch `junseong/main` |
| SafeDrive 조사 기준 | `6ed73948d272218c5fd34c3eddf2dc0d26510393` |
| Drive-JEPA 소스 | `/rhome/junseong/PlanningAwareFuturePrediction/reference_repositories/Drive-JEPA` |
| 공식 소스 기준 | `548bb8215e3aae18e162a0f12f1ba83b4d3eb57e`, [공식 저장소](https://github.com/linhanwang/Drive-JEPA/tree/548bb8215e3aae18e162a0f12f1ba83b4d3eb57e) |
| 현재 호스트 | `user-ESC8000A-E11`; 이전 AXE-080 명칭과 구분 |
| GPU 0 / 1 | RTX A6000, 각 약48GB. 점유는 변동하며 이번 실행 직전0번25MiB/compute 없음,1번기존 프로세스 보존 |
| 공용 원본 | `/home/user/data/Dataset/` 전체 **절대 직접 수정하지 않음** |
| NAVSIM 읽기 링크 | `PlanningAwareFuturePrediction/dataset -> /home/user/data/Dataset/navsim -> /mnt/nfs/data/open_dataset/navsim` |
| 코드·변환·cache·결과 | `/rhome/junseong/` 아래 사용자 작업공간 |
| 신규 원본 다운로드 | `/home/user/data/processed_dataset/`, 사용자 지정 총 1 TB 한도 |
| 평가 CSV | `exp/safedrive/**/traj_*.csv` 22개. 존재만 확인; run별 의미와 학습 provenance 재감사 필요 |
| O0 last.ckpt | `exp/safedrive/o0_nofuture/lightning_logs/checkpoints/last.ckpt`, 1,382,312,902 bytes |
| F3 last.ckpt | `exp/safedrive/f3_nopairnc/lightning_logs/checkpoints/last.ckpt`, 1,382,302,278 bytes |
| 분석 축 | `analysis/context_axes.csv`; 생성 코드의 일부 축은 미래 reference 기반 |
| 없는 자산 | 현재 작업공간의 `ckpts/`, `exp/metric_cache_navtest`, `exp/safedrive_train_cache` |
| 환경 | 기존 base / alpasim-cuda128 보존; 공식 full-stack safedrive / drive-jepa 환경 미구성 |
| 기존 torch | torch 2.8.0+cu128 읽기 전용 상속; CPU와 작은 visual pilot만 실행 확인 |
| CPU fixture 실행 환경 | `runtime/environments/future_prediction_cpu`, Python 3.12.13 / 기존 torch 읽기 전용 참조 venv. 완전 독립 dependency 환경 아님 |
| 영상 pilot 환경 | `runtime/environments/visual_future_prediction_pilot`, 별도 venv overlay; timm1.0.30 설치 |
| Official encoder weight | `runtime/checkpoints/drive_jepa/vitl_merge_3dataset_e50.pt`; hash/strict load 검증 |
| Visual 결과 | `results/visual_diagnostics/visual_future_pilot_verified_20261001.json`; local ROI 그림 `outputs/visual_pilot/` |
| 최신 coverage/split | `results/data_surveys/navsim_visual_target_coverage_summary_20261001.json`, `navsim_recording_split_manifest_20261001.json` |
| 제한 탐색 계획 | `configs/exploration/target_ablation_plan.json`; 계획이며 trainer/cache/학습 미실행 |
| EgoFSD official source 조사 | `reference_repositories/EgoFSD`, `23fec8aba3e828ef228939e30e3020240d8b0cae`; README/assets only |

체크포인트 크기는 기존 AICA 전송 기록과 일치한다. 체크섬·내용·resume 적합성은 검증하지 않았다.
공식 Drive-JEPA 전체 cache/weight 다운로드 명령을 무작정 실행하지 않는다.

## 4. 지금까지 확인한 중요한 코드 사실

### SafeDrive

`navsim/agents/safedrive/safedrive_model.py`:

- `_forward_swnet`(859행), `select_topk`(1266행)는 각 ego proposal 경로와 현재 객체 위치의
  최소 거리를 사용한다. 1301행의 `topk(...).indices`로 선택하고 query를 gather한다.
- 선택된 **feature 내용**으로는 gradient가 흐를 수 있지만, 정수 index 선택 자체에는
  선택 정책을 학습시키는 미분 경로가 없다. 현재 선택기는 학습된 importance selector가 아니다.
- world/planning decoder가 agent와 ego query를 함께 갱신한다. 이 경로를 곧바로
  별도 JEPA 미래 latent predictor → planner 구조라고 부르지 않는다.
- `_proposal_object_detection`의 training denoising 경로는 `targets`를 사용한다(603~611행).
  신규 graph 검사에서는 미래 GT의 forward 경로를 분리하고, 우선 DN을 끄는 안을 검토한다.
  이것만으로 기존 코드에 미래 정보 누출이 입증됐다고 주장하지 않는다.

`safedrive_features.py` 284~322행과 `_align_future_agent_states`(689행)는 미래 객체를
현재 ego frame과 현재 track-token 순서로 정렬한다. 활용 가능한 target 생성 참고 코드다.
검출 query ↔ GT token 대응 및 entity latent target 생성은 추가 설계가 필요하다.

`safedrive_loss.py`의 `prediction_loss_weight`와 `pair_Disp_loss_weight`는 별도 경로다.
F2의 과거 결과를 “미래 agent 정보 완전 제거”로 해석하려면 당시 설정·코드를 복원해야 한다.

### Drive-JEPA — 현재 감사한 분기만

공식 commit의 `navsim_v1/navsim/agents/drive_jepa_perception_free/`:

- `drive_jepa_features.py::compute_features`와 `DriveJEPAFeatureDIBuilder`는 현재 또는
  과거+현재 이미지와 driving command / ego velocity / acceleration을 만든다.
- `drive_jepa_model.py::DriveJEPAModel.forward`(87행)는 이미지 encoder → pooling/projection
  → status 결합 → transformer → ego waypoint head를 호출한다.
- `vjepa2/evals/image_classification_frozen/modelcustom/vit_encoder.py::init_module`(39행)은
  checkpoint에서 **encoder**를 로드한다.
- `drive_jepa_agent.py::compute_loss`(128행)는 ego trajectory의 length-normalized L1이다.

이 perception-free downstream 경로에는 별도 entity 미래 latent predictor를 호출하는 코드가 없다.
따라서 “Drive-JEPA를 그대로 사용하면 selector→future predictor→planner가 이미 연결된다”는
가정은 성립하지 않는다. 이후 v1 perception-based 내부도 감사한 근거는 별도
[baseline 감사](baseline_and_target_adapter_audit.md)에 있다. 미래 state head의 train-only 출력과
현재 ego proposal query를 안정적 entity 미래 latent와 혼동하지 않는다.

## 5. 이전 해석을 정정 / 유보한 것

- **마스크 token이 학습됨**과 **대상을 고르는 selector가 학습됨**은 다르다.
- AD-E2E-JEPA v1은 2026-09-28 공개다. “3주 전” 표기는 정정한다.
  [원문 §3.4](https://arxiv.org/html/2609.34085v1)는 downstream IL에 patch predictor를
  제거하는 경로를 설명한다. goal-conditioned zero-shot 경로와 이 IL 경로를 섞지 않는다.
- 모든 기존 JEPA 마스크가 입력과 무관하다는 일반화, 두 비교 열이 모두 “아니오”이면 novelty가
  확보된다는 결론은 삭제한다. 직접 선행연구 비교가 더 필요하다.
- `context_axes.py`는 `mc.trajectory`로 `fwd/lat/dheading/bow`를 계산한다. 이는 미래 reference
  기반 분석 축이지, 그대로 online selector 입력으로 사용 가능한 맥락 정보가 아니다.
  `mc.observation.unique_objects`로 얻은 주변 객체 수도 현재 시점 정보인지 추가 감사해야 한다.
- SafeDrive ablation은 미래 감독·용량·객체 수·클래스가 섞여 있다. 모두를 동일한 “미래 정보
  선택” 실험으로 묶지 않는다. DAC 상승만으로 과적합 원인을 확정하지 않는다.
- 다른 split에서 학습한 context 선택은 **cross-fitted 선택 규칙**이지 보장된 oracle upper bound가 아니다.
- navtest는 반복 개발에 사용된 상태다. 최종 독립 평가라고 부르지 않는다.
  navhard 접근 가능성·사용 이력·프로토콜은 별도로 확인해야 한다.

## 6. 다음 담당자가 바로 할 일

1. 최신 [기반추천·통제실험명세](future_prediction_foundation_decision.md)를먼저읽는다.
   과거pilot target비교/학습계획은완료또는보류이력이며다시실행하지않는다.
2. 준성이WA native spatial patch-tube범위축소를승인할지, 객체instance를유지할지결정한다.
3. 승인후에만WA전용Conda/worktree·정확source/config/checkpoint를pin하고strictfull loading/소수train-dev
   officialforward/gradient·VRAM/latency를확인한다. 현재는모두미실행이며공식점수를우리결과로복사하지않는다.
4. Prediction전packing과all-ID원본동등성을확인한뒤고정K/horizon matched통제학습의상한을등록한다.
   Learnedselector는hardindex만으로학습되지않으며planning→position ST경로와auxiliarypolicydetach를검증한다.
5. Original/random/fixed-rule/planning-conditioned/all-future참조를동일split/core초기화/pairedseed로설계한다.
   Native recording단위train/dev/독립holdout;navtest fitting/tuning금지. Navhard/scorer/cache는versiongate후별도계획.
6. Pilot·데이터확대·residual·동적K/horizon은계속보류. GPU0·1도이번조사에서는쓰지않았다.

## 7. Codex ↔ ChatGPT 협업 규약

- Git의 `junseong/main`이 공통 기준이다. ChatGPT에는 **commit URL과 이 두 문서**를 전달한다.
  push만으로 다른 대화의 모델이 자동으로 문서를 읽거나 메모리를 공유하지는 않는다.
- 확인된 사실 / 설계 제안 / 실행 결과 / 미확인 항목을 분리하고 근거의 commit·경로·함수를 남긴다.
- ChatGPT의 방법 제안은 아직 구현·검증된 결과가 아니다. Codex가 코드와 실제 결과로 대조한 뒤
  상태를 갱신한다. 의견이 충돌하면 가정·근거·영향을 기록한다.
- 작업 종료 시 상태 문서와 HANDOFF를 갱신하고 RESUME_NOTES에는 실제 실행한 일만 덧붙인다.
  기존 실험 기록은 지우거나 새 결과처럼 재해석하지 않는다.
- 대용량 데이터·checkpoint·비밀키·토큰·환경 credential은 커밋하지 않는다.
- 재현 기준: source commit, config, split, seed, 실행 명령, 환경, 측정 결과를 함께 남긴다.

이 문서를 공유할 때의 첫 요청 예:

> 이 commit의 research_status.md와 future_prediction_foundation_decision.md를 읽어 주세요.
> 코드 사실과 미구현 설계를 구분하여 WA spatial-tube 범위 변경, position ST gradient 경로,
> original/full-reference 통제와 실제 연산 절감 명세를 검토해 주세요.

## 8. 가독성 refactor와 현재 진입점

작업 루트는 `/rhome/junseong/PlanningAwareFuturePrediction`다. 이전 SafeDrive 이름의 checkout을 재명명했다.
현재 package는 `src/planning_aware_future_prediction/`, 검사는 `tests/`에 있다.
SafeDrive 원본 코드/CSV/checkpoint는 변경하지 않고 중단된 참고 자산으로만 유지한다.
명명 규칙과 이전 경로 대응은 `docs/naming_conventions.md`와 `docs/directory_migration.md`를 읽는다.
원격은 협업 이력 보존을 위해 기존 SafeDrive 저장소를 유지한다. 원격 이름은 baseline 결정이 아니다.
