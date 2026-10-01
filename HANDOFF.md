# HANDOFF — 이 파일 하나로 다음 에이전트가 이어받는다

마지막 갱신: 2026-10-01 16:37 KST (Codex)

세션 시작: 이 파일 + `git log -10` + `AGENTS.md`.
세션 끝: 상태 문서 갱신 + `tools/handoff-commit.sh` + `git push mine`.
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
현재 작업 루트: `/rhome/junseong/PlanningAwareFuturePrediction/`.
명명 규칙: [docs/naming_conventions.md](docs/naming_conventions.md).
경로 이전: [docs/directory_migration.md](docs/directory_migration.md).

## 0. 현재 연구 의도 — 최신 사용자 프롬프트가 우선

**현재 주행 맥락·ego 의도·planning 목적에 따라 같은 예측 예산에서 유용한 객체의 미래를
선택하도록 학습할 수 있는가?** JEPA 채택 자체가 핵심 기여는 아니다.

- H1: 상황에 따라 유리한 미래 정보 구성이 다를 수 있다. 아직 일반적 사실로 확립되지 않았다.
- H2: 맥락/planning-conditioned 선택이 같은 예산의 강한 비교군을 넘는가.
- H3: 동적 K/horizon은 나중 확장이다. 현재는 **고정 K·고정 horizon의 개발 기반**.
  EgoFSD 중복 때문에 fixed-K 객체 선택을 최종 novelty로 전제하지 않는다.
  Target을 좁힌 뒤 고정 K별 추가 예측의 맥락적 이득부터 측정한다.
- 관측 마스킹 / 미래 target 선택 / planner 입력 선택을 구분한다. 주 초안은 미래 target 선택.
- SafeDrive는 기존 motivation 자산과 코드 참고다. 주 baseline으로 임의 회귀하지 않는다.
- Drive-JEPA는 encoder/재현 후보다. **직접 연결형 future predictor의 확정 baseline은 아니다.**
- 과거 “모든 기존 JEPA 마스크는 입력과 무관” / “두 비교 열이 아니오면 novelty 확보” 주장은 철회한다.

## 1. 실행 중인 작업

**실행 중인 학습 없음**. 기준 `9353acf`의cache373/seed29 A–E200은 반복하지 않았다.
C/E mean·persistence·분산축·교란 입력량과raw track/time/target373구간을 검사했다.
A seed11/47 각200 추가, 기존seed29 A–E를model/AdamW200에서1000까지 이어 학습,
F29와E11/F11을1000회 신규 학습했다. 추가7400update/전체235.33s로 등록 상한에서 종료했다.
F는planner 입력만detach하고 양head aux는 학습한다. 구checkpoint는RNG가 없어
optimizer-state continuation이지bitwise exact resume이라고 주장하지 않는다.
질문은미래 감독/gradient 결합이며 selector·동적예산·확률모델·SafeDrive 재학습은 없다.
Raw: `outputs/future_prediction_diagnostics/bounded_followup_1000_v1/`,
공유: `results/future_prediction_diagnostics/bounded_followup_v1.json`.
협업 출발 `95015df`, 현재루트/package/기존SafeDrive 자산은 유지했다.

현재 호스트는 `user-ESC8000A-E11`. 사용자 승인 GPU는 **0·1**이고 두 카드 모두
RTX A6000 약48GB다. 이번에는GPU0/1 각각15MiB/compute process 없음 확인 후GPU0만 사용했다.
종료 후GPU0/1은15MiB, GPU4–7 타인PID21618/33392/33393/33394는 전후 유지됐다.
다른 연구원 작업 중지/환경 변경 없음. 과거H100 설정은 적용하지 않는다.

이전 중단 작업: O0 epoch 1 / F3 epoch 0 checkpoint라는 인수인계가 있다.
실제 checkpoint 내부 epoch와 resume 적합성은 이번에 검증하지 않았다.
**O0·O1×4를 재개하지 않고, F3/F4도 현재 실행하지 않는다.**
큰 cache 재생성이나 SafeDrive 재학습은 새 연구 방향을 확인한 다음 별도 결정한다.

## 2. 최근 결과와 조사 사실

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

- 완료작업 재사용; C/E persistence/train-mean/horizon·분산축·swap 입력변화 진단 추가.
- Raw373구간의track/time/spatial/gather 일관성, normalization/loss 분모/currentROI 입력 검사.
- JPEG 왕복과실제보정필요성구분; export미확인가정유지/cache재생성없음.
- F planner-only detach: forward불변/planning→P차단/두aux→P보존. 관련8tests/Ruff통과.
- 추가7400update: A추가2seed/seed29 A–E1000까지continuation/대응seed29·11 E–F1000.
- C MSE.951→.794/분산비.012→.214, 여전히persistence.395보다나쁨; E/F차이작음/원인미확정.
- 구RNG snapshot없음명시; 새final10checkpoint strict/finite/optimizer/sampler/CPU RNG검증.
- iPad Table4 90.5→91.7은General→Proposal-centric prediction교체임을원문확인/선택효과로과장안함.
- 공유JSON/보고서/config/재현명령/status/HANDOFF/일지갱신. 원본/cache/구checkpoint/타인process보존.

## 4. 다음 단계 — 영상 persistence보다 나쁜 예측의 후속 결정

1. 최신저분산 보고서/JSON/F경계/continuation 제한을ChatGPT·Claude에검수공유한다.
2. 학습량관련근거는생겼지만 모든horizon persistence보다나쁨. Visual을버리거나계속학습하지말고
   다음하나의통제ablation을결정한다. 초기에작았던swap으로branch무시를확정하지않는다.
3. 후보는현재feature+delta residual만변경하는C대조(target/K/계수유지),
   별도config1000update/600s상한제안. 아직실행하지않았고효과주장전양쪽대응seed필요.
4. JPEG원본export근거/occlusion, multi-view후보중복제거/turn·interaction coverage를보완한다.
   Dev left6/right1/merge미확인으로일반적상황별K효과를주장하지않는다.
5. Target근거후random/강한규칙/ego-attention/ST와고정K추가예산을비교한다.
   확률적calibration/uncertainty planning은후속설계만; novelty/독립평가도별도gate다.

navtest는 개발·진단용이며 최종 독립 평가가 아니다. navhard 접근·사용 이력·공식 프로토콜을
확인한 뒤 최종 평가 경로를 정한다. 이전 서버의 데이터 크기·GPU-hour를 현재 실측치로 취급하지 않는다.
**현재 대규모 학습·SafeDrive 재학습·전체 cache 생성은 시작하지 않는다.**

## 5. 확정 범위 / 미결

**확정**: 미래 예측 대상/필요성·예산 배분이 연구 질문; 고정 K·horizon은 개발 기반으로 사용;
첫 target 비교는 selector를 고정하고 감독부터 분리; GPU 0·1만 사용;
공용 원본 직접 수정 금지; 기존 환경/프로세스 보존; 작업공간은 `/rhome/junseong`;
새 데이터셋 원본 다운로드만 `/home/user/data/processed_dataset/`에 총 1 TB 한도.

**미결**: 최종 visual/spatial/mixed target/baseline·공식 평가, ST 공동 학습 안정성,
multiview/occlusion/GT 대체association, 미래 활용·동일 예산 효과·novelty delta·독립 holdout.
이번1000update도A–E는단일seed(별도E/F만대응2seed)이고persistence를못넘었다.
C/E저분산은추가학습으로개선됐지만 capacity/조건부평균/regularization/실제미래활용의분리는남았다.
JPEG original-distorted 취급은 명시적 운영 가정으로 별도 원본 byte 증거는 없다.
Frozen visual teacher와 GT ROI는 구현됐지만 deployment perception/일반화는 검증하지 않았다.
현재 ST는 편향된 임시 추정이다. 작은 연결 검사 성공을 성능·효율로 일반화하지 않는다.
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
