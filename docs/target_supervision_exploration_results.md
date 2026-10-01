# 고정 K4 미래 감독 비교 — 200-update 실행 결과

2026-10-01, 출발 commit `09913b5`. **A–E 각 200 update 완료**, seed29, batch8,
GPU0. 이번 질문은 ‘누구를 선택하는가’가 아니라 **선택된 객체에 어떤 미래 감독을 주면
planning 학습에 도움이 되는가**다. 최종 target/novelty/H1·H2는 확정하지 않았다.
실행 원본·hash·곡선·상황별 지표: [공유 결과 JSON](../results/target_supervision_exploration/seed29_updates200_20261001.json).

**9353acf 이후 검토 정정**: 아래는 당시200update의 기록이며 수치는 보존한다.
낮은 예측 분산과 작은 swap Δ만으로 branch 무시/collapse를 확정하지 않는다.
JPEG 왕복 오차는 저장 영상에 보정이 필요한지의 증거가 아니다.
입력 변화량/mean/persistence/variance-axis·1,000update·E/F 후속 결과는
[최신 저분산 진단](future_prediction_variance_followup_results.md)을 읽는다.

## 1. 투영·cache에서 확정한 것과 남은 가정

- 네 개 직진/회전 train/dev 표본에서 NAVSIM의 K/D/1920×1080이 원본 nuPlan DB와 일치함을 확인했다.
  DB는 SQLite read-only/immutable로 열었다. CameraIntrinsic list만 해독하고 legacy devkit을 설치하지 않았다.
- 운영 규약: **저장 JPEG에 저장 K/D로 in-memory rectification, 새 K는 같은 K → 상하28px crop
  → cv2 linear512×256 → pinhole ROI → stride16/aligned ROIAlign**. 현재·미래에 같은 규약을 쓴다.
  Resize pixel center는 `(x+.5)*512/1920-.5`, `(y-28+.5)*256/1024-.5`다.
  이전 edge 좌표와 최대0.375px 차이를 검사했다. 과거 pilot의 기본 전처리는 그대로 보존했다.
- 네 표본의 원본/보정 ROI sheet를 직접 확인했다. Calibrated FOV point의 distort→undistort
  수치 왕복 오차는 최대1.14e-12px다. **이미지-3D 정합 정확도 또는 visibility 측정값은 아니다.**
- OpenScene은 nuPlan sensor 직접 연결을 안내하고, nuPlan 원문 camera schema는 Caltech D 순서를
  정의한다. nuPlan 영상을 쓰는 MTGS도 distortion/pose timing의 문제와 undistortion을 보고한다.
  [OpenScene 데이터 규약](https://github.com/OpenDriveLab/OpenScene/blob/main/docs/getting_started.md),
  [nuPlan schema](https://github.com/motional/nuplan-devkit/blob/master/docs/nuplan_schema.md),
  [MTGS Appendix A.1](https://arxiv.org/html/2503.12552v2#A1),
  [OpenCV 변환 규약](https://docs.opencv.org/4.13.0/d9/d0c/group__calib3d.html).
- **별도 원본 sensor JPEG가 없어 로컬 export 가공 이력은 byte 비교로 확인하지 못했다.** 위 근거로
  저장 JPEG를 original-distorted로 취급하는 운영 가정을 명시했으며, 완전히 입증됐다고 하지 않는다.
  Projection-valid는 visibility가 아니다. 가려진 객체/background, sensor 시간 차이는 남는다.
  Raw distorted cuboid corner의 FOV 밖 polynomial 외삽은 큰 허위 ROI를 만들 수 있어 cache에 쓰지 않았다.

Manifest 그대로 train12recording/277window, dev4recording/96window만 cache했다.
Current clip은 `t-.5,t`, teacher target clip은 각 미래 시점에 끝나는 두 frame이며 training-only다.
Online input과 future target/mask를 별도 dict로 저장한다. Encoder는 frozen/eval이며 clip 사이 batch attention이 없다.
Current grid `[1024,16,32]`, current entity `[32,1034]`, current valid/ID `[32]`, ego8;
future visual `[32,8,1024]`, spatial `[32,8,6]`, 별도 mask와 ego trajectory `[8,3]`다.
Grid/ROI 및 current combined feature(geometry10 포함)는 FP16 저장 후 FP32 학습한다.
거리 규칙도 저장된 현재 좌표를 쓰므로 미세한 quantization이 있다. Spatial/ego target은 FP32다.

Cache373개 합계616,266,713bytes, 전체 directory 약617MB로2GiB cap 이하다.
Train8 profile58.65s(첫 kernel 시작43.95s 포함), 나머지365개188.50s, 총3,357clip을 처리했다.
Cache peak allocated1,479,913,472bytes/reserved1,648,361,472bytes.
전체 split의 현재front>K4는train132/277, dev54/96; current front0도train18/dev2를 그대로 남겼다.
공용log16/image3,730개의 사전·사후 SHA256이 모두 같다. 공용 원본 쓰기/새 원본 download 없음.

## 2. 동일 조건과 실제 학습

선택은 현재 거리+stable ID 동점 규칙 K4이며 learned scorer는 freeze하고 호출하지 않는다.
현재 관측은 유지, 미래8step/4s, 공식 encoder303,885,312parameter frozen,
신규 predictor/planner를 쓴다. **공식 Drive-JEPA 전체 baseline 재현/encoder 사전학습은 아니다.**
B–E 모두 visual1024/spatial6 두 head와 동일 memory adapter를 유지하며 aux 종류/계수만 바꾼다.
Predictor 출력은 train-only whitening 좌표이며 그대로 planner에 넣고, forecast 지표만 역변환한다.

초기 state hash, recording-balanced batch sequence, optimizer, clipping, cache를 조건 간 고정했다.
AdamW lr1e-4/weight decay1e-4/global clip1; 조건당1,600draw(반복 포함). Step0/50/100/150/200
동일 split 평가, **마지막200 checkpoint** 보관. Dev best checkpoint 선택이나 계수 변경 없음.

Normalization은 train의 selected common-valid4,785slot/time에서만 fit했고 std floor0.05에 닿는
채널은 없다. Common mask는 현재 선택 valid∩미래 visual-valid∩spatial-valid로 **loss에만** 적용한다.
모든 조건에서 같은27659유효slot/time와133 zero-common sample draw를 사용했다.
Dev의 common1961/native spatial2516, zero-common4window도 planning 평가에 포함했다.
A/B는 미래 aux를 학습하지 않지만 같은 availability를 진단용으로 기록했다.

첫 update에서 planning은B–E predictor/planner에 gradient를 주고, aux→planner gradient는 정확히0이다.
모든200update의 loss/gradient가 finite이고 P/D가 실제로 갱신됐다. Scorer 변화0, A의 predictor 변화0.
활성 parameter는A762,627 / B–E2,223,283이며 stored scaffold2,487,988과 구분한다.
E의0.05+0.05는 C/D의0.1과 총 계수만 같으며 gradient 크기를 맞춘 실험은 아니다.

## 3. 마지막 checkpoint의 초기 경향

ADE는 **scene-macro(m), 낮을수록 좋음**. Imitation waypoint proxy이며 PDMS/안전 성능이 아니다.

| 조건 | 감독 | Train planning loss | Dev planning loss | Train ADE | Dev ADE | Dev FDE |
|---|---|---:|---:|---:|---:|---:|
| A | Branch 없음 | 0.14899 | 0.17995 | 4.950 | 5.921 | 12.065 |
| B | Planning만 | 0.14782 | 0.17940 | 4.853 | 5.881 | 12.212 |
| C | Visual aux0.1 | 0.14951 | 0.18031 | 4.968 | 5.921 | 12.067 |
| D | Spatial aux0.1 | 0.14517 | 0.17615 | 4.753 | 5.749 | 12.035 |
| E | Mixed 각0.05 | 0.14730 | 0.17869 | 4.846 | 5.872 | 12.046 |

모든 dev ADE가 약9.25에서 내려왔고 **150→200에도 약0.74~0.88m 감소**했다.
아직 수렴한 비교가 아니다. D는 B보다0.132m 낮지만 단일 seed/작은 dev에서의 초기 관찰이지
최종 target 선택/공간 감독의 우월성 입증이 아니다. A–B는 활성 용량도 다르다.
Paired scene-macro/recording-cluster1,000resample은 JSON에 설명용으로만 남겼다.
Dev cluster4개/86scene여서 유의성·일반화 주장을 하지 않는다. Context 표본 부족 flag도 유지했다.

### 예측 품질과 branch 의존도 — 경고 포함

Common mask 기준 normalized MSE와 normalized prediction/target 채널 분산 비율.
Swap/제거 Δ는 **window mean ADE 증가(m)**로 위 scene-macro 지표와 집계가 다르다.

| 조건 | Visual MSE | Spatial MSE | Visual 분산 비율 | Future swap ΔADE | Branch 제거 ΔADE |
|---|---:|---:|---:|---:|---:|
| B | 1.737 | 1.445 | 0.743 | +0.443 | +0.468 |
| C | 0.951 | 0.760 | 0.012 | +0.0006 | +0.011 |
| D | 1.616 | 0.355 | 0.649 | +0.239 | +0.433 |
| E | 0.977 | 0.386 | 0.120 | +0.087 | +0.093 |

- **C는 평균 회귀/낮은 조건부 분산 및 branch 무시 경고**다. Horizon별 분산도 낮으며
  frozen encoder가 collapse했다는 뜻은 아니다. Aux MSE 감소만으로 좋은 미래 표현이라 하지 않는다.
- E의 visual 분산도 아직 낮다. 동시 공간 감독을 줬다고 이 위험이 해결됐다는 보장은 없다.
- Current ROI/state persistence의 MSE는 visual0.395/spatial0.153으로 **현재 모든 예측보다 낮다**.
  아직 미래 예측 학습이 충분하지 않거나 현재 상태/appearance가 강한 기준인 상황이다.
- B는 branch에 의존하지만 명시적 미래 예측 오차가 크다. 현재 정보 side-channel일 수 있다.
  Zero/swap/제거는 의존도 진단이며 재학습 A/B 대조를 대체하지 않는다.
  Swap donor는 다른 recording/scene의 **예측**이고 미래 GT가 아니다. Recipient current padding을 유지했다.
  Zero는 time/type token이 남고, branch 제거는 미래 memory 자체를 생략한다. Detach는 forward 교란이 아니다.

학습update 합계26.22s, 조건별4.05/5.71/4.86/5.35/6.25s. 전체 invocation250.21s에는
평가·진단·checkpoint/JSON IO가 포함된다. Cached-head latency이지 전체 영상 pipeline 지연은 아니다.
Training peak allocated는A172MB~E239MB; process VRAM sample 최대A550/B598/C608/D606/E626MiB
(일부 NVML sample unavailable). GPU0 메모리는 종료 후1MiB로 복귀, GPU1 기존 사용은 보존했다.

## 4. 검증·재현과 다음 gate

38/38 unittest(기존33+새 fixed-rule/cache 계약5), Ruff check/format, source/config hash와
pre/post 원본 hash 검사를 통과했다. 과거 synthetic/단일 pilot 결과와 SafeDrive 자산은 보존했다.
사용자 전원 중단 보고 후15:43 복구 점검에서도 cache373/checkpoint5 hash·strict loading·200update,
공유 결과/source hash와38tests를 재확인했다. 이미 완료된 학습은 다시 실행하지 않았다.
실행 전 sandbox GPU 접근/초기 CUDA memory 초기화 및 OpenCV5 API 오류를 수정하고 성공 실행을 남겼다.
실패 output은 지우지 않았다. 기존 env를 upgrade하거나 추가로 설치하지 않았다.

```bash
# 승인 GPU 점유 재확인; 각 output은 존재하지 않는 새 경로를 지정한다.
runtime/environments/visual_future_prediction_pilot/bin/python scripts/audit_front_camera_projection.py --output-directory outputs/projection_audit/<새경로>
runtime/environments/visual_future_prediction_pilot/bin/python scripts/cache_target_supervision_features.py --output-directory outputs/feature_caches/<새경로> --profile-only
# profile 검토 후 같은 cache를 --resume으로 373window까지만 완성
runtime/environments/visual_future_prediction_pilot/bin/python scripts/cache_target_supervision_features.py --output-directory outputs/feature_caches/<새경로> --resume
# configs/exploration/target_supervision_run_v1.json의 cache_directory와 실행 source를 고정한다.
runtime/environments/visual_future_prediction_pilot/bin/python scripts/train_target_supervision_ablation.py --output-directory outputs/target_supervision_exploration/<새경로>
```

Code: `scripts/cache_target_supervision_features.py`, `scripts/train_target_supervision_ablation.py`,
`scripts/summarize_target_supervision_run.py`, `models/fixed_distance_future_supervision.py`.
Raw cache/checkpoint/구간별기록/곡선은 Git 제외 `outputs/`에, 실제 공유 JSON은 `results/`에 있다.
원본 영상/ROI sheet를 Git 원격에 올리지 않았다.

**다음**: C/E 평균 회귀와 predictor 무시를 원인 후보로 검토하고, 이미 강한 persistence 및
current-target/current-feature 대조를 어떻게 분리할지 정한다. 아직 내려가는 동일 조건의 곡선을
충분한 학습·다른 seed로 확인하기 전 target을 탈락시키지 않는다. 자동 shortlist/추가 학습은 하지 않았다.
그 확인 뒤 같은 target/K에서 selector 비교로 돌아간다. Learned selector/동적K·horizon,
공식 평가/대규모 학습/전체 cache/SafeDrive 재학습은 이번에 시작하지 않았다.
