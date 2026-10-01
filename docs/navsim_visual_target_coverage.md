# NAVSIM 여러-log 영상 target 유효율 — CPU 조사 결과

2026-10-01, 개발 기반 commit `1231767` 이후 실행. **Projection/target availability 조사**이며
encoder 실행·학습·planning 성능 평가가 아니다. 공용 원본에는 쓰지 않았다.

## 1. 표본과 split을 고정한 방법

- 설정: [data_survey.json](../configs/exploration/data_survey.json).
  Mini의64segment `.pkl`는 capture timestamp+vehicle 기준52recording group으로 묶인다.
  Segment 시작/끝 suffix만 다른 파일을 서로 다른 train/dev log로 취급하지 않는다.
- Filename hash와 고정 seed로16group, group당1segment를 선택했다. Train12/dev4group.
  이전 smoke에서 확인한 `2021.05.12.22.00.38_veh-35` 전체는 development에만 둔다.
  선택한 recording의 다른 segment alias도 같은 split에 귀속한다.
- **Coverage/모델 결과를 보기 전에** 로컬 `pre_survey_recording_split.json`을 저장했다.
  Window는 history4+future8의 연속12frame, 0.5±0.05s cadence, 동일scene/native-log를 요구한다.
  비중첩 window pool에서 시간상 균등 간격으로 group당최대24개를 pilot에 고정하고8개를 조사했다.
  미래 ROI 유효율은 window/현재 candidate 선정에 사용하지 않는다.
- 조사:128window(train96/dev32). 이후 탐색용 manifest:373window(train277/dev96).
  Recording group 및 읽은 native log token이 train/dev에서 겹치지 않는 것을 assertion으로 확인했다.
  최종 평가용 독립 holdout이나 공식 benchmark train/test split이라고 부르지 않는다.
- 공유 [split manifest](../results/data_surveys/navsim_recording_split_manifest_20261001.json)에
  전체alias, source hash, log token, eligible/pilot/survey start index를 보존했다.
  Mini 편의 표본으로 Las Vegas104/Pittsburgh24window이며 대표성은 보장하지 않는다.

## 2. 무엇을 ‘유효’로 셌나

현재 GT 차량·보행자 중40m 이내 최근접최대32개를 후보로 만든다. GT state/track은 privileged input이다.
현재 front `CAM_F0`로 투영한3D box가 기존 adapter의 cropped/resized 영상 ROI 규칙을 통과하면
`current_front_valid`다. **가려짐, segmentation, 실제 객체 pixel 존재는 보장하지 않는다.**

Visual future-valid는 현재 front-valid AND 동일track의 미래 state-valid AND 미래 front projection-valid
AND 미래2frame clip 파일 존재다. Spatial future-valid는 현재 front-valid AND 미래 state-valid다.
모든 horizon의 잔존율 분모는 현재 front-valid 객체621개다. 미래 이미지 전체 decoding/encoder
feature 유효성 검사는 하지 않았으며, 이 조사는 geometric/file availability 측정이다.
현재 두frame 이미지는 전체128window에서1920×1080 header를 확인했다.
현재 front/left/right 파일도 모두 존재했다. 신규 multi-view 모델은 구현하지 않았다.

## 3. 실제 수치

| 조사 범위 | Window/group | Front 유효 객체 수 합 | Front 수 중앙값 | Front 수 > K4 | +4s visual 잔존 | +4s spatial 잔존 |
|---|---:|---:|---:|---:|---:|---:|
| 전체 | 128/16 | 621 | 4 | 45.3% (58/128) | 65.4% (406/621) | 93.6% (581/621) |
| Train | 96/12 | 448 | 4 | 43.8% | 63.8% | 94.0% |
| Development | 32/4 | 173 | 4.5 | 50.0% | 69.4% | 92.5% |
| Left-turn heading proxy | 11/8 | 33 | 2 | 27.3% | 36.4% (12/33) | 93.9% |
| Low-turn heading proxy | 117/16 | 588 | 4 | 47.0% | 67.0% | 93.5% |
| Right-turn heading proxy | 0/0 | 0 | — | — | — | — |

Left/right는 **미래4s ego heading 변화 ±15°**로 만든 사후 분석 proxy다. Driving command의 의미,
교차로/합류/차선 변경 정답 label을 확인한 결과가 아니다. 이 미래-derived 값을 selector에 넣지 않는다.
Right-turn 표본은0개며 합류 semantic label도 이번에 확보하지 못했다. 해당 상황 coverage는 미확인이다.
Command는 해석하지 않은 index0/1/2/3에11/81/2/34window이며 의미를 추측해 이름 붙이지 않는다.

| 미래 offset (약 s) | Visual 유효/621 | Visual 잔존 | Spatial 유효/621 | Spatial 잔존 |
|---|---:|---:|---:|---:|
| 0.5 | 568 | 91.5% | 618 | 99.5% |
| 1.0 | 535 | 86.2% | 616 | 99.2% |
| 1.5 | 505 | 81.3% | 612 | 98.6% |
| 2.0 | 482 | 77.6% | 608 | 97.9% |
| 2.5 | 467 | 75.2% | 604 | 97.3% |
| 3.0 | 442 | 71.2% | 593 | 95.5% |
| 3.5 | 423 | 68.1% | 586 | 94.4% |
| 4.0 | 406 | 65.4% | 581 | 93.6% |

같은 조사에서 추가로 집계한 내용:

- 현재 후보2,647개 중front-valid는23.5%. Front에는 없지만left/right 중 하나에 투영 가능한 대상은
  26.9%. Side projection도 가려짐을 판별하지 않으며 front-only가 중요 대상을 모두 포함한다는 증거가 아니다.
- Front-valid가0개인window는7/128(5.5%). 이들도split에서 제외하지 않고 current-only planning을 유지한다.
- 고정nearest-current-front-K4는391active slot(평균3.05/window, 명목 최대512)을 선택한다.
  +4s visual238/391=60.9%, spatial368/391=94.1%. 명목K4와 실제 예측 대상 수를 분리해 기록한다.
- +4s에spatial만 유효한 대상은175개다. C/D/E에서 독립mask를 쓰면target 내용뿐 아니라 감독 수도 바뀐다.
  첫 주 비교는**공통 visual∩spatial mask**. Native mask 예측 오차·coverage는 별도로 보고한다.

## 4. ROI review와 camera 규약

Train/dev × left/low-turn에서 **처음 관측한** 4window의 contact sheet를 만들고 모두 직접 확인했다.
Front/left/right 현재 영상과 +4s front를 나란히 표시했다. 차량·보행자 투영은 대체로 대상 영역에 맞지만
건물·식생·다른 차량에 가려진 GT box도 projection-valid가 된다. ROI feature에 배경/가림 객체가 들어갈 수 있다.
잘 나온 이미지만 다시 고르지 않았으며 전체 후보의 정합 정확도 측정이나 calibration 증명은 아니다.
공유JSON의 `projection_review_examples` 에 git에서 제외한 로컬PNG 위치를 기록했다.

Distortion은 비제로다. 예: `[-0.356123,0.172545,-0.002130,0.000464,-0.052310]`.
확인한 공식/기존 코드의 범위:

- [nuPlan schema](https://github.com/motional/nuplan-devkit/blob/master/docs/nuplan_schema.md?plain=1)는
  Caltech형식 `[k1,k2,p1,p2,k3]` 를 기록한다.
- [nuPlan Image.load_as/render](https://github.com/motional/nuplan-devkit/blob/master/nuplan/database/nuplan_db_orm/image.py)
  는JPEG를PIL로 읽고 표시 투영에는intrinsics를 쓴다. 해당 경로에는undistort가 없다.
- Drive-JEPA pinned source `548bb8215e3aae18e162a0f12f1ba83b4d3eb57e`의
  `navsim_v1/navsim/common/dataclasses.py::Cameras.from_camera_dict` 와
  `navsim_v1/navsim/visualization/camera.py`도 이미지load와pinhole 투영에undistort를 넣지 않는다.

**Stored JPEG가 사전에rectified됐는지는 이것만으로 확정하지 못한다.** Nonzero distortion 저장도
raw 이미지라는 증거는 아니다. 현재pilot의pinhole 규약을 명시하고 작은front target 탐색에 한정한다.
최종geometry/다시점 평가 전에export 사양/제공원에서 확인한다. 공유 이미지·calibration을 추측으로 수정하지 않는다.

## 5. 재현·공유와 판단

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 \
  runtime/environments/visual_future_prediction_pilot/bin/python \
  scripts/survey_navsim_visual_target_coverage.py \
  --output-directory outputs/data_surveys/navsim_visual_target_coverage_v1_20261001
```

기존output을 덮어쓰지 않으므로 실행한 디렉터리로 재실행하면 거부한다.
재조사 시 다른run directory를 지정하고 원래 결과를 보존한다.
최종CPU run약8.08s, encoder/GPU를 사용하지 않았다. 읽은log16file과review 이미지16file의SHA256은
전후 일치. 모든 미래 이미지의byte hash를 확인했다는 뜻은 아니다. 처리 대상 공용 파일에 대한 쓰기는0.

공유: [summary JSON](../results/data_surveys/navsim_visual_target_coverage_summary_20261001.json)과
[split manifest](../results/data_surveys/navsim_recording_split_manifest_20261001.json).
128window별raw JSON/PNG는 `outputs/data_surveys/navsim_visual_target_coverage_v1_20261001/` 에 저장했다.
이전 조사output도 삭제·덮어쓰기하지 않았다. 공유summary의source hash는 해당script와 일치한다.

**판단**: 작은front-only target 효과 탐색은 준비 가능하다. 절반 이상window에서K4 선택 여지가 없고
turn 시visual supervision 결손이 크므로 최종 선택/상황별 예산 연구로 일반화하지 않는다.
Right/merge·occlusion·rectification·multiview 대응은 미해결로 남긴다.
학습 실행은 아직 없으며 다음은 [최소target 비교 계획](minimal_target_ablation_plan.md)의 검토와 구현이다.
