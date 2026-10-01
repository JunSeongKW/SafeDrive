# 실제 front-video / GT ROI 미래 예측 pilot

2026-10-01, 사용자·ChatGPT의 `35fbdcf` 검토 이후 수행. 하위 질문은 **실제 영상 표현으로도
선택→선택적 미래 예측→planning의 gradient 경계를 구현할 수 있는가?**다.
H1/H2, 자율주행 성능, novelty, 공식 Drive-JEPA 전체 재현을 검증한 실험이 아니다.
SafeDrive 주 baseline 연구는 계속 잠정 중단이다.

## 1. 이번에 실제 구현하고 실행한 것

- 공식 driving video encoder checkpoint **한 개만** 다운로드하고 크기·SHA256 확인.
  `target_encoder` tensor292개를 `strict=True`로 로딩했다. Missing/mismatch를 임의 초기값으로
  대체하지 않는다. `weights_only=True, mmap=True`를 썼고 unsafe pickle fallback은 하지 않았다.
- 별도 `visual_future_prediction_pilot` venv에 `timm==1.0.30`만 추가. 기존 환경 upgrade 없음.
  PyTorch 등은 기존 `alpasim-cuda128`을 읽기 전용 상속한 overlay이므로 완전 hermetic 환경이 아니다.
- 공용 NAVSIM mini 한 구간: 현재/미래2-frame front clip9개, same-track GT box ROI와 teacher target.
- 새로 초기화한 작은 selector/predictor/Transformer trajectory decoder에 영상/공간 target 연결.
  공식 encoder만 실제 재사용하고, planner는 공식 image/status-memory+trajectory-query 패턴을
  새 작은 모듈로 구현했다. 공식 planner weight·scorer·제안256개 pipeline을 재현한 것이 아니다.
- 단위 검사33개(기존28+visual5), 실제 영상 gradient 검사·공동 backward·Adam 단일 update 통과.
  추가 합성 학습/accuracy tuning, GT-state 검사 확대, benchmark 평가, 본 학습 없음.
- Log1개와 읽은 front image10개의 SHA256 전후 동일. 공유 원본은 생성/수정/이동/삭제하지 않았다.

공유 수치 원본: `results/visual_diagnostics/visual_future_pilot_verified_20261001.json`.
현재·미래 ROI 그림은 로컬 `outputs/visual_pilot/visual_future_pilot_verified_20261001_roi_contact_sheet.png`.
PNG와 weight/env는 git 제외다. 그림의 노란 선택은 **학습 전** 선택이지 planning 중요도 증거가 아니다.

## 2. 공식 자산·환경·실행 범위

| 항목 | 고정 / 실제 확인 |
|---|---|
| 공식 코드 | Drive-JEPA `548bb8215e3aae18e162a0f12f1ba83b4d3eb57e`, clean clone |
| 공식 weight | HF dataset `LinhanWang/Drive-JEPA`, `vitl_merge_3dataset_e50.pt` |
| HF revision | `65e0d7284f69bf29d1a4864affcfd84ca4e97a2e` |
| 파일 크기 | 5,127,748,765 bytes |
| SHA256 | `4649182770ef68f84a001780c6579435345948fd80f186fa3616ab078ced668f` |
| 모델 | ViT-L/16, tubelet2, RoPE,1024dim, encoder303,885,312 parameters |
| 입력 | RGB, top/bottom28px crop, cv2 linear512×256, ImageNet normalization, old→new2frames |
| 출력 | clip `[1,3,2,256,512]` → `[1,1024,16,32]` |
| Teacher / student features | 같은 frozen pretrained encoder; eval, no_grad, EMA update 없음 |
| Runtime | Python3.12.13, torch2.8.0+cu128, torchvision0.23.0+cu128, timm1.0.30, cv2 5.0.0 |
| 실행 GPU | 승인 범위의 GPU0 RTX A6000만 사용; GPU1/타 프로세스 건드리지 않음 |
| 표본 | mini log `2021.05.12.22.00.38_veh-35_01008_01518`, window0, current index3 |
| Scene | `165060762e765a5a`; 과거 state smoke와 같은 scene, 독립 평가 표본 아님 |

근거: [공식 encoder 코드](https://github.com/linhanwang/Drive-JEPA/blob/548bb8215e3aae18e162a0f12f1ba83b4d3eb57e/navsim_v1/vjepa2/src/models/vision_transformer.py),
[공식 front feature builder](https://github.com/linhanwang/Drive-JEPA/blob/548bb8215e3aae18e162a0f12f1ba83b4d3eb57e/navsim_v1/navsim/agents/drive_jepa_perception_free/drive_jepa_features.py),
[공식 normalization/planner 패턴](https://github.com/linhanwang/Drive-JEPA/blob/548bb8215e3aae18e162a0f12f1ba83b4d3eb57e/navsim_v1/navsim/agents/drive_jepa_perception_free/drive_jepa_model.py),
[고정 revision checkpoint](https://huggingface.co/datasets/LinhanWang/Drive-JEPA/blob/65e0d7284f69bf29d1a4864affcfd84ca4e97a2e/vitl_merge_3dataset_e50.pt).
PyTorch 버전, GPU-hour 등은 이 pilot 조건이지 공식 논문 실험 환경/비용이 아니다.

## 3. 무엇의 미래를 예측하는가 — 위치가 ROI에서 사라지는 문제

Target을 명시적으로 두 부분으로 나눴다.

| 변수 | Shape / 내용 | 생성 / 사용 |
|---|---|---|
| 현재 영상 맥락 | `[1,1024,16,32]` | 현재·직전0.5s clip, frozen encoder |
| 현재 entity | `[1,32,1034]` | GT projected ROI1024 + 현재 GT geometry/class10 |
| Selector ego 입력 | `[1,8]` | 현재 driving command4 + 속도/가속4 |
| Future appearance target | `[1,32,8,1024]` | 미래2-frame clip encoder에서 미래 **동일 track** ROIAlign |
| Future spatial target | `[1,32,8,6]` | 미래 x/40,y/40,sin yaw,cos yaw,vx/10,vy/10; **현재 ego 좌표계** |
| 선택적 예측 | visual `[1,4,8,1024]`, spatial `[1,4,8,6]` | predictor가 현재 입력으로 예측; 미래 GT 입력 아님 |
| Planner 입력 | 현재 영상128tokens+현재 ego+현재 entity+예측 미래32tokens | 현재 정보는 유지하며 미래 GT/mask 미전달 |
| Ego output | `[1,8,3]` | x,y,yaw, 현재 rear-axle frame |

단위 ROI가 차량 외형을 담는 것과 이동/ego 관계를 담는 것은 다르다. 이번에는 **위치·속도를
별도의 명시적 spatial head/target으로 보존**한다. 시각 target에 그것이 충분히 들어 있다고
주장하지 않는다. 이 spatial target은 GT 좌표/속도 감독이며 JEPA latent가 아니다.
따라서 전체는 **frozen visual-latent + explicit spatial-state 혼합 감독 pilot**이다.
순수 visual JEPA나 semantic detector 출력 기반 배포 가능한 방법으로 부르지 않는다.
최종 target 설계는 visual-only / spatial-only / mixed 비교와 novelty 검토 후 결정한다.

Teacher는 미래 영상/box를 학습 label 생성에만 사용한다. Forward signature에는 미래 target,
미래 영상, 미래 box, 미래 target-valid가 없다. 미래 clip은 `[t+(j-1)Δ,t+jΔ]`의 tubelet feature라
순간 t+jΔ만의 객체 표현은 아니다. 현재 clip도 `[t-Δ,t]`이므로 온라인 미래 누출은 없다.
GT annotation geometry와 camera calibration으로 box를 투영하고 ROIAlign3×3 평균을 사용한다.
Attention이 전체 영상에서 동작하므로 ROI feature는 객체만의 외형이 아니라 배경/맥락도 포함한다.

현재 후보는 차량·보행자,40m radius,가까운32개 cap이라는 **고정 전처리 제한**을 갖는다.
그중 현재 front projection-valid만 S의 후보다. Stable ID는 tie/association용이며 score 입력이 아니다.
미래에는 모든 annotation의 track을 먼저 대응하고 미래 radius/topK로 다시 필터하지 않는다.
미래 front ROI-invalid는 visual loss만 제외한다. Track GT가 존재하면 화면 밖이어도 spatial loss는 가능하다.
새로 나타난 객체는 현재 후보가 아니므로 이번 예측 대상에 포함되지 않는다.
Occlusion과 추적 오류는 이 pilot에서 해결하지 않았다.

## 4. 실제 gradient 결과와 한계

첫 실행의 norm은 아래와 같고 최종 format/시각화 정리 후 재실행에서도 동일 zero/nonzero 계약을
확인했다. 공유 JSON의 최종 수치를 우선한다.

| Loss / 대조 | Selector | Predictor | Planner |
|---|---:|---:|---:|
| Planning MSE | 0.029646 | 0.466658 | 25.917812 |
| Future visual auxiliary | 0 | 3.903199 | 0 |
| Future spatial auxiliary | 0 | 6.385962 | 0 |
| Future detach + planning | 0 | 0 | 25.917812 |
| No future branch + planning | 0 | 0 | 25.631276 |

Auxiliary는 `hard_selection.detach() @ current_entity`로 predictor를 **별도 호출**한다.
이로써 loss의 직접 S/D gradient를 차단한다. P 업데이트로 이후 선택 양상이 간접적으로 바뀔 수
있으므로 예측 난이도/visibility 편향이 전부 제거됐다고 주장하지 않는다.
Planning은 ST의 편향된 surrogate로 S에 전달한다. 연결 검사와 좋은 선택 학습은 다르다.
`detach`는 forward가 정확히 같고 backward만 다르다. Branch 제거는 forward 구조도 달라진다.
No-branch 대조에서도 동일 현재 GT entity/영상/ego 정보를 planner에 제공한다.
같은 Python model에 P/S parameter가 남아 있으나 그 forward 호출/활성 경로는 생략된다.

실제 loss는 planning MSE+visual MSE+spatial MSE의 단순합으로 한 번 update했다.
이 가중치/latent scale은 학습 안정성이나 최종 성능에 적합한 값으로 검증되지 않았다.
Current target으로 label만 교체해도 forward는 동일했다. Future zeroing에 따른 미학습 ego 출력 차이는
약0.03767로 연결을 확인할 뿐, 학습한 planner가 미래를 유용하게 쓴다는 증거가 아니다.

## 5. ROI/target 내용의 실제 관찰

- 현재32 후보 중 front ROI 유효13(40.625%). 고정 K4; 선택된4개 모두8시점의 visual target 유효.
- 미래 각 시점에서 현재32 track의 front ROI 유효수는 `[12,10,10,10,10,10,11,12]`.
  이는 화면 투영 유효수이며 occlusion을 반영한 실제 가시성 수가 아니다.
- 현재/future ROI cosine 평균0.89360, 최소0.74484. 대응되는 metric displacement 평균0.91031m,
  최대13.81093m. 단일 구간의 설명 통계다. Feature가 위치를 못 담는다는 인과 증명도,
  visual target이 planning에 필요한 motion 정보를 충분히 담는다는 증명도 아니다.
- Overlay를 직접 확인한 결과 차량과 투영 영역이 대체로 정합한다. 일부는 가려지고 ROI가 겹친다.
  공식 pinhole 투영을 따랐으며 distortion/rectification 규약은 아직 독립 확인하지 않았다.
- Front-only는 좌회전/합류의 측면 중요 객체를 보장하지 않는다. 원래 가설 전체로 일반화하지 않는다.
  상황별 coverage 및 multicamera 후보/association 정책 확인이 본 학습 이전 gate다.
- 새 scaffold parameter2,487,988; frozen encoder 약304M은 별도다. K4×8 예측 visual32768values,
  spatial192values. Encoder는 전체 front clip을 처리한다. FLOPs/지연 절감은 검증하지 않았다.
- 첫 연결 검사23.67s, CUDA 최대 **allocated**1,324,247,552 bytes(약1.23GiB); hash/weight load 포함.
  이것은 reserved/전체 process VRAM 또는 epoch GPU-hour가 아니다. 최종 실행값은 JSON 참조.

## 6. 대조 설계 — 서로 다른 질문을 섞지 않는다

아래는 다음 작은 공동 학습을 위한 **계획**이다. 현재 수행한 것은 label/branch 경계 진단뿐이다.

| 대조 | Future branch | Visual auxiliary target | Spatial auxiliary target | 분리하려는 효과 |
|---|---|---|---|---|
| 제안 mixed pilot | 있음, K4 | 미래 ROI | 미래 state | 선택적 미래 경로 전체 |
| Visual JEPA auxiliary 없음 | 동일 | 없음 | 동일 미래 state | 영상 미래 감독의 추가 효과 |
| 미래 감독 전부 없음 | 동일 | 없음 | 없음 | 추가 모듈만으로 생기는 효과 |
| 현재 visual target | 동일 | 현재 ROI 반복 | 동일 미래 state | 영상 미래 vs 현재 감독 |
| 현재 visual+state target | 동일 | 현재 ROI 반복 | 현재 state 반복 | 현재 정보 side-channel 진단 |
| 미래 branch 없음 | 없음 | 없음 | 없음 | 현재 맥락/entity/ego만 사용하는 planner |
| 전체 entity 미래 예측 | 있음, 모든 현재 valid | 미래 ROI | 미래 state | **다른 예산**의 참조 |

Random/거리·TTC/제안 선택은 같은 mixed architecture, 동일 K/horizon/학습량/seed로 비교한다.
먼저 visual/spatial 감독과 branch 효과를 분리해야 visual JEPA 기여를 주장할 수 있다.
GT-state-only 비교도 이 단계에 유용하며 mixed 개선을 영상 표현의 성과로 바로 부르지 않는다.
모든 row를 처음부터 전량 학습하지 않는다. 공통 backbone이 동작하는 지금, 표본/log 분할과
작은 학습의 진행·중단 조건을 정하고 꼭 필요한 비교만 선정한다.
전체 예측 참조의 우위는 선택 연구의 필수 진행 조건이 아니다.
Current-feature 직접 전달 adapter는 추가 확인 후보며 현재 구현한 것은 **current-target loss 대조**다.

## 7. 재현 명령 / 파일 경계

새 머신은 고정 official source를 `reference_repositories/Drive-JEPA`에 준비하고 dataset read-only
symlink를 만든다. 기존 root `setup.py`나 SafeDrive 환경/캐시 script를 실행하지 않는다.
이미 준비된 현재 서버에서는 env 재생성/weight 재다운로드가 필요 없다.

```bash
cd /rhome/junseong/PlanningAwareFuturePrediction
# 현재 서버의 overlay 생성 이력; 기존 env를 변경하지 않음
/rhome/junseong/miniconda3/envs/alpasim-cuda128/bin/python -m venv \
  --system-site-packages runtime/environments/visual_future_prediction_pilot
runtime/environments/visual_future_prediction_pilot/bin/python -m pip install \
  --no-deps -r configs/visual_pilot/requirements_overlay.txt
runtime/environments/visual_future_prediction_pilot/bin/python \
  scripts/download_visual_encoder_checkpoint.py

PYTHONPATH=src CUDA_VISIBLE_DEVICES='' \
  runtime/environments/visual_future_prediction_pilot/bin/python \
  -m unittest discover -s tests -v

# GPU 점유를 먼저 확인. Existing result overwrite는 거부하므로 새 명칭 사용.
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTHONDONTWRITEBYTECODE=1 \
  runtime/environments/visual_future_prediction_pilot/bin/python \
  scripts/validate_visual_future_prediction_pilot.py --device cuda \
  --output outputs/visual_pilot/visual_future_pilot_recheck.json
```

파일·함수: `adapters/navsim_visual_entities.py::project_lidar_box_to_front_roi/front_track_rois/
pool_projected_entity_rois`, `models/frozen_driving_video_encoder.py::FrozenDrivingVideoEncoder`,
`models/visual_entity_future_planning_pilot.py::VisualEntityFuturePlanningPilot/
detached_selection_auxiliary_losses/CurrentAndPredictedFutureTrajectoryDecoder`.
실행 script가 source SHA256, dataset log/images SHA256, checkpoint provenance, shape, gradient,
coverage, single update 및 measured runtime을 기록한다.
최초 테스트 명령에서 `PYTHONPATH=src`를 빠뜨려 기존 module2개 import 실패했으나 명령을
바로잡아 전체33개를 통과했다. upstream timm import/CUDA attention의 deprecation 경고는 남고,
기능 실패가 아니라며 공식 source를 임의 수정하지 않았다.

## 8. 다음 gate / 미확인

1. Mixed target vs pure latent라는 주장 범위를 검토하고 visual/state/current-branch 대조 확정.
2. 관련 연구 원문/공식 코드 novelty 표 완성. 이 pilot 성공은 novelty 확보가 아니다.
3. 여러 log의 front/multiview ROI coverage·occlusion·GT association 의존도 조사; split을 log단위로 고정.
4. 작은 공동 학습에서 random/강한 규칙/제안 선택 비교. Collapse, 미래 무시, current side-channel,
   미래 spatial 감독 의존도, command 조건 효과 확인. 예산·학습량·parameter 효과 통제.
5. Official metric/evaluation path와 독립 holdout 연결 후 본 학습 여부 결정. 큰 학습/cache는 아직 시작 안 함.

최종 baseline·학습 안정성·공식 planning score·일반화·효율·추론 검출/association·novelty는 미확정이다.
