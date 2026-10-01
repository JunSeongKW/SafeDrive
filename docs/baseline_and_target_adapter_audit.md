# 실제 entity/target adapter와 baseline 소스 감사

2026-10-01, Codex. ChatGPT의 `fe8c930` 검토를 반영했다. 이번 작업의 parent는 `9b7d6e1`.
**확인 사실 / 권고 설계 / 실행 결과 / 미확인**을 아래에서 구분한다.
연구 질문은 바꾸지 않는다: 현재 관측·ego 의도에 따라 같은 예산에서 planning에 유용한 객체의
미래 예측 대상을 학습할 수 있는가? SafeDrive baseline 연구와 재학습은 계속 잠정 중단이다.

## 1. 결론 — 재사용할 것은 encoder·평가 인프라, 없는 연결은 새로 구현

**권고**: Drive-JEPA의 front-video encoder와 perception-free trajectory decoder를 연구용
scaffold로 우선 검증한다. Perception-based proposal planner 전체나 SafeDrive를 현재 주 baseline으로
채택하지 않는다. **공식 Drive-JEPA 그대로의 재현**과 **encoder를 재사용한 신규 연구 scaffold**는
다르며, 신규 scaffold가 공식 점수를 재현한다고 가정하지 않는다.

첫 entity는 GT track/box로 image token의 ROI를 정의하는 **privileged geometry pilot**을 권고한다.
검출 query와 GT association을 한꺼번에 새로 만드는 대신 선택·미래 target 연결부터 점검한다.
이 pilot은 perception-free도, 배포 가능한 end-to-end 시스템도 아니다. 이후 동일 backbone의
실제 detected/tracked entity로 옮겨야 한다. Front camera의 시야 밖 객체·occlusion 문제 때문에
멀티뷰 완성안으로 일반화하지 않으며, 시야 제한이 명제의 실험을 막으면 기존 multi-view feature
backbone의 ROI adapter를 다음 후보로 비교한다. BEV 도입은 필수 전제가 아니다.

**지금 실행한 것**: 실제 NAVSIM mini의 GT 상태 입력·미래 상태 target으로 작은 모델의
forward/loss/backward를 검사했다. 이것은 visual encoder/JEPA 또는 공식 baseline 재현이 아니다.
**아직 실행하지 않은 것**: 공식 encoder checkpoint 로딩, image feature ROI, 시각 target encoder,
planner와 공식 평가 경로의 실제 실행. 본 학습 baseline 확정은 이 gate를 통과한 다음이다.

## 2. 소스 기준과 근거

- Drive-JEPA 공식 clone: `reference_repositories/Drive-JEPA`, clean at
  `548bb8215e3aae18e162a0f12f1ba83b4d3eb57e`.
  아래 `D:` 경로는 이 clone의 `navsim_v1/`에 상대적이다.
  [고정 공식 소스](https://github.com/linhanwang/Drive-JEPA/tree/548bb8215e3aae18e162a0f12f1ba83b4d3eb57e).
- SafeDrive 참고 코드: 현재 root의 `navsim/agents/safedrive/`. 아래 `S:`는 그 디렉터리.
  최초 감사 commit `6ed73948d272218c5fd34c3eddf2dc0d26510393`이며 이번 작업에서 변경하지 않았다.
- Drive-JEPA 논문 v2 §3.2/§3.3: video encoder transfer와 waypoint proposal refinement를 설명한다.
  소스의 head 실행 조건과 detach 판정은 논문 요약 대신 아래 함수의 구현을 근거로 했다.
  [논문 v2](https://arxiv.org/html/2601.22032v2#S3).
- 이 감사는 **NAVSIM v1 경로**에 한정한다. v2/Bench2Drive나 학습 provenance 전체를 감사했다고
  주장하지 않는다. 아래 shape는 소스와 기본 config에서 계산했으며 공식 모델을 실행한 shape가 아니다.

### Drive-JEPA perception-based — 실제 함수 경계

기본 config: `drive_jepa_config.py`: proposal 32, pose 8, feature 256, refinements 4,
`agent_pred=True`, `bev_agent=False`, `bev_map=False`, `image_architecture=vit_large`.
파일은 모두 `D:navsim/agents/drive_jepa_perception_based/`에 있다.

| 파일·함수 | 입력→출력 의미/shape | gradient·특권 입력·제약 |
|---|---|---|
| `drive_jepa_features.py::DriveJEPAFeatureBuilder.compute_features` (66) / `_get_camera_feature` (37) | 과거/현재 front 2장, 각 `[3,256,512]`; ego history `[history,11]`; calibration | 현재까지의 sensor/dynamics/command만 feature로 생성. ego pose도 history-relative. 이 출력에 객체 instance token은 없음 |
| `bevformer/image_encoder.py::ImgEncoder.forward` (61) | `[B,2,3,256,512]`→video ViT `[B,512,1024]`→FPN/embedding `[1,512,B,256]` | Image grid이지 object query나 dense BEV가 아님. GridMask는 student 학습 증강. pretrain path가 다른 사용자의 절대경로로 hard-coded (32) |
| `drive_jepa_model.py::DriveJEPAModel.forward` (47) | ego 11-d encoding + `init_feature` 256개→`[B,32*8,256]` | 변수 `bev_feature`는 초기에는 **ego proposal/time query**다. entity H로 바로 쓸 수 없음. front branch가 feature dict의 lidar2img를 slice하여 변경 |
| `traj_refiner.py::Traj_refiner.forward` (26) | proposal query→`[B,32,8,3]` ego proposals→visual cross-attention으로 query 갱신 | 4개 refiner entry는 model constructor가 **동일 module 객체를 반복**한 weight sharing. 주변 객체 미래 latent predictor 아님 |
| `bevformer/bev_refiner.py::Bev_refiner.forward` (107) | proposal coordinates + queries + image feature→갱신 proposal queries | `ref_2d=pose.detach()` (114): reference 위치를 통한 gradient는 차단. query 내용 경로는 차단하지 않음 |
| `score_module/scorer.py::Scorer.forward` (48) | proposal query time-pooling→`[B,32,6]` score logits | Score head는 query에서 직접 계산. agent/area prediction을 score 입력으로 전달하지 않음 |
| 같은 함수 `pred_col_agent` (65–66) | 학습 시만 `[B,32,8,5,2,9]` 기본 reshape | 마지막 9=box corners 8값+valid logit. 5축은 40개 0.1s 샘플을 8개 query로 나눈 것: **5 agent instance가 아님**. 평가 시 실행 안 함 |
| `score_module/compute_navsim_score.py::get_sub_score` (48) | simulator-scored proposal→target scores / collision corners / area labels | target corners `[P,40,2,4,2]`, valid `[P,40,2]`: proposal-conditioned collision objects. 안정적 current instance→future latent 대응이 아님 |
| `drive_jepa_model.py::forward` (95–103) | score 마지막 항목 sigmoid→argmax→ego trajectory `[B,8,3]` | hard final trajectory 선택. 학습에서는 proposal/score 각각 loss가 있지만 정수 argmax 선택 자체는 미분 안 됨 |
| `drive_jepa_features.py::DriveJEPATargetBuilder.compute_targets` (141) | future ego trajectory `[8,3]` + token | object/map target 생성은 주 return에서 주석 처리됨. optional `bev_agent=True`를 켜는 것만으로 target 공급이 해결되지 않음 |
| `drive_jepa_agent.py::compute_score` (102), `pad_loss` (249), `score_loss` (145) | proposal.detach→CPU simulator/metric-cache로 학습 label 생성; proposal imitation+score/corner/area losses | GT는 학습 label. detached simulator label에 역전파하지 않음. training 시 metric cache / anchor scores와 worker infrastructure 필요 |

위 `pred_col_agent`의 시간축 해석은 reshape만으로 정하지 않았다. `compute_navsim_score.py`의
`proposal_sampling=num_poses40, interval0.1` 및 corner target/loss reshape를 함께 대조했다.
`bev_agent=True`의 optional `_agent_head`도 `MyTransformeDecoder`가 출력 heads를 바로 적용하므로,
현재 코드가 latent detection query를 외부에 제공하거나 identity association을 해결하지는 않는다.

**확인된 판단**: 이 분기에서 agent future-state head는 scorer와 공유한 feature를 학습하는
보조 supervision이다. 그 **예측 출력 자체**는 proposal generator나 score head의 입력이 아니다.
공유 representation을 통해 학습에 영향을 줄 수 있다는 사실과 추론 중 미래 출력을 사용한다는
주장을 구분한다. 따라서 `selector→entity future latent→planner`는 새 연결이 필요하다.

### 재사용 후보별 H / Z / association / planner

| 후보 / source 함수 | 현재 entity H | 미래 target Z·association | planner 입력 / gradient / 필요한 추가 코드 |
|---|---|---|---|
| SafeDrive `S:safedrive_model.py::_proposal_object_detection` (594), `_forward_swnet` (859) | 기본 detection query `[layer,B,30,256]`에서 마지막 layer의 H를 사용 가능; 좌표·valid 및 DN padding별 shape는 실행 재확인 필요 | `S:safedrive_features.py::compute_targets`, `_align_future_agent_states` (689)의 GT track 정렬은 참고 가능. 현재 검출 query↔GT instance matching과 미래 visual target encoder는 새로 필요 | joint agent/ego decoder에 넣음. `select_topk` (1266)의 index policy는 미분 안 됨. `det_coord_detach` 및 DN target 사용을 별도 감사해야 함. 미래 출력→planner의 새 boundary와 MM stack/cache 재구성 필요 |
| Drive-JEPA perception-based 위 함수들 | 기본 proposal query는 entity가 아님. optional agent head는 box/label 출력만 반환 | 미래 latent teacher 및 현재 entity track association 모두 없음 | proposal visual decoder 및 score infrastructure는 재사용 가능하나 instance adapter+미래 경로 추가와 외부 dependencies/caches 필요 |
| **권고 pilot** `D:navsim/agents/drive_jepa_perception_free/drive_jepa_model.py::forward` (87), `D:vjepa2/evals/image_classification_frozen/modelcustom/vit_encoder.py::init_module` (39) | 기존 front image grid `[B,512,1024]`에서 current box ROI로 새 instance token 생성 | 동일 고정 encoder의 future clip grid에서 같은 GT track ROI. 미래 box는 training target 위치만 제공; 현재 input에 미래 넣지 않음 | 현재 grid+ego + 새 predicted entity tokens를 trajectory Transformer의 memory로 concat. frozen encoder `no_grad` 후 S/P/D는 미분 유지. image ROI + mask + teacher + memory adapter는 새로 필요 |

권고안은 prediction/scoring을 포함한 전체 공식 Drive-JEPA보다 좁은 재사용이다. 아직 코드가 없는
entity visual adapter 비용을 “selector만 추가”로 축소하지 않는다. SafeDrive의 association 아이디어는
참고하되 수천 줄 모델의 기본 학습 스택을 새 연구에 재도입하지 않는다.

## 3. 권고 visual adapter v0 명세 — 아직 미구현

다음은 **설계 결정 후보**이며 실행 사실이 아니다. 실제 weight batch에서 shape·유효 ROI 개수를
확인한 뒤 채택한다. first fixed budget 예시는 K=4, horizon=4s/8 future slots; 본 학습 전에
데이터 유효율·메모리를 실측하며 성능이 좋은 값으로 사후 고르지 않는다.

1. Current clip `[t-0.5,t]`, front image `[B,3,2,256,512]`를 공식 frozen encoder에 넣는다.
   원본 front 1920×1080의 위/아래 28px crop, 512×256 resize를 그대로 적용한다.
   patch16/tubelet2 → output grid `[B,16,32,1024]`를 검사한다.
2. **H**: 현재 GT lidar box의 8 corner를 camera extrinsics/intrinsics로 투영하고 crop/resize한
   image 좌표에 맞춘다. ROI pooling된 **image grid feature**와 현재 geometry/class를 묶는다.
   이것은 detection query도 BEV ROI도 아니다. 현재 valid mask는 현재 frame의 projection으로만 정한다.
   near-plane clipping, screen intersection, distortion/rectification의 데이터 규약을 실제로 확인해야 한다.
   front projection 유효는 완전한 관측/occlusion 확인을 뜻하지 않는다.
3. **Z**: future step j의 clip `[t+0.5(j-1),t+0.5j]`을 같은 weight의 **frozen eval teacher**로 encode.
   clip 끝의 동일 track GT box로 ROI pooling하여 `[B,current_entity_count,8,1024]`를 만든다.
   미래 target은 stop-gradient; teacher weight/정규화 고정. 초기에는 EMA teacher를 새로 학습하지 않는다.
   tubelet feature가 두 시점 관측을 섞는 표현이라는 한계를 명시한다. 미래 clip은 마지막 관측보다
   더 뒤의 frame을 사용하지 않는다.
4. **association**: current track tokens가 candidate 집합의 기준이다. future annotations 전체에서
   matching한 뒤 ROI 유효성을 판정한다. future top-K/range를 먼저 잘라 track을 잃지 않는다.
   새로 등장한 객체는 첫 pilot의 예측 대상 밖; 사라짐/시야 밖/invalid teacher ROI는 **loss mask**로만
   처리한다. 미래 mask로 현재 candidate나 선택 score를 변경하면 누출이므로 금지한다.
5. **좌표**: visual target은 각 future camera의 image feature다. future ego pose로 현재 위치를 warp한
   BEV latent라고 부르지 않는다. current geometry/ego 상태·명령과 future offset embedding으로
   predictor가 해당 image-coordinate target을 예측한다. ego action 반사실/새 의도 결과까지 검증한
   world model이 아니며, commanded direction 변경 시 logged visual future를 새 GT로 취급하지 않는다.
6. **planner boundary**: Drive-JEPA perception-free model의 image/status memory `keyval_final`에
   `[B,K*8,256]` projected predicted-future tokens와 해당 valid slot mask를 붙여 Transformer memory로
   전달한다. 현재 전체 image memory는 유지한다. 새 함수 `build_planning_memory`에서 **예측값만**
   추가하고 target은 입력받지 않는다. dynamic memory 길이/pos embedding/padding mask를 수정해야 한다.
   기존 최상위 forward를 무변경으로 재사용 가능한 것은 아니다.

계산 경로:

```text
history/current images ─ frozen E ─ current grid ─ ROI(current boxes) ─ H ─ S(C,u) ─ ST K ─ P ─ future predictions ─┐
                               └── full current image/status context C ────────────────────────────────────┤ D → ego plan → Lplan
future images ─ frozen E_target ─ ROI(same-track future boxes) ─ stopgrad Z ─ Lfuture(P(hard-selection.detach)) ┘ [loss only]
```

`Lplan→D→P→S` 유지; `Lfuture→P`는 유지하되 selector 직접 경로 차단. current/target encoder는
frozen이므로 둘에 gradient가 없다는 것이 예상 계약이다. target teacher의 future metadata는
training-only다. hard-K 운영은 현재 순차 ST를 임시 사용하고 joint learning 전에 collapse/bypass를 검사한다.
현재 H 경로를 없애 미래 의존을 강제로 만드는 것은 기본안이 아니다.

## 4. 구현된 실제 GT-state 진단 — 시각 모델과 구분

`src/planning_aware_future_prediction/adapters/navsim_tracked_state.py`:

- `build_tracked_state_online_inputs(history_frames, config)`에는 미래 frame 인자가 없다.
  정확히 4개 history/current frame으로 H·ego history/dynamics C·현재 command u를 만든다.
- H `[1,N,10]` = normalized x/y, sin/cos heading, length/width, vx/vy, vehicle/pedestrian class.
  반경40m 이내 현재 GT 차량·보행자 중 거리 순 최대32개. **현재 GT 특권 정보**이며 시야/검출 검증 아님.
- `build_tracked_state_training_targets(current_frame, future_frames, online_inputs, config)`는
  same track의 future **state** `[1,N,8,6]`(x/y, sin/cos heading, vx/vy)와 ego `[1,8,3]`를 별도 생성.
  velocity/box는 lidar→global→현재 planar ego frame 변환; ego pose yaw는 공식 pyquaternion 규약과 맞춤.
  GT association·미래 ego 변환은 target 생성에만 사용한다. 이름이 `future_latent_targets`인 generic
  fixture loss 인자로 state를 넣었어도 **시각 latent 또는 JEPA 실험으로 보고하지 않는다**.
- 미래 출현/소멸, annotation 순서 변경, invalid NaN, 현재 빈 집합, 먼 미래 대상, timestamp gap을 검사.
  현재 track ID hash는 tie/association만 하고 scorer content에는 넣지 않는다.
- 기존 작은 graph를 waypoint8 출력으로 확장했고 기본 1-step fixture interface는 유지했다.
  미래 state aux direct gradient 차단과 planning S/P/D gradient를 실제 입력에서 확인했다.

실행 결과와 한계는 [실제 데이터 진단 보고서](navsim_state_adapter_validation.md).
공식 entity visual input, perception stack의 label 경계 또는 encoder forward를 검증한 것으로
대신하지 않는다. 이미지는 파일 header/무결성 확인만 했다. 공용 원본은 read-only이고 log/image
hash가 실행 전후 같았다.

## 5. 현재 feature 전달 대조와 최소 실험표 — 미실행

공통 backbone·entity H·planner memory boundary·K·time slots·train split·step·optimizer·head 용량 고정.

| 조건 | selector | 같은 predictor/adapter 출력 크기 | auxiliary supervision | 분리하려는 효과 |
|---|---|---|---|---|
| Future-random | random valid K | K×8×latent_dim | 미래 visual Z | 단순 선택 대비 |
| Future-rule | 현재 distance/TTC 규칙 K | 동일 | 미래 visual Z | 강한 입력-only 규칙 대비; TTC 정의·0/음수 closing speed 처리 확인 |
| Future-context | current H/C/u + planning gradient | 동일 | 미래 visual Z; selector detach | 주 가설 H2 |
| Same-architecture no-future-aux | 위와 같음 | **모듈·parameter 수까지 동일** | 미래 auxiliary weight0 | 미래 감독 효과. predictor가 미래 정보를 전혀 담지 않는다는 뜻 아님 |
| Current-feature adapter | 위와 같음 | 동일 구조/출력 slot/parameter 수, current-feature adapter로 해석 | 미래 Z 대신 현재 teacher ROI를 8개 slot에 반복한 target | 미래 감독과 현재 feature 전달의 비교. 같은 용량만으로 정보량이 완전히 같다고 주장하지 않음 |

첫 visual batch gate가 통과한 뒤 random/rule/context의 작은 **공동 학습**부터 시작한다.
미래 감독 on/off 및 current-target 대조는 같은 seed 초기값·학습량을 맞춘다. random/rule에는
제안 scorer와 같은 학습 parameter가 없으므로 향상 해석에 parameter-matched entity-only no-intent
학습 조건을 추가한다. 처음부터 전체 조합을 실행하지 않는다.

GT-state 버전은 **engineering/privileged-state diagnostic**으로만 별도 표시한다. 공식 navtest score와
섞지 않는다. navmini는 smoke 전용, 학습/개발 split의 공식 log 목록을 대조한 뒤 고정한다.
navtest는 반복 진단용; 독립 final holdout라고 부르지 않는다.

진행 gate: 실제 encoder 출력 finite, projected ROI/track valid율 기록, target/online 분리,
planning→S gradient와 auxiliary direct 차단, 단일 joint update finite. 다음으로 작은 log-disjoint
공동 학습에서 여러 seed·선택 다양성·future-path 제거/교환의 **학습 후 성능**을 확인한다.
Gradient nonzero나 untrained 출력 변화만으로 학습 유용성을 주장하지 않는다.
개선이 추가 용량/학습/토큰/연산 차이로 설명되면 H2 주장을 축소한다.

## 6. 자원·막힌 지점·가장 작은 다음 행동

**로컬 확인**: 현재 CPU venv는 torch2.8.0+cu128, Python3.12.13을 기존 환경에서 읽기 전용 참조.
numpy/torchvision/PIL/pyquaternion은 존재하지만 nuplan/mmcv/mmdet/timm은 없음.
공식 안내는 Python3.9, torch2.1.0/cu121, requirements의 `mmcv_full1.7.2`, `mmdet2.28.2`다.
기존 환경을 업그레이드하지 않았다. MM 스택은 권고 front encoder/ROI 최소 경로에는 넣지 않고,
공식 perception-based 재현이 필요해질 때 독립 환경에서 다룬다.

**온라인 원본 확인**: [공식 weight 목록](https://huggingface.co/datasets/LinhanWang/Drive-JEPA/tree/main)
(2026-10-01 열람, 표시 revision `65e0d72`)에는 pretrained encoder `vitl_merge_3dataset_e50.pt`
5.13GB, perception-free ViT checkpoint3.72GB, perception-based ViT3.74GB, ResNet34117MB가 표시된다.
파일이 공개되어 있다는 것은 다운로드·checkpoint key 호환·재현 성공을 뜻하지 않는다.
5.13GB pretrained 파일과3.72GB downstream 파일을 둘 다 받거나 전체80GB bundle을 받지 않았다.
파일별 full revision/size/hash 및 필요한 encoder state-dict prefix를 확인해 **한 파일만 선택**해야 한다.

다음 행동은 **독립 front-encoder 환경 구성안 고정 → 선택한 공식 weight 한 개의 metadata/key
확인·선택 다운로드 → 최소 current/future visual batch + ROI/track adapter 구현·검사**다.
weight 저장·cache·변환은 사용자 `/rhome/junseong/` 작업공간이며 새 dataset 원본만
`/home/user/data/processed_dataset/` 1TB 규칙을 따른다. 공용 dataset에는 어떠한 파일도 생성하지 않는다.
GPU0·1 사용만 승인됐으며 이번 작업은 CPU만 썼다. GPU-hour/VRAM/학습 안정성은 미측정이다.

비용 비교는 “SafeDrive는 cache/검출/query/target 교체”, “perception-based는 MM/proposal/scorer/entity
adapter”, “권고 pilot은 frozen video encoder+image ROI+별도future memory adapter”라는 코드 작업량
차이에 근거한다. 일수/GPU-hour를 실측 확정값으로 적지 않는다. 논문 novelty 전체 감사는 별도 미결이다.
