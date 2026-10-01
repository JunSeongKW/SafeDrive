# 선택적 미래 예측 기반 결정 — 코드 감사와 첫 통제 실험 명세

기준: 프로젝트 `578be6e168e86cd5bebfdf78a0d554fe556c634b`, 2026-10-01.
**이번 산출물은 기반 추천과 설계다. 모델 이동·selector 구현·새 학습·추가 전체 평가는 하지 않았다.**

## 1. 결정 요약과 보존 범위

**WA-JEPA를 다음 구현 기반으로 추천한다.** 공개 joint predictor가 추론 중 미래 장면과 ego 궤적을
같이 갱신하고, 학습의 trajectory loss에서 미래 scene hidden으로 gradient가 연결된다.
Drive-JEPA PB는 학습용 미래 객체 head가 있지만, 그 출력을 planning에 소비하지 않는다.
새 미래 predictor/planner를 다시 개발하기보다 기존 joint predictor를 재사용하는 편이 연구 질문에 직접적이다.

단, **공간 patch-tube 선택을 첫 구현 단위로 승인하는 결정**과 **WA 공식 weight/config의 실제 호환성 검증**이
남았다. 이를 통과하기 전 추천을 실행 확정·재현 성공으로 표현하지 않는다.
객체 instance 선택을 유지하려면 entity-to-patch/association을 추가해야 한다. Patch를 entity라고 부르지 않는다.

완료한 공식 Drive-JEPA PF ViT-L baseline은 그대로 보존한다:

- `results/official_drive_jepa_reproduction/`: 원본 scene CSV, 설정/scorer 감사, source/checkpoint/hash, 비용.
- 12,146/12,146 성공, PDMS **89.224320**. 논문 89.0 대비 +0.224320점 원인은 미확정.
- PF forward는 encoder→trajectory decoder이며 **추론 중 별도 미래 predictor 호출 없음**.
- 공식 평가 baseline 확보와 선택적 미래 예측 구현 기반 확보는 별개의 일이다.
- 기존 결과·환경·checkpoint·pilot·SafeDrive 자산을 변경하지 않았다. 점수 맞추기/전체 평가 반복 없음.

## 2. 직접 확인 수준과 소스 고정

| 후보 | 직접 읽은 것 | 이번 실행 | 아직 확인하지 않은 것 |
|---|---|---|---|
| Drive-JEPA PF | 기존 공식 forward와 재현 기록 | 저장 scene 결과 CPU 분석만 | +0.224320 원인 |
| Drive-JEPA PB | model/backbone/refiner/scorer/targets/agent loss/Lightning training/eval script | 코드 읽기 + 공식 HF 파일 목록 | PB weight 로딩·실제 forward/gradient·PB score |
| WA-JEPA | joint attention/predictor/positioner, teacher/flow losses, train loader/trainer, 추론 agent/config/strict loader | 코드 읽기 + 공식 HF 파일 목록 | full checkpoint/config 호환·full gradient·VRAM/latency·공식 score |
| ForeDrive v2 | 원문 §3/Eq.(7)/Appendix H 및 제한된 공식 링크 검색 | 문헌 확인만 | 공식 repo/weight 위치, 실제 구현·gradient·비용 |

공식 로컬 clone 둘 다 변경 없이 읽었다. 과거 tiny WA attention backward는 full-model 검증을 대신하지 않는다.
HF 메타데이터만 재조회했고 새 모델 파일은 다운로드하지 않았다.

- Drive-JEPA: [`548bb8215e3aae18e162a0f12f1ba83b4d3eb57e`](https://github.com/linhanwang/Drive-JEPA/tree/548bb8215e3aae18e162a0f12f1ba83b4d3eb57e).
- WA-JEPA: [`bec29660f5ea46ac73db8e2d0c33c8d1a72c23ad`](https://github.com/AFARI-Research/WA-JEPA/tree/bec29660f5ea46ac73db8e2d0c33c8d1a72c23ad).
- 공개 파일 이름/용량/LFS hash: [metadata JSON](../results/foundation_selection/public_checkpoint_metadata.json).

## 3. 코드 근거 비교표

아래 PB는 NAVSIM v1 ViT-L preset, WA는 `configs/wa_jepa_navsim_pdms.yaml` 기준이다.
파일/함수 번호는 아래 근거 목록에서 고정 commit에 연결된다.

| 확인 항목 | Drive-JEPA PF | Drive-JEPA PB | WA-JEPA | ForeDrive v2 — 문헌만 |
|---|---|---|---|---|
| 실제 센서·시간 | front 2 frame | ViT는 front 2 frame. SensorConfig는 다른 3 camera의 현재도 요청하지만 pixel encoder는 front만 소비. ResNet 변형은 4 camera 현재 1 frame | left/front/right/back 4 camera × history4, 2Hz, 256×512; future8, 4초 | front 현재 영상, 복수 미래 horizon |
| 현재 표현 | 영상 patch + ego 상태 | 영상 patch→proposal/waypoint query. `bev_feature`라는 이름을 dense BEV/instance로 해석하지 않음 | view별 16×32 image patch, history2 tubelet, projected dim512 | image patch latent |
| 미래 predictor train/infer | 사전학습과 downstream 구분; downstream 없음 | 충돌 관련 미래 객체/도로 auxiliary head **train만** 호출 | joint scene/trajectory flow predictor **train과 infer 모두** 호출 | 미래 latent predictor와 planner 연결을 기술 |
| 미래 target | downstream ego trajectory | scorer cache의 충돌/TTC 관련 최대2객체 corner/valid, ego area. Visual JEPA target 아님 | no_grad teacher의 미래 영상 patch latent, frozen/EMA target path; ego trajectory flow target | EMA visual target 및 ego 상태 감독 |
| 예측 미래→planner | 없음 | **없음**. pred_agents_states는 반환/감독만, final trajectory는 proposal score argmax | trajectory attention이 scene hidden K/V를 사용. 여러 inference step에서 예측 scene으로 다음 step 상태 갱신 | future injection 존재 |
| planning→predictor | 해당 모듈 없음 | trajectory L1/score BCE→shared backbone/refiner/feature. Auxiliary 미래 head 출력은 planning의 ancestor 아님 | trajectory flow MSE→scene hidden/projections 경로 있음. **최종 scene_out은 단일 training forward에서 직접 연결 아님** | Eq.(7) future injection detach: planning↛future predictor |
| 예측 전 선택 지점 | 새 predictor 추가 필요 | 새 미래 입력 branch/target 필요 | future noisy queries/condition/position를 joint blocks **전에** pack할 위치 있음 | 코드 미확인 |
| 미선택 예측 계산 생략 | 현재 경로에 대상 미래 예측 없음 | head/loss mask 제거는 원래 연구의 selective prediction 아님 | **원본은 dense 전량 처리**. Sparse shape/index interface 변경 필요 | 코드 미확인 |
| 공식 planning weight/eval | 이미 전체 평가 완료 | PB v1/v2 weights와 eval script 존재; 미재현 | 공개 stage2 weights + v1 PDMS/v2 EPDMS launcher/presets 존재; 미재현 | 이번 조사에서 공식 링크 확인 못함 |

### Drive-JEPA PB 파일·함수 근거

다음 상대 경로는 모두 공식 `navsim_v1/navsim/agents/drive_jepa_perception_based/` 아래다.

1. [`drive_jepa_model.py#L47`](https://github.com/linhanwang/Drive-JEPA/blob/548bb8215e3aae18e162a0f12f1ba83b4d3eb57e/navsim_v1/navsim/agents/drive_jepa_perception_based/drive_jepa_model.py#L47), `DriveJEPAModel.forward`:
   `camera_feature_2/_1`→`ImgEncoder`→`Traj_refiner`→`Scorer`→score argmax.
   Ego11 channels=current pose3/velocity2/acceleration2/command4.
2. [`bevformer/simple_image_encoder.py#L40`](https://github.com/linhanwang/Drive-JEPA/blob/548bb8215e3aae18e162a0f12f1ba83b4d3eb57e/navsim_v1/navsim/agents/drive_jepa_perception_based/bevformer/simple_image_encoder.py#L40), `ImgEncoder.forward`:
   input `[batch,2,3,256,512]`; patch `[batch,512,1024]`→projection256,
   flattened image feature `[1,512,batch,256]`.
3. [`traj_refiner.py#L26`](https://github.com/linhanwang/Drive-JEPA/blob/548bb8215e3aae18e162a0f12f1ba83b4d3eb57e/navsim_v1/navsim/agents/drive_jepa_perception_based/traj_refiner.py#L26), `Traj_refiner.forward`,
   [`bevformer/bev_refiner.py#L112`](https://github.com/linhanwang/Drive-JEPA/blob/548bb8215e3aae18e162a0f12f1ba83b4d3eb57e/navsim_v1/navsim/agents/drive_jepa_perception_based/bevformer/bev_refiner.py#L112), `Bev_refiner.forward`:
   proposals `[batch,32,8,3]`, proposal-point feature `[batch,256,256]`;
   proposal reference coordinates use `pose.detach()`. 256 query는 객체 256개가 아니다.
4. [`score_module/scorer.py#L48`](https://github.com/linhanwang/Drive-JEPA/blob/548bb8215e3aae18e162a0f12f1ba83b4d3eb57e/navsim_v1/navsim/agents/drive_jepa_perception_based/score_module/scorer.py#L48), `Scorer.forward`:
   `if self.training` 내부에만 `pred_col_agent/pred_area`.
   NAVSIM agent output은 `[batch,32,8,5,2,9]`, 즉 proposal별40시점×2collision slot×(corner8+valid1).
   현재 track 집합 전체의 instance 미래 latent가 아니다.
5. [`score_module/compute_navsim_score.py#L48`](https://github.com/linhanwang/Drive-JEPA/blob/548bb8215e3aae18e162a0f12f1ba83b4d3eb57e/navsim_v1/navsim/agents/drive_jepa_perception_based/score_module/compute_navsim_score.py#L48), `get_sub_score`:
   simulator/scorer의 future observation에서 collision/TTC offending track의 corner와 valid를 가져온다.
6. [`drive_jepa_agent.py#L102`](https://github.com/linhanwang/Drive-JEPA/blob/548bb8215e3aae18e162a0f12f1ba83b4d3eb57e/navsim_v1/navsim/agents/drive_jepa_perception_based/drive_jepa_agent.py#L102),
   `compute_score/score_loss/trajectory_loss_anchors/pad_loss/compute_loss`:
   proposals.detach→CPU numpy scorer, BCE score proxy와 trajectory L1, 미래 auxiliary loss를 더한다.
   **공식 PDMS가 미분 가능한 loss가 아니다.**
7. [`drive_jepa_features.py#L142`](https://github.com/linhanwang/Drive-JEPA/blob/548bb8215e3aae18e162a0f12f1ba83b4d3eb57e/navsim_v1/navsim/agents/drive_jepa_perception_based/drive_jepa_features.py#L142), `DriveJEPATargetBuilder.compute_targets`:
   ego trajectory와 token을 생성; agent/map target 코드는 호출되지 않는 주석 경로와 구분.
   Lightning `AgentLightningModule._step`은 agent.forward→agent.compute_loss→Lightning backward.
8. [`scripts/evaluation/eval_drive_jepa_perception_based.sh`](https://github.com/linhanwang/Drive-JEPA/blob/548bb8215e3aae18e162a0f12f1ba83b4d3eb57e/navsim_v1/scripts/evaluation/eval_drive_jepa_perception_based.sh):
   navtest/PB/latent=False. HF의 PB ViT-L v1 weight는3,735,352,913bytes. 공개 존재≠strict loading 검증.
   Training은 train metric cache/anchors score index와 MMCV deformable modules도 필요하다.

### WA-JEPA 파일·함수 근거와 중요한 gradient 예외

모델 파일은 [`models/multiview_causal_future_jepa.py`](https://github.com/AFARI-Research/WA-JEPA/blob/bec29660f5ea46ac73db8e2d0c33c8d1a72c23ad/models/multiview_causal_future_jepa.py).

| 함수 / 행 | 직접 읽은 내용 |
|---|---|
| `SceneInputPositioner` 498–604 | camera/time/spatial embedding. 현재 interface `[batch,views*local_tokens,512]`, index `[batch,local_tokens]`를 **모든 camera에 공유**. 독립 camera 선택에는 index interface 수정 필요 |
| `ModalitySpecificJointAttention.forward` 687–758 | trajectory query가 context/scene/trajectory K/V에 attention; `traj_loss_grad_to_scene_flow=true`면 scene K/V detach 안 함 |
| `JointFlowBlock.forward` 769–835 | context도 scene/trajectory와 같이 갱신; current-context bypass까지 합친 실제 그래프를 봐야 함 |
| `SceneTrajectoryFlowPredictor.__init__/forward` 1011–1193 | future token 수=views×future_steps×grid라는 고정 조건, dense shape 검사와 full future indices. 마지막 `scene_out(scene)`과 trajectory decode는 별도 출력 |
| `_teacher_targets/_target_future_scene` 1631/1754 | 미래 clip teacher/target projector no_grad; patch latent 감독, instance association 없음 |
| `_flow_predict_groups` 1799–1897 | 학습 때 teacher 미래와 ego GT를 noisy interpolation하여 denoising 입력으로 사용. 이는 **학습 전용 flow 입력**이며 online selector 관측에 넣으면 안 됨 |
| `_scene_flow_loss/_trajectory_flow_loss` 1899/1924 | scene MSE와 normalized ego trajectory MSE; official PDMS와 다른 학습 surrogate |
| `predict_trajectory` 2256–2347 | 현재/과거만 encoder; future zero-placeholder/full mask. Gaussian scene+trajectory→joint predictor4회→Euler 갱신, final ego path. no future GT |

**단일 training forward의 trajectory loss는 `scene_out`의 결과를 읽지 않는다.**
따라서 scene hidden K/V와 일부 공유/scene projection은 planning loss의 ancestor지만,
최종 scene output head까지 모두 planning gradient를 받는다고 쓰면 틀리다.
추론에서는 `scene_out`으로 갱신한 future state가 다음 step trajectory에 영향을 줄 수 있다.
이는 코드의 계산 의존성 확인이지 pretrained 모델의 실측 기여/gradient norm 확인은 아니다.

`scene_loss_grad_to_traj_flow=false`는 trajectory K/V 경계다. Shared context·time·joint modules까지
scene loss가 전혀 학습하지 않는다는 뜻이 아니다. 최종 모델 전체의 gradient 분리는 향후 autograd로 검증해야 한다.

기타 직접 확인:

- [`eval/navsim_agent.py`](https://github.com/AFARI-Research/WA-JEPA/blob/bec29660f5ea46ac73db8e2d0c33c8d1a72c23ad/eval/navsim_agent.py), `WorldModelFeatureBuilder.compute_features/WorldModelNavsimAgent.initialize/compute_trajectory`:
  native image resize→[-1,1]→model의 ImageNet normalization; ego_status8와 history trajectory4×3;
  GT ROI/우리 영상 보정/pilot planner는 사용하지 않는다.
- [`datasets/navsim_official_dataset.py#L393`](https://github.com/AFARI-Research/WA-JEPA/blob/bec29660f5ea46ac73db8e2d0c33c8d1a72c23ad/datasets/navsim_official_dataset.py#L393):
  **현재 stored command 누락 시 `infer_navigation_command(fut_poses,...)` fallback**.
  선택 실험에서는 이를 금지하고 누락을 명시적 unknown 또는 해당 current-input 결손으로 기록해야 한다.
- [`configs/wa_jepa_navsim_pdms.yaml`](https://github.com/AFARI-Research/WA-JEPA/blob/bec29660f5ea46ac73db8e2d0c33c8d1a72c23ad/configs/wa_jepa_navsim_pdms.yaml):
  full_mask_prob1.0, predictor12layer/512dim/8head, inference4step, trajectory gradient→scene true.
  `dynamic_collapse_topk=64`는 collapse **진단**이지 planning selector가 아니다.
- [`training/checkpoint.py#L251`](https://github.com/AFARI-Research/WA-JEPA/blob/bec29660f5ea46ac73db8e2d0c33c8d1a72c23ad/training/checkpoint.py#L251):
  내부 load_state_dict(strict=False) 뒤 key/shape diff 검사, 기본 strict=True에서 누락/추가/shape mismatch면 raise.
  다음 실행은 이 fail-closed 계약을 보존하고 key report만 프로젝트 outputs에 기록한다.
- [`docs/installation.md`](https://github.com/AFARI-Research/WA-JEPA/blob/bec29660f5ea46ac73db8e2d0c33c8d1a72c23ad/docs/installation.md):
  공개 stage2 checkpoint와 별도 Meta encoder weight를 NAVSIM model construction에 요구한다.
  v1.1 PDMS/v2.2 EPDMS config/launcher 존재. 우리 Drive fork/cache와 호환성은 미검증, 섞지 않는다.

### ForeDrive — 관련되지만 이번 구현 기반으로 확정 못 함

[원문 v2 §3/Eq.(7)/Appendix H](https://arxiv.org/html/2609.26299v2)는 future latent를 planner에
주입하지만 해당 연결을 detach하는 비대칭 학습을 기술한다. 따라서 planning-conditioned 선택기를
predictor 앞에 붙이기만 하면 planning gradient가 전달된다고 가정할 수 없다.
원문과 제한 검색에서 공식 code/weight 링크를 확인하지 못했다. 공개 구현 부재를 단정하지 않으며,
파일·함수·strict loading·실제 sparse 연산 항목은 **미확인**으로 남긴다.

## 4. 기반 대안과 변경 위험

| 안 | 재사용할 것 | 새로 필요한 것 | 가장 큰 위험 / 비용 |
|---|---|---|---|
| Drive-JEPA 확장 | 검증된 PF encoder/planner/scorer, 또는 PB proposal/refiner | 미래 predictor/target adapter/새 planner 연결/학습, entity이면 detection·association | 좋은 planning 성능을 새 branch에 보존할지 미확인. 자체 predictor 개발로 되돌아갈 위험; PB 추가 compiled deps/cache |
| **WA-JEPA native token 선택 — 추천** | 공개 encoder·scene/trajectory predictor·target encoder·eval agent | sparse query packing, camera별 position index, selector gradient adapter | full weight/config 미검증, dense→sparse distribution shift, ST 편향. 객체가 아닌 공간 tube 선택으로 범위 좁아짐 |
| WA-JEPA entity 선택 유지 | 위 모델 자산 | detection/GT association, entity→patch support 및 중복/이동 ROI 처리 | 기존 weights/positional grammar를 바꾸고 GT pilot 의존 증가; 첫 최소 변경안 아님 |
| ForeDrive 활용 | 문헌상 연결 구조 | 공식 구현 위치 확보 또는 재구현, gradient reroute | 실행 가능한 공개 자산 미확인. 현재 즉시 이전 추천하지 않음 |

추천 이유는 기존 투자 시간이나 논문 점수 순위가 아니라 **예측 미래가 이미 planning과 연결된 코드**다.
WA README의91.7 EPDMS는 공개 주장이고 우리 실측이 아니다. Drive PF89.224 PDMS와 protocol이 다른 숫자를 비교하지 않는다.

## 5. 최소 실험 계산 그래프 — 설계만, 미구현

### 선택 단위·고정 범위

첫 구현안은 **camera별 spatial patch-tube**: 4camera×16×32=2,048후보.
한 tube는 같은 image-grid 위치의 미래4tubelet(원본8frame/4초)을 모두 유지한다.
객체/BEV/track 단위가 아니며 움직이는 객체를 따라가는 ROI도 아니다.

사전 설계 예산은 camera당128tube, 총512tube→**2,048future token**.
Dense는 camera당512tube→8,192future token. 현재 context4,096token과 ego8point는 양쪽 모두 유지한다.
K/horizon은 고정하고 camera별 quota로 처음부터 camera를 버리는 confound를 막는다.
이 예산은 성능을 본 뒤 선택하지 않았으며, 실제 memory/profile로 실행 불가할 때만 결과 비교 전 protocol 재등록한다.

```text
현재/과거 4-view 영상 ── 기존 encoder/projector ── current_context[batch,4096,512]
                                              │
현재 ego 상태8 + command + 과거 ego path4×3 ──────┤
                                              ▼
                              current-only policy → camera별128개의 tube ID
                                              │
         canonical noise/position dictionary ──┤ pack BEFORE joint blocks
                                              ▼
current_context 유지 + selected_future_states[batch,2048,512] + ego_states[batch,8,3]
                   ── 기존 joint scene/trajectory predictor ──┬─ future scene update
                                                            └─ ego trajectory update
                   inference는 동일4step 반복 (미래 GT 입력 없음)

TRAIN ONLY: 미래 영상 → no_grad teacher → hard-ID gather → scene auxiliary target
            ego GT → training flow target/interpolation → trajectory surrogate loss
```

현재 feature는 `[batch,4camera,2history_tubelet,512spatial,512dim]`에서
각 현재 spatial 위치의 과거 요약+ego intent로 score를 만든다. 미래 visibility, future GT,
예측 오차, scorer 결과는 policy 입력/후보 유효성/예산에 넣지 않는다.
Official full future masking을 유지해 encoder의 visible-future conditioning으로 미래가 섞이는 것도 방지한다.

### 선택기 gradient 계약 — hard index만으로는 학습되지 않음

초기 후보는 기존 CPU graph의 **중복 없는 순차 ST**를 native tube ID에 적용하는 것이다.
그 코드를 WA에 붙인 상태가 아니며 다음은 검증해야 할 명세다.

- `hard_selection`: camera별 without-replacement, canonical ID tie-break, 고정128slot.
- `straight_through_selection = hard - stopgrad(soft) + soft`: hard-forward/soft-backward.
  한 softmax를128번 복제하지 않는다. Temperature1.0으로 첫 명세 고정, dynamic schedule/tuning 없음.
- Hard-ID gather만 하면 policy로 gradient가 안 간다. **기존 camera/time/spatial positional dictionary를
  ST weight로 pack하는 경로**를 predictor 입력에 둔다. Forward는 원래 selected ID 위치 embedding과 동일해야 한다.
- Future GT와 flow-interpolated noisy target은 detached hard-ID로 gather한다.
  Soft mixture에 teacher GT를 넣거나 target gather의 gradient로 policy를 학습시키지 않는다.
  Canonical per-scene/token noise를 공유해 정책이 난수 선택 차이만 이용하는 것을 줄인다.
  Scene noise를 짧게 뽑아 ego noise의 RNG 위치가 달라지지 않도록 ego noise도 동일하게 고정한다.
- Planning surrogate→trajectory attention→selected scene hidden/position→ST→policy.
  이는 **position-mediated 편향된 surrogate**다. 실제 대상 중요도를 학습한다는 보장은 없고 작은 학습 검사 필요.
- Scene auxiliary는 detached hard selection으로 별도 predictor forward를 사용해 policy 직접 gradient=0을 보장한다.
  Predictor/core에는 scene MSE가 전달되고 teacher에는0. 두 호출의 추가 **학습 비용**을 비교군에도 동일 적용한다.
  Official shared context/time 모듈과 trajectory-specific head를 구분하여 autograd 표를 실제로 확인한다.
- Trajectory surrogate는 official normalized ego flow MSE, scene MSE 가중치는 우선 official1.0/0.5 유지.
  Official PDMS/EPDMS scorer는 평가 전용이며 이를 미분 가능한 planning objective라고 부르지 않는다.
- `scene_out` 단일-forward planning gradient가0일 수 있는 원본 예외는 보존한다.
  Planning이 모든 future-predictor parameter를 직접 업데이트해야 한다는 잘못된 통과 조건을 만들지 않는다.

이 경로가 실제 모델에서 학습 불가능하면 soft GT routing·새 predictor로 자동 변경하지 않고 설계를 다시 검토한다.
전체 미래를 dense 예측한 뒤 선택하는 대조는 별도로 **planner bottleneck**이며 sparse prediction 실험과 구분한다.

### 필수 비교군 — 원본과 full-reference의 역할을 구분

| 이름 | 선택/학습 | 역할 |
|---|---|---|
| `official_unmodified` | 공개 weight, 원본 dense 추론, 수정/학습 없음 | 원본 동작·score 기준. Sparse fine-tune의 인과 대조와 혼동하지 않음 |
| `random_same_budget` | camera당128, 현재와 무관한 seeded uniform selection | 동일 token 예산 random 기준 |
| `fixed_rule_same_budget` | camera당128: grid의 row/column index가 둘 다 짝수인 spatial lattice | 결과와 무관한 동일 예산 fixed/규칙 기준. Ego-attention 규칙은 후속 하나의 별도 대조로만 검토 |
| `planning_conditioned_same_budget` | current patch features+ego command/status로128선택, planning surrogate로 학습 | H2 대상; fixed K/horizon만 |
| `all_future_reference` | 동일 sparse-compatible interface에서 camera당512, 같은 freeze/train scope와 update | Packed-all 원본 동등성 검사 후 대응 fine-tune 참조. 첫 행과 이름만 다른 중복 재평가 아님 |

공통 encoder/teacher를 첫 비교에서는 frozen으로 유지하고 pretrained projector/joint predictor를 동일 범위
fine-tune하는 후보안이다. 원본 row 외 비교군은 동일 초기 core state/batch order/noise/loss weights/steps,
같은 data split, 대응 최소3seed를 적용한다. 실제 update 수·scheduler·계산 상한은 profile 후 실행 전 등록한다.
Policy 추가 parameter 수/활성 수를 보고한다. Params가 같다고 planning-conditioned 추가 용량 confound가
자동 제거되는 것은 아니다. 이득을 주장하기 전 동일 policy의 ego-command 제거 대조로 의도 조건 효과를 확인한다.

`current_feature_delivery_control`은 같은 selected slot 수/차원·trajectory 모듈을 유지하고,
selected 현재 patch feature를 미래 시점에 전달하도록 **별도로 학습하는** 정보 경로 대조다.
Auxiliary만 끄는 비교와 같지 않으며 첫 five-condition 결과의 해석에 필요할 때 하나만 추가한다.
미래 zero/swap은 기존 joint 모델의 scene state를 perturb하는 **의존도 진단**이다.
이미 scene에 영향받은 context/ego hidden이 남을 수 있고 OOD/교환 입력 변화량 문제도 있으므로,
그 결과는 retrained no-future baseline이나 실제 유용성의 단독 증거가 아니다.

### 예산·데이터·평가 계약

- Token 수/차원/horizon은 운영 가능한 정보 예산 proxy다. 같은 token 수≠같은 정보량.
- Pack은 scene QKV/FFN/attention 전에 실행. Dense8192의 loss만 줄이면 **연산 절감 주장 불가**.
  Current encoder/teacher 계산은 dense로 남는다. Selector/packing/양방향 joint 비용까지 측정한다.
- Dense joint token12,296 vs sparse6,152라는 shape 감소는 설계상 수치다. 전체 latency/FLOPs/VRAM 절감률,
  GPU-hour는 아직 실측하지 않았고 이 숫자로 추정 결과를 보고하지 않는다.
- Train은 navtrain에 속하는 trainval; native recording `log_token` 기준 train/dev/최종holdout 분리를
  먼저 고정한다. 같은 recording의 exported segment를 서로 다른 split에 넣지 않는다.
- 기존 mini 반복 평가 recording은 새로운 최종holdout에서 제외한다. 이전3620manifest는 참고이지만
  4camera×history4/future8 파일 completeness를 확인하기 전 WA-ready dataset이라고 부르지 않는다.
- 현재 camera 입력 결손과 미래 teacher supervision 결손을 분리; future validity로 현재 후보/window를
  선별하지 않는다. 필요한 미래-label mask는 loss에만 쓰고 동일 availability로 비교한다.
- Camera4개 공식 coverage를 유지한다. Spatial후보2,048개가 객체2,048개 또는 interaction coverage를 뜻하지 않는다.
  View간 객체 중복, 시야 밖 물체, ego-motion/occlusion, fixed-image cell의 의미 변화를 한계로 기록한다.
- Official PDMS/하위 NC/DAC/TTC/EP/Comfort/DDC + 실패/누락 수; v2를 쓰면 EPDMS protocol을 별도로 고정.
  Current speed/command 기준 사전 상황 분해와 recording-level paired/cluster bootstrap, seed별 차이를 보고한다.
- Navtest는 이미 baseline을 확인한 진단/benchmark 데이터다. Selector fitting/하이퍼파라미터 선택에는 쓰지 않는다.
  새로운 독립 recording holdout이 최종 평가이고, navhard는 checkpoint/devkit/scorer/cache 버전 gate 후 별도 계획.

## 6. 변경 예정 함수·비용·진행/중단 gate

다음은 **변경 예정**이지 이번 변경이 아니다. Official reference tree는 읽기 전용으로 보존하고 별도 worktree에서 한다.

| 파일 / 함수 | 최소 변경 후보 | 통과 조건 |
|---|---|---|
| WA model `SceneInputPositioner._position_embeddings/forward` | camera별 canonical ID와 packed selected position 지원 | all-ID가 원본 출력과 동일; camera/time 위치와 순서/padding 검사 |
| `SceneTrajectoryFlowPredictor.__init__/forward` | fixed8192 shape→명시 selected count/IDs; selected states만 처리 | 실제 QKV/FFN shape가 줄어듦, 기존 weight key/shape 보존 |
| `_flow_predict_groups/_scene_flow_loss` | hard-ID target/noise gather, 동일 loss 분모, aux policy detach 호출 | 누출/gradient 계약, 유효 감독 count 동일 |
| `predict_trajectory` | current-only ID 선택→4step selected state update | no future GT, fixed seed/current-only deterministic, finite ego path |
| 새 `FutureSpatialTubeSelector` 모듈 | current feature/intent→quota 내 ST 선택 | policy gradient 존재 + 작은 선택 학습, 아직 구현 안 함 |
| `NavsimOfficialDataset.__getitem__` adapter | future-derived command fallback 금지, provenance/availability 기록 | online inputs만 policy에 사용; 공용 원본 변경 없음 |
| evaluation/config/runner | original vs matched-reference, paired seeds/splits/cost/scene completeness | 원본 preprocessing/scorer 보존, tuning split 제한 |

실측/확정 비용: 이번 CPU 상황 분석16.29초, GPU 사용0/학습0; 추가다운로드는 작은 API metadata뿐.
WA stage2 weight1,575,763,741bytes는 **공개 용량**, 다운로드하지 않았다. NAVSIM preset construction은 별도
Meta encoder도 요구한다. 그 파일/환경 추가 용량과 RTX A6000 48GB의 full forward/gradient peak 비용은 미측정이다.
Full-model peak/latency와 sparse profile로 확인할 값이지 README 수치를 가져와 확정 GPU-hour를 쓰지 않는다.

진행 순서(이번에는 실행하지 않음):

1. 준성: native spatial-tube라는 **범위 축소**를 승인하거나 객체 instance 유지안을 선택.
2. 승인 후 WA 전용 Conda/worktree/정확 config/source/weights strict loading과 소수 **train/dev** 장면 공식 동작 확인.
   v1.1/v2.2와 기존 Drive fork/scorer/cache 호환성을 먼저 대조. 미호환 cache 재사용/새 navhard 평가 금지.
3. Sparse-compatible **all-ID equivalence**부터 검증. Weight key/shape를 임의 제외해 통과시키지 않음.
4. 실제 gradient/teacher leak/packing/작은 선택 학습/비용 gate 후 five-condition 실행 계획 상한 등록.
5. 공식 planning 성능을 보존할 수 있는지 통제 비교. 실패하면 기반/adapter 판단을 재검토하며 H1/H2를 자동 기각하지 않음.

중단/보고 조건: strict weight/config 불일치, 4-view current 입력 부족, GPU0·1 메모리/비용 실행 불가,
packing-all 동등성 실패, planning→selector 단절 또는 GT leak. 이런 조건을 숨기기 위해 새 predictor나
목표/target을 바꾸지 않는다. Dynamic K/horizon, 확률적 predictor, 자체 pilot 확대는 계속 보류한다.

## 7. 이번 실제 실행과 검수 진입점

- [현재 입력 기반 navtest 현황](official_navtest_current_context_summary.md): 전체12146scene, metadata 누락0.
- [CPU 요약 script](../scripts/summarize_official_navtest_current_context.py), [사전 bin 정의](../configs/analysis/official_navtest_current_context_v1.json).
- [집계 JSON](../results/foundation_selection/current_context_navtest_v1/context_summary.json),
  [scene별 현재 metadata](../results/foundation_selection/current_context_navtest_v1/current_input_metadata.csv).
- 새7개 현재-context tests + 기존5개 공식 집계 tests 통과. 이는 분석/무결성 검사이고 새 모델 검증이 아니다.
- 공식 Drive PF 결과와 source/weights/config/scorer를 수정하지 않았다. 새 환경/대용량 다운로드/GPU/학습 없음.

준성의 결정: **추천 WA native spatial-tube 선택으로 시작** vs **객체 instance 가설을 엄격히 유지하고
더 큰 entity adapter 변경을 수용**. 양쪽 모두 현재 관측·ego 의도→같은 예산의 planning 유용성을 검증한다.
Native token 결과를 객체 선택 성공이나 독자적 novelty 확립으로 확대 해석하지 않는다.
