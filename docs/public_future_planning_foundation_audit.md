# 공개 future-planning 기반 감사와 다음 선택 실험

이 문서는 `5c6e6d5` 당시 조사/계획 이력이다. `578be6e` 이후의 새 PB/WA 코드 감사,
single-forward scene_out gradient 예외와 현재 명세는
[최신 기반 결정](future_prediction_foundation_decision.md)을 먼저 읽는다. 과거 계획을 자동 실행하지 않는다.

## 결정 요약

최신 사용자 지시에 따라 predictor 개발·대규모 pilot 확대 대신 **이미 미래 예측과 planning이 연결된
공개 모델의 재현→선택·예산 가설 검증**을 우선한다. 현재 추천은 WA-JEPA의 official checkpoint
동작을 먼저 재현하는 것이다. 모델 이동, 새 flow matching 구현, multi-view 우리 모델 구현 또는
최종 selection 단위 변경을 이미 완료했다고 보고하지 않는다.

## 확인 수준

| 후보 | 원문/README | 직접 읽은 코드 | 실제 실행 | 현재 활용 판단 |
|---|---|---|---|---|
| WA-JEPA | 공개 future-action model·공식 점수 주장 | joint inference, training loss, gradient flag, dense shape, feature/eval loader | official attention의 tiny CPU backward만 | **첫 재현 후보**, full loading/평가 미확인 |
| ForeDrive v2 | future latent→planner, asymmetric gradient, horizon ablation | 공식 구현 위치 확인 못함 | 없음 | 직접 관련 연구·대조 설계 참고, 당장 실행 기반으로 확정 못함 |
| Drive-JEPA | 사전학습/공식 planning | 기존 감사 재사용: perception-free 및 perception-based 경계 | 우리 frozen encoder 로딩/기존 pilot 연결 | encoder 자산. 조사한 downstream에 명시적 future latent→planner가 없어 최소 수정 기반으로 덜 적합 |
| 현재 pilot | 우리 진단 모델 | GT ROI+track+새 planner, gradient 계약 | 연결·조건부 target 학습·이번 bounded 비교 | 보존할 engineering fixture, 주 성능 기반 튜닝 중단 |

## WA-JEPA: source에 직접 확인한 경로

Official repository: [AFARI-Research/WA-JEPA](https://github.com/AFARI-Research/WA-JEPA).
이번 clone source는 **`bec29660f5ea46ac73db8e2d0c33c8d1a72c23ad`**다.
아래 파일·함수는 이 commit 기준이며 README의 성능 수치를 독립 재현한 것은 아니다.

1. [`eval/navsim_agent.py`](https://github.com/AFARI-Research/WA-JEPA/blob/bec29660f5ea46ac73db8e2d0c33c8d1a72c23ad/eval/navsim_agent.py):
   `WorldModelFeatureBuilder.compute_features`→`WorldModelAgent.compute_trajectory`→`model.predict_trajectory`.
   4 history×4camera(`cam_l0/f0/r0/b0`) 이미지, ego command/velocity/acceleration, 과거 ego trajectory를 만든다.
   Image builder는 저장 JPEG를 직접 resize하여 [-1,1]로 바꾸며 pilot의 GT-ROI undistortion 경로를 쓰지 않는다.
2. [`models/multiview_causal_future_jepa.py`](https://github.com/AFARI-Research/WA-JEPA/blob/bec29660f5ea46ac73db8e2d0c33c8d1a72c23ad/models/multiview_causal_future_jepa.py):
   `MultiViewCausalFutureMaskedJEPA.predict_trajectory`(2257행)는 현재/과거를 encode하고
   future image zero-placeholder와 full future mask를 만든다. **미래 GT 영상 없이** noisy future scene과
   noisy ego trajectory를 `SceneTrajectoryFlowPredictor`에서 함께 갱신한다. Official NAVSIM preset은
   4 inference step이다. 별도로 미래를 모두 예측한 후 작은 planner를 붙이는 우리 pilot 구조와 다르다.
3. `SceneTrajectoryFlowPredictor.forward`(1113행)의 joint attention에서 trajectory query가 scene K/V를 사용한다.
   `ModalitySpecificJointAttention.forward`(687행)는 `traj_loss_grad_to_scene_flow`가 false일 때만
   해당 scene K/V를 detach한다. Preset은 **true**, 반대방향 `scene_loss_grad_to_traj_flow=false`다.
4. Training `_flow_predict_groups`(1799행)→joint predictor→`_trajectory_flow_loss`(1924행)의 ego target MSE다.
   Train의 future target/flow interpolation과 inference의 GT 없는 initialization을 구분한다.
   `predict_trajectory`의 `@no_grad`는 inference wrapper이며 training gradient 차단이라는 뜻이 아니다.
5. `dynamic_collapse_topk=64`의 top-K는 `_scene_token_collapse_metrics`(2031행)의 **진단 대상 선정**이다.
   Planning-aware predictor target selector나 동적 inference budget으로 해석하지 않는다.

실행한 [official attention CPU 검사](../results/pilot_foundation_decision/official_wa_jepa_attention_gradient.json)는
직접 upstream class에 작은 tensor를 넣었다. `traj_loss_grad_to_scene_flow=true`일 때 trajectory loss→scene
input norm0.01519 / scene QKV norm0.11358, false일 때 둘 다0이다. Full-model loss→새 selector 검증,
official pretrained encoder/weight loading 또는 성능 재현을 대신하지 않는다.

### 선택 단위와 실제 계산 절감

Preset: 256×512, patch16, tubelet2, history4/future8, 4camera. 따라서 spatial512×history2×view4=
**context4096**, spatial512×future4×view4=**future8192** scene token이며 ego 미래8point와 joint 처리한다.
여기서 time token은 0.5초 원본 frame과 동일하지 않고 2frame tubelet 단위다.

Native 단위는 **view·공간 patch·시간 tubelet**이며 객체 instance/track이 없다.
공식 future mask sampler는 가려진 관측/학습 문제를 구성하지만, `SceneTrajectoryFlowPredictor.forward`는
`num_scene_tokens`와 두 future tensor의 길이가 같음을 assert하고 **전체8192 token을 처리한다**.
Loss mask만 줄이거나 dense prediction 뒤 planner 정보만 제거하면 predictor 연산 절감을 주장할 수 없다.

최소 sparse adaptation 후보(아직 미구현):

- 현재 encoder context와 ego intent에서 **future query index**를 선택한다. 현재 관측은 유지한다.
- Future noisy state / condition / positional index를 선택된 index로 함께 pack한다.
- 고정8192 shape assertion과 positional indexing, group/padding/valid mask, 출력 복원 인터페이스를
  변경하되 projection·attention·trajectory head weight는 가능한 재사용한다.
- Train-only target encoder의 미래 latent를 동일 index로 gather하여 감독한다. Selector에는 미래 target,
  future visibility, future GT importance를 입력하지 않는다.
- 실제 batch/행별 padding 길이, predictor attention 비용, encoder/selector overhead, 전체 latency까지 측정한다.
  Token 수가 같다는 것이 정보량이 완전히 같다는 보장은 아니다.

**위 adaptation은 단순 flag 하나가 아니며 작동/성능/절감이 확인되지 않았다.**
객체 단위를 고수하면 추가 association/entity-to-patch adapter가 필요하다. Spatial/time token 선택을 먼저
연구하는 것은 원래 질문의 구현 가능한 하위 문제지만 객체 선택과 동일한 주장으로 바꾸면 안 된다.

### Checkpoint와 실행 가능성

- 공식 [Hugging Face release](https://huggingface.co/AFARI-Research/WA-JEPA), revision
  `15c0770ebd233665214590bb2190a90907499f9e`의 API/tree를 직접 조회했다. Public/ungated,
  `model_state_dict.pt` **1,575,763,741bytes**, `state.pt`7073bytes와 LFS SHA256 확인.
  [메타데이터 JSON](../results/pilot_foundation_decision_metadata/official_wa_jepa_checkpoint_metadata.json).
  Weight 자체는 다운로드/strict load하지 않았다. 공개 파일 존재와 checkpoint 완전성·호환성은 다르다.
- [`docs/installation.md`](https://github.com/AFARI-Research/WA-JEPA/blob/bec29660f5ea46ac73db8e2d0c33c8d1a72c23ad/docs/installation.md)
  및 official NAVSIM presets는 construction에 Meta `vjepa2_1_vitl_dist_vitG_384.pt` 경로도 요구한다.
  기존 Drive-JEPA weight를 이름만 바꿔 재사용하지 않는다. 별도의 isolated environment/commit pin과
  loader key/shape 검증이 필요하다.
- NAVSIM v1.1 PDMS/v2.2 EPDMS 평가 entrypoint와 script는 존재한다. NAVSIM 의존성(numpy1.23.4 등)과
  현 CPU pilot Python3.12 환경이 그대로 호환된다고 가정하지 않는다. 기존 환경을 업그레이드하지 않는다.
- GPU0/1은 RTX A6000 약48GB. 조사 종료 시25MiB/compute process 없음이지만 실행 직전 다시 확인한다.
  **Full model VRAM·throughput·공식 score는 실측하지 않았다.** README의 파라미터/속도 주장을 실측으로 표기하지 않는다.
- Official 모델 자체의 4-view 재현은 upstream 동작 확인이지 새 multi-view 모델 구현이 아니다.
  Front-only로 checkpoint 구조를 축소한 결과를 official reproduction이라고 부르면 안 된다.

## ForeDrive와 Drive-JEPA의 제한된 비교

[ForeDrive v2 §3/Appendix F/H](https://arxiv.org/html/2609.26299v2)는 현재 front image의 patch에서
다중 horizon latent와 ego status를 예측하고 planner에 주입한다. Eq.(7)/Appendix H의 injected future
detach 때문에 planning→predictor가 차단된다. Selector 위치가 detach 앞이면 함께 차단될 수 있으므로,
planning으로 target selection을 학습하려면 gradient 경계를 다시 설계해야 한다. Horizon 구성을 비교한
실험은 상황별 학습된 예산 배분을 입증한 실험이 아니다.
이번 제한 검색과 arXiv 원문에서 공식 code/checkpoint 링크를 확인하지 못했다.
**공개 구현이 존재하지 않는다고 단정하지 않는다.** 재현 구현부터 새로 작성하는 우선순위는 낮다.

[기존 Drive-JEPA source 감사](baseline_and_target_adapter_audit.md)는 commit
`548bb8215e3aae18e162a0f12f1ba83b4d3eb57e` 기준이다. Perception-free encoder→trajectory,
perception-based proposal refinement와 train-only object-motion head를 조사했다. 그 head의 미래
출력이 planner 입력이라는 경로는 없었다. 이번에 공식 전체 model을 재감사/재평가하지 않았다.

## 다음 실행: 공식 재현 후 세 가설로 복귀

**0. Official reproduction gate** — 별도 env/reference checkout에 source/weight/config를 pin하고
strict-load report, 동일 입력 결정론, 실제 NAVSIM train/development history batch의 finite trajectory,
frame/unit/time, latency/VRAM을 확인한다. 새 held-out과 navtest를 configuration 튜닝에 쓰지 않는다.
공식 benchmark 비교는 전처리·평가 설정을 고정한 뒤 별도 실행하며 공개 논문 수치를 우리 결과로 복사하지 않는다.
이 단계는 미래 예측 가치를 처음부터 증명하는 긴 연구가 아니라 **기반이 공식대로 동작하는지 확인**하는 단계다.

**1. 상황별 차이** — 같은 backbone/현재 context/동일 future token 예산에서 fixed/random/intent-rule
query 구성을 비교한다. 현재 입력에서 정의한 축·recording split을 사전에 고정하고 맥락별 차이를
paired recording 분석한다. Dense 미래를 계산한 뒤 제거하는 검사는 저비용 의존도 진단으로만
기록하고, 실제 선택적 예측 효율의 증거로 쓰지 않는다. 단순 추론 제거의 OOD 실패는 matched
fine-tuning 대조 없이 정책 우열로 확정하지 않는다.

**2. 학습형 선택** — 현재 관측·ego intent selector를 같은 budget에서 위 비교군과 비교한다.
Training planning loss→selector의 실제 backward를 검사하고 강한 current-only bypass로 미래가
무시되는지도 검사한다. 동일 frozen/fine-tuned 모듈·추가 파라미터·학습량과 대응 seed를 통제한다.

**3. 예산 배분** — 먼저 여러 고정 budget의 맥락별 marginal benefit을 측정한 뒤,
**동일 평균 token/연산 예산**의 context-conditioned 배분과 고정·입력 독립 allocation을 비교한다.
실현 budget 분포·encoder/predictor/selector 비용·공식 안전/진행 지표를 함께 보고한다.
동적 K/horizon method는 아직 구현하지 않는다. 위험은 routing collapse, attention/position/index 오류,
미선택 정보의 안전 관련 손실, 정보량/연산량 confound다.

Branch 없음/잘못된 future/identity future는 해석용 대조로 유지하되 독립적인 긴 target 튜닝 과제로
늘리지 않는다. Reproduction gate가 실패하면 compatibility/cost를 구체적으로 보고하고 다른 공개 기반을
검토한다. 새로운 predictor를 잘 만드는 것이 필수 기여는 아니다.

## 이번 종료 시 권고

사용자 옵션 **2(공개 모델 기반 우선)**를 권고한다. WA-JEPA가 가장 먼저 재현할 후보이며
최종 선택은 full checkpoint/eval/비용 gate 뒤 확정한다. Option3의 spatial/time token은 native 구현 후보로
검토하되 객체 가설의 검증 성공으로 일반화하지 않는다. 현재 pilot 확대 재개를 공식 재현의 선행 조건으로 두지 않는다.
