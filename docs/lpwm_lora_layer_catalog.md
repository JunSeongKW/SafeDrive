# LPWM의 LoRA 적용 후보와 현재 planning 경로

## 실행 상태

2026-10-06 최신 사용자 지시: **계층 설명을 먼저 읽고 사용자가 적용 대상을 선택한다.**
기존 attention LoRA의 첫 epoch를 보존·중단했으며, 새 geometry LoRA 본학습·queue·monitor는 시작하지 않았다.
Geometry head 코드는 후보 구현이다. 실제 영상/official planning loss 검사2update와 DDP2update만 완료했고,
검사용 가중치는 본학습에 사용하지 않는다. `outputs/lpwm_drivor_geometry_lora_v1/pause.requested`를 유지한다.

## 기준 코드와 목록 범위

공개 Sketchy checkpoint를 strict loading한 현재 모델 객체를 순회했다. 원래 weight map 기준
**Linear234개 / Conv2d70개**가 있으며, 적용 중인 Q/V adapter의 원래 Linear는 한 번만 센다.
실제 모듈 전체 경로·weight shape·원래 파라미터 수는
[CSV](../results/lpwm_drivor_geometry_lora_v1/linear_convolution_inventory.csv)와
[JSON](../results/lpwm_drivor_geometry_lora_v1/linear_convolution_inventory.json)에 있다.
이 개수는 모든 map에 adapter를 붙였다는 뜻이 아니다.

기준: `reference_repositories/LPWM/modules/modules.py`, `models.py`,
`src/planning_aware_future_prediction/object_centric/lpwm_drivor_joint.py`.
원본 public model의 옵션에 따라 생성되지 않는 계층도 있으므로 다른 LPWM 설정에 개수를 일반화하지 않는다.

## 요청한 세 head와 영상 CNN

현재 head의 공통 경로는 `encoder_module.particle_enc.particle_attribute_enc`다.

1. `xy_head`: attribute CNN의2048D특징→256D→4D. 마지막4개 출력은 위치 offset 평균2개와
   log-variance2개다. 최종 현재 위치는 prior CNN이 만든 `z_base`에 offset을 더한다.
   현재 deterministic 추론/학습은 평균을 사용한다. 좌표는 영상의 정규화 좌표이며 차량의 월드 위치나 ego 궤적이 아니다.
2. `scale_xy_head`: 같은2048D특징→256D→4D. 크기의 latent 평균2개와 log-variance2개를 만든다.
   scale prior와 평균을 합하고 sigmoid로 영상 대비 glimpse 크기로 변환한다. 이 크기는 이후 appearance CNN이
   crop할 영역과 재구성 배치에 쓰인다. 정답 객체 bbox와 같아야 한다는 감독은 없다.
3. `obj_on_head`: 같은2048D특징→256D→1D. 출력을 두 Beta 분포 파라미터로 변환하며,
   deterministic 경로는 Beta 평균을 presence로 사용한다. 이는 차량/보행자 클래스 확률이나 주행 중요도 점수가 아니다.
4. `particle_attribute_enc.cnn`: 위 head가 판단할 지역 RGB 특징을 추출한다. Conv-LoRA를 넣으면
   출력 해석뿐 아니라 head에 전달되는 시각 특징도 바꿀 수 있다. 여기서 말하는 CNN과 particle 외관을 인코딩하는
   `particle_features_enc.cnn`은 서로 다른 모듈이다.

이 세 head의 두 Linear에 추가하는 후보는 hidden rank8, 출력 rank4/4/1, alpha=rank다.
작은 마지막 Linear는 출력 차원만큼 rank를 쓰므로 엄격한 저랭크 제약의 이득이 작다.
더 큰 첫 Linear에서 파라미터 절감이 크며, 최종 작은 Linear에도 원본 고정·zero-init residual이라는 장점은 남는다.

## 전체 적용 후보 — 역할별 지도

아래의 'planning 경로'는 현재 단일 관측 영상→prior context→미래 rollout→DrivoR planner 연결을 기준으로 한다.
모든 후보에 실제 adapter를 넣어 gradient를 실측한 것은 아니다. 직접 실측한 신규 후보는 위 세 geometry head다.

| 영역 | 실제 경로 예시 (world model 내부) | 하는 일 / 바꿀 수 있는 것 | 방식·현재 상태 |
|---|---|---|---|
| 위치 초깃값 CNN | `encoder_module.particle_enc.prior_encoder.enc` | patch별 위치 제안 `z_base` 생성 | Conv-LoRA 후보; 현재 고정 |
| Geometry용 CNN | `encoder_module.particle_enc.particle_attribute_enc.cnn` | 위치·크기·presence head의 지역 영상 특징 | Conv-LoRA 후보; 현재 고정 |
| 현재 위치 head | `...particle_attribute_enc.xy_head` | 현재 위치 offset | Linear LoRA 후보 구현·연결검사 완료, 본학습 미실행 |
| 현재 크기 head | `...particle_attribute_enc.scale_xy_head` | 현재 glimpse 크기 | 위와 동일 |
| 현재 presence head | `...particle_attribute_enc.obj_on_head` | particle 활성/presence | 위와 동일 |
| 지역 외관 CNN/MLP | `encoder_module.particle_enc.particle_features_enc.cnn`, `.to_mu` | 결정된 위치·크기로 crop한 영역의 외관 정보 | Conv/Linear LoRA 후보; 좌표 직접 출력 경로는 아님 |
| 배경 CNN/MLP | `encoder_module.bg_encoder.bg_cnn_enc`, `.to_mu` | 배경/장면 전체 특징 | Conv/Linear LoRA 후보 |
| Interaction의 영상 context CNN | `encoder_module.particle_inter_enc.ctx_cnn_enc`, `.to_latent` | 전체 영상 정보를 particle 상호작용에 제공 | Conv-LoRA 후보; `.to_latent`는1×1Conv |
| 현재 속성 projection | `encoder_module.particle_inter_enc.basic_particle_proj.*` | 위치·크기·presence·외관 등을 attention token으로 변환 | Linear LoRA 후보 |
| Particle interaction attention | `encoder_module.particle_inter_enc.pte.blocks.*.attn` | 현재 particle끼리 정보를 교환 | Q/V는 기존 적용; K/출력`proj`도 후보 |
| Particle interaction FFN/출력 | 같은 block의 `.mlp.fc_1`, `.mlp.proj`, `pte.head` | 상호작용 후 특징 가공 | Linear LoRA 후보 |
| 현재 depth/feature 출력 | `encoder_module.particle_inter_enc.particle_decoder.depth_head`, `.features_head`, `.bg_backbone`, `.bg_features_head` | 현재 depth latent·외관·배경 특징 | Linear LoRA 후보; 이 구성에서는 interaction이 현재xy/scale을 다시 출력하지 않음 |
| Context 입력 projection | `ctx_module.basic_particle_proj.*` | particle 속성을 context 모델 입력으로 변환 | Linear LoRA 후보 |
| Context attention/FFN/조건변환 | `ctx_module.pte.blocks.*.{spatio_block,temp_block}`의 Q/K/V/proj, MLP, `c_proj`; `pte.head` | particle·시간 관계를 통해 미래 context를 계산 | Q/V는 기존 적용; 나머지 활성 Linear도 후보 |
| Context prior 출력 | `ctx_module.prior_decoder.context_head` | 관측/예측 상태만으로 rollout용 context 분포를 출력 | Linear LoRA 후보; 현재planning 경로에서 사용 |
| Context posterior 출력 | `ctx_module.posterior_decoder.context_head` | 관측 시퀀스에서 posterior context를 추론 | Linear LoRA 가능; 현재planner는 이 결과를 사용하지 않아 별도loss/연결 없이는 학습 신호 없음 |
| Dynamics 입력·조건 projection | `dyn_module.particle_projection.*`, `.context_proj` | 현재 particle과 context를 미래 예측 token/조건으로 변환 | Linear LoRA 후보 |
| Dynamics attention/FFN/조건변환 | `dyn_module.particle_transformer.blocks.*.{spatio_block,temp_block}`의 Q/K/V/proj, MLP, `c_proj`; transformer `head` | 미래 particle의 공간·시간 상호작용과 context 반영 | Q/V는 기존 적용; 나머지 활성 Linear도 후보 |
| 미래 위치·크기·presence head | `dyn_module.particle_decoder.offset_xy_head`, `.scale_xy_head`, `.obj_on_head` | 미래 particle geometry | Linear LoRA 후보; 현재 위치 head와 다른 모듈 |
| 미래 depth·외관·배경 head | 같은 decoder의 `.depth_head`, `.features_head`, `.bg_backbone`, `.bg_features_head` | 미래 particle 정보 및 배경 latent | Linear LoRA 후보 |
| RGB foreground decoder | `decoder_module.particle_dec.from_latent_lin`, `.cnn` | particle 특징을 RGB/alpha로 복원 | Linear/Conv-LoRA 가능; 현재planning-only forward에서 미사용 |
| RGB background decoder | `decoder_module.bg_dec.latent_to_feat_map`, `.cnn` | 배경 RGB 복원 | 위와 동일; 재구성loss 등을 연결해야 학습됨 |

`ctx_module`, `encoder_module.ctx_enc`, `dyn_module.context_decoder`는 이 모델에서 같은 context 객체를 공유한다.
`prior_module`도 encoder prior의 별도 이름이다. 이름이 여러 개여도 중복 adapter를 삽입하면 안 된다.
현재 depth는 재구성용 latent depth이므로 metric3D거리나 detection 정답으로 자동 해석하지 않는다.

## 활성화 옵션·비행렬 계층·LPWM 외부 모듈

- 현재 checkpoint는 native action/language/image-goal conditioning 경로를 사용하지 않는다.
  다른 설정의 `action_proj`, `lang_proj`, `goal_proj`, cross-attention도 Linear LoRA 후보지만,
  현재 존재하지 않는 경로를 켜는 일은 기존 weight에 LoRA를 붙이는 것과 다르다. 새 입력·초기화·학습 설계가 필요하다.
- LayerNorm/RMSNorm의 scale·bias, bias-only, position/particle/view embedding을 조정하는 방법도 있다.
  Norm/bias tuning은 별도의 미세조정 방식이며, embedding에는 전용 저랭크 delta를 설계할 수 있다.
  현재 구현의 Linear LoRA wrapper를 이런 파라미터에 그대로 씌우지는 않는다.
- ReLU/GELU/SiLU, sigmoid/softmax, pooling, spatial transform 자체에는 LoRA로 분해할 학습 weight가 없다.
  이 연산 앞뒤의 Linear/Conv를 적응시킨다. Top-k 인덱스·명시적 detach 경로를 LoRA가 자동으로 미분 가능하게 만들지는 않는다.
- 확률분포 head의 log-variance 행은 deterministic 경로에서 사용하지 않거나 confidence용으로 detach될 수 있다.
  분산 출력까지 유의미하게 학습하려면 별도의 확률/uncertainty 목적을 검토해야 한다.
- 우리가 추가한 command FiLM, `particle_trajectory_projection`, camera embedding은 원래 LPWM 구성과 구분한다.
  현재 새 모듈은 전체 학습하며, planner도 전체 학습한다. 공개 사전학습 weight가 없는 새 모듈에 LoRA만 쓰는 것은
  기존 LPWM 적응과 다른 제약이다.

## 선택할 때 구분할 질문

| 원하는 변화 | 직접적인 후보 |
|---|---|
| 현재 particle의 위치·크기·presence | 현재 geometry head, 필요시 attribute/prior CNN |
| 같은 위치에서 더 유용한 정보를 추출 | appearance CNN·feature head, interaction Q/K/V/proj·FFN |
| 미래의 위치·상태·상호작용을 개선 | context prior·dynamics projection/attention/FFN/미래 출력 head |
| 이미지 복원도 함께 적응 | RGB decoder와 재구성 목적의 연결 |

LoRA는 원래 행렬을 고정한 채 `W0 + (alpha/r) B A`의 A/B를 학습한다.
CNN에도 kernel의 저랭크 residual을 구성할 수 있지만, 일반 Linear wrapper와는 별도 구현이 필요하다.
가중치 고정은 activation gradient를 끊는 `no_grad`/detach와 다르다.
원래 layer가 고정돼 있어도 그 앞의 trainable module이나 LoRA까지 gradient가 흐를 수 있다.
근거: [LoRA 원논문](https://arxiv.org/abs/2106.09685),
[LPWM 공식 모듈](https://github.com/taldatech/lpwm/blob/main/modules/modules.py).

현재 **어느 후보를 다음 본학습에 쓸지는 미확정**이다. 사용자의 계층 선택 이후 새 설정을 확정하고 등록한다.
