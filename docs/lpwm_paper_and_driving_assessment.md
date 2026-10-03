# LPWM의 주행 객체 표현 적용 검토

## 연구 질문과 판단 기준

현재 연구의 하위 질문은 **선택할 만한 객체 단위가 encoder에서 실제로 만들어지는가**다.
이 단위가 확보되어야 현재 맥락과 ego intent에 따라 미래 정보의 대상·시간·예산을 선택하는
실험을 해석할 수 있다. 기존 Drive-JEPA의 patch 선택 결과를 객체 선택 결과로 재해석하지 않는다.

LPWM은 이 질문을 검사할 가치가 있다. 위치와 크기를 직접 조작하고, 영상 위에서 표현을 확인하며,
표현 학습과 미래 dynamics를 함께 갱신할 수 있기 때문이다. 그러나 복원에 필요한 시각적 부분이
planning에 필요한 객체와 일치하는지는 별도 가설이다. 이번에는 객체 표현 적응을 검사하고,
Drive-JEPA planner에 연결하거나 PDMS 개선을 주장하지 않는다.

## 읽은 자료와 고정한 구현

- [LPWM 논문](https://arxiv.org/abs/2603.04553), v1, 본문과 부록 포함 49쪽.
  방법 A.2–A.4, 정책 A.5, 관련 연구 A.6, 데이터·비교군 A.7–A.8, 학습 A.9,
  추가 실험 A.10을 확인했다. 저자 공개 PDF는 로컬 literature 폴더에 보존했다.
- [공식 코드](https://github.com/taldatech/lpwm), 기본 `main`의
  `4cf53c403433e64c01652ac2adbec66231a46dea`, MIT license.
- 저자가 연결한 Sketchy checkpoint와 동봉 `hparams.json`을 사용한다.
  공개 MEGA 파일의 MAC 검증을 통과했고 ZIP SHA256은
  `979a0cee748c67bf9f6dda3a86b5caffca1f2ba204d54ba483189bcdfcf4454f`다.
- OpenReview 토론 페이지는 브라우저 검증 화면으로 열리지 않았다. 리뷰 내용을 확인했다고 주장하지 않는다.

## 논문에서 우리 연구에 직접 영향을 주는 내용

LPWM은 DLP 계열의 위치·크기·외형·투명도·합성 순서 표현에 입자별 latent action과 dynamics를
결합한다. Encoder는 프레임별로 병렬 처리한다. 입자는 원래 patch 근처에 머물며 객체 정보가
이웃 입자로 전달되는 구조여서, 입자 번호를 객체 ID로 해석할 수 없다. `depth`는 물리적 거리가
아닌 합성 순서다. A.3–A.4를 근거로 한 해석이다. [논문](https://arxiv.org/html/2603.04553v1)

학습에서는 미래를 본 inverse head와 과거를 보는 latent policy를 KL로 맞춘다. 영상 복원과
시간적 KL이 encoder까지 갱신한다. 논문의 일부 PSNR·LPIPS 평가는 미래에서 얻은 latent action을
조건으로 사용하며, prior에서 생성하는 평가는 구분된다. 따라서 그 수치를 NAVSIM의 과거 입력만
사용한 예측 정확도로 옮길 수 없다. A.4.6·A.9·A.10.2를 확인했다. [논문](https://arxiv.org/html/2603.04553v1)

정책 실험은 로봇 조작이다. OGBench에서는 강한 task와 실패하는 task가 함께 있으며 전체 성능에서
모든 비교군을 이기지 않는다. 저자도 작은 카메라 움직임과 반복되는 상황에 의존하는 한계를
명시한다. 주행 일반화와 planning 효용은 검증되지 않았다. §5.2·§6·A.10.4를 확인했다.
[논문](https://arxiv.org/html/2603.04553v1)

## Slot Attention부터 LPWM까지의 적용 차이

| 기반 | 가져올 수 있는 것 | 현재 연구에서 확인할 점 |
|---|---|---|
| Slot Attention | 경쟁적 attention으로 feature를 slot에 묶는 인터페이스 | slot을 붙였다는 사실만으로 객체·시간 대응·planning 중요도가 생기지 않는다 |
| DLP | 위치와 외형을 분리한 작은 표현, 위치 불확실성 | 영상 복원으로 발견한 부분과 차량·보행자의 대응 |
| DDLP | 위치·크기 등의 속성과 시간 dynamics, 표현 조작 | 큰 프레임 간 이동과 새 객체 진입에서 추적의 안정성 |
| LPWM | 공식 전체 모델, 입자별 확률적 전이, 조건 입력과 decoding | 주행의 카메라 이동, 작은 객체, 입자 간 정보 전달, 과거만 사용하는 rollout |

위 각 기반의 기능은 [Slot Attention 원문](https://arxiv.org/abs/2006.15055),
[DLP 저자 자료](https://taldatech.github.io/deep-latent-particles-web/),
[DDLP 저자 자료](https://taldatech.github.io/ddlp-web/),
[LPWM 공식 구현](https://github.com/taldatech/lpwm)에 근거한다.
마지막 열은 이번 연구의 검증 질문이다. 이 네 방법을 같은 데이터에서 모두 재학습한 비교는 아니다.

## 공식 코드에서 확인한 실제 계산 경로

아래 경로는 `reference_repositories/LPWM/` 기준이다. 공식 파일은 수정하지 않았다.

| 코드 | 확인 내용 | 우리 코드에서 주의할 점 |
|---|---|---|
| `modules/modules.py::AlternativeSpatialSoftmaxKP` | 반환은 height, width 순서 | 주석의 x, y 표현만 믿으면 투영이 뒤집힌다 |
| `DLPEncoder`, `ParticleEncoder` | proposal → attribute → appearance → particle interaction | image encoder 단계와 context 단계의 조건 입력을 구분한다 |
| `models.py::encode_all` | foreground 64개와 background를 사용 | decoder의 30개만을 전체 latent 개수로 세지 않는다 |
| `models.py::decode_all` | 복원은 낮은 위치 분산으로 입자를 골라 배치 | top transparency 16개 평가와 decoder의 실제 선택은 별개다 |
| `DLPDecoder` | sigmoid scale, RGBA glimpse, STN, depth 가중 합성 | 마스크는 학습된 시각적 분해이며 semantic mask 정답이 아니다 |
| `DLPContext` | inverse posterior와 policy prior를 별도로 출력 | 미래 posterior를 온라인 입력으로 넣지 않는다 |
| `DLPDynamics`, `PINT` | 시간에 causal한 spatial/temporal attention | 입자별 local motion과 ego camera motion의 혼합 가능성 |
| `models.py::forward` | dynamics 입력 detach가 False | 보조 미래 loss가 encoder에 도달할 경로가 있다 |
| `models.py::calc_dyn_elbo` | 정적·동적 KL, context KL, 복원, 투명도 정규화 | 재구현한 유사 loss 대신 공식 함수를 직접 호출한다 |
| `models.py::sample_from_x` | `use_all_ctx=True`는 전체 영상에서 context를 추출 | 이번 평가는 False, 실제 인자로 관측 4프레임만 전달한다 |
| `utils/loss_functions.py::LossLPIPS` | pixel MSE와 가중치 0.1의 LPIPS | VGG와 LPIPS 가중치도 공식 hash로 확인했다 |

이 표는 로컬 소스를 직접 확인한 결과다. 전체 원본과 함수명은
[공식 코드](https://github.com/taldatech/lpwm/tree/4cf53c403433e64c01652ac2adbec66231a46dea)에서 확인할 수 있다.

### 논문 설명과 코드를 구분해야 하는 부분

`calc_dyn_elbo`의 투명도 정규화는 static frame의 입자 투명도 **합을 제곱**한다.
논문에 적힌 입자별 제곱합과 그대로 같다고 설명하면 안 된다. 이번 실행은 release의 계산을 유지했다.
또 공식 학습 script는 `use_ep_done_mask` 대신 `ep_done_mask`라는 키를 읽는 부분이 있다.
이번 고정 길이 clip은 episode 경계를 넘지 않아 해당 옵션을 사용하지 않는다.

Table 5의 Sketchy proposal patch size는 8로 표기돼 있지만, 공개 checkpoint의 `hparams.json`은
16이고 proposal 수는 64다. PDF 표도 직접 확인했다. 이번에는 checkpoint와 strict 호환되는
동봉 설정을 따른다. 논문 표의 설정과 공개 artifact 설정을 동일하다고 가정하지 않는다.

공식 loss는 입력 길이를 읽는 대신 model의 horizon 값으로 reshape한다. 8프레임 입력을 위해
strict loading 후 바깥 `timestep_horizon`만 7로 설정했다. 기존 20-step 위치 embedding의 크기와
가중치는 유지하고 앞부분을 사용한다. TorchScript KL은 batch/time slice에 `.view`를 사용해
실패했으므로 함수 경계에서 `.contiguous()`만 적용한다. 값·gradient·수식은 바꾸지 않는다.
처음 실패한 smoke 로그와 성공 로그를 모두 보존했다.

## 카메라 움직임과 가림에 관한 구체적인 가설

**카메라 움직임 가설:** 영상 좌표의 변화에는 차량 자체 이동, ego 회전, ego 병진에 따른
깊이별 시차가 섞인다. 회전은 calibration과 관측 pose로 보정할 수 있다. 이번 변형은
`K_ref R_ref^T R_t K_t^-1`로 첫 관측 좌표계에 정렬한다. 병진·시차·새로 드러나는 영역은 남는다.
따라서 이 대조에서 효과가 없더라도 3D 또는 ego motion 분리 전체가 불필요하다는 뜻은 아니다.

**가림 가설:** 프레임별 encoder는 현재 보이지 않는 물체를 이미지 자체에서 복구할 수 없다.
이를 지속적으로 다루려면 과거 dynamics의 belief와 현재 관측을 결합하는 상태가 필요하다.
박스 내부에 particle 점이 남아 있다는 사실만으로 객체 permanence가 입증되지 않는다.
외형 보존·재등장 시 대응·불확실성·해당 객체가 planner에 미치는 영향을 함께 검사해야 한다.

**해상도 가설:** 공식 checkpoint의 128×128 입력에서는 원거리 객체의 픽셀 수가 매우 작다.
이번에는 원본 전방 영상의 상하 28행을 자르고 128×128로 늘려 공식 구조와 호환시킨다.
이 전처리는 비등방 변형을 포함한다. 처음부터 늘어난 backbone이나 고해상도 모델로 바꾸면
공식 checkpoint 적응과 구조 확장의 효과가 섞이므로 이번에는 유지한다.

## Novelty를 주장하기 전에 비교해야 하는 선행연구

[SAVi++](https://arxiv.org/abs/2206.07764)는 LiDAR의 sparse depth를 학습 목표로 사용해,
이동 카메라의 실제 Waymo 영상에서 객체 분해와 추적을 다뤘다. Depth supervision과
자율주행 객체 표현 자체는 새 주장으로 삼기 어렵다. 초기 박스 조건의 사용 여부도 맞춰 비교해야 한다.

[3D-DLP](https://eubooks3003.github.io/3d-dlp/)는 RGB-D 또는 voxel을 3D particle로 표현하고,
같은 정책에서 2D·3D 표현을 바꾸는 비교를 제시한다. DLP에 3D 좌표와 크기를 추가하는 발상도
이미 존재한다. 이 자료에서의 로봇 조작 성능을 주행 성능 근거로 사용하지 않는다.

우리 연구의 후보 기여는 **ego intent와 planning 목적에 따라 유지해야 할 객체의 미래 상태와
불확실성을 학습하고, 같은 예산에서 이를 활용하는 표현**이다. 다음은 아직 검증하지 않은 설계다.

1. 관측 ego motion으로 설명되는 변화와 객체의 잔여 변화를 분리한다.
2. 의미 특징·깊이·시간 대응을 학습 목표로 사용해 작은 주행 객체의 표현을 보강한다.
3. 가려진 객체에는 존재 확률과 상태 불확실성을 유지하고, 재관측 시 갱신한다.
4. intent를 image/particle encoder 내부에 조건으로 넣고, 동일 영상에서의 표현 변화와
   relevant-object 정보 보존을 측정한다. Context/dynamics에만 intent를 넣은 조건을 대조군으로 둔다.
5. 최종적으로 같은 planner·같은 예산에서 미래 감독 없음, intent 없음, 카메라 분리 없음,
   memory 없음의 대조군을 통해 planning 효과를 분리한다.

“이미지 복원이 좋아졌다 → 객체를 이해했다 → planning이 좋아진다”는 추론은 이번 연구에서
각각 별도의 실험으로 검증해야 한다.

## 이번 적응 실험과 재현 진입점

등록 설정은 `configs/lpwm_navsim_adaptation/pilot_v1.json`, 선택 목록은
`results/lpwm_navsim_adaptation_v1/selection_manifest.json`이다.
90 train clip/57 recording, 30 development clip/17 recording으로 나누며 recording은 겹치지 않는다.
이 development recording은 이전 연구에서 사용했으므로 새로운 독립 test라고 부르지 않는다.
직진·회전·투영 겹침 순서의 수는 train 10/40/40, development 6/12/12다.

공식 checkpoint로부터 원영상·회전 보정 영상 각각 3 seed × 300 update를 실행한다.
Encoder·decoder·context·dynamics 전체를 갱신한다. Detector·GT box·track·planning loss는 모델의
학습 입력이나 loss에 사용하지 않는다. GT 박스는 clip의 상황별 선정과 평가에 사용한다.
회전 보정은 pose와 calibration을 추가로 사용한다.
이 차이는 정보량까지 완전히 같은 비교가 아니라 기하 정보 활용의 통제된 진단이다.

평가는 같은 particle 예산의 박스/점 대응, 균일 격자·무작위 점 대조, 복원, 과거 4프레임에서
미래 4프레임까지의 rollout을 함께 본다. 가림 후보는 depth 순서가 있는 투영 박스 겹침이며
visibility 정답이 아니다. 알려진 위치를 회색 사각형으로 가리는 추가 진단도 실제 가림과 구분한다.
Stabilized 모델의 미래 camera rotation은 과거 두 회전에서만 외삽한다. 미래 GT pose로 복원하지 않는다.

코드 진입점은 `scripts/prepare_lpwm_navsim_clips.py`,
`src/planning_aware_future_prediction/object_centric/lpwm_bridge.py`,
`scripts/run_lpwm_navsim_adaptation.py`, `scripts/evaluate_lpwm_navsim_adaptation.py`다.
실행은 `scripts/launch_lpwm_navsim_processes.py`가 두 GPU의 잠금과 PID 기록을 가진 queue를 시작한다.
완료한 checkpoint는 재학습하지 않는다. 실험 결과와 실제 particle 그림은 별도 결과 보고서로 정리한다.
