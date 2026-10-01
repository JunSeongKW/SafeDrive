# 저분산 진단의 범위·전처리·직접 선행연구 확인 수준

기준 `9353acf`. 완료된 cache373/train277/dev96와 A–E200update를 재사용한다.
이번 하위 질문은 **저분산이 구현·target·학습량 중 무엇과 관련되며, 같은 미래 감독에서
planning→predictor gradient 결합이 어떤 차이를 만드는가**다. 선택/예산 효과의 검증은 아니다.

## 실행 전에 고정한 범위

[고정 설정](../configs/exploration/future_prediction_diagnostic_followup_v1.json)에 실행 상한을 먼저 기록했다.

- A만 seed11/47 각200update 추가. 기존29와 합쳐 A의 초기 변동을 측정한다.
- 기존 seed29 A–E의 model/AdamW200 state를 복원하고 각각 총1000update까지800회 추가한다.
- F29 및 E11/F11은 같은 seed의 공유 초기 weight·batch 순서로 각1000회 신규 학습한다.
  E–F는 seed29/11의 대응 비교이며 A의 seed 변동으로 C/D/E의 변동을 대신하지 않는다.
- 새 update는 합7400, run별 evaluation 포함600s, invocation1800s 보호 상한.
  평가0/200/500/750/1000; 기존200 이하의 평가와 학습은 다시 실행하지 않는다.
- 고정 거리 K4/front/공통 loss mask/whitening/lr/clip/architecture/aux 계수를 유지한다.
  F는 planner 입력만 detach; 반환 predictor 출력의 두 aux loss는 계속 학습한다.
- NaN/무결성/시간 상한 위반이면 중단한다. 초기 순위로 조건을 탈락시키지 않는다.
  이 실행 후 residual/delta-target 추가는 자동으로 하지 않는다. 별도 하나의 bounded ablation으로만 검토한다.
- Learned selector·동적K/horizon·확률적 predictor·encoder 학습·full cache·SafeDrive 재학습 없음.

## JPEG 전처리: 수치 일관성과 실제 필요성을 분리

기존 4개 표본 원본DB K/D와1920×1080 일치, 왕복1.14e-12px,
원본/보정 contact sheet를 재사용했고 직진/회전 대표2장을 다시 직접 확인했다.
왕복은 **동일 모델의 invertibility**다. 실제 JPEG가 보정 전인지/후인지 알려주지 않는다.
기존 sheet의 ROI 좌표 shift0 또한 두 열에 같은 pinhole box를 그렸기 때문이지,
픽셀 appearance나 정합이 같다는 증거가 아니다. 주변부 모습은 보정 적용 시 변한다.
Projection-valid는 실제 visibility나 occlusion 대응이 아니다.

- [OpenScene download 규약](https://github.com/OpenDriveLab/OpenScene/blob/main/docs/getting_started.md):
  nuPlan sensor를 직접 링크할 수 있고 downsampling으로 제공한다고 명시한다.
  이는 original-distorted 취급을 지지하지만 **로컬 JPEG의 실제 export 이력 증명은 아니다**.
- [공식 collect_data.py](https://github.com/OpenDriveLab/OpenScene/blob/main/DriveEngine/process_data/collect_data.py):
  기존 metadata pkl을 train/val로 합치는 script다. 이 파일은 JPEG 가공 경로를 보여주지 않는다.
- [MTGS v2 Appendix A.1](https://arxiv.org/html/2503.12552v2#A1): nuPlan distortion과 calibration 문제,
  OpenCV undistortion을 서술한다. 다른 연구의 처리이고 **현재 JPEG의 이중 보정 여부를 확인한 것은 아니다**.
- 로컬 original nuPlan DB calibration은 읽기 전용 확인됐지만 별도 원본 sensor JPEG가 없어
  byte/파일 provenance 대조가 아직 불가능하다. 표본 정합도 정량 GT visibility 라벨이 없다.

따라서 현재 `rectified_v1b`는 **저장 이미지가 원래 왜곡 영상이라는 조건부 가정**으로 유지한다.
추가 보정이 옳다고 확정하거나 이중 보정 가능성을 닫지 않는다. 잘못됐다는 결정적 근거도 없어
adapter/full cache를 무작정 바꾸지 않았다. 기존 raw/cache/checkpoint 모두 보존한다.
이번 감독/gradient 비교의 주장은 이 공통 전처리 가정 아래로 제한한다.
후속 visual target 확정 또는 multi-view 전환 전에는 원본 sensor byte provenance 또는
annotation으로 통제된 정합/소수 동일 표본의 feature 민감도 검사를 확보해야 한다.

## 문헌 확인 수준과 iPad 수치의 올바른 조건

기존 [EgoFSD/ForeDrive 감사](egofsd_foredrive_evidence_audit.md)는 당시 원문 직접 검토 결과이며
EgoFSD selection autograd/ForeDrive 공개 모델 재현은 미확인이다. 이번 실행에서 이를 독립 재현하지 않았다.

이번에는 [iPad arXiv2505.15111v1](https://arxiv.org/html/2505.15111v1#S4.SS3)의 §3.4·§4.3/Table4와
[공식 README](https://github.com/Kguo-cs/iPad)를 직접 읽었다. Source 전체/학습 재실행은 하지 않았다.
Table4의 마지막 두 행은 proposal refinement/ProFormer/proposal-centric mapping을 유지하고,
prediction을 **General→Proposal-centric**으로 교체한 PDMS90.5→91.7이다.
89.8→91.7은 mapping까지 함께 바뀌므로 prediction 하나의 효과로 인용하지 않는다.
두 비교 모두 predictor 없음→선택적 미래 예측의 비교가 아니며 동일K learned selector나
adaptive budget 검증도 아니다. §3.4는 proposal과 충돌/잠재 충돌 관련 객체의 미래 state를
training auxiliary로 구성한다. 강한 직접 관련 선행연구로 반영하되 우리 효과나 novelty 증거로 옮기지 않는다.

## 후속 선택/예산 연구의 진행 조건

- 현재 front-only/GT ROI/거리 선택은 개발용이다. 후보>K인 train132/277,dev54/96와
  활성 slot수를 함께 보고 K2/4/8의 selection opportunity를 확인해야 한다.
- Right/merge·횡방향 interaction coverage 부족을 보완하고 multi-view 후보 풀과
  같은 객체의 view간 중복 제거/track 대응을 설계한 뒤 일반적 상황별 예산을 주장한다.
- 현재 관측은 유지하며 선택하지 않은 객체도 planner의 기본 인지/예측에서 무시하지 않는다.
  Encoder/기본 예측/선택 비용을 포함하고 selected predictor 효율과 전체 효율을 구분한다.
- 확률적 확장은 아직 **설계 후보**: 공간 평균·공분산부터, calibration과 planning을 따로 평가.
  Uncertainty token 입력과 probabilistic collision constraint는 다른 설계다.
  EgoFSD noise 학습·확률적 미래 예측·Tube-MPC를 같은 방법으로 취급하지 않는다.
  중요도×불확실성 예산 배분의 novelty는 직접 문헌 검증 전까지 유보한다.
