# LPWM의 NAVSIM 적응 실험 결과

## 결론

**LPWM은 객체 중심 미래 표현을 설계하는 선행연구로 활용 가치가 있다. 그러나 이번 작은 적응만으로
planning에 필요한 객체를 안정적으로 보존하는 표현을 얻지는 못했다.**
영상 복원은 개선됐고 일부 차량 영역을 분리했지만, 주요 particle은 도로·나무·건물과 차량 일부를
함께 표현한다. 과거 영상만 사용하는 미래 예측은 마지막 프레임 유지 기준선과 비슷했다.
첫 관측 영상으로 회전을 보정하는 변형은 시야를 잃는 문제가 커서 채택하지 않는다.

이는 300-update 적응 결과다. LPWM의 충분한 주행 사전학습 가능성을 기각하는 결과도,
새 아키텍처의 planning/PDMS 효과를 평가한 결과도 아니다.
논문과 코드의 자세한 검토는 [LPWM 검토 문서](lpwm_paper_and_driving_assessment.md)에 있다.

## 실제 실행 범위

| 항목 | 실행 |
|---|---|
| 공식 source | `taldatech/lpwm`, `main`, `4cf53c403433e64c01652ac2adbec66231a46dea` |
| 시작 가중치 | 저자 공개 Sketchy checkpoint, MAC 검증과 strict loading 통과 |
| 학습 | 원영상 / 회전 보정 영상 × seed 29, 47, 71 × 300 update |
| 학습 대상 | 109,545,263 parameter 전체, encoder·decoder·context·dynamics |
| 입력 | 128×128 전방 RGB, 8프레임, batch 2, 2 Hz |
| 목적함수 | 공식 temporal ELBO와 MSE + 0.1 LPIPS, Adam 8e-5 |
| 학습 자료 | 90 clip, 57 recording |
| 개발 평가 | 30 clip, 17 recording, train recording과 겹침 없음 |
| 상황 | 직진 6 / 회전 12 / 투영 겹침·가림 후보 12개 개발 clip |
| 미래 평가 | 관측 4프레임 → 미래 4프레임, 최대 2초, deterministic prior rollout |
| 비용 | 6회 합계 학습 시간 807.39초, 최대 allocated 10.145 GiB |
| 실행 자원 | GPU 0·1에 각각 한 프로세스, 6 GiB 여유 메모리 보호 |

모든 학습은 같은 마지막 300 update checkpoint를 평가했다. 개발 결과로 학습 길이·hyperparameter를
선택하지 않았다. 서로 다른 recording으로 나눴지만 기존 연구에서 사용했던 개발 자료이므로 독립 test는 아니다.
GT 객체 박스·track은 모델 입력·loss에 사용하지 않았으며, clip의 상황별 선정과 평가에 사용했다.
이번에는 intent를 encoder에 넣거나 planning loss로 학습하지 않았다.
회전 보정 조건은 관측 camera pose와 calibration을 추가 입력으로 사용한다.

이미지는 상하 28행을 잘라 1920×1024로 만든 뒤 정사각형 128×128로 resize했다.
해상도와 비등방 변형이 특히 작은 객체에 불리할 수 있다. 사전학습 도메인 차이·짧은 적응·입력 해상도가
모두 제한이므로, 결과만으로 실패 원인을 한 가지로 확정하지 않는다.

## 전체 비교

적응 조건은 3 seed 평균이다. 객체 지표는 현재 프레임에 3픽셀 이상 투영되는 평가 대상이 있는
24개 clip의 평균이며, 영상 지표는 전체 30개 clip 평균이다. 평가 객체는 vehicle·pedestrian·bicycle다.

| 조건 | 객체 박스 안 점 포함률 ↑ | 일대일 박스 대응률 IoU≥0.3 ↑ | 현재 영상 복원 MSE ↓ | 과거만 사용한 미래 MSE ↓ |
|---|---:|---:|---:|---:|
| 공식 모델 그대로 | 40.73% | 17.71% | 0.05680 | 0.06934 |
| NAVSIM 원영상 적응 | 31.76% | 18.47% | 0.01640 | 0.03015 |
| 회전 보정만 적용한 공식 모델 | 33.22% | 13.67% | 0.08518 | 0.11162 |
| NAVSIM 회전 보정 적응 | 15.14% | 8.50% | 0.05121 | 0.08633 |
| 균일 격자 16개 | 21.89% | 11.28% | 해당 없음 | 해당 없음 |
| 마지막 관측 프레임 유지 | 해당 없음 | 해당 없음 | 해당 없음 | 0.03032 |

**점 포함률**은 transparency 상위 16개 particle 중심 중 하나라도 GT 투영 박스 안에 있는 객체의 비율이다.
이는 semantic recognition 정확도가 아니다. 큰 박스·겹치는 박스는 우연히 점을 포함할 수 있다.
**박스 대응률**은 particle의 위치·크기로 만든 박스를 GT 박스와 Hungarian 방식으로 일대일 대응시킨
뒤 IoU≥0.3인 객체 비율이다. Decoder 마스크의 segmentation 정확도와 다르다.
균일 격자의 박스 크기는 32×32이며, 학습 없이 위치를 덮는 참고 기준선이다.
복원 MSE는 8프레임 각각을 해당 프레임의 posterior로 복원한 평균이며 미래 예측 오차와 구분한다.

복원 MSE는 약 71% 감소했지만, 박스 대응률의 추가 개선은 **0.76 percentage point**였다.
Recording 단위 paired bootstrap 95% 구간은 **−5.80∼+6.43 pp**로 0을 포함한다.
점 포함률은 **8.97 pp 감소**했고, 구간은 **−15.67∼−3.03 pp**다.
복원 개선과 주행 객체 표현 개선이 함께 일어났다고 주장할 수 없다.

미래 MSE는 원영상 적응이 마지막 영상 유지와 **0.00017** 차이다.
Seed별 값은 0.03037 / 0.02940 / 0.03068로 기준선을 넘는 방향도 일관되지 않았다.
적응 모델에서 기준선을 뺀 차이의 95% 구간은 −0.00338∼+0.00307로 0을 포함한다.
이 평가는 multimodal future의 분포 품질을 측정하지 않는다. 30개 clip으로 FVD를 보고하지 않았다.

불확실성은 2,000회 recording cluster bootstrap으로 계산했다. Seed 평균의 clip 차이를 사용하므로
학습 seed의 불확실성 전체를 포함하지 않으며, 여러 지표의 다중 비교 보정도 하지 않았다.

## 직진과 회전과 가림 후보의 차이

| 상황 | 객체 평가 clip | 공식 모델 박스 대응률 | 원영상 적응 박스 대응률 | 회전 보정 적응 박스 대응률 | 원영상 적응 미래 MSE |
|---|---:|---:|---:|---:|---:|
| 직진 | 4 / 6 | 0.00% | 4.17% | 4.17% | 0.03124 |
| 회전 | 8 / 12 | 37.92% | 31.67% | 16.39% | 0.03172 |
| 투영 겹침·가림 후보 | 12 / 12 | 10.14% | 14.43% | 4.68% | 0.02804 |

직진 표본에서 원거리 보행자·자전거 같은 작은 대상은 충분히 분리되지 않았다.
시각화의 회전 장면은 차량의 일부와 도로·배경을 함께 분해한다.
가림 후보에서는 일부 가까운 차량의 외형은 표현하지만, 여러 겹친 차량을 각각 안정된 단위로
나누는 결과는 확인하지 못했다. 세 시각화 사례는 모델 결과를 보지 않고 선택 목록의 첫 장면을 사용했다.
위 장면별 관찰을 30개 전체에서의 semantic segmentation 성능으로 일반화하지 않는다.

상황 정의는 투영 겹침≥35%를 먼저 분류하고, 나머지 중 yaw 범위≥15도를 회전,
yaw 범위≤5도와 속도≥1 m/s를 직진으로 분류했다. 가림 후보에는 회전도 포함될 수 있으므로
카메라 운동과 가림을 독립적으로 분리한 factorial 실험은 아니다. Yaw label은 원래 12프레임 창에서
정한 retrospective 평가 label이며 실제 모델 입력은 앞 8프레임이다.

## 회전 보정의 실패를 어떻게 해석하는가

변형은 첫 관측 camera 방향을 기준으로 모든 영상을 warp한다. 미래 출력은 마지막 두 관측 회전에서
constant angular velocity를 외삽해 현재 camera 쪽으로 되돌린다. **미래 GT pose는 사용하지 않았다.**
회전이 끝나는 시점이나 큰 회전에서 시야가 크게 잘려, 검은 영역이 출력에 남는 사례가 관찰됐다.

시야 손실과 모델 오차를 구분하려고 동일한 유효 영역에서 추가로 비교했다.

| 동일 유효 영역 평가 | 원영상 적응 | 회전 보정 적응 |
|---|---:|---:|
| 복원 MSE | 0.01614 | 0.01654 |
| 미래 MSE | 0.03025 | 0.03319 |

공통 영역은 평균 관측 영상의 85.94%, 미래 영상의 77.56%다. 미래 영역의 mask 역시 과거에서
외삽한 변환으로 정의했다. 미래 MSE 차이는 +0.00294, 95% 구간은 −0.00016∼+0.00675다.
즉 전체 영상에서 보인 큰 악화에는 시야 손실이 상당 부분 포함된다. 공통 영역에서도 개선 근거는 없다.
이 결과는 해당 고정 기준 영상 warp를 채택하지 않을 근거이며, 일반적인 ego motion 분리나
metric 3D 표현 전체가 무효라는 근거는 아니다.

## Particle 수와 가림 진단

| 예산 | 공식 모델 점 포함률 | 원영상 적응 점 포함률 | 공식 모델 박스 대응률 | 원영상 적응 박스 대응률 |
|---|---:|---:|---:|---:|
| 8 | 27.53% | 19.08% | 14.29% | 9.63% |
| 16 | 40.73% | 31.76% | 17.71% | 18.47% |
| 32 | 51.94% | 47.87% | 21.77% | 25.25% |

32개에서는 박스 대응률이 증가하지만 8개에서는 감소한다. 예산에 일관된 주요 객체 보존이
확인된 결과가 아니다. 이 순위는 transparency 기준이며 planning 중요도를 학습한 selector가 아니다.

현재 가장 큰 GT 박스 영역을 회색으로 바꾸는 **인공 가림 진단**도 수행했다.
가린 박스 안에 점이 남는 비율은 공식 모델 33.33%, 원영상 적응 20.83%, 회전 보정 적응 15.28%다.
이 숫자는 객체 기억을 측정하지 않는다. 고정 grid나 배경 particle도 가린 위치에 남을 수 있기 때문이다.
가림 전후와 다음 실제 관측의 출력을 시각화하고 특징 변화량은 clip별 JSON에 저장했다.
자연 가림 성능과 구분한다.

투영 GT track에 IoU≥0.1로 대응되는 인접 프레임에서 같은 particle 번호가 유지되는 비율은
공식 모델 44.84%, 원영상 적응 46.62%다. LPWM은 입자 간 정보 전달을 허용하므로 이 값은
정식 tracking 정확도가 아니며, 이 변화로 가림 후 re-identification 성공을 주장하지 않는다.
자연 가림의 visibility label과 가림 전후 appearance 대응을 갖춘 평가가 추가로 필요하다.

## 다음 연구 방향

**공식 LPWM의 RGB 복원 목적을 그대로 확대하기보다, particle이 보존해야 할 주행 정보를 바꾸는
방향을 우선 제안한다.** 이번 pilot에서는 아래 구조를 구현하거나 학습하지 않았다.

1. **주행 의미 특징과 기하를 학습 목표로 추가한다.** 기존 Drive-JEPA 특징을 teacher로 쓰는
   feature reconstruction, sparse depth, 객체별 residual motion을 분리 비교한다. 원거리 객체를
   평가할 수 있도록 해상도·ROI 품질도 함께 관리한다. 처음부터 모든 변경을 한 조건에 넣지 않는다.
2. **영상 전체를 과거 화면에 고정시키는 대신 particle 상태에서 ego motion을 처리한다.**
   원래 RGB 시야는 보존하고, metric depth를 가진 입자 상태의 좌표 변환과 객체의 잔여 dynamics를
   분리하는 방식이다. LPWM의 compositing depth를 metric depth로 간주해서는 안 된다.
3. **가림 동안 belief를 유지하고 새 객체 진입을 다룬다.** 과거 예측 상태와 현재 관측의
   불확실성을 결합하고, 가림·재등장·시야 이탈을 나눠 평가한다.
4. **Ego intent를 encoder 내부에 넣고 planning 목적과 연결한다.** 동일 영상에서 intent를 바꿀 때
   선택 위치만 달라지는지, 관련 객체의 상태·미래 정보를 담는 표현까지 달라지는지를 검사한다.
   Encoder 조건 입력 없음, dynamics에만 조건 입력, planning loss 없음의 대조군을 둔다.

깊이나 3D particle을 추가하는 것 자체는 [SAVi++](https://arxiv.org/abs/2206.07764)와
[3D-DLP](https://eubooks3003.github.io/3d-dlp/)를 고려할 때 충분한 novelty가 아니다.
우리 명제의 기여는 **동일 장면에서 의도별로 필요한 미래 객체 정보를 같은 예산으로 보존하고,
그 차이가 planning에 도움이 되는가**에서 검증해야 한다.

## 결과 이미지와 파일

로컬 시각화 폴더:
`/rhome/junseong/PlanningAwareFuturePrediction/outputs/lpwm_navsim_adaptation_v1/visualization/`

- `01_particles_and_masks.png`: 실제 영상, 공식 모델, 적응 모델, 회전 보정 모델, 실제 decoder 마스크.
- `02_reconstruction_and_causal_forecast.png`: 현재 복원과 과거만 사용한 2초 미래 예측.
- `03_quantitative_summary.png`: 객체 대응과 미래 영상 오차.
- `04_controlled_occlusion.png`: 인공 가림 전후와 다음 관측의 particle.
- `particles_straight.gif`, `particles_turn.gif`, `particles_projected_overlap.gif`: 8프레임 particle 변화.
- `index.html`: 로컬 갤러리.

공유 [시각화 PDF](../results/lpwm_navsim_adaptation_v1/lpwm_navsim_visualizations.pdf),
[전체 집계 JSON](../results/lpwm_navsim_adaptation_v1/summary.json),
[clip별 결과](../results/lpwm_navsim_adaptation_v1/per_clip_metrics.json),
[검증 기록](../results/lpwm_navsim_adaptation_v1/validation.json).

## 검증과 보존

- 실제 공식 checkpoint strict load, 전체 encoder gradient와 가중치 변화, finite loss를 확인했다.
- STN 좌표 순서·pixel center, camera rotation projection, 과거 회전 외삽의 검사 3개를 통과했다.
- 미래 영상만 바꾸는 intervention에서 관측 particle 변화는 8개 모델 모두 0이었다.
- 처음 두 평가에는 마스크·복원 LPIPS 저장을 추가하기 위해 checkpoint만 재추론했다.
  이전 지표와 차이는 모두 정확히 0이며 초기 결과를 별도 보존했다. 재학습은 하지 않았다.
- 공식 source의 tracked 파일, 기존 Drive-JEPA 모델·결과·환경, 공용 원본 자료는 변경하지 않았다.
- 새 환경 `runtime/environments/lpwm_navsim_adaptation/`, 실행 별칭 `kjs-lpwm-navsim`을 사용했다.
  추가 의존성은 이 환경에만 설치했고 source·가중치·환경 버전을 공유 JSON에 기록했다.
- 등록 학습과 평가가 종료됐다. WA 또는 이전 encoder 실험을 재개하지 않았다.
