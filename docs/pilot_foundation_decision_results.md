# Pilot 기반 판단 — `607da52` 이후 완료 기록

## 결론과 최신 우선순위

사용자의 후속 지시에 따라 **predictor 튜닝과 확대 pilot 학습은 보류**한다. 작은 pilot은
연결·gradient·target 대응 진단 자산으로 보존한다. 원래 질문은 다음과 같다.

> 미래 예측과 planning이 연결된 기반에서, 모든 대상을 동일하게 예측하는 대신 현재 상황과
> ego 의도에 맞춰 예측 대상·범위·평균 예산을 배분하면 유용성 또는 효율이 좋아지는가?

현재 모델이 단순 기준선보다 약하다는 사실은 **이 pilot의 기반 적합성 문제**다.
미래 예측의 가치를 처음부터 다시 입증해야 하거나 연구 가설이 실패했다는 결론이 아니다.
[공개 기반 감사와 선택 실험 진입 계획](public_future_planning_foundation_audit.md)을 다음 진입점으로 삼는다.

## 재사용 / 새 실행 / 미실행

- 재사용: 기존 rectified v1b 373-window cache(train277/dev96, recording12/4), train-only normalization,
  `607da52`의 seed29 A–F 1000-update checkpoint 및 seed별 공통 초기 state/batch 순서.
- 새 실행: CPU 물리·ridge·공간/영상 기준선 평가, 하나의 visual-only residual 대조(3 대응 seed),
  navtrain manifest 조사, 4표본 전처리 민감도, 공식 WA-JEPA attention의 작은 CPU gradient 검사.
- 미실행: 200-window encoder profile, 확대 cache, 확대 residual A/B/E/F 학습, 새로운 selector,
  적응형 예산, 공식 WA-JEPA full checkpoint 로딩·추론·성능 재현, held-out 모델 평가.
- 기존 SafeDrive 코드/CSV/checkpoint, 공용 원본, 기존 cache/결과, 타인 프로세스를 변경하지 않았다.

## 1. 단위·좌표·기준선 재현

`current_ego_status=[command4,vx,vy,ax,ay]`, 속도 m/s·가속도 m/s²다. Ego 목표는
**현재 rear-axle 기준 고정 XY frame**의 m 좌표. 8시점은 명목상 0.5–4.0초이고,
기존 실제 timestamp 조사 최대 오차는 0.012979초였다.
현재 entity geometry tail은 `[x/40,y/40,sin(yaw),cos(yaw),width/10,length/10,vx/10,vy/10,class2]`,
미래 spatial target은 `[x/40,y/40,sin(yaw),cos(yaw),vx/10,vy/10]`다.
미래 ego frame으로 움직여 가는 좌표가 아니다. 영상은 동일 frozen teacher ROI의 1024차원 latent다.
Native velocity/acceleration 추출 코드와 과거 pose 차분 4표본도 대조했다. 작은 차분 일치는
모든 장면의 calibration 또는 가속도 정의를 독립 검증한 결과는 아니다.

Dev 지표는 기존과 같은 **scene-macro XY ADE**다: window 내 8시점 거리 평균→같은
recording+scene 내 window 평균→scene 균등 평균. Dev96window/86scene/4recording이다.

| Ego 기준선 / 기존 모델 | Dev ADE(m) |
|---|---:|
| 정지 | 9.1239 |
| 현재 vector 등속 | 1.0951 |
| 현재 vector 등가속 | 0.9260 |
| Train-fitted ego-status ridge | **0.7677** |
| A: branch 없음, seed29/1000update | 1.6717 |
| B: branch/planning only | 1.7445 |
| C: visual aux | 1.6485 |
| D: spatial aux | 1.6991 |
| E: mixed aux | 1.5747 |
| F: E + planner-input detach | 1.5536 |

Ridge는 **단순 학습 기준선**이다. Train277window만 사용해 8채널을 표준화하고 unpenalized
intercept를 포함한 16차원 XY 회귀를 적합했다. λ grid
`[0.0001,0.001,0.01,0.1,1,10,100]`는 train 내 3 recording-group fold의 scene-macro ADE로
선택했다. Fold마다 표준화도 train fold에서 다시 적합했다. Dev로 λ를 선택하지 않았다.
정확한 선택 λ/fold 점수는 [공유 기준선 JSON](../results/pilot_foundation_decision/physical_and_ridge_references.json)에 있다.

공간 예측은 정규화 MSE뿐 아니라 위치 m, 속도 vector/vx/vy/speed m/s, wrapped heading rad를
각 horizon·mask별로 계산했다. 아래는 **+4초 common mask, 215 slot 관측**이다.

| 객체 기준선 / 감독된 predictor | 위치 평균오차(m) | 속도 vector 평균오차(m/s) | 방향 MAE(rad) |
|---|---:|---:|---:|
| 현재 위치 유지 | 10.8494 | 0.9674 | 0.0631 |
| 현재 속도 등속 외삽 | **2.0197** | 0.9674 | 0.0631 |
| D | 6.3530 | 1.9683 | 0.4019 |
| E | 5.5511 | 1.9103 | 0.4096 |
| F | 5.6630 | 1.9347 | 0.4228 |

위치도 CV보다 나쁘다(G2). 이것만으로 spatial 감독이 planning에 도움이 없다고 결론 내리지 않는다.
영상 normalized common-mask MSE는 persistence0.3950/train-mean0.9496/C0.7936/E0.8344/F0.8234다.
모든 기준선은 같은 target·train normalization·mask를 사용했다. Common/native 결과와 전체 horizon은 JSON에 있다.

## 2. 사전 상한 내 visual-only residual 대조

등록 [설정](../configs/exploration/pilot_foundation_decision_v1.json)을 먼저 고정했다. 기존 C의 target,
K4, aux0.1, 공간 aux0, 모듈 크기는 유지하고 **현재 ROI + 학습 visual delta**만 추가했다.
Ego/spatial residual은 이 대조에서 껐다. 세 seed29/11/47, 1000update, batch8, AdamW
lr/weight decay 각1e-4·constant schedule·clip1.0; 평가0/200/500/1000, per-run600s/전체1800s.
완료 C29는 재사용하고 나머지 5run=5000update만 새로 학습했다.

| Seed | Absolute C ADE | Visual-residual C ADE | Residual−absolute(m) | Absolute / residual 영상 MSE |
|---|---:|---:|---:|---:|
| 29 | 1.6485 | 1.6189 | −0.0297 | 0.7936 / 0.3789 |
| 11 | 1.7126 | 1.8104 | +0.0978 | 0.7911 / 0.3815 |
| 47 | 1.7588 | 1.8341 | +0.0753 | 0.7800 / 0.3833 |
| Mean ± seed sample STD | 1.7066 ± 0.0554 | 1.7545 ± 0.1180 | +0.0478 ± 0.0680 | — |

초기 common module state와 기존 1000-batch sequence를 실제 파일로 대조했다. Residual visual
출력행만 zero init이고, update0은 persistence와 같다. 따라서 0.79→0.38의 큰 차이는 대부분
**persistence를 구조에 넣은 효과**다. 학습 delta의 추가 이득은 0.3950→약0.3812(약3.5%)로 구분한다.
Planning 개선은 대응 seed에서 일관되지 않고 평균은 악화했다. Ridge보다 여전히 약하다(G1).

세 seed 차이를 window별로 평균한 후 recording cluster bootstrap(2000회, seed20261002)을
계산했다. Residual−absolute 평균+0.04783m, 조건부95% percentile 구간
`[+0.01372,+0.08644]m`. 네 recording 모두 평균 차이가 양수다. **고정된 세 학습 seed와 반복 사용한
4 dev recording에 조건부인 탐색 결과**이며, 이 CI는 training-seed 불확실성을 포함하지 않는다.
"차이가 seed STD보다 작으면 차이 없음" 규칙을 사용하지 않았다.
공유 [대응 통계](../results/pilot_foundation_decision/visual_residual_paired_statistics.json),
[window별 차이](../results/pilot_foundation_decision/visual_residual_per_window_paired.json),
[평가 시점별 결과](../results/pilot_foundation_decision/visual_residual_runs.json).

Residual 계약 코드·테스트에서는 ego CA prior, spatial CV prior, visual persistence를 같은 단위에
더하고 활성 correction의 마지막 layer만 zero init했다. Update0 prior 동일, invalid slot0,
planner-input detach의 forward 불변/aux 학습 보존을 검사했다. Zero head 때문에 초기 upstream
gradient0였다가 한 update 후 nonzero가 되는 것도 검사했다. **모든 residual의 공동 학습 성능은 검증하지 않았다.**

## 3. 전처리 / split / 확대 중단 기록

OpenScene pinned helper `72860746787a67946bef07aa1f78bbbc6b20e445`는 원본 JPEG 경로와 K/D를
복사하지만 JPEG 재배포/export 과정은 설명하지 않는다. 로컬 독립 nuPlan 원본 JPEG나 export manifest도
찾지 못했다. 4 동일 표본의 stored-pinhole / stored-distortion / undistorted-pinhole overlay를 생성했다.
보정 시 ROI 좌표 평균 변화4.19–7.56px, feature RMS 변화0.463–0.919였다.
이 수치는 **전처리에 민감함**을 보이지 어느 가정이 맞는지를 증명하지 않는다. 대표 정합 역시 독립
alignment GT가 없다. 왕복 오차를 근거로 해결했다고 기록하지 않았다. 기존 cache는 조건부 가정 그대로 보존했다.
[전처리 evidence](../results/pilot_foundation_decision/preprocessing_evidence.json).

Navtrain(trainval)의 공식 list1192segment 모두 존재하고 실제 trainval1310pkl를 확인했다.
Native log_name을 vehicle+capture recording162개로 묶었다. Mini52group을 held-out에서 제외하고
결과 확인 전 assignment를 기록했다. History4/future8/0.5s, 비중첩, **이전 cap24 유지**다.
현재 history front4파일은 필수 입력이고 미래 이미지 누락은 window 제외 사유가 아니라 loss-only mask다.

| Split | Recording | Window |
|---|---:|---:|
| Train | 82 | 1857 |
| Dev | 40 | 865 |
| Held-out | 40 | 898 |

162×24의 이론 상한3888로 요청한 12000window를 만들 수 없다. Cap을 임의로 풀거나 중복으로 수를
맞추지 않았다. Native log/current token 교차·recording overlap·mini-held leakage 없음 검사 통과.
Navtest/Navhard/private_test는 fitting에 사용하지 않았다. Held-out은 **manifest만 조사**했으며 모델 평가 없음.
[공유 manifest](../results/pilot_foundation_decision/navtrain_three_way_manifest.json).

Config/manifest 기반 cache runner는 구현했지만 전처리 미해소로 **fail-closed**했다.
200-window encoder profile은 미실행이며 속도/VRAM/전체cache 용량을 실측했다고 보고하지 않는다.
당시 등록한 32GiB/cache4h/profile1800s/free64GiB 상한과 A/B/E/F 확대 계획은 남아 있지만,
**최신 사용자 우선순위로 확대 실행은 보류; 자동 재개하면 안 된다.**
[실행 gate](../results/pilot_foundation_decision/expanded_cache_execution_gate.json).
조사 초반 GPU0/1은 타인 CARLA가 점유했다. 종료 직전 재확인에서는 각각25MiB/compute process 없음이었다.
GPU 점유는 시시각각 바뀌므로 다음 실행 전에 다시 확인한다. 이번 신규 작업은 전부 CPU였다.

## 4. 실측 비용 / 검증 / 재현

| 실제 새 작업 | CPU wall time |
|---|---:|
| 기준선·기존 A–F 평가 | 33.67s |
| Visual-only residual 대응3seed / 새5000update | 364.26s |
| 1192 segment·3-way manifest 조사 | 228.75s |
| Frozen encoder 동일4표본 전처리 민감도 | 39.69s |
| 공식 attention 작은 autograd 검사 | 0.023s |

전체 54 unittest와 새 코드 Ruff 통과. 최초 test discovery는 `PYTHONPATH` 미설정으로 기존 test
import1건이 실패했고, 올바른 package path로 재검사했다. GPU graphics 메모리 점유 guard 검사까지
추가한 최종 전체 재검사에서는 54/54 통과했다.
새 final checkpoint5개의 strict load/finite model·AdamW/sampler1001/RNG presence를 확인했다.
새로 학습을 이어 실행해 resume를 검증한 것은 아니다. Run 후 import/format만 정리했으며
과거 raw source hash·timestamp·수치·curve를 새 commit hash로 덮어쓰지 않았다.

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
  PYTHONPATH=src:scripts PYTHONDONTWRITEBYTECODE=1 \
  runtime/environments/visual_future_prediction_pilot/bin/python -m unittest discover -s tests -v
# 아래는 새 output path가 필수인 재현 진입점. 완료 학습을 다시 실행할 필요는 없다.
# scripts/evaluate_kinematic_and_ridge_references.py --output-directory outputs/...새경로...
# scripts/train_registered_visual_residual_control.py --output-directory outputs/...새경로...
# scripts/prepare_navtrain_recording_split.py --output-directory outputs/...새경로...
# scripts/audit_preprocessing_sensitivity.py --output-directory outputs/...새경로...
# scripts/summarize_pilot_foundation_decision.py --output-directory results/...새경로...
```

Raw window별 모든 horizon/mask 오차는 `outputs/pilot_foundation_decision/physical_ridge_references_v1b/`의
train/dev JSON에 보존했다. Git 공유에는 aggregate/horizon, 작은 ego window JSON, seed별 대응 window,
manifest와 evidence를 담았다. GPU-hour 추정·공식 안전성·일반적 선택 성공은 주장하지 않는다.
성능 약세의 원인은 데이터·분포/GT/front-only·구조·최적화·전처리 후보로 남아 있고 "데이터 부족 과적합"으로 확정하지 않는다.

## 다음 결정

**권고2: 공개 future-planning 기반을 우선 재현한다. WA-JEPA가 첫 후보다.** 공식 높은 점수 때문이
아니라 source상 inference 연결과 planning→scene gradient, 공개 weight/evaluation 경로가 확인됐기 때문이다.
기존 pilot은 유지하되 주 기반으로 개선하지 않는다. 공간·시간 token으로의 최종 연구 단위 변경은
별도 설계 판단이며 객체와 동등하다고 주장하지 않는다. 세 선택·예산 실험과 진입 gate는 다음 감사 문서에 있다.
