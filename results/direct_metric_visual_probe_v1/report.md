# Frozen 시각 feature만으로 NC·DAC 직접 예측: 실제 pilot 결과

## 결과 해석

**시각 입력을 실제로 읽는 경로는 만들었으나, NC·DAC별로 읽는 영역을 분리하는 추가 이득은 확인하지 못했다.**

- 기존 78.48 → 새 shared/specific 79.55 PDMS다. Specific−기존은 +1.08점이며 recording bootstrap 95% 구간은 [-1.04, +2.91]이다. 단일 seed/기존 개발셋의 결과이므로 성능 향상을 확정하지 않는다.
- Shared와 Specific은 **192장면 모두 같은 후보**를 선택했다. NC/DAC attention 교환도 선택을 바꾸지 않았다. 지표별 spatial reading의 planning 이득은 입증되지 않았다.
- NC 실패는 기존 3/192 → 새 모델 5/192, DAC 실패는 21/192로 같았다. 사전 등록한 안전 기준을 충족하지 못했다. 평균 PDMS 증가만으로 성공으로 판단하지 않는다.
- NC/DAC 평균 BCE는 기존 0.3205, Specific 0.4832로 나빠졌다. 장면 안에서 후보 위험을 구분하는 AUC도 기존 NC 0.898/DAC 0.866 대비 새 NC 0.775/DAC 0.591였다. 이는 원 scorer보다 해당 지표를 더 잘 이해했다는 근거가 없다.
- 다만 train-only 상수 prior보다 BCE가 작고, visual feature를 상수로 바꾸면 BCE가 커지며 선택이 15/192 바뀌었다. 사전 등록한 약한 시각 읽기 진단은 통과했다. 이것이 semantic 특화나 인과적 정보 가치의 증명은 아니다.
- 같은 이미지·후보·공통 색상 척도에서 이전의 하늘 쪽 강조는 줄고 도로가 있는 하단 쪽으로 가중치가 옮겨지는 모습이 나왔다. 고정12장면의 상단/하단 농도는 NC 0.488/1.649, DAC 0.477/1.650다. 좌표 기준 측정이며 하늘/도로 segmentation을 평가한 것이 아니다. **NC는 차량, DAC는 차선이라는 분리는 아직 확인하지 못했다.**
- 마지막 500-step의 개발 PDMS는 shared 75.15, specific 74.71였다. 체크포인트 선택은 튜닝64에서 결정한100step을 유지했다. 추가 epoch 확대 근거로 삼지 않는다.

### 다음 한 실험 — 아직 미실행

**BCE만 사용하는 NC/DAC 직접 예측 대조군**을 같은 예산에서 비교하는 것을 추천한다. 현재 composite PDMS ranking loss는 고정된 다른 네 지표의 예측 오차를 NC/DAC 출력으로 보상하게 만들 가능성이 있다. 이 가능성은 실패 원인으로 확정하지 않았다. 지표별 표현의 학습 목적을 먼저 분리하고, 해당 비교의 checkpoint 선택 기준도 실행 전에 등록해야 한다. 추가 본학습/semantic 감독은 자동 시작하지 않았다.

학습 step 시간 합계는 미니배치 로딩·동기화를 포함한 파이프라인 경과시간이다. 순수 GPU kernel 시간으로 해석하지 않는다. 짧은 profiling의 약16.8초 예상과 실제 약378.4초가 달랐으며, 긴 step의 세부 I/O/대기 원인을 계측하지 않았다. 실제 비용을 기준으로 보고한다.

**사전 등록한 planning gate: 미충족. 시각 읽기 gate: 통과.**

## 구현과 비교 조건

- 고정된 DrivoR 축소 실험 pass5/3200update에서 추출한 DINO 공간 cache를 재사용했다. 전방1대·과거현재2frame·512×256/crop28/padding; 장면당1406×384FP16이다.
- 새 query는 ego11·후보24·metric16을 concat한 뒤 비선형 MLP로 만든다. 단순 가산 query의 metric/candidate 제약을 제거했다.
- Head는 attention의64차원 visual readout만 받는다. 원 scorer feature/logit 및 직접 ego/경로 입력은 없다. Position은key에만 들어가고value에는 들어가지 않는다.
- Shared는 두 metric attention logit을 평균해서 같은 표현을 읽는다. Specific은 개별 attention이다. 같은 파라미터 수/초기state/실제 배치·pair 순서/seed0/batch8/500step/AdamW3e-4/BCE+PDMS-gap ranking을 사용했다.
- NAVTRAIN1024/101recording, 튜닝NAVVAL64/15recording, 개발NAVVAL192/37recording. 그룹 간 recording중복0. 이전에 사용한 개발패널이며 독립 최종NAVTEST가 아니다.
- Native64 후보와 원 scorer의 다른4지표는 고정했다. 새 NC/DAC logit으로 대체해 원 점수식을 적용했다. OFF는 원6logit을 정확히 복원한다.
- 별도 튜닝64의PDMS로100/250/500 중 체크포인트를 선택했다. 두조건 모두500step을 학습했고 profiling 모델은 폐기했다.
- 이전 residual pilot과는 query·head 입력·position의value입력·예측 방식이 함께 달라졌다. 이전 대비 변화를 단일 요인의 효과로 해석하지 않는다. 원 scorer의NC/DAChead도3200update학습된 기존 모델이므로 새 head500step과 학습 이력이 다르다.

## 동일한192장면·64후보 평가

| 조건 | PDMS | 후보최고PDMS | 선택손실 | NC실패 | DAC실패 | NC/DAC평균BCE |
|---|---:|---:|---:|---:|---:|---:|
| unchanged | 78.4753 | 97.5683 | 19.0930 | 1.56% | 10.94% | 0.32050 |
| shared | 79.5545 | 97.5683 | 18.0138 | 2.60% | 10.94% | 0.48356 |
| metric_specific | 79.5545 | 97.5683 | 18.0138 | 2.60% | 10.94% | 0.48317 |
| train_only_metric_prior | 78.6998 | 97.5683 | 18.8684 | 2.60% | 11.98% | 0.52726 |

Specific−shared +0.0000점. Recording bootstrap95%구간 [0.0, 0.0]. 단일seed의 학습 변동을 bootstrap이 반영하는 것은 아니다.
선택한 checkpoint: {'shared': 100, 'metric_specific': 100}. 마지막500step의PDMS도 별도로 기록했다: {'shared': 75.15296173095703, 'metric_specific': 74.71117401123047}.

Train-only metric prior는 학습1024의NC/DAC평균 정답을 모든 후보에 예측하는 진단용 상수다. 파라미터/조건 정보가 같은 대조군은 아니다.

| 명령 | 장면수 | 기존 | Shared | Specific |
|---|---:|---:|---:|---:|
| 좌회전 | 49 | 71.61 | 76.96 | 76.96 |
| 직진 | 106 | 83.65 | 84.22 | 84.22 |
| 우회전 | 37 | 72.75 | 69.61 | 69.61 |

## 추가 학습 없는 기능 개입

| Specific개입 | PDMS | NC/DAC평균BCE | 선택변경 |
|---|---:|---:|---:|
| uniform_attention | 78.6998 | 0.48614 | 7.81% |
| swap_metric_attention | 79.5545 | 0.48331 | 0.00% |
| patch_constant | 78.6998 | 1.01867 | 7.81% |

Patch constant는 학습영상token의평균만 사용한다. 조건query를 통해영상정보가 기능적으로 사용되는지를 확인하는 내부개입이며 실제객체를 제거하는 인과검사가 아니다. Attention교환/균일화도encoder영상을 직접 바꾸지 않는다.

## 표현과 검증

- CPU검사6개와 실제cache검사를 통과했다. 동일value일 때ego/path/position변경에예측불변,첫updatequery/key/valuegradient유한양수,metric×candidate상호작용,invalidtoken제외,head입력경로차단,다른4지표불변/OFF정확성을 확인했다.
- 고정12장면 entropy/max=0.97771,NC/DAC attention TV=0.00660,후보간TV=0.00600. Metric상대tokenodds의후보의존 최대값=0.012180.
- 실제 읽기 가중치를 같은원영상/같은후보/공통척도로 역투영했다. Semantic정답은없고NC차량/DAC도로분할 정확도는측정하지않았다. Featuregradient가존재해도semantic특화나planning이득의증명이아니다.
- Top140token평균대체와random동수대체의지표별logit변화는functional_diagnostics.json에저장했다. 높은attention의유용성을별도확인하기위한진단이다.

## 실제 계산 비용과 보존

- shared: 90,402개학습파라미터,500step GPUstep합계193.32초,step중앙값0.0259초,peakallocated0.134GiB,선택checkpoint100.
- metric_specific: 90,402개학습파라미터,500step GPUstep합계185.13초,step중앙값0.0256초,peakallocated0.135GiB,선택checkpoint100.
- 두조건GPUtraining step합계378.45초. 별도profiling0.63초,평가GPUphase wall7.37초. 학습step시간과cache/평가비용을구분했다.
- 물리GPU0만사용/process2.5GiB정책/전체카드관측최대41.648decimalGB<48GB. 원LPWM보호대상289파일불변. 새encoderforward/새cache생성/새공식candidate평가호출은모두0.
- CUDAtraining시deterministic algorithms와CUBLASworkspace를설정하고단계별환경을기록했다. Bitwise재실행은미검증이다. Cachedforward latency는추가probe만/배치8/fileIO·원encoder·원scorer제외이며병행GPU부하가있다.
- Cachedforward/sample 실측(ms): {'shared': 0.3334810957312584, 'metric_specific': 0.2709461749570134}.

## 해석과 다음 한 실험

Planning gate와 시각정보읽기 gate를 각각 보고한다. 한쪽통과가다른쪽의개선을뜻하지않고,이개발panel의양수차이를논문수준우위로확대해석하지않는다. 공간표현이서로달라도안전지표예측과후보선택에이득이없으면추가본학습은진행하지않는다.
후속우선순위는기능개입에서실제정보사용이확인되는지를기준으로정한다. 이번작업에서fullscale/추가seed/semanticmask를자동추가하지않았다.

## 산출물과 재실행

- `nc_dac_before_after.png`: 이전residual reader/새초기/새학습후의동일이미지비교.
- `nc_dac_shared_vs_specific.png`: 두새조건의NC/DACmap비교.
- `nc_dac_training_evolution.png`: 초기/선택checkpoint/마지막500step. 마지막500은평가선택checkpoint와구분한다.
- `results.json`, `comparison.csv`, `scene_results.csv`, `evaluation_predictions.npz`,검증·manifest·source/inputhash·환경·기능진단을저장했다.

```bash
cd /rhome/junseong/PlanningAwareFuturePrediction
bash scripts/run_direct_metric_visual_probe_pilot.sh
```

완료phase는건너뛴다. 설정이나등록source를바꾸려면새output에등록한다.

## 후보별 위험 구분 추가 진단

후보별 점수를 읽는 능력을 확인하기 위해 NC/DAC0(위험)와1(안전)만 이용해 AUC를 계산했다. NC0.5는 이 binary 진단에서만 제외했다. 장면내AUC는 두label이모두있는장면만집계하며semantic영역정확도는아니다.

| 조건 | NC 장면내AUC | DAC 장면내AUC |
|---|---:|---:|
| unchanged | 0.8983 | 0.8662 |
| shared | 0.7737 | 0.5926 |
| metric_specific | 0.7748 | 0.5913 |
| metric_specific_patch_constant | 0.5000 | 0.5000 |

정보가있는장면수 NC94/DAC163이다. BinaryNC에서제외한partialtarget은174개다. PooledAUC·개입별전체수치는candidate_metric_discrimination.json에저장했다. 재실행은동일환경에서 `python scripts/analyze_direct_metric_candidate_discrimination.py`.

## 튜닝 loss 추가 진단

저장된100/250/500step 모델을 CPU에서 튜닝64의NC/DAC BCE로만 진단했다. 아래PDMS는 본실행에서 기록한GPU값이며 재선택하지 않았다.

| 조건 | step | 튜닝 NC/DAC BCE | 기존 튜닝 PDMS |
|---|---:|---:|---:|
| shared | 100 | 0.50146 | 78.58 |
| shared | 250 | 0.45336 | 73.67 |
| shared | 500 | 0.43041 | 75.69 |
| metric_specific | 100 | 0.50135 | 78.58 |
| metric_specific | 250 | 0.45486 | 73.42 |
| metric_specific | 500 | 0.42691 | 76.43 |

추가 읽기 전용 진단 재실행:

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 PYTHONPATH=src runtime/environments/kjs-nc-dac-visual-probe/bin/python scripts/analyze_direct_metric_candidate_discrimination.py
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 PYTHONPATH=src runtime/environments/kjs-nc-dac-visual-probe/bin/python scripts/summarize_direct_metric_visual_probe.py
```
