# NC·DAC 지표별 시각 표현: 실제 pilot 결과

**결론: 이번 설정에서 planning 개선과 NC·DAC별 의미 있는 공간 특화를 확인하지 못했다.** 구현·CPU/실제캐시 검증·두조건500step·공통후보평가·heatmap·기능적 개입을 모두 완료했다. 사전 등록한 replication gate는 미충족이며 추가 본학습을 시작하지 않았다.

## 실제 비교 조건

- Local DrivoR pass5/3200update checkpoint. 원논문의 충분한 학습을 재현한 baseline이 아니다.
- 전방1대·과거현재2frame·512×256/crop28/우6·하10padding. 백본의기존LoRA/native/원planner모두freeze/eval.
- 같은forward finalDINO norm에서19×37×2=1406개/384차원FP16공간token캐싱. CLS/register prefix를제외했으며,기존register16×256/후보scoredfeature/logit/ego는그대로재사용.
- 공식NAVTRAIN1024/101recording. NAVVAL64/15recording에서checkpoint선택,별도192/37recording평가. Recording중복0,이전에사용한개발패널이므로독립최종NAVTEST가아니다.
- 후보native64 고정,각장면64개공식label재사용. Label/후보수/후보context/백본/기존scorer/원score조합동일. 새로운candidate획득·semantic감독·futurepredictor·encoderFT없음.
- Shared/specific각126978trainableparameter,commonstate/batch8/500step/seed0/AdamW3e-4/배치·pair순서동일. 공간patch추가입력도동일.
- Query=공통ego+path MLP 출력+metric embedding. Shared는2metric attentionlogit평균후softmax,Specific은각metricsoftmax. 따라서Shared는두metric이같은feature를읽는다. Effective query자유도차이는구조개입이며파라미터수동일이모든표현용량동일을뜻하지는않는다.
- 읽은feature와기존candidatefeature를보정MLP에입력하고NC/DAC2logit만boundedresidual을더한다. 다른4component는bitwise원출력유지. 기존candidatefeature우회경로가있다는한계는아래에명시한다.
- 공식NC/DACtarget BCE+실제PDMSgap가중pairwise softplus. 공식evaluator/argmax를미분하지않는다.

## 실제 별도 개발192장면 결과

| 조건 | PDMS | 같은64후보최고PDMS | 선택손실 | NC실패 | DAC실패 |
|---|---:|---:|---:|---:|---:|
| unchanged | 78.4753 | 97.5683 | 19.0930 | 1.56% | 10.94% |
| shared | 78.2336 | 97.5683 | 19.3346 | 1.56% | 10.94% |
| metric_specific | 78.3413 | 97.5683 | 19.2269 | 1.56% | 10.94% |

Specific−shared **+0.1077점**, recordingpaired-bootstrap95%구간**[-1.446248871805236, 1.7384990815265307]**. Specific−원scorer -0.1339점. 개선근거없으며통계적악화확정도아니다.

두조건모두500step학습했다. Tuning기준으로Shared는500step,Specific은100step모델을선택했다. 평가와기능진단의Specific은100step이다. 별도evolutionPNG에는같은이미지/같은후보의마지막500step도포함하며,이를평가checkpoint와혼동하지않는다.

| 주행명령 | 장면수 | 기존 | Shared | Specific |
|---|---:|---:|---:|---:|
| 좌회전 | 49 | 71.61 | 68.79 | 69.45 |
| 직진 | 106 | 83.65 | 84.38 | 84.36 |
| 우회전 | 37 | 72.75 | 73.14 | 72.87 |

## 실제 기능적 의존성과 표현 진단

- Specific은원scorer와14.06%장면에서선택이달라졌다. OFF시모든선택복원/score최대차4.77e-7.
- Uniform attention으로바꾸거나NC/DACattention을교환해도192장면모두최종선택이그대로였다. 점수의작은변화는있지만지표별공간분리가선택에유의미한역할을한다는근거가없다.
- Train-onlymeanpatch로추가patch입력을대체하면선택이4.69%바뀌었다. PDMS는78.7172이다. 이변형은추가patch분기만바꾸며기존candidatefeature의시각정보는유지한다. 영상정보전체의인과효과로해석하지않는다.
- 고정12장면(각실제명령첫4scene)에서attention entropy/maximum=0.99243,NC/DAC totalvariation=0.02333,candidate간totalvariation=0.01008. 이미지전체를넓게읽고두지표의분리는약했다.
- 초기대비attention totalvariation=0.13432; 학습에따른변화와metric전문화/실제선택이득은별개다.
- 상위10%token mean대체의NCabslogit변화 0.00638,random동수 0.00201. DAC는상위 0.00442,random 0.00882. 모든지표에서상위attention이더유용하다는일관된근거없음.
- Featuregradient는유한하며reader가학습되고patch정보에기능적으로의존한다. 이사실만으로차량/보행자/도로를목적별로이해했다고주장하지않는다. Semantic정답감독·객체대응평가를이번에추가하지않았다.

## 검증과 계산 비용

CPU6검사통과: 초기/OFF원출력,나머지4지표불변,같은파라미터수/sharedattention동일,NaNpadding제외,실제lossgradient,feature/metric/candidate기능개입. 실제캐시2step에서head첫gradient와reader후속gradientpositive/finite;cacheinputs/nativeweightsgradient없음. 원modelcacheforwardscene/경로오차0,score9.54e-7,원tensorhash불변. CacheFP16의상대L2오차최대.00020881(0.0209%).

학습코드의모델에는encoder나원scorer모듈이없고캐시만읽는다. Profile가중치는본학습에재사용하지않았다. 첫head0초기화때reader첫gradient0은의도된현상이며첫head업데이트후실제gradient를확인했다.

학습 함수는 deterministic algorithms를 활성화한다. results.json의 environment는 별도 report 프로세스의 스냅샷이며 해당 값은 false다. 학습 시점의 플래그를 별도로 기록하거나 bitwise 재실행 일치를 검증하지 않았으므로, 완전한 재현성을 입증했다고 주장하지 않는다.

- Spatialcache 추출GPU0 loop 49.85초,peakallocated 0.232GiB. 모델초기화전체wall은이loop에포함하지않는다.
- 두조건실제GPU training step합계 52.76초. Cache추출·profiling·tuning·평가/진단비용은별도로기록. 모든GPU작업은물리0번만사용.
- Cached adapter GPUforward/sample(batch8)은Shared 0.190ms,Specific 0.331ms. 단일짧은실측/병행GPU실행이며batch/fileIO/nativeencoder/scorer/deployment총시간미포함;두조건속도차를구조적우위로확정하지않는다.
- 새작업중cache/profile/train/report에서관측한전체GPU0메모리최대 41.743decimalGB<48GB. 기존LPWM등록289파일불변,기존학습계속.
- 공식라벨은이전검증된evaluator의native64만재사용:이번추가공식평가0개. 원라벨생성비용은이전pilot보고서참조. 새보정기의추론순위로고정후보의실제공식score를집계했다.

## 확인한 사실과 추정 원인

확인:loss감소/readergradient/attention변화는있었으나개발PDMS·ranking·NC/DACcalibration개선은없다. Uniform/swap개입도최종선택을바꾸지않았다. Attention이NC차량/DAC도로로뚜렷하게특화됐다는증거가없다.

미확정원인:보정MLP가기존candidatefeature/ego/path를이용하는우회경로,scalar NC/DAC감독의공간식별성부족,additive metricembedding의제한적상호작용,작은학습량/약한원scorer/개발분포shift. 특히현재query는metric과candidate의강한비선형교차작용을보장하지않는다. 이들중하나를실패원인으로확정하거나frozenvisualfeature가무용하다고확대해석하지않는다.

## 다음 한 실험 — 미실행

기존scored feature와원logit을입력에서제외한작은NC/DACprobe를같은frozenpatch+ego+candidate에서학습해,공간feature가해당지표예측정보를실제로제공하는지먼저검증한다. Semantic라벨/새worldmodel/fullscale을추가하기전에직접읽기분기와기존표현우회경로의효과를분리한다. 이번작업에서추가학습을자동기동하지않았다.

## 시각화·재실행

- `nc_dac_attention_overlays.png`: 고정첫장면/명령별원영상,NC전후,DAC,빨강·파랑겹침. 평가Specific100step이다.
- `nc_dac_attention_density.png`: 원가중치density/token그래프. Uniformdensity=1,패널전체같은scale.
- `nc_dac_attention_training_evolution.png`: 같은이미지/후보의초기→100→500step. 마지막500은표현진단이며공식비교의선택checkpoint가아니다.
- 전후시각화는현재frame의실제attention만원crop/resize/pad역매핑했다. RGBdecoder생성영상이아니다. 빨강/파랑은읽기비중이고segmentation/위험GT가아니다.

```bash
cd /rhome/junseong/PlanningAwareFuturePrediction
bash scripts/run_metric_visual_evidence_pilot.sh
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 PYTHONPATH=src runtime/environments/kjs-scorer-selection-pilot/bin/python scripts/visualize_metric_evidence_training_steps.py --configuration configs/metric_visual_evidence/pilot_v1.json
CUDA_VISIBLE_DEVICES='' PYTHONPATH=src runtime/environments/kjs-scorer-selection-pilot/bin/python scripts/package_metric_visual_evidence_results.py --configuration configs/metric_visual_evidence/pilot_v1.json
```

완료단계는skip하며새설정은새output에서등록한다. Config/source/입력hash·checkpoint·environment·검증·JSON/CSV/NPZ는같은폴더에있다. 단일seed/기존개발NAVVAL/약한원baseline이라논문성능이나독립finaltest우위를주장하지않는다.
