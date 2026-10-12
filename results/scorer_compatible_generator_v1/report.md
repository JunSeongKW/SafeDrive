# Scorer가 정확히 평가하는 후보 분포를 학습하는 generator pilot

## 결론

등록한 후속 확대 기준 미충족: 현재 방법의 planning 개선은 입증되지 않았다. 대규모 학습을 자동 시작하지 않는다.

C−B 실제 선택 PDMS +0.0892점, recording paired bootstrap95%CI [-0.07742758113577661, 0.3016813284513116]. **Teacher 오차 감소·MMD 감소와 실제 주행 이득은 구분한다.**

## Stage 0 — 설계와 선행연구

[학습 전 등록한 연구 검토·수식·gate](stage0_research_design.md). 이 방법은 조건부 후보분포 증류와 겹치며 frozen scorer/reparameterization만으로 novelty를 주장하지 않는다. C의 scoring-accuracy 항이 품질-only B보다 추가 이득을 주는지가 핵심이다.

## Stage 1 — 실제 cache·teacher·gradient 검사

NAVTRAIN 학습128/53recording, 점검32/15recording, 기존 노출 NAVTEST 개발64/38recording. 집합 간 recording 중복0. 공개 checkpoint가 trainval로 학습돼 있어 NAVTRAIN 점검은 원 모델의 완전 미관측 데이터가 아니다.
현재4camera/1148×672/1image frame, 이전·현재 ego history 유지. 실제 scene64×256·최종 decoder trajectory token64×256·ego1×256 FP32 cache. 원head 복원·실제 finite gradient·현재 sample 대응·원checkpoint 보존 검사 통과. NAVTEST target는 teacher/학습/체크포인트 선택에 사용0.

| 학습용 teacher | 실제 품질 가중 평균 | scorer 예측오차 가중 평균 | target 유효 후보 수 중앙값 |
|---|---:|---:|---:|
| B_quality | 97.09 | 12.40 | 51.34 |
| C_compatible | 96.85 | 8.87 | 48.91 |

이 표는 정답을 이용해 만든 학습용 teacher의 통계다. 학습된 generator의 성능이나 inference 가능 upper-bound 정책이 아니다. 품질 손실 최대1점 제약도 teacher에만 적용된다.
[Stage1 teacher 이미지](stage1_teacher_distributions.png): 현재원native64의실제/예측점수·B/C확률·Ctargetmass의배치를보여준다. 점 크기는 target probability, 색은 실제PDMS다.

## Stage 2 — generator 학습과 분포 이동

원generator `traj_head[-1]` 복사본 1,341,464parameters를 B/C 각각300step/seed0/batch8/AdamW1e-5로학습했다. 원본encoder/ego/query/trajectorydecoder/scorer는 고정·eval. 학습중 encoder forward0. 새로운필터·scorerhead를추가한실험이아니다.
Gradient: MMD+anchor+coverage loss→후보 좌표→native output MLP head. Official evaluator와 scorer를 역전파하지 않는다. 두 조건의 차이는 teacher target뿐이며 초기화/배치순서/학습량/나머지loss는동일하다.
[학습 loss·실제 좌표 이동](stage2_training_and_movement.png) / [고정 장면의 후보 형상](stage2_generated_candidate_shapes.png). Target support가 원native64로제한되고출력head만갱신하므로전체generator적응·CEM학습·새로운분포모드발견의검증은아니다.

## Stage 3 — 실제 경로 재채점·공식 평가

최종표는두조건모두고정300step이며test점수로checkpoint를고르지않았다. Scorer는매번각조건의완전64경로집합을채점했다. 변경된모든경로의공식라벨을새로계산했고원라벨을재사용하지않았다.

| 조건 | 선택 PDMS | 후보 최고 PDMS | 선택손실 | 후보 MAE | 후보 RMS 퍼짐(m) | 평균 좌표 이동(m) |
|---|---:|---:|---:|---:|---:|---:|
| A_original | 93.6087 | 99.8745 | 6.2658 | 17.4045 | 2.060 | 0.000 |
| B_quality | 92.0348 | 99.8906 | 7.8558 | 17.2452 | 1.931 | 0.166 |
| C_compatible | 92.1240 | 99.8574 | 7.7334 | 16.7826 | 1.896 | 0.213 |

| 조건 | 선택 NC 실패 | 선택 DAC 실패 | 선택 TTC 실패 | 선택 comfort 실패 | 진행률 평균 |
|---|---:|---:|---:|---:|---:|
| A_original | 1/64 | 0/64 | 5/64 | 0/64 | 0.9310 |
| B_quality | 1/64 | 1/64 | 5/64 | 0/64 | 0.9151 |
| C_compatible | 1/64 | 1/64 | 5/64 | 0/64 | 0.9172 |

| 조건 | 좌회전(n=21) | 직진(n=22) | 우회전(n=21) |
|---|---:|---:|---:|
| A_original | 88.4972 | 97.5651 | 94.5754 |
| B_quality | 83.6780 | 97.6049 | 94.5564 |
| C_compatible | 83.7290 | 97.7887 | 94.5847 |

[Stage3 실제 planning/분포 비교](stage3_planning_and_distribution_comparison.png). PDMS는각장면의Top1선택경로를공식평가한64장면평균이며,후보64점수의평균이아니다. Oracle도그조건의생성64개중실제최고다.

## 실패 위험과 해석

- 오류가작은원native후보쪽으로MMD를줄여도이동한좌표에서scorer오류가작아진다는보장은없다. teacher는고정64context에서의원후보오차라새context/경로에서달라진다.
- under-scored좋은후보를피하는경향은좋은주행모드를버릴수있다. 그래서품질-only B·target품질guard·공통coverage·최종oracle최고를함께확인한다.
- 후보퍼짐감소가선택손실감소를만들었더라도oracle최고하락·NC/DAC추가실패가있으면등록gate미충족이다. MMD를성공기준으로바꾸지않았다.
- nativeposehead는제어범위제약을직접보장하지않는다. 공식tracker/comfort로실제검사했다. CPU두조건의작은trainset300step/한seed개발pilot이므로다른기법의성공/불가능성으로확대하지않는다.
- 이64개NAVTEST장면은이미분석에노출됐다. 독립최종평가·전체NAVTEST·EPDMS결과가아니다. 대규모자동후속학습없음.

## TOAD 관련 질문

앞선CEM OFF 동일128후보에서원DrivoR선택93.6972/TOAD배포가중치95.9485는각scorer의Top1선택경로실제PDMS평균이다. 후보128개에준점수의평균이아니다. 동일후보에서도차이가있으므로그차이는CEM재샘플링효과로설명할수없다. 두weight/학습recipe차이는남는다.
CEM은frozen scorer를변경하지않고높은예측reward의후보를찾는다. 최종PDMS·그분포에서의점수MAE·후보순위정확도는별도로평가해야한다. 이pilot은teacher정확도를반영해generator를학습하는별도가설이다.

## 비용·검증·다음 단계

비용: {"cache_gpu_wall_seconds": 207.99287152197212, "cache_maximum_card_used_bytes": 42010214400, "head_training_cpu_seconds": 22.94079414900625, "prediction_cpu_seconds": 6.938855982996756, "native_labels_cpu_seconds": 62.44275539700175, "modified_labels_cpu_seconds": 120.30679678899469, "native_official_candidates": 10240, "evaluation_official_candidates": 30720, "training_gpu_seconds": 0, "head_cpu_inference_mean_seconds": 0.0018224219892848245}

등록gate: {"C_minus_B_selected95ci_lower_above0": false, "C_minus_A_oracle_drop_at_most1point": true, "C_no_extra_selected_nc_or_dac_failure": false, "all_passed": false}

원cache/공개checkpoint·기존LPWM등록293파일과대기열보존. 공식pair evaluator최대차<1e-8 및원경로새라벨동등성검사통과. CPU학습의작은공개head검사에서gradient가존재한다는사실은planning이득증거가아니다.
다음하나는이결과에따라정한다. 등록gate통과시두학습seed와독립recording을늘려C−B를복제한다. 미통과시추가모듈/학습량을늘리기전에 teacher distribution의matching과실제새경로ranking의불일치를분해한다. 자동실행하지않았다.

```bash
bash scripts/run_scorer_compatible_generator_pilot.sh
```

원cache준비가없는환경은stage0설정과subset_manifest에기록된publiccheckpoint/원본NAVSIM/metriccache가필요하다. Pythonenv와local경로는cache_complete/prediction_complete에기록했다. 결과수치JSON/장면별CSV/config·단계별검사·checkpoint는보존했다.

## NAVSIM 실제 사례

- [largest_improvement / 46369836a9b654ee](stage3_largest_improvement_navsim.png): post-hoc C−A 변화 사례.
- [largest_degradation / 131ab0263b07507c](stage3_largest_degradation_navsim.png): post-hoc C−A 변화 사례.

## Stage 4 — 손실 원인 분해(추가 학습 없음)

| 조건 | 원query를 유지하고 좌표만 변경한 효과 | 변경된set에서 재선택 효과 | 총 변화 |
|---|---:|---:|---:|
| B_quality | -1.4791 | -0.0948 | -1.5739 |
| C_compatible | -1.0463 | -0.4383 | -1.4847 |

Query identity가유지되는같은head의미세조정이므로이분해가가능하다. 이는정확한점수차회계이며인과분리실험은아니다.
가장큰악화는131ab0263b07507c이다. A와C가모두query37를선택했지만그경로의좌표가평균0.133m/최대0.367m이동하면서실제PDMS100→0, DAC1→0이됐다. C의DAC예측은0.843여서scorer는도로이탈을충분히낮게평가하지못했다.
따라서주요악화사례를'다른후보를잘못고름'만으로설명할수없다. 미세한좌표이동이안전경계를넘고고정scorer가그위험을놓쳤다. 이한장면을제거하면평균을다르게보이게할수있지만,전체64장면을주결과에그대로남겼다.
[MMD 적합과 planning 불일치](stage4_matching_vs_planning_diagnosis.png). 고정train전체의MMD감소를실제측정했다. 새경로의oracle라벨을재확인해야하며,오래된nativecandidate teacher와의거리만으로정확한새분포가보장되지않는다.
이결과를본뒤성공기준을변경하거나현재generator를추가학습하지않았다. 가장작은다음검토는학습용생성경로의officiallabel을한번갱신해target의stale-label문제가얼마나큰지확인하는것이다. 평가NAVTEST라벨로teacher를재구성하지않는다.

### 원head CPU 재생 정밀도 확인

도로경계악화사례에서학습하지않은원head를CPU로재생한64개경로와원GPU캐시64개를공식평가했다. NC/DAC/TTC/comfort모두동일,전체지표최대차0,선택query37은양쪽모두100점이다. 따라서이번100→0실패를원head의CPU/GPU정밀도차로설명할근거는없다. 추가학습·GPU0,검사범위는이한장면이다.

### 실제 tracked footprint와 도로 경계

[확대 시각화](stage4_tracked_footprint_road_boundary.png): 공식tracker rollout에서C는2.3초에처음drivable영역밖corner가발생했다. A는전구간0회/C는10timestep이다. DAC는중심선만이아니라차량footprint4corner중하나라도주행가능영역밖에나가면0이된다. 앞선BEV는요청waypoint였고이그림은실제trackedfootprint다.
