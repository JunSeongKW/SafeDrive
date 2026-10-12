# D — 예측오차만으로 목표 분포를 만드는 generator 추가 실험

## 결론

D의후보예측MAE는원A17.4045→16.6068로줄었으나실제선택PDMS는A93.6087→91.3739였다. 현재accuracy-only목적의planning이득을입증하지못했다. 추가fullscale학습은시작하지않았다. 작은개발표본이므로모든유사방법의불가능성이나평균악화의통계적확정으로확대하지않는다.

사용자 요청에 따라 실제 품질 가중·품질 guard·품질 coverage를 모두 제거했다. 원좌표 anchor만 유지했고 원래 head/cache/배치순서/seed0/300step/batch8/LR1e-5는 B/C와 같다. B/C에서 이어학습하지 않았다.

```text
q_D(i|scene) = softmax(-abs(predicted_PDMS_i - actual_PDMS_i) / 0.15)
L_D = MMD²(uniform generated64, weighted native64 by q_D) + 0.1 × native-coordinate anchor
```

**작은 예측오차가 좋은 주행을 뜻하지는 않는다.** 실제0점 후보를0점으로맞혀도D는높은가중치를줄수있다. 실제정답은오차정의에만사용하며품질선호나안전teacher로는사용하지않는다. NAVTRAIN128에서만target/gradient를만들고NAVTEST는평가에만사용했다.

학습용Dteacher실제품질평균83.59,예측오차평균5.38,실제0점후보targetmass10.71%다. Teacher통계는inference성능이아니다.

| 조건 | 선택 PDMS | 후보 최고 | 선택손실 | 후보 예측 MAE | NC/DAC/TTC 실패 수 |
|---|---:|---:|---:|---:|---|
| A_original | 93.6087 | 99.8745 | 6.2658 | 17.4045 | 1/0/5 |
| B_quality | 92.0348 | 99.8906 | 7.8558 | 17.2452 | 1/1/5 |
| C_compatible | 92.1240 | 99.8574 | 7.7334 | 16.7826 | 1/1/5 |
| D_accuracy_only | 91.3739 | 99.8376 | 8.4637 | 16.6068 | 1/1/6 |

위PDMS는동일NAVTEST64/38recording에서각scorerTop1선택경로를공식평가한평균이다. 후보점수평균이나전체NAVTEST결과가아니다. 동일publicscorer/full64context를사용하고변경된경로official라벨은새로계산했다.

D의대응차이와95%recording구간: {"D_minus_A_original": {"selected_pdms_difference": -2.2348191522104415, "recording_bootstrap95ci": [-5.828397166163923, 0.07893123231503933]}, "D_minus_B_quality": {"selected_pdms_difference": -0.6609691506473478, "recording_bootstrap95ci": [-2.064256726334907, 0.1599117036430873]}, "D_minus_C_compatible": {"selected_pdms_difference": -0.7501543551665343, "recording_bootstrap95ci": [-2.1404412444899616, 0.02352944931214385]}}

단일seed/128학습장면/반복노출개발패널이다. D는target품질항뿐아니라explicitqualitycoverage도OFF이므로C−D를어느한항만의순수효과로부르지않는다. 원본A/B/C수치·checkpoint·cache는변경하지않았다. 전체후보MAE감소와최종선택정확도는별도결과이며, 낮은품질targetmass가이번선택악화의원인이라는인과분리도실시하지않았다.

[A/B/C/D 비교·D 학습곡선](accuracy_only_comparison_and_training.png). 현재조건에서예측오차를작게만드는것과좋은후보·선택경로를만드는것이함께개선됐는지각지표로판단한다.

[D가 선호하는 학습 후보](accuracy_only_teacher_distribution.png): 좌측actual/prediction,우측D확률. actual품질로가중하지않는다.

비용: {"training_cpu_seconds": 10.973385403980501, "prediction_cpu_seconds": 2.619068227999378, "complete": true, "official_candidates": 12288, "unchanged_pair_checks": 630, "maximum_pair_evaluator_difference": 0.0, "cpu_wall_seconds": 58.046855641005095, "gpu_seconds": 0, "new_gpu_seconds": 0}. GPU/encoder호출0,모든학습stepgradient유한. 원래LPWM등록293파일불일치0·학습대기열보존. fullscale자동확장없음.

```bash
bash scripts/run_scorer_compatible_accuracy_only.sh
```
