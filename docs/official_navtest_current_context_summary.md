# 공식 Drive-JEPA baseline — 현재 입력 기반 navtest 상황별 현황

기준 commit `578be6e`. **추론 재실행 없이 저장된 공식 scene 결과만 분석했다.**
이 분석은 baseline 약점·coverage의 기술 통계이고 상황별 필요한 미래 정보가 다르다는 증거가 아니다.
Selector fitting/하이퍼파라미터 tuning/모델 선택에 navtest 결과를 사용하지 않았다.

## 사전 정의와 completeness

[설정](../configs/analysis/official_navtest_current_context_v1.json)을 결과 join 전에 작성했다.
공식 SceneLoader의 current frame index3에서만 metadata를 만들고 **모든 scene의 label을 완성한 뒤**
score CSV를 읽어 join했다. 미래 경로/heading/target visibility/model error로 구간을 정하지 않았다.

- Ego speed: raw `ego_dynamic_state[0:2]`의 norm, m/s. 나머지2channels는 acceleration.
- Speed bins: `[0,.1)`, `[.1,2)`, `[2,5)`, `[5,10)`, `[10,∞)` m/s, missing/invalid 별도.
- Command: 저장된4channel one-hot의 raw index. 원본 미래 trajectory에서 turn label을 만들지 않음.
  OpenScene source `72860746787a67946bef07aa1f78bbbc6b20e445`의
  [`docs/dataset_stats.md#L122`](https://github.com/OpenDriveLab/OpenScene/blob/72860746787a67946bef07aa1f78bbbc6b20e445/docs/dataset_stats.md#L122),
  [`helpers/driving_command.py#L40`](https://github.com/OpenDriveLab/OpenScene/blob/72860746787a67946bef07aa1f78bbbc6b20e445/DriveEngine/process_data/helpers/driving_command.py#L40)
  기준0left/1forward/2right/3unknown-route-command. Route-derived 명령이지 실제 미래 좌/우회전 관측값은 아니다.
  로컬 export가 정확히 이 commit에서 생성됐는지는 미확인이라 결과 CSV에는 raw index를 보존했다.
- Native recording은 raw current `log_token`, exported log는 `log_name`; 결손을 proxy로 조용히 대체하지 않음.
- 집계: official scene별 동일 가중 평균×100. 아래 recording 수는 여러 범주에 중복 참여하므로 더하면 안 된다.
- 사전 최소표본:100scene/10native recording. 미달 범주는 수치를 보여도 해석 유보.

12,146scene, 136native recording/136exported log. 현재 frame/speed/command/recording ID 누락 **모두0**.
0개 unknown command와 missing category도 JSON/CSV에 남겼다. Speed×command30개 cell도 exhaustive 집계했다.
Source 로그만 읽고 sensor image를 로드하지 않았다. CPU wall16.29초, GPU/추론/학습 없음.

## 속도별 현황 — 단위 점수 %

| 현재 속도 (m/s) | Scenes | Recording / log | PDMS | NC | DAC | TTC | EP |
|---|---:|---:|---:|---:|---:|---:|---:|
| 전체 | 12,146 | 136 / 136 | 89.224 | 99.082 | 96.559 | 96.023 | 83.034 |
| 정지에 가까움 `<0.1` | 487 | 106 / 106 | 93.256 | 100.000 | 100.000 | 98.152 | 85.661 |
| 저속 이동 `[0.1,2)` | 2,052 | 130 / 130 | 94.082 | 99.172 | 99.220 | 98.051 | 89.188 |
| `[2,5)` | 3,458 | 127 / 127 | 87.906 | 98.569 | 96.761 | 95.315 | 80.212 |
| `[5,10)` | 5,181 | 132 / 132 | 87.412 | 99.151 | 95.252 | 95.233 | 81.280 |
| `≥10` | 968 | 71 / 71 | 91.313 | 99.897 | 95.455 | 97.417 | 88.138 |

## 저장 command별 현황

| 현재 stored command | Scenes | Recording / log | PDMS | NC | DAC | TTC | EP |
|---|---:|---:|---:|---:|---:|---:|---:|
| index0 / left | 2,501 | 96 / 96 | 88.806 | 99.340 | 95.762 | 96.242 | 82.663 |
| index1 / forward | 8,070 | 134 / 134 | 90.154 | 99.126 | 97.249 | 96.121 | 84.345 |
| index2 / right | 1,575 | 93 / 93 | 85.127 | 98.444 | 94.286 | 95.175 | 76.910 |
| index3 / unknown-route-command | 0 | 0 / 0 | — | — | — | — | — |
| missing/invalid | 0 | 0 / 0 | — | — | — | — | — |

Right command 범주의 EP가 낮다는 기술 통계는 얻었지만 scene 내용/속도/recording 구성을 통제하지 않았다.
이것은 원인 특정·미래 정보의 필요성·선택 정책의 개선 가능성을 증명하는 결과가 아니다.
Command×speed의 소표본 cell은 사전 설정의 플래그에 따라 해석을 유보한다.

## 원본 보존과 재현

Baseline score SHA256 before/after:
`56010fc01ae91818daef49ec63f38614bf1db6913b6e672c354d650891f99ce5`.
Protocol config SHA256:
`484b867dd0cb8eaeb82cef2889b8d49eb985795fdba90541e1aee39fe9f2fd02`.
원본 전체 mean89.22432019890219와 새 집계가 일치한다.

```bash
cd /rhome/junseong/PlanningAwareFuturePrediction
env CUDA_VISIBLE_DEVICES='' PYTHONNOUSERSITE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  runtime/environments/drive_jepa_official_evaluation/bin/python \
  scripts/summarize_official_navtest_current_context.py
```

위는 이번에 실행한 명령이다. 기존 결과 directory가 있으면 fail하여 덮어쓰지 않는다.
재실행이 필요하면 새 config/result path를 명시한다. 이번에는 반복할 필요 없다.

共有:
[summary JSON](../results/foundation_selection/current_context_navtest_v1/context_summary.json),
[全区分CSV](../results/foundation_selection/current_context_navtest_v1/context_summary.csv),
[scene metadata CSV](../results/foundation_selection/current_context_navtest_v1/current_input_metadata.csv).
새7개 CPU tests와 기존5개 official 집계 tests를 통과했다. 경계값/velocity 단위/raw command/결손/
recording count/전체 partition/hash를 검사했다.
Pandas empty-bin/NaN CSV serialization warning은 빈 범주의 요약값에서만 발생했고 scene score의 invalid/비유한 값은0이다.
학습·PDMS 계산·checkpoint 재검사는 아니다.
