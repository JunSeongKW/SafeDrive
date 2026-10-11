# Frozen DrivoR scorer 검증

원래 연구의 하위 질문: 시각 조건부 탐색을 학습하기 전에, 고정 scorer가 유용한 제어 후보를 판별하는가?

64개 기존 공식 navval 개발 장면. 좌21/직진22/우21; 점수 확인 전 seed로 선택했다. 독립 최종 test·전체 NAVTEST가 아니다.

**결론: 사전 등록한 scorer 품질 기준 미충족 — sampler 학습 보류.**

| 후보 묶음 | 후보 수 | 선택 PDMS | 후보 내 최고 PDMS | 선택 손실 | 충돌·도로이탈 선택률 |
|---|---:|---:|---:|---:|---:|
| native64 | 64 | 76.8109 | 98.8412 | 22.0303 | 9.38% |
| nearby64 | 64 | 78.8271 | 84.6712 | 5.8441 | 6.25% |
| nearby16 | 16 | 80.1637 | 84.5054 | 4.3417 | 6.25% |

## 통제와 해석

- Native64는 기존 generator의 64개 경로다. Nearby64는 원래 선택 경로 fallback 1개, 제어→BM roundtrip center 1개, TOAD 초기 std의0.25배로 생성한62개 후보다. 학습한 sampler가 아니다.
- Nearby16은 동일 nearby pool의 앞16개 경로를 다른 candidate context로 재채점한다. 공통경로의 점수 변화는 scorer의 candidate self-attention에 따른다. M16의 실제 TOAD std1.0 분포를 검증한 것은 아니다.
- Native/nearby 모두 원래 neural score로 선택한다. 공식 PDMS로 얻은 best는 사후 oracle 진단이며 inference 선택에 사용하지 않는다.
- 품질 기준: 모든 native64/nearby64/nearby16의 평균 selection regret≤5점, native 대비 collision-or-offroad 선택률 악화≤1pp. 결과를 보고 기준을 완화하지 않았다.
- 공식 simulator는 전체후보를 일괄 처리하고 progress를 reference+개별후보 기준으로 복원한다. selected/best/center/gradient 후보 일부를 수정 없는 공식 evaluator와 모든 하위 지표까지 대조했다.
- 공식 evaluator 최대차 0. Cached/native neural score 최대차 0.

## Gradient 방향의 실제 효과

Scorer의 BM center score만 미분하고, 나머지63개 후보는 고정했다. reference std 좌표에서 unit gradient 방향과 반대 방향으로 동일 크기 보정한다. 이 검사는 optimizer 학습이 아니다.

| 보정 반경 | Neural score 증가율 | +gradient 실제 PDMS 변화 | 반대 방향 실제 PDMS 변화 | Neural↑·PDMS↓ 비율 |
|---|---:|---:|---:|---:|
| 0.05 | 100.0% | +0.6240 | +0.0276 | 71.9% |
| 0.1 | 100.0% | +0.5972 | +0.0546 | 71.9% |
| 0.25 | 100.0% | +1.9867 | +0.1311 | 71.9% |

**평균 상승은 소수 장면의 큰 개선에 의존한다.**
세 보정 반경 모두 평균 변화의 recording-bootstrap95%구간이0을포함한다. 71.9%의하락은1e-6점 이상 변화 기준이며 대부분 작은 변화다. 모든 반경에서1점초과 하락은0장면이었다.

| 보정 반경 | 평균변화95%구간 | 중앙값 변화 | 0.1점초과 하락 | 1점초과 개선 |
|---|---|---:|---:|---:|
| 0.05 | [-0.0320, +2.0971] | -0.0116 | 2/64 | 1/64 |
| 0.1 | [-0.0643, +2.0569] | -0.0288 | 12/64 | 1/64 |
| 0.25 | [-0.1440, +5.5536] | -0.0676 | 28/64 | 2/64 |

Native 후보의 neural-vs-official Spearman 장면평균은 +0.3444, nearby64는 -0.3659이다. Near pool에서는53개의비상수 장면만 포함한다. 좁은후보군에서progress의작은차이가상관에영향을줄수있으므로 음수상관을큰안전악화의증거로바꾸지않는다.

## 선택 손실의 하위 지표

| 후보 묶음 | 선택/최고 progress | 선택/최고 DAC | 선택/최고 TTC | 선택/최고 DDC |
|---|---|---|---|---|
| native64 | 0.6216/0.9722 | 0.9062/1.0000 | 0.9531/1.0000 | 0.9531/0.8281 |
| nearby64 | 0.6418/0.6915 | 0.9375/0.9688 | 0.9375/0.9844 | 0.9453/0.9531 |
| nearby16 | 0.6427/0.6875 | 0.9375/0.9688 | 0.9688/0.9844 | 0.9531/0.9531 |

NAVSIMv1 공식 기본점수와 이neural scorer의 DDC 가중치는0이다. 따라서oracle최고PDMS를 모든주행조건을만족하는안전경로라고해석하지않는다. Native oracle은DDC평균0.8281로선택경로0.9531보다낮다. 원래metric을변경하지않고DDC도별도보고한다.

BM roundtrip 평균 ADE 0.4112m. Gradient 성능은 원래 base가 아니라 BM center에 대한 대응 변화다.
현재·이전 ego pose와 실제 timestamp 차이로 yaw rate를 복원했다. 미래 관측은 feature/comfort boundary에 입력하지 않았다.

## 비용·보존

Cache 생성 CPU 219.3s, cached scorer 진단 7.3s, 공식 평가 37.0s /2worker. GPU0초·optimizer0step.
Encoder는 cache 생성 때만 실행했고 cached gradient 단계 호출0회, 전체 frozen weight digest 불변이다. 기존LPWM 학습 source/config SHA를 전후검사했다.

## 이전 검토의 체크포인트 정정

이전192장면 selected72.76/oracle97.81/regret25.05는 pass3.pt(1920update) 결과다. 이전 타당성 검토에서 이를 pass5 scorer와 동일 checkpoint로 연결한 해석은 정정한다. 이번은 pass5.pt(3200update)의 별도 실제 측정이며 과거 로그·수치는 변경하지 않았다.

## 다음 작업

Hold sampler training; inspect scorer ranking and off-support calibration before optimizing its reward
이번 검증은 scorer 품질과 국소 gradient를 검사하며 visual 정보의 추가 가치·sampler 성공·논문 novelty를 입증하지 않는다. Bootstrap은 recording 변동이며 학습 seed 변동이 아니다.

![검증 요약](scorer_calibration_summary.png)

## 재실행

```bash
cd /rhome/junseong/PlanningAwareFuturePrediction
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 PYTHONPATH=src runtime/environments/kjs-lpwm-drivor-joint/bin/python -u scripts/calibrate_frozen_drivor_scorer.py --phase run --configuration configs/residual_control_sampler/scorer_calibration_20261011.json
```

완료 marker가 있는 phase는 재실행하지 않는다. 새로운 조건은 새 config/output 폴더로 등록한다.
