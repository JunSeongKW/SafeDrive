# 선택 오류 중심 sampler와 scorer 보정: 실제 pilot 결과

**결론: 이번 설정에서 성능 개선을 확인하지 못했다. 사전 등록한 후속 확장 기준도 미충족이다.**
코드 구현·검증·세 조건 학습·공식 평가를 완료했으며, 추가 학습을 자동 시작하지 않았다.

## 비교 조건

- 기존 DrivoR 축소실험 pass5/3,200update. 공개 논문의 충분히 학습된 모델 재현은 아니다.
- 전방1대, 과거·현재2frame,512×256. Backbone·기존 generator·scorer 가중치 완전 고정.
- 실제16×256FP32 장면 토큰, ego, 후보 경로, scorer 후보 표현256차원과6개 logits를 캐싱.
- Scorer 보정기89,222파라미터, 동일 초기값·AdamW3e-4·batch16·500step·seed0.
- 각 조건은 장면당 native64+추가32후보의 공식 라벨96개를 사용. Scorer 손실은6개 하위 지표 BCE와 실제 점수 차이로 가중한 pairwise ranking loss.
- A: 무작위 제어 family 획득. B: GT·지도·미래 객체 geometry 경계 주변 획득. C: 장면·ego·기존 제어·분산 조건부 획득.
- C의 sampler는56,542파라미터이며 공통128학습 scout만으로300step 지도학습. 추론에서는 sampler를 사용하지 않고 같은 generator64후보를 채점한다.
- B는 geometry로 직접 경계를 찾는 참고 구현이다. Versatile Samples 논문의 tracked simulator margin solver를 그대로 재현한 조건이 아니다.

## 데이터와 실제 점수

NAVTRAIN1,024장면/101recording. NAVVAL64장면/15recording으로100/250/500체크포인트를 선택하고, 별도192장면/37recording에서 아래 결과를 얻었다. 집합 간 recording 중복0이다. 기존에 개발에 사용한 NAVVAL이며 독립 최종 NAVTEST가 아니다.

| 조건 | 선택 PDMS | 같은 후보 내 최고 PDMS | 선택 손실 | 충돌 또는 도로 이탈 선택률 |
|---|---:|---:|---:|---:|
| 기존 고정 scorer | 78.4753 | 97.5683 | 19.0930 | 12.50% |
| A 무작위 획득 + 보정 | 78.0014 | 97.5683 | 19.5669 | 12.50% |
| B 경계 획득 + 보정 | 77.5620 | 97.5683 | 20.0063 | 13.02% |
| C 선택 오류 획득 + 보정 | 77.5871 | 97.5683 | 19.9811 | 13.02% |

모든 조건의 평가 후보와 candidate context가 동일하다. 후보 최고점이 낮아져 선택 손실이 작아지는 효과는 없다.

- C−A −0.4142점, recording paired-bootstrap95%구간[−1.3969,+0.0756]. 추가 이득 근거 없음; 통계적 악화를 확정하는 결과도 아니다.
- C−기존 −0.8881점,95%구간[−3.4336,+1.7334]. 평균 개선 근거 없음.
- C는 좌회전71.1544/직진83.2668/우회전69.8346점. 표본49/106/37이며 상태는 실제 one-hot 주행 명령에서 읽었다. 정지·교차로 라벨을 임의로 만들지 않았다.
- 평가 후보 최선의 DDC도 별도 저장했다. NAVSIMv1 기본 DDC 가중치0이므로 높은 oracle PDMS를 모든 주행 규칙 준수로 해석하지 않는다.

## 확인된 사실과 아직 추정인 부분

**확인된 사실**

- 기능적 변화는 발생했다. C는 기존 scorer와 약51.0%장면에서 다른 경로를 선택했다. 보정 OFF 시 원래 선택 index는 모든 평가 장면에서 복원된다.
- 시각 입력을0으로 바꾸면 sampler 확률이 변한다. 이 검사는 기능적 의존성이고 영상 정보의 인과적 성능 이득 검사가 아니다.
- 학습 native 후보에서 C는 기존 대비+0.6372점이나 별도 개발에서는−0.8881점이다. A도 학습+0.5505/개발−0.4739점이다.
- C의 gap-weighted pair 정확도는 기존82.90%에서81.52%로 낮아졌다. BCE/ranking 학습 loss가 내려간 것이 실제 선택 개선으로 이어지지 않았다.
- C의 추가 후보에서 충돌/도로이탈 비율은6.16%, A는6.82%였다. 따라서 현재 획득 proxy가 실제로 더 유익한 오류를 더 많이 찾았다는 증거는 없다.
- B의 추가 후보 comfort 실패율은26.23%였다. 경계 샘플 생성 방식과 실제 tracker 동작 사이의 차이를 무시할 수 없다.

**가능한 원인, 미확정**

작은 보정기의 일반화,128scout로 학습한 획득 proxy의 타당성, 학습 acquired32와 평가 native64의 candidate context 차이, six-subscore BCE와 최종 hard 선택 목적 사이의 불일치가 함께 남는다. 이 결과를 sampler 자체의 불가능성이나 시각 feature 무용성으로 확대하지 않는다.

## 검증과 비용

- CPU 기능 검증5개 통과: 초기/OFF 복원, 원 score 조합, head·adapter gradient, 순위 loss 방향, sampler gradient·조건 의존성.
- 원래 경로와 cache가 선택하는 index1,280/1,280일치. 원래/cached scorer 최대 오차9.54e-7.
- 수정 없는 공식 evaluator8,300후보 대조, 모든7개 지표 최대 오차0.
- 실제 공식 후보188,416개 평가. 라벨 생성 CPU wall511.86초,4worker.
- 성공한 cache 추출은 GPU0 연산 루프 약35.18초, peak allocated0.226GiB/ reserved0.260GiB. 이 시간은 모델 초기화 전체 wall을 포함하지 않는다.
- 보정기 학습 CPU A7.11/B6.41/C5.83초. Sampler0.57초. 작은 캐시 MLP만 학습하므로 영상 encoder 학습 시간과 다르다.
- 최초 sandbox에서는 CUDA가 보이지 않아 CPU cache 추출을 중단했다. 부분 cache·진행기록은 보존; 실제 비교는 새 GPU cache만 사용. 그 실패 시도의 비용은 성공 경로 비용에 포함하지 않았다.
- 모델 환경의 nuplan 부재로 boundary 단계1회 실패; 같은 geometry 알고리즘만 기존 공식 평가 환경으로 분리했다. 기존 scout 라벨과 sampler 상태 재사용, config/model 및 완료·학습 함수 AST 불변 검증.
- 기존 LPWM 학습 등록source/config289개 불변, 기존 학습은 계속 진행한다.

## 추천하는 다음 한 실험 — 아직 미실행

**Native64 후보만 사용해 같은 작은 scorer 보정기를 학습하는 대조군**을 먼저 검토한다. 이번 동일1024장면/초기화/500step/평가192를 사용하고 획득 sampler를 제외한다. 같은64candidate context에서 보정기 자체가 이득을 줄 수 있는지 확인한다. 이를 통과하기 전에는 sampler 획득 전략이나 본 학습 규모를 늘리지 않는다.

## 파일과 재실행

- `report.md`: 전체 설계·통제·기본 결과
- `results.json`, `comparison.csv`, `scene_results.csv`: 실제 수치
- `validation.json`, `unit_validation.json`, `cached_selection_matching.json`: 검증
- `acquisition_and_ranking_diagnostics.json`, `training_vs_development.json`: 실패 진단
- `selection_regret_comparison.png`: 고정 후보 점수·선택 손실·학습된 획득 확률
- `selected_trajectory_examples.png`: 최대 개선/최대 악화/중앙 변화 사례. 모집단 대표 사례가 아니다. 그림은 요청 waypoint 경로이고 점수는 공식 tracker rollout의 평가다.

```bash
cd /rhome/junseong/PlanningAwareFuturePrediction
bash scripts/run_scorer_selection_pilot.sh
```

완료 단계는 다시 학습하지 않는다. 과학 설정을 바꿀 때에는 새 config/output 등록을 사용한다.
