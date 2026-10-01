# 연구 질문·target 결정 v1 — 고정 K는 개발 기반

2026-10-01, 기준 `1231767` 이후 사용자/ChatGPT 지시. **결정 문서이지 학습 결과가 아니다.**

## 지금 답할 한 문장

**동일한 현재 정보·객체 선택 규칙·K·예측 branch에서 어떤 미래 감독(영상 latent / 객체 공간 상태 /
혼합)이 현재-only 또는 감독 없는 branch보다 log-held-out ego 경로 예측에 유용한가?**

이는 최종 planning 연구에 앞선 target 선정 질문이다. 작은 ADE/FDE 비교는 공식 주행 성능이나
원래 H1/H2 검증을 대체하지 않는다. 유효한 target을 찾은 뒤 **미래 예측의 추가 예산이 주는
이득이 맥락마다 달라지는가?**를 고정 K별 동일 조건 비교로 묻는다. 동적 K/horizon은 구현하지 않는다.

## 현재 확정한 범위와 미정 사항

- **개발 기반**: 공식 Drive-JEPA frozen video encoder+GT box/track ROI, 신규 작은 predictor/planner.
  공식 planner는 복제/재현하지 않았다. SafeDrive baseline/retraining은 계속 중단한다.
- **현재 구현**: 객체 instance를 선택하고 각 선택 객체의 visual+spatial 미래를 함께 예측한다.
  정보 종류 자체를 선택하는 모델이 아니다. Frozen visual target와 explicit GT-state 감독의 혼합이다.
- **첫 탐색**: 고정 nearest-current-distance, K4, future8/~4s, 동일 현재 입력 및 branch 폭.
  Selector를 학습하지 않고 target 효과부터 분리한다. 현재 ST는 이후 비교 후보로 보존한다.
- **최종 기여 후보**: 맥락별 미래 예측 필요성과 추가 예산의 marginal planning benefit을 측정하고,
  근거가 생기면 예산 배분을 학습하는 방법. **Novelty 미확정**이며 단순 동적 K만으로 확보되지 않는다.
- **미정**: visual/spatial/mixed 중 최종 target, multiview/검출·association, 공식 evaluator,
  독립 holdout, encoder fine-tuning, 학습·예산 효과와 최종 선행연구 대비 delta.

## 직접 선행연구 때문에 제외하는 기여 주장

- EgoFSD의 ego-conditioned 객체 선택→joint prediction/planning과 중복하므로, “중요 객체를 고른다”
  또는 고정 K 선택만을 첫 기여로 내세우지 않는다.
- ForeDrive의 미래 latent 소비와 중복하므로 “JEPA 예측을 planner에 넣었다”만을 기여로 내세우지 않는다.
- 원문/공식 repository 근거와 gradient 구현의 미확인은
  [직접 선행연구 감사](egofsd_foredrive_evidence_audit.md)에 구분했다.
  ST의 gradient 연결 차이도 성능·학습량·예산/기하 prior를 통제하기 전에는 연구 기여가 아니다.

## 데이터 조사에 따른 현재 판단

16recording/128구간에서 front 후보>K4는45.3%, 4s visual target 잔존65.4% vs spatial93.6%.
좌회전 heading proxy11구간은4s visual36.4%; right-turn proxy0, merge label 없음.
따라서 **작은 front-only target 탐색은 가능**, **최종 상황별 선택·예산 주장은 coverage 미해결**이다.
미래 target-valid로 현재 후보/학습 구간을 걸러 문제를 숨기지 않는다.
Primary C/D/E는 같은 visual∩spatial mask를 사용해 감독 표본 차이를 통제한다.
실제 수치·split은 [데이터 유효율 보고](navsim_visual_target_coverage.md).

## 다음 실행과 진행/중단 판단

1. 공유 split의 train277/development96 window를 유지한다. Train only normalization, 같은 초기값/순서,
   A/B/C/D/E 각각200update의 탐색 계획을 [최소 학습 계획](minimal_target_ablation_plan.md)에 고정했다.
2. 실행 전에 train 일부로 메모리/시간/loss scale을 측정한다. 환경 재설치·큰 학습·전체 cache 금지.
   현재 trainer/제한 feature cache/normalization은 **아직 구현·생성·실행하지 않았다**.
3. Target 결과로 shortlist 후 다른 seed와 현재-target 반복 대조를 확인한다. 안정된 후보에서만
   random/규칙/ego-attention/ST와 고정 K2/4/8 비교로 간다. Ego-attention은 공식 EgoFSD 재현이라고 부르지 않는다.
4. 미래 효과가 불분명하면 coverage·loss scale·표본/학습량·현재 side-channel을 진단한다.
   작은 실패로 모든 적응형 예측 가능성을 기각하지 않고, 큰 학습으로 바로 보상하지 않는다.
5. 단일 seed/개발 ADE만으로 최종 기여를 확정하지 않는다. 공식 안전·진행 지표와 별도 독립 평가,
   관련 연구 대비 동일 예산/연산 조건까지 확보해야 논문 주장으로 발전시킨다.
