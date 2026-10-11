# NC·DAC visual evidence pilot — 결과 확인 전 등록한 비교

등록 설정: `configs/metric_visual_evidence/pilot_v1.json`. 원래 연구의 하위 질문은 **현재 관측에서 후보 경로·평가 목적에 맞춰 필요한 정보를 읽는 것이 동일 후보의 선택 오류를 줄이는가**이다. 명시적 미래 predictor의 효용은 이번 실험 대상이 아니다.

## 공통 조건

- 기존 local DrivoR pass5/3,200 update. 원논문의 완전 학습 checkpoint 재현이 아니다.
- 같은 전방 카메라의 과거·현재 2frame, crop28/512×256/patch14 정렬 padding6·10. 백본의 기존 LoRA를 포함한 모든 native 가중치 고정/eval.
- 같은 encoder forward의 마지막 norm에서 공간 patch를 추출. 기존 register 경로와 generator·scorer 예측을 재검사한다. 위치 없는 register16×256을 이미지 grid로 취급하지 않는다.
- 이전 pilot의 동일 NAVTRAIN1024, NAVVAL tuning64·평가192/recording101·15·37. Recording 중복0이며 이전에 개발에 사용한 패널이다. 독립 최종 NAVTEST로 보고하지 않는다.
- 후보64개/후보 context/공식 지표 라벨은 그대로 고정·재사용. 새 trajectory acquisition, generator 학습, encoder 미세조정, semantic 감독 없음.
- 두 학습 조건은 같은 state_dict·파라미터 수·batch8·500step·seed0·AdamW3e-4·batch순서·pair순서·NC/DAC BCE+PDMS차 가중 pair-ranking loss를 사용한다.
- 원래 NC/DAC logit에 bounded residual만 더하고 나머지4지표는 원출력 그대로 유지한다. 초기 residual=0, head는 첫step부터 gradient를 받고 reader는 head 업데이트 뒤 학습한다. OFF는 기존 scorer를 복원한다.

## 비교

1. 기존 frozen scorer: 추가 모듈 없음.
2. Shared adapter: 두 metric query의 attention logits를 평균한 뒤 softmax하여 두 지표가 같은 visual evidence를 읽는다. 두 query 모두 gradient를 받는다.
3. Metric-specific adapter: 같은 모듈/파라미터에서 NC·DAC attention을 각각 사용한다.

Shared와specific의공통imagepatch추가입력및보정MLP를맞춰추가시각정보/용량효과를통제한다. Shared 조건은 query 차이의 평균만 쓰므로 effective query 자유도가 다르며, 이것이 구조적 개입이다. 동일 파라미터 수만으로 모든 effective capacity가 같다는 뜻은 아니다.

## 판정과 비용 제한

- Tuning64에서100/250/500 checkpoint 중 최종 PDMS가 높은 것을 선택한다. 두 조건 모두500step은 실제 실행한다.
- 주요판정: specific−shared의평가PDMS recording-bootstrap95%구간하한>0.
- 안전조건: specific의collision/offroad선택률이shared와원scorer각각보다1%p초과악화하지않음.
- 게이트를통과하면다른seed·개발recording으로replication을검토하고,이번작업에서fullscale을자동시작하지않는다.
- Profile는본학습이전별도가중치로12step,공통batch8/500step예산이합계0.5GPUh이내인지확인한다. GPU0만/새processallocator2.5GiB/전체카드48decimalGB한도. 기존LPWM학습/등록코드/배치/LR변경없음.

## 평가·표현 검증

같은64후보oracle PDMS−selected PDMS,NC/DAC failure,Brier/BCE/ranking,실제명령별sample수·PDMS,실측GPU시간/메모리/캐시비용/보정기추론비용을기록한다.

Specific에는재학습없이OFF·uniform attention·NC/DAC attention swap·train-only mean patch 대체를수행한다. Mean 대체는추가patch분기만바꾸고기존scored feature의시각정보는유지하므로"모든영상정보제거"검사로부르지않는다.

표현진단은각주행명령의첫4개평가장면으로고정하며heatmap품질/PDMS개선으로사례를선별하지않는다. 실제attention의uniform 대비density·entropy·metric차이·candidate차이·전후변화와상위10%token대체/동수random대체의logit변화,patch gradient를측정한다. Actualattention을current/past별원이미지로역매핑하고모든패널색상scale을같게한다.

Heatmap은모델의읽기비중이지object mask/위험확률/객체의인과적중요도GT가아니다. NC/DAC 감독만으로공간적의미가보장되지않으며,semantic특화주장에는별도annotation대응/위험cause검증이필요하다.

## 재실행

```bash
cd /rhome/junseong/PlanningAwareFuturePrediction
bash scripts/run_metric_visual_evidence_pilot.sh
```

완료marker는단계를건너뛴다. 등록source/config/inputhash가달라지면중단하며설정변경은새output에서진행한다.
