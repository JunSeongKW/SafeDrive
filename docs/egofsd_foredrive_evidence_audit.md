# EgoFSD / ForeDrive — 직접 중복과 미확인 경계

2026-10-01 확인. 원문과 official repository를 읽었으며 **모델 구현 재현/독립 성능 평가 없음**.
문헌 관찰과 우리 설계 판단을 구분한다. 다른 JEPA·선택 연구 전체의 novelty 감사는 여전히 미완료다.

## 근거와 확인 범위

| 항목 | EgoFSD | ForeDrive |
|---|---|---|
| 원문 기준 | arXiv2409.09777v6,2026-02-09 | arXiv2609.26299v2,2026-09-23 |
| 예측 대상 / 선택 | agent/map query를 ego 의도·attention·기하 prior로 계층 선별 후 joint motion | 전체 visual patch future와 future **ego** status; entity top-K 선택은 확인하지 못함 |
| 주요 근거 | §3.4,Eq.(2)(3),§3.5,Table3 | latent model Eq.(1)-(3),fusion Eq.(4),gradient Eq.(7),Appendix F/H |
| 예산 | §3.4의3stage 비율10/5/2%; 장면별 학습된 동적 예산으로 확인하지 않음 | 고정 horizon set과 sample-shared horizon gate; 장면별 budget 정책으로 확인하지 않음 |
| Gradient | direct planning→discrete selection 정책의 실제 autograd **미확인** | Eq.(7)은 planning→shared encoder/fusion/planner, **planning↛predictor**; injected futures detach |
| Train/inference | geometry response map은 미래 ego trajectory 기반 학습 감독; 추론 GT 여부를 실제 코드로 감사 못함 | 미래 image EMA target은 train only; inference는 예측 futures 사용 |
| 평가 | nuScenes/Bench2Drive | NAVSIM v1/v2; 원문 보고이며 재평가 아님 |
| 구현 | official tree에는 README/assets만 있음 | 원문/제한된 검색에서 official 구현 링크 확인 못함; “전 세계 비공개/없음”으로 단정하지 않음 |

EgoFSD official tree `23fec8aba3e828ef228939e30e3020240d8b0cae`를 shallow/no-checkout으로 확인했다.
파일은 `.gitignore`, `README.md`, `assets/*.png`이며 `.py`/training config/model weight가 없다.
README의 setup도 미완이다. 따라서 source 함수/gradient/top-K surrogate를 독립 감사했다고 보고하지 않는다.
v3 HTML은 DiFSD라는 명칭을 쓰지만 이번 판단은 최신 **v6 EgoFSD**로 고정했다.

ForeDrive Appendix F는 고정 horizon 구성을 비교하고 H는 detach/joint injection을 비교한다.
그 실험은 우리의 context별 marginal budget benefit이나 객체 target 선택을 이미 검증했다는 뜻은 아니다.
반대로 미래 사용·horizon 변화·detach 자체가 미개척 문제라고 말할 수도 없다.

## 우리 연구에 대한 판단 — 아래는 추론/결정

1. 객체 선택과 joint planning은 직접 중복한다. ST가 있다는 이유만으로 고정 K 선택의 novelty를
   확보했다고 주장하지 않는다. 이후 비교에 ego-attention+current geometry의 강한 대조가 필요하다.
2. Future latent 직접 conditioning도 직접 중복한다. ForeDrive의 **ego status**와 우리 **주변 객체 state**는
   다르지만, 이 차이만으로 별도 기여가 확보되지는 않는다.
3. 현재 frozen encoder / P까지 planning gradient / GT ROI는 ForeDrive의 학습 경로와 같지 않다.
   이 차이는 engineering 선택이며 우위의 증거가 아니다. Gradient interference도 후속 위험이다.
4. 다음 연구 질문은 target을 정한 뒤, 동일한 고정 정책·모듈에서 예산 증가의 효과가 상황별로
   안정적으로 달라지는지 측정하는 것으로 좁힌다. 동적 예산 method는 그 결과와 추가 문헌 검토 후 결정한다.

## 원문·코드 링크

- [EgoFSD v6 §3.4/3.5와Table3](https://arxiv.org/html/2409.09777v6#S3.SS4)
- [EgoFSD official README pinned commit](https://github.com/suhaisheng/EgoFSD/blob/23fec8aba3e828ef228939e30e3020240d8b0cae/README.md)
- [EgoFSD official file tree pinned commit](https://github.com/suhaisheng/EgoFSD/tree/23fec8aba3e828ef228939e30e3020240d8b0cae)
- [ForeDrive v2 method/Appendix F/H](https://arxiv.org/html/2609.26299v2)

위 표는 관련 원문을 요약한 것이며 paper score를 비교표에 재사용해 우리 모델 성능처럼 보고하지 않는다.

## 9353acf 이후 관련 근거 추가

[iPad의실제ablation 조건과확인수준](future_prediction_diagnostic_scope_and_evidence.md)을 확인했다.
당시 EgoFSD/ForeDrive 원문 감사와 이번 iPad 원문/README 확인을 구분한다.
이번에 세 연구의 모델 구현을 독립 재현하거나 전체 source autograd를 새로 감사한 것은 아니다.
EgoFSD의noise 학습, 확률적future, Tube-MPC는 서로 다른 방법이며 현재확률적후속아이디어의novelty는미확정이다.
