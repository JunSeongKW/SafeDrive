# Drive-JEPA 미래 경로의 구조별 추가 학습

## 실행 전 고정한 질문과 범위

사용자의 추가 학습 승인에 따라, 기존 planner를 보존하면서 미래 경로의 표현력과
ego 의도 반영을 강화하면 작은 개발 split에서 planning 오차를 줄일 수 있는지 확인한다.
본 연구의 선택·예산 가설을 입증하는 실험이 아니라, 다음 선택 실험을 위한 구조 비교다.

기준 HEAD는 `2185ce5`이며 이후 완료한 선택 비교의 미커밋 코드·결과를 먼저 보존한다.
기존 비교에서 learned K4는 200 update의 dev scene-macro XY ADE가 0.242582m로
원본 0.220644m보다 나빴다. 초반 개선 뒤 dev 오차가 증가했고 미래 MSE도 persistence보다
나빴다. 이를 데이터 부족·과적합 하나로 확정하지 않으며 구조와 최적화 모두 후보로 남긴다.

이번에는 기존 128 train/64 dev, recording 16/8, frozen cache를 그대로 쓴다.
Held-out/navtest/navhard를 읽거나 평가하지 않으며 기존 GT ROI pilot과 WA를 재개하지 않는다.

## 구조 비교

| 조건 | Predictor | Selector | Encoder 변화 | 보조 target |
|---|---|---|---|---|
| mlp_recipe_control | 기존 절대 feature MLP | 기존 additive ego conditioning | 없음 | 미래 |
| contextual_residual | 전체 현재 patch를 cross-attention으로 참조, 현재+변화량 | 기존 | 없음 | 미래 |
| ego_query_residual | 위와 같음 | ego query와 patch key 내적 | 없음 | 미래 |
| ego_query_residual_lora | 위와 같음 | ego query | 미래 branch에 복제한 마지막 4 block의 QKV LoRA | 미래 |
| lora_current_target_control | 위와 같음 | ego query | 위와 같음 | frozen 현재 feature를 4시점에 반복 |

미래 prediction query는 K4×4시점이다. Current context는 전체 512 patch를 유지하므로
encoder 전체의 연산 절감을 주장하지 않는다. Residual predictor는 128차원, 4-head,
2-layer cross-attention이며 출력 delta head를 0으로 초기화한다.
Ego 8차원은 기존에도 들어갔다. 새 구조는 입력 추가가 아니라 conditioning 방식 변경이다.

LoRA는 rank4/alpha4/dropout0, 마지막 block20–23의 QKV에만 적용한다.
원본 planner 입력과 원본 encoder 가중치는 고정한다. 별도 future branch encoder tail만 적응한다.
Teacher는 full planning checkpoint의 encoder로 고정하며 EMA를 도입하지 않는다.
이는 미래 latent 회귀를 쓰는 JEPA형 확장이지 공식 JEPA 사전학습 전체 재현이 아니다.

## Gradient와 관측 경계

Planning loss는 bridge, predictor, selector와 활성 LoRA에 전달한다.
보조 loss는 detached hard selection으로 predictor를 다시 호출하여 selector와 bridge에는
전달하지 않고 predictor와 LoRA만 갱신한다. Teacher feature와 미래 유효 mask는 loss에만 쓴다.
미래 GT는 online forward, candidate 선택, 현재 encoder prefix에 들어가지 않는다.
LoRA가 selector도 읽는 현재 표현을 바꾸므로, auxiliary 업데이트가 이후 선택에 주는 **간접 영향**은
남는다. Selector parameter로의 직접 gradient 차단을 선택 정책에 대한 모든 auxiliary 영향 제거라고
해석하면 안 된다. 이번 비교에서는 이 shared-representation 효과를 별도로 분리하지 않았다.

Bridge zero-init은 시작 시 원본 trajectory를 보존한다. Delta zero-init은 persistence를
구조적으로 구현한다. 초기 개선과 학습으로 얻은 미래 변화량 개선을 구분해야 한다.
LoRA B=0일 때 A의 초기 gradient가 0인 것은 예상된 현상이며 그래프 단절이 아니다.

## 공통 학습 절차와 상한

설정은 `configs/drive_jepa_selective_future/architecture_followup_v1.json`에 실행 전에 고정한다.
각 조건은 seed29/47/83, random hard K4의 aux-only warmup100회 후 learned joint200회다.
모든 조건에서 같은 batch 순서와 대응되는 공통 모듈의 초기화를 사용한다.
단계 경계에서 optimizer를 새로 만들며, 기존 실험의 정확한 resume이라고 부르지 않는다.
기존 MLP도 같은 새 절차로 재학습하므로 구조 변경과 새 학습 절차의 영향을 분리할 수 있다.

AdamW, batch8, weight decay0.01, gradient clip1, predictor/bridge LR1e-4,
selector/LoRA LR2e-5, 각 단계 cosine decay로 초기 LR의 0.1까지 낮춘다.
공식 length-normalized L1 planning loss와 raw latent MSE를 쓴다.
보조 가중치는 warmup1, joint0.01이며 sweep하지 않는다.
Joint0/50/100/200에서 평가하고 마지막 고정 checkpoint를 보고한다. Best dev 선택은 하지 않는다.

전체 15run/4500update, 총 wall-time90분, 조건당15분, allocated GPU12GiB 상한이다.
시작 free16GiB, 실행 reserve6GiB. 실제 여유를 확인한 GPU1 한 프로세스만 사용한다.
이 수치는 실행 상한이지 예상 사용량 실측값이 아니다.

LoRA에는 frozen 첫20 block 출력이 필요하므로 같은192개 현재 관측만 추가로 읽는다.
새 prefix cache는 약384MiB 예상/1GiB 상한, 기존 cache와 다른 경로다.
공식 전처리와 고정 encoder 출력을 재대조한다. FP32 batch/kernel 차이 허용값은
atol=rtol=1e-5로 사전 설정한다. 원본 planner 출력 보존은 같은 실행 모드에서 bitwise로 검사한다.
단순 CPU autograd-on/off 경로의 다른 attention kernel은 수치 동일성과 혼동하지 않는다.

## 평가와 해석

Scene-macro XY ADE와 recording별/command별 오차, seed별 차이, seed 평균·표준편차를 보고한다.
같은 selected patch의 미래 MSE와 persistence를 비교하고 horizon별 오차를 보존한다.
정책에 따라 target patch가 달라지므로 조건 간 raw MSE만으로 predictor 우열을 판정하지 않는다.
Current-target 조건은 현재 feature를 반복 감독하는 대조이며 direct feature adapter가 아니다.
Zero/swap은 입력 변화량과 trajectory 변화량을 함께 기록하는 의존도 진단이다.
이는 재학습 no-future 대조나 안전성 평가를 대신하지 않는다.

성공하더라도 반복 사용한 작은 dev split의 탐색 결과다. 대응 seed·recording 불확실성을
분리하며 공식 PDMS 향상 또는 학습된 최적 선택을 주장하지 않는다.
상한에서 종료하고 결과가 좋지 않아도 조건·데이터·loss를 추가 탐색하지 않는다.

## 실행과 상태

구현 및 CPU 계약 검사 후 실제 공식 모듈의 초기 출력·tail 동등성 검사를 통과해야 학습한다.
Raw 출력은 `outputs/drive_jepa_selective_future/architecture_followup_v1_20261002/` 아래 새로 생성한다.
위 절까지는 실행 전 계획이다. 아래는 2026-10-03 완료 후 확인한 결과다.

## 완료 결과

사전 고정 커밋 `d3bbced`의 코드로 **15run/4500update를 모두 완료**했다.
중간 중단·OOM·조건 탈락 없이 마지막 joint200 checkpoint를 보고한다.
Raw 디렉터리의 `20261002`는 시작일이며, 종료 및 공유 export는 2026-10-03이다.
실행 후 가독성 formatting과 동기 실행 lambda의 loop variable 명시적 binding만 정리했다.
정확한 실행 코드의 SHA는 공유 summary의 execution.source_hashes에 있으며 원문은 `d3bbced`에 있다.

| 조건 | Train ADE 평균(m) | Dev ADE 평균 ± seed 표준편차(m) | Dev 미래 MSE | 같은 patch의 persistence MSE |
|---|---:|---:|---:|---:|
| 원본 no-branch | 0.273974 | 0.220644 | — | — |
| 기존 MLP + 새 학습 절차 | 0.234828 | **0.209975 ± 0.008071** | 3.099922 | 2.191076 |
| Contextual residual predictor | 0.201145 | 0.223808 ± 0.006583 | 2.112470 | 2.179885 |
| 위 구조 + ego-query selector | 0.205381 | 0.216938 ± 0.008604 | 2.403973 | 2.507297 |
| 위 구조 + encoder-tail LoRA | 0.205984 | 0.216459 ± 0.009297 | 2.403980 | 2.508950 |
| LoRA 구조 + 현재 feature 반복 target | 0.204671 | 0.219851 ± 0.009602 | 2.501165 | 2.494441 |

ADE는 개발용 scene-macro XY trajectory 오차이며 PDMS가 아니다.
수치는 train123scene/dev62scene, 128/64window, recording16/8의 작은 반복 개발 표본에 한정한다.
MSE는 각 조건이 고른 patch의 미래 target을 대상으로 평가하므로 조건 간 target 집합이 다르다.
현재 target 대조의 표 역시 학습 target이 아니라 **실제 미래 target**으로 평가한 MSE다.

### 개선한 부분과 아직 개선하지 못한 부분

- 기존 MLP에 새 학습 절차를 적용한 조건은 원본보다 평균 0.010669m(약4.84%) 낮았다.
  세 seed 모두 낮았지만, recording-cluster 95% CI는 차이 기준 **[-0.025015, +0.000101]m**다.
  8개 recording의 불확실성과 반복 dev 사용 때문에 일반적인 성능 향상을 확정하지 않는다.
  이전 learned 결과0.242582와 비교할 때 학습 절차뿐 아니라 모듈 초기화 규칙도 달라졌다.
  따라서 이를 warmup 하나의 인과적 이득이라고 부를 수 없다.
- Contextual residual은 train 오차와 동일 patch 미래 MSE를 낮췄지만, 새 절차의 MLP보다
  dev ADE가 세 seed 모두 높았다. 평균 차이 +0.013833m, cluster CI[+0.004742,+0.024249]m이다.
  구조적 persistence를 넘는 학습된 변화량의 이득은 있으나 planning 이득으로 자동 연결되지 않는다.
- Ego-query 추가 효과는 seed29/47에서 약0.00086m 악화, seed83에서0.02233m 개선으로 불균일했다.
  평균 차이 -0.006870m, cluster CI[-0.020105,+0.002794]m다. 입력 반영과 유용한 선택을 구분한다.
- LoRA 추가의 dev ADE 차이는 -0.000479m, cluster CI[-0.001177,+0.000231]m다.
  **이번 rank4·last4·LR·학습량에서는 추가 이득을 확인하지 못했다.**
  LoRA 일반의 효과 부재나 encoder fine-tuning 불필요를 증명한 결과가 아니다.
- LoRA의 미래 target 대 현재 target 차이는 -0.003392m, CI[-0.007769,+0.000450]m다.
  미래 감독의 planning 유용성을 확정할 근거는 아직 부족하다.

위 CI는 seed 평균 scene 차이에 대해 recording을 재표집한 2000회 bootstrap이다.
Seed 변동은 별도 표준편차와 대응 seed 차이로 보고한다. 여러 탐색 비교에 대한 confirmatory
다중검정 주장은 하지 않는다. 표준편차보다 작은 차이라고 곧바로 차이 없다고 판단하지 않는다.

### 학습 곡선

| 조건 | Joint0 | Joint50 | Joint100 | Joint200 |
|---|---:|---:|---:|---:|
| MLP | 0.220644 | 0.208170 | 0.208339 | 0.209975 |
| Contextual residual | 0.220644 | 0.210819 | 0.218923 | 0.223808 |
| Ego query | 0.220644 | 0.211627 | 0.215593 | 0.216938 |
| Ego query + LoRA | 0.220644 | 0.211669 | 0.215809 | 0.216459 |
| 현재 target 대조 | 0.220644 | 0.211195 | 0.217323 | 0.219851 |

모두 3seed 평균 dev ADE다. 곡선에서 좋은 시점을 사후 선택하지 않고 고정200 결과를 유지했다.
복잡한 모델의 train 개선과 dev 후반 악화는 일반화·최적화 문제 후보를 지지하지만,
데이터 부족·과적합·target 특성 중 하나를 단독 원인으로 식별하지는 못한다.

## 원본 보존과 gradient 실측

전체15run 초기 bridge-on/off 출력은 동일한 평가 모드에서 bitwise로 일치했다.
추가 current-prefix cache는192/192 기존 최종 feature와 **최대 차이0**이었다.
Batch8에서 copied tail과 단일-window cache의 최대 차이는1.1444e-5이며 사전 atol+rtol 기준을 통과했다.
상대 오차도 쓰는 검사를 순수 absolute1e-5 검사라고 기술하지 않는다.

실제 공식 모델의 모든 조건에서 planning→selector/predictor/bridge가 연결됐다.
LoRA 조건의 seed29 planning gradient norm은 selector0.035136, predictor0.029967,
bridge0.162120, LoRA0.000295였다. Aux→selector/bridge는0,
predictor0.405343/LoRA0.023228이었다. 다른 seed도 동일 경계를 통과했다.
LoRA trainable factor는65,536개, 전체 새 trainable은1,734,400개다.
LoRA factor가 갱신됐다는 사실과 dev 성능이 좋아졌다는 주장은 별개다.

학습 후 각 조건 seed29 checkpoint를 엄격한 key/shape 대조로 복원해 실제 RGB2window로 검사했다.
동결·warmup 조건을 일치시키면 **branch-off는 다섯 조건 모두 원본과 bitwise 일치**했다.
Online 영상 경로와 cache 경로의 trajectory 최대 차이는1.90735e-6으로 사전 FP32 기준 안이다.
원본309,981,955개 parameter hash는 학습 전·후·최종 검사에서
`05af3ccd8a8e7db1b9b2ae2e77f81ef753446a3309989cef08c161ce0614476a`로 동일했다.

첫 postflight는 동결 설정·warmup 이전 참조와 이후 출력 사이9.53674e-7 차이로 bitwise gate에 실패했다.
동시에 다시 호출한 원본과 branch-off의 차이는0이고 parameter hash도 같았다.
참조 실행 조건을 동결·warmup으로 통일한 뒤 bitwise gate를 다시 통과했다. 허용 오차를 늘리거나
checkpoint를 수정하지 않았다. 첫 호출 차이의 backend 내부 원인을 하나로 확정하지 않는다.
실패 수치도 `postflight_equivalence_failure.json`으로 보존한다.

## 비용과 실행 상태

전체1300.33초(21분40초), 순차15run 합1205.60초, 나머지는 로딩/해시/current-prefix 준비였다.
새 prefix cache402,888,384bytes. 기존 약1.88GiB cache와 모든 결과는 그대로 보존했다.
최대 **PyTorch allocated2.674GiB**; 이는 CUDA context·reserved cache·다른 사용자 메모리를 포함한
`nvidia-smi` 전체 점유와 다른 수치다. OOM0, 총90분/조건15분 상한 이내였다.
학습과 최종 추론 검사 프로세스는 모두 종료했다. 타인 작업을 중단하지 않았다.

LoRA 없는 run은43–56초, LoRA run은115–149초였다.
최종 실제 RGB2window 배치 추론은 원본81.1ms, MLP90.5ms, contextual89.9ms,
LoRA83.6ms 등으로 측정됐으나 공유 GPU에서 편차가 컸다.
LoRA가 더 빠르다고 해석하지 않는다. 같은 checkpoint의 cached 미래 경로+planner는
MLP5.14ms, ego-query8.16ms, LoRA20.32ms였다. LoRA tail을 추가 실행하는 비용이 포함된다.
이 측정은 warmup2/repeat7/동기화이며 IO·teacher·scorer를 제외한 소수 표본 진단이다.

## 결론과 다음 진행 조건

**더 큰 구조를 기본 모델로 승격하지 않는다.** 새 절차의 기존 MLP를 현재 최소 비용 개발 참조로
보존하고, contextual residual은 실제 미래 feature 정확도를 개선한 후보로만 남긴다.
추가 LoRA sweep, EMA, planner fine-tuning, 데이터 확대 또는 benchmark 평가는 자동 실행하지 않았다.
현재 미래 MSE 개선과 planning 개선이 분리됐으므로, 후속 연구는 같은 조건의 fixed/random/learned
선택 비교로 돌아갈지 검토하는 것이 우선이다. 이번 실험에는 새 절차의 fixed/random 비교가 없으므로
학습형 선택의 우월성이나 원래 H1/H2를 검증했다고 할 수 없다.

## 재현과 검수 자료

전용 Conda는 `/rhome/junseong/envs/kjs-drive-jepa-extension`이며 패키지 설치·기존 환경 변경은 없다.
아래 학습 명령의 출력은 **반드시 미존재 새 경로**를 지정한다. 완료 경로는 덮어쓸 수 없다.

```bash
CUDA_VISIBLE_DEVICES=1 \
LD_LIBRARY_PATH=/rhome/junseong/PlanningAwareFuturePrediction/runtime/environments/drive_jepa_official_evaluation/lib \
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
PYTHONPATH=/rhome/junseong/PlanningAwareFuturePrediction/src \
/rhome/junseong/envs/kjs-drive-jepa-extension/bin/python \
scripts/run_drive_jepa_architecture_followup.py \
--output-directory /rhome/junseong/PlanningAwareFuturePrediction/outputs/drive_jepa_selective_future/architecture_followup_reproduction
```

Checkpoint는 trainable delta/optimizer/scheduler/RNG/batch schedule을 보존한다.
학습 resume CLI는 아직 없고, 원본 strict weights + trainable delta key/shape 복원은 postflight에서 검증했다.
새 초기화 재실행을 정확한 optimizer resume이라고 부르지 않는다.

- [공유 요약·대응 seed·recording CI](../results/drive_jepa_selective_future/architecture_followup_v1_20261003/summary.json)
- [학습 곡선](../results/drive_jepa_selective_future/architecture_followup_v1_20261003/training_curves.json)
- [Window별 최종 오차](../results/drive_jepa_selective_future/architecture_followup_v1_20261003/final_window_metrics.csv)
- [실제 gradient·초기화 계약](../results/drive_jepa_selective_future/architecture_followup_v1_20261003/architecture_contracts.json)
- [학습 후 원본 보존·추론 비용](../results/drive_jepa_selective_future/architecture_followup_v1_20261003/trained_checkpoint_postflight.json)
- [모델 코드](../src/planning_aware_future_prediction/models/drive_jepa_adaptive_future.py)
- [학습 runner](../scripts/run_drive_jepa_architecture_followup.py)

공유 문서는 작성 스킬의 사실/설계/해석 분리 기준에 맞춰 기존 저장소에 기록했다.
원격 공개 설정은 그대로이며 push는 기존 VSCode Git credential socket 오류로 실패했다.
최종 commit은 로컬에 보존하므로 인증 복구 후 `git push mine junseong/main`이 필요하다.
