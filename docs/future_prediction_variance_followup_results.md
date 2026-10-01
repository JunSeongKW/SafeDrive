# C/E 저분산 진단과 제한된 1,000-update 비교

2026-10-01, 기준 `9353acf`. **추가7400update 완료**. 초기 C는 train-mean 수준이었고 추가 학습으로
분산·오차가 개선됐다. 검사 범위에서 정렬/gradient 오류를 찾지 못했지만 **모든 horizon에서 영상
persistence보다 MSE가 높다**. 시간적 미래 예측의 유용성·최종 target·novelty/H1·H2는 미확정이다.

공유 [전체 실측 JSON](../results/future_prediction_diagnostics/bounded_followup_v1.json),
[target 정렬](../results/future_prediction_diagnostics/cached_target_alignment_v1.json),
[새 checkpoint 검사](../results/future_prediction_diagnostics/new_checkpoint_contracts_v1.json),
[전처리·문헌·진행 조건](future_prediction_diagnostic_scope_and_evidence.md).

## 재사용 / 새 실행 / 동일 조건

재사용: cache373/train277/dev96, train-only normalization, seed29 A–E200의 model/AdamW/곡선/감독 수.
Cache 생성·200update 학습·기존 recovery는 반복하지 않았다. 새 실행은 C/E checkpoint 진단,
raw annotation과373구간 대응 검사, A seed11/47 각200, 기존 seed29 A–E 각각800회 추가,
F29·E11·F11 각1000회 신규 학습이다. 합7400update/invocation235.33s(평가/저장 포함),
등록 상한1800s/run600s 이내. GPU0만 사용/종료 후15MiB, GPU1 미사용, GPU4–7 타인 PID 유지.
공용 원본·환경·기존 결과·저장소 공개 설정은 변경하지 않았다.

거리K4/8step4s/frozen teacher/공통mask/초기weight/batch/AdamW lr1e-4/clip1을 유지했다.
B–F는 동일2head/P/D 용량, scorer는 frozen/미호출이다.
F는 `FixedDistanceFutureSupervisionPilot.forward(detach_future_for_planning=True)`로 planner 입력만
detach한다. 반환 예측은 attached여서 두 aux가 P를 갱신한다. Predictor를 no_grad로 실행하지 않는다.

| 조건 | Active params | Visual aux | Spatial aux | Planning→P |
|---|---:|---:|---:|---|
| A | 762627 | 0 | 0 | branch 없음 |
| B | 2223283 | 0 | 0 | 전달 |
| C | 2223283 | .10 | 0 | 전달 |
| D | 2223283 | 0 | .10 | 전달 |
| E | 2223283 | .05 | .05 | 전달 |
| F | 2223283 | .05 | .05 | 차단 |

P1327536/D895747, stored2487988(비활성 scorer 포함), encoder303885312 frozen.
Normalization train4785관측, std floor.05에 닿는 채널0. Dev common1961/native spatial2516.
E/F1000 common 감독은 seed29 136258/seed11 136788로 각각 순서까지 같다. 기존200 common27659 보존.
A/B는 availability만 기록하고 C/D는 해당head, E/F는 두head를 같은 common mask로 감독한다.
미래 mask는 loss/metric에만 사용하고 current selection/window/planner padding에 쓰지 않는다.

## 학습곡선과 대응 seed의 E–F

ADE는 정답8waypoint와의 XY 거리를 scene별로 먼저 평균한 **scene-macro(m)**다.
Imitation proxy이지 official PDMS/충돌/폐루프 지표가 아니다. A–E1000은 seed29 단일 탐색 비교다.

| 조건 | Train loss200→1000 | Dev loss200→1000 | Train ADE200→1000 | Dev ADE200→1000 | Dev FDE1000 |
|---|---|---|---|---|---:|
| A | .14899→.00360 | .17995→.02446 | 4.950→.553 | 5.921→1.672 | 3.490 |
| B | .14782→.00352 | .17940→.02504 | 4.853→.531 | 5.881→1.745 | 3.665 |
| C | .14951→.00343 | .18031→.02338 | 4.968→.517 | 5.921→1.649 | 3.512 |
| D | .14517→.00314 | .17615→.02497 | 4.753→.470 | 5.749→1.699 | 3.521 |
| E | .14730→.00319 | .17869→.02160 | 4.846→.479 | 5.872→1.575 | 3.446 |
| F | 신규→.00305 | 신규→.02096 | 신규→.460 | 5.884(200)→1.554 | 3.397 |

| Dev ADE(seed29) | 200 | 500 | 750 | 1000 |
|---|---:|---:|---:|---:|
| A | 5.921 | 2.387 | 1.671 | 1.672 |
| B | 5.881 | 2.414 | 1.961 | 1.745 |
| C | 5.921 | 2.254 | 1.992 | 1.649 |
| D | 5.749 | 2.292 | 1.944 | 1.699 |
| E | 5.872 | 2.403 | 2.019 | 1.575 |
| F | 5.884 | 2.256 | 1.977 | 1.554 |

초기 순위가 바뀌었다. A200(seed29/11/47)는5.921/6.322/5.659m, 평균5.967/sample std.334m.
**A의 변동으로 다른 조건의 변동을 대신하지 않는다**. A750→1000은 비슷하지만 다른 조건에는 감소가
남는다. Train/dev gap, dev4recording/86scene, GT/front와 반복 train draw의 한계가 있다.

| Seed | E dev ADE | F dev ADE | E−F ADE | E dev loss | F dev loss |
|---|---:|---:|---:|---:|---:|
| 29 | 1.5747 | 1.5536 | +.0211 | .02160 | .02096 |
| 11 | 1.8538 | 1.8030 | +.0508 | .02635 | .02659 |

E/F 초기weight·batch/common 수 일치. F 첫step planning→P norm0, visual/spatial→P는
seed29 .10560/.21620, seed11 .12845/.20129, aux→D는0이다. 두head가 정상 학습된다.
F의 ADE가 조금 낮지만 seed11 planning loss는 조금 높다. 작은 차이/작은 dev로 detach 우월성을
확정하지 않는다. **Gradient 결합 대조이지 미래 정보의 인과적 기여를 단독 증명한 것이 아니다.**

## Mean / persistence / predictor의 같은 mask 비교

Train-mean은 train-only global channel mean(정규화 좌표0), persistence는 현재ROI를 전horizon에 복사.
모두 같은whitening/common mask/channel-mean MSE다. Target에 미래GT가 있어도 forward로 전달하지 않는다.

| Dev predictor | Visual normalized MSE | Visual raw MSE | Spatial normalized MSE | Visual 예측/target 분산 비율 |
|---|---:|---:|---:|---:|
| C200 | .950855 | 3.115516 | .759966 | .011997 |
| E200 | .977312 | 3.070186 | .386421 | .120223 |
| C1000(29) | .793564 | 2.270076 | 1.032829(공간aux 없음) | .213669 |
| E1000(29) | .834415 | 2.396238 | .288063 | .202536 |
| F1000(29) | .823366 | 2.342244 | .292313 | .218067 |
| E1000(11) | .835277 | JSON 참조 | JSON 참조 | .201292 |
| F1000(11) | .825550 | JSON 참조 | JSON 참조 | .212541 |

Dev visual **mean .949572 / persistence .395004**. Spatial persistence .152658.
B1000 visual/spatial1.836622/1.515181, D1.709427/.304915. 보조 감독 없는 head를 미래 예측 성공으로
해석하지 않는다. C/E/F1000 trainvisual .678717/.756311/.732308로 학습 대상 fit도 완전하지 않다.

| Horizon(s) | 공통 유효 수 | Persistence | Train-mean | C200 | C1000 | E1000(29) | F1000(29) |
|---|---:|---:|---:|---:|---:|---:|---:|
| .5 | 298 | .20364 | .97319 | .97446 | .80036 | .83856 | .82729 |
| 1.0 | 270 | .28857 | .95416 | .95593 | .78828 | .83060 | .81979 |
| 1.5 | 253 | .34209 | .94070 | .94158 | .78045 | .82256 | .81004 |
| 2.0 | 241 | .38590 | .93292 | .93527 | .77663 | .81711 | .80807 |
| 2.5 | 235 | .45016 | .94071 | .94351 | .79024 | .82894 | .81897 |
| 3.0 | 229 | .51661 | .95558 | .95699 | .80442 | .84295 | .83092 |
| 3.5 | 220 | .51798 | .94058 | .94064 | .79756 | .84491 | .83123 |
| 4.0 | 215 | .55072 | .95269 | .95210 | .81318 | .85296 | .84395 |

E200/train 전체baseline/horizon/raw metric도 JSON에 있다. 높은 현재→미래cosine을 외형만 학습했다는
증거로 해석하지 않는다. **Persistence를 넘어 유용한 시간적 예측을 배웠다는 주장은 유보한다.**

## 분산 축·입력·target 대응 검사

기존pooled는 sample×entity×time을 합쳤다. 이번에는 다른 축 고정/각 축 population variance를
valid2개 이상인 그룹에서 계산했다. Across-window slot은 거리 순위이지 동일 객체 identity가 아니다.

| Dev normalized visual 분산 | Target | C200 | E200 | C1000 | E1000 |
|---|---:|---:|---:|---:|---:|
| Pooled sample/entity/time | .91679 | .01100 | .11022 | .19589 | .18568 |
| Window축(slot/time 고정) | .90739 | .00832 | .09746 | .19005 | .17991 |
| Entity축(window/time 고정) | .30727 | .00218 | .02616 | .05314 | .05197 |
| Time축(window/entity 고정) | .12977 | .00934 | .07870 | .04007 | .05701 |

시간 분산은 단조 증가하지 않는다. Target future−current RMS .62849,
C200 .97226→C1000 .88444, E200 .98650→E1000 .90641(정규화 좌표).
큰 예측 변화량만으로 실제 변화를 잘 맞춘 것은 아니다.

- Raw373window: track 순서 불일치0/spatial target 오차0/valid 불일치0,
  visual-valid지만 track/projection 없음0/selected gather 오차0.
  Nominal timestamp 최대편차.012979s(허용.05s내). 같은builder로source/cache의 일관성을 검사했고
  좌표 규약 자체의 독립 현실 정합 증명은 아니다. Frozen 값은 전체 encoder로 재생성하지 않았다.
- `cache_window`: clip0=current, clip[j+1]=future[j] 종단, ROI는 current track 순서로 future lookup.
- `predict_selected_future`: currentROI+GT state+현재grid 평균+ego.
  C/E200 currentROI gradient norm115.03/613.00, ROI zero 예측 RMS변화.101/.308.
  입력 경로 존재 증거이지 유용한 예측/선택 성능 증거는 아니다.
- Manual valid-entry MSE와loss 차이 최대5.96e-8, normalize→inverse 오차1.91e-6.
  Broadcast[window,slot,step,channel], 분모valid slot-time×channel. 미래mask는loss에만 사용.

## Zero/swap 입력 변화와 planning 변화

입력RMS는whitened 좌표에서recipient current-active slot만 계산. ΔADE는window 평균이다.

| 조건/step | 교란 | Visual RMS 변화 | Spatial RMS 변화 | Ego XY 변화(m) | Δwindow ADE(m) |
|---|---|---:|---:|---:|---:|
| B200 | Zero | .8789 | .8293 | .69762 | +.42028 |
| B200 | Swap | .7819 | .7515 | .70343 | +.44253 |
| C200 | Zero | .1081 | .1138 | .01111 | +.00276 |
| C200 | Swap | .1204 | .1223 | .00611 | +.00057 |
| D200 | Zero | .8143 | .6744 | .61385 | +.39712 |
| D200 | Swap | .7652 | .8327 | .45047 | +.23880 |
| E200 | Zero | .3459 | .5508 | .17839 | +.06117 |
| E200 | Swap | .4171 | .6869 | .31830 | +.08741 |
| C1000 | Zero | .4472 | .4894 | .14700 | +.03505 |
| C1000 | Swap | .5755 | .5735 | .14252 | +.02585 |
| E1000 | Zero | .4427 | .7613 | .53638 | +.24400 |
| E1000 | Swap | .5632 | .9633 | 1.10534 | +.57133 |

Swap은 다른recording의 예측(미래GT 없음). 단319active recipient slot 중120에서donor가 inactive여서
availability confound가 있다. 작은 C200 입력 변화로 branch 무시를 확정하지 않는다.
1000에서 증가한 의존도도 유용성의 인과 증거가 아니다. Zero는OOD 가능성/type·time token 잔존,
branch 제거는memory 구조 변경. Detach는 같은forward의 backward 검사다.
기존 B–E Zero/swap/제거와 새 E/F 두seed 결과도 JSON에 함께 보관했다.
B/D 입력 변화량은 별도 CPU-only [공유 진단](../results/future_prediction_diagnostics/remaining_branch_perturbations_v1.json)에 기록했다.
A는 미래branch가 없으므로 이 교란의 대상이 아니다.

## 좁혀진 원인·남은 가설·다음 판단

**확인**: C200은 평균 수준, frozen target에는 분산이 남는다. 추가 학습으로 C/E MSE·조건부 분산이
개선돼 초기 저분산에 학습량이 관련된 근거가 있다. 검사 범위의 정렬/normalization/gradient는 정상이고
F의 두head가 학습된다. 현재 absolute predictor는 전horizon에서persistence에 못 미친다.

**미분리**: 유한 학습에 의한 평균 수축, hidden128 용량/최적화, 불확실한target의 조건부 평균,
aux regularization vs 현재 정보 side-channel vs 유용한 미래 정보. JPEG/occlusion/시간 정합도 미확인이다.
Planning gradient가 있다는 것만으로 세 효과를 분리하지 못한다.
Encoder collapse/visual 무효/F 우월/H1·H2/안전/novelty를 주장하지 않는다. B–F 용량은 같지만
A와는 다르며 같은 용량만으로regularization 설명을 제거하지 못한다.

등록 상한에서 멈췄고 residual/delta-target/selector/확률 모델을 추가하지 않았다.
다음 후보는 **현재feature+예측delta residual 한 가지**: target/계수/K를 유지하고architecture만 바꾸는
별도ablation이다. 제안 상한seed29/C와 같은1000update/600s, 기존 C1000 대조 재사용.
**미실행 제안**이며 실행 전config 고정이 필요하다. Persistence/mean/분산을 판정 기준으로 삼고,
한 번 더 실패하면 일반적target 무효를 주장하지 않고front/target 제약을 검토하며 진단 연장을 멈춘다.
효과 주장 전에는 C/residual 양쪽에 대응seed가 필요하다.

K 실험 전coverage: train active838/nominal1108,dev319/384; 후보>K132/277,54/96.
Dev future-heading proxy left6/right1,merge label 미확인. 미래heading은 분석 전용이다.
Multi-view 후보·중복 제거·turn/interaction coverage·JPEG 근거를 보완해야 일반적 맥락별 예산을 논할 수 있다.
확률적 공간 분포/calibration/Tube-MPC 구분은[후속 설계](future_prediction_diagnostic_scope_and_evidence.md)에만 기록했다.

## 이어 학습·검증·재현·공유

구 checkpoint에는model/AdamW는 있으나scheduler/RNG snapshot은 없다. 첫200 sampler prefix/window 순서,
initial hash/AdamW step200/weight를 확인하고800회 추가했다.
**Optimizer-state continuation이며 bitwise exact resume이라고 부르지 않는다**.
새final10checkpoint strict/finite/optimizer step/sampler-next/Python·NumPy·Torch CPU RNG replay 통과.
CUDA RNG는1visible GPU snapshot을 저장했고CPU 검사에서CUDA replay를 실행한 것은 아니다.
새3test(F detach/variance2)+관련 기존5test, 총8test 및Ruff/diff 통과. 전체구38test/recovery는 반복하지 않았다.

Raw는`outputs/future_prediction_diagnostics/bounded_followup_1000_v1/` 약767MB,
초기진단은`seed29_update200_v2/`; sandbox GPU 실패 때 만든 빈v1도 보존.
JSON에source/config/cache/초기weight/구checkpoint hash와 모든curve/metric을 기록했다.
원본영상/cache/checkpoint는Git에 올리지 않는다.

```bash
# 승인GPU 점검 후 새출력 경로 지정. 기존cache 재생성 없음.
runtime/environments/visual_future_prediction_pilot/bin/python scripts/diagnose_visual_future_prediction_variance.py --output-directory outputs/future_prediction_diagnostics/<새진단경로> --physical-gpu 0
runtime/environments/visual_future_prediction_pilot/bin/python scripts/train_bounded_future_prediction_followup.py --output-directory outputs/future_prediction_diagnostics/<새학습경로> --physical-gpu 0
runtime/environments/visual_future_prediction_pilot/bin/python scripts/audit_cached_future_target_alignment.py
runtime/environments/visual_future_prediction_pilot/bin/python scripts/summarize_bounded_future_prediction_followup.py
runtime/environments/visual_future_prediction_pilot/bin/python scripts/verify_followup_checkpoint_contracts.py
runtime/environments/visual_future_prediction_pilot/bin/python scripts/diagnose_remaining_branch_perturbations.py
PYTHONPATH=src runtime/environments/visual_future_prediction_pilot/bin/python -m unittest discover -s tests -p test_future_prediction_diagnostics.py -v
```

Share 생성용script는 이번 고정raw path를 사용하며 기존share에는 덮어쓰지 않고 거부한다.
ChatGPT/Claude는본 보고서+JSON+F forward+continuation 제한을 함께 검수한다.
Push만으로다른 대화에 자동동기화되지는 않는다.
