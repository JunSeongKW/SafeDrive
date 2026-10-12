# Stage 0 — 고정 scorer가 정확히 평가하는 후보 분포를 학습할 수 있는가?

조사일: 2026-10-12 KST. 본 문서는 점수를 확인하기 전의 설계이며 성공 결과가 아니다.

## 연구 질문과 가장 가까운 방법

같은 후보64/같은 고정 scorer에서, **실제 주행 품질을 유지하며 scorer가 정확히 평가하는 조건부 분포 쪽으로 generator를 적응시키면 최종 선택손실이 감소하는가?** 장면별 안전·진행 판단에 필요한 표현을 어떤 후보 탐색에 활용할지에 관한 하위 질문이다.

- [TOAD 원문 §3, §5](https://arxiv.org/html/2606.07170v1): 고정 scorer를 reward로 삼아 제어 공간에서 CEM으로 높은 점수 후보를 탐색한다. Anchor/comfort 규제가 있다. Scorer가 제안 밖에서도 일반화해야 한다는 문제의식은 가깝다. 이번은 generator를 학습하며, **예측 점수 최대화와 실제 평가 정확도 최대화를 구분**한다. 원문 검색에서 random augmentation 설명은 확인하지 못했으므로 공개 random-bouquet 가중치의 효과를 원문이 분리 입증했다고 주장하지 않는다.
- [Learning Sampling Distributions for Model Predictive Control, Sacks & Boots, CoRL 2022/PMLR 2023, §3](https://proceedings.mlr.press/v205/sacks23a/sacks23a.pdf): 조건부 normalizing flow와 latent MPC 갱신을 bi-level/BPTT로 학습한다. 좋은 분포를 학습해 적은 후보로 계획한다는 목적이 겹친다. 이번은 고정된 학습 scorer의 oracle 오차를 이용한 offline teacher와 분포 matching이며 flow·MPC unrolling은 추가하지 않는다.
- [MOPO, NeurIPS 2020 원문](https://proceedings.neurips.cc/paper/2020/file/a322852ce0df73e204b7e67cbbef0d0a-Paper.pdf): 학습 모델의 오류를 이용하는 정책을 막기 위해 uncertainty로 reward를 낮춘다. 이는 이번 오류 회피 동기의 참고다. 여기의 직접 측정한 scorer 오차는 모델 uncertainty가 아니며 MOPO의 return lower-bound 이론을 가져올 수 없다.

공식 코드는 `reference_repositories/DrivoR`와 `TOADNav1OfficialCalibration`의 `drivor_model.py`, `transformer_decoder.py`, `layers/utils/mlp.py`를 확인했다. 검색어는 `trajectory scorer distribution TOAD DrivoR generator`, `trajectory sampling learned cost distribution planning scorer`, `Learning Sampling Distributions for Model Predictive Control`, `policy model exploitation learned reward regularization MOPO`다. 조사일 이후 자료는 사용하지 않았다.

현재 형태의 novelty는 **조건부**다. 좋은 후보를 teacher로 학습하는 분포 증류와 겹친다. 실제로 기여를 주장하려면 품질만 사용하는 B 대비 정확도 항의 이득이 있어야 하고, 동일 후보수/고정 scorer에서 oracle coverage를 유지하면서 선택손실이 감소해야 한다. 적은 학습 파라미터 자체는 구현 장점이지 충분한 기여가 아니다.

## 먼저 피해야 할 잘못된 목표

1. 'Scorer가 좋아하는 분포'를 예측 점수가 높은 분포로 정의하면 기존 reward hacking을 재현할 수 있다.
2. 오류가 작은 낮은 점수 후보만 생성하면 평가가 쉬워지지만 주행 품질이 떨어진다.
3. 후보를 하나의 쉬운 경로로 모으면 다양성·oracle 최고점이 줄고 선택손실만 작아질 수 있다.
4. Scorer에는 candidate self-attention이 있어 개별 경로의 정확도가 후보 집합 구성에 의존한다. 원래64 문맥에서 얻은 정확도가 새 분포에서도 보존된다는 보장은 없다.

따라서 **좋은 품질과 정확한 평가가 동시에 있는, 장면별 teacher**를 만든다. 하늘/차량 같은 정적 feature 분류로 쉬운 분포를 가정하지 않는다.

## 첫 실행 범위와 모델

- 공개 DrivoR Scaling checkpoint SHA `617e22c5ebcf8b24c542d42a470514b09c3cadce1a2630071d49f2d422d76672`에서 시작한다.
- 원래 현재4카메라/1148×672/카메라당16 scene/register token을 유지한다. Scene features 64×256, 최종 trajectory decoder 출력64×256, ego token1×256을 FP32로 캐싱한다.
- 원 encoder/ego encoder/query/trajectory decoder/scorer를 고정·eval한다. 학습 대상은 원래 `traj_head[-1]` MLP(256→1024→1024→24)의 복사본이다. 원head/원checkpoint는 수정하지 않는다.
- 새로운 sampling head나 후보 필터만 붙이는 실험이 아니다. 원래 generator의 출력 head를 업데이트하므로 64개 경로의 실제 좌표가 바뀐다. 전체 generator 미세조정이나 CEM 분포 학습의 검증으로 확대하지 않는다.
- NAVTRAIN128 학습/별도 recording NAVTRAIN32 점검, 이미 노출된 NAVTEST64 개발평가. 집합 간 recording 중복0. 공개 모델 자체는 trainval로 사전학습돼 있으므로 NAVTRAIN 점검은 원 모델의 완전 미관측 검증이 아니다. NAVVAL을 독립 검증으로 취급하지 않는다.
- 현재 이미지1프레임/이전·현재 ego 이력은 기존 forward와 같다. 미래 정답은 teacher loss/공식 평가에만 사용한다.

## A/B/C와 loss

각 장면의 원 후보에 대해 실제 품질 `J_i`와 고정 scorer 추정 `S_i`를 0–1 범위로 계산한다.

```text
q_quality(i|scene) = softmax(J_i / 0.10)
q_accurate(i|scene) ∝ q_quality(i|scene) × exp(-abs(S_i-J_i) / 0.15)
q_compatible = (1-mix) × q_quality + mix × q_accurate
```

`mix<=0.8`이며 target의 평균 실제 점수가 quality-only보다 1 PDMS점 넘게 낮아지지 않도록 scene별로 낮춘다. 이는 **학습용 target 품질 제약**이고 학습된 경로의 품질 보장은 아니다. NAVTEST 정보로 target을 만들지 않는다.

- A: 수정 없는 공개 generator와 scorer.
- B: 원 native64의 실제 품질 가중 teacher를 학습.
- C: B에 측정된 scorer 예측오차 가중을 더한 teacher를 학습.

B/C 모두 같은 초기 head·seed0·batch 순서·AdamW1e-5·batch8·300step. 최종 비교는 둘 다300step으로 고정하고 100step은 양상 확인용이다. 추가 SSL/semantic/scorer loss는 없다.

```text
L = MMD²(uniform generated64, weighted teacher-native64)
    + 0.1 × original-coordinate anchor
    + 0.01 × high-quality native-candidate coverage
```

MMD는 8 waypoint XY와 sin/cos heading 좌표에 대해 1/2/4m RBF kernel을 평균한다. Coverage는 B/C 공통 quality-only target과 generated set의 soft nearest XY RMS 거리를 사용한다. Anchor도 동일하다. Target는 stop-gradient이며 **MMD→생성 경로 좌표→원 generator 출력 head**로 gradient가 흐른다. Official evaluator나 frozen scorer를 역전파하지 않는다.

MMD의 정의는 [Gretton et al., A Kernel Two-Sample Test, JMLR 2012](https://www.jmlr.org/papers/v13/gretton12a.html)를 참고한다. 이번은 두 empirical distribution의 kernel 거리 loss이며 논문의 검정 통계 유의성 결과를 planning 성공 판단으로 사용하지 않는다. 이 출처는 Stage4 결과 확인 후 방법 출처를 보충한 것이며 등록 loss/수치/gate는 변경하지 않았다.

MMD가 낮아지는 것과 최종 planning 개선은 다른 결과다. 기존 native 후보의 작은 영역에 머무르는 첫 진단이므로 support 밖 새로운 좋은 경로 탐색은 제한적이다.

## 단계와 사전 판정 기준

1. Cache/sample matching, 원head 복원, 실제 official teacher 라벨 및 finite gradient 검사. 오류가 있으면 수정 후 학습한다. 결과 `stage1_teacher_distributions.png`와 JSON.
2. 같은 예산으로 B/C 학습. 학습 loss와 원 후보에서의 좌표 이동·분포 변화를100/300step에 기록한다. 결과 `stage2_training_and_movement.png`.
3. 바뀐64경로 집합을 frozen scorer로 다시 채점하고 공식 PDMS도 다시 계산한다. 원라벨을 이동한 경로에 재사용하지 않는다. A/B/C 최종 선택·후보 최고·안전·진행·comfort·예측오차·다양성 및 상황별 집계를 공개한다. NAVSIM 카메라/BEV 사례 시각화도 남긴다.

후속 확대 판단: C−B 선택 PDMS의 paired recording-bootstrap95% 하한>0, C−A 후보 최고 PDMS≥−1점, C의 선택 NC/DAC 실패 수가 A보다 늘지 않음. 작은 단일seed pilot에서 이 기준을 통과해도 복제 실험만 지지하며 논문 채택/전체 NAVTEST 향상을 주장하지 않는다. 기준을 결과 확인 후 바꾸거나 장시간 fullscale로 자동 확장하지 않는다.

기존 LPWM 학습은 보존한다. GPU0에서 캐시만 최대4.5GB·전체카드48GB 이내/600초상한. 그 이후 새 학습·채점은 CPU다. 비용은 단계별로 별도 측정한다.

## TOAD의 개선과 이 가설의 관계

앞선 동일128후보/CEM OFF 비교에서 원DrivoR93.6972 vs TOAD배포95.9485였다. 따라서 이 차이를 CEM이 scoring하기 쉬운 후보를 골랐기 때문이라고 설명할 수 없다. 두 공개 checkpoint는 다른 weights/학습 recipe이고 scorer/decoder 구조는 동일하다. Random-bouquet 공식 코드 옵션도 확인했지만 augmentation만의 순수 효과는 현재 비교로 분리되지 않았다.

CEM 자체는 frozen scorer를 업데이트하지 않는다. CEM이 만든 집합의 **실제 점수 오차/순위 정확도**가 올라가는지는 동일 scorer·초기/후기 집합·공식 oracle 라벨을 비교해야 한다. 높은 neural reward만으로 정확도를 주장하지 않는다. 이번 파일럿은 학습으로 후보 분포를 바꾸는 별도 가설을 시험한다.
