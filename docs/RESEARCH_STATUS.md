# 연구 상태 — Codex / ChatGPT 공통 인수인계

갱신: 2026-10-01. 협업 출발점은 `95015df`다. 현재 **계산 그래프 v1 CPU 진단까지 완료**했다.
실제 NAVSIM 모델·visual JEPA·baseline 구현 완료 보고는 아니다.
동적인 상태는 이 파일과 `HANDOFF.md`, 계산 그래프는 `SELECTIVE_FUTURE_GRAPH.md`,
시간순 이력은 `RESUME_NOTES.md`에서 관리한다.

## 1. 최신 사용자 의도와 범위

연구 질문: **현재 주행 맥락과 ego 의도를 이용하여, 같은 예측 예산에서 planning에 유용한
객체의 미래를 선택적으로 예측하도록 학습할 수 있는가?** JEPA 채택 자체는 기여가 아니다.

- H1: 상황에 따라 유리한 미래 정보 구성이 다를 수 있다. 아직 일반적인 사실로 확립되지 않았다.
- H2: context/planning-conditioned 선택이 동일 예산의 강한 random/규칙/학습 비교군보다
  좋은 planning을 만들 수 있다. 검증할 주 가설이다.
- H3: 정보량 K 또는 horizon의 맥락 적응. 첫 버전에서는 하지 않는다.
- 첫 범위: 고정 K, 고정 horizon, 차량·보행자 instance의 **미래 예측 대상 선택**.
  정차 차량도 후보에서 배제하지 않는다. 도로·신호 등은 공통 맥락으로 유지하는 후보안이다.
- 관측 마스킹, 미래 예측 대상 선택, 예측 후 planner 입력 선택을 구분한다.
- 사용자 승인 GPU는 **0, 1**뿐이다. 승인됐다고 기존 타인 프로세스를 중지하거나
  카드가 비어 있다고 가정하지 않는다.

최신 상세 사용자 프롬프트가 과거 HANDOFF의 강한 주장보다 우선한다.
연구 질문의 변경, 큰 학습·cache 생성은 대안과 비용을 먼저 보고한다.

## 2. 실제 조사 및 변경 현황

### 95015df까지 실행한 조사

1. SafeDrive 지침, git 상태, HANDOFF, 기존 메모리·분석 자산 및 데이터 경로를 읽었다.
2. SafeDrive의 instance 선택, joint world/planning decoder, loss 및 미래 target 정렬 코드를 추적했다.
3. Drive-JEPA 공식 저장소를 작업공간에 shallow clone하고 아래 commit을 고정해 기록했다.
   perception-free model/agent/feature builder 및 encoder loader를 읽었다.
   perception-based 최상위 모델은 읽었지만 내부 scorer/BEV 모듈 감사는 미완료다.
4. Drive-JEPA / AD-E2E-JEPA 원문에서 pretraining과 downstream 경로를 구분해 확인했다.
   다른 출발 논문의 제한적 원문 조사는 있었으나, 전체 novelty 비교표와 코드 검증은 미완료다.
5. 호스트의 GPU 종류와 기존 프로세스를 읽기 전용으로 확인했다.
6. 이 상태 문서와 계산 그래프 **초안**을 작성하고, 과거 인수인계의 오해 소지를 정정했다.

### 이번 v1에서 추가 실행한 것

- 사용자/ChatGPT 검토를 반영해 순차 조건부 ST의 K-slot, invalid, temperature, ID tie,
  permutation 및 모든 객체가 예산에 들어가는 경우를 구현했다.
- CPU fixture에 selector/predictor/planner와 loss별 gradient 경계를 구현했다.
  기존 SafeDrive/Drive-JEPA model에는 아직 붙이지 않았다.
- **13개 unittest 통과**: planning→selector, auxiliary 차단, detach의 forward 불변과
  backward 차단, target 교체 독립성, permutation, padding/empty/invalid/tie 등을 검사했다.
- **3-seed 합성 선택 학습 통과**: 정해진 analytic predictor/planner를 두고 selector만 planning
  MSE로 학습했다. 별도 holdout에서 두 relevant entity의 exact-set 정확도는 99.9756~100%다.
- Command 교란 / no-intent entity-only 학습 / random / motion / fixed-semantic / hindsight
  oracle을 비교했다. 쉬운 인위적 과제이므로 자율주행 H1/H2 증거로 사용하지 않는다.
- 결과·환경·config·source SHA256은 `analysis/research/graph_v1_validation.json`,
  해석·한계·재현 명령은 `docs/GRAPH_V1_VALIDATION.md`에 남겼다.

### 아직 하지 않은 것

- 실제 NAVSIM selector / future predictor / target encoder / entity adapter 연결은 구현하지 않았다.
- 실제 batch forward/backward 및 공식 checkpoint 평가를 수행하지 않았다.
- 새 외부 의존성 설치, 기존 환경 업그레이드, GPU 학습, 전체 cache 재생성을 하지 않았다.
- 공용 데이터셋에 쓰거나 새 데이터셋을 다운로드하지 않았다.
- 최종 baseline 및 tensor adapter를 확정하지 않았다.

95015df는 문서·작업 규칙 변경이었다. 이번 v1은 독립 synthetic fixture와 결과 기록을 추가한다.
기존 모델, loss, 실험 결과 CSV는 변경하지 않았다.

## 3. 실제 경로·버전·자산

| 항목 | 조사 결과 / 범위 |
|---|---|
| SafeDrive | `/rhome/junseong/SafeDrive`, branch `junseong/main` |
| SafeDrive 조사 기준 | `6ed73948d272218c5fd34c3eddf2dc0d26510393` |
| Drive-JEPA 소스 | `/rhome/junseong/research_sources/Drive-JEPA` |
| 공식 소스 기준 | `548bb8215e3aae18e162a0f12f1ba83b4d3eb57e`, [공식 저장소](https://github.com/linhanwang/Drive-JEPA/tree/548bb8215e3aae18e162a0f12f1ba83b4d3eb57e) |
| 현재 호스트 | `user-ESC8000A-E11`; 이전 AXE-080 명칭과 구분 |
| GPU 0 / 1 | RTX A6000, 각 약 48 GB. 확인 당시 양쪽에 기존 프로세스가 있음 |
| 공용 원본 | `/home/user/data/Dataset/` 전체 **절대 직접 수정하지 않음** |
| NAVSIM 읽기 링크 | `SafeDrive/dataset -> /home/user/data/Dataset/navsim -> /mnt/nfs/data/open_dataset/navsim` |
| 코드·변환·cache·결과 | `/rhome/junseong/` 아래 사용자 작업공간 |
| 신규 원본 다운로드 | `/home/user/data/processed_dataset/`, 사용자 지정 총 1 TB 한도 |
| 평가 CSV | `exp/safedrive/**/traj_*.csv` 22개. 존재만 확인; run별 의미와 학습 provenance 재감사 필요 |
| O0 last.ckpt | `exp/safedrive/o0_nofuture/lightning_logs/checkpoints/last.ckpt`, 1,382,312,902 bytes |
| F3 last.ckpt | `exp/safedrive/f3_nopairnc/lightning_logs/checkpoints/last.ckpt`, 1,382,302,278 bytes |
| 분석 축 | `analysis/context_axes.csv`; 생성 코드의 일부 축은 미래 reference 기반 |
| 없는 자산 | 현재 작업공간의 `ckpts/`, `exp/metric_cache_navtest`, `exp/safedrive_train_cache` |
| 환경 | 기존 base / alpasim-cuda128만 확인; safedrive / drive-jepa 전용 환경 없음 |
| 기존 torch | alpasim-cuda128에서 torch 2.8.0+cu128을 읽기 전용 조회. 본 연구 호환성 검증 아님 |
| CPU fixture 실행 환경 | `exp/graph_cpu_env`, Python 3.12.13 / 기존 torch 읽기 전용 참조 venv. 완전 독립 dependency 환경 아님 |

체크포인트 크기는 기존 AICA 전송 기록과 일치한다. 체크섬·내용·resume 적합성은 검증하지 않았다.
공식 Drive-JEPA 전체 cache/weight 다운로드 명령을 무작정 실행하지 않는다.

## 4. 지금까지 확인한 중요한 코드 사실

### SafeDrive

`navsim/agents/safedrive/safedrive_model.py`:

- `_forward_swnet`(859행), `select_topk`(1266행)는 각 ego proposal 경로와 현재 객체 위치의
  최소 거리를 사용한다. 1301행의 `topk(...).indices`로 선택하고 query를 gather한다.
- 선택된 **feature 내용**으로는 gradient가 흐를 수 있지만, 정수 index 선택 자체에는
  선택 정책을 학습시키는 미분 경로가 없다. 현재 선택기는 학습된 importance selector가 아니다.
- world/planning decoder가 agent와 ego query를 함께 갱신한다. 이 경로를 곧바로
  별도 JEPA 미래 latent predictor → planner 구조라고 부르지 않는다.
- `_proposal_object_detection`의 training denoising 경로는 `targets`를 사용한다(603~611행).
  신규 graph 검사에서는 미래 GT의 forward 경로를 분리하고, 우선 DN을 끄는 안을 검토한다.
  이것만으로 기존 코드에 미래 정보 누출이 입증됐다고 주장하지 않는다.

`safedrive_features.py` 284~322행과 `_align_future_agent_states`(689행)는 미래 객체를
현재 ego frame과 현재 track-token 순서로 정렬한다. 활용 가능한 target 생성 참고 코드다.
검출 query ↔ GT token 대응 및 entity latent target 생성은 추가 설계가 필요하다.

`safedrive_loss.py`의 `prediction_loss_weight`와 `pair_Disp_loss_weight`는 별도 경로다.
F2의 과거 결과를 “미래 agent 정보 완전 제거”로 해석하려면 당시 설정·코드를 복원해야 한다.

### Drive-JEPA — 현재 감사한 분기만

공식 commit의 `navsim_v1/navsim/agents/drive_jepa_perception_free/`:

- `drive_jepa_features.py::compute_features`와 `DriveJEPAFeatureDIBuilder`는 현재 또는
  과거+현재 이미지와 driving command / ego velocity / acceleration을 만든다.
- `drive_jepa_model.py::DriveJEPAModel.forward`(87행)는 이미지 encoder → pooling/projection
  → status 결합 → transformer → ego waypoint head를 호출한다.
- `vjepa2/evals/image_classification_frozen/modelcustom/vit_encoder.py::init_module`(39행)은
  checkpoint에서 **encoder**를 로드한다.
- `drive_jepa_agent.py::compute_loss`(128행)는 ego trajectory의 length-normalized L1이다.

이 perception-free downstream 경로에는 별도 entity 미래 latent predictor를 호출하는 코드가 없다.
따라서 “Drive-JEPA를 그대로 사용하면 selector→future predictor→planner가 이미 연결된다”는
가정은 성립하지 않는다. perception-based 내부 경로 전체에 같은 결론을 일반화하지 않는다.

## 5. 이전 해석을 정정 / 유보한 것

- **마스크 token이 학습됨**과 **대상을 고르는 selector가 학습됨**은 다르다.
- AD-E2E-JEPA v1은 2026-09-28 공개다. “3주 전” 표기는 정정한다.
  [원문 §3.4](https://arxiv.org/html/2609.34085v1)는 downstream IL에 patch predictor를
  제거하는 경로를 설명한다. goal-conditioned zero-shot 경로와 이 IL 경로를 섞지 않는다.
- 모든 기존 JEPA 마스크가 입력과 무관하다는 일반화, 두 비교 열이 모두 “아니오”이면 novelty가
  확보된다는 결론은 삭제한다. 직접 선행연구 비교가 더 필요하다.
- `context_axes.py`는 `mc.trajectory`로 `fwd/lat/dheading/bow`를 계산한다. 이는 미래 reference
  기반 분석 축이지, 그대로 online selector 입력으로 사용 가능한 맥락 정보가 아니다.
  `mc.observation.unique_objects`로 얻은 주변 객체 수도 현재 시점 정보인지 추가 감사해야 한다.
- SafeDrive ablation은 미래 감독·용량·객체 수·클래스가 섞여 있다. 모두를 동일한 “미래 정보
  선택” 실험으로 묶지 않는다. DAC 상승만으로 과적합 원인을 확정하지 않는다.
- 다른 split에서 학습한 context 선택은 **cross-fitted 선택 규칙**이지 보장된 oracle upper bound가 아니다.
- navtest는 반복 개발에 사용된 상태다. 최종 독립 평가라고 부르지 않는다.
  navhard 접근 가능성·사용 이력·프로토콜은 별도로 확인해야 한다.

## 6. 다음 담당자가 바로 할 일

1. `SELECTIVE_FUTURE_GRAPH.md`와 `GRAPH_V1_VALIDATION.md`의 구현 범위·결과·편향 한계를 읽는다.
   Auxiliary의 selector 직접 gradient 차단은 초기 실험 선택이지 보편 원칙이 아니다.
2. perception-based Drive-JEPA 내부와 entity/target adapter 비용을 끝까지 확인하여 baseline을
   선택한다. SafeDrive를 주 baseline으로 되돌리는 변경은 임의로 확정하지 않는다.
3. 실제 입력/target tensor, ego-motion 정렬 및 미래 GT 경계를 정의하고 작은 adapter test를 만든다.
4. 구현 경로가 성립하면 독립 환경과 최소 NAVSIM batch로 검증한다. GPU 사용은 0·1로 제한한다.
5. 현재-feature 전달 대조와 강한 동일-K 비교군, novelty 검증을 마친 후 본 실험 진행 여부를 판단한다.

## 7. Codex ↔ ChatGPT 협업 규약

- Git의 `junseong/main`이 공통 기준이다. ChatGPT에는 **commit URL과 이 두 문서**를 전달한다.
  push만으로 다른 대화의 모델이 자동으로 문서를 읽거나 메모리를 공유하지는 않는다.
- 확인된 사실 / 설계 제안 / 실행 결과 / 미확인 항목을 분리하고 근거의 commit·경로·함수를 남긴다.
- ChatGPT의 방법 제안은 아직 구현·검증된 결과가 아니다. Codex가 코드와 실제 결과로 대조한 뒤
  상태를 갱신한다. 의견이 충돌하면 가정·근거·영향을 기록한다.
- 작업 종료 시 상태 문서와 HANDOFF를 갱신하고 RESUME_NOTES에는 실제 실행한 일만 덧붙인다.
  기존 실험 기록은 지우거나 새 결과처럼 재해석하지 않는다.
- 대용량 데이터·checkpoint·비밀키·토큰·환경 credential은 커밋하지 않는다.
- 재현 기준: source commit, config, split, seed, 실행 명령, 환경, 측정 결과를 함께 남긴다.

이 문서를 공유할 때의 첫 요청 예:

> 이 commit의 RESEARCH_STATUS.md와 SELECTIVE_FUTURE_GRAPH.md를 읽어 주세요.
> 구현된 사실과 초안을 구분해, 선택기의 planning gradient 경로와 가장 작은 검증 실험을 검토해 주세요.
