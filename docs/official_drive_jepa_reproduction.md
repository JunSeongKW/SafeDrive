# Drive-JEPA 공식 planning checkpoint 재현

목적은 신뢰할 수 있는 공개 baseline의 추론·평가 동작 확인이다. Selector·pilot 성능 개선·학습은 하지 않는다.
시작 commit: `5c6e6d5fc8f2e20ea5dad90dabce5e1e3f310de8`; 재현 준비/실행 시작 commit: `18663fc`.
**전체 평가 완료**: 공식 perception-free ViT-L planning checkpoint, navtest 12,146/12,146 성공,
PDMS **89.224320**, 논문89.0 대비 **+0.224320 percentage points**. 실패/누락/중복0.
공식 전체 추론·평가 동작은 확인했지만, checkpoint별 result card/허용 오차가 없어
논문 수치의 정확한 재현 성공을 임의 기준으로 선언하지 않는다.

## 안전 정리와 범위

시작 worktree는 clean. 우리 pilot 학습/cache 프로세스는 없었으며 종료한 기존 프로세스 없음.
기존 cache/checkpoint/CSV/manifest/환경 보존, 확대 cache·학습·held-out 평가 미실행 상태를 유지한다.
WA-JEPA는 다음 순위이며 이 작업 종료 후에도 자동 시작하지 않는다. Pilot 재개는 별도 사용자 결정이 필요하다.
GPU0·1만 승인; GPU4–7의 다른 사용자 프로세스는 그대로 둔다.
공용 NAVSIM 원본은 `dataset` 심볼릭 링크를 통해 읽기만 한다. **NAVSIM 원본 재다운로드 없음.**
새 다운로드는 planning checkpoint 3.72GB와 metric cache archive 3.18GB, 기존 encoder 5.13GB 재사용이다.

## 공식 재현 대상 고정

| 항목 | 이번 실행 기준 | 직접 근거 |
|---|---|---|
| 공식 source | `548bb8215e3aae18e162a0f12f1ba83b4d3eb57e` | [공식 저장소](https://github.com/linhanwang/Drive-JEPA/tree/548bb8215e3aae18e162a0f12f1ba83b4d3eb57e), 원격 master와 기존 clean clone 일치 |
| full planning checkpoint | `drive_jepa_perception_free_agent_vitl.ckpt` | [공식 공개 파일](https://huggingface.co/datasets/LinhanWang/Drive-JEPA/blob/65e0d7284f69bf29d1a4864affcfd84ca4e97a2e/drive_jepa_perception_free_agent_vitl.ckpt), README 연결의 HF dataset |
| encoder 초기화 weight | `vitl_merge_3dataset_e50.pt` | 공식 constructor 초기화용. 이것만으로 planning 재현을 주장하지 않음 |
| 모델 | perception-free ViT-L + **공식 전체 Transformer trajectory decoder** | `drive_jepa_perception_free/{drive_jepa_agent,drive_jepa_model}.py` |
| 센서/프레임 | front camera만, history index2·3 = 직전/현재 2프레임 | `get_sensor_config()`, official evaluation `double_image=true` |
| 전처리 | RGB JPEG→상하28px crop→OpenCV512×256→ToTensor→ImageNet 정규화 | `DriveJEPAFeatureDIBuilder`, `DriveJEPAModel.make_transform`; 별도 undistortion 없음 |
| 출력 | 현재 ego rear-axle local XY(m)/heading(rad), 8pose, 0.5s, 4s | 공식 `Trajectory`, agent sampling yaml |
| 데이터 | NAVSIM v1 **navtest 전체**, logs/sensors의 `test` | 공식 split yaml의 token/log 목록 사용; pilot mini/held-out과 별개 |
| devkit | Drive-JEPA `navsim_v1` fork package1.1.0 + NuPlan v1.2 commit`ce3c323af01c0d7ec5672f7832ef53f9c679aab0` | 공식 requirements/setup 및 tagged source |
| scorer | 40pose×0.1s, EP5/TTC5/C2, DDC weight0 | `config/pdm_scoring/default_scoring_parameters.yaml`, 공식 `pdm_score` |
| cache | 공식 `metric_cache.tar`, HF revision`65e0d728…` | [공식 공개 파일](https://huggingface.co/datasets/LinhanWang/Drive-JEPA/blob/65e0d7284f69bf29d1a4864affcfd84ca4e97a2e/metric_cache.tar); 새 다운로드 derivative만 metadata 절대 경로 재작성 |
| 논문 대응 | v2 Table2, **첫 perception-free block의 Ours / ViT-L / Camera** | [원문 PDF, p8](https://arxiv.org/pdf/2601.22032v2); NC98.7/DAC96.2/EP82.9/C100/TTC95.5/PDMS89.0 |
| 공식 명령 | `scripts/evaluation/eval_drive_jepa_perception_free.sh` | 아래 wrapper는 동일 모델·scorer overrides, 경로·thread 상한만 명시 |

선택 이유: 서버에 v1 test logs/sensors/maps가 있고, full PF planning weight·공식 cache·eval script를
명확히 매칭할 수 있다. Perception-based 전용 deformable/MMCV stack을 섞지 않는 첫 재현이다.
이는 최고 headline 모델 재현을 대신한 것이라고 주장하지 않는다.

실제 full planning 파일은3,720,285,081bytes이며 SHA256은
`982960aed9e82900584d20c676ba9a7498add912bed94a3e7fa5f4d22bff6237`이다.
Encoder 초기화와 metric cache의 byte/hash는 [공식 asset 검증 기록](../results/official_drive_jepa_reproduction/verified_assets.json)에 있다.

공개 체크포인트의 정확한 row 동일성은 공식 파일명·script의 연결로 추론했다. 별도 checkpoint별
실측 result card/오차 허용 기준은 발견하지 못했다. 논문 수치와 서버 수치는 별도로 보고한다.
**공식 README/arXiv abstract metadata의 PB93.3과 v2 PDF Table2의 PB93.7은 불일치한다.**
이번 PF 행89.0과 혼동하지 않으며 점수 맞추기를 위해 checkpoint/config를 바꾸지 않는다.

## 격리 환경과 코드 차이

독립 **Conda 환경**: `runtime/environments/drive_jepa_official_evaluation`, Python3.9.23,
Torch2.1.0+cu121/vision0.16.0+cu121/audio2.1.0+cu121. 기존 base·alpasim·pilot env 수정 없음.
OS/CUDA driver 설치·업그레이드 없음. CUDA runtime은 이 Conda prefix의 wheel 내부 것을 사용한다.
패키지 cache도 `runtime/package_cache/pip`로 지정한다.

```bash
conda activate /rhome/junseong/PlanningAwareFuturePrediction/runtime/environments/drive_jepa_official_evaluation
python -m pip --version  # 이 prefix를 가리키는지 확인
```

공식 source 별도 detached worktree `reference_repositories/DriveJEPAOfficialEvaluation`.
기존 `reference_repositories/Drive-JEPA`와 pilot import를 건드리지 않는다.
PF 추론/scorer에 필요한 패키지를 `configs/official_drive_jepa/evaluation_requirements.txt`로 설치한다.
PB 전용 `mmcv_full/mmdet`은 이 변형의 import 경로에서 사용되지 않아 설치하지 않는다.
공식 요구에 없는 버전 선택은 성공 환경의 freeze로 공개하고 환경 일치의 한계로 남긴다.

최소 upstream 차이: encoder 초기화 loader의 silent shape replacement/`strict=False`를 제거하고
missing/shape mismatch는 즉시 실패, `strict=True`로 강화한다.
`configs/official_drive_jepa/strict_encoder_loading.patch`에 보존한다. 순전파·전처리·scorer 변경 없음.
Full planning initialize는 원래부터 strict=True(default)이며 추가 key/shape audit를 수행한다.

## 실행 및 completeness

```bash
cd /rhome/junseong/PlanningAwareFuturePrediction
runtime/environments/drive_jepa_official_evaluation/bin/python scripts/prepare_official_drive_jepa_assets.py
PYTHONNOUSERSITE=1 CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  runtime/environments/drive_jepa_official_evaluation/bin/python scripts/audit_official_drive_jepa_evaluation.py --stage preflight
PYTHONNOUSERSITE=1 CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  runtime/environments/drive_jepa_official_evaluation/bin/python scripts/audit_official_drive_jepa_evaluation.py --stage smoke
# Smoke와GPU점유 확인 후 새 run directory로만 실행:
bash scripts/run_official_drive_jepa_full_evaluation.sh \
  /rhome/junseong/PlanningAwareFuturePrediction/outputs/official_drive_jepa_reproduction/full_navtest_run_v2 0
```

사용자 후속 요청으로 실제 전체 실행은 GPU0·1 병렬, 각worker2/총4이다. 공식136log를 deterministic
largest-count-first로 배분해6075/6071scene, 각각68log, token겹침0/합집합전체12146을 보장한다.
모델·전처리·scorer는 동일하며 새로운 benchmark split이나 scene selection이 아니다.
실행 명령의 마지막 argument `0`/`1`은 execution shard다:

```bash
# prepare_shards는 CPU, 결과 확인 전에 고정하며 모델 결과를 사용하지 않는다.
CUDA_VISIBLE_DEVICES='' runtime/environments/drive_jepa_official_evaluation/bin/python \
  scripts/audit_official_drive_jepa_evaluation.py --stage prepare_shards
# GPU 점유 확인 후 각각 별도 전용 세션에서 실행한다.
bash scripts/run_official_drive_jepa_full_evaluation.sh \
  /rhome/junseong/PlanningAwareFuturePrediction/outputs/official_drive_jepa_reproduction/full_navtest_gpu0_v2 0 0
bash scripts/run_official_drive_jepa_full_evaluation.sh \
  /rhome/junseong/PlanningAwareFuturePrediction/outputs/official_drive_jepa_reproduction/full_navtest_gpu1_v2 1 1
# 끝난 두 CSV에 대해 --csv-path를 두 번 전달하여 원본 scene-wise mean을 구한다.
```

위 `v2` 경로는 재실행이 필요할 때의 예시이며 **이번 v1 평가를 다시 실행하지 않는다**.
실제 v1은 전용 `tmux -L drive-jepa-official-evaluation`의 gpu0/gpu1 세션에서 동시에 실행했다.
완료 후 두 세션과 GPU 작업은 종료됐다. 평가 중 worker 수를 변경하거나 재시작하지 않았다.
각shard가사용하지않는나머지cache token warning은분할실행에따른것이며, 최종합집합에서누락0을검사한다.

공식 runner는 scene/cache token 교집합을 사용한다. 별도 preflight가 공식 token 목록·log·현재 front
이미지·cache 파일의 **누락 0**을 먼저 요구하여 조용한 제외를 방지한다. 결과도 token uniqueness와
모든 scene 성공을 검사하며 실패/누락이 있으면 성공 subset 점수를 전체 재현 수치로 부르지 않는다.
소수3scene smoke 점수는 재현 수치가 아니다. 주점수는 공식 scene-wise mean×100;
pilot의 scene-macro ADE와 다른 지표다. DDC도 출력되나 이번 v1 composite의 weight는0이다.

## 실제 전체 결과 — 2026-10-01

세asset byte/hash확인완료. 독립Conda/PF import/pip check통과.
Preflight공식token12146/log136/front14247image/공식cache12146, 누락0.
OldSafeDrive YAML도동일12146token. 과거sampleCSV의12147행중마지막1행은`average`이며원본12146scene다.
소수3scene fullplanner·encoder strict/finite8×3trajectory/공식scorer통과. Missing/unexpected/shape mismatch모두0.
Warm-up이후inference0.070–0.078s/scene, scorer0.058–0.068s, CUDApeakallocated1.286GB(소수검사).
공식checkpoint metadata epoch36/globalstep49173; file size/hash를근거로파일역할을구분한다.
원본timestampscadence는대부분약0.5s이나소수약1s도있다. 저장원본·공식전처리를그대로사용한다.

| 항목 (0–100 scale) | 논문 v2 Table2 PF ViT-L | 서버 전체 실측 | 차이 (percentage points) |
|---|---:|---:|---:|
| NC — no-at-fault collision | 98.7 | 99.082002 | +0.382002 |
| DAC — drivable area compliance | 96.2 | 96.558538 | +0.358538 |
| EP — ego progress | 82.9 | 83.034487 | +0.134487 |
| Comfort | 100.0 | 99.983534 | −0.016466 |
| TTC | 95.5 | 96.023382 | +0.523382 |
| **PDMS** | **89.0** | **89.224320** | **+0.224320** |
| DDC (v1 score weight0) | 해당 행 미보고 | 98.196937 | 비교 불가 |

논문 값은 한 자리 반올림된 수치다. 서버 값은 공식 scene-wise score의 평균×100이며,
집계된 하위 지표로 composite를 다시 계산하거나 두 shard 평균을 단순 평균하지 않았다.
소수 scene smoke 점수와 자체 pilot ADE는 이 표에 포함하지 않았다.

| Completeness | 수 |
|---|---:|
| 공식 예상 scene / 결과 unique scene / 성공 scene | 12,146 / 12,146 / 12,146 |
| 실패 / 비유한 metric / 누락 / 추가 / 중복 | 0 / 0 / 0 / 0 / 0 |
| GPU0 / GPU1 처리 scene | 6,075 / 6,071 |
| 공식 log / unique required front image | 136 / 14,247 |

각 shard CSV의 마지막 `average` 행 두 개를 제외하고 원본 scene rows를 합쳤다.
결과 무결성5tests(unequal shard 집계/누락/중복/invalid/NaN)와 관련Ruff 검사를 통과했다.
기존 인수인계의12,147는 실제 검토한 과거 CSV에서 average행을 포함한 row count다.
Old split YAML도12,146token으로 같으며, 이번 cache에서 scene 한 개가 빠진 것이 아니다.

원본 CSV와 로그는 `outputs/official_drive_jepa_reproduction/full_navtest_gpu{0,1}_v1/` 및
`outputs/official_drive_jepa_reproduction/logs/`에 보존한다. 원본 CSV SHA256:

- GPU0: `08acb586fefc7c346cff1e661c881afd74df86fb76108f96a2dfc673018953b6`
- GPU1: `cbeb652fae042484f275b53d3fc547af3871e84404436489d1d6e8e18a09b4da`

집계와 설정 대조만 다시 실행할 때 (추론 반복 없음):

```bash
cd /rhome/junseong/PlanningAwareFuturePrediction
PYTHONNOUSERSITE=1 CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  runtime/environments/drive_jepa_official_evaluation/bin/python \
  scripts/audit_official_drive_jepa_evaluation.py --stage summarize \
  --csv-path /rhome/junseong/PlanningAwareFuturePrediction/outputs/official_drive_jepa_reproduction/full_navtest_gpu0_v1/2026.10.01.21.27.02.csv \
  --csv-path /rhome/junseong/PlanningAwareFuturePrediction/outputs/official_drive_jepa_reproduction/full_navtest_gpu1_v1/2026.10.01.21.26.32.csv
PYTHONNOUSERSITE=1 CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  runtime/environments/drive_jepa_official_evaluation/bin/python \
  scripts/summarize_official_drive_jepa_execution.py
PYTHONNOUSERSITE=1 CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  runtime/environments/drive_jepa_official_evaluation/bin/python -m unittest discover \
  -s tests -p test_official_drive_jepa_results.py -v
```

## 실측 비용과 worker 판단

| 항목 | 실측 |
|---|---|
| Full evaluation 시작 / 전체 완료 (KST) | 21:14:44 / 21:27:05 |
| 전체 병렬 wall time | 약741s = 12분21초 |
| GNU time GPU0 / GPU1 wall time | 740.59s / 710.22s |
| 처리율 (초기화/IO/scorer 포함) | 전체16.391scene/s |
| Worker | GPU마다2 process, 총4; BLAS/OMP thread1 |
| 샘플된 device VRAM peak | GPU0 7,810MiB / GPU1 7,811MiB |
| GNU time maximum reported child RSS | GPU0 7,547,264KiB / GPU1 7,134,856KiB |
| Shard wall의 합 / 3600 | 0.403 GPU-allocation-hour; 실제 SM active time/FLOPs가 아님 |

시간은 full evaluation 실행부터 완료까지이며 다운로드·환경 구성·preflight·smoke는 제외한다.
VRAM telemetry는 실행 도중 시작한15초 간격 표본이므로 전체 실행의 연속 peak가 아니고,
GNU time RSS도 concurrent worker 메모리 총합이 아니다.
완료 뒤 GPU0·1은 각각25MiB/compute process 없음으로 확인했다. 다른 GPU 작업은 건드리지 않았다.

GPU memory에는 더 많은 worker를 담을 여지가 있었지만, utilization 한 시점만으로 최적 worker 수를
정할 수 없다. 공식 runner는 결과를 메모리에 누적하고 끝에 CSV를 쓰므로 worker 증가는 현 실행의
재시작을 요구했다. 이미 진행된 작업을 보존해 끝까지 마쳤다. Worker scaling benchmark는 하지 않았고
GPU마다4개가 더 빠르다고 검증한 것은 아니다. 필요하다면 이후 고정 소수 표본에서만 profile한다.

## 논문 대비 차이 감사와 판단 범위

| 확인 순서 | 직접 확인된 사실 | 남은 불확실성 |
|---|---|---|
| 1. Checkpoint / 변형 | 공식 HF revision의 full PF ViT-L weight byte/SHA 일치; strict missing/unexpected/shape0; epoch36/global_step49173 | 논문 행의 정확한 원 실행과 이 공개 파일의1:1 result card 없음 |
| 2. Split / 누락 | 공식 navtest token12,146, 전부 성공; raw scene별 union/duplicate/finite 검사 | 논문 원 실행별 scene CSV 미공개여서 per-scene 대조 불가 |
| 3. NAVSIM / scorer / cache | source548bb82의v1 fork/NuPlanv1.2/공식cache hash; 저장된 두 Hydra의 scorer·sampling이 pinned 설정과 같음 | 논문 원 실행의 전체 package lock/cache 생성 source provenance와 직접 대조하지 못함 |
| 4. 전처리 / 센서 / 궤적 | 공식2front frame·crop/resize/ImageNet·float32·8pose/0.5s; pilot ROI/undistortion/predictor 미사용 | 원 dataset export bytes까지 논문 실행과 동일함을 확인하지 못함 |
| 5. 실행 환경 / seed | 독립Py3.9/Torch2.1+cu121, 공식eval/no_grad/dropout0, autocast없음; GPU0·1 동일 model/scorer 확인 | 원 평가 seed/정밀도/하드웨어 전체 정보 없음; 차이의 정확한 원인 미확정 |

위 점검에서 checkpoint loading, 설정 불일치, 누락 scene 또는 잘못된 shard 집계는 발견하지 못했다.
**+0.224320점의 원인은 확정하지 않았다.** 수치를 맞추기 위한 scorer/전처리/checkpoint 변경이나
추가 sweep을 하지 않았다. “논문보다 개선된 새 방법” 또는 임의 허용 오차의 “정확 재현 성공”을 주장하지 않는다.

재현된 것: 공개 **전체 planning checkpoint** strict loading, 공식 입력/decoder/trajectory/scorer,
공식 navtest 전체 추론·평가. 미확인: 논문의 원 실행과 수치가 정확히 같아야 하는 조건/허용 오차.
이번 perception-free forward는 encoder→trajectory decoder이며 추론 중 별도 미래 predictor를 호출하지 않는다.
따라서 이 결과는 신뢰 가능한 planning 평가 baseline 확보이지 선택적 미래 예측/동일 예산 효과의 검증이 아니다.
학습·fine-tuning·selector 추가는 없었으며 pilot·WA-JEPA는 보류 상태다. 다음 기반 활용 방향은 별도 검토한다.

## ChatGPT / Claude 검수 자료

- [실측 점수·누락/실패 검사 JSON](../results/official_drive_jepa_reproduction/full_navtest_results.json)
- [12,146 scene별 원본 지표 CSV](../results/official_drive_jepa_reproduction/official_scene_scores.csv)
- [비용·실제 Hydra 설정 대조](../results/official_drive_jepa_reproduction/execution_cost.json),
  [GPU 표본 telemetry](../results/official_drive_jepa_reproduction/execution_telemetry.json)
- [공식 파일 크기/hash](../results/official_drive_jepa_reproduction/verified_assets.json),
  [환경 정보](../results/official_drive_jepa_reproduction/environment.json),
  [pip freeze](../results/official_drive_jepa_reproduction/environment_pip_freeze.txt)
- [고정 설정](../configs/official_drive_jepa/reproduction_v1.json),
  [최소 upstream patch](../configs/official_drive_jepa/strict_encoder_loading.patch)
- [Runner](../scripts/run_official_drive_jepa_full_evaluation.sh),
  [fail-closed 집계](../scripts/audit_official_drive_jepa_evaluation.py),
  [5개 결과 무결성 tests](../tests/test_official_drive_jepa_results.py)

공유 시 이 보고서를 포함한 결과 commit SHA를 고정해 GitHub의 `blob/<SHA>/...` 링크로 전달한다.
체크포인트·대용량 metric cache/환경은 Git에 넣지 않는다. 기존 공개 설정과 원격은 유지한다.
