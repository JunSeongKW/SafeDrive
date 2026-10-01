# Drive-JEPA 공식 planning checkpoint 재현

목적은 신뢰할 수 있는 공개 baseline의 추론·평가 동작 확인이다. Selector·pilot 성능 개선·학습은 하지 않는다.
시작 commit: `5c6e6d5fc8f2e20ea5dad90dabce5e1e3f310de8`. 최종 수치는 평가 완료 후 갱신한다.

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
  /rhome/junseong/PlanningAwareFuturePrediction/outputs/official_drive_jepa_reproduction/full_navtest_run_v1 0
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
  /rhome/junseong/PlanningAwareFuturePrediction/outputs/official_drive_jepa_reproduction/full_navtest_gpu0_v1 0 0
bash scripts/run_official_drive_jepa_full_evaluation.sh \
  /rhome/junseong/PlanningAwareFuturePrediction/outputs/official_drive_jepa_reproduction/full_navtest_gpu1_v1 1 1
# 끝난 두 CSV에 대해 --csv-path를 두 번 전달하여 원본 scene-wise mean을 구한다.
```

지속 실행 세션: `tmux -L drive-jepa-official-evaluation ls`, 세션gpu0/gpu1.
각shard가사용하지않는나머지cache token warning은분할실행에따른것이며, 최종합집합에서누락0을검사한다.

공식 runner는 scene/cache token 교집합을 사용한다. 별도 preflight가 공식 token 목록·log·현재 front
이미지·cache 파일의 **누락 0**을 먼저 요구하여 조용한 제외를 방지한다. 결과도 token uniqueness와
모든 scene 성공을 검사하며 실패/누락이 있으면 성공 subset 점수를 전체 재현 수치로 부르지 않는다.
소수3scene smoke 점수는 재현 수치가 아니다. 주점수는 공식 scene-wise mean×100;
pilot의 scene-macro ADE와 다른 지표다. DDC도 출력되나 이번 v1 composite의 weight는0이다.

## 결과 (실행 완료 후 갱신)

세asset byte/hash확인완료. 독립Conda/PF import/pip check통과.
Preflight공식token12146/log136/front14247image/공식cache12146, 누락0.
OldSafeDrive YAML도동일12146token. 과거sampleCSV의12147행중마지막1행은`average`이며원본12146scene다.
소수3scene fullplanner·encoder strict/finite8×3trajectory/공식scorer통과. Missing/unexpected/shape mismatch모두0.
Warm-up이후inference0.070–0.078s/scene, scorer0.058–0.068s, CUDApeakallocated1.286GB(소수검사).
공식checkpoint metadata epoch36/globalstep49173; file size/hash를근거로파일역할을구분한다.
**전체공식navtest는GPU0·1로실행중이며최종점수미확정**. 소수점수를재현점수로부르지않는다.
원본timestampscadence는대부분약0.5s이나소수약1s도있다. 저장원본·공식전처리를그대로사용한다.
Shared JSON: `results/official_drive_jepa_reproduction/` (실측 생성 후 commit).
이 작업은 학습·선택 가설의 검증이나 안전성 개선을 주장하지 않는다.
