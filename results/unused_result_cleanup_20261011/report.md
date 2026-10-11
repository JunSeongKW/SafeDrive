# 사용하지 않는 실험 산출물 정리 — 2026-10-11

## 완료 결과

사용자 요청으로 폐기된 profile 가중치와 이후 상태로 대체된 복구 snapshot을 정리했다.
**87개 파일 / 85개 고유 inode / 54,441,963,520 bytes = 54.44 GB = 50.70 GiB**의 파일 할당량을 해제했다.
파일별 논리 용량을 더하면 하드링크가 중복되므로, 확보 용량은 마지막 링크가 삭제된 inode를 한 번씩 계산했다.
실제 삭제 완료: **2026-10-11 20:17:50 KST**.

정리 후 프로젝트 전체를 다시 집계했다: **325,744,136,192 bytes = 325.74 GB = 303.37 GiB**.
정리 전 `du -h`는 반올림된 355 GiB였다. 실행 중인 학습의 추가 저장으로 전체 용량은 계속 변할 수 있다.

- [삭제 파일 전체 목록](deleted_files.csv): 경로, 사유, 기존 할당량, inode, 하드링크 수, 대체 checkpoint.
- [실제 검사·용량·학습 상태](summary.json).
- 로컬 상세 감사: `outputs/unused_result_cleanup_20261011/audit.json`, `quarantine_verified.json`, `cleanup_complete.json`.

## 삭제 범위와 판단 근거

1. 배치/워커/DDP/SDPA/oracle/offload/병렬 실행 검사에만 사용한 `.pt` 가중치와 gradient dump.
2. 중단된 512×256 축소실험의 옛 메모리 복구·정책 변경 snapshot. 각각 더 진행된 `latest.pt`를 보존했다.
   JEPA 911/1,428/1,704 → 보존 3,200; LPWM sequential 743/1,317/1,937 → 보존 3,200;
   DrivoR 2,453 → 보존 3,200; LPWM joint 87/285 → 보존 2,264.
3. `ddp_engineering` 아래의 `epoch_01.pt` 한 개는 128장면/epoch당2update의 실행 검사 가중치다.
   실제 103,288장면 본학습의 `navsim_v1/epoch_01.pt`는 보존했다.

경로 이름만으로 현재 학습이 중단됐다고 판단하지 않았다. 실행 중인 128×128 전방4프레임 Stage2의 root 전체를 삭제 범위에서 제외했다.
과거의 고유 최종 가중치를 임의로 삭제하지 않았으며, 고유 최종·에폭별 scientific checkpoint와 원본 cache는 이번 정리 범위가 아니다.
삭제된 engineering/복구 snapshot 자체를 다시 읽는 과거 명령은 재실행할 수 없다.
과거 JSON의 경로·SHA·수치·실패 이력은 원문대로 남겼고, 파일 존재 여부는 이 삭제 manifest와 함께 해석해야 한다.

## 확보 용량

| 출력 폴더 | 확보 GB (10^9 bytes) |
|---|---:|
| `four_model_small_corpus_v1` | 21.333 |
| `lpwm_card_budget_measured_v4` | 4.991 |
| `lpwm_object_future_planning_v3` | 4.027 |
| `lpwm_adapter_original_batch_rebalance_v1` | 3.574 |
| `lpwm_drivor_lora_v1` | 3.426 |
| `lpwm_48gb_planning_v1` | 3.303 |
| `lpwm_drivor_optimized_execution_v1` | 2.804 |
| `lpwm_partial_planning_profiles` | 2.425 |
| `lpwm_drivor_joint_v1` | 1.555 |
| `lpwm_driving_video_512x256_v1` | 1.315 |
| `lpwm_card_budget_expandable_v3` | 0.963 |
| `lpwm_card_budget_planning_v2` | 0.963 |
| `lpwm_frozen_control_v1` | 0.913 |
| `lpwm_stage1_effect_v1` | 0.913 |
| `lpwm_drivor_planning_path_lora_v1` | 0.769 |
| `lpwm_drivor_geometry_lora_v1` | 0.729 |
| `lpwm_navsim_posttraining_v1` | 0.439 |
| **합계** | **54.442** |

## 보존 및 검사

- **현재 학습**: `outputs/lpwm_front_history_stage1_lora_v1/` 전체, source/config/input SHA **293개** 일치.
- **Stage1**: `outputs/lpwm_navsim_full_posttraining_v2/stage1/checkpoint.pt`, optimizer 포함 `latest.pt`, 이미지 cache와 manifest.
  Raw Stage1 SHA `71478ee548376bec21a4a22bc219929955aa0bfdd876f704676ade169fb4831f` 일치.
- **이전 모델**: Adapter13,480 / 4camera planning-path LoRA5,493 / 축소 DrivoR·JEPA·LPWM의 최종·에폭별 checkpoint.
- **후속 연구 재사용**: encoder/spatial feature cache, metric cache, GT와 candidate label, 실행 환경, 공식 참고 저장소.
- **실험 기록**: JSON/CSV/로그/NPZ/문서/시각화는 삭제하지 않았다. 요청된 particle PNG **1,615개** 존재 확인.
- **Git**: 정리 직전 추적 파일 **1,794개**를 SHA 비교해 원문 보존 확인.
- **공용 데이터와 다른 프로젝트**: 접근·삭제 대상이 아니다. 원본 데이터셋과 symlink는 보존했다.

Host에서 사용자 작업 **238개**의 cmdline/cwd/fd/maps를 검사했고 후보 참조는 0개였다.
FD 조회가 금지된 OS 인증 `sd-pam`과 SSH transport 2개는 명시적으로 제외했다. SSH 자식 작업은 별도로 검사했다.
삭제 후보와 무관한 기존 NFS stale FD 2개를 기록했으며, 후보의 모든 하드링크 이름을 목록으로 확인했다.
`.trash-unused-experiment-artifacts-20261011/`로 먼저 이동 → 동일 inode·크기·mtime·보호 의존성·사용 참조 재검사 → unlink 순서로 삭제했다.
검사 실패 시 unlink 전에 원래 경로로 복원하도록 구현했다. 격리 폴더는 현재 제거 완료다.

## 학습 연속성

삭제 전 **12,413** → 삭제 후 **12,420 / 29,425 update**,
11번째 epoch 학습 중. 현재 queue PID1533508, torchrun574536을 변경하거나 중단하지 않았다.
새 GPU 작업·학습·평가는 실행하지 않았다.

## 정리 코드

`scripts/cleanup_unused_experiment_artifacts.py`는 이번 날짜와 명시된 출력 계열만 처리한다.
기본 action은 read-only audit이며, cleanup은 저장된 감사 결과를 재검사한 뒤 격리·삭제한다.
이번 실행 명령:

```bash
cd /rhome/junseong/PlanningAwareFuturePrediction
python3 scripts/cleanup_unused_experiment_artifacts.py --action audit
python3 scripts/cleanup_unused_experiment_artifacts.py --action cleanup
```

Host `/proc`의 실제 작업을 보도록 승인된 호스트 실행을 사용했다. 샌드박스 안에서 학습 PID가 보이지 않으면 삭제를 거부한다.
이미 완료된 cleanup 재실행도 거부한다. 별도 미사용 파일을 삭제하려면 새 감사와 새 범위 기록이 필요하다.
