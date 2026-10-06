# 첫 epoch 검토 후 나머지24epoch 재개

## 현재 결정과 실행 상태

사용자가 제안한 중간 검토를 위해, 첫 epoch의 정확한 재개 상태를 보존한 뒤 본학습을 대기시키는 CPU 제어를 등록했다.
기존 등록 trainer/model/config/optimizer 설정은 변경하지 않았다. 총25epoch 스케줄도 유지한다.

- 제어: `scripts/queue_lpwm_drivor_epoch_review.py --config configs/lpwm_drivor_review/epoch1.json`.
- 제어 PID:2949030. Root:`outputs/lpwm_drivor_epoch1_review_v1`.
- 대상:epoch1/1614updates, 이후38736updates(24epoch)가 남는다.
- 2026-10-06 09:40KST 확인:1541update, 제어상태`waiting_for_epoch_boundary`, 아직 학습 pause 아님.
- 최근100update 속도19.21초 기준 첫 epoch 예상10:04KST. 공유 부하에 따라 변동한다.
- 본학습 완료→96장면 표현 진단→DrivoR 비교·해석을 위한 대기. DrivoR 비교 방식은 사용자에게 선택을 요청했으며 미확정이다.
- 현재 남은24epoch의 자동 재개는 보류한다. 기존 후속navtest/v2/EPDMS 계획은 보존된다.

## 정확히 무엇을 보존하는가

Epoch 끝에서 trainer가 `latest.pt`에 model/AdamW/scheduler/두rank RNG/다음epoch 위치를 저장한 뒤
`epoch_01.pt`를 만든다. 제어는 이 marker를 감지하고 `latest.pt`를 먼저 열린 파일로 확보한 뒤 기존 pause 신호를 쓴다.
열린 파일을 보존하므로 trainer가 `latest.pt`를 atomic replace해도 복사 대상은 바뀌지 않는다.

원래 trainer는 update 경계에서 pause하므로, 다음 epoch의 update가 이미 진행 중이면 한 번 더 계산될 수 있다.
그 상태는 `stopped_process_resume.pt`로 별도 보존하고, **검토 및 재개 기준은1614의 `epoch_01_resume.pt`**로 고정한다.
`training_held.json`에 정확한epoch checkpoint와 실제중단update 차이를 기록한다. 추가update를 조용히1epoch에 포함하지 않는다.

보존 상태 검사는 다음을 요구한다.

- `completed_updates=1614`, `epoch=1`(다음 epoch의 zero-based index), `next_update_in_epoch=0`.
- `total_updates=40350` 및 기존 configuration SHA.
- AdamW state/step, scheduler state, 두rank RNG 존재.
- 원래LPWM frozen SHA 및 checkpoint SHA 기록.

최신파일교체경쟁, 기존사용자pause보호, 잘못된epoch checkpoint거부의 CPU 검사3개를 통과했다.
기존 학습·후속queue·monitor가 pause를 처리해 종료하면 같은1614checkpoint로 기존 읽기전용 표현진단을 실행한다.
따라서 기존queue의`stopped/Training or queue paused` 표시는 이 제어가 만든 의도된 중단일 수 있다.

## 첫 epoch 결과를 읽을 때 주의할 학습 스케줄

현재 스케줄은1epoch완결학습이 아닌 **25epoch학습의첫1epoch**다. 등록설정상 warmup은3322updates이며,
첫epoch1614는여전히warmup안에있다. 첫epoch결과를최종수렴성능으로판단하지않는다.
`epochs=1`로 바꿔 다시 scheduler를 만들면 warmup/cosine이 달라지므로 같은 학습이 아니다.
재개는 저장된 optimizer/scheduler/RNG를 복원하고 기존25epoch목표를 유지한다.

## 검증 범위와 데이터 문제

### 첫 epoch 점검의 타당성과 선행연구 근거 (2026-10-06)

첫 epoch의 checkpoint를 평가하는 것은 학습 연결·안정성·초기 변화 확인에 적합하다.
같은 학습량의 두 모델을 비교해도 알 수 있는 것은 그 예산에서의 성능이며, 최종 수렴 후 순위를 보장하지 않는다.
논문 저자들이 모두 첫 epoch 점수로 나머지 학습의 진행 여부를 결정한다고 주장할 근거는 없다.

공식 공개 설정을 확인한 결과는 다음과 같다.

| 연구 | 공개된 학습·검증 설정 | 해석 |
|---|---|---|
| DrivoR | NAVSIM v1 25 epoch / v2 10 epoch; 기본 `check_val_every_n_epoch=1`, `val_check_interval=1.0`; `trainer.fit`에 validation loader 전달 | 매 epoch 검증 경로가 있으나, 1 epoch를 최종 성능으로 보고하는 설정은 아님 |
| DiffusionDrive | NAVSIM 학습 명령 `trainer.params.max_epochs=100` | 공식 최종 학습 예산은 100 epoch; 이 문서만으로 중간 검증 빈도는 단정하지 않음 |
| VAD base | Stage 1 48 epoch / Stage 2 12 epoch; 두 설정 모두 checkpoint 매 epoch 저장, `evaluation.interval=total_epochs` | 저장 주기와 평가 주기는 다르며, 기본 평가는 각 단계 마지막 |

근거: [DrivoR README](https://github.com/valeoai/DrivoR),
[DrivoR 기본 설정](https://github.com/valeoai/DrivoR/blob/main/navsim/planning/script/config/training/default_training.yaml),
[DrivoR training 진입점](https://github.com/valeoai/DrivoR/blob/main/navsim/planning/script/run_training_full.py),
[DiffusionDrive 학습 문서](https://github.com/hustvl/DiffusionDrive/blob/main/docs/train_eval.md),
[VAD Stage 1](https://github.com/hustvl/VAD/blob/main/projects/configs/VAD/VAD_base_stage_1.py),
[VAD Stage 2](https://github.com/hustvl/VAD/blob/main/projects/configs/VAD/VAD_base_stage_2.py).

DrivoR의 `validation_run=false`는 검증 전용 실행 대신 `trainer.fit`을 선택한다는 뜻이며,
학습 중 validation loader를 비활성화한다는 뜻이 아니다. 단, v1 full recipe의 train/val 중복은 아래처럼 별개 문제다.

현재 첫 epoch는 1614 update이고 warmup은 3322 update다. 낮은 첫 epoch 점수만으로 LPWM 가설을 탈락시키지 않는다.
진단에서는 gradient·실제 parameter 변화·readout·particle/미래 개입을 먼저 확인한다.
현재 geometry head에 직접 LoRA가 없는 구조는 학습량과 별도로 판단할 수 있다. Epoch를 더 늘려도 그 경로가 새로 생기지는 않는다.
DrivoR 동일 1 epoch 비교는 초기 학습 속도를 비교하는 선택지이며, 이 진단의 선행 필수 조건은 아니다.
이후 성능 추세는 warmup 이후 미리 정한 checkpoint에서도 비교하는 것이 적절하다. 이번 설명으로 학습 예산·hold·queue를 변경하지 않았다.

현재v1은navtrain85109+navval18179=103288장면 모두 학습한다. 따라서 **navval은 독립 검증셋이 아니다**.
기존96장면monitor도 training-distribution 진단이며 여기서 나온oracle score를navtest PDMS로 부르지 않는다.

첫epoch에서 바로 가능한 검사는:

1. Loss 추세/수치안정성/영역별gradient/원래LPWM freeze/실제parameter 변경.
2. 고정96장면의particle 분포,객체·미래readout,명령경로분리,미래/관련particle 개입.
3. 같은고정장면에서DrivoR와궤적·oracle·실패사례 비교(모델이학습한분포임을명시).

독립적인 일반화 검증을 위해서는 처음부터제외한development recording이 필요하다. 이미학습한navval을
뒤늦게독립검증이라고부를수없다. 이번중간review를이유로navtest를반복튜닝용으로자동전환하지않는다.
현재설계에서는최종fullnavtest/EPDMS를고정계획으로보존한다. 추가독립평가의용도·비교군은별도명시해야한다.

## DrivoR 비교 선택지

사용자에게 아래두방식을 구분해 선택을 요청했다.

| 방식 | 알 수 있는 것 | 제한 |
|---|---|---|
| DrivoR도동일스케줄의첫1epoch학습 | 같은데이터노출·update수에서학습진행정도비교 | 추가학습시간필요;다른backbone/pretraining/해상도차이는남음 |
| 공개DrivoR최종모델과참고비교 | 현재모델과완성된baseline의성능·실패양상차이 | 1epoch 대25epoch로학습량이다르므로표현우월성비교가아님 |

공식DrivoR README의v1recipe는25epoch다. 공개모델은[공식Releases](https://github.com/valeoai/DrivoR/releases)에서제공된다.
동일학습량비교를선택하면backend초기state/seed/data순서/effective64/학습update와25epoch scheduler를맞춰야한다.
특히`max_epochs=1`로scheduler까지단축한DrivoR와현재모델을비교하지않는다.
해상도·사전학습·encoder intent·LoRA예산차이는기존공정성감사대로보고한다.
새DrivoR학습/다운로드/benchmark평가는이번제어등록으로자동실행하지않았다.

## 재개할 때 지켜야 하는 순서

1. `status.json`, `resume_state.json`, `training_held.json`과epoch1진단·DrivoR비교결과를확인한다.
2. 구조/loss를그대로유지하면`epoch_01_resume.pt`를기준으로기존25epoch훈련을이어간다.
3. 기존실중단`latest.pt`및launch/pause메타데이터를보존한뒤,검사된1614파일을atomic하게`latest.pt`로복원한다.
4. 이번제어가만든pause만해제한다. 별도의사용자pause가있으면그대로존중한다.
5. 동일`train_lpwm_drivor_lora_parallel.py`와batch16/loader2/oracle4execution을사용한다.
6. 새trainer PID로launch기록을갱신하고,기존후속queue·checkpoint monitor를한번씩재기동한다.
7. Resume메타데이터의optimizer/scheduler/RNG복원과1615부터진행됨을확인한다.

Geometry-LoRA를추가하는등모델/loss를바꾸면동일실험의나머지24epoch이라고부를수없다.
그경우기존조건과checkpoint를보존하고새조건으로등록한다. 현재geometry추가안은여전히미적용이다.
