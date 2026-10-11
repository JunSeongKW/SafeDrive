#!/usr/bin/env python
"""Package measured metric-evidence artifacts without training or rescoring."""
import argparse
import json
from pathlib import Path
import sys

PROJECT_ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(PROJECT_ROOT/"scripts"))
from calibrate_frozen_drivor_scorer import file_sha256,protected_training_audit,write_json


def main(configuration_path):
    configuration=json.loads(configuration_path.read_text())
    output=PROJECT_ROOT/configuration["output_directory"]
    result=PROJECT_ROOT/configuration["results_directory"]
    results=json.loads((result/"results.json").read_text())
    diagnostics=json.loads((result/"functional_diagnostics.json").read_text())
    evolution=json.loads((result/"attention_training_evolution.json").read_text())
    registration=json.loads((output/"registration.json").read_text())
    assert all(file_sha256(PROJECT_ROOT/name)==digest for name,digest in registration["sources"].items())
    protection=protected_training_audit(configuration)
    source_hashes={**registration["sources"],str(Path(__file__).relative_to(PROJECT_ROOT)):file_sha256(Path(__file__)),
        "scripts/visualize_metric_evidence_training_steps.py":file_sha256(PROJECT_ROOT/"scripts/visualize_metric_evidence_training_steps.py")}
    checkpoints={path.name:file_sha256(path) for path in output.glob("*_step*.pt")}
    write_json(result/"execution_provenance.json",{"sources":source_hashes,"input_hashes":registration["inputs"],
        "checkpoints":checkpoints,"environment":results["environment"],
        "determinism_context": {
            "environment_snapshot_phase": "report",
            "training_source_enables_deterministic_algorithms": True,
            "training_source": "scripts/run_metric_visual_evidence_pilot.py:376",
            "training_runtime_flag_separately_recorded": False,
            "bitwise_replay_verified": False,
        },"protected_training":protection,
        "first_launcher_attempt":"Background shell exited before creating outputs; empty log and no duplicate host pilot process. Foreground nohup session37149/controller37151 then executed the registered queue.",
        "sandbox_cuda_unavailable":True,"host_gpu_execution_authorized":True,"original_native_weights_unchanged":True,
        "official_labels_source":"Previous scorer-selection pilot; native64 slices only. Its unchanged pair evaluator8300 checks had maximum7-metric error0.",
        "completed_after_core_queue":"Read-only CPU attention evolution and packaging; original learning sources/config unchanged"})
    main_names=("unchanged","shared","metric_specific")
    lines=["# NC·DAC 지표별 시각 표현: 실제 pilot 결과", "",
        "**결론: 이번 설정에서 planning 개선과 NC·DAC별 의미 있는 공간 특화를 확인하지 못했다.** 구현·CPU/실제캐시 검증·두조건500step·공통후보평가·heatmap·기능적 개입을 모두 완료했다. 사전 등록한 replication gate는 미충족이며 추가 본학습을 시작하지 않았다.","",
        "## 실제 비교 조건","",
        "- Local DrivoR pass5/3200update checkpoint. 원논문의 충분한 학습을 재현한 baseline이 아니다.",
        "- 전방1대·과거현재2frame·512×256/crop28/우6·하10padding. 백본의기존LoRA/native/원planner모두freeze/eval.",
        "- 같은forward finalDINO norm에서19×37×2=1406개/384차원FP16공간token캐싱. CLS/register prefix를제외했으며,기존register16×256/후보scoredfeature/logit/ego는그대로재사용.",
        "- 공식NAVTRAIN1024/101recording. NAVVAL64/15recording에서checkpoint선택,별도192/37recording평가. Recording중복0,이전에사용한개발패널이므로독립최종NAVTEST가아니다.",
        "- 후보native64 고정,각장면64개공식label재사용. Label/후보수/후보context/백본/기존scorer/원score조합동일. 새로운candidate획득·semantic감독·futurepredictor·encoderFT없음.",
        "- Shared/specific각126978trainableparameter,commonstate/batch8/500step/seed0/AdamW3e-4/배치·pair순서동일. 공간patch추가입력도동일.",
        "- Query=공통ego+path MLP 출력+metric embedding. Shared는2metric attentionlogit평균후softmax,Specific은각metricsoftmax. 따라서Shared는두metric이같은feature를읽는다. Effective query자유도차이는구조개입이며파라미터수동일이모든표현용량동일을뜻하지는않는다.",
        "- 읽은feature와기존candidatefeature를보정MLP에입력하고NC/DAC2logit만boundedresidual을더한다. 다른4component는bitwise원출력유지. 기존candidatefeature우회경로가있다는한계는아래에명시한다.",
        "- 공식NC/DACtarget BCE+실제PDMSgap가중pairwise softplus. 공식evaluator/argmax를미분하지않는다.","",
        "## 실제 별도 개발192장면 결과","",
        "| 조건 | PDMS | 같은64후보최고PDMS | 선택손실 | NC실패 | DAC실패 |",
        "|---|---:|---:|---:|---:|---:|"]
    for name in main_names:
        summary=results["summaries"][name]
        lines.append(f"| {name} | {summary['selected_pdms']:.4f} | {summary['oracle_pdms']:.4f} | {summary['selection_regret']:.4f} | {100*summary['NC_failure_rate']:.2f}% | {100*summary['DAC_failure_rate']:.2f}% |")
    lines.extend(["",f"Specific−shared **{results['metric_specific_minus_shared_pdms']:+.4f}점**, recordingpaired-bootstrap95%구간**{results['metric_specific_minus_shared_recording_95ci']}**. Specific−원scorer {results['summaries']['metric_specific']['minus_unchanged_pdms']:+.4f}점. 개선근거없으며통계적악화확정도아니다.",
        "", "두조건모두500step학습했다. Tuning기준으로Shared는500step,Specific은100step모델을선택했다. 평가와기능진단의Specific은100step이다. 별도evolutionPNG에는같은이미지/같은후보의마지막500step도포함하며,이를평가checkpoint와혼동하지않는다.","",
        "| 주행명령 | 장면수 | 기존 | Shared | Specific |","|---|---:|---:|---:|---:|"])
    for command,label in (("0","좌회전"),("1","직진"),("2","우회전")):
        values=[results["summaries"][name]["by_command"][command] for name in main_names]
        lines.append(f"| {label} | {values[0]['scenes']} | {values[0]['selected_pdms']:.2f} | {values[1]['selected_pdms']:.2f} | {values[2]['selected_pdms']:.2f} |")
    lines.extend(["","## 실제 기능적 의존성과 표현 진단","",
        f"- Specific은원scorer와{100*results['summaries']['metric_specific']['selected_index_changed_from_original_fraction']:.2f}%장면에서선택이달라졌다. OFF시모든선택복원/score최대차4.77e-7.",
        "- Uniform attention으로바꾸거나NC/DACattention을교환해도192장면모두최종선택이그대로였다. 점수의작은변화는있지만지표별공간분리가선택에유의미한역할을한다는근거가없다.",
        f"- Train-onlymeanpatch로추가patch입력을대체하면선택이{100*results['interventions']['metric_specific']['patch_mean']['selected_index_changed_fraction']:.2f}%바뀌었다. PDMS는{results['summaries']['metric_specific_patch_mean']['selected_pdms']:.4f}이다. 이변형은추가patch분기만바꾸며기존candidatefeature의시각정보는유지한다. 영상정보전체의인과효과로해석하지않는다.",
        f"- 고정12장면(각실제명령첫4scene)에서attention entropy/maximum={diagnostics['mean_normalized_attention_entropy']:.5f},NC/DAC totalvariation={diagnostics['mean_NC_DAC_attention_total_variation']:.5f},candidate간totalvariation={diagnostics['mean_candidate_attention_total_variation']:.5f}. 이미지전체를넓게읽고두지표의분리는약했다.",
        f"- 초기대비attention totalvariation={diagnostics['mean_before_after_attention_total_variation']:.5f}; 학습에따른변화와metric전문화/실제선택이득은별개다.",
        f"- 상위10%token mean대체의NCabslogit변화 {diagnostics['feature_replacement_summary']['NC']['mean_important_abs_logit_change']:.5f},random동수 {diagnostics['feature_replacement_summary']['NC']['mean_random_abs_logit_change']:.5f}. DAC는상위 {diagnostics['feature_replacement_summary']['DAC']['mean_important_abs_logit_change']:.5f},random {diagnostics['feature_replacement_summary']['DAC']['mean_random_abs_logit_change']:.5f}. 모든지표에서상위attention이더유용하다는일관된근거없음.",
        "- Featuregradient는유한하며reader가학습되고patch정보에기능적으로의존한다. 이사실만으로차량/보행자/도로를목적별로이해했다고주장하지않는다. Semantic정답감독·객체대응평가를이번에추가하지않았다.","",
        "## 검증과 계산 비용","",
        "CPU6검사통과: 초기/OFF원출력,나머지4지표불변,같은파라미터수/sharedattention동일,NaNpadding제외,실제lossgradient,feature/metric/candidate기능개입. 실제캐시2step에서head첫gradient와reader후속gradientpositive/finite;cacheinputs/nativeweightsgradient없음. 원modelcacheforwardscene/경로오차0,score9.54e-7,원tensorhash불변. CacheFP16의상대L2오차최대.00020881(0.0209%).",
        "", "학습코드의모델에는encoder나원scorer모듈이없고캐시만읽는다. Profile가중치는본학습에재사용하지않았다. 첫head0초기화때reader첫gradient0은의도된현상이며첫head업데이트후실제gradient를확인했다.","",
        "학습 함수는 deterministic algorithms를 활성화한다. results.json의 environment는 별도 report 프로세스의 스냅샷이며 해당 값은 false다. 학습 시점의 플래그를 별도로 기록하거나 bitwise 재실행 일치를 검증하지 않았으므로, 완전한 재현성을 입증했다고 주장하지 않는다.","",
        f"- Spatialcache 추출GPU0 loop {results['spatial_cache']['gpu_loop_seconds']:.2f}초,peakallocated {results['spatial_cache']['peak_allocated_gpu_bytes']/1024**3:.3f}GiB. 모델초기화전체wall은이loop에포함하지않는다.",
        f"- 두조건실제GPU training step합계 {sum(value['gpu_training_step_seconds'] for value in results['training']['conditions'].values()):.2f}초. Cache추출·profiling·tuning·평가/진단비용은별도로기록. 모든GPU작업은물리0번만사용.",
        f"- Cached adapter GPUforward/sample(batch8)은Shared {results['adapter_latency']['shared']['cached_adapter_gpu_forward_ms_per_scene_batched8']:.3f}ms,Specific {results['adapter_latency']['metric_specific']['cached_adapter_gpu_forward_ms_per_scene_batched8']:.3f}ms. 단일짧은실측/병행GPU실행이며batch/fileIO/nativeencoder/scorer/deployment총시간미포함;두조건속도차를구조적우위로확정하지않는다.",
        f"- 새작업중cache/profile/train/report에서관측한전체GPU0메모리최대 {max(results['spatial_cache']['maximum_whole_card_bytes'],results['profile']['maximum_whole_card_bytes'],*[value['maximum_whole_card_bytes'] for value in results['training']['conditions'].values()] ,results['maximum_whole_card_bytes'])/1e9:.3f}decimalGB<48GB. 기존LPWM등록289파일불변,기존학습계속.",
        "- 공식라벨은이전검증된evaluator의native64만재사용:이번추가공식평가0개. 원라벨생성비용은이전pilot보고서참조. 새보정기의추론순위로고정후보의실제공식score를집계했다.","",
        "## 확인한 사실과 추정 원인","",
        "확인:loss감소/readergradient/attention변화는있었으나개발PDMS·ranking·NC/DACcalibration개선은없다. Uniform/swap개입도최종선택을바꾸지않았다. Attention이NC차량/DAC도로로뚜렷하게특화됐다는증거가없다.",
        "", "미확정원인:보정MLP가기존candidatefeature/ego/path를이용하는우회경로,scalar NC/DAC감독의공간식별성부족,additive metricembedding의제한적상호작용,작은학습량/약한원scorer/개발분포shift. 특히현재query는metric과candidate의강한비선형교차작용을보장하지않는다. 이들중하나를실패원인으로확정하거나frozenvisualfeature가무용하다고확대해석하지않는다.","",
        "## 다음 한 실험 — 미실행","",
        "기존scored feature와원logit을입력에서제외한작은NC/DACprobe를같은frozenpatch+ego+candidate에서학습해,공간feature가해당지표예측정보를실제로제공하는지먼저검증한다. Semantic라벨/새worldmodel/fullscale을추가하기전에직접읽기분기와기존표현우회경로의효과를분리한다. 이번작업에서추가학습을자동기동하지않았다.","",
        "## 시각화·재실행","",
        "- `nc_dac_attention_overlays.png`: 고정첫장면/명령별원영상,NC전후,DAC,빨강·파랑겹침. 평가Specific100step이다.",
        "- `nc_dac_attention_density.png`: 원가중치density/token그래프. Uniformdensity=1,패널전체같은scale.",
        "- `nc_dac_attention_training_evolution.png`: 같은이미지/후보의초기→100→500step. 마지막500은표현진단이며공식비교의선택checkpoint가아니다.",
        "- 전후시각화는현재frame의실제attention만원crop/resize/pad역매핑했다. RGBdecoder생성영상이아니다. 빨강/파랑은읽기비중이고segmentation/위험GT가아니다.","",
        "```bash","cd /rhome/junseong/PlanningAwareFuturePrediction","bash scripts/run_metric_visual_evidence_pilot.sh",
        "CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 PYTHONPATH=src runtime/environments/kjs-scorer-selection-pilot/bin/python scripts/visualize_metric_evidence_training_steps.py --configuration configs/metric_visual_evidence/pilot_v1.json",
        "CUDA_VISIBLE_DEVICES='' PYTHONPATH=src runtime/environments/kjs-scorer-selection-pilot/bin/python scripts/package_metric_visual_evidence_results.py --configuration configs/metric_visual_evidence/pilot_v1.json","```","",
        "완료단계는skip하며새설정은새output에서등록한다. Config/source/입력hash·checkpoint·environment·검증·JSON/CSV/NPZ는같은폴더에있다. 단일seed/기존개발NAVVAL/약한원baseline이라논문성능이나독립finaltest우위를주장하지않는다."])
    (result/"report.md").write_text("\n".join(lines)+"\n")
    write_json(output/"complete.json",{"complete":True,"replication_gate_passed":results["replication_gate_passed"],
        "new_training_started_after_pilot":False,"results_directory":str(result),"protected_training":protection})


if __name__=="__main__":
    parser=argparse.ArgumentParser();parser.add_argument("--configuration",type=Path,required=True)
    main(parser.parse_args().configuration)
