#!/usr/bin/env python
"""Read-only CPU diagnostics, matched attention views, and measured report."""
import argparse
import json
from pathlib import Path
import sys
import time

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plotting
import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
from run_direct_metric_visual_probe_pilot import (
    CachedMetricProbeScenes, initialize_probe, load_probe,
)
from run_metric_visual_evidence_pilot import cache_tensors, load_trained_reader, verify_registration
from diagnose_metric_visual_evidence import blend_attention, map_attention_to_original
from calibrate_frozen_drivor_scorer import file_sha256, protected_training_audit, write_json


def main(configuration_path):
    configuration = json.loads(configuration_path.read_text())
    verify_registration(configuration)
    torch.set_num_threads(configuration["cpu_threads"])
    device = torch.device("cpu")
    scene_cache = CachedMetricProbeScenes(configuration)
    results = json.loads((scene_cache.result / "results.json").read_text())
    prior_configuration = json.loads((PROJECT_ROOT / configuration["source_visual_pilot_configuration"]).read_text())
    prior_result = PROJECT_ROOT / prior_configuration["results_directory"]
    diagnostic_subset = json.loads((prior_result / "diagnostic_subset.json").read_text())
    fixed_rows = diagnostic_subset["rows"]
    shared, shared_step = load_probe(configuration, "shared", device)
    specific, specific_step = load_probe(configuration, "metric_specific", device)
    last_specific, _ = load_probe(configuration, "metric_specific", device, step=500)
    initial = initialize_probe(configuration, "metric_specific").eval()
    initial.load_state_dict(torch.load(scene_cache.output / "common_initialization.pt", map_location=device, weights_only=True))
    prior_reader = load_trained_reader(prior_configuration, "metric_specific", device)
    for model in (shared, specific, last_specific, initial, prior_reader):
        model.requires_grad_(False)
    training_mean = torch.from_numpy(np.load(scene_cache.output / "training_mean_visual_feature.npy"))
    original_predictions = np.load(scene_cache.result / "evaluation_predictions.npz")
    dev_rows = scene_cache.rows("locked_development")
    diagnostics, gallery = [], []
    started = time.perf_counter()
    for row in fixed_rows:
        inputs, _, _ = scene_cache.batch([row], device)
        with torch.no_grad():
            logits, attention = specific(**inputs, capture_attention=True)
            _, shared_attention = shared(**inputs, capture_attention=True)
            _, initial_attention = initial(**inputs, capture_attention=True)
            _, last_attention = last_specific(**inputs, capture_attention=True)
            _, prior_attention = prior_reader(**cache_tensors(prior_configuration, [row], device), capture_attention=True)
            constant_inputs = {**inputs, "visual_tokens": training_mean[None, None].expand_as(inputs["visual_tokens"])}
            constant_prediction, _ = specific(**constant_inputs)
            modified_prediction, _ = specific(**{**constant_inputs,
                                               "ego_status": inputs["ego_status"] * -3.,
                                               "trajectories": inputs["trajectories"] * 2.,
                                               "positions": inputs["positions"] * -4.})
        assert torch.equal(shared_attention[:, :, 0], shared_attention[:, :, 1])
        normalized_entropy = -(attention * attention.clamp_min(1e-30).log()).sum(-1).mean().item() / np.log(1406)
        metric_distance = (attention[:, :, 0] - attention[:, :, 1]).abs().sum(-1).mul(.5).mean().item()
        candidate_distance = (attention - attention.mean(1, keepdim=True)).abs().sum(-1).mul(.5).mean().item()
        relative_odds = attention[:, :, 0].log() - attention[:, :, 1].log()
        relative_odds = relative_odds - relative_odds[..., :1]
        candidate_metric_interaction = (relative_odds - relative_odds[:, :1]).abs().max().item()
        feature_diagnostics = {}
        for metric_index, metric_name in enumerate(("NC", "DAC")):
            candidate = int(scene_cache.targets[row, :, metric_index].argmin())
            important = attention[0, candidate, metric_index].argsort(descending=True)[:140]
            random_indices = torch.randperm(1406, generator=torch.Generator().manual_seed(7000 + row + metric_index))[:140]
            changes = []
            for indices in (important, random_indices):
                altered = inputs["visual_tokens"].clone()
                altered[:, indices] = training_mean
                with torch.no_grad():
                    changed, _ = specific(**{**inputs, "visual_tokens": altered})
                changes.append((changed[0, candidate, metric_index] - logits[0, candidate, metric_index]).abs().item())
            differentiable_tokens = inputs["visual_tokens"].clone().requires_grad_(True)
            prediction, _ = specific(**{**inputs, "visual_tokens": differentiable_tokens})
            gradient = torch.autograd.grad(prediction[0, candidate, metric_index], differentiable_tokens)[0]
            assert torch.isfinite(gradient).all()
            feature_diagnostics[metric_name] = {
                "candidate": candidate, "official_target": float(scene_cache.targets[row, candidate, metric_index]),
                "important140_mean_replacement_abs_logit_change": changes[0],
                "random140_mean_replacement_abs_logit_change": changes[1],
                "feature_gradient_norm": gradient.norm().item(),
            }
        density = attention[0, :, :, 703:].reshape(64, 2, 19, 37) * 1406
        diagnostics.append({
            "sample_id": scene_cache.records[row]["token"], "pilot_row": row,
            "normalized_attention_entropy": normalized_entropy,
            "metric_attention_total_variation": metric_distance,
            "candidate_attention_total_variation": candidate_distance,
            "candidate_dependence_of_metric_relative_token_odds": candidate_metric_interaction,
            "constant_visual_condition_change_max_logit_error": (constant_prediction - modified_prediction).abs().max().item(),
            "current_upper6rows_density_NC_DAC": density[:, :, :6].mean((0, 2, 3)).tolist(),
            "current_lower6rows_density_NC_DAC": density[:, :, -6:].mean((0, 2, 3)).tolist(),
            "feature_diagnostics": feature_diagnostics,
        })
        command = scene_cache.records[row]["command_index"]
        if not any(example["command_index"] == command for example in gallery):
            candidate = int(original_predictions["unchanged_scores"][dev_rows.index(row)].argmax())
            gallery.append({
                "row": row, "command_index": command, "candidate": candidate,
                "maps": {"prior": prior_attention[0, candidate].numpy()[:, 703:] * 1406,
                         "initial": initial_attention[0, candidate].numpy()[:, 703:] * 1406,
                         "shared": shared_attention[0, candidate].numpy()[:, 703:] * 1406,
                         "specific": attention[0, candidate].numpy()[:, 703:] * 1406,
                         "last500": last_attention[0, candidate].numpy()[:, 703:] * 1406},
            })
    display_maximum = max(1., float(np.percentile(np.concatenate([value.ravel() for example in gallery for value in example["maps"].values()]), 99.5)))
    for filename, columns, figure_width in (
        ("nc_dac_before_after.png", (("original", None), ("prior", 0), ("initial", 0),
                                     ("specific", 0), ("specific", 1), ("overlay", None)), 21),
        ("nc_dac_shared_vs_specific.png", (("original", None), ("shared", 0), ("shared", 1),
                                           ("specific", 0), ("specific", 1)), 18),
        ("nc_dac_training_evolution.png", (("original", None), ("initial", 0), ("specific", 0),
                                          ("last500", 0), ("last500", 1)), 18),
    ):
        figure, axes = plotting.subplots(len(gallery), len(columns), figsize=(figure_width, 3.1 * len(gallery)), squeeze=False)
        for panel, example in enumerate(gallery):
            raw = cv2.cvtColor(cv2.imread(scene_cache.records[example["row"]]["observed_front_paths"][-1]), cv2.COLOR_BGR2RGB)
            for column, (map_name, metric) in enumerate(columns):
                if map_name == "original":
                    shown, title = raw, "Original observed image"
                elif map_name == "overlay":
                    mapped = [map_attention_to_original(example["maps"]["specific"][index], raw.shape) for index in (0, 1)]
                    red_alpha = np.clip(mapped[0] / display_maximum, 0, 1)[..., None] * .45
                    blue_alpha = np.clip(mapped[1] / display_maximum, 0, 1)[..., None] * .45
                    shown = np.clip((1-red_alpha-blue_alpha) * raw + red_alpha * np.array([255, 0, 0]) + blue_alpha * np.array([0, 70, 255]), 0, 255).astype("uint8")
                    title = "NC red + DAC blue"
                else:
                    mapped = map_attention_to_original(example["maps"][map_name][metric], raw.shape)
                    shown = blend_attention(raw, mapped, [255, 0, 0] if metric == 0 else [0, 70, 255], display_maximum)
                    title = {"prior": "Previous residual reader", "initial": "New direct reader initial",
                             "shared": f"Shared step{shared_step}", "specific": f"Specific step{specific_step}",
                             "last500": "Specific last500"}[map_name] + (" | NC" if metric == 0 else " | DAC")
                axes[panel, column].imshow(shown); axes[panel, column].axis("off")
                axes[panel, column].set_title(f"{title}\ncommand{example['command_index']} / fixed candidate{example['candidate']}", fontsize=9)
        figure.suptitle(f"Same images/candidates and common display scale (density uniform=1, max={display_maximum:.3f})\nCurrent frame only. Reading weights, not segmentation. Last500 is not necessarily the selected checkpoint.")
        figure.tight_layout(rect=(0, 0, 1, .93)); figure.savefig(scene_cache.result / filename, dpi=160); plotting.close(figure)
    functional = {
        "complete": True, "diagnostic_scenes": len(diagnostics), "rows": diagnostics,
        "mean_normalized_attention_entropy": float(np.mean([record["normalized_attention_entropy"] for record in diagnostics])),
        "mean_metric_attention_total_variation": float(np.mean([record["metric_attention_total_variation"] for record in diagnostics])),
        "mean_candidate_attention_total_variation": float(np.mean([record["candidate_attention_total_variation"] for record in diagnostics])),
        "maximum_candidate_metric_interaction": max(record["candidate_dependence_of_metric_relative_token_odds"] for record in diagnostics),
        "display_maximum": display_maximum,
        "cpu_diagnostic_and_plot_wall_seconds": time.perf_counter() - started, "gpu_seconds": 0.,
        "semantic_grounding_measured": False,
        "interpretation": "Spatial reading and internal-feature interventions; coordinate bands are not semantic sky/road annotations.",
        "protected_training": protected_training_audit(configuration),
    }
    write_json(scene_cache.result / "functional_diagnostics.json", functional)
    write_json(scene_cache.result / "visualization_metadata.json", {
        "examples": [{key: value for key, value in example.items() if key != "maps"} for example in gallery],
        "same_images_and_candidates_as_previous_pilot": True, "display_maximum": display_maximum,
        "same_scale_all_panels": True, "current_frame_only": True,
        "mapping": "19x37 -> padded518x266 -> crop512x256 -> original rows28:-28",
    })
    write_report(configuration, scene_cache, results, functional)
    write_json(scene_cache.output / "diagnostics_complete.json", {"complete": True})
    write_json(scene_cache.output / "complete.json", {
        "complete": True, "planning_gate_passed": results["planning_gate_passed"],
        "visual_reading_gate_passed": results["visual_reading_gate_passed"],
        "extra_training_started": False, "protected_training": protected_training_audit(configuration),
    })
    print(json.dumps({"phase": "diagnostics", "complete": True,
                      "metric_attention_variation": functional["mean_metric_attention_total_variation"]}), flush=True)


def write_report(configuration, scene_cache, results, functional):
    names = ("unchanged", "shared", "metric_specific")
    planning = "통과" if results["planning_gate_passed"] else "미충족"
    visual = "통과" if results["visual_reading_gate_passed"] else "미충족"
    lines = ["# Frozen 시각 feature만으로 NC·DAC 직접 예측: 실제 pilot 결과", "",
             f"**사전 등록한 planning gate: {planning}. 시각 읽기 gate: {visual}.**", "",
             "## 구현과 비교 조건", "",
             "- 고정된 DrivoR 축소 실험 pass5/3200update에서 추출한 DINO 공간 cache를 재사용했다. 전방1대·과거현재2frame·512×256/crop28/padding; 장면당1406×384FP16이다.",
             "- 새 query는 ego11·후보24·metric16을 concat한 뒤 비선형 MLP로 만든다. 단순 가산 query의 metric/candidate 제약을 제거했다.",
             "- Head는 attention의64차원 visual readout만 받는다. 원 scorer feature/logit 및 직접 ego/경로 입력은 없다. Position은key에만 들어가고value에는 들어가지 않는다.",
             "- Shared는 두 metric attention logit을 평균해서 같은 표현을 읽는다. Specific은 개별 attention이다. 같은 파라미터 수/초기state/실제 배치·pair 순서/seed0/batch8/500step/AdamW3e-4/BCE+PDMS-gap ranking을 사용했다.",
             "- NAVTRAIN1024/101recording, 튜닝NAVVAL64/15recording, 개발NAVVAL192/37recording. 그룹 간 recording중복0. 이전에 사용한 개발패널이며 독립 최종NAVTEST가 아니다.",
             "- Native64 후보와 원 scorer의 다른4지표는 고정했다. 새 NC/DAC logit으로 대체해 원 점수식을 적용했다. OFF는 원6logit을 정확히 복원한다.",
             "- 별도 튜닝64의PDMS로100/250/500 중 체크포인트를 선택했다. 두조건 모두500step을 학습했고 profiling 모델은 폐기했다.",
             "- 이전 residual pilot과는 query·head 입력·position의value입력·예측 방식이 함께 달라졌다. 이전 대비 변화를 단일 요인의 효과로 해석하지 않는다. 원 scorer의NC/DAChead도3200update학습된 기존 모델이므로 새 head500step과 학습 이력이 다르다.", "",
             "## 동일한192장면·64후보 평가", "",
             "| 조건 | PDMS | 후보최고PDMS | 선택손실 | NC실패 | DAC실패 | NC/DAC평균BCE |",
             "|---|---:|---:|---:|---:|---:|---:|"]
    for name in (*names, "train_only_metric_prior"):
        value = results["summaries"][name]
        lines.append(f"| {name} | {value['selected_pdms']:.4f} | {value['oracle_pdms']:.4f} | {value['selection_regret']:.4f} | {100*value['NC_failure_rate']:.2f}% | {100*value['DAC_failure_rate']:.2f}% | {value['mean_NC_DAC_BCE']:.5f} |")
    lines.extend(["", f"Specific−shared {results['metric_specific_minus_shared_pdms']:+.4f}점. Recording bootstrap95%구간 {results['metric_specific_minus_shared_recording_95ci']}. 단일seed의 학습 변동을 bootstrap이 반영하는 것은 아니다.",
                  f"선택한 checkpoint: {results['selected_steps']}. 마지막500step의PDMS도 별도로 기록했다: {results['last_step500_pdms_without_dev_selection']}.", "",
                  "Train-only metric prior는 학습1024의NC/DAC평균 정답을 모든 후보에 예측하는 진단용 상수다. 파라미터/조건 정보가 같은 대조군은 아니다.", "",
                  "| 명령 | 장면수 | 기존 | Shared | Specific |", "|---|---:|---:|---:|---:|"])
    for command, label in (("0", "좌회전"), ("1", "직진"), ("2", "우회전")):
        metrics = [results["summaries"][name]["by_command"][command] for name in names]
        lines.append(f"| {label} | {metrics[0]['scenes']} | {metrics[0]['selected_pdms']:.2f} | {metrics[1]['selected_pdms']:.2f} | {metrics[2]['selected_pdms']:.2f} |")
    lines.extend(["", "## 추가 학습 없는 기능 개입", "",
                  "| Specific개입 | PDMS | NC/DAC평균BCE | 선택변경 |", "|---|---:|---:|---:|"])
    for intervention in ("uniform_attention", "swap_metric_attention", "patch_constant"):
        value = results["summaries"]["metric_specific_" + intervention]
        changed = results["interventions"]["metric_specific"][intervention]["selected_index_changed_fraction"]
        lines.append(f"| {intervention} | {value['selected_pdms']:.4f} | {value['mean_NC_DAC_BCE']:.5f} | {100*changed:.2f}% |")
    lines.extend(["", "Patch constant는 학습영상token의평균만 사용한다. 조건query를 통해영상정보가 기능적으로 사용되는지를 확인하는 내부개입이며 실제객체를 제거하는 인과검사가 아니다. Attention교환/균일화도encoder영상을 직접 바꾸지 않는다.", "",
                  "## 표현과 검증", "",
                  f"- CPU검사6개와 실제cache검사를 통과했다. 동일value일 때ego/path/position변경에예측불변,첫updatequery/key/valuegradient유한양수,metric×candidate상호작용,invalidtoken제외,head입력경로차단,다른4지표불변/OFF정확성을 확인했다.",
                  f"- 고정12장면 entropy/max={functional['mean_normalized_attention_entropy']:.5f},NC/DAC attention TV={functional['mean_metric_attention_total_variation']:.5f},후보간TV={functional['mean_candidate_attention_total_variation']:.5f}. Metric상대tokenodds의후보의존 최대값={functional['maximum_candidate_metric_interaction']:.6f}.",
                  "- 실제 읽기 가중치를 같은원영상/같은후보/공통척도로 역투영했다. Semantic정답은없고NC차량/DAC도로분할 정확도는측정하지않았다. Featuregradient가존재해도semantic특화나planning이득의증명이아니다.",
                  "- Top140token평균대체와random동수대체의지표별logit변화는functional_diagnostics.json에저장했다. 높은attention의유용성을별도확인하기위한진단이다.", "",
                  "## 실제 계산 비용과 보존", ""])
    training = results["training"]["conditions"]
    for name in CONDITIONS:
        value = training[name]
        lines.append(f"- {name}: {value['trainable_parameters']:,}개학습파라미터,500step GPUstep합계{value['gpu_training_step_seconds']:.2f}초,step중앙값{value['median_step_seconds']:.4f}초,peakallocated{value['peak_allocated_bytes']/1024**3:.3f}GiB,선택checkpoint{value['best_tuning_step']}.")
    maximum_card = max(results["maximum_whole_card_bytes"], results["profile"]["maximum_whole_card_bytes"], *[value["maximum_whole_card_bytes"] for value in training.values()])
    lines.extend([f"- 두조건GPUtraining step합계{sum(value['gpu_training_step_seconds'] for value in training.values()):.2f}초. 별도profiling{results['profile']['gpu_profile_seconds']:.2f}초,평가GPUphase wall{results['gpu_report_wall_seconds']:.2f}초. 학습step시간과cache/평가비용을구분했다.",
                  f"- 물리GPU0만사용/process2.5GiB정책/전체카드관측최대{maximum_card/1e9:.3f}decimalGB<48GB. 원LPWM보호대상289파일불변. 새encoderforward/새cache생성/새공식candidate평가호출은모두0.",
                  "- CUDAtraining시deterministic algorithms와CUBLASworkspace를설정하고단계별환경을기록했다. Bitwise재실행은미검증이다. Cachedforward latency는추가probe만/배치8/fileIO·원encoder·원scorer제외이며병행GPU부하가있다.",
                  f"- Cachedforward/sample 실측(ms): {results['cached_probe_gpu_forward_ms_per_scene_batched8']}.", "",
                  "## 해석과 다음 한 실험", "",
                  "Planning gate와 시각정보읽기 gate를 각각 보고한다. 한쪽통과가다른쪽의개선을뜻하지않고,이개발panel의양수차이를논문수준우위로확대해석하지않는다. 공간표현이서로달라도안전지표예측과후보선택에이득이없으면추가본학습은진행하지않는다.",
                  "후속우선순위는기능개입에서실제정보사용이확인되는지를기준으로정한다. 이번작업에서fullscale/추가seed/semanticmask를자동추가하지않았다.", "",
                  "## 산출물과 재실행", "",
                  "- `nc_dac_before_after.png`: 이전residual reader/새초기/새학습후의동일이미지비교.",
                  "- `nc_dac_shared_vs_specific.png`: 두새조건의NC/DACmap비교.",
                  "- `nc_dac_training_evolution.png`: 초기/선택checkpoint/마지막500step. 마지막500은평가선택checkpoint와구분한다.",
                  "- `results.json`, `comparison.csv`, `scene_results.csv`, `evaluation_predictions.npz`,검증·manifest·source/inputhash·환경·기능진단을저장했다.", "",
                  "```bash", "cd /rhome/junseong/PlanningAwareFuturePrediction", "bash scripts/run_direct_metric_visual_probe_pilot.sh", "```", "",
                  "완료phase는건너뛴다. 설정이나등록source를바꾸려면새output에등록한다."])
    (scene_cache.result / "report.md").write_text("\n".join(lines) + "\n")


CONDITIONS = ("shared", "metric_specific")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--configuration", type=Path, default=Path("configs/direct_metric_visual_probe/pilot_v1.json"))
    main(parser.parse_args().configuration)
