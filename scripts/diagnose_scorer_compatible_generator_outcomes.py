#!/usr/bin/env python
"""Separate native candidate-coordinate changes from scorer index reselection."""
import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
from calibrate_frozen_drivor_scorer import write_json
from run_scorer_compatible_generator import native_head, inputs


def diagnose(configuration):
    import torch
    from planning_aware_future_prediction.models.scorer_compatible_generator import distribution_matching_losses
    started = time.perf_counter(); torch.set_num_threads(configuration["cpu_threads"])
    output, registration, cache = inputs(configuration); destination = PROJECT_ROOT / configuration["results_directory"]
    metadata = json.loads((output / "prediction_complete.json").read_text()); records = metadata["records"]
    arrays = np.load(output / "evaluation_candidates.npz"); metrics = np.load(output / "evaluation_metrics.npy")
    indices = np.asarray([index for index, record in enumerate(records) if record["pilot_split"] == "locked_development"])
    actual = metrics[..., -1] * 100; chosen = arrays["log_scores"].argmax(axis=-1)
    diagnosis = {}
    for condition, model_index in [("B_quality", 3), ("C_compatible", 4)]:
        original_quality = actual[0, indices, chosen[0, indices]]
        fixed_query_quality = actual[model_index, indices, chosen[0, indices]]
        final_quality = actual[model_index, indices, chosen[model_index, indices]]
        coordinate_change = fixed_query_quality - original_quality
        reselection_change = final_quality - fixed_query_quality
        total_change = final_quality - original_quality
        assert np.max(np.abs(total_change - coordinate_change - reselection_change)) < 1e-10
        worst_offset = int(total_change.argmin()); scene_index = int(indices[worst_offset]); record = records[scene_index]
        selected_index = int(chosen[model_index, scene_index]); original_selected_index = int(chosen[0, scene_index])
        selected_path_movement = np.linalg.norm(arrays["trajectories"][model_index, scene_index, selected_index, :, :2] - arrays["trajectories"][0, scene_index, selected_index, :, :2], axis=-1)
        old_unsafe = (metrics[0, indices, :, 0] < 1) | (metrics[0, indices, :, 1] < 1)
        current_unsafe = (metrics[model_index, indices, :, 0] < 1) | (metrics[model_index, indices, :, 1] < 1)
        diagnosis[condition] = {"mean_coordinate_change_under_original_query_selection_pdms": float(coordinate_change.mean()),
            "mean_reselection_change_on_modified_set_pdms": float(reselection_change.mean()), "mean_total_pdms_change": float(total_change.mean()),
            "decomposition_meanings": "Query IDs match because only the output head changes. This is algebraic bookkeeping, not causal identification.",
            "native_safe_queries_becoming_unsafe_candidate_count": int(np.sum(~old_unsafe & current_unsafe)),
            "native_unsafe_queries_becoming_safe_candidate_count": int(np.sum(old_unsafe & ~current_unsafe)),
            "worst_scene": {"token": record["token"], "command": record["command_name"], "original_selected_index": original_selected_index,
                "current_selected_index": selected_index, "selected_query_unchanged": original_selected_index == selected_index,
                "original_selected_pdms": float(original_quality[worst_offset]), "modified_same_query_pdms": float(fixed_query_quality[worst_offset]),
                "modified_final_pdms": float(final_quality[worst_offset]),
                "current_selected_dac_estimate": float(arrays["metric_probabilities"][model_index, scene_index, selected_index, 1]),
                "current_selected_actual_dac": float(metrics[model_index, scene_index, selected_index, 1]),
                "same_query_mean_waypoint_movement_m": float(selected_path_movement.mean()), "same_query_max_waypoint_movement_m": float(selected_path_movement.max())},
            "sum_other_scene_changes_excluding_worst_for_diagnosis_only": float(np.delete(total_change, worst_offset).sum()),
            "all_64_scenes_kept_in_primary_result": True}
    training_rows = [record["pilot_row"] for record in registration["records"] if record["pilot_split"] == "train"]
    teachers = np.load(output / "teacher_distributions.npz")
    quality = torch.from_numpy(teachers["quality"][training_rows])
    original_paths = torch.from_numpy(np.array(cache["native_trajectories"][training_rows])); tokens = torch.from_numpy(np.array(cache["trajectory_tokens"][training_rows]))
    training_matching = {}
    with torch.inference_mode():
        for condition, target_name in [("B_quality", "quality"), ("C_compatible", "compatible")]:
            weights = torch.from_numpy(teachers[target_name][training_rows]); checkpoint_losses = []
            for step in [0, 100, 300]:
                head = native_head(configuration).eval()
                if step:
                    head.load_state_dict(torch.load(output / condition / f"step{step}.pt", map_location="cpu", weights_only=True)["state_dict"], strict=True)
                generated = head(tokens).reshape(len(tokens), 64, 8, 3)
                losses = distribution_matching_losses(generated, original_paths, weights, quality, configuration["kernel_bandwidths_m"], configuration["heading_distance_scale_m"], configuration["coverage_temperature_m2"])
                checkpoint_losses.append({"step": step, **{name: float(value) for name, value in losses.items()}})
            training_matching[condition] = checkpoint_losses
    result = {"development_decomposition": diagnosis, "full_training_set_target_matching": training_matching,
        "optimizer_steps": 0, "encoder_calls": 0, "gpu_seconds": 0, "cpu_wall_seconds": time.perf_counter() - started}
    write_json(destination / "outcome_diagnosis.json", result)
    # Explicit precision check request for the road-edge failure: compare the
    # cached GPU baseline with an unchanged head replayed on CPU. This is not
    # another learned condition and does not change the primary measurements.
    worst_token = diagnosis["C_compatible"]["worst_scene"]["token"]
    precision_records = [record for record in registration["records"] if record["token"] == worst_token]
    assert len(precision_records) == 1
    record = precision_records[0]; row = record["pilot_row"]
    with torch.inference_mode():
        original_head = native_head(configuration).eval()
        cpu_replay = original_head(torch.from_numpy(np.array(cache["trajectory_tokens"][row:row+1]))).reshape(1, 64, 8, 3)[0].numpy()
    original_paths = np.array(cache["native_trajectories"][row]); original_scores = np.array(cache["native_log_scores"][row])
    np.savez(output / "baseline_precision_paths.npz", trajectories=np.concatenate((original_paths, cpu_replay)), log_scores=np.tile(original_scores, 2))
    write_json(output / "baseline_precision_request.json", {"record": record, "maximum_unchanged_head_gpu_cache_vs_cpu_error_m_or_rad": float(np.abs(original_paths - cpu_replay).max()), "original_selected_index": diagnosis["C_compatible"]["worst_scene"]["original_selected_index"]})
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    figure, axes = plt.subplots(1, 2, figsize=(13, 5))
    for position, (condition, color) in enumerate([( "B_quality", "#4c78a8"), ("C_compatible", "#d04467")]):
        row = diagnosis[condition]
        axes[0].bar(np.arange(3) + position*.3, [row["mean_coordinate_change_under_original_query_selection_pdms"], row["mean_reselection_change_on_modified_set_pdms"], row["mean_total_pdms_change"]], width=.28, color=color, label=condition)
        rows = training_matching[condition]; axes[1].plot([row["step"] for row in rows], [row["distribution_mmd"] for row in rows], marker="o", color=color, label=condition)
    axes[0].set_xticks(np.arange(3)+.15, ["Coordinates at original query", "Reselection on new set", "Total"]); axes[0].axhline(0, color="#444444", linewidth=1)
    axes[0].set_ylabel("Mean actual PDMS change from A"); axes[0].set_title("Development score change: exact algebraic decomposition")
    axes[1].set_xlabel("Optimizer step"); axes[1].set_ylabel("MMD squared to own training target"); axes[1].set_title("Fixed full training set, same teacher throughout")
    for axis in axes:
        axis.legend(fontsize=9); axis.grid(axis="y", alpha=.15)
    figure.suptitle("Stage 4: fitting the teacher does not guarantee safe new coordinates or better selection", fontsize=12)
    figure.tight_layout(rect=(0, 0, 1, .92)); figure.savefig(destination / "stage4_matching_vs_planning_diagnosis.png", dpi=145); plt.close(figure)
    lines = ["\n## Stage 4 — 손실 원인 분해(추가 학습 없음)", "", "| 조건 | 원query를 유지하고 좌표만 변경한 효과 | 변경된set에서 재선택 효과 | 총 변화 |", "|---|---:|---:|---:|"]
    for condition, row in diagnosis.items():
        lines.append(f"| {condition} | {row['mean_coordinate_change_under_original_query_selection_pdms']:+.4f} | {row['mean_reselection_change_on_modified_set_pdms']:+.4f} | {row['mean_total_pdms_change']:+.4f} |")
    worst = diagnosis["C_compatible"]["worst_scene"]
    lines += ["", "Query identity가유지되는같은head의미세조정이므로이분해가가능하다. 이는정확한점수차회계이며인과분리실험은아니다.",
        f"가장큰악화는{worst['token']}이다. A와C가모두query{worst['current_selected_index']}를선택했지만그경로의좌표가평균{worst['same_query_mean_waypoint_movement_m']:.3f}m/최대{worst['same_query_max_waypoint_movement_m']:.3f}m이동하면서실제PDMS100→0, DAC1→0이됐다. C의DAC예측은{worst['current_selected_dac_estimate']:.3f}여서scorer는도로이탈을충분히낮게평가하지못했다.",
        "따라서주요악화사례를'다른후보를잘못고름'만으로설명할수없다. 미세한좌표이동이안전경계를넘고고정scorer가그위험을놓쳤다. 이한장면을제거하면평균을다르게보이게할수있지만,전체64장면을주결과에그대로남겼다.",
        "[MMD 적합과 planning 불일치](stage4_matching_vs_planning_diagnosis.png). 고정train전체의MMD감소를실제측정했다. 새경로의oracle라벨을재확인해야하며,오래된nativecandidate teacher와의거리만으로정확한새분포가보장되지않는다.",
        "이결과를본뒤성공기준을변경하거나현재generator를추가학습하지않았다. 가장작은다음검토는학습용생성경로의officiallabel을한번갱신해target의stale-label문제가얼마나큰지확인하는것이다. 평가NAVTEST라벨로teacher를재구성하지않는다.", ""]
    with (destination / "report.md").open("a") as stream:
        stream.write("\n".join(lines))
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(); parser.add_argument("--configuration", type=Path, required=True)
    arguments = parser.parse_args(); diagnose(json.loads(arguments.configuration.read_text()))
