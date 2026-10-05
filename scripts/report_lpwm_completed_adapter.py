"""Report three completed LPWM adaptation runs using saved development rows."""
import hashlib
import json
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from report_lpwm_completed_lora import PROJECT_ROOT, RUN_DIRECTORIES, read_json
from summarize_lpwm_posttraining import metric_means, paired_recording_interval


REPORT_DIRECTORY = PROJECT_ROOT / "results/lpwm_card_budget_measured_v4/completed_adapter_review_20261005"


def main():
    directories = {**RUN_DIRECTORIES, "residual_adapter": PROJECT_ROOT /
        "outputs/lpwm_card_budget_measured_v4/residual_adapter/batch8/metric_plus_world"}
    planning_rows, world_rows, initial_rows, summaries, costs, source_hashes = {}, {}, {}, {}, {}, {}
    for method, directory in directories.items():
        for filename in ("trained.json", "world.json", "initial.json", "summary.json"):
            path = directory / "trend_evaluation" / filename
            source_hashes[str(path.relative_to(PROJECT_ROOT))] = hashlib.sha256(path.read_bytes()).hexdigest()
        planning_rows[method] = read_json(directory / "trend_evaluation/trained.json")
        world_rows[method] = read_json(directory / "trend_evaluation/world.json")
        initial_rows[method] = read_json(directory / "trend_evaluation/initial.json")
        summaries[method] = read_json(directory / "trend_evaluation/summary.json")
        training = read_json(directory / "training_summary.json")
        inventory = read_json(directory / "parameter_inventory.json")
        assert training["completed_updates"] == 4707 and not training["profile_only"]
        assert not summaries[method]["engineering_only"]
        assert training["checkpoint_sha256"] == summaries[method]["checkpoint_sha256"]
        costs[method] = {key: training[key] for key in ("seconds", "peak_allocated_gib", "completed_updates")}
        costs[method].update({key: inventory[key] for key in (
            "trainable_parameters", "trainable_world_parameters", "trainable_planner_parameters")})
        costs[method]["evaluation_seconds"] = summaries[method]["seconds"]
    adapter = "residual_adapter"
    comparisons = {}
    for reference in ("partial_output_layers", "attention_lora"):
        for collection in (planning_rows, world_rows, initial_rows):
            first = [(row["token"], row["recording_group"]) for row in collection[adapter]]
            second = [(row["token"], row["recording_group"]) for row in collection[reference]]
            assert first == second and len({token for token, _ in first}) == len(first)
        comparisons["adapter_minus_" + reference] = {
            "planning": {metric: paired_recording_interval(planning_rows[adapter], planning_rows[reference], metric)
                for metric in ("pdms", "ade_meters", "fde_meters", "metric_bce")},
            "world": {metric: paired_recording_interval(world_rows[adapter], world_rows[reference], metric)
                for metric in ("reconstruction_lpips", "forecast_lpips")},
            "initial_candidate_mismatch_count": sum(first["candidate_index"] != second["candidate_index"]
                for first, second in zip(initial_rows[adapter], initial_rows[reference])),
            "trained_candidate_mismatch_count": sum(first["candidate_index"] != second["candidate_index"]
                for first, second in zip(planning_rows[adapter], planning_rows[reference])),
        }
    stage1 = read_json(PROJECT_ROOT / "outputs/lpwm_navsim_full_posttraining_v2/evaluation/posttrained/metrics.json")
    stage1_by_token = {row["token"]: row for row in stage1["records"]}
    stage1_panel = [stage1_by_token[row["token"]] for row in world_rows[adapter]]
    world_means = {"stage1": metric_means(stage1_panel), **{
        method: metric_means(rows) for method, rows in world_rows.items()}}
    scenarios = {}
    for scenario in sorted({row["scenario"] for row in stage1_panel}):
        selected = [row for row in world_rows[adapter] if stage1_by_token[row["token"]]["scenario"] == scenario]
        selected_tokens = {row["token"] for row in selected}
        scenarios[scenario] = {"clips": len(selected),
            "adapter_forecast_lpips": metric_means(selected)["forecast_lpips"],
            "stage1_forecast_lpips": metric_means([row for row in stage1_panel if row["token"] in selected_tokens])["forecast_lpips"],
            "adapter_minus_stage1": paired_recording_interval(selected, stage1_panel, "forecast_lpips"),
            "adapter_minus_lora": paired_recording_interval(selected, world_rows["attention_lora"], "forecast_lpips")}
    report = {
        "created_at": datetime.now(ZoneInfo("Asia/Seoul")).isoformat(),
        "scope": "Same1024 navtrain internal-development scenes,1021 valid cached official PDM scores,40 recordings;256 world clips. Not official navtest.",
        "training": {"scenes": 75297, "epochs": 1, "updates": 4707, "seed": 47, "direct_object_gt_auxiliary": False},
        "planning_means": {method: metric_means(rows) for method, rows in planning_rows.items()},
        "world_means": {method: {key: values[key] for key in ("reconstruction_lpips", "forecast_lpips")}
            for method, values in world_means.items()},
        "world_relative_change_percent_vs_stage1": {method: {
            metric: (world_means[method][metric] / world_means["stage1"][metric] - 1) * 100
            for metric in ("reconstruction_lpips", "forecast_lpips")} for method in directories},
        "paired_comparisons": comparisons,
        "interval_method": "2000 paired recording-cluster bootstrap resamples, seed20261003; sample-weighted means; not training-seed uncertainty.",
        "adapter_predicted_minus_persistent_future": summaries[adapter]["predicted_minus_persistent_future"],
        "adapter_predicted_and_persistent_means": {key: summaries[adapter]["means"][key] for key in ("trained", "persistent_future")},
        "checks_by_method": {method: summary["checks"] for method, summary in summaries.items()},
        "adapter_world_vs_stage1": summaries[adapter]["world_vs_stage1"],
        "adapter_world_risk_breakdown": summaries[adapter]["world_risk_breakdown"],
        "world_scenario_breakdown": scenarios,
        "scenario_scope": "Previously registered Stage1 labels, whole-image LPIPS; projected overlap is an occlusion proxy, not confirmed occlusion or object-state accuracy. Exploratory intervals without multiple-comparison correction.",
        "measured_costs": costs,
        "missing_pdms_tokens": [row["token"] for row in planning_rows[adapter] if row["metrics"]["pdms"] is None],
        "missing_pdms_reason": "Three scenes lack fully finite cached candidate metric labels; same exclusions across compared methods, trajectory errors remain available.",
        "source_sha256": source_hashes,
        "limitations": [
            "One seed and one epoch; no convergence or independent-test claim.",
            "No separately trained frozen-LPWM planner control; adaptation benefit is not isolated from planner learning.",
            "Adaptation locations, capacity and historical microbatch schedules differ.",
            "Persistent-future replacement is an inference intervention, not a separately trained no-future baseline.",
            "World-video LPIPS does not directly validate object-state retention or planning-aware particle specialization.",
            "Different shared GPU loads and batch histories preclude attributing elapsed-time differences to adaptation method alone.",
        ],
        "runtime_changes": False,
    }
    REPORT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    (REPORT_DIRECTORY / "summary.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    figure, axes = plt.subplots(2, 2, figsize=(12, 8))
    labels = ["Partial layers", "LoRA", "Adapter"]
    colors = ["#d7953b", "#287aa6", "#368568"]
    for axis, metric, title, scale, upper in (
        (axes[0, 0], "pdms", "Development PDMS (higher is better)", 100, 100),
        (axes[0, 1], "ade_meters", "Average trajectory error (lower is better)", 1, 1.55),
    ):
        values = [report["planning_means"][method][metric] * scale for method in directories]
        bars = axis.bar(labels, values, color=colors, width=.55)
        axis.bar_label(bars, labels=[f"{value:.3f}" for value in values], padding=5)
        axis.set_ylim(0, upper)
        axis.set_title(title, pad=13)
        axis.grid(axis="y", alpha=.15)
        axis.set_axisbelow(True)
    axis = axes[1, 0]
    values = [world_means[method]["forecast_lpips"] for method in ("stage1", *directories)]
    bars = axis.bar(["Stage1", *labels], values, color=["#a6adb2", *colors], width=.55)
    axis.bar_label(bars, labels=[f"{value:.4f}" for value in values], padding=5)
    axis.set_ylim(0, .52)
    axis.set_title("Future-video LPIPS (lower is better)", pad=13)
    axis.set_xlabel("Same 256 clips; Stage1 before planning adaptation")
    axis.grid(axis="y", alpha=.15)
    axis.set_axisbelow(True)
    axis = axes[1, 1]
    intervals = [comparisons["adapter_minus_" + method]["planning"]["pdms"]
        for method in ("partial_output_layers", "attention_lora")]
    centers = np.array([item["mean_difference"] * 100 for item in intervals])
    lows = np.array([item["ci95"][0] * 100 for item in intervals])
    highs = np.array([item["ci95"][1] * 100 for item in intervals])
    axis.errorbar(centers, [1, 0], xerr=np.stack((centers-lows, highs-centers)), fmt="o", capsize=5, color=colors[-1])
    axis.axvline(0, color="#777777", linewidth=1, linestyle="--")
    axis.set_yticks([1, 0], ["Adapter - Partial", "Adapter - LoRA"])
    axis.set_ylim(-.7, 1.7)
    axis.set_title("PDMS differences and paired 95% intervals", pad=13)
    axis.set_xlabel("PDMS points; recording-cluster bootstrap")
    figure.suptitle("Completed LPWM adaptation: Adapter development results", fontsize=17, y=.98)
    figure.text(.5, .935, "One epoch / one seed | 1,024 scenes (1,021 valid PDMS) | NOT official navtest", ha="center", color="#555555")
    figure.tight_layout(rect=(0, 0, 1, .91), h_pad=3, w_pad=3)
    figure.savefig(REPORT_DIRECTORY / "comparison.png", dpi=170)
    figure.savefig(REPORT_DIRECTORY / "comparison.pdf")
    plt.close(figure)
    print(json.dumps({key: report[key] for key in ("planning_means", "world_means", "paired_comparisons", "world_scenario_breakdown")}, indent=2))


if __name__ == "__main__":
    main()
