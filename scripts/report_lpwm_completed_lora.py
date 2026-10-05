"""Summarize completed LPWM adaptation results from saved rows, without inference."""
import hashlib
import json
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from summarize_lpwm_posttraining import metric_means, paired_recording_interval


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RUN_DIRECTORIES = {
    "partial_output_layers": PROJECT_ROOT / "outputs/lpwm_48gb_planning_v1/partial_output_layers/batch8/metric_plus_world",
    "attention_lora": PROJECT_ROOT / "outputs/lpwm_card_budget_measured_v4/attention_lora/batch8/metric_plus_world",
}
REPORT_DIRECTORY = PROJECT_ROOT / "results/lpwm_card_budget_measured_v4/completed_lora_review_20261005"


def read_json(path):
    return json.loads(path.read_text())


def main():
    REPORT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    planning_rows, world_rows, initial_rows, summaries, costs, source_digests = {}, {}, {}, {}, {}, {}
    for method, directory in RUN_DIRECTORIES.items():
        for filename in ("trained.json", "world.json", "initial.json", "summary.json"):
            path = directory / "trend_evaluation" / filename
            source_digests[str(path.relative_to(PROJECT_ROOT))] = hashlib.sha256(path.read_bytes()).hexdigest()
        planning_rows[method] = read_json(directory / "trend_evaluation/trained.json")
        world_rows[method] = read_json(directory / "trend_evaluation/world.json")
        initial_rows[method] = read_json(directory / "trend_evaluation/initial.json")
        summaries[method] = read_json(directory / "trend_evaluation/summary.json")
        training = read_json(directory / "training_summary.json")
        inventory = read_json(directory / "parameter_inventory.json")
        assert training["completed_updates"] == 4707 and not training["profile_only"]
        assert not summaries[method]["engineering_only"]
        costs[method] = {key: training[key] for key in ("seconds", "peak_allocated_gib", "completed_updates")}
        costs[method].update({key: inventory[key] for key in (
            "trainable_parameters", "trainable_world_parameters", "trainable_planner_parameters")})
        costs[method]["evaluation_seconds"] = summaries[method]["seconds"]

    partial, lora = "partial_output_layers", "attention_lora"
    for collection in (planning_rows, world_rows, initial_rows):
        assert [(row["token"], row["recording_group"]) for row in collection[partial]] == [
            (row["token"], row["recording_group"]) for row in collection[lora]]
    paired_planning = {metric: paired_recording_interval(planning_rows[lora], planning_rows[partial], metric)
                       for metric in ("pdms", "ade_meters", "fde_meters", "metric_bce")}
    paired_world = {metric: paired_recording_interval(world_rows[lora], world_rows[partial], metric)
                   for metric in ("reconstruction_lpips", "forecast_lpips")}
    stage1_records = read_json(PROJECT_ROOT /
        "outputs/lpwm_navsim_full_posttraining_v2/evaluation/posttrained/metrics.json")["records"]
    stage1_by_token = {row["token"]: row for row in stage1_records}
    stage1_panel = [stage1_by_token[row["token"]] for row in world_rows[lora]]
    world_means = {"stage1": metric_means(stage1_panel), **{
        method: metric_means(rows) for method, rows in world_rows.items()}}
    world_changes = {method: {
        metric: (world_means[method][metric] / world_means["stage1"][metric] - 1) * 100
        for metric in ("reconstruction_lpips", "forecast_lpips")}
        for method in RUN_DIRECTORIES}
    initial_mismatches = [first["token"] for first, second in zip(initial_rows[partial], initial_rows[lora])
                          if first["candidate_index"] != second["candidate_index"]]
    final_candidate_changes = sum(first["candidate_index"] != second["candidate_index"]
                                  for first, second in zip(planning_rows[partial], planning_rows[lora]))
    # Risk flags are existing Stage1 definitions, not bins based on model scores.
    risk_groups = {}
    for risk in summaries[lora]["world_risk_breakdown"]:
        selected = [row for row in world_rows[lora] if row["risk_flags"].get(risk)]
        risk_groups[risk] = {
            "clips": len(selected),
            "forecast_change_vs_stage1": paired_recording_interval(selected, stage1_panel, "forecast_lpips"),
            "scope": "Exploratory whole-image metric in flagged scenes, not object-specific information retention",
        }
    report = {
        "created_at": datetime.now(ZoneInfo("Asia/Seoul")).isoformat(),
        "scope": "Same1024 internal-development planning scenes and256 video clips; not navtest",
        "training": {"scenes": 75297, "epochs": 1, "updates": 4707, "seed": 47, "object_gt_auxiliary": False},
        "planning_means": {method: metric_means(rows) for method, rows in planning_rows.items()},
        "world_means": {method: {metric: values[metric] for metric in ("reconstruction_lpips", "forecast_lpips")}
                        for method, values in world_means.items()},
        "world_relative_change_percent_vs_stage1": world_changes,
        "lora_minus_partial_planning": paired_planning,
        "lora_minus_partial_world": paired_world,
        "interval_method": "2000 paired recording-cluster bootstrap resamples, seed20261003; sample-weighted means",
        "lora_predicted_minus_persistent_future": summaries[lora]["predicted_minus_persistent_future"],
        "lora_checks": summaries[lora]["checks"],
        "partial_checks": summaries[partial]["checks"],
        "lora_risk_groups": risk_groups,
        "measured_costs": costs,
        "initial_candidate_mismatch_count": len(initial_mismatches),
        "initial_candidate_mismatch_tokens": initial_mismatches,
        "initial_comparison_note": "Same configured Stage1/seed, but stored initial selections differ; cause not established in this read-only report.",
        "trained_candidate_changes_between_methods": final_candidate_changes,
        "source_sha256": source_digests,
        "limitations": [
            "One seed/epoch; intervals measure scene/recording variability, not training-seed variability.",
            "No separately trained frozen-LPWM control; initial reference is an untrained planner.",
            "Adaptation locations, capacity and historical microbatch schedules differ.",
            "Stored initial candidate selections are not identical; do not claim exact initial output equivalence across full panels.",
            "Persistent-future replacement changes inference inputs; it does not isolate the causal benefit of LPWM fine-tuning.",
            "World LPIPS retention is not proof of object-state preservation or planning-aware particle specialization.",
            "Training wall times include different shared GPU loads and execution histories; no isolated speed claim.",
        ],
        "runtime_changes": False,
    }
    (REPORT_DIRECTORY / "summary.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")

    plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False})
    figure, axes = plt.subplots(2, 2, figsize=(11, 7.8))
    colors = ["#d7953b", "#287aa6"]
    labels = ["Partial layers", "LoRA"]
    for axis, metric, title, scale, limit in (
        (axes[0, 0], "pdms", "Planning PDMS (higher is better)", 100, 100),
        (axes[0, 1], "ade_meters", "Average trajectory error (lower is better)", 1, 1.6),
        (axes[1, 0], "fde_meters", "Final trajectory error (lower is better)", 1, 3.7),
    ):
        values = [report["planning_means"][method][metric] * scale for method in (partial, lora)]
        bars = axis.bar(labels, values, color=colors, width=.52)
        axis.bar_label(bars, labels=[f"{value:.2f}" if metric == "pdms" else f"{value:.3f} m" for value in values], padding=5)
        axis.set_ylim(0, limit)
        axis.set_title(title, fontsize=12, pad=14)
        axis.grid(axis="y", alpha=.16)
        axis.set_axisbelow(True)
        interval = paired_planning[metric]
        axis.set_xlabel(f"LoRA - Partial: {interval['mean_difference'] * scale:+.3f}; "
                        f"95% CI [{interval['ci95'][0] * scale:+.3f}, {interval['ci95'][1] * scale:+.3f}]", fontsize=9)
    axis = axes[1, 1]
    values = [world_means[method]["forecast_lpips"] for method in ("stage1", partial, lora)]
    bars = axis.bar(["Stage1", *labels], values, color=["#9ba5ac", *colors], width=.55)
    axis.bar_label(bars, labels=[f"{value:.4f}" for value in values], padding=5)
    axis.set_ylim(0, .52)
    axis.set_title("Future-video LPIPS (lower is better)", fontsize=12, pad=14)
    axis.set_xlabel("Same 256 clips; Stage1 reference before planning adaptation", fontsize=9)
    axis.grid(axis="y", alpha=.16)
    axis.set_axisbelow(True)
    figure.suptitle("Completed LPWM adaptation: partial layers vs LoRA", fontsize=17, y=.98)
    figure.text(.5, .925, "One epoch | 1,024 development scenes (1,021 valid PDMS) | NOT official navtest",
                ha="center", fontsize=10, color="#555555")
    figure.tight_layout(rect=(0, .01, 1, .90), h_pad=3, w_pad=2)
    figure.savefig(REPORT_DIRECTORY / "comparison.png", dpi=170)
    figure.savefig(REPORT_DIRECTORY / "comparison.pdf")
    plt.close(figure)
    print(json.dumps({key: report[key] for key in (
        "planning_means", "world_means", "world_relative_change_percent_vs_stage1", "lora_minus_partial_planning",
        "initial_candidate_mismatch_count", "trained_candidate_changes_between_methods")}, indent=2))


if __name__ == "__main__":
    main()
