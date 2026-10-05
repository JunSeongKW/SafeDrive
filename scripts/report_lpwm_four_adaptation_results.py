"""Compare four completed LPWM runs from saved rows; never launch inference."""
import hashlib
import itertools
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
    "partial_output_layers": "outputs/lpwm_48gb_planning_v1/partial_output_layers/batch8/metric_plus_world",
    "attention_lora": "outputs/lpwm_card_budget_measured_v4/attention_lora/batch8/metric_plus_world",
    "residual_adapter": "outputs/lpwm_card_budget_measured_v4/residual_adapter/batch8/metric_plus_world",
    "full_low_learning_rate": "outputs/lpwm_card_budget_measured_v4/full_low_learning_rate/batch4/metric_plus_world",
}
REPORT_DIRECTORY = PROJECT_ROOT / "results/lpwm_card_budget_measured_v4/completed_four_method_review_20261005"


def main():
    source_hashes = {}

    def read_record(relative_path):
        path = PROJECT_ROOT / relative_path
        contents = path.read_bytes()
        source_hashes[str(relative_path)] = hashlib.sha256(contents).hexdigest()
        return json.loads(contents)

    registered = read_record("results/lpwm_card_budget_measured_v4/queue/four_method_summary.json")
    assert registered["complete"]
    planning_rows, world_rows, initial_rows, summaries, final_learning_rates = {}, {}, {}, {}, {}
    for method, relative_directory in RUN_DIRECTORIES.items():
        directory = Path(relative_directory)
        planning_rows[method] = read_record(directory / "trend_evaluation/trained.json")
        world_rows[method] = read_record(directory / "trend_evaluation/world.json")
        initial_rows[method] = read_record(directory / "trend_evaluation/initial.json")
        summaries[method] = read_record(directory / "trend_evaluation/summary.json")
        training = read_record(directory / "training_summary.json")
        assert training["completed_updates"] == 4707 and not training["profile_only"]
        assert not summaries[method]["engineering_only"]
        assert training["checkpoint_sha256"] == summaries[method]["checkpoint_sha256"]
        assert len(planning_rows[method]) == 1024 and len(world_rows[method]) == 256
        log_path = directory / "training_log.jsonl"
        log_contents = (PROJECT_ROOT / log_path).read_bytes()
        source_hashes[str(log_path)] = hashlib.sha256(log_contents).hexdigest()
        latest_training_row = json.loads(log_contents.splitlines()[-1])
        assert latest_training_row["update"] == 4707
        final_learning_rates[method] = {metric: latest_training_row[metric]
            for metric in ("lpwm_lr", "planner_lr")}
    reference_method = "residual_adapter"
    for collection in (planning_rows, world_rows, initial_rows):
        expected = [(row["token"], row["recording_group"]) for row in collection[reference_method]]
        assert len({token for token, _ in expected}) == len(expected)
        for rows in collection.values():
            assert [(row["token"], row["recording_group"]) for row in rows] == expected
    missing_tokens = {
        method: [row["token"] for row in rows if row["metrics"]["pdms"] is None]
        for method, rows in planning_rows.items()
    }
    assert all(tokens == missing_tokens[reference_method] for tokens in missing_tokens.values())
    assert len(missing_tokens[reference_method]) == 3
    stage1 = read_record("outputs/lpwm_navsim_full_posttraining_v2/evaluation/posttrained/metrics.json")
    stage1_by_token = {row["token"]: row for row in stage1["records"]}
    stage1_panel = [stage1_by_token[row["token"]] for row in world_rows[reference_method]]
    world_means = {"stage1": metric_means(stage1_panel), **{
        method: metric_means(rows) for method, rows in world_rows.items()}}
    planning_means = {method: metric_means(rows) for method, rows in planning_rows.items()}
    for method, metrics in planning_means.items():
        for metric in ("pdms", "ade_meters", "fde_meters", "metric_bce"):
            assert np.isclose(metrics[metric], registered["reports"][method]["means"]["trained"][metric], atol=1e-10)
    paired = {}
    for first, second in itertools.combinations(RUN_DIRECTORIES, 2):
        key = second + "_minus_" + first
        paired[key] = {
            "planning": {metric: paired_recording_interval(planning_rows[second], planning_rows[first], metric)
                for metric in ("pdms", "ade_meters", "fde_meters", "metric_bce")},
            "world": {metric: paired_recording_interval(world_rows[second], world_rows[first], metric)
                for metric in ("reconstruction_lpips", "forecast_lpips")},
        }
        assert paired[key] == registered["paired_comparisons"][key]
    scenarios = {}
    for scenario in sorted({row["scenario"] for row in stage1_panel}):
        selected_tokens = {row["token"] for row in stage1_panel if row["scenario"] == scenario}
        scenarios[scenario] = {"clips": len(selected_tokens), "forecast_lpips": {
            method: metric_means([row for row in rows if row["token"] in selected_tokens])["forecast_lpips"]
            for method, rows in {"stage1": stage1_panel, **world_rows}.items()}}
    report = {
        "created_at_kst": datetime.now(ZoneInfo("Asia/Seoul")).isoformat(),
        "scope": "1024 internal-development scenes, 1021 valid cached PDMS, 40 recordings; 256 world clips. Not official navtest.",
        "training": {"navtrain_scenes": 75297, "epochs": 1, "updates": 4707, "seed": 47,
            "direct_object_gt_auxiliary": False},
        "planning_means": planning_means,
        "world_means": {method: {metric: metrics[metric] for metric in ("reconstruction_lpips", "forecast_lpips")}
            for method, metrics in world_means.items()},
        "world_change_percent_vs_stage1": {method: {
            metric: 100 * (world_means[method][metric] / world_means["stage1"][metric] - 1)
            for metric in ("reconstruction_lpips", "forecast_lpips")} for method in RUN_DIRECTORIES},
        "paired_comparisons": paired,
        "interval_method": "2000 paired recording-cluster bootstrap resamples, fixed seed20261003; not training-seed uncertainty. Exploratory unadjusted intervals.",
        "checks_by_method": {method: summary["checks"] for method, summary in summaries.items()},
        "predicted_minus_persistent_future": {
            method: summary["predicted_minus_persistent_future"] for method, summary in summaries.items()},
        "world_scenario_breakdown": scenarios,
        "world_risk_breakdown": {method: summary["world_risk_breakdown"] for method, summary in summaries.items()},
        "measured_costs": registered["measured_costs"],
        "final_learning_rates": final_learning_rates,
        "missing_pdms_tokens": missing_tokens[reference_method],
        "initial_candidate_difference_counts_vs_adapter": {
            method: sum(row["candidate_index"] != reference["candidate_index"]
                for row, reference in zip(rows, initial_rows[reference_method]))
            for method, rows in initial_rows.items()},
        "limitations": [
            "One epoch and one seed; no convergence or independent-test conclusion.",
            "All six pairwise PDMS intervals include zero; mean ranking is not established superiority.",
            "Full tuning preserves the earlier conv_in command placement, LPWM1e-6 LR and20epoch schedule; others use conv_out, LPWM1e-5 and1epoch schedule.",
            "Frozen-control training is separate and its final result is not part of this four-method report.",
            "Timing includes prior-run active seconds for resumed methods; shared GPU load and microbatch histories differ. Not an isolated speed benchmark.",
            "Whole-image forecast LPIPS and projected-overlap labels do not measure semantic object retention or true occlusion recovery.",
            "Replacing predicted futures with repeated observations is an inference intervention, not a separately retrained no-future baseline.",
        ],
        "source_sha256": source_hashes,
        "runtime_changes": False,
    }
    REPORT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    (REPORT_DIRECTORY / "summary.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    create_figure(report)
    print(json.dumps({"report_directory": str(REPORT_DIRECTORY), "planning": planning_means,
        "forecast_change_percent": {method: values["forecast_lpips"] for method, values in report["world_change_percent_vs_stage1"].items()},
        "world_scenarios": scenarios}, ensure_ascii=False, indent=2))


def create_figure(report):
    methods = list(RUN_DIRECTORIES)
    labels = ["Partial layers", "LoRA", "Adapter", "Full low-LR"]
    colors = ["#c78937", "#287aa6", "#328568", "#9470a2"]
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    figure, axes = plt.subplots(2, 2, figsize=(12.5, 8.6))
    axis = axes[0, 0]
    values = [100 * report["planning_means"][method]["pdms"] for method in methods]
    bars = axis.bar(labels, values, color=colors, width=.58)
    axis.bar_label(bars, labels=[f"{value:.2f}" for value in values], padding=5)
    axis.set(title="Development PDMS (higher is better)", ylim=(0, 100))
    axis = axes[0, 1]
    positions = np.arange(len(methods))
    for offset, metric, label, color in ((-.18, "ade_meters", "ADE", "#368568"), (.18, "fde_meters", "FDE", "#7da4c0")):
        values = [report["planning_means"][method][metric] for method in methods]
        bars = axis.bar(positions + offset, values, width=.35, label=label, color=color)
        axis.bar_label(bars, fmt="%.3f", padding=4, fontsize=8)
    axis.set_xticks(positions, labels)
    axis.set(title="Trajectory errors (lower is better)", ylabel="Meters", ylim=(0, 4.15))
    axis.legend(frameon=False, loc="upper right")
    axis = axes[1, 0]
    values = [report["world_change_percent_vs_stage1"][method]["forecast_lpips"] for method in methods]
    bars = axis.bar(labels, values, color=colors, width=.58)
    axis.bar_label(bars, labels=[f"{value:+.2f}%" for value in values], padding=5)
    axis.axhline(0, color="#777777", linewidth=.8)
    axis.axhline(10, color="#a64b4b", linestyle="--", linewidth=1, label="10% retention margin")
    axis.set(title="Future-video error change from Stage1", ylabel="LPIPS change (%) - lower is better", ylim=(-2, 15))
    axis.legend(frameon=False, fontsize=9)
    axis = axes[1, 1]
    intervals = []
    for reference in ("partial_output_layers", "attention_lora", "full_low_learning_rate"):
        key = "residual_adapter_minus_" + reference
        if key in report["paired_comparisons"]:
            item = report["paired_comparisons"][key]["planning"]["pdms"]
            intervals.append((item["mean_difference"], *item["ci95"]))
        else:
            item = report["paired_comparisons"][reference + "_minus_residual_adapter"]["planning"]["pdms"]
            intervals.append((-item["mean_difference"], -item["ci95"][1], -item["ci95"][0]))
    centers, lower, upper = np.asarray(intervals).T * 100
    axis.errorbar(centers, [2, 1, 0], xerr=np.stack((centers-lower, upper-centers)), fmt="o", capsize=5, color=colors[2])
    axis.axvline(0, color="#777777", linestyle="--", linewidth=1)
    axis.set_yticks([2, 1, 0], ["Adapter - Partial", "Adapter - LoRA", "Adapter - Full"])
    axis.set(title="PDMS differences: paired 95% intervals", xlabel="PDMS points; every interval crosses zero", ylim=(-.6, 2.6))
    for axis in axes.flat:
        axis.grid(axis="y", alpha=.15)
        axis.set_axisbelow(True)
    figure.suptitle("Four completed LPWM adaptation experiments", fontsize=17, y=.98)
    figure.text(.5, .934, "One epoch / one seed | Same internal development panels | NOT official navtest", ha="center", color="#555555")
    figure.text(.5, .021, "Full tuning retains a different LR schedule and intent-input location. Intervals do not capture training-seed variability.", ha="center", fontsize=9, color="#555555")
    figure.tight_layout(rect=(0, .045, 1, .91), h_pad=3, w_pad=3)
    figure.savefig(REPORT_DIRECTORY / "comparison.png", dpi=160)
    figure.savefig(REPORT_DIRECTORY / "comparison.pdf")
    plt.close(figure)


if __name__ == "__main__":
    main()
