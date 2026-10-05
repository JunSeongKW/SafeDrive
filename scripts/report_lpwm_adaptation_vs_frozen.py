"""Audit and visualize completed LPWM adaptation versus frozen-planner results."""
import hashlib
import json
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from report_lpwm_four_adaptation_results import RUN_DIRECTORIES as ADAPTATION_DIRECTORIES
from summarize_lpwm_posttraining import metric_means, paired_recording_interval


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONTROL_METHOD = "frozen_lpwm_planner_control"
RUN_DIRECTORIES = {
    CONTROL_METHOD: "outputs/lpwm_frozen_control_v1/batch8/metric_plus_world",
    **ADAPTATION_DIRECTORIES,
}
REPORT_DIRECTORY = PROJECT_ROOT / "results/lpwm_frozen_control_v1/completed_comparison_20261005"


def main():
    source_hashes = {}

    def read_record(relative_path):
        contents = (PROJECT_ROOT / relative_path).read_bytes()
        source_hashes[str(relative_path)] = hashlib.sha256(contents).hexdigest()
        return json.loads(contents)

    registered = read_record("results/lpwm_frozen_control_v1/queue/adaptation_vs_frozen_summary.json")
    queue_state = read_record("outputs/lpwm_frozen_control_v1/queue/queue_state.json")
    completion = read_record("outputs/lpwm_frozen_control_v1/queue/queue_completion.json")
    assert registered["complete"] and completion["complete"] and queue_state["status"] == "complete"
    assert registered["frozen_weights_and_buffers_verified_unchanged"]
    planning_rows, world_rows, initial_rows = {}, {}, {}
    for method, relative_directory in RUN_DIRECTORIES.items():
        directory = Path(relative_directory)
        planning_rows[method] = read_record(directory / "trend_evaluation/trained.json")
        world_rows[method] = read_record(directory / "trend_evaluation/world.json")
        initial_rows[method] = read_record(directory / "trend_evaluation/initial.json")
        evaluation = read_record(directory / "trend_evaluation/summary.json")
        training = read_record(directory / "training_summary.json")
        assert training["completed_updates"] == 4707 and not training["profile_only"]
        assert not evaluation["engineering_only"]
        assert training["checkpoint_sha256"] == evaluation["checkpoint_sha256"]
        assert len(planning_rows[method]) == 1024 and len(world_rows[method]) == 256
        assert evaluation == registered["reports"][method]
    for collection in (planning_rows, world_rows, initial_rows):
        expected = [(row["token"], row["recording_group"]) for row in collection[CONTROL_METHOD]]
        assert len({token for token, _ in expected}) == len(expected)
        for rows in collection.values():
            assert [(row["token"], row["recording_group"]) for row in rows] == expected
    missing_tokens = {
        method: [row["token"] for row in rows if row["metrics"]["pdms"] is None]
        for method, rows in planning_rows.items()
    }
    assert len(missing_tokens[CONTROL_METHOD]) == 3
    assert all(tokens == missing_tokens[CONTROL_METHOD] for tokens in missing_tokens.values())
    for relative_path, expected_hash in registered["source_sha256"].items():
        assert source_hashes[relative_path] == expected_hash
    for domain, collection in (("planning", planning_rows), ("world", world_rows)):
        for method, rows in collection.items():
            measured = metric_means(rows)
            for metric, value in registered[domain + "_means"][method].items():
                assert np.isclose(measured[metric], value, rtol=0, atol=1e-12)
            if method == CONTROL_METHOD:
                continue
            for metric, interval in registered["paired_comparisons"][method + "_minus_frozen"][domain].items():
                assert paired_recording_interval(rows, collection[CONTROL_METHOD], metric) == interval
    stage1 = read_record("outputs/lpwm_navsim_full_posttraining_v2/evaluation/posttrained/metrics.json")
    stage1_by_token = {row["token"]: row for row in stage1["records"]}
    stage1_panel = [stage1_by_token[row["token"]] for row in world_rows[CONTROL_METHOD]]
    scenarios = {}
    for scenario in sorted({row["scenario"] for row in stage1_panel}):
        selected_tokens = {row["token"] for row in stage1_panel if row["scenario"] == scenario}
        scenarios[scenario] = {"clips": len(selected_tokens), "forecast_lpips": {
            method: metric_means([row for row in rows if row["token"] in selected_tokens])["forecast_lpips"]
            for method, rows in {"stage1": stage1_panel, **world_rows}.items()}}
    report = {
        **registered,
        "created_at_kst": datetime.now(ZoneInfo("Asia/Seoul")).isoformat(),
        "completed_at_kst": datetime.fromtimestamp(completion["finished_unix"], ZoneInfo("Asia/Seoul")).isoformat(),
        "training": {"navtrain_scenes": 75297, "epochs": 1, "updates": 4707, "seed": 47,
            "direct_object_gt_auxiliary": False},
        "evaluation_definition": "Same 1024 development scenes from 40 recordings; PDMS uses cached official PDM scores for selected unchanged vocabulary candidates. Three common invalid teacher scenes are excluded (1021 valid). Not full navtest or independent test.",
        "interval_method": registered["interval_method"] + "; exploratory intervals without multiple-comparison correction",
        "conclusion": "Additional planning benefit of adapting the representation is not demonstrated under these settings. All four PDMS difference intervals include zero. This is not evidence of equivalence or universal ineffectiveness.",
        "world_scenario_breakdown": scenarios,
        "world_forecast_change_percent_vs_frozen": {method: 100 * (
            registered["world_means"][method]["forecast_lpips"] /
            registered["world_means"][CONTROL_METHOD]["forecast_lpips"] - 1)
            for method in RUN_DIRECTORIES},
        "checks_by_method": {method: summary["checks"] for method, summary in registered["reports"].items()},
        "missing_pdms_tokens": missing_tokens[CONTROL_METHOD],
        "source_sha256": source_hashes,
        "runtime_changes": False,
    }
    REPORT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    (REPORT_DIRECTORY / "summary.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    create_figure(report)
    print(json.dumps({"report_directory": str(REPORT_DIRECTORY), "audit_passed": True,
        "world_scenario_breakdown": scenarios}, ensure_ascii=False, indent=2))


def create_figure(report):
    methods = list(RUN_DIRECTORIES)
    labels = ["Frozen LPWM", "Partial", "LoRA", "Adapter", "Full low-LR"]
    colors = ["#475569", "#bf8336", "#287aa6", "#328568", "#9470a2"]
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    figure, axes = plt.subplots(2, 2, figsize=(13, 8.8))
    axis = axes[0, 0]
    values = [100 * report["planning_means"][method]["pdms"] for method in methods]
    bars = axis.bar(labels, values, color=colors, width=.58)
    axis.bar_label(bars, labels=[f"{value:.2f}" for value in values], padding=5)
    axis.set(title="PDMS on the common development panel", ylim=(0, 100), ylabel="PDMS points (higher is better)")
    axis = axes[0, 1]
    for position, method in enumerate(methods[1:]):
        interval = report["paired_comparisons"][method + "_minus_frozen"]["planning"]["pdms"]
        center = 100 * interval["mean_difference"]
        lower, upper = np.asarray(interval["ci95"]) * 100
        axis.errorbar(center, 3 - position, xerr=[[center - lower], [upper - center]],
            fmt="o", capsize=5, color=colors[position + 1])
        axis.annotate(f"{center:+.2f} [{lower:+.2f}, {upper:+.2f}]", (upper + .08, 3 - position),
            va="center", fontsize=8)
    axis.axvline(0, color="#666666", linestyle="--", linewidth=1)
    axis.set_yticks([3, 2, 1, 0], labels[1:])
    axis.set(title="Adaptation minus frozen: paired 95% intervals", xlabel="PDMS points (positive favors adaptation)",
        xlim=(-3.6, 3.1), ylim=(-.6, 3.6))
    axis = axes[1, 0]
    positions = np.arange(len(methods))
    for offset, metric, label, color in ((-.18, "ade_meters", "ADE", "#368568"), (.18, "fde_meters", "FDE", "#7da4c0")):
        values = [report["planning_means"][method][metric] for method in methods]
        bars = axis.bar(positions + offset, values, width=.35, label=label, color=color)
        axis.bar_label(bars, fmt="%.3f", padding=4, fontsize=8)
    axis.set_xticks(positions, labels)
    axis.set(title="Trajectory error (lower is better)", ylabel="Meters", ylim=(0, 4.1))
    axis.legend(frameon=False, loc="upper right")
    axis = axes[1, 1]
    values = [report["world_forecast_change_percent_vs_frozen"][method] for method in methods]
    bars = axis.bar(labels, values, color=colors, width=.58)
    axis.bar_label(bars, labels=[f"{value:+.2f}%" for value in values], padding=5)
    axis.axhline(0, color="#777777", linewidth=.8)
    axis.set(title="Future-video error change from frozen LPWM", ylabel="LPIPS change (%) - lower is better", ylim=(-2, 15))
    for axis in axes.flat:
        axis.grid(axis="y", alpha=.15)
        axis.set_axisbelow(True)
    figure.suptitle("Does adapting LPWM improve planning beyond training the planner?", fontsize=16, y=.98)
    figure.text(.5, .934, "No additional benefit established | 1 epoch / 1 seed | Internal development: 1,024 scenes, 1,021 valid PDMS", ha="center", color="#555555")
    figure.text(.5, .024, "40-recording paired bootstrap; no seed uncertainty. Full uses a different LR schedule / intent location. NOT official navtest.", ha="center", fontsize=9, color="#555555")
    figure.tight_layout(rect=(0, .045, 1, .91), h_pad=3, w_pad=2.4)
    figure.savefig(REPORT_DIRECTORY / "comparison.png", dpi=160)
    figure.savefig(REPORT_DIRECTORY / "comparison.pdf")
    plt.close(figure)


if __name__ == "__main__":
    main()
