"""Compare pretrained/post-trained LPWM and enforce the registered stage2 gate."""
import argparse
from collections import defaultdict
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def paired_recording_interval(records, reference_records, metric, reference_metric=None):
    reference_metric = reference_metric or metric
    references = {record["token"]: record for record in reference_records}
    grouped_differences = defaultdict(list)
    differences = []
    for record in records:
        reference = references[record["token"]]
        first_value = record["metrics"].get(metric)
        second_value = reference["metrics"].get(reference_metric)
        if first_value is None or second_value is None:
            continue
        difference = float(first_value - second_value)
        if not np.isfinite(difference):
            raise ValueError("Nonfinite paired metric")
        grouped_differences[record["recording_group"]].append(difference)
        differences.append(difference)
    if not grouped_differences:
        return {"mean_difference": None, "ci95": [None, None], "clips": 0, "recordings": 0}
    recording_sums = np.array([sum(values) for values in grouped_differences.values()])
    recording_counts = np.array([len(values) for values in grouped_differences.values()])
    generator = np.random.default_rng(20261003)
    sampled_groups = generator.integers(len(recording_sums), size=(2000, len(recording_sums)))
    sampled_means = recording_sums[sampled_groups].sum(1) / recording_counts[sampled_groups].sum(1)
    return {"mean_difference": float(np.mean(differences)),
        "ci95": np.quantile(sampled_means, [.025, .975]).tolist(),
        "clips": len(differences), "recordings": len(recording_sums)}


def metric_means(records):
    names = sorted({name for record in records for name, value in record["metrics"].items()
        if value is not None and isinstance(value, (float, int))})
    return {name: float(np.mean([record["metrics"][name] for record in records
        if record["metrics"].get(name) is not None])) for name in names}


def summarize(specification):
    output_root = PROJECT_ROOT / specification["output_directory"]
    shared_root = PROJECT_ROOT / specification["shared_results_directory"]
    shared_root.mkdir(parents=True, exist_ok=True)
    published = json.loads((output_root / "evaluation/published/metrics.json").read_text())
    adapted = json.loads((output_root / "evaluation/posttrained/metrics.json").read_text())
    training = json.loads((output_root / "stage1/training_summary.json").read_text())
    manifest = json.loads((output_root / "manifest.json").read_text())
    assert not training["profile_only"] and not published["profile_only"] and not adapted["profile_only"]
    assert training["checkpoint_sha256"] == adapted["checkpoint_sha256"]
    assert training["initial_checkpoint_sha256"] == published["checkpoint_sha256"]
    expected_tokens = {row["current_frame_token"] for row in manifest["records"] if row["split"] == "development"}
    assert {row["token"] for row in published["records"]} == expected_tokens
    assert {row["token"] for row in adapted["records"]} == expected_tokens
    comparisons = {
        "reconstruction_lpips_vs_published": paired_recording_interval(adapted["records"], published["records"], "reconstruction_lpips"),
        "forecast_lpips_vs_published": paired_recording_interval(adapted["records"], published["records"], "forecast_lpips"),
        "forecast_lpips_vs_persistence": paired_recording_interval(adapted["records"], adapted["records"], "forecast_lpips", "persistence_lpips"),
        "forecast_object_region_mse_vs_persistence": paired_recording_interval(adapted["records"], adapted["records"], "forecast_object_region_mse", "persistence_object_region_mse"),
        "object_box_recall_vs_published": paired_recording_interval(adapted["records"], published["records"], "top16_object_box_recall_iou_010"),
    }
    checks = {name: value["ci95"][1] is not None and value["ci95"][1] < 0
        for name, value in comparisons.items() if name != "object_box_recall_vs_published"}
    interval = comparisons["object_box_recall_vs_published"]["ci95"]
    checks["object_box_recall_noninferiority"] = interval[0] is not None and interval[0] >= -.02
    checks["all_core_modules_updated"] = all(value > 0 for value in training["parameter_tensors_changed_in_sample"].values())
    checks["all_core_modules_receive_gradient"] = all(value["norm"] > 0 and np.isfinite(value["norm"]) for value in training["module_gradients"].values())
    for name, evaluation in (("published", published), ("posttrained", adapted)):
        checks[name + "_causal_forecast"] = any(record["metrics"].get("future_intervention_forecast_difference", 1.) < 1e-5
            and record["metrics"].get("future_intervention_observed_particle_difference", 1.) < 1e-5 for record in evaluation["records"])
    risk_configuration = json.loads((PROJECT_ROOT / "configs/lpwm_navsim_adaptation/driving_risks_v1.json").read_text())
    risk_breakdown = {}
    for risk_name in sorted({name for record in adapted["records"] for name in record["risk_flags"]}):
        post_rows = [row for row in adapted["records"] if row["risk_flags"][risk_name]]
        original_rows = [row for row in published["records"] if row["risk_flags"][risk_name]]
        interval = paired_recording_interval(post_rows, post_rows, "forecast_lpips", "persistence_lpips")
        record_count = interval["recordings"]
        means = metric_means(post_rows) if post_rows else {}
        risk_breakdown[risk_name] = {"clips": len(post_rows), "recordings": record_count,
            "published": metric_means(original_rows) if original_rows else {}, "posttrained": means,
            "forecast_lpips_vs_persistence": interval}
        if record_count >= risk_configuration["minimum_recordings_for_stratum_gate"]:
            margin = risk_configuration["gate_noninferiority_relative_margin"] * means["persistence_lpips"]
            checks["risk_noninferiority_" + risk_name] = interval["ci95"][1] <= margin
            risk_breakdown[risk_name]["noninferiority_margin"] = margin
        else:
            risk_breakdown[risk_name]["inference_limitation"] = "insufficient recording coverage for a stratum gate"
    passed = all(checks.values())
    scenario_breakdown = {}
    for scenario in ("straight", "turn", "projected_overlap", "other"):
        post_rows = [row for row in adapted["records"] if row["scenario"] == scenario]
        original_rows = [row for row in published["records"] if row["scenario"] == scenario]
        if post_rows:
            scenario_breakdown[scenario] = {"clips": len(post_rows), "published": metric_means(original_rows),
                "posttrained": metric_means(post_rows),
                "forecast_lpips_vs_persistence": paired_recording_interval(post_rows, post_rows, "forecast_lpips", "persistence_lpips")}
    # Alternative current-context definitions avoid conflating overlap and turning.
    alternative_breakdown = {}
    for name, predicate in (
        ("turn_yaw_at_least_10deg", lambda row: row["yaw_range_degrees"] >= 10),
        ("turn_yaw_at_least_15deg", lambda row: row["yaw_range_degrees"] >= 15),
        ("overlap_at_least_025", lambda row: row["projected_overlap_fraction"] >= .25),
        ("overlap_at_least_050", lambda row: row["projected_overlap_fraction"] >= .5),
    ):
        rows = [row for row in adapted["records"] if predicate(row)]
        alternative_breakdown[name] = {"clips": len(rows), "means": metric_means(rows) if rows else {}}
    summary = {"adaptation_gate_passed": passed, "gate_checks": checks,
        "unmet_criteria": [name for name, value in checks.items() if not value],
        "stage2_status": "ready_for_low_lr_lpwm_and_full_planner_training" if passed else "not_started_adaptation_gate_not_met",
        "published": metric_means(published["records"]), "posttrained": metric_means(adapted["records"]),
        "paired_recording_comparisons": comparisons, "scenario_breakdown": scenario_breakdown,
        "alternative_scenario_definitions": alternative_breakdown, "driving_risk_breakdown": risk_breakdown, "training": training,
        "counts": manifest["counts"], "recording_counts": manifest["recording_counts"],
        "limitations": ["One adaptation seed, recording-disjoint development, not independent test",
            "Projected boxes and overlap are object/occlusion proxies, not segmentation ground truth",
            "Passing this gate does not establish planning utility or persistent object identity",
            "Merged stage2 is not included in this world-model report"]}
    for path in (output_root / "adaptation_gate.json", shared_root / "summary.json"):
        path.write_text(json.dumps(summary, indent=2) + "\n")
    updates = [json.loads(line) for line in (output_root / "stage1/training_log.jsonl").read_text().splitlines()]
    figure, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    axes[0].plot([row["update"] for row in updates], [row["loss"] for row in updates])
    axes[0].set(title="Official temporal ELBO", xlabel="Optimizer update", ylabel="Training objective")
    horizon = np.arange(1, 9) * .5
    for name, evaluation in (("Published", published), ("Post-trained", adapted)):
        axes[1].plot(horizon, np.mean([row["metrics"]["forecast_lpips_by_horizon"] for row in evaluation["records"]], 0), marker="o", label=name)
    axes[1].plot(horizon, np.mean([row["metrics"]["persistence_lpips_by_horizon"] for row in adapted["records"]], 0), linestyle="--", label="Last frame")
    axes[1].set(title="Past-only future prediction", xlabel="Future time (s)", ylabel="LPIPS (lower is better)")
    axes[1].legend()
    plotted_scenarios = list(scenario_breakdown)
    positions = np.arange(len(plotted_scenarios))
    for offset, name in ((-.18, "published"), (.18, "posttrained")):
        axes[2].bar(positions + offset, [scenario_breakdown[scenario][name].get("top16_object_box_recall_iou_010", np.nan) for scenario in plotted_scenarios], width=.36, label=name)
    axes[2].set_xticks(positions, [name.replace("projected_overlap", "overlap proxy") for name in plotted_scenarios], rotation=20)
    axes[2].set(title="Current object correspondence", ylabel="Top16 box recall, IoU >= 0.1")
    axes[2].legend()
    figure.tight_layout()
    figure.savefig(shared_root / "stage1_summary.png", dpi=160)
    plt.close(figure)
    print("ADAPTATION_GATE", "PASSED" if passed else "NOT_MET", summary["unmet_criteria"], flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs/lpwm_navsim_adaptation/posttraining_v1.json")
    arguments = parser.parse_args()
    summarize(json.loads(arguments.config.read_text()))
