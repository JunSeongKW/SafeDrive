"""Measure NAVSIM Stage1's contribution using two independently trained frozen planners."""
import argparse
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from evaluate_lpwm_full_planning import digest, write_json
from lpwm_stage1_effect_protocol import PROJECT_ROOT, verify_stage1_effect_configuration
from summarize_lpwm_posttraining import metric_means, paired_recording_interval


def summarize(configuration_path):
    queue_configuration = json.loads(configuration_path.read_text())
    output_directory = PROJECT_ROOT / queue_configuration["output_directory"]
    shared_directory = PROJECT_ROOT / queue_configuration["shared_results_directory"]
    selection = json.loads((output_directory / "execution_selection.json").read_text())
    experiment_path = Path(selection["selected_configuration"])
    specification = verify_stage1_effect_configuration(experiment_path)
    contrast = specification["stage1_effect"]
    directories = {
        "public_pretrained_frozen": PROJECT_ROOT / specification["output_directory"] / "metric_plus_world",
        "navsim_posttrained_frozen": PROJECT_ROOT / contrast["reference_result_directory"],
    }
    planning, worlds, reports, training_reports, source_hashes = {}, {}, {}, {}, {}
    for method, directory in directories.items():
        reports[method] = json.loads((directory / "trend_evaluation/summary.json").read_text())
        training = json.loads((directory / "training_summary.json").read_text())
        training_reports[method] = training
        assert training["completed_updates"] == 4707 and training["epochs"] == 1
        assert training["effective_batch_size"] == 16 and training["train_clips"] == 75297
        assert not training["profile_only"] and training["frozen_state_unchanged"]
        assert not reports[method]["engineering_only"]
        assert digest(directory / "checkpoint.pt") == training["checkpoint_sha256"] == reports[method]["checkpoint_sha256"]
        planning[method] = json.loads((directory / "trend_evaluation/trained.json").read_text())
        worlds[method] = json.loads((directory / "trend_evaluation/world.json").read_text())
        for name in ("training_summary.json", "trend_evaluation/summary.json", "trend_evaluation/trained.json", "trend_evaluation/world.json"):
            source_hashes[str((directory / name).relative_to(PROJECT_ROOT))] = digest(directory / name)
    public_name, adapted_name = directories
    public_training = training_reports[public_name]
    assert public_training["navsim_stage1_performed"] is False
    assert public_training["initial_lpwm_checkpoint_sha256"] == contrast["public_checkpoint_sha256"]
    saved = torch.load(directories[public_name] / "checkpoint.pt", map_location="cpu", weights_only=True)
    initial = torch.load(PROJECT_ROOT / contrast["public_checkpoint"], map_location="cpu", weights_only=True)
    assert {name.removeprefix("world_model.") for name in saved if name.startswith("world_model.")} == set(initial)
    assert all(torch.equal(saved["world_model." + name], tensor) for name, tensor in initial.items())
    frozen_digest = hashlib.sha256()
    for prefix in ("world_model.", "command_feature_modulation."):
        for name, tensor in saved.items():
            if name.startswith(prefix):
                frozen_digest.update(name.encode())
                frozen_digest.update(tensor.contiguous().reshape(-1).view(torch.uint8).numpy().tobytes())
    assert frozen_digest.hexdigest() == public_training["frozen_representation_sha256"]
    del saved, initial
    for collection, expected_count in ((planning, 1024), (worlds, 256)):
        public_keys = [(row["token"], row["recording_group"]) for row in collection[public_name]]
        assert len(public_keys) == expected_count and len(set(public_keys)) == expected_count
        assert public_keys == [(row["token"], row["recording_group"]) for row in collection[adapted_name]]
    missing_tokens = {method: [row["token"] for row in rows if row["metrics"]["pdms"] is None]
        for method, rows in planning.items()}
    assert len(missing_tokens[public_name]) == 3 and missing_tokens[public_name] == missing_tokens[adapted_name]
    comparisons = {
        "planning": {metric: paired_recording_interval(planning[adapted_name], planning[public_name], metric)
            for metric in ("pdms", "ade_meters", "fde_meters", "metric_bce")},
        "world": {metric: paired_recording_interval(worlds[adapted_name], worlds[public_name], metric)
            for metric in ("reconstruction_lpips", "forecast_lpips")},
    }
    public_reference = json.loads((PROJECT_ROOT / contrast["initial_world_evaluation"]).read_text())["records"]
    by_token = {row["token"]: row for row in public_reference}
    scenarios = {}
    for scenario in sorted({by_token[row["token"]]["scenario"] for row in worlds[public_name]}):
        selected_tokens = {row["token"] for row in worlds[public_name] if by_token[row["token"]]["scenario"] == scenario}
        scenarios[scenario] = {"clips": len(selected_tokens), "forecast_lpips": {
            method: metric_means([row for row in rows if row["token"] in selected_tokens])["forecast_lpips"]
            for method, rows in worlds.items()}}
    interval = comparisons["planning"]["pdms"]
    if interval["ci95"][0] > 0:
        conclusion = "NAVSIM Stage1 improves PDMS on this development panel under the matched frozen-planner recipe; one-seed uncertainty and independent-test performance remain unresolved."
    elif interval["ci95"][1] < 0:
        conclusion = "NAVSIM Stage1 reduces PDMS under this recipe on this development panel; no planning benefit established."
    else:
        conclusion = "Additional PDMS benefit from NAVSIM Stage1 is not established; the paired interval includes zero. This is not evidence of equivalence."
    result = {
        "complete": True, "configuration_sha256": digest(configuration_path),
        "public_configuration_sha256": digest(experiment_path),
        "reference_reused_without_retraining": True,
        "public_weights_and_buffers_match_official_checkpoint": True,
        "frozen_representation_sha256": frozen_digest.hexdigest(),
        "scope": "1epoch/seed47;75297navtrain training scenes; same1024development scenes/1021valid cached PDM scores/40recordings;256world clips; not navtest.",
        "planning_means": {method: metric_means(rows) for method, rows in planning.items()},
        "world_means": {method: metric_means(rows) for method, rows in worlds.items()},
        "adapted_minus_public": comparisons, "conclusion": conclusion,
        "world_scenario_breakdown": scenarios,
        "world_risk_breakdown": {risk: paired_recording_interval(
            [row for row in worlds[adapted_name] if row["risk_flags"].get(risk)], worlds[public_name], "forecast_lpips")
            for risk in sorted({risk for row in worlds[adapted_name] for risk, present in row["risk_flags"].items() if present})},
        "reports": reports, "training_reports": training_reports,
        "missing_pdms_tokens": missing_tokens[public_name],
        "interval_method": "2000 paired recording-cluster bootstrap draws, seed20261003; exploratory unadjusted intervals, no training-seed uncertainty.",
        "source_sha256": source_hashes,
        "limitations": [
            "The public checkpoint is pretrained on Sketchy; this tests NAVSIM adaptation, not pretraining versus random initialization.",
            "Both fixed representations include causal future rollout. This contrast does not isolate particle structure, prediction versus current state, or object-state preservation.",
            "Shared GPU load differs across runs; training times are descriptive, not a matched speed benchmark.",
            "Single seed/epoch and exposed development scenes do not establish convergence or independent-test generalization.",
        ],
    }
    write_json(output_directory / "stage1_effect_summary.json", result)
    write_json(shared_directory / "stage1_effect_summary.json", result)
    create_figure(result, shared_directory)
    print("STAGE1_EFFECT_COMPARISON_COMPLETE", json.dumps(result["planning_means"]), flush=True)


def create_figure(result, directory):
    methods = ["public_pretrained_frozen", "navsim_posttrained_frozen"]
    labels = ["Public LPWM\n(no NAVSIM Stage1)", "NAVSIM-adapted LPWM\n(completed control)"]
    figure, axes = plt.subplots(1, 3, figsize=(13, 4.5))
    for axis, means, metric, title, factor in (
        (axes[0], result["planning_means"], "pdms", "Development PDMS (higher is better)", 100),
        (axes[1], result["planning_means"], "ade_meters", "ADE in meters (lower is better)", 1),
        (axes[2], result["world_means"], "forecast_lpips", "Future LPIPS (lower is better)", 1),
    ):
        values = [means[method][metric] * factor for method in methods]
        bars = axis.bar(labels, values, color=["#aa7a41", "#426e86"], width=.55)
        axis.bar_label(bars, fmt="%.3f", padding=4)
        axis.set_title(title, fontsize=10)
        axis.set_ylim(0, max(values) * 1.18)
        axis.spines[["top", "right"]].set_visible(False)
    interval = result["adapted_minus_public"]["planning"]["pdms"]
    center = interval["mean_difference"] * 100
    lower, upper = np.array(interval["ci95"]) * 100
    figure.suptitle(f"NAVSIM Stage1 effect: PDMS {center:+.2f} points, paired 95% CI [{lower:+.2f}, {upper:+.2f}]", fontsize=13)
    figure.text(.5, .025, "Both LPWMs frozen; same planner training, batch16, seed47,1epoch. Internal development, NOT navtest.", ha="center", fontsize=9)
    figure.tight_layout(rect=(0, .07, 1, .9))
    figure.savefig(directory / "stage1_effect_comparison.png", dpi=160)
    figure.savefig(directory / "stage1_effect_comparison.pdf")
    plt.close(figure)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    summarize(parser.parse_args().config.resolve())
