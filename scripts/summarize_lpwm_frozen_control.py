"""Compare completed adaptation methods with the trained fixed-representation planner."""
import argparse
import hashlib
import json
from pathlib import Path

import torch

from evaluate_lpwm_full_planning import digest, write_json
from summarize_lpwm_posttraining import metric_means, paired_recording_interval
from lpwm_frozen_control_protocol import PROJECT_ROOT, verify_control_configuration


def summarize(config_path):
    specification = json.loads(config_path.read_text())
    root = PROJECT_ROOT / specification["output_directory"]
    shared = PROJECT_ROOT / specification["shared_results_directory"]
    selected = json.loads((root / "execution_selection.json").read_text())
    control_path = Path(selected["selected_configuration"])
    control_config = verify_control_configuration(control_path)
    control_root = PROJECT_ROOT / control_config["output_directory"] / "metric_plus_world"
    control_training = json.loads((control_root / "training_summary.json").read_text())
    assert control_training["completed_updates"] == 4707 and control_training["frozen_state_unchanged"]
    # Verify every frozen weight and persistent buffer in the final artifact.
    saved = torch.load(control_root / "checkpoint.pt", map_location="cpu", weights_only=True)
    frozen_digest = hashlib.sha256()
    for prefix in ("world_model.", "command_feature_modulation."):
        for name, tensor in saved.items():
            if name.startswith(prefix):
                frozen_digest.update(name.encode())
                frozen_digest.update(tensor.contiguous().reshape(-1).view(torch.uint8).numpy().tobytes())
    del saved
    assert frozen_digest.hexdigest() == control_training["frozen_representation_sha256"]
    predecessor = PROJECT_ROOT / specification["predecessor_queue_directory"]
    predecessor_specification = json.loads((PROJECT_ROOT / specification["predecessor_configuration"]).read_text())
    directories = {"frozen_lpwm_planner_control": control_root}
    for method in predecessor_specification["methods"]:
        selection = json.loads((predecessor / (method["method"] + "_execution_selection.json")).read_text())
        config = json.loads(Path(selection["selected_configuration"]).read_text())
        directories[method["method"]] = PROJECT_ROOT / config["output_directory"] / "metric_plus_world"
    planning, worlds, initial, summaries, costs, source_hashes = {}, {}, {}, {}, {}, {}
    for method, directory in directories.items():
        summaries[method] = json.loads((directory / "trend_evaluation/summary.json").read_text())
        training = json.loads((directory / "training_summary.json").read_text())
        assert not summaries[method]["engineering_only"] and training["completed_updates"] == 4707
        assert summaries[method]["checkpoint_sha256"] == training["checkpoint_sha256"]
        inventory = json.loads((directory / "parameter_inventory.json").read_text())
        planning[method] = json.loads((directory / "trend_evaluation/trained.json").read_text())
        worlds[method] = json.loads((directory / "trend_evaluation/world.json").read_text())
        initial[method] = json.loads((directory / "trend_evaluation/initial.json").read_text())
        costs[method] = {name: training[name] for name in ("seconds", "peak_allocated_gib", "completed_updates")}
        costs[method].update({name: inventory[name] for name in ("trainable_parameters", "trainable_world_parameters", "trainable_planner_parameters")})
        for filename in ("trained.json", "world.json", "initial.json", "summary.json"):
            path = directory / "trend_evaluation" / filename
            source_hashes[str(path.relative_to(PROJECT_ROOT))] = digest(path)
    control_name = "frozen_lpwm_planner_control"
    comparisons = {}
    for method in directories:
        if method == control_name:
            continue
        for collection in (planning, worlds, initial):
            assert [(row["token"], row["recording_group"]) for row in collection[method]] == [
                (row["token"], row["recording_group"]) for row in collection[control_name]]
        comparisons[method + "_minus_frozen"] = {
            "planning": {metric: paired_recording_interval(planning[method], planning[control_name], metric)
                for metric in ("pdms", "ade_meters", "fde_meters", "metric_bce")},
            "world": {metric: paired_recording_interval(worlds[method], worlds[control_name], metric)
                for metric in ("reconstruction_lpips", "forecast_lpips")},
            "initial_candidate_mismatch_count": sum(first["candidate_index"] != second["candidate_index"]
                for first, second in zip(initial[method], initial[control_name])),
            "world_risk_breakdown": {risk: paired_recording_interval(
                [row for row in worlds[method] if row["risk_flags"].get(risk)], worlds[control_name], "forecast_lpips")
                for risk in sorted({risk for row in worlds[method] for risk, present in row["risk_flags"].items() if present})},
        }
    result = {"complete": True, "scope": "Same internal-development1024 planning scenes and256 world clips, one epoch/seed. Not navtest.",
        "configuration_sha256": digest(config_path), "frozen_configuration_sha256": digest(control_path),
        "frozen_weights_and_buffers_verified_unchanged": True,
        "frozen_representation_sha256": frozen_digest.hexdigest(),
        "planning_means": {method: metric_means(rows) for method, rows in planning.items()},
        "world_means": {method: metric_means(rows) for method, rows in worlds.items()},
        "paired_comparisons": comparisons, "reports": summaries, "measured_costs": costs,
        "interval_method": "2000 paired recording-cluster bootstrap resamples, seed20261003; no training-seed uncertainty",
        "source_sha256": source_hashes,
        "limitations": [
            "Control freezes LPWM weights, buffers and zero-output encoder FiLM and uses eval mode; contrasts adaptation including FiLM and train-mode regularization.",
            "Same planner architecture and seeded initialization, but4,480 encoder-FiLM parameters are intentionally frozen only in the control.",
            "Partial/LoRA microbatch histories and full conv_in conditioning/20epoch schedule differ; report practical comparisons, not isolated module effects.",
            "One seed/epoch and previously exposed development panels; no final method ranking or independent-test claim.",
            "Future-replacement diagnostics and whole-image LPIPS are not object-state preservation measurements.",
        ]}
    write_json(root / "adaptation_vs_frozen_summary.json", result)
    write_json(shared / "adaptation_vs_frozen_summary.json", result)
    print("FROZEN_CONTROL_COMPARISON_COMPLETE", json.dumps(result["planning_means"]), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    summarize(parser.parse_args().config.resolve())
