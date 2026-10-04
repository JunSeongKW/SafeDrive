"""Paired planning/world retention and measured cost of partial tuning vs LoRA."""
import argparse
import json
from pathlib import Path

from evaluate_lpwm_partial_planning import paired_metrics
from evaluate_lpwm_full_planning import digest, write_json

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def summarize(config_path):
    specification = json.loads(config_path.read_text())
    root = PROJECT_ROOT / specification["output_directory"]
    selected = json.loads((root / "lora_execution_selection.json").read_text())
    configurations = {"partial_output_layers": PROJECT_ROOT / specification["partial_config"],
        "attention_lora": Path(selected["selected_configuration"])}
    reports, predictions, initial_predictions, world_predictions, costs = {}, {}, {}, {}, {}
    for method, path in configurations.items():
        config = json.loads(path.read_text())
        condition_root = PROJECT_ROOT / config["output_directory"] / "metric_plus_world"
        reports[method] = json.loads((condition_root / "trend_evaluation/summary.json").read_text())
        assert not reports[method]["engineering_only"]
        assert reports[method]["configuration_sha256"] == digest(path)
        predictions[method] = json.loads((condition_root / "trend_evaluation/trained.json").read_text())
        initial_predictions[method] = json.loads((condition_root / "trend_evaluation/initial.json").read_text())
        world_predictions[method] = json.loads((condition_root / "trend_evaluation/world.json").read_text())
        training = json.loads((condition_root / "training_summary.json").read_text())
        assert training["completed_updates"] == 4707 and not training["profile_only"]
        inventory = json.loads((condition_root / "parameter_inventory.json").read_text())
        costs[method] = {name: training[name] for name in ("seconds", "peak_allocated_gib", "trainable_parameters", "completed_updates")}
        costs[method].update({name: inventory[name] for name in ("trainable_world_parameters", "trainable_planner_parameters")})
        costs[method]["microbatch_size_per_gpu"] = config["microbatch_size_per_gpu"]
        costs[method]["gradient_accumulation"] = config["gradient_accumulation"]
    partial, lora = "partial_output_layers", "attention_lora"
    assert [row["token"] for row in predictions[partial]] == [row["token"] for row in predictions[lora]]
    assert [row["token"] for row in world_predictions[partial]] == [row["token"] for row in world_predictions[lora]]
    initial_max_difference = 0.
    for first, second in zip(initial_predictions[partial], initial_predictions[lora]):
        assert first["token"] == second["token"]
        assert first["candidate_index"] == second["candidate_index"]
        for name in ("ade_meters", "fde_meters", "pdms", "metric_bce"):
            first_value, second_value = first["metrics"][name], second["metrics"][name]
            if first_value is None or second_value is None:
                assert first_value is second_value
            else:
                initial_max_difference = max(initial_max_difference, abs(first_value - second_value))
    assert initial_max_difference <= 1e-5, initial_max_difference
    planning = paired_metrics(predictions[lora], predictions[partial], ("pdms", "ade_meters", "fde_meters", "metric_bce"))
    lower, upper = planning["pdms"]["ci95"]
    if lower is not None and lower > 0 and reports[lora]["trend_checks_passed"]:
        decision = "attention_lora_promising_requires_replication"
    elif upper is not None and upper < 0 and reports[partial]["trend_checks_passed"]:
        decision = "partial_output_layers_promising_requires_replication"
    else:
        decision = "method_advantage_not_established"
    report = {"complete": True, "configuration_sha256": digest(config_path), "decision": decision,
        "initial_prediction_max_difference": initial_max_difference, "lora_minus_partial_planning": planning,
        "lora_minus_partial_world": paired_metrics(world_predictions[lora], world_predictions[partial],
            ("reconstruction_lpips", "forecast_lpips")), "reports": reports, "measured_costs": costs,
        "object_auxiliary_supervision": "off in both methods; object GT experiment deferred",
        "limitations": specification["limitations"] + ["Timing measured at different shared-GPU loads",
            "Different internal adaptation locations; not an isolated comparison of low-rank vs unrestricted updates at identical layers"]}
    write_json(root / "method_comparison_summary.json", report)
    write_json(PROJECT_ROOT / specification["shared_results_directory"] / "method_comparison_summary.json", report)
    print("ADAPTATION_METHOD_COMPARISON_DONE", decision, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    summarize(parser.parse_args().config.resolve())
