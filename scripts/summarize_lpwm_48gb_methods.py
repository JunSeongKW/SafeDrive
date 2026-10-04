"""Paired development outcomes and cumulative cost for four adaptation strategies."""
import argparse
from itertools import combinations
import json
from pathlib import Path

from evaluate_lpwm_partial_planning import paired_metrics
from evaluate_lpwm_full_planning import digest, write_json

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def summarize(config_path):
    specification = json.loads(config_path.read_text())
    root = PROJECT_ROOT / specification["output_directory"]
    configurations = {method["method"]: Path(json.loads((root / (method["method"] + "_execution_selection.json")).read_text())["selected_configuration"]) for method in specification["methods"]}
    reports, predictions, worlds, costs = {}, {}, {}, {}
    initial_rows = {}
    for method, path in configurations.items():
        config = json.loads(path.read_text())
        directory = PROJECT_ROOT / config["output_directory"] / "metric_plus_world"
        report = json.loads((directory / "trend_evaluation/summary.json").read_text())
        assert not report["engineering_only"] and report["configuration_sha256"] == digest(path)
        training = json.loads((directory / "training_summary.json").read_text())
        assert training["completed_updates"] == 4707 and not training["profile_only"]
        inventory = json.loads((directory / "parameter_inventory.json").read_text())
        reports[method] = report
        predictions[method] = json.loads((directory / "trend_evaluation/trained.json").read_text())
        worlds[method] = json.loads((directory / "trend_evaluation/world.json").read_text())
        initial_rows[method] = json.loads((directory / "trend_evaluation/initial.json").read_text())
        costs[method] = {name: training[name] for name in ("seconds", "peak_allocated_gib", "completed_updates")}
        costs[method].update({name: inventory[name] for name in
            ("trainable_world_parameters", "trainable_planner_parameters", "trainable_parameters")})
        costs[method].update(microbatch_size_per_gpu=config["microbatch_size_per_gpu"],
            gradient_accumulation=config["gradient_accumulation"], lpwm_learning_rate=config["lpwm_learning_rate"],
            starting_completed_updates=training.get("starting_completed_updates", 0),
            additional_updates_this_execution=training.get("additional_updates_this_execution", 4707))
    comparisons = {}
    for first, second in combinations(configurations, 2):
        assert [row["token"] for row in predictions[first]] == [row["token"] for row in predictions[second]]
        assert [row["token"] for row in worlds[first]] == [row["token"] for row in worlds[second]]
        comparisons[second + "_minus_" + first] = {
            "planning": paired_metrics(predictions[second], predictions[first], ("pdms", "ade_meters", "fde_meters", "metric_bce")),
            "world": paired_metrics(worlds[second], worlds[first], ("reconstruction_lpips", "forecast_lpips"))}
    result = {"complete": True, "configuration_sha256": digest(config_path), "reports": reports,
        "paired_comparisons": comparisons, "measured_costs": costs,
        "scope": "Four practical adaptation configurations; previously exposed development panels, one cumulative epoch each.",
        "full_method_continuation": "2095 previous updates reused plus2612 new updates, native conv_in command and original20epoch scheduler preserved.",
        "object_gt_auxiliary": "off in every condition; future object-supervision experiment remains deferred",
        "limitations": specification["limitations"],
        "interpretation": "Report PDMS/ADE, world retention, costs and intervals together; a method/configuration association is not a causal isolation of PEFT choice."}
    write_json(root / "four_method_summary.json", result)
    write_json(PROJECT_ROOT / specification["shared_results_directory"] / "four_method_summary.json", result)
    print("FOUR_METHOD_COMPARISON_COMPLETE", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    summarize(parser.parse_args().config.resolve())
