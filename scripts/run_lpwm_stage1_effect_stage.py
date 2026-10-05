"""Train one public frozen LPWM planner using the completed control's recipe."""
import argparse
import json
import os
from pathlib import Path

import evaluate_lpwm_partial_planning as evaluation
import train_lpwm_frozen_control as training
from evaluate_lpwm_full_planning import digest, write_json
from lpwm_stage1_effect_protocol import (
    PROJECT_ROOT, load_stage1_effect_inputs, verify_stage1_effect_configuration)
from lpwm_measured_card_budget_execution import install_allocator_budget, check_card_budget
from planning_aware_future_prediction.object_centric.lpwm_frozen_control import build_frozen_control_model
from summarize_lpwm_posttraining import paired_recording_interval, metric_means


def run(arguments):
    specification = verify_stage1_effect_configuration(arguments.config)
    contrast = specification["stage1_effect"]
    training.load_training_inputs = load_stage1_effect_inputs
    evaluation.load_training_inputs = load_stage1_effect_inputs
    adapted = contrast["representation_condition"] == "navsim_posttrained"
    directory = PROJECT_ROOT / specification["output_directory"] / arguments.condition
    if arguments.action == "train":
        install_allocator_budget(specification, arguments.profile)
        training.check_gpu_reserve = check_card_budget
        training.run(arguments)
        if int(os.environ["RANK"]) == 0:
            path = directory / ("profile" if arguments.profile else "") / "training_summary.json"
            if path.exists():
                summary = json.loads(path.read_text())
                # The unchanged trainer uses this historical field name for its
                # start checkpoint. Initialization provenance is explicit below.
                assert summary["stage1_checkpoint_sha256"] == contrast["initial_lpwm_checkpoint_sha256"]
                summary.update(representation_condition=contrast["representation_condition"],
                    initial_lpwm_checkpoint_sha256=contrast["initial_lpwm_checkpoint_sha256"],
                    navsim_stage1_performed=adapted,
                    legacy_stage1_checkpoint_field="Alias for initial_lpwm_checkpoint_sha256; public condition never loads adapted LPWM weights.")
                write_json(path, summary)
                write_json(PROJECT_ROOT / specification["shared_results_directory"] /
                    (("profile_" if arguments.profile else "") + arguments.condition + "_training_summary.json"), summary)
    else:
        evaluation.build_partial_planning_model = build_frozen_control_model
        evaluation.check_gpu_reserve = check_card_budget
        evaluation.evaluate(arguments)
        engineering = arguments.engineering_checkpoint is not None
        destination = directory / ("engineering_evaluation" if engineering else "trend_evaluation")
        path = destination / "summary.json"
        summary = json.loads(path.read_text())
        rows = json.loads((destination / "world.json").read_text())
        initial_reference = json.loads((PROJECT_ROOT / contrast["initial_world_evaluation"]).read_text())["records"]
        reference_by_token = {row["token"]: row for row in initial_reference}
        matched_reference = [reference_by_token[row["token"]] for row in rows]
        for row, reference in zip(rows, matched_reference):
            assert row["recording_group"] == reference["recording_group"]
        retained = {metric: paired_recording_interval(rows, matched_reference, metric)
            for metric in ("reconstruction_lpips", "forecast_lpips")}
        baseline = metric_means(matched_reference)
        summary["checks"].update({metric: interval["ci95"][1] <= .1 * baseline[metric]
            for metric, interval in retained.items()})
        summary["trend_checks_passed"] = all(summary["checks"].values())
        summary.update(experiment_kind="public_frozen_lpwm_stage1_effect",
            stage1_effect=contrast, navsim_stage1_performed=adapted,
            world_vs_initial_representation=retained,
            initial_world_reference_sha256=digest(PROJECT_ROOT / contrast["initial_world_evaluation"]),
            world_risk_reference="world_risk_breakdown and world_vs_stage1 compare with NAVSIM-adapted Stage1; retention checks use each condition's initial checkpoint baseline.")
        summary["limitations"] = [
            "One seed/epoch and previously exposed development panels; not navtest or convergence.",
            "LPWM including buffers and encoder-command FiLM remains frozen in eval mode; only the fresh same-seed planner trains.",
            "The public condition reuses NAVSIM caches/preprocessing only and never loads NAVSIM-adapted LPWM weights.",
            "The completed frozen Stage1 planner is reused as the comparison; GPU batch8/effective16 and the same training recipe are retained.",
            "Cached official PDM scores evaluate selected unchanged vocabulary candidates.",
            "Whole-image LPIPS and inference future replacement do not establish semantic object preservation.",
        ]
        write_json(path, summary)
        write_json(PROJECT_ROOT / specification["shared_results_directory"] /
            (("engineering_" if engineering else "") + arguments.condition + "_trend_summary.json"), summary)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--action", choices=("train", "evaluate"), required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--condition", default="metric_plus_world")
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--gpu", type=int, choices=(0, 1), default=0)
    parser.add_argument("--engineering-checkpoint", type=Path)
    arguments = parser.parse_args()
    arguments.config = arguments.config.resolve()
    run(arguments)
