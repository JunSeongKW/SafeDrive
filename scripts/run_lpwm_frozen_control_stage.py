"""Bind dedicated frozen-control training to the unchanged evaluation protocol."""
import argparse
import json
from pathlib import Path

import evaluate_lpwm_partial_planning as evaluation
import train_lpwm_frozen_control as training
from evaluate_lpwm_full_planning import write_json
from lpwm_frozen_control_protocol import verify_control_configuration, PROJECT_ROOT
from lpwm_measured_card_budget_execution import install_allocator_budget, check_card_budget
from planning_aware_future_prediction.object_centric.lpwm_frozen_control import build_frozen_control_model


def run(arguments):
    specification = verify_control_configuration(arguments.config)
    if arguments.action == "train":
        install_allocator_budget(specification, arguments.profile)
        training.check_gpu_reserve = check_card_budget
        training.run(arguments)
    else:
        evaluation.build_partial_planning_model = build_frozen_control_model
        evaluation.check_gpu_reserve = check_card_budget
        evaluation.evaluate(arguments)
        directory = PROJECT_ROOT / specification["output_directory"] / arguments.condition
        engineering = arguments.engineering_checkpoint is not None
        path = directory / ("engineering_evaluation" if engineering else "trend_evaluation") / "summary.json"
        summary = json.loads(path.read_text())
        summary.update(experiment_kind="fixed_representation_planner_control", frozen_control=specification["frozen_control"])
        summary["limitations"] = [
            "One seed/epoch; previously exposed internal development, not independent test.",
            "Compared with Adapter/LoRA, this control freezes both LPWM adaptation and encoder-command FiLM.",
            "Frozen LPWM uses evaluation mode during training; adaptation runs use train mode.",
            "World SSL is retained as a detached monitor; it cannot update the fixed representation or planner.",
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
