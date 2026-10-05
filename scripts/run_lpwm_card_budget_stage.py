"""Run immutable LPWM trainers with a dynamically derived shared-card memory cap."""
import argparse
import json
from pathlib import Path

import run_lpwm_48gb_stage as original_stage
from lpwm_card_budget_execution import check_card_budget, install_allocator_budget


def run(arguments):
    configuration = json.loads(arguments.config.read_text())
    original_stage.check_gpu_limits = check_card_budget
    if arguments.action == "train":
        install_allocator_budget(configuration, arguments.profile)
    original_stage.run(arguments)


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
