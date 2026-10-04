"""Process-local bindings preserve registered models/trainers and override resource guards."""
import argparse
import json
from pathlib import Path

import train_lpwm_partial_planning as partial_training
import evaluate_lpwm_partial_planning as evaluation
from lpwm_48gb_execution import verify_execution_configuration, check_gpu_limits
from planning_aware_future_prediction.object_centric.lpwm_partial_finetuning import (
    build_partial_planning_model, parameter_inventory)
from planning_aware_future_prediction.object_centric.lpwm_lora_finetuning import build_lora_planning_model, lora_parameter_inventory
from planning_aware_future_prediction.object_centric.lpwm_adapter_full_finetuning import (
    build_adapter_or_full_planning_model, adaptation_parameter_inventory)

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def run(arguments):
    config = json.loads(arguments.config.read_text())
    verify_execution_configuration(config, PROJECT_ROOT)
    method = config["adaptation_method"]
    builders = {"partial_output_layers": (build_partial_planning_model, parameter_inventory),
        "attention_lora": (build_lora_planning_model, lora_parameter_inventory),
        "residual_adapter": (build_adapter_or_full_planning_model, adaptation_parameter_inventory),
        "full_low_learning_rate": (build_adapter_or_full_planning_model, adaptation_parameter_inventory)}
    builder, inventory = builders[method]
    if arguments.action == "train":
        training = partial_training
        if method == "full_low_learning_rate":
            import train_lpwm_resumed_full_planning as training
        training.build_planning_model = builder
        training.parameter_inventory = inventory
        training.check_gpu_reserve = check_gpu_limits
        training.run(arguments)
    else:
        evaluation.build_partial_planning_model = builder
        evaluation.check_gpu_reserve = check_gpu_limits
        evaluation.evaluate(arguments)


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
