"""Use the registered Stage2 trainer with a separate LoRA model builder."""
import argparse
from pathlib import Path

import train_lpwm_partial_planning as training
from planning_aware_future_prediction.object_centric.lpwm_lora_finetuning import (
    build_lora_planning_model, lora_parameter_inventory)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--condition", default="metric_plus_world")
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--resume", action="store_true")
    arguments = parser.parse_args()
    arguments.config = arguments.config.resolve()
    training.build_planning_model = build_lora_planning_model
    training.parameter_inventory = lora_parameter_inventory
    training.run(arguments)
