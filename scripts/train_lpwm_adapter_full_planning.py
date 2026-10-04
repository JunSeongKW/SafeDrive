"""Use the unchanged registered trainer with adapter/full LPWM builders."""
import argparse
from pathlib import Path

import train_lpwm_partial_planning as training
from planning_aware_future_prediction.object_centric.lpwm_adapter_full_finetuning import (
    build_adapter_or_full_planning_model, adaptation_parameter_inventory)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--condition", default="metric_plus_world")
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--resume", action="store_true")
    arguments = parser.parse_args()
    arguments.config = arguments.config.resolve()
    import json
    if json.loads(arguments.config.read_text())["adaptation_method"] == "full_low_learning_rate":
        import train_lpwm_resumed_full_planning as training
    training.build_planning_model = build_adapter_or_full_planning_model
    training.parameter_inventory = adaptation_parameter_inventory
    training.run(arguments)
