"""Run exactly the same registered development evaluations for LoRA."""
import argparse
from pathlib import Path

import evaluate_lpwm_partial_planning as evaluation
from planning_aware_future_prediction.object_centric.lpwm_lora_finetuning import build_lora_planning_model


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--condition", default="metric_plus_world")
    parser.add_argument("--gpu", type=int, choices=(0, 1), default=0)
    parser.add_argument("--engineering-checkpoint", type=Path)
    arguments = parser.parse_args()
    arguments.config = arguments.config.resolve()
    evaluation.build_partial_planning_model = build_lora_planning_model
    evaluation.evaluate(arguments)
