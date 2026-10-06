"""Evaluate the geometry-adapted model with the existing final benchmark protocol."""
import argparse
from pathlib import Path

import infer_lpwm_drivor_lora as inference
from planning_aware_future_prediction.object_centric.lpwm_drivor_geometry_lora import LPWMDrivoRGeometryLoRAModel


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--split", required=True, choices=["navtest", "warmup_two_stage", "navhard_two_stage"])
    inference.LPWMDrivoRLoRAModel = LPWMDrivoRGeometryLoRAModel
    inference.run(parser.parse_args())
