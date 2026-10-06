"""Run the sealed DDP training loop with the explicitly extended geometry model."""
import argparse
from pathlib import Path

import train_lpwm_drivor_lora_parallel as training
from planning_aware_future_prediction.object_centric.lpwm_drivor_geometry_lora import LPWMDrivoRGeometryLoRAModel


def run(arguments):
    # A separate entrypoint preserves the registered previous experiment source.
    training.LPWMDrivoRLoRAModel = LPWMDrivoRGeometryLoRAModel
    arguments.config = arguments.config.resolve()
    training.run(arguments)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--execution", type=Path, required=True)
    parser.add_argument("--engineering-updates", type=int, default=0)
    run(parser.parse_args())
