"""Run the original DDP protocol with geometry, appearance and future LoRA."""
import argparse
import json
from pathlib import Path

import train_lpwm_drivor_lora_parallel as training
from planning_aware_future_prediction.object_centric.lpwm_drivor_planning_path_lora import LPWMDrivoRPlanningPathLoRAModel


def run(arguments):
    # A separate entrypoint preserves the registered previous experiment source.
    training.LPWMDrivoRLoRAModel = LPWMDrivoRPlanningPathLoRAModel
    arguments.config = arguments.config.resolve()
    if not arguments.engineering_updates:
        configuration = json.loads(arguments.config.read_text())
        registration = json.loads((training.PROJECT_ROOT / configuration["registration"]).read_text())
        for filename, expected in registration["immutable_inputs"].items():
            assert training.digest(training.PROJECT_ROOT / filename) == expected, filename
    training.run(arguments)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--execution", type=Path, required=True)
    parser.add_argument("--engineering-updates", type=int, default=0)
    run(parser.parse_args())
