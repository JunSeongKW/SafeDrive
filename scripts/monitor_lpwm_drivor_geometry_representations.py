"""Use the same fixed-panel diagnostics for the separate geometry-LoRA condition."""
import argparse
from pathlib import Path

import monitor_lpwm_drivor_representations as monitoring
from planning_aware_future_prediction.object_centric.lpwm_drivor_geometry_lora import LPWMDrivoRGeometryLoRAModel


def configure_monitor():
    monitoring.OUTPUT = monitoring.ROOT / "outputs/lpwm_drivor_geometry_representation_monitor_v1"
    monitoring.TRAINING = monitoring.ROOT / "outputs/lpwm_drivor_geometry_lora_v1/navsim_v1"
    monitoring.LPWMDrivoRLoRAModel = LPWMDrivoRGeometryLoRAModel
    # The inherited watcher must spawn this entrypoint, not the older model.
    monitoring.__file__ = str(Path(__file__).resolve())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--device", default="cuda", choices=["cpu", "cuda"])
    parser.add_argument("--scene-limit", type=int, default=0)
    parser.add_argument("--watch", action="store_true")
    arguments = parser.parse_args()
    configure_monitor()
    if arguments.watch:
        monitoring.watch()
    else:
        monitoring.evaluate_checkpoint(arguments)
