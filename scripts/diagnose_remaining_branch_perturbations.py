"""Complete B/D input-amplitude table using original checkpoints, CPU only."""

import json
from pathlib import Path

import torch
import train_target_supervision_ablation as exploration
from diagnose_visual_future_prediction_variance import (
    load_existing_experiment,
    perturbation_amplitudes,
)


def main():
    torch.set_num_threads(4)
    device = torch.device("cpu")
    _configuration, windows, _normalization, original_directory = (
        load_existing_experiment(device)
    )
    development_windows = [
        window for window in windows if window["metadata"]["split"] == "development"
    ]
    report = {
        "device": "cpu",
        "training_updates": 0,
        "conditions": {},
        "source_sha256": exploration.file_sha256(Path(__file__)),
        "helper_sha256": exploration.file_sha256(
            Path(__file__).with_name("diagnose_visual_future_prediction_variance.py")
        ),
    }
    for label in ("B", "D"):
        model = exploration.FixedDistanceFutureSupervisionPilot().eval()
        checkpoint_path = original_directory / label / "last_update_200.pt"
        checkpoint = torch.load(checkpoint_path, weights_only=True, map_location="cpu")
        model.load_state_dict(checkpoint["model"], strict=True)
        report["conditions"][label] = {
            "checkpoint_sha256": exploration.file_sha256(checkpoint_path),
            "perturbations": perturbation_amplitudes(
                model, development_windows, device
            ),
        }
    output_path = (
        exploration.PROJECT_ROOT
        / "results/future_prediction_diagnostics/remaining_branch_perturbations_v1.json"
    )
    if output_path.exists():
        raise RuntimeError("refusing overwrite")
    output_path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
