"""Capture real DDP gradients while profiling a saved primary checkpoint."""
import argparse
import os
from pathlib import Path

import torch
import train_lpwm_drivor_optimized_execution as training


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--execution", type=Path, required=True)
    parser.add_argument("--resume-checkpoint", type=Path, required=True)
    parser.add_argument("--benchmark-output", type=Path, required=True)
    parser.add_argument("--benchmark-updates", type=int, default=2)
    options = parser.parse_args()
    options.config = options.config.resolve()
    options.engineering_updates = 0
    original_step = torch.optim.AdamW.step
    captured = []

    def record_first_gradients(optimizer, *arguments, **keywords):
        if not captured:
            gradients = [parameter.grad.detach().cpu().clone()
                         for group in optimizer.param_groups for parameter in group["params"]
                         if parameter.grad is not None]
            torch.save(gradients, options.benchmark_output / f"rank{os.environ['RANK']}_first_gradients.pt")
            captured.append(True)
        return original_step(optimizer, *arguments, **keywords)

    torch.optim.AdamW.step = record_first_gradients
    training.run(options)
