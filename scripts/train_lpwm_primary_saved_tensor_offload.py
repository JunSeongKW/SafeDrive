"""Preserve primary batch16 training while storing saved autograd tensors on CPU."""
import argparse
import os
from pathlib import Path

import torch
import train_lpwm_drivor_optimized_execution as training


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--execution", type=Path, required=True)
    parser.add_argument("--resume-checkpoint", type=Path)
    parser.add_argument("--benchmark-output", type=Path)
    parser.add_argument("--benchmark-updates", type=int, default=0)
    parser.add_argument("--engineering-updates", type=int, default=0)
    options = parser.parse_args()
    options.config = options.config.resolve()
    if options.benchmark_updates:
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
    with torch.autograd.graph.save_on_cpu(pin_memory=True):
        training.run(options)
