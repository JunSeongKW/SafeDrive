"""Bounded full LPWM adaptation using the official temporal ELBO and LPIPS loss."""
import argparse
import json
import os
import random
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
from planning_aware_future_prediction.object_centric.lpwm_bridge import ARTIFACT_ROOT, checkpoint_digest, image_tensor, load_official_lpwm


def gpu_guard(device_index):
    available = int(subprocess.check_output(["nvidia-smi", f"--id={device_index}", "--query-gpu=memory.free", "--format=csv,noheader,nounits"], text=True).strip())
    if available < 6144:
        raise RuntimeError("GPU reserve below 6 GiB; stopping our run")


def run_adaptation(arguments):
    assert arguments.gpu in (0, 1)
    torch.set_num_threads(4)
    torch.cuda.set_device(arguments.gpu)
    device = f"cuda:{arguments.gpu}"
    gpu_guard(arguments.gpu)
    torch.manual_seed(arguments.seed)
    np.random.seed(arguments.seed)
    random.seed(arguments.seed)
    os.environ["TORCH_HOME"] = str(ARTIFACT_ROOT / "torch")
    os.chdir(ARTIFACT_ROOT)
    model, official_config = load_official_lpwm(device)
    from utils.loss_functions import LossLPIPS
    reconstruction_loss = LossLPIPS(normalized_rgb=False).to(device).eval()
    manifest = json.loads((ARTIFACT_ROOT / "clip_manifest.json").read_text())
    train_records = [record for record in manifest["records"] if record["split"] == "train"]
    field = "raw_images" if arguments.variant == "raw" else "stabilized_images"
    train_clips = [np.load(PROJECT_ROOT / record["clip_path"])[field][:8] for record in train_records]
    assert train_clips and all(len(clip) == 8 for clip in train_clips)
    run_directory = ARTIFACT_ROOT / "runs" / f"{arguments.variant}_seed{arguments.seed}"
    if arguments.smoke:
        run_directory = ARTIFACT_ROOT / "smoke" / f"{arguments.variant}_seed{arguments.seed}"
    run_directory.mkdir(parents=True, exist_ok=True)
    if (run_directory / "training_summary.json").exists():
        raise FileExistsError("Completed runs are immutable")
    optimizer = torch.optim.Adam(model.parameters(), lr=8e-5, betas=(.9, .999), eps=1e-6)
    generator = np.random.default_rng(arguments.seed)
    training_start = time.monotonic()
    initial_encoder_parameter = next(model.encoder_module.parameters()).detach().clone()
    updates = 2 if arguments.smoke else arguments.updates
    model.train()
    with (run_directory / "training_log.jsonl").open("w") as training_log:
        for update in range(updates):
            if update % 20 == 0:
                gpu_guard(arguments.gpu)
                if time.monotonic() - training_start > 2400:
                    raise RuntimeError("Per-run 40 minute wall-clock cap reached")
            selected_indices = generator.integers(len(train_clips), size=2)
            observed_training_video = torch.stack([image_tensor(train_clips[index], device) for index in selected_indices])
            optimizer.zero_grad(set_to_none=True)
            output = model(observed_training_video, with_loss=True, recon_loss_type="vgg", recon_loss_func=reconstruction_loss,
                           beta_kl=.08, beta_obj=.08, beta_dyn=.2, beta_rec=1, kl_balance=.01,
                           num_static=7 if update < 20 and not (arguments.smoke and update == 1) else 1)
            losses = output["loss_dict"]
            if not torch.isfinite(losses["loss"]):
                raise FloatingPointError("Nonfinite official ELBO")
            losses["loss"].backward()
            encoder_gradient = float(torch.stack([parameter.grad.detach().square().sum() for parameter in model.encoder_module.parameters() if parameter.grad is not None]).sum().sqrt())
            torch.nn.utils.clip_grad_norm_(model.parameters(), 100, error_if_nonfinite=True)
            optimizer.step()
            if torch.cuda.max_memory_allocated(device) > 16 * 1024**3:
                raise RuntimeError("16 GiB allocated-memory cap exceeded")
            if update % 20 == 0 or update == updates - 1:
                record = {"update": update + 1, "elapsed_seconds": time.monotonic() - training_start,
                          "encoder_gradient_norm": encoder_gradient,
                          **{key: float(value.detach()) for key, value in losses.items() if value.numel() == 1}}
                training_log.write(json.dumps(record) + "\n")
                training_log.flush()
                print(json.dumps(record), flush=True)
            del output, losses, observed_training_video
    parameter_change = float((next(model.encoder_module.parameters()).detach() - initial_encoder_parameter).abs().max())
    assert parameter_change > 0
    checkpoint_path = run_directory / "checkpoint.pt"
    torch.save(model.cpu().state_dict(), checkpoint_path)
    summary = {"variant": arguments.variant, "seed": arguments.seed, "updates": updates,
               "train_clips": len(train_clips), "batch_size": 2, "sequence_frames": 8,
               "trainable_parameters": sum(parameter.numel() for parameter in model.parameters()),
               "encoder_parameter_max_change": parameter_change, "seconds": time.monotonic() - training_start,
               "peak_allocated_gib": torch.cuda.max_memory_allocated(device) / 1024**3,
               "checkpoint_sha256": checkpoint_digest(checkpoint_path), "checkpoint": str(checkpoint_path),
               "official_elbo": True, "reconstruction": "official pixel MSE + 0.1 LPIPS", "warmup_static_frames": 7,
               "warmup_updates": min(20, updates), "optimizer": "Adam lr=8e-5, eps=1e-6, clip_grad_norm=100"}
    (run_directory / "training_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print("TRAINING_DONE", json.dumps(summary), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", choices=("raw", "rotation_stabilized"), required=True)
    parser.add_argument("--seed", type=int, default=29)
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--updates", type=int, default=300)
    parser.add_argument("--smoke", action="store_true")
    run_adaptation(parser.parse_args())
