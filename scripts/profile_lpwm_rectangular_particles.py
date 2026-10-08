"""Measure actual 512x256 LPWM SSL updates for reduced particle counts."""
import argparse
import json
import os
from pathlib import Path
import pickle
import sys
import time

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from planning_aware_future_prediction.object_centric.lpwm_rectangular import load_rectangular_lpwm
from planning_aware_future_prediction.object_centric.lpwm_bridge import ARTIFACT_ROOT


def main(arguments):
    torch.set_num_threads(2)
    torch.manual_seed(47)
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    torch.cuda.set_per_process_memory_fraction(44_000_000_000 / torch.cuda.get_device_properties(device).total_memory)
    log = sorted((ROOT / "dataset/navsim_logs/trainval").glob("*.pkl"))[0]
    with log.open("rb") as stream:
        frames = pickle.load(stream)
    paths = [ROOT / "dataset/sensor_blobs/trainval" / frame["cams"]["CAM_F0"]["data_path"] for frame in frames]
    exists = [path.is_file() for path in paths]
    start = next(index for index in range(len(frames) - 7) if all(exists[index:index + 8]))
    images = []
    for path in paths[start:start + 8]:
        rgb = cv2.cvtColor(cv2.imread(str(path)), cv2.COLOR_BGR2RGB)
        images.append(cv2.resize(rgb[28:-28], (512, 256), interpolation=cv2.INTER_LINEAR))
    video = torch.from_numpy(np.stack(images)).permute(0, 3, 1, 2).unsqueeze(0).float().to(device) / 255
    model, configuration = load_rectangular_lpwm(device, foreground_particles=arguments.particles)
    model.train()
    os.environ["TORCH_HOME"] = str(ARTIFACT_ROOT / "torch")
    os.chdir(ARTIFACT_ROOT)
    from utils.loss_functions import LossLPIPS
    reconstruction = LossLPIPS(normalized_rgb=False).to(device).eval()
    reconstruction.requires_grad_(False)
    optimizer = torch.optim.Adam(model.parameters(), lr=8e-5, eps=1e-6)
    measurements = []
    for update in range(arguments.updates):
        optimizer.zero_grad(set_to_none=True)
        torch.cuda.synchronize()
        started = time.perf_counter()
        output = model(video, deterministic=False, with_loss=True, warmup=False, num_static=1,
            recon_loss_func=reconstruction, recon_loss_type="vgg", beta_kl=.08, beta_obj=.08,
            beta_dyn=.2, beta_rec=.125, beta_dyn_rec=1., kl_balance=.01)
        loss = output["loss_dict"]["loss"]
        torch.cuda.synchronize()
        forward_seconds = time.perf_counter() - started
        loss.backward()
        torch.cuda.synchronize()
        backward_seconds = time.perf_counter() - started - forward_seconds
        gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 100, error_if_nonfinite=True)
        optimizer.step()
        torch.cuda.synchronize()
        groups = {"encoder": model.encoder_module, "context": model.ctx_module,
                  "dynamics": model.dyn_module, "decoder": model.decoder_module}
        norms = {name: float(torch.stack([parameter.grad.detach().float().square().sum()
            for parameter in module.parameters() if parameter.grad is not None]).sum().sqrt())
            for name, module in groups.items()}
        measured = {"update": update + 1, "seconds": time.perf_counter() - started,
            "forward_seconds": forward_seconds, "backward_seconds": backward_seconds,
            "loss": float(loss.detach()), "gradient_norm": float(gradient_norm), "gradient_groups": norms,
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(), "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
            "reconstruction_shape": list(output["rec_rgb"].shape), "particle_shape": list(output["z"].shape)}
        assert all(np.isfinite(value) and value > 0 for value in norms.values())
        assert output["z"].shape[-2] == arguments.particles
        assert tuple(output["rec_rgb"].shape[-2:]) == (256, 512)
        measurements.append(measured)
        print(json.dumps(measured), flush=True)
        del output, loss
    destination = arguments.output
    destination.mkdir(parents=True, exist_ok=True)
    result = {"passed": True, "foreground_particles": arguments.particles, "configuration": configuration,
        "profile_only_weights_discarded": True, "batch_clips": 1, "frames_per_clip": 8,
        "precision": "float32", "input_log": log.name, "input_start": start, "measurements": measurements,
        "median_seconds_excluding_warmup": float(np.median([row["seconds"] for row in measurements[1:]]))}
    (destination / f"profile_particles{arguments.particles}.json").write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--particles", type=int, choices=(8, 16, 32, 64), required=True)
    parser.add_argument("--updates", type=int, default=3)
    parser.add_argument("--output", type=Path, default=ROOT / "results/lpwm_driving_video_512x256_v1/reduced_particle_profiles")
    main(parser.parse_args())
