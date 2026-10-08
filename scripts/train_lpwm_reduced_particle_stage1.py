"""Train native-resolution SSL candidates and a selected OpenScene preparation run.

This does not claim to reproduce Drive-JEPA's unavailable 330-hour clip manifest.
No planning labels enter the loss. Validation records are excluded from training.
"""
import argparse
import ctypes
import hashlib
import json
import os
from pathlib import Path
import random
import signal
import subprocess
import sys
import time

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from planning_aware_future_prediction.object_centric.lpwm_rectangular import load_rectangular_lpwm
from planning_aware_future_prediction.object_centric.lpwm_bridge import ARTIFACT_ROOT

STOP_REQUESTED = False


def request_stop(*_arguments):
    global STOP_REQUESTED
    STOP_REQUESTED = True


def write_json(path, content):
    pending = path.with_suffix(path.suffix + ".pending")
    pending.write_text(json.dumps(content, indent=2, allow_nan=False) + "\n")
    pending.replace(path)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def card_used_bytes():
    physical_gpu = os.environ["CUDA_VISIBLE_DEVICES"]
    assert physical_gpu in ("0", "1"), physical_gpu
    return int(subprocess.check_output(["nvidia-smi", "--id=" + physical_gpu,
        "--query-gpu=memory.used", "--format=csv,noheader,nounits"], text=True).strip()) * 1024**2


def load_clip(paths):
    images = []
    for path in paths:
        image = cv2.imread(path)
        if image is None:
            raise FileNotFoundError(path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        images.append(cv2.resize(image[28:-28], (512, 256), interpolation=cv2.INTER_LINEAR))
    return torch.from_numpy(np.stack(images)).permute(0, 3, 1, 2).contiguous()


class DrivingClips(Dataset):
    def __init__(self, records):
        self.records = records

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        return load_clip(self.records[index]["paths"])


def prepare_records(manifest, validation_count):
    episodes = [json.loads(line) for line in manifest.read_text().splitlines()]
    training, validation_candidates = [], {}
    for episode in episodes:
        for start in range(0, episode["frame_count"] - 7, 8):
            record = {"recording": episode["recording"], "log": episode["log"],
                "start": start, "paths": episode["frame_paths"][start:start + 8]}
            if episode["split"] == "train":
                training.append(record)
            else:
                validation_candidates.setdefault(record["recording"], []).append(record)
    generator = random.Random(47)
    generator.shuffle(training)
    recordings = sorted(validation_candidates)
    generator.shuffle(recordings)
    validation = [generator.choice(validation_candidates[recording]) for recording in recordings[:validation_count]]
    assert not {row["recording"] for row in training} & {row["recording"] for row in validation}
    assert len(validation) == validation_count
    return training, validation


@torch.no_grad()
def evaluate(model, validation, reconstruction, output, label):
    model.eval()
    rows, visual = [], []
    for index, record in enumerate(validation):
        clip = load_clip(record["paths"])[None].to("cuda").float() / 255
        encoded = model.encode_all(clip, deterministic=True)
        decoded = model.decode_all(encoded["z"], encoded["z_scale"], encoded["z_features"],
            encoded["obj_on"], encoded["z_depth"], encoded["z_bg_features"],
            z_ctx=encoded["z_context"][:, 1:].contiguous())["rec_rgb"].reshape_as(clip)
        # Only the two observed frames are supplied: no future-posterior context.
        forecast = model.sample_from_x(clip[:, :2].contiguous(), cond_steps=2, num_steps=6,
            deterministic=True, use_all_ctx=False, n_pred_eq_gt=False)[:, 2:]
        truth = clip[:, 2:]
        persistence = clip[:, 1:2].expand_as(truth)
        assert forecast.shape == truth.shape
        row = {"recording": record["recording"], "log": record["log"], "start": record["start"],
            "reconstruction_mse": float((decoded - clip).square().mean()),
            "forecast_mse": float((forecast - truth).square().mean()),
            "persistence_mse": float((persistence - truth).square().mean()),
            "forecast_lower_half_mse": float((forecast[..., 128:, :] - truth[..., 128:, :]).square().mean()),
            "reconstruction_lpips": float(reconstruction.perceptual_loss(
                decoded.flatten(0, 1) * 2 - 1, clip.flatten(0, 1) * 2 - 1).mean()),
            "forecast_lpips": float(reconstruction.perceptual_loss(
                forecast.flatten(0, 1) * 2 - 1, truth.flatten(0, 1) * 2 - 1).mean()),
            "mean_presence": float(encoded["obj_on"].mean()),
            "position_std": float(encoded["z"].std(dim=-2).mean()),
            "appearance_std": float(encoded["z_features"].std(dim=-2).mean())}
        assert all(np.isfinite(value) for value in row.values() if isinstance(value, float))
        rows.append(row)
        if index < 3:
            visual.append({"input": clip[0, 1].cpu().numpy(), "truth": truth[0, -1].cpu().numpy(),
                "reconstruction": decoded[0, 1].cpu().numpy(), "forecast": forecast[0, -1].cpu().numpy(),
                "positions": encoded["z"][0, 1].cpu().numpy(),
                "scales": encoded["z_scale"][0, 1].sigmoid().cpu().numpy(),
                "presence": encoded["obj_on"][0, 1].cpu().numpy()})
    statistics = {key: float(np.mean([row[key] for row in rows])) for key in rows[0]
                  if isinstance(rows[0][key], float)}
    report = {"label": label, "mean": statistics, "records": rows, "future_gt_used_as_input": False,
        "planning_performance_verified": False, "object_detection_or_small_object_preservation_verified": False}
    write_json(output / (label + ".json"), report)
    np.savez_compressed(output / (label + "_visuals.npz"),
        **{f"scene{index}_{key}": value for index, scene in enumerate(visual) for key, value in scene.items()})
    render_validation(output, label, visual)
    return report


def render_validation(output, label, scenes):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plot
    from matplotlib.patches import Rectangle
    figure, axes = plot.subplots(len(scenes), 4, figsize=(16, 3 * len(scenes)))
    for row, scene in enumerate(scenes):
        for column, (key, title) in enumerate((("input", "Observed + particles"),
                ("reconstruction", "Reconstruction"), ("truth", "Future GT (+3s)"), ("forecast", "Causal forecast (+3s)"))):
            axis = axes[row, column]
            axis.imshow(scene[key].transpose(1, 2, 0).clip(0, 1))
            axis.set_title(title)
            axis.axis("off")
        for position, scale, presence in zip(scene["positions"], scene["scales"], scene["presence"]):
            center = (position[::-1] + 1) * np.array([256, 128]) - .5
            size = scale[::-1] * np.array([512, 256])
            axes[row, 0].scatter(*center, s=5 + 30 * float(np.asarray(presence).reshape(-1)[0]), c="orange")
            axes[row, 0].add_patch(Rectangle(center - size / 2, *size, fill=False, edgecolor="orange", linewidth=.7))
    figure.suptitle(f"{label}: LPWM glimpse boxes, not object detections")
    figure.tight_layout()
    figure.savefig(output / (label + ".png"), dpi=140)
    plot.close(figure)


def save_checkpoint(output, model, optimizer, completed, configuration, training_hash):
    pending = output / "latest.pending.pt"
    torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(), "completed_updates": completed,
        "configuration": configuration, "training_order_sha256": training_hash,
        "torch_rng": torch.get_rng_state(), "cuda_rng": torch.cuda.get_rng_state(),
        "numpy_rng": np.random.get_state(), "python_rng": random.getstate()}, pending)
    pending.replace(output / "latest.pt")


def main(arguments):
    output = arguments.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    ctypes.CDLL(None).prctl(15, b"kjs-lpwm-stage1", 0, 0, 0)
    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)
    torch.set_num_threads(2)
    cv2.setNumThreads(1)
    torch.manual_seed(47)
    np.random.seed(47)
    random.seed(47)
    torch.cuda.set_device(0)
    available = min(42_000_000_000, 48_000_000_000 - card_used_bytes() - 2_000_000_000)
    assert available > 20_000_000_000, "Wait for at least 20GB allocator budget"
    torch.cuda.set_per_process_memory_fraction(available / torch.cuda.get_device_properties(0).total_memory)
    training, validation = prepare_records(arguments.manifest, arguments.validation_clips)
    training_hash = hashlib.sha256(json.dumps(training).encode()).hexdigest()
    configuration = vars(arguments).copy()
    configuration = {key: str(value) if isinstance(value, Path) else value for key, value in configuration.items()}
    configuration.update({"train_clip_count": len(training), "train_validation_recording_overlap": 0,
        "training_order_sha256": training_hash, "manifest_sha256": digest(arguments.manifest),
        "source_sha256": {str(path.relative_to(ROOT)): digest(path) for path in
            (Path(__file__), ROOT / "src/planning_aware_future_prediction/object_centric/lpwm_rectangular.py")},
        "loss": {"name": "official_temporal_ELBO", "beta_kl": .08, "beta_obj": .08, "beta_dyn": .2,
                 "beta_rec": .125, "beta_dyn_rec": 1., "kl_balance": .01},
        "learning_rate": 8e-5, "precision": "float32", "foreground_particles": arguments.particles,
        "background_particles": 1, "effective_batch_clips": arguments.accumulation,
        "full_330h_training": False, "object_gt_supervision": False, "allocator_budget_bytes": available})
    if (output / "registration.json").exists():
        previous = json.loads((output / "registration.json").read_text())
        assert previous["source_sha256"] == configuration["source_sha256"], "Registered source changed"
        assert previous["training_order_sha256"] == training_hash
    else:
        write_json(output / "registration.json", configuration)
    model, _ = load_rectangular_lpwm("cuda", foreground_particles=arguments.particles)
    optimizer = torch.optim.Adam(model.parameters(), lr=8e-5, eps=1e-6)
    os.environ["TORCH_HOME"] = str(ARTIFACT_ROOT / "torch")
    os.chdir(ARTIFACT_ROOT)
    from utils.loss_functions import LossLPIPS
    reconstruction = LossLPIPS(normalized_rgb=False).to("cuda").eval()
    reconstruction.requires_grad_(False)
    completed = 0
    if (output / "latest.pt").exists():
        saved = torch.load(output / "latest.pt", map_location="cpu", weights_only=False)
        assert saved["training_order_sha256"] == training_hash
        model.load_state_dict(saved["model"], strict=True)
        optimizer.load_state_dict(saved["optimizer"])
        completed = saved["completed_updates"]
        torch.set_rng_state(saved["torch_rng"])
        torch.cuda.set_rng_state(saved["cuda_rng"])
        np.random.set_state(saved["numpy_rng"])
        random.setstate(saved["python_rng"])
        del saved
    else:
        evaluate(model, validation, reconstruction, output, "before_training")
    target_updates = arguments.updates or (len(training) // arguments.accumulation)
    assert target_updates * arguments.accumulation <= len(training), "This run uses at most one nonrepeated epoch"
    remaining = training[completed * arguments.accumulation:target_updates * arguments.accumulation]
    loader_generator = torch.Generator().manual_seed(47)
    loader = DataLoader(DrivingClips(remaining), batch_size=1, shuffle=False, num_workers=arguments.workers,
        pin_memory=True, generator=loader_generator, persistent_workers=arguments.workers > 0)
    iterator = iter(loader)
    started = time.time()
    while completed < target_updates:
        if STOP_REQUESTED or (output / "pause.requested").exists() or card_used_bytes() > 46_500_000_000:
            save_checkpoint(output, model, optimizer, completed, configuration, training_hash)
            write_json(output / "paused.json", {"completed_updates": completed, "saved": True})
            return
        model.train()
        optimizer.zero_grad(set_to_none=True)
        update_started = time.time()
        total_loss = 0.
        for _ in range(arguments.accumulation):
            clip = next(iterator).to("cuda", non_blocking=True).float() / 255
            prediction = model(clip, deterministic=False, with_loss=True, warmup=False, num_static=1,
                recon_loss_func=reconstruction, recon_loss_type="vgg", **{key: value for key, value in
                    configuration["loss"].items() if key != "name"})
            loss = prediction["loss_dict"]["loss"] / arguments.accumulation
            assert torch.isfinite(loss), "Nonfinite training loss"
            total_loss += float(loss.detach())
            loss.backward()
            del prediction, loss
        gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 100., error_if_nonfinite=True)
        gradient_groups = {}
        if completed == 0 or (completed + 1) % 50 == 0:
            for name, module in (("encoder", model.encoder_module), ("context", model.ctx_module),
                                 ("dynamics", model.dyn_module), ("decoder", model.decoder_module)):
                norm = float(torch.stack([parameter.grad.detach().float().square().sum()
                    for parameter in module.parameters() if parameter.grad is not None]).sum().sqrt())
                assert np.isfinite(norm) and norm > 0, name
                gradient_groups[name] = norm
        optimizer.step()
        torch.cuda.synchronize()
        completed += 1
        status = {"completed_updates": completed, "target_updates": target_updates, "loss": total_loss,
            "gradient_norm": float(gradient_norm), "seconds": time.time() - update_started,
            "card_used_bytes": card_used_bytes(), "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
            "elapsed_seconds": time.time() - started, "foreground_particles": arguments.particles,
            "gradient_groups": gradient_groups}
        with (output / "training.jsonl").open("a") as stream:
            stream.write(json.dumps(status) + "\n")
        write_json(output / "progress.json", status)
        if completed % 10 == 0:
            print(json.dumps(status), flush=True)
        if completed % 50 == 0:
            save_checkpoint(output, model, optimizer, completed, configuration, training_hash)
        if completed % 500 == 0 and completed < target_updates:
            evaluate(model, validation, reconstruction, output, f"update{completed}")
    save_checkpoint(output, model, optimizer, completed, configuration, training_hash)
    report = evaluate(model, validation, reconstruction, output, "after_training")
    write_json(output / "complete.json", {"completed_updates": completed, "validation": report["mean"],
        "planning_performance_verified": False, "full_330h_training": False})


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--particles", type=int, choices=(8, 16, 32, 64), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, default=ROOT / "outputs/lpwm_driving_video_512x256_v1/corpus/local_openscene_episodes.jsonl")
    parser.add_argument("--updates", type=int, default=200, help="0: one complete local epoch")
    parser.add_argument("--accumulation", type=int, default=4)
    parser.add_argument("--validation-clips", type=int, default=32)
    parser.add_argument("--workers", type=int, default=4)
    main(parser.parse_args())
