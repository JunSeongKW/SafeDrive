"""Particle and trajectory snapshots during merged-stage2 fine-tuning."""
import copy
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from visualize_lpwm_posttraining_progress import capture_particle_snapshot, select_visualization_records, render_progress


def capture_planning_snapshot(model, frames, stage1_manifest, planning_records, targets,
                              output_root, stage1_configuration, device, optimizer_update):
    specification = copy.deepcopy(stage1_configuration)
    specification["visualization"]["initial_label"] = "After stage1 / before planning"
    selected = select_visualization_records(stage1_manifest)
    index_by_token = {record["current_frame_token"]: index for index, record in enumerate(planning_records)}
    was_training = model.training
    cpu_rng, cuda_rng = torch.get_rng_state(), torch.cuda.get_rng_state(device)
    model.eval()
    try:
        for record in selected:
            record_index = index_by_token[record["current_frame_token"]]
            status = torch.from_numpy(targets["ego_status"][record_index:record_index + 1].copy()).to(device)
            with model.encoder_command(status):
                capture_particle_snapshot(model.world_model, frames, {"records": [record]}, output_root,
                    specification, device, optimizer_update)
            current_images = torch.from_numpy(np.array(frames[record["frame_cache_indices"][:4]])).to(device).permute(0, 3, 1, 2).float()[None] / 255
            with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                prediction = model(current_images, status)["trajectory"][0].float().cpu().numpy()
            truth = targets["ego_trajectory_target"][record_index]
            scene_root = Path(output_root) / "visualization" / record["current_frame_token"]
            np.savez_compressed(scene_root / f"planning_update_{optimizer_update:06d}.npz", trajectory=prediction, target=truth)
            figure, axis = plt.subplots(figsize=(5, 5))
            axis.plot(truth[:, 1], truth[:, 0], "o-", label="Expert trajectory")
            axis.plot(prediction[:, 1], prediction[:, 0], "o-", label="LPWM planner")
            axis.set(title=f"{record['scenario']} | update {optimizer_update:,}", xlabel="Lateral (m)", ylabel="Forward (m)")
            axis.set_aspect("equal", adjustable="datalim")
            axis.grid(alpha=.25)
            axis.legend()
            figure.tight_layout()
            figure.savefig(scene_root / f"trajectory_update_{optimizer_update:06d}.png", dpi=150)
            plt.close(figure)
        render_progress(Path(output_root) / "visualization", selected)
        (Path(output_root) / "visualization/latest_snapshot.json").write_text(json.dumps({
            "optimizer_update": optimizer_update, "stage": "merged_stage2",
            "scenes": [{key: record[key] for key in ("current_frame_token", "scenario", "recording_group")} for record in selected]}, indent=2) + "\n")
    finally:
        model.train(was_training)
        torch.set_rng_state(cpu_rng)
        torch.cuda.set_rng_state(cuda_rng, device)
