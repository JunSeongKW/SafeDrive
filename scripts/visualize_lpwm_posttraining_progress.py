"""Fixed-scene before/during/after particle diagnostics for official LPWM training."""
import hashlib
import html
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
from PIL import Image
import torch


def select_visualization_records(manifest, scenes_per_scenario=2):
    selected = []
    for scenario in ("straight", "turn", "projected_overlap", "other"):
        candidates = [record for record in manifest["records"]
            if record["split"] == "development" and record["scenario"] == scenario]
        candidates.sort(key=lambda record: hashlib.sha256(
            ("lpwm-posttraining-visual:" + record["current_frame_token"]).encode()).hexdigest())
        selected.extend(candidates[:scenes_per_scenario])
    return selected


def draw_particles(axis, rgb, centers, sizes, presence):
    axis.imshow(np.clip(rgb, 0, 1))
    color_map = plt.get_cmap("hsv")
    colors = color_map(np.arange(len(centers)) / len(centers))
    colors[:, 3] = .25 + .75 * presence.clip(0, 1)
    axis.scatter(centers[:, 0], centers[:, 1], c=colors,
        s=7 + 16 * presence.clip(0, 1), linewidths=.3, edgecolors="black")
    for particle_index in np.argsort(-presence, kind="stable")[:16]:
        left, top = centers[particle_index] - sizes[particle_index] / 2
        width, height = sizes[particle_index]
        axis.add_patch(Rectangle((left, top), width, height, fill=False,
            edgecolor=colors[particle_index], linewidth=.8))
    axis.set_xlim(-.5, 127.5)
    axis.set_ylim(127.5, -.5)
    axis.axis("off")


def render_progress(visualization_root, selected_records):
    visualization_root = Path(visualization_root)
    cards = []
    for record in selected_records:
        token = record["current_frame_token"]
        scene_directory = visualization_root / token
        snapshots = sorted(scene_directory.glob("update_*.npz"))
        if not snapshots:
            continue
        loaded = [dict(np.load(snapshot)) for snapshot in snapshots]
        columns = len(loaded) + 1
        figure, axes = plt.subplots(3, columns, figsize=(2.55 * columns, 8.0), squeeze=False)
        reference = loaded[0]
        axes[0, 0].imshow(reference["current_rgb"])
        axes[0, 0].set_title("Fixed current image", fontsize=10)
        axes[1, 0].imshow(reference["current_rgb"])
        axes[1, 0].set_title("Reconstruction target", fontsize=10)
        axes[2, 0].imshow(reference["future_rgb"][-1])
        axes[2, 0].set_title("GT at +4.0 s", fontsize=10)
        for axis in axes[:, 0]:
            axis.axis("off")
        for column, snapshot in enumerate(loaded, start=1):
            update = int(snapshot["optimizer_update"])
            title = str(snapshot.get("initial_label", "Published / before")) if update == 0 else f"Update {update:,}"
            draw_particles(axes[0, column], snapshot["current_rgb"], snapshot["centers"], snapshot["sizes"], snapshot["presence"])
            axes[0, column].set_title(title + f"\npresence sum {snapshot['presence'].sum():.1f}", fontsize=10)
            axes[1, column].imshow(np.clip(snapshot["reconstruction"], 0, 1))
            axes[1, column].set_title(f"Reconstruction MSE {float(snapshot['reconstruction_mse']):.4f}", fontsize=9)
            axes[2, column].imshow(np.clip(snapshot["prediction"][-1], 0, 1))
            axes[2, column].set_title(f"Prior rollout +4 s\n8-frame MSE {float(snapshot['forecast_mse']):.4f}", fontsize=9)
            axes[1, column].axis("off")
            axes[2, column].axis("off")
        figure.suptitle(f"LPWM particle learning\n{record['scenario']} | {token}\nSame images; 64 centers / top16 boxes\nColors denote patch IDs, not object tracks", fontsize=10)
        figure.tight_layout(rect=(0, 0, 1, .88))
        figure.savefig(scene_directory / "particle_evolution.png", dpi=140, facecolor="white")
        plt.close(figure)
        gif_frames = []
        for snapshot in loaded:
            figure, axis = plt.subplots(figsize=(5, 5))
            draw_particles(axis, snapshot["current_rgb"], snapshot["centers"], snapshot["sizes"], snapshot["presence"])
            axis.set_title(f"{record['scenario']} | update {int(snapshot['optimizer_update']):,}\n64 centers; top16 boxes; patch IDs are not object tracks", fontsize=10)
            figure.tight_layout()
            figure.canvas.draw()
            gif_frames.append(Image.fromarray(np.asarray(figure.canvas.buffer_rgba())[:, :, :3].copy()))
            plt.close(figure)
        gif_frames[0].save(scene_directory / "particle_evolution.gif", save_all=True,
            append_images=gif_frames[1:], duration=1200, loop=0)
        cards.append(f'<section><h2>{html.escape(record["scenario"])} — {token}</h2>'
            f'<a href="{token}/particle_evolution.png"><img src="{token}/particle_evolution.png"></a>'
            f'<p><a href="{token}/particle_evolution.gif">Particle animation across training checkpoints</a></p></section>')
    (visualization_root / "index.html").write_text('<!doctype html><meta charset="utf-8"><title>LPWM particle learning progress</title>'
        '<style>body{font:16px system-ui;background:#f8fafc;color:#0f172a;margin:30px}section{background:white;padding:20px;margin:20px 0}img{max-width:100%}</style>'
        '<h1>LPWM: fixed-scene particle changes during NAVSIM post-training</h1>'
        '<p>Snapshots are taken at registered optimizer updates. Missing future columns mean training has not reached that update. '
        'Projected overlap is an occlusion proxy. Particle ID is not an object track. Ground truth future is shown for evaluation only.</p>' + ''.join(cards))


def capture_particle_snapshot(model, frames, manifest, output_root, specification, device, optimizer_update):
    visualization_root = Path(output_root) / "visualization"
    visualization_root.mkdir(parents=True, exist_ok=True)
    selected_records = select_visualization_records(manifest, specification["visualization"]["scenes_per_scenario"])
    training_mode = model.training
    # Visualization must not change the stochastic training trajectory.
    cpu_rng, cuda_rng = torch.get_rng_state(), torch.cuda.get_rng_state(device)
    model.eval()
    try:
        with torch.inference_mode():
            for record in selected_records:
                scene_directory = visualization_root / record["current_frame_token"]
                scene_directory.mkdir(parents=True, exist_ok=True)
                snapshot_path = scene_directory / f"update_{optimizer_update:06d}.npz"
                if snapshot_path.exists():
                    continue
                rgb_frames = frames[record["frame_cache_indices"]].astype(np.float32) / 255
                observed_video = torch.from_numpy(rgb_frames[:4]).to(device).permute(0, 3, 1, 2)[None].contiguous()
                encoded = model(observed_video, deterministic=True)
                reconstruction = encoded["rec_rgb"].reshape(1, 4, 3, 128, 128)[0, 3].permute(1, 2, 0).cpu().numpy()
                generated = model.sample_from_x(observed_video, num_steps=8, cond_steps=4,
                    deterministic=True, use_all_ctx=False, n_pred_eq_gt=False)
                prediction = generated[0, -8:].permute(0, 2, 3, 1).cpu().numpy()
                centers = (encoded["z"][0, 3].cpu().numpy()[:, ::-1] + 1) * 64 - .5
                sizes = torch.sigmoid(encoded["z_scale"][0, 3]).cpu().numpy()[:, ::-1] * 128
                presence = encoded["obj_on"][0, 3].cpu().numpy().reshape(-1)
                np.savez_compressed(snapshot_path, optimizer_update=np.array(optimizer_update),
                    initial_label=np.array(specification["visualization"].get("initial_label", "Published / before")),
                    current_rgb=rgb_frames[3], future_rgb=rgb_frames[4:], centers=centers, sizes=sizes,
                    presence=presence, reconstruction=reconstruction, prediction=prediction,
                    reconstruction_mse=np.array(np.mean((reconstruction - rgb_frames[3])**2)),
                    forecast_mse=np.array(np.mean((prediction - rgb_frames[4:])**2)))
                del encoded, generated, observed_video
    finally:
        torch.set_rng_state(cpu_rng)
        torch.cuda.set_rng_state(cuda_rng, device)
        model.train(training_mode)
    render_progress(visualization_root, selected_records)
    (visualization_root / "latest_snapshot.json").write_text(json.dumps({"optimizer_update": optimizer_update,
        "scenes": [{key: record[key] for key in ("current_frame_token", "scenario", "recording_group")} for record in selected_records],
        "index": str(visualization_root / "index.html")}, indent=2) + "\n")
