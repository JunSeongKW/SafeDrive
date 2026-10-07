"""Render observed particle changes ranked across a saved diagnostic panel, on CPU."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from publish_lpwm_particle_geometry_overlays import draw_geometry_overlay, font
from planning_aware_future_prediction.object_centric.particle_planning_diagnostics import particle_geometry

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CAMERA_NAMES = ("front", "back", "left", "right")
BEFORE_COLOR = (0, 220, 255)
AFTER_COLOR = (255, 180, 0)


def file_sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def draw_particle_snapshot(rgb_image, attributes, selected_particle_indices, color):
    """Show all centers at fixed radius, and boxes for the same selected indices."""
    display = Image.fromarray(rgb_image).resize((384, 384), Image.Resampling.NEAREST).convert("RGBA")
    overlay = Image.new("RGBA", display.size)
    drawing = ImageDraw.Draw(overlay)
    centers, _, boxes, _ = particle_geometry(attributes)
    for particle_index in selected_particle_indices:
        drawing.rectangle(tuple(boxes[particle_index] * 3), outline=(*color, 220), width=2)
    for horizontal, vertical in centers * 3:
        drawing.ellipse((horizontal - 3, vertical - 3, horizontal + 3, vertical + 3), fill=(*color, 240))
    return Image.alpha_composite(display, overlay).convert("RGB")


def main(arguments):
    monitor = arguments.monitor.resolve()
    destination = arguments.output.resolve()
    if destination.exists() and any(destination.iterdir()):
        raise FileExistsError(f"Preserve existing figures: {destination}")
    checkpoint_root = monitor / f"update_{arguments.updates:06d}"
    completion = json.loads((checkpoint_root / "complete.json").read_text())
    assert completion["complete"] and completion["completed_updates"] == arguments.updates
    initial_path = monitor / "update_000000/particle_attributes.npy"
    trained_path = checkpoint_root / "particle_attributes.npy"
    initial_attributes = np.asarray(np.load(initial_path, mmap_mode="r")[:, :, 0])
    trained_attributes = np.asarray(np.load(trained_path, mmap_mode="r")[:, :, 0])
    images = np.load(monitor / "images.npy", mmap_mode="r")
    panel = json.loads((monitor / "panel.json").read_text())
    assert initial_attributes.shape == trained_attributes.shape == (96, 4, 64, 14)
    assert images.shape == (96, 4, 128, 128, 3)
    assert np.isfinite(initial_attributes).all() and np.isfinite(trained_attributes).all()
    initial_centers, initial_sizes, _, initial_presence = particle_geometry(initial_attributes)
    trained_centers, trained_sizes, _, trained_presence = particle_geometry(trained_attributes)
    center_shift = np.linalg.norm(trained_centers - initial_centers, axis=-1)
    relative_size_change = np.abs(trained_sizes - initial_sizes) / np.maximum(initial_sizes, 1e-6) * 100
    particle_size_change = relative_size_change.mean(axis=-1)
    image_statistics = []
    for scene_index, record in enumerate(panel["records"]):
        for camera_index, camera_name in enumerate(CAMERA_NAMES):
            image_statistics.append({
                "scene_index": scene_index, "camera_index": camera_index, "camera": camera_name,
                "token": record["token"], "scene_type": record["scene_type"],
                "center_mean_input_px": float(center_shift[scene_index, camera_index].mean()),
                "center_max_input_px": float(center_shift[scene_index, camera_index].max()),
                "size_mean_absolute_percent": float(relative_size_change[scene_index, camera_index].mean()),
            })
    criteria = (
        ("center_mean_input_px", "largest_mean_position_change", "Largest mean position change"),
        ("size_mean_absolute_percent", "largest_mean_size_change", "Largest mean size change"),
        ("center_max_input_px", "largest_individual_position_change", "Largest single-particle position change"),
    )
    destination.mkdir(parents=True, exist_ok=True)
    selected_examples = []
    figure_images = []
    for metric_name, artifact_name, heading in criteria:
        selected = max(image_statistics, key=lambda record: record[metric_name])
        scene_index, camera_index = selected["scene_index"], selected["camera_index"]
        before = initial_attributes[scene_index, camera_index]
        after = trained_attributes[scene_index, camera_index]
        ranking_values = particle_size_change[scene_index, camera_index] if metric_name == "size_mean_absolute_percent" else center_shift[scene_index, camera_index]
        selected_particles = np.argsort(-ranking_values, kind="stable")[:8]
        rgb_image = np.asarray(images[scene_index, camera_index])
        reference_path = Path(panel["records"][scene_index]["current_camera_paths"][camera_index])
        with Image.open(reference_path) as original:
            expected_input = np.asarray(original.convert("RGB").resize((128, 128), Image.Resampling.BICUBIC))
        assert np.array_equal(expected_input, rgb_image)
        figure = Image.new("RGB", (1576, 656), "#f4f5f7")
        drawing = ImageDraw.Draw(figure)
        drawing.text((16, 12), f"{heading} | Scene {scene_index} | {selected['camera']} camera | {selected['scene_type']}", font=font(24), fill="black")
        drawing.text((16, 49), f"Before vs update {arguments.updates:,} | Mean shift {selected['center_mean_input_px']:.2f} input px | Max shift {selected['center_max_input_px']:.2f} px | Mean size change {selected['size_mean_absolute_percent']:.2f}%", font=font(21), fill="black")
        drawing.text((16, 83), "Cyan: before. Orange: after. White arrows: actual center displacement; no exaggeration.", font=font(19), fill="black")
        drawing.text((16, 114), "All 64 centers. Boxes: 8 particles with largest change for this criterion, fixed across columns. Dot radii are fixed.", font=font(18), fill="black")
        panels = (
            Image.fromarray(rgb_image).resize((384, 384), Image.Resampling.NEAREST),
            draw_particle_snapshot(rgb_image, before, selected_particles, BEFORE_COLOR),
            draw_particle_snapshot(rgb_image, after, selected_particles, AFTER_COLOR),
            draw_geometry_overlay(rgb_image, before, after, selected_particles, 384),
        )
        for column, (title, picture) in enumerate(zip(("Model input", "Before training", f"Update {arguments.updates:,}", "Before / after overlay"), panels)):
            horizontal = 8 + column * 392
            drawing.text((horizontal, 153), title, font=font(21), fill="black")
            figure.paste(picture, (horizontal, 188))
        highest_particle = int(selected_particles[0])
        selected_particle_statistics = [{
            "particle_index": int(particle_index),
            "center_shift_input_px": float(center_shift[scene_index, camera_index, particle_index]),
            "size_mean_absolute_percent": float(particle_size_change[scene_index, camera_index, particle_index]),
            "initial_presence": float(initial_presence[scene_index, camera_index, particle_index]),
            "trained_presence": float(trained_presence[scene_index, camera_index, particle_index]),
        } for particle_index in selected_particles]
        drawing.text((16, 587), f"Largest-change particle #{highest_particle}: presence {initial_presence[scene_index,camera_index,highest_particle]:.3f} -> {trained_presence[scene_index,camera_index,highest_particle]:.3f}. Ranking includes low-presence particles.", font=font(18), fill="black")
        drawing.text((16, 618), "Selected extremes from 96 scenes x 4 cameras; not typical examples or evidence of planning benefit. Boxes are glimpses.", font=font(18), fill="black")
        figure.save(destination / f"{artifact_name}.png")
        figure_images.append(figure)
        selected_examples.append({**selected, "selection_metric": metric_name, "rank": 1,
                                  "figure": f"{artifact_name}.png", "box_particle_statistics": selected_particle_statistics,
                                  "original_camera_path": str(reference_path)})
    combined = Image.new("RGB", (1576, sum(figure.height for figure in figure_images)), "#f4f5f7")
    vertical = 0
    for figure in figure_images:
        combined.paste(figure, (0, vertical))
        vertical += figure.height
    combined.save(destination / "largest_particle_changes_comparison.png")
    with (destination / "all_camera_image_changes.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(image_statistics[0]))
        writer.writeheader()
        writer.writerows(image_statistics)
    report = {
        "completed_updates": arguments.updates, "checkpoint_sha256": completion["checkpoint_sha256"],
        "panel_sha256": file_sha256(monitor / "panel.json"),
        "initial_attributes_sha256": file_sha256(initial_path), "trained_attributes_sha256": file_sha256(trained_path),
        "source_sha256": file_sha256(Path(__file__)), "cpu_only": True, "training_changes": False,
        "candidate_images": len(image_statistics), "particles_per_image": 64,
        "selection": "Argmax of mean center displacement, mean absolute width/height relative change, and maximum individual center displacement across all 384 camera images. Includes low-presence particles.",
        "size_definition": "Mean over 64 particles and width/height axes of 100 * abs(after-before) / before; not box-area change.",
        "center_definition": "Euclidean displacement in 128x128 model-input pixels for identical particle indices on the identical image.",
        "global_center_mean_input_px": float(center_shift.mean()),
        "global_size_mean_absolute_percent": float(relative_size_change.mean()),
        "box_selection": "Largest8 by the respective per-particle criterion, not initial-presence top16 used in previous figures.",
        "displacement_magnification": 1.0, "markers_encode_presence": False,
        "selected_examples": selected_examples,
        "scope": "User-requested extreme geometry examples, not representative or causal planning evidence.",
    }
    (destination / "selection_and_geometry_report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--monitor", type=Path, default=PROJECT_ROOT / "outputs/lpwm_drivor_planning_path_representation_monitor_v1")
    parser.add_argument("--updates", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    main(parser.parse_args())
