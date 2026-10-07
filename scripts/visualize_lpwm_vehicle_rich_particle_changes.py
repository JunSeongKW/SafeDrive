"""Render fixed vehicle-rich scenes using saved particle representations, without training."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from publish_lpwm_particle_geometry_overlays import draw_geometry_overlay, font
from visualize_lpwm_planning_path_geometry import draw_particles, geometry_changes

ROOT = Path(__file__).resolve().parents[1]
CAMERA_NAMES = ("front", "back", "left", "right")
SCENE_DESCRIPTIONS = {41: "Dense downtown traffic", 21: "Traffic under an overpass", 27: "Nearby queued vehicles"}


def digest(path):
    checksum = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024**2), b""):
            checksum.update(chunk)
    return checksum.hexdigest()


def main(arguments):
    monitor, output = arguments.monitor.resolve(), arguments.output.resolve()
    assert not output.exists() or not any(output.iterdir()), "Preserve completed visualization"
    panel = json.loads((monitor / "panel.json").read_text())
    diagnostic = monitor / f"update_{arguments.updates:06d}"
    completion = json.loads((diagnostic / "complete.json").read_text())
    assert completion["complete"] and completion["completed_updates"] == arguments.updates
    initial_attributes = np.asarray(np.load(monitor / "update_000000/particle_attributes.npy", mmap_mode="r")[:, :, 0])
    trained_attributes = np.asarray(np.load(diagnostic / "particle_attributes.npy", mmap_mode="r")[:, :, 0])
    images = np.load(monitor / "images.npy", mmap_mode="r")
    assert initial_attributes.shape == trained_attributes.shape == (96, 4, 64, 14)
    assert np.isfinite(initial_attributes).all() and np.isfinite(trained_attributes).all()
    selected = arguments.scene_indices
    assert len(selected) == 3 and len(set(selected)) == 3
    assert len({panel["records"][index]["recording_group"] for index in selected}) == 3
    output.mkdir(parents=True, exist_ok=True)
    comparison = Image.new("RGB", (1576, 1450), "#f4f5f7")
    comparison_draw = ImageDraw.Draw(comparison)
    comparison_draw.text((16, 10), f"Vehicle-rich scenes | Before training vs update {arguments.updates:,}", font=font(25), fill="black")
    comparison_draw.text((16, 46), "Cyan: before. Orange: after. Overlay: dashed/solid glimpse boxes and white arrows for actual center movement.", font=font(18), fill="black")
    comparison_draw.text((16, 74), "All 64 centers; fixed top16 initial-presence boxes. Overlay dot radius is fixed; other columns show presence.", font=font(18), fill="black")
    overlay = Image.new("RGB", (1568, 712), "#f4f5f7")
    overlay_draw = ImageDraw.Draw(overlay)
    overlay_draw.text((16, 10), f"Vehicle-rich particle overlays | Before vs update {arguments.updates:,}", font=font(25), fill="black")
    overlay_draw.text((16, 46), "Cyan hollow centers / dashed boxes: BEFORE. Orange centers / solid boxes: AFTER. White arrows: actual movement.", font=font(18), fill="black")
    overlay_draw.text((16, 74), "Same 64 particle indices; fixed top16 initial-presence glimpse boxes. These boxes are not object detections.", font=font(18), fill="black")
    overlay_draw.text((16, 100), "Overlay dot radii are fixed. Vehicle counts below refer to projected GT annotations, including occluded objects.", font=font(18), fill="black")
    originals = Image.new("RGB", (1952, 426), "#f4f5f7")
    originals_draw = ImageDraw.Draw(originals)
    originals_draw.text((16, 8), "Original front-camera images | reference only; model input is the full image resized to 128x128", font=font(23), fill="black")
    scenes = []
    for row, scene_index in enumerate(selected):
        record = panel["records"][scene_index]
        before, after = initial_attributes[scene_index, 0], trained_attributes[scene_index, 0]
        box_indices = np.argsort(-before[:, 4], kind="stable")[:16]
        vehicle_annotations = [obj for obj in panel["objects"][record["object_start"]:record["object_end"]]
                               if obj["camera_index"] == 0 and obj["category"] == "vehicle"]
        substantial_annotations = [obj for obj in vehicle_annotations
                                   if obj["box"][2] - obj["box"][0] >= 4 and obj["box"][3] - obj["box"][1] >= 3]
        assert len(vehicle_annotations) >= 10
        label = SCENE_DESCRIPTIONS.get(scene_index, f"Scene {scene_index}")
        row_top = 112 + row * 428
        panels = (
            Image.fromarray(images[scene_index, 0]).resize((384, 384), Image.Resampling.NEAREST),
            draw_particles(images[scene_index, 0], before, box_indices, (0, 220, 255)),
            draw_particles(images[scene_index, 0], after, box_indices, (255, 180, 0)),
            draw_geometry_overlay(images[scene_index, 0], before, after, box_indices, 384),
        )
        for column, (heading, picture) in enumerate(zip(("Model input", "Before training", f"Update {arguments.updates}", "Overlay"), panels)):
            horizontal = 8 + column * 392
            comparison_draw.text((horizontal, row_top), label if column == 0 else heading, font=font(18), fill="black")
            comparison.paste(picture, (horizontal, row_top + 27))
        overlay_horizontal = 8 + row * 520
        overlay_draw.text((overlay_horizontal, 133), label, font=font(21), fill="black")
        overlay_draw.text((overlay_horizontal, 158), f"Scene {scene_index} | GT vehicle projections: {len(vehicle_annotations)}", font=font(17), fill="black")
        overlay.paste(draw_geometry_overlay(images[scene_index, 0], before, after, box_indices, 512), (overlay_horizontal, 190))
        original_path = Path(record["current_camera_paths"][0])
        with Image.open(original_path) as original:
            original_dimensions = original.size
            original_rgb = original.convert("RGB")
            resized_input = np.asarray(original_rgb.resize((128, 128), Image.Resampling.BICUBIC))
            assert np.array_equal(resized_input, images[scene_index, 0]), "Use the same full-image preprocessing as the existing diagnostic"
            originals.paste(original_rgb.resize((640, 360), Image.Resampling.LANCZOS), (8 + row * 648, 58))
        originals_draw.text((8 + row * 648, 35), label, font=font(19), fill="black")
        scene_report = {"scene_index": scene_index, "token": record["token"], "recording_group": record["recording_group"],
                        "scene_type": record["scene_type"], "camera": "front", "description": label,
                        "original_camera_path": str(original_path), "original_camera_size": list(original_dimensions),
                        "gt_projected_vehicle_annotations": len(vehicle_annotations),
                        "gt_projected_vehicle_boxes_width_ge_4_height_ge_3": len(substantial_annotations),
                        "gt_boxes_used_as_model_input_or_loss": False,
                        "box_particle_indices": box_indices.tolist(), "changes": geometry_changes(before, after)}
        scenes.append(scene_report)
        # Include every input camera for each selected traffic scene.
        all_cameras = Image.new("RGB", (1576, 558), "#f4f5f7")
        cameras_draw = ImageDraw.Draw(all_cameras)
        cameras_draw.text((16, 10), f"{label} | Four-camera overlays, before vs {arguments.updates:,}", font=font(23), fill="black")
        cameras_draw.text((16, 43), "Cyan dashed: before. Orange solid: after. White arrows: actual center movement. Fixed-radius overlay dots.", font=font(17), fill="black")
        for camera_index, name in enumerate(CAMERA_NAMES):
            camera_before = initial_attributes[scene_index, camera_index]
            camera_after = trained_attributes[scene_index, camera_index]
            camera_boxes = np.argsort(-camera_before[:, 4], kind="stable")[:16]
            horizontal = 8 + camera_index * 392
            cameras_draw.text((horizontal, 81), name, font=font(21), fill="black")
            all_cameras.paste(draw_geometry_overlay(images[scene_index, camera_index], camera_before, camera_after, camera_boxes, 384), (horizontal, 114))
        all_cameras.save(output / f"scene_{scene_index:03d}_four_camera_overlays.png")
    comparison_draw.text((16, 1401), "Fixed traffic examples chosen by projected vehicle density and visual inspection, independently of particle movement or PDMS.", font=font(18), fill="black")
    comparison.save(output / "vehicle_rich_before_after_with_overlay.png")
    overlay.save(output / "vehicle_rich_particle_overlays.png")
    originals.save(output / "vehicle_rich_original_camera_images.png")
    report = {"completed_updates": arguments.updates, "checkpoint_sha256": completion["checkpoint_sha256"],
              "source_sha256": digest(Path(__file__)), "panel_sha256": digest(monitor / "panel.json"),
              "initial_attributes_sha256": digest(monitor / "update_000000/particle_attributes.npy"),
              "trained_attributes_sha256": digest(diagnostic / "particle_attributes.npy"),
              "selection": "Front-camera projected GT vehicle density screening followed by visual inspection for dense moving traffic, an overpass queue, and nearby queued vehicles; distinct recordings. No particle-change or planning-score criterion.",
              "vehicle_count_interpretation": "Projected GT boxes, not a count of fully visible or detected vehicles; occlusion is not resolved",
              "scene_indices": selected, "scenes": scenes, "overlay_displacement_magnification": 1.0,
              "scope": "Illustrative fixed training-panel cases; geometry changes do not establish planning benefit",
              "training_changes": False, "cpu_only": True}
    (output / "selection_and_geometry_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"output": str(output), "updates": arguments.updates,
                      "scenes": [{"index": scene["scene_index"], "vehicles": scene["gt_projected_vehicle_annotations"],
                                  "mean_center_shift_px": scene["changes"]["center_shift_input_pixels"]["mean"]} for scene in scenes]}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--monitor", type=Path, default=ROOT / "outputs/lpwm_drivor_planning_path_representation_monitor_v1")
    parser.add_argument("--output", type=Path, default=ROOT / "results/lpwm_drivor_planning_path_lora_v1/vehicle_rich_particle_visualization_update3000")
    parser.add_argument("--updates", type=int, default=3000)
    parser.add_argument("--scene-indices", type=int, nargs=3, default=[41, 21, 27])
    main(parser.parse_args())
