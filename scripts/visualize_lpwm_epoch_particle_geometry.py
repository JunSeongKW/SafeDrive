"""Compare deterministic particle centers, glimpse sizes and presence on a fixed panel."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from planning_aware_future_prediction.object_centric.particle_planning_diagnostics import particle_geometry


def distribution(values):
    values = np.asarray(values, dtype=np.float64)
    return {"mean": float(values.mean()), "median": float(np.median(values)),
            "p90": float(np.quantile(values, .9)), "p99": float(np.quantile(values, .99)),
            "maximum": float(values.max())}


def geometry_changes(initial, trained):
    initial_centers, initial_sizes, _, initial_presence = particle_geometry(initial)
    trained_centers, trained_sizes, _, trained_presence = particle_geometry(trained)
    center_shift = np.linalg.norm(trained_centers - initial_centers, axis=-1)
    size_change = np.abs(trained_sizes - initial_sizes)
    relative_size_change = size_change / np.maximum(initial_sizes, 1e-6)
    presence_change = np.abs(trained_presence - initial_presence)
    return {"particle_count": int(center_shift.size),
            "centers_bitwise_identical": bool(np.array_equal(initial[..., :2], trained[..., :2])),
            "sizes_bitwise_identical": bool(np.array_equal(initial[..., 2:4], trained[..., 2:4])),
            "center_shift_input_pixels": distribution(center_shift),
            "size_axis_absolute_change_input_pixels": distribution(size_change),
            "size_axis_relative_change_percent": distribution(relative_size_change * 100),
            "presence_absolute_change": distribution(presence_change),
            "fraction_center_shift_over_one_input_pixel": float((center_shift > 1).mean()),
            "fraction_particles_any_size_axis_change_over_five_percent": float((relative_size_change.max(-1) > .05).mean())}


def font(size):
    return ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", size)


def draw_particles(rgb, attributes, fixed_indices, color, opacity=210):
    image = Image.fromarray(rgb).resize((384, 384), Image.Resampling.NEAREST).convert("RGBA")
    layer = Image.new("RGBA", image.size)
    draw = ImageDraw.Draw(layer)
    centers, _, boxes, presence = particle_geometry(attributes)
    for center in centers:
        horizontal, vertical = center * 3
        draw.ellipse((horizontal-2, vertical-2, horizontal+2, vertical+2), fill=(*color, 130))
    for particle in fixed_indices:
        draw.rectangle(tuple(boxes[particle] * 3), outline=(*color, opacity), width=2)
        horizontal, vertical = centers[particle] * 3
        radius = 2 + 4 * presence[particle]
        draw.ellipse((horizontal-radius, vertical-radius, horizontal+radius, vertical+radius), fill=(*color, opacity))
    return Image.alpha_composite(image, layer).convert("RGB")


def main(arguments):
    monitor = arguments.monitor
    trained_root = monitor / f"update_{arguments.updates:06d}"
    complete = json.loads((trained_root / "complete.json").read_text())
    assert complete["complete"] and complete["completed_updates"] == arguments.updates
    initial = np.load(monitor / "update_000000/particle_attributes.npy")[:, :, 0]
    trained = np.load(trained_root / "particle_attributes.npy")[:, :, 0]
    assert initial.shape == trained.shape and initial.shape[0] == 96
    assert np.isfinite(initial).all() and np.isfinite(trained).all()
    panel = json.loads((monitor / "panel.json").read_text())
    images = np.load(monitor / "images.npy", mmap_mode="r")
    output = arguments.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    assert not (output / "geometry_report.json").exists(), "Preserve completed report"
    selected = [next(index for index, record in enumerate(panel["records"]) if record["scene_type"] == scenario)
                for scenario in ("straight", "left_turn", "right_turn")]
    report = {"completed_updates": arguments.updates, "checkpoint_sha256": complete["checkpoint_sha256"],
              "input_resolution": [128, 128], "scene_count": 96, "cameras_per_scene": 4,
              "scope": "Fixed training panel; geometry change is not evidence of planning improvement",
              "all_particles": geometry_changes(initial, trained), "by_scene_type": {}, "by_camera": {},
              "visualized_scene_indices": selected, "plot_selection": "First scene per pre-existing scenario group; front camera; fixed top16 initial presence boxes and all64 centers",
              "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    for scenario in sorted(set(record["scene_type"] for record in panel["records"])):
        mask = np.array([record["scene_type"] == scenario for record in panel["records"]])
        report["by_scene_type"][scenario] = geometry_changes(initial[mask], trained[mask])
    for camera, name in enumerate(("front", "back", "left", "right")):
        report["by_camera"][name] = geometry_changes(initial[:, camera], trained[:, camera])
    canvas = Image.new("RGB", (1184, 1450), "#f4f5f7")
    draw = ImageDraw.Draw(canvas)
    draw.text((16, 10), "LPWM particle geometry: initialization vs epoch 1", font=font(25), fill="black")
    draw.text((16, 47), "Dots: all 64 centers. Boxes: glimpse sizes of fixed top16 initial-presence particles.", font=font(17), fill="black")
    draw.text((16, 70), "Dot radius = presence. Boxes are LPWM glimpses, not GT / detected object boxes.", font=font(17), fill="black")
    for row, scene_index in enumerate(selected):
        record = panel["records"][scene_index]
        vertical = 112 + row * 428
        before, after = initial[scene_index, 0], trained[scene_index, 0]
        indices = np.argsort(-before[:, 4], kind="stable")[:16]
        raw = Image.fromarray(images[scene_index, 0]).resize((384,384), Image.Resampling.NEAREST)
        panels = (raw, draw_particles(images[scene_index, 0], before, indices, (0,220,255)),
                  draw_particles(images[scene_index, 0], after, indices, (255,180,0)))
        for column, (label, picture) in enumerate(zip(("Input", "Before training", "After epoch 1"), panels)):
            horizontal = 8 + column * 392
            draw.text((horizontal, vertical), record["scene_type"] + " | " + label, font=font(18), fill="black")
            canvas.paste(picture, (horizontal, vertical+27))
    stats = report["all_particles"]
    draw.text((16, 1401), f"96 scenes x 4 cameras: mean center shift {stats['center_shift_input_pixels']['mean']:.3f} input px; "
              f"mean size change {stats['size_axis_relative_change_percent']['mean']:.3f}%", font=font(18), fill="black")
    canvas.save(output / "before_after_particle_geometry.png")
    # Separate exact-scale overlays expose small changes without magnifying arrows.
    overlay = Image.new("RGB", (1184, 486), "#f4f5f7")
    drawing = ImageDraw.Draw(overlay)
    drawing.text((16, 10), "Same particle IDs: before (cyan) / epoch 1 (orange). No displacement magnification.", font=font(19), fill="black")
    for column, scene_index in enumerate(selected):
        before, after = initial[scene_index, 0], trained[scene_index, 0]
        indices = np.argsort(-before[:, 4], kind="stable")[:16]
        picture = draw_particles(images[scene_index, 0], before, indices, (0,220,255)).convert("RGBA")
        layer = Image.new("RGBA", picture.size)
        local_draw = ImageDraw.Draw(layer)
        centers, _, boxes, _ = particle_geometry(after)
        for particle in indices:
            local_draw.rectangle(tuple(boxes[particle]*3), outline=(255,180,0,190), width=1)
            horizontal, vertical = centers[particle]*3
            local_draw.ellipse((horizontal-3,vertical-3,horizontal+3,vertical+3), fill=(255,180,0,255))
        overlay.paste(Image.alpha_composite(picture, layer).convert("RGB"), (8+column*392, 75))
        drawing.text((8+column*392,45),panel["records"][scene_index]["scene_type"], font=font(18),fill="black")
    overlay.save(output / "overlaid_particle_geometry.png")
    (output / "geometry_report.json").write_text(json.dumps(report, indent=2, allow_nan=False)+"\n")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--updates", type=int, default=1614)
    parser.add_argument("--monitor", type=Path, default=ROOT / "outputs/lpwm_drivor_representation_monitor_v1")
    parser.add_argument("--output", type=Path, default=ROOT / "results/lpwm_drivor_epoch1_particle_review_v1")
    main(parser.parse_args())
