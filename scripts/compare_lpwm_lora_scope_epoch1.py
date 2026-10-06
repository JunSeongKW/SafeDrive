"""CPU-only comparison of the two exact first-epoch particle representations."""
import argparse
import csv
import ctypes
import fcntl
import hashlib
import html
import json
import os
from pathlib import Path
import sys
import time

import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from planning_aware_future_prediction.object_centric.particle_planning_diagnostics import particle_geometry
from visualize_lpwm_epoch_particle_geometry import draw_particles, font, geometry_changes

CAMERAS = ("front", "back", "left", "right")
OLD_COLOR = (0, 220, 255)
NEW_COLOR = (255, 180, 0)


def digest(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024**2), b""):
            result.update(chunk)
    return result.hexdigest()


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_suffix(".pending.json")
    pending.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    pending.replace(path)


def load_configuration(path):
    configuration = json.loads(path.read_text())
    for key in ("old_monitor", "new_monitor", "training_root", "output"):
        configuration[key] = ROOT / configuration[key]
    return configuration


def verify_registration(configuration, config_path):
    registration = json.loads((configuration["output"] / "registration.json").read_text())
    assert digest(config_path) == registration["configuration_sha256"]
    for filename, expected in registration["sources"].items():
        assert digest(ROOT / filename) == expected, filename
    for filename, expected in registration["old_evidence"].items():
        assert digest(ROOT / filename) == expected, filename
    assert digest(configuration["old_monitor"] / "panel.json") == digest(configuration["new_monitor"] / "panel.json")
    return registration


def load_attributes(monitor, update):
    folder = monitor / f"update_{update:06d}"
    metadata = json.loads((folder / "complete.json").read_text())
    assert metadata["complete"] and metadata["completed_updates"] == update and metadata["scene_count"] == 96
    attributes = np.load(folder / "particle_attributes.npy", mmap_mode="r")
    assert attributes.shape[:4] == (96, 4, 9, 64) and attributes.shape[-1] == 14
    current = np.array(attributes[:, :, 0])
    assert np.isfinite(current).all()
    return current, metadata


def draw_comparison_overlay(rgb, old_attributes, new_attributes, fixed_indices):
    canvas = draw_particles(rgb, old_attributes, fixed_indices, OLD_COLOR).convert("RGBA")
    layer = Image.new("RGBA", canvas.size)
    drawing = ImageDraw.Draw(layer)
    old_centers, _, _, _ = particle_geometry(old_attributes)
    centers, _, boxes, presence = particle_geometry(new_attributes)
    for center in centers:
        horizontal, vertical = center * 3
        drawing.ellipse((horizontal-2, vertical-2, horizontal+2, vertical+2), fill=(*NEW_COLOR, 180))
    for particle in fixed_indices:
        drawing.rectangle(tuple(boxes[particle] * 3), outline=(*NEW_COLOR, 220), width=2)
        origin, destination = old_centers[particle] * 3, centers[particle] * 3
        drawing.line([tuple(origin), tuple(destination)], fill=(255, 255, 255, 200), width=1)
        horizontal, vertical = destination
        radius = 2 + 4 * presence[particle]
        drawing.ellipse((horizontal-radius, vertical-radius, horizontal+radius, vertical+radius), fill=(*NEW_COLOR, 240))
    return Image.alpha_composite(canvas, layer).convert("RGB")


def render_grid(output, filename, selections, panel, images, initial, old, new, include_raw):
    # One overview across scenarios, and detailed four-camera sheets per scene.
    canvas = Image.new("RGB", (1584, 134 + 422 * len(selections)), "#f4f5f7")
    drawing = ImageDraw.Draw(canvas)
    drawing.text((12, 8), "Exact epoch 1 comparison | both models: 1,614 updates", font=font(25), fill="black")
    drawing.text((12, 43), "Old: attention Q/V LoRA. New: geometry + appearance + future LoRA.", font=font(19), fill="black")
    drawing.text((12, 70), "Dots = 64 particle centers; boxes = LPWM glimpse sizes for fixed top16 initial-presence IDs.", font=font(17), fill="black")
    drawing.text((12, 94), "Dot radius = presence. Cyan = old, orange = new. Same-index comparison is not object tracking.", font=font(17), fill="black")
    for row, (scene_index, camera_index) in enumerate(selections):
        record = panel["records"][scene_index]
        attributes = (initial[scene_index, camera_index], old[scene_index, camera_index], new[scene_index, camera_index])
        fixed_indices = np.argsort(-attributes[0][:, 4], kind="stable")[:16]
        rgb = images[scene_index, camera_index]
        pictures = [draw_particles(rgb, attributes[0], fixed_indices, (145, 220, 80)),
                    draw_particles(rgb, attributes[1], fixed_indices, OLD_COLOR),
                    draw_particles(rgb, attributes[2], fixed_indices, NEW_COLOR)]
        labels = ["Public initialization", "Old / epoch 1", "New / epoch 1"]
        if include_raw:
            pictures.insert(0, Image.fromarray(rgb).resize((384, 384), Image.Resampling.NEAREST))
            labels.insert(0, "Input")
        else:
            pictures.append(draw_comparison_overlay(rgb, attributes[1], attributes[2], fixed_indices))
            labels.append("Old + new overlay")
        vertical = 132 + row * 422
        for column, (label, picture) in enumerate(zip(labels, pictures)):
            horizontal = 8 + column * 394
            drawing.text((horizontal, vertical), f"{record['scene_type']} / {CAMERAS[camera_index]}", font=font(16), fill="black")
            drawing.text((horizontal, vertical+18), label, font=font(16), fill="black")
            canvas.paste(picture, (horizontal, vertical+38))
    canvas.save(output / filename)


def build_report(configuration, registration):
    output = configuration["output"]
    if (output / "complete.json").exists():
        return
    old_monitor, new_monitor = configuration["old_monitor"], configuration["new_monitor"]
    initial, initial_metadata = load_attributes(old_monitor, 0)
    new_initial, new_initial_metadata = load_attributes(new_monitor, 0)
    old, old_metadata = load_attributes(old_monitor, 1614)
    new, new_metadata = load_attributes(new_monitor, 1614)
    # A common initial reference is valid only if deterministic attributes agree.
    assert np.allclose(initial, new_initial, rtol=0, atol=1e-6), "Initial representations differ; inspect before pooling baselines"
    assert old_metadata["checkpoint_sha256"] == registration["old_epoch1_checkpoint_sha256"]
    assert old_metadata["original_weights_frozen_sha256"] == new_metadata["original_weights_frozen_sha256"]
    panel = json.loads((old_monitor / "panel.json").read_text())
    images = np.load(old_monitor / "images.npy", mmap_mode="r")
    pairs = {"initial_to_old": (initial, old), "initial_to_new": (initial, new), "old_to_new": (old, new)}
    groups = {"all": np.ones(96, dtype=bool)}
    groups.update({scenario: np.array([record["scene_type"] == scenario for record in panel["records"]])
                   for scenario in ("straight", "left_turn", "right_turn")})
    groups["projected_box_overlap_proxy"] = np.array([record["projected_overlap_proxy"] for record in panel["records"]], dtype=bool)
    report = {"completed_updates_each": 1614, "scene_count": 96, "camera_count": 4,
        "input_resolution": [128, 128], "initial_attributes_max_abs_difference": float(np.abs(initial-new_initial).max()),
        "old_checkpoint_sha256": old_metadata["checkpoint_sha256"], "new_checkpoint_sha256": new_metadata["checkpoint_sha256"],
        "selected_scenes_before_new_results": configuration["gallery_scene_indices"],
        "same_native_particle_indices_not_semantic_object_matches": True,
        "sizes_are_glimpses_not_detection_boxes": True, "training_distribution_diagnostic_not_pdms": True,
        "larger_geometry_change_is_not_necessarily_better": True, "comparisons": {}}
    for name, (reference, changed) in pairs.items():
        report["comparisons"][name] = {
            "all_particles": geometry_changes(reference, changed),
            "by_scene_group": {group: geometry_changes(reference[mask], changed[mask]) for group, mask in groups.items() if mask.any()},
            "by_camera": {camera: geometry_changes(reference[:, index], changed[:, index]) for index, camera in enumerate(CAMERAS)}}
    with (output / "particle_geometry.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["token", "scene_type", "camera", "particle_index", "condition", "center_x_px", "center_y_px",
                         "width_px", "height_px", "presence", "center_shift_from_initial_px", "width_change_from_initial_percent", "height_change_from_initial_percent"])
        original_centers, original_sizes, _, _ = particle_geometry(initial)
        for condition, values in (("initial", initial), ("old_epoch1", old), ("new_epoch1", new)):
            centers, sizes, _, presence = particle_geometry(values)
            displacement = np.linalg.norm(centers-original_centers, axis=-1)
            relative_size = 100 * (sizes-original_sizes) / np.maximum(original_sizes, 1e-6)
            for scene_index, record in enumerate(panel["records"]):
                for camera_index, camera in enumerate(CAMERAS):
                    for particle_index in range(64):
                        coordinates = (scene_index, camera_index, particle_index)
                        writer.writerow([record["token"], record["scene_type"], camera, particle_index, condition,
                            *centers[coordinates], *sizes[coordinates], presence[coordinates], displacement[coordinates], *relative_size[coordinates]])
    selected = configuration["overview_scene_indices"]
    render_grid(output, "epoch1_comparison.png", [(index, 0) for index in selected], panel, images, initial, old, new, True)
    render_grid(output, "epoch1_overlays.png", [(index, 0) for index in selected], panel, images, initial, old, new, False)
    links = []
    for index in configuration["gallery_scene_indices"]:
        filename = f"scene_{index:03d}_all_cameras.png"
        render_grid(output, filename, [(index, camera) for camera in range(4)], panel, images, initial, old, new, False)
        links.append(f'<li><a href="{filename}">{html.escape(panel["records"][index]["token"])} — {panel["records"][index]["scene_type"]}</a></li>')
    write_json(output / "summary.json", report)
    (output / "index.html").write_text('''<!doctype html><html lang="ko"><meta charset="utf-8"><title>LPWM 두 LoRA 조건의 첫 epoch 비교</title>
<style>body{font-family:sans-serif;max-width:1600px;margin:24px auto;padding:0 16px}img{width:100%;height:auto}table{border-collapse:collapse}td,th{border:1px solid #ccc;padding:8px}</style>
<h1>공통 초기 상태 / 이전 attention LoRA 1epoch / 새 세 경로 LoRA 1epoch</h1>
<p>모두 같은 장면·카메라·주행 명령. 두 학습 결과는 정확히1,614update입니다. 위치는128×128 입력 이미지의pixel, 사각형은LPWM glimpse 크기입니다.
같은particle 번호는객체identity가 아니며, projected overlap은가림의보조proxy입니다. 더많이이동했다고planning이좋아졌다는의미는아닙니다.</p>
<p><a href="summary.json">전체96장면·상황별·카메라별변화량</a> / <a href="particle_geometry.csv">모든particle좌표·크기·presence CSV</a></p>
<h2>같은장면비교</h2><img src="epoch1_comparison.png"><h2>초기 / 이전 / 새조건 / 겹쳐보기</h2><img src="epoch1_overlays.png">
<h2>사전고정장면의4카메라상세</h2><ul>''' + "".join(links) + "</ul></html>")
    write_json(output / "complete.json", {"complete": True, "completed_unix": time.time(),
        "completed_updates_each": 1614, "summary_sha256": digest(output / "summary.json"),
        "old_checkpoint_sha256": old_metadata["checkpoint_sha256"], "new_checkpoint_sha256": new_metadata["checkpoint_sha256"],
        "scene_count": 96, "no_gpu_or_training_changes": True})


def watch(configuration, config_path):
    ctypes.CDLL(None).prctl(15, b"kjs-epoch1-viz", 0, 0, 0)
    output = configuration["output"]
    lock = (output / "watch.lock").open("w")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    while True:
        registration = verify_registration(configuration, config_path)
        if (output / "complete.json").exists():
            write_json(output / "status.json", {"stage": "complete"})
            return
        if any(path.exists() for path in (output / "pause.requested", configuration["training_root"] / "pause.requested",
                                         configuration["training_root"] / "navsim_v1/paused.json")):
            write_json(output / "status.json", {"stage": "paused_with_training"})
            return
        dependencies = [configuration["new_monitor"] / f"update_{update:06d}/complete.json" for update in (0, 1614)]
        if all(path.exists() for path in dependencies):
            write_json(output / "status.json", {"stage": "rendering_exact_epoch1"})
            build_report(configuration, registration)
            continue
        monitor_state_path = configuration["new_monitor"] / "watch_status.json"
        monitor_state = json.loads(monitor_state_path.read_text()) if monitor_state_path.exists() else {}
        assert monitor_state.get("stage") != "diagnostic_failed_training_unaffected", "Representation monitor failed; inspect its log"
        progress_path = configuration["training_root"] / "navsim_v1/progress.json"
        progress = json.loads(progress_path.read_text()) if progress_path.exists() else {}
        write_json(output / "status.json", {"stage": "waiting_for_exact_epoch1_representations",
            "target_updates": 1614, "completed_training_updates": progress.get("completed_updates"),
            "missing": [str(path.relative_to(ROOT)) for path in dependencies if not path.exists()], "updated_unix": time.time()})
        time.sleep(30)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--watch", action="store_true")
    options = parser.parse_args()
    configuration = load_configuration(options.config)
    try:
        if options.watch:
            watch(configuration, options.config)
        else:
            build_report(configuration, verify_registration(configuration, options.config))
    except Exception as error:
        write_json(configuration["output"] / "status.json", {"stage": "failed_training_unaffected", "error": repr(error), "updated_unix": time.time()})
        raise
