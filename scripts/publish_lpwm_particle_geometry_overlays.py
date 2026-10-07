"""Publish CPU-only before/after overlays from completed particle diagnostics."""
import argparse
import ctypes
import hashlib
import html
import json
import os
from pathlib import Path
import sys
import time

import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from planning_aware_future_prediction.object_centric.particle_planning_diagnostics import particle_geometry

BEFORE_COLOR = (0, 220, 255)
AFTER_COLOR = (255, 180, 0)
SCENARIOS = ("straight", "left_turn", "right_turn")


def digest(path):
    checksum = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024**2), b""):
            checksum.update(chunk)
    return checksum.hexdigest()


def read_json(path):
    return json.loads(path.read_text())


def write_json(path, value):
    pending = path.with_suffix(".pending.json")
    pending.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    pending.replace(path)


def font(size):
    return ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", size)


def dashed_line(drawing, start, end, color, dash_length=6):
    displacement = np.asarray(end) - np.asarray(start)
    length = float(np.linalg.norm(displacement))
    if length == 0:
        return
    direction = displacement / length
    for distance in np.arange(0, length, 2 * dash_length):
        drawing.line((tuple(np.asarray(start) + direction * distance),
                      tuple(np.asarray(start) + direction * min(distance + dash_length, length))), fill=color, width=2)


def draw_dashed_box(drawing, box, color):
    left, top, right, bottom = box
    corners = ((left, top), (right, top), (right, bottom), (left, bottom))
    for index in range(4):
        dashed_line(drawing, corners[index], corners[(index + 1) % 4], color)


def draw_geometry_overlay(rgb_image, before_attributes, after_attributes, box_indices, display_size):
    """Uniformly scale image and geometry; retain actual displacement and fixed dot radius."""
    image = Image.fromarray(np.asarray(rgb_image)).resize((display_size, display_size), Image.Resampling.NEAREST).convert("RGBA")
    layer = Image.new("RGBA", image.size)
    drawing = ImageDraw.Draw(layer)
    display_scale = display_size / rgb_image.shape[0]
    before_centers, _, before_boxes, _ = particle_geometry(before_attributes)
    after_centers, _, after_boxes, _ = particle_geometry(after_attributes)
    for particle_index in box_indices:
        draw_dashed_box(drawing, before_boxes[particle_index] * display_scale, (*BEFORE_COLOR, 185))
        drawing.rectangle(tuple(after_boxes[particle_index] * display_scale), outline=(*AFTER_COLOR, 220), width=2)
    for before_center, after_center in zip(before_centers * display_scale, after_centers * display_scale):
        displacement = after_center - before_center
        length = float(np.linalg.norm(displacement))
        if length >= 1:
            drawing.line((tuple(before_center), tuple(after_center)), fill=(0, 0, 0, 190), width=3)
            drawing.line((tuple(before_center), tuple(after_center)), fill=(255, 255, 255, 245), width=1)
            # The arrow tip is exactly at the new center; short moves stay short.
            direction = displacement / length
            normal = np.array((-direction[1], direction[0]))
            head_length = min(5.0, length * .45)
            base = after_center - direction * head_length
            drawing.polygon((tuple(after_center), tuple(base + normal * head_length * .45),
                             tuple(base - normal * head_length * .45)), fill=(255, 255, 255, 245))
    for center in before_centers * display_scale:
        horizontal, vertical = center
        drawing.ellipse((horizontal - 4, vertical - 4, horizontal + 4, vertical + 4), outline=(*BEFORE_COLOR, 255), width=2)
    for center in after_centers * display_scale:
        horizontal, vertical = center
        drawing.ellipse((horizontal - 2, vertical - 2, horizontal + 2, vertical + 2), fill=(*AFTER_COLOR, 255))
    return Image.alpha_composite(image, layer).convert("RGB")


def save_image(image, path):
    pending = path.with_suffix(".pending.png")
    image.save(pending, format="PNG")
    pending.replace(path)


def publish_update(monitor, output, update, panel, images, initial_attributes, source_sha256):
    diagnostic = monitor / f"update_{update:06d}"
    geometry = diagnostic / "geometry"
    completion = read_json(diagnostic / "complete.json")
    geometry_report = read_json(geometry / "geometry_report.json")
    assert completion["complete"] and completion["completed_updates"] == geometry_report["completed_updates"] == update
    destination = output / f"update_{update:06d}"
    metadata_path = destination / "overlay_report.json"
    if metadata_path.exists():
        existing = read_json(metadata_path)
        assert existing["source_sha256"] == source_sha256
        assert existing["checkpoint_sha256"] == completion["checkpoint_sha256"]
        return
    after_attributes = np.asarray(np.load(diagnostic / "particle_attributes.npy", mmap_mode="r")[:, :, 0])
    assert initial_attributes.shape == after_attributes.shape == (96, 4, 64, 14)
    assert np.isfinite(initial_attributes).all() and np.isfinite(after_attributes).all()
    selected_indices = geometry_report["visualized_scene_indices"]
    assert [panel["records"][index]["scene_type"] for index in selected_indices] == list(SCENARIOS)
    destination.mkdir(parents=True, exist_ok=True)
    canvas = Image.new("RGB", (1568, 704), "#f4f5f7")
    drawing = ImageDraw.Draw(canvas)
    drawing.text((16, 10), f"LPWM particle overlay | Before training vs update {update:,}", font=font(25), fill="black")
    drawing.text((16, 46), "Cyan hollow dots / dashed boxes: BEFORE. Orange dots / solid boxes: AFTER.", font=font(20), fill="black")
    drawing.text((16, 75), "White arrows: same particle index, actual movement. All 64 centers; fixed top16 initial-presence glimpse boxes.", font=font(18), fill="black")
    drawing.text((16, 102), "Overlay dot radii are fixed to expose position changes; they do not show presence. Boxes are not object detections.", font=font(18), fill="black")
    scenes = []
    for column, scene_index in enumerate(selected_indices):
        before = initial_attributes[scene_index, 0]
        after = after_attributes[scene_index, 0]
        box_indices = np.argsort(-before[:, 4], kind="stable")[:16]
        picture = draw_geometry_overlay(images[scene_index, 0], before, after, box_indices, 512)
        horizontal = 8 + column * 520
        drawing.text((horizontal, 137), SCENARIOS[column], font=font(21), fill="black")
        canvas.paste(picture, (horizontal, 171))
        before_centers, before_sizes, _, _ = particle_geometry(before)
        after_centers, after_sizes, _, _ = particle_geometry(after)
        scene = {"scene_index": scene_index, "token": panel["records"][scene_index]["token"], "scene_type": SCENARIOS[column],
                 "camera": "front", "box_particle_indices": box_indices.tolist(),
                 "mean_center_shift_input_px": float(np.linalg.norm(after_centers - before_centers, axis=-1).mean()),
                 "mean_size_axis_absolute_change_percent": float((np.abs(after_sizes - before_sizes) / np.maximum(before_sizes, 1e-6) * 100).mean())}
        scenes.append(scene)
    save_image(canvas, destination / "particle_geometry_overlay.png")
    # Append a fourth column to an exact copy of the user's existing comparison.
    with Image.open(geometry / "before_after_particle_geometry.png") as existing:
        comparison = Image.new("RGB", (1576, existing.height), "#f4f5f7")
        comparison.paste(existing, (0, 0))
    drawing = ImageDraw.Draw(comparison)
    drawing.rectangle((0, 0, comparison.width, 104), fill="#f4f5f7")
    drawing.text((16, 10), f"LPWM particle geometry: before / update {update:,} / overlay", font=font(25), fill="black")
    drawing.text((16, 47), "Overlay: cyan hollow centers + dashed boxes (before), orange centers + solid boxes (after), white movement arrows.", font=font(18), fill="black")
    drawing.text((16, 75), "Same particle indices. Image and movement scale together. Overlay dots have fixed radius; other columns retain presence.", font=font(18), fill="black")
    for row, scene in enumerate(scenes):
        scene_index = scene["scene_index"]
        vertical = 112 + row * 428
        drawing.text((1184, vertical), scene["scene_type"] + " | Overlay", font=font(18), fill="black")
        picture = draw_geometry_overlay(images[scene_index, 0], initial_attributes[scene_index, 0],
                                        after_attributes[scene_index, 0], scene["box_particle_indices"], 384)
        comparison.paste(picture, (1184, vertical + 27))
    save_image(comparison, destination / "particle_geometry_comparison_with_overlay.png")
    all_changes = geometry_report["all_particles"]
    metadata = {"complete": True, "completed_updates": update, "checkpoint_sha256": completion["checkpoint_sha256"],
                "source_sha256": source_sha256, "panel_sha256": digest(monitor / "panel.json"),
                "initial_attributes_sha256": digest(monitor / "update_000000/particle_attributes.npy"),
                "trained_attributes_sha256": digest(diagnostic / "particle_attributes.npy"),
                "coordinate_order": "native y,x converted to pixel x,y by existing particle_geometry",
                "center_matching": "same particle index, not semantic object tracking", "displacement_magnification": 1.0,
                "centers_per_camera": 64, "boxes_per_camera": 16, "box_selection": "fixed highest16 initial-presence particles",
                "overlay_dot_radius": "fixed; does not encode presence", "selected_scenes": scenes,
                "all_particles": all_changes, "created_unix": time.time(), "cpu_only": True,
                "training_changes": False, "scientific_interpretation": "Geometry visualization; not attention, object detection, or evidence of planning benefit"}
    write_json(metadata_path, metadata)
    print(json.dumps({"completed_updates": update, "output": str(destination), "scenes": scenes}), flush=True)


def publish_index(monitor, output):
    reports = sorted((read_json(path) for path in output.glob("update_*/overlay_report.json")), key=lambda report: report["completed_updates"])
    links = []
    for report in reversed(reports):
        update = report["completed_updates"]
        directory = f"update_{update:06d}"
        original = os.path.relpath(monitor / directory / "geometry/before_after_particle_geometry.png", output)
        links.append(f'<tr><td>{update}</td><td><a href="{html.escape(original)}">전·후 비교</a></td>'
                     f'<td><a href="{directory}/particle_geometry_overlay.png">겹침·이동 화살표</a></td>'
                     f'<td><a href="{directory}/particle_geometry_comparison_with_overlay.png">비교 + 겹침 4열</a></td></tr>')
    latest = reports[-1]["completed_updates"] if reports else None
    preview = f'<img src="update_{latest:06d}/particle_geometry_overlay.png">' if latest else ""
    document = '<!doctype html><html lang="ko"><meta charset="utf-8"><meta http-equiv="refresh" content="60"><title>LPWM particle 겹침 비교</title><style>body{font-family:sans-serif;max-width:1600px;margin:24px auto}img{width:100%;height:auto}table{border-collapse:collapse}td,th{border:1px solid #bbb;padding:8px}</style><h1>학습 전·후 particle 겹침 비교</h1><p>청록색: 학습 전 · 주황색: 학습 후 · 흰 화살표: 같은 particle 번호의 실제 이동. 겹침 그림은 점 크기를 고정합니다. 박스는 LPWM glimpse이며 객체 검출 박스가 아닙니다.</p><p>기존 매500 update 및 epoch 진단이 완료되면 CPU에서 자동 생성됩니다.</p>' + preview + '<table><tr><th>Update</th><th>기존 이미지</th><th>겹침</th><th>통합</th></tr>' + ''.join(links) + '</table></html>'
    pending = output / "index.pending.html"
    pending.write_text(document)
    pending.replace(output / "index.html")
    write_json(output / "status.json", {"latest_completed_update": latest, "published_updates": [report["completed_updates"] for report in reports],
                                       "updated_unix": time.time(), "cpu_only": True})


def main(arguments):
    monitor, output = arguments.monitor.resolve(), arguments.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    source_sha256 = digest(Path(__file__))
    panel = read_json(monitor / "panel.json")
    images = np.load(monitor / "images.npy", mmap_mode="r")
    initial_attributes = np.array(np.load(monitor / "update_000000/particle_attributes.npy", mmap_mode="r")[:, :, 0])
    if arguments.watch:
        import fcntl
        lock = (output / "publisher.lock").open("a")
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        ctypes.CDLL(None).prctl(15, b"kjs-lpwm-overlay", 0, 0, 0)
    while True:
        assert digest(Path(__file__)) == source_sha256, "Do not change a running publisher"
        updates = arguments.updates or sorted(int(path.parent.parent.name.split("_")[1]) for path in monitor.glob("update_*/geometry/geometry_report.json"))
        for update in updates:
            if update:
                publish_update(monitor, output, update, panel, images, initial_attributes, source_sha256)
        publish_index(monitor, output)
        if not arguments.watch:
            break
        if (output / f"update_{arguments.final_update:06d}/overlay_report.json").exists():
            break
        if (output / "stop.requested").exists():
            break
        time.sleep(arguments.poll_seconds)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--monitor", type=Path, default=ROOT / "outputs/lpwm_drivor_planning_path_representation_monitor_v1")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/lpwm_drivor_particle_geometry_overlays_v1")
    parser.add_argument("--updates", type=int, nargs="+")
    parser.add_argument("--watch", action="store_true")
    parser.add_argument("--poll-seconds", type=int, default=30)
    parser.add_argument("--final-update", type=int, default=40350)
    main(parser.parse_args())
