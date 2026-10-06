"""Read a training snapshot and compare current particles on the fixed 96-scene panel."""
import argparse
from contextlib import redirect_stdout
import csv
import hashlib
import html
import io
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from planning_aware_future_prediction.object_centric.lpwm_drivor_planning_path_lora import LPWMDrivoRPlanningPathLoRAModel
from planning_aware_future_prediction.object_centric.lpwm_planning_finetuning import particle_attributes
from planning_aware_future_prediction.object_centric.particle_planning_diagnostics import particle_geometry
from visualize_lpwm_planning_path_geometry import draw_particles, geometry_changes

MONITOR = ROOT / "outputs/lpwm_drivor_planning_path_representation_monitor_v1"
TRAINING = ROOT / "outputs/lpwm_drivor_planning_path_lora_v1"
CAMERA_LABELS = ("전방", "후방", "좌측", "우측")
SCENARIO_LABELS = {"straight": "직진", "left_turn": "좌회전", "right_turn": "우회전"}
BEFORE_COLOR, AFTER_COLOR = (0, 220, 255), (255, 180, 0)


def file_digest(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024**2), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path, content):
    path.write_text(json.dumps(content, indent=2, ensure_ascii=False, allow_nan=False) + "\n")


def verify_registered_sources():
    checked = {}
    for registration_path in (TRAINING / "registration.json", TRAINING / "parallelism_registration.json",
                              TRAINING / "queue_registration.json", MONITOR / "registration.json"):
        registration = json.loads(registration_path.read_text())
        for field in ("sources", "configurations"):
            for filename, expected in registration.get(field, {}).items():
                if filename not in checked:
                    checked[filename] = file_digest(ROOT / filename)
                assert checked[filename] == expected, filename
    return len(checked)


def total_card_bytes():
    usage = subprocess.check_output(["nvidia-smi", "--id=0", "--query-gpu=memory.used",
                                     "--format=csv,noheader,nounits"], text=True).strip()
    return int(usage) * 1024**2


def current_attributes(encoder, rgb, command):
    encoder.active_command = command
    try:
        with torch.inference_mode(), torch.autocast("cuda", enabled=False):
            encoder_rgb = rgb.float()[:, None]
            if encoder.world_model.normalize_rgb:
                encoder_rgb = encoder_rgb * 2 - 1
            encoded = encoder.world_model.encoder_module(encoder_rgb, deterministic=True)
            return particle_attributes(encoded, "obj_on")[:, 0]
    finally:
        encoder.active_command = None


def overlay_particles(rgb, before, after, selected):
    picture = draw_particles(rgb, before, selected, BEFORE_COLOR).convert("RGBA")
    layer = Image.new("RGBA", picture.size)
    drawing = ImageDraw.Draw(layer)
    centers, _, boxes, _ = particle_geometry(after)
    for center in centers:
        horizontal, vertical = center * 3
        drawing.ellipse((horizontal - 1, vertical - 1, horizontal + 1, vertical + 1), fill=(*AFTER_COLOR, 230))
    for particle_index in selected:
        drawing.rectangle(tuple(boxes[particle_index] * 3), outline=(*AFTER_COLOR, 230), width=1)
    return Image.alpha_composite(picture, layer).convert("RGB")


def render_comparison(output, filename, selections, images, records, initial, trained, updates, statistics):
    font_path = subprocess.check_output(["fc-match", "-f", "%{file}", "Noto Sans CJK KR"], text=True).strip()
    fonts = {size: ImageFont.truetype(font_path, size) for size in (17, 20, 23, 30)}
    canvas = Image.new("RGB", (1592, 174 + 448 * len(selections) + 82), "#f4f5f7")
    drawing = ImageDraw.Draw(canvas)
    drawing.text((16, 10), f"LPWM 학습 중 particle 변화: 초기 → {updates} update ({updates / 1614:.3f} epoch)", font=fonts[30], fill="#15243c")
    drawing.text((16, 57), "같은 장면·카메라·명령 | 하늘색: 학습 전 / 주황색: 학습 후 | 이동량 과장 없음", font=fonts[23], fill="#34435b")
    drawing.text((16, 91), "점: 전체 64개 중심 / 사각형: 초기 presence 상위 16개의 glimpse 범위 (객체 정답 박스 아님)", font=fonts[20], fill="#34435b")
    drawing.text((16, 124), "128×128 입력을 표시용으로 3배 확대. 같은 particle 번호는 객체 identity 추적을 뜻하지 않습니다.", font=fonts[20], fill="#34435b")
    for row, (scene_index, camera_index) in enumerate(selections):
        before, after = initial[scene_index, camera_index], trained[scene_index, camera_index]
        selected = np.argsort(-before[:, 4], kind="stable")[:16]
        rgb = np.array(images[scene_index, camera_index])
        pictures = (Image.fromarray(rgb).resize((384, 384), Image.Resampling.NEAREST),
                    draw_particles(rgb, before, selected, BEFORE_COLOR),
                    draw_particles(rgb, after, selected, AFTER_COLOR),
                    overlay_particles(rgb, before, after, selected))
        labels = ("실제 입력", "학습 전", f"{updates} update 후", "전후 겹쳐보기")
        vertical = 172 + row * 448
        record = records[scene_index]
        scenario = SCENARIO_LABELS[record["scene_type"]]
        for column, (picture, label) in enumerate(zip(pictures, labels)):
            horizontal = 8 + column * 396
            drawing.text((horizontal, vertical), f"{scenario} · {CAMERA_LABELS[camera_index]} | {label}", font=fonts[20], fill="#15243c")
            canvas.paste(picture, (horizontal, vertical + 32))
        local = geometry_changes(before, after)
        drawing.text((16, vertical + 418), f"{record['token']} | 이 영상: 중심 평균 {local['center_shift_input_pixels']['mean']:.4f}px · 크기 축별 평균 변화 {local['size_axis_relative_change_percent']['mean']:.3f}%", font=fonts[17], fill="#34435b")
    vertical = 174 + 448 * len(selections)
    drawing.text((16, vertical), f"전체 96장면 × 4카메라: 중심 평균 {statistics['center_shift_input_pixels']['mean']:.4f}px / 최대 {statistics['center_shift_input_pixels']['maximum']:.3f}px", font=fonts[20], fill="#15243c")
    drawing.text((16, vertical + 31), f"크기 축별 평균 변화 {statistics['size_axis_relative_change_percent']['mean']:.3f}% | 변화량 자체는 planning 개선의 증거가 아닙니다.", font=fonts[20], fill="#15243c")
    canvas.save(output / filename)


def main(arguments):
    assert not arguments.output.exists(), "Use a new output directory to preserve prior results"
    checked_sources = verify_registered_sources()
    assert total_card_bytes() < 42_000_000_000, "Training has priority; defer diagnostic"
    arguments.output.mkdir(parents=True)
    snapshot = arguments.output / "checkpoint_snapshot.pt"
    shutil.copyfile(arguments.checkpoint, snapshot)
    state = torch.load(snapshot, map_location="cpu", weights_only=False)
    updates = int(state["completed_updates"])
    assert updates == arguments.expected_updates, (updates, arguments.expected_updates)
    torch.set_num_threads(2)
    torch.manual_seed(2)
    torch.cuda.set_per_process_memory_fraction(3 * 1024**3 / torch.cuda.get_device_properties(0).total_memory)
    configuration = json.loads((ROOT / "configs/lpwm_drivor_planning_path_lora/navsim_v1.json").read_text())
    records = json.loads((MONITOR / "panel.json").read_text())["records"]
    images = np.load(MONITOR / "images.npy", mmap_mode="r")
    initial = np.array(np.load(MONITOR / "update_000000/particle_attributes.npy", mmap_mode="r")[:, :, 0])
    with redirect_stdout(io.StringIO()):
        model = LPWMDrivoRPlanningPathLoRAModel(ROOT / configuration["public_checkpoint"]).eval()
    native_digest = model.frozen_native_digest()
    model.to("cuda")
    first_rgb = torch.from_numpy(np.array(images[0, 0])).permute(2, 0, 1)[None].to("cuda").float() / 255
    first_ego = np.load(Path(records[0]["cache_directory"]) / "ego.npy", mmap_mode="r")[records[0]["cache_row"]]
    first_command = torch.tensor(np.array(first_ego[7:11]), device="cuda")[None]
    initial_replay = current_attributes(model.particle_encoder, first_rgb, first_command).cpu().numpy()[0]
    initial_error = float(np.max(np.abs(initial_replay - initial[0, 0])))
    assert initial_error < 1e-6, initial_error
    model.load_state_dict(state["model"], strict=True)
    assert native_digest == model.frozen_native_digest() == state["frozen_native_sha256"]
    del state
    with torch.inference_mode():
        complete_path = model.particle_encoder._encode_camera(first_rgb, first_command)[:, 0]
        geometry_path = current_attributes(model.particle_encoder, first_rgb, first_command)
        replay_error = float((complete_path - geometry_path).abs().max())
        assert replay_error < 1e-6, replay_error
    started = time.time()
    collected, maximum_card_bytes = [], total_card_bytes()
    for scene_index, record in enumerate(records):
        card_bytes = total_card_bytes()
        maximum_card_bytes = max(maximum_card_bytes, card_bytes)
        assert card_bytes < 46_500_000_000, "Stop diagnostic before the 48GB card limit"
        ego = np.load(Path(record["cache_directory"]) / "ego.npy", mmap_mode="r")[record["cache_row"]]
        command = torch.tensor(np.array(ego[7:11]), device="cuda")[None]
        camera_attributes = []
        for camera_index in range(4):
            rgb = torch.from_numpy(np.array(images[scene_index, camera_index])).permute(2, 0, 1)[None].to("cuda").float() / 255
            camera_attributes.append(current_attributes(model.particle_encoder, rgb, command).cpu().numpy()[0])
        collected.append(camera_attributes)
        if (scene_index + 1) % 16 == 0:
            print(f"GEOMETRY {scene_index + 1}/{len(records)} elapsed={time.time()-started:.1f}s", flush=True)
    trained = np.asarray(collected)
    assert trained.shape == initial.shape == (96, 4, 64, 14)
    assert np.isfinite(trained).all()
    assert model.frozen_native_digest() == native_digest
    reserved_bytes = torch.cuda.max_memory_reserved()
    del model
    torch.cuda.empty_cache()
    np.save(arguments.output / "current_particle_attributes.npy", trained)
    selections = [next(index for index, record in enumerate(records) if record["scene_type"] == scenario)
                  for scenario in ("straight", "left_turn", "right_turn")]
    report = {"completed_updates": updates, "epoch_fraction": updates / 1614,
              "checkpoint_sha256": file_digest(snapshot), "panel_sha256": file_digest(MONITOR / "panel.json"),
              "source_sha256": file_digest(Path(__file__)), "scene_count": 96, "camera_count": 4,
              "input_size": [128, 128], "inference_seconds": time.time() - started,
              "checked_registered_sources": checked_sources, "native_frozen_sha256": native_digest,
              "initial_replay_max_abs_difference": initial_error, "full_path_current_max_abs_difference": replay_error,
              "max_sampled_total_card_bytes": maximum_card_bytes, "max_diagnostic_reserved_bytes": reserved_bytes,
              "visualized_scene_indices": selections, "selection": "First scene per pre-existing scenario; independent of measured change",
              "scope": "Current particle geometry on fixed training panel; no planning or future improvement claim",
              "all_particles": geometry_changes(initial, trained),
              "by_camera": {label: geometry_changes(initial[:, index], trained[:, index]) for index, label in enumerate(CAMERA_LABELS)},
              "by_scene_type": {scenario: geometry_changes(initial[[r['scene_type'] == scenario for r in records]], trained[[r['scene_type'] == scenario for r in records]]) for scenario in SCENARIO_LABELS}}
    report["previous_update100"] = json.loads((MONITOR / "update_000100/geometry/geometry_report.json").read_text())["all_particles"]
    report["registered_sources_unchanged_after"] = verify_registered_sources()
    write_json(arguments.output / "geometry_report.json", report)
    render_comparison(arguments.output, "before_after_overview.png", [(index, 0) for index in selections], images, records, initial, trained, updates, report["all_particles"])
    for scene_index in selections:
        render_comparison(arguments.output, f"scene_{scene_index:03d}_four_cameras.png", [(scene_index, camera) for camera in range(4)], images, records, initial, trained, updates, report["all_particles"])
    before_centers, before_sizes, _, before_presence = particle_geometry(initial)
    after_centers, after_sizes, _, after_presence = particle_geometry(trained)
    with (arguments.output / "particle_geometry_changes.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["scene_token", "camera", "particle_index", "before_x", "before_y", "after_x", "after_y", "before_width", "before_height", "after_width", "after_height", "before_presence", "after_presence"])
        for scene_index, record in enumerate(records):
            for camera_index, label in enumerate(CAMERA_LABELS):
                for particle_index in range(64):
                    indices = (scene_index, camera_index, particle_index)
                    writer.writerow([record["token"], label, particle_index, *before_centers[indices], *after_centers[indices], *before_sizes[indices], *after_sizes[indices], before_presence[indices], after_presence[indices]])
    links = "".join(f'<li><a href="scene_{index:03d}_four_cameras.png">{html.escape(records[index]["scene_type"])}: 4카메라</a></li>' for index in selections)
    (arguments.output / "index.html").write_text(f'<!doctype html><html lang="ko"><meta charset="utf-8"><title>LPWM {updates} update particle 비교</title><style>body{{font-family:sans-serif;max-width:1600px;margin:24px auto}}img{{max-width:100%}}</style><h1>초기 / {updates} update 비교</h1><p>고정96장면·4카메라, 같은 명령. 위치/크기 변화는 planning 개선을 뜻하지 않습니다.</p><img src="before_after_overview.png"><ul>{links}</ul><a href="geometry_report.json">통계</a> · <a href="particle_geometry_changes.csv">모든 particle 수치</a></html>')
    write_json(arguments.output / "complete.json", {"complete": True, "completed_updates": updates, "completed_unix": time.time(), "checkpoint_sha256": report["checkpoint_sha256"], "training_changed": False})
    print(json.dumps({"output": str(arguments.output), "updates": updates, "all_particles": report["all_particles"]}, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, default=TRAINING / "navsim_v1/latest.pt")
    parser.add_argument("--expected-updates", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    main(parser.parse_args())
