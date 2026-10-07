"""Publish all saved camera comparisons and diverse new examples without GPU work."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from publish_lpwm_particle_geometry_overlays import draw_geometry_overlay, font
from visualize_lpwm_largest_particle_changes import draw_particle_snapshot, file_sha256
from planning_aware_future_prediction.object_centric.particle_planning_diagnostics import particle_geometry

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CAMERA_NAMES = ("front", "back", "left", "right")
SCENARIO_NAMES = ("straight", "left_turn", "right_turn")
PREVIOUSLY_SHOWN_SCENES = {0, 2, 3, 21, 27, 41, 84, 94}


def describe_distribution(values):
    values = np.asarray(values, dtype=np.float64)
    return {"count": int(values.size), "mean": float(values.mean()), "median": float(np.median(values)),
            "p10": float(np.quantile(values, .1)), "p90": float(np.quantile(values, .9)),
            "minimum": float(values.min()), "maximum": float(values.max())}


def summarize_images(records):
    return {metric: describe_distribution([record[metric] for record in records]) for metric in
            ("center_mean_input_px", "size_mean_absolute_percent", "presence_before_mean", "presence_after_mean")}


def render_comparison(rgb_image, before, after, record, updates):
    # Keep the initial-presence top16 fixed; never replace them with trained top16.
    box_indices = np.argsort(-before[:, 4], kind="stable")[:16]
    overlay = draw_geometry_overlay(rgb_image, before, after, box_indices, 384)
    figure = Image.new("RGB", (1576, 552), "#f4f5f7")
    drawing = ImageDraw.Draw(figure)
    drawing.text((16, 10), f"Scene {record['scene_index']:02d} | {record['scenario']} | {record['camera']} camera | Before vs update {updates:,}", font=font(24), fill="black")
    drawing.text((16, 44), f"Mean position shift {record['center_mean_input_px']:.2f} input px | Mean size change {record['size_mean_absolute_percent']:.2f}% | Mean presence {record['presence_before_mean']:.3f} -> {record['presence_after_mean']:.3f}", font=font(20), fill="black")
    drawing.text((16, 75), "Cyan: before. Orange: after. White arrows: actual displacement. All 64 centers; fixed initial-presence top16 boxes.", font=font(18), fill="black")
    panels = (Image.fromarray(rgb_image).resize((384, 384), Image.Resampling.NEAREST),
              draw_particle_snapshot(rgb_image, before, box_indices, (0, 220, 255)),
              draw_particle_snapshot(rgb_image, after, box_indices, (255, 180, 0)), overlay)
    for column, (heading, picture) in enumerate(zip(("Model input", "Before training", f"Update {updates:,}", "Before / after overlay"), panels)):
        horizontal = 8 + column * 392
        drawing.text((horizontal, 110), heading, font=font(21), fill="black")
        figure.paste(picture, (horizontal, 143))
    drawing.text((16, 530), "Dot radii are fixed (not presence). Boxes are LPWM glimpses, not object detections. No displacement exaggeration.", font=font(16), fill="black")
    return figure, overlay, box_indices.tolist()


def choose_additional_front_examples(records):
    selected = []
    used_recordings = set()
    for scenario in SCENARIO_NAMES:
        candidates = [record for record in records if record["camera"] == "front" and record["scenario"] == scenario
                      and record["scene_index"] not in PREVIOUSLY_SHOWN_SCENES]
        assert len(candidates) >= 6
        candidates.sort(key=lambda record: (record["center_mean_input_px"], record["token"]))
        selected_scene_indices = set()
        for quantile in (.10, .25, .40, .60, .75, .90):
            eligible = [(rank, record) for rank, record in enumerate(candidates) if record["scene_index"] not in selected_scene_indices]
            rank, choice = min(eligible, key=lambda item: (abs(item[0] / (len(candidates) - 1) - quantile),
                                                         item[1]["recording_group"] in used_recordings, item[1]["token"]))
            selected.append({**choice, "target_quantile": quantile,
                             "selected_rank_quantile_among_new_scenario_front_images": rank / (len(candidates) - 1)})
            selected_scene_indices.add(choice["scene_index"])
            used_recordings.add(choice["recording_group"])
    assert len(selected) == 18 and len({record["scene_index"] for record in selected}) == 18
    return selected


def render_summary_plot(records, destination, updates):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    figure, axes = plt.subplots(2, 2, figsize=(13, 9), constrained_layout=True)
    center_values = np.array([record["center_mean_input_px"] for record in records])
    size_values = np.array([record["size_mean_absolute_percent"] for record in records])
    axes[0, 0].hist(center_values, bins=20, color="#1d7995", edgecolor="white")
    axes[0, 0].axvline(center_values.mean(), color="#dd9100", label=f"Mean {center_values.mean():.2f} px")
    axes[0, 0].set(xlabel="Mean center displacement per image (128x128 input px)", ylabel="Camera images", title="Position changes across all 384 images")
    axes[0, 0].legend()
    axes[0, 1].hist(size_values, bins=20, color="#1d7995", edgecolor="white")
    axes[0, 1].axvline(size_values.mean(), color="#dd9100", label=f"Mean {size_values.mean():.2f}%")
    axes[0, 1].set(xlabel="Mean absolute width/height relative change (%)", ylabel="Camera images", title="Glimpse size changes across all 384 images")
    axes[0, 1].legend()
    for camera, color in zip(CAMERA_NAMES, ("#247BA0", "#D27C00", "#6A994E", "#9B5DE5")):
        camera_records = [record for record in records if record["camera"] == camera]
        axes[1, 0].scatter([record["center_mean_input_px"] for record in camera_records],
                           [record["size_mean_absolute_percent"] for record in camera_records], s=16, alpha=.65, color=color, label=camera)
        axes[1, 1].scatter([record["presence_before_mean"] for record in camera_records],
                           [record["presence_after_mean"] for record in camera_records], s=16, alpha=.65, color=color, label=camera)
    axes[1, 0].set(xlabel="Mean center displacement (input px)", ylabel="Mean size change (%)", title="Position and size changes by camera")
    axes[1, 0].legend()
    axes[1, 1].plot([0, 1], [0, 1], "--", color="gray", linewidth=1)
    axes[1, 1].set(xlabel="Mean presence before training", ylabel=f"Mean presence at update {updates:,}", title="Presence is separate from geometry and planner importance", xlim=(0, 1), ylim=(0, 1))
    figure.suptitle(f"LPWM geometry | Initial vs update {updates:,} | 96 fixed scenes x 4 cameras\nDescriptive training-panel statistics; no planning-benefit claim", fontsize=14)
    figure.savefig(destination / "all_camera_geometry_summary.png", dpi=150)
    figure.savefig(destination / "all_camera_geometry_summary.pdf")
    plt.close(figure)


def publish_gallery(records, destination, updates):
    document = """<!doctype html><html lang="ko"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>LPWM particle 전후 비교</title><style>
body{font-family:system-ui,sans-serif;background:#f5f7fa;color:#1f2937;max-width:1500px;margin:24px auto;padding:0 20px}
h1{font-size:26px}p{line-height:1.65}.controls{display:flex;gap:16px;flex-wrap:wrap;align-items:center;padding:14px;background:white;border-radius:10px;position:sticky;top:0;z-index:2}
select,button{font:inherit;padding:7px}#cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(290px,1fr));gap:16px;margin:20px 0}
article{background:white;padding:12px;border-radius:10px}img{width:100%;height:auto}h2{font-size:17px}a{color:#145994}small{display:block;margin:8px 0}.paging{text-align:center;margin:20px}</style>
<h1>LPWM 학습 전 → __UPDATES__ update: 전체 384개 이미지</h1>
<p>96장면 × 전방·후방·좌측·우측. 청록색은 학습 전, 주황색은 학습 후, 흰 화살표는 같은 particle 번호의 위치 변화입니다.<br>
점은 64개 전부이며 반경은 고정입니다. 박스는 초기 presence 상위16개 particle 번호를 유지한 glimpse 범위입니다. 실제 객체 검출 박스나 attention이 아닙니다.<br>
전체 평균은 전체384개 이미지로 계산했습니다. 아래 필터와 정렬은 탐색용이며, 변화가 큰 사례를 전체 경향으로 해석하지 않습니다.</p>
<p><a href="front_camera_additional_18_overlays.png">새 전방18장면 한눈에 보기 PNG</a> · <a href="all_camera_geometry_summary.png">전체 변화량 그래프 PNG</a> · <a href="all_camera_image_changes.csv">전체 수치 CSV</a></p>
<div class="controls"><label>카메라 <select id="camera"><option value="front">전방</option><option value="all">전체</option><option value="back">후방</option><option value="left">좌측</option><option value="right">우측</option></select></label>
<label>장면 <select id="scenario"><option value="all">전체</option><option value="straight">직진</option><option value="left_turn">좌회전</option><option value="right_turn">우회전</option></select></label>
<label>정렬 <select id="sort"><option value="scene">장면 번호</option><option value="center_desc">위치 변화 큰 순</option><option value="center_asc">위치 변화 작은 순</option><option value="size_desc">크기 변화 큰 순</option><option value="presence_desc">presence 변화 큰 순</option></select></label><span id="count"></span></div>
<div id="cards"></div><div class="paging"><button id="previous">이전</button> <span id="page"></span> <button id="next">다음</button></div>
<script>const records=__RECORDS__;let page=0;const pageSize=24;
function render(){const camera=document.getElementById('camera').value,scenario=document.getElementById('scenario').value,sort=document.getElementById('sort').value;
let selected=records.filter(row=>(camera==='all'||row.camera===camera)&&(scenario==='all'||row.scenario===scenario));
selected.sort((a,b)=>sort==='center_desc'?b.center_mean_input_px-a.center_mean_input_px:sort==='center_asc'?a.center_mean_input_px-b.center_mean_input_px:sort==='size_desc'?b.size_mean_absolute_percent-a.size_mean_absolute_percent:sort==='presence_desc'?b.presence_mean_absolute_change-a.presence_mean_absolute_change:a.scene_index-b.scene_index||a.camera_index-b.camera_index);
const pages=Math.max(1,Math.ceil(selected.length/pageSize));page=Math.min(page,pages-1);document.getElementById('count').textContent=selected.length+'개 이미지';
document.getElementById('page').textContent=(page+1)+' / '+pages;document.getElementById('previous').disabled=page===0;document.getElementById('next').disabled=page>=pages-1;
document.getElementById('cards').innerHTML=selected.slice(page*pageSize,(page+1)*pageSize).map(row=>`<article><h2>Scene ${row.scene_index} · ${row.camera} · ${row.scenario}</h2><a href="${row.comparison_png}" target="_blank"><img loading="lazy" src="${row.overlay_png}" alt="학습 전후 particle 겹침"></a><small>평균 위치 ${row.center_mean_input_px.toFixed(2)}px · 평균 크기 ${row.size_mean_absolute_percent.toFixed(2)}%<br>평균 presence ${row.presence_before_mean.toFixed(3)} → ${row.presence_after_mean.toFixed(3)}</small><a href="${row.comparison_png}" target="_blank">입력 / 학습 전 / 학습 후 / 겹침 PNG</a></article>`).join('');}
for(const id of ['camera','scenario','sort'])document.getElementById(id).onchange=()=>{page=0;render();};document.getElementById('previous').onclick=()=>{page--;render();};document.getElementById('next').onclick=()=>{page++;render();};render();</script></html>"""
    document = document.replace("__UPDATES__", f"{updates:,}").replace("__RECORDS__", json.dumps(records, ensure_ascii=False).replace("<", "\\u003c"))
    (destination / "index.html").write_text(document)


def main(arguments):
    monitor, destination = arguments.monitor.resolve(), arguments.output.resolve()
    if destination.exists() and any(destination.iterdir()):
        raise FileExistsError(f"Preserve completed gallery: {destination}")
    checkpoint_root = monitor / f"update_{arguments.updates:06d}"
    completion = json.loads((checkpoint_root / "complete.json").read_text())
    assert completion["complete"] and completion["completed_updates"] == arguments.updates
    initial_path, trained_path = monitor / "update_000000/particle_attributes.npy", checkpoint_root / "particle_attributes.npy"
    before = np.asarray(np.load(initial_path, mmap_mode="r")[:, :, 0])
    after = np.asarray(np.load(trained_path, mmap_mode="r")[:, :, 0])
    images = np.load(monitor / "images.npy", mmap_mode="r")
    panel = json.loads((monitor / "panel.json").read_text())
    assert before.shape == after.shape == (96, 4, 64, 14) and images.shape == (96, 4, 128, 128, 3)
    assert np.isfinite(before).all() and np.isfinite(after).all()
    before_centers, before_sizes, _, before_presence = particle_geometry(before)
    after_centers, after_sizes, _, after_presence = particle_geometry(after)
    shifts = np.linalg.norm(after_centers - before_centers, axis=-1)
    size_changes = np.abs(after_sizes - before_sizes) / np.maximum(before_sizes, 1e-6) * 100
    destination.mkdir(parents=True)
    for directory_name in ("comparisons", "overlays", "scenario_sheets"):
        (destination / directory_name).mkdir()
    records = []
    for scene_index, scene in enumerate(panel["records"]):
        for camera_index, camera in enumerate(CAMERA_NAMES):
            artifact_name = f"scene_{scene_index:03d}_{camera}.png"
            record = {"scene_index": scene_index, "token": scene["token"], "recording_group": scene["recording_group"],
                      "scenario": scene["scene_type"], "camera_index": camera_index, "camera": camera,
                      "center_mean_input_px": float(shifts[scene_index, camera_index].mean()),
                      "center_max_input_px": float(shifts[scene_index, camera_index].max()),
                      "size_mean_absolute_percent": float(size_changes[scene_index, camera_index].mean()),
                      "presence_before_mean": float(before_presence[scene_index, camera_index].mean()),
                      "presence_after_mean": float(after_presence[scene_index, camera_index].mean()),
                      "presence_mean_absolute_change": float(np.abs(after_presence-before_presence)[scene_index, camera_index].mean()),
                      "comparison_png": f"comparisons/{artifact_name}", "overlay_png": f"overlays/{artifact_name}"}
            figure, overlay, particle_indices = render_comparison(images[scene_index, camera_index], before[scene_index, camera_index], after[scene_index, camera_index], record, arguments.updates)
            figure.save(destination / record["comparison_png"])
            overlay.save(destination / record["overlay_png"])
            record["box_particle_indices"] = particle_indices
            records.append(record)
        if (scene_index + 1) % 12 == 0:
            print(json.dumps({"rendered_images": len(records), "total_images": 384}), flush=True)
    featured = choose_additional_front_examples(records)
    for scenario in SCENARIO_NAMES:
        scenario_examples = [record for record in featured if record["scenario"] == scenario]
        for part in range(2):
            sheet = Image.new("RGB", (1576, 552 * 3), "#f4f5f7")
            for row_index, record in enumerate(scenario_examples[part*3:(part+1)*3]):
                with Image.open(destination / record["comparison_png"]) as figure:
                    sheet.paste(figure, (0, row_index * 552))
            sheet.save(destination / "scenario_sheets" / f"{scenario}_part{part+1}.png")
    overview = Image.new("RGB", (1616, 1180), "#f4f5f7")
    drawing = ImageDraw.Draw(overview)
    drawing.text((16, 12), f"18 additional front-camera scenes | Before vs update {arguments.updates:,}", font=font(26), fill="black")
    drawing.text((16, 51), "Rows: straight / left turn / right turn. Within each row: smaller -> larger mean center displacement.", font=font(20), fill="black")
    drawing.text((16, 84), "Cyan before / orange after. Fixed initial-presence top16 boxes, all64 centers, fixed marker radii, no magnification.", font=font(18), fill="black")
    for row_index, scenario in enumerate(SCENARIO_NAMES):
        for column, record in enumerate(record for record in featured if record["scenario"] == scenario):
            horizontal, vertical = 8 + column * 268, 125 + row_index * 346
            drawing.text((horizontal, vertical), f"{scenario} | Scene {record['scene_index']}", font=font(18), fill="black")
            drawing.text((horizontal, vertical + 28), f"Shift {record['center_mean_input_px']:.2f}px | Size {record['size_mean_absolute_percent']:.1f}%", font=font(17), fill="black")
            scene_index = record["scene_index"]
            picture = draw_geometry_overlay(images[scene_index, 0], before[scene_index, 0], after[scene_index, 0], record["box_particle_indices"], 256)
            overview.paste(picture, (horizontal, vertical + 58))
    overview.save(destination / "front_camera_additional_18_overlays.png")
    render_summary_plot(records, destination, arguments.updates)
    publish_gallery(records, destination, arguments.updates)
    csv_fields = [key for key in records[0] if key != "box_particle_indices"]
    with (destination / "all_camera_image_changes.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=csv_fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(records)
    source_paths = [Path(__file__), PROJECT_ROOT / "scripts/visualize_lpwm_largest_particle_changes.py", PROJECT_ROOT / "scripts/publish_lpwm_particle_geometry_overlays.py", PROJECT_ROOT / "src/planning_aware_future_prediction/object_centric/particle_planning_diagnostics.py"]
    report = {"complete": True, "completed_updates": arguments.updates, "checkpoint_sha256": completion["checkpoint_sha256"],
              "source_sha256": {str(path.relative_to(PROJECT_ROOT)): file_sha256(path) for path in source_paths},
              "panel_sha256": file_sha256(monitor / "panel.json"), "images_sha256": file_sha256(monitor / "images.npy"),
              "initial_attributes_sha256": file_sha256(initial_path), "trained_attributes_sha256": file_sha256(trained_path),
              "scenes": 96, "camera_images": len(records), "cpu_only": True, "training_changes": False,
              "all_images": summarize_images(records),
              "by_camera": {camera: summarize_images([record for record in records if record["camera"] == camera]) for camera in CAMERA_NAMES},
              "by_scenario": {scenario: summarize_images([record for record in records if record["scenario"] == scenario]) for scenario in SCENARIO_NAMES},
              "featured_selection": "18 new front images: six per scenario near mean-center-shift rank quantiles .10/.25/.40/.60/.75/.90 after excluding previously shown scenes. Size and presence are not selection criteria.",
              "excluded_previously_shown_scene_indices_for_featured_only": sorted(PREVIOUSLY_SHOWN_SCENES),
              "featured_examples": featured,
              "scope": "All 384 images in the existing fixed training diagnostic panel; not representative of all NAVSIM. Geometry and presence do not establish planning importance or benefit.",
              "display": {"centers": 64, "box_selection": "fixed highest16 initial-presence indices", "presence_encoded_in_dot_radius": False, "displacement_magnification": 1.0}}
    (destination / "gallery_report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"complete": True, "output": str(destination), "all_images": report["all_images"], "featured_scene_indices": [record["scene_index"] for record in featured]}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--monitor", type=Path, default=PROJECT_ROOT / "outputs/lpwm_drivor_planning_path_representation_monitor_v1")
    parser.add_argument("--updates", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    main(parser.parse_args())
