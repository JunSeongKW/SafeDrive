"""CPU visualization and saved-weight audit of partial/full LPWM geometry tuning."""
import argparse
import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np

from visualize_lpwm_adapter_vs_frozen_particles import (
    PROJECT_ROOT, UPDATES, GEOMETRY_KEYS, SCENARIO_NAMES, load_snapshot,
    summarize_shifts, show_image, draw_particles, finish_figure, encode_image,
)


RUN_ROOTS = {
    "frozen": PROJECT_ROOT / "outputs/lpwm_frozen_control_v1/batch8/metric_plus_world",
    "partial": PROJECT_ROOT / "outputs/lpwm_48gb_planning_v1/partial_output_layers/batch8/metric_plus_world",
    "full": PROJECT_ROOT / "outputs/lpwm_card_budget_measured_v4/full_low_learning_rate/batch4/metric_plus_world",
}
FULL_INITIAL_ROOT = PROJECT_ROOT / "outputs/lpwm_object_future_planning_v3/metric_plus_world/visualization"
METHOD_NAMES = {"frozen": "LPWM 고정", "partial": "일부 계층 미세조정", "full": "전체 미세조정"}


def snapshot_path(method, token, update):
    root = FULL_INITIAL_ROOT if method == "full" and update == 0 else RUN_ROOTS[method] / "visualization"
    return root / token / f"update_{update:06d}.npz"


def audit_geometry_weights():
    import torch

    torch.set_num_threads(1)
    stage1_path = PROJECT_ROOT / "outputs/lpwm_navsim_full_posttraining_v2/stage1/checkpoint.pt"
    initial_state = torch.load(stage1_path, map_location="cpu", weights_only=True, mmap=True)
    reports = {}
    for method in ("partial", "full"):
        checkpoint_path = RUN_ROOTS[method] / "checkpoint.pt"
        state = torch.load(checkpoint_path, map_location="cpu", weights_only=True, mmap=True)
        inventory = json.loads((RUN_ROOTS[method] / "parameter_inventory.json").read_text())
        trainable_names = set(inventory["trainable_parameter_names"])
        report = {"checkpoint": str(checkpoint_path), "heads": {}}
        for head in ("xy_head", "scale_xy_head", "obj_on_head"):
            prefix = f"world_model.encoder_module.particle_enc.particle_attribute_enc.{head}."
            selected = {name: value for name, value in state.items() if name.startswith(prefix)}
            assert selected and all(name in trainable_names for name in selected)
            changed_elements, total_elements, maximum_change = 0, 0, 0.
            for name, value in selected.items():
                difference = value - initial_state[name.removeprefix("world_model.")]
                changed_elements += int((difference != 0).sum())
                total_elements += value.numel()
                maximum_change = max(maximum_change, float(difference.abs().max()))
            assert changed_elements > 0
            report["heads"][head] = {"all_parameters_registered_trainable": True,
                "changed_elements": changed_elements, "total_elements": total_elements,
                "max_absolute_weight_change": maximum_change}
        reports[method] = report
        del state
    return {"initial_checkpoint": str(stage1_path), "methods": reports,
            "interpretation": "Trainable heads and actual changes verified; joint planning + SSL + optimizer regularization changes, not planning-only attribution."}


def render_comparisons(scenes, destination):
    representatives = [next(scene for scene in scenes if scene["scenario"] == scenario)
                       for scenario in ("straight", "turn", "projected_overlap")]
    for boxes, filename in ((True, "geometry_comparison.png"), (False, "geometry_centers.png")):
        figure, axes = plt.subplots(3, 4, figsize=(15, 11.6))
        for row, scene in enumerate(representatives):
            show_image(axes[row, 0], scene["frozen"][0], f"{SCENARIO_NAMES[scene['scenario']]} · 원본\n{scene['current_frame_token']}")
            for column, method in enumerate(RUN_ROOTS, start=1):
                snapshot = scene[method][-1]
                subtitle = "학습 전과 동일" if method == "frozen" else f"평균 이동 {scene['metrics'][method]['center_shift_mean_pixels']:.3f}px"
                show_image(axes[row, column], snapshot, METHOD_NAMES[method] + "\n" + subtitle)
                draw_particles(axes[row, column], snapshot, scene["box_ids"], boxes)
        finish_figure(figure, destination / filename, "위치 head도 학습한 LPWM: 고정군 · 일부 계층 · 전체 미세조정",
            "동일 개발 장면 / 최종 4,707 update / 전체 64개 중심 / 128×128 입력의 실제 좌표\n"
            "같은 색 = 같은 인덱스 (객체 ID 아님) / 박스는 고정군 상위 16개 인덱스를 공통 적용 / 이동량 과장 없음", pdf=boxes)

    for method in ("partial", "full"):
        figure, axes = plt.subplots(3, 4, figsize=(15, 11.6))
        for row, scene in enumerate(representatives):
            show_image(axes[row, 0], scene[method][0], SCENARIO_NAMES[scene["scenario"]] + " · 원본")
            for column, (update, snapshot) in enumerate(zip(UPDATES, scene[method]), start=1):
                stage = "학습 전" if update == 0 else "학습 중" if update == 2353 else "학습 후"
                show_image(axes[row, column], snapshot, f"{stage} · {update:,}")
                draw_particles(axes[row, column], snapshot, scene["box_ids"])
        caption = "학습 전 = NAVSIM Stage1 적응 완료 / 같은 장면·같은 인덱스 / planning + SSL 공동학습"
        if method == "full":
            caption += "\n학습 전은 원래 full 실행의 snapshot, 이후는 2,095 update checkpoint를 이어받은 실행입니다."
        finish_figure(figure, destination / f"{method}_particle_evolution.png", METHOD_NAMES[method] + ": 학습 전 · 중 · 후", caption, pdf=True)

    for scene in scenes:
        figure, axes = plt.subplots(2, 3, figsize=(12, 8.8))
        for row, method in enumerate(("partial", "full")):
            initial, final = scene[method][0], scene[method][-1]
            show_image(axes[row, 0], initial, METHOD_NAMES[method] + " · 학습 전")
            show_image(axes[row, 1], final, "학습 후 · 4,707")
            draw_particles(axes[row, 0], initial, scene["box_ids"])
            draw_particles(axes[row, 1], final, scene["box_ids"])
            show_image(axes[row, 2], initial, "중심 이동 · 실제 크기")
            for origin, target in zip(initial["centers"], final["centers"]):
                axes[row, 2].plot([origin[0], target[0]], [origin[1], target[1]], color="white", linewidth=.8)
            axes[row, 2].scatter(*initial["centers"].T, facecolors="none", edgecolors="#00efff", s=30, linewidths=.8)
            axes[row, 2].scatter(*final["centers"].T, c="#ff35be", marker="+", s=25, linewidths=.8)
        finish_figure(figure, destination / f"scene_{scene['current_frame_token']}.png",
            f"{SCENARIO_NAMES[scene['scenario']]} · {scene['current_frame_token']}",
            "중심 겹침: 청록 원 = 학습 전 / 분홍 + = 학습 후 / 흰 선 = 같은 인덱스의 위치 차이\n"
            "박스는 학습된 공간 범위이며 객체 검출 박스가 아닙니다. Presence는 planning 중요도가 아닙니다.")

    figure, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    for method, color in (("partial", "#365fb4"), ("full", "#a43178")):
        distances = np.concatenate([np.linalg.norm(scene[method][-1]["centers"] - scene[method][0]["centers"], axis=-1) for scene in scenes])
        axes[0].hist(distances, bins=np.linspace(0, 2.6, 27), histtype="step", linewidth=2, color=color, label=METHOD_NAMES[method])
        mean_shifts = [np.mean([np.linalg.norm(scene[method][step]["centers"] - scene[method][0]["centers"], axis=-1).mean() for scene in scenes]) for step in range(3)]
        axes[1].plot(UPDATES, mean_shifts, marker="o", color=color, label=METHOD_NAMES[method])
    axes[0].set(xlabel="학습 전 대비 중심 이동 (입력 픽셀)", ylabel="particle 수", title="8장면 × 64개 중심의 이동")
    axes[1].set(xlabel="Optimizer update", ylabel="평균 중심 이동 (입력 픽셀)", title="학습 전·중·후 위치 변화")
    axes[1].set_xticks(UPDATES)
    for axis in axes:
        axis.legend(); axis.grid(alpha=.2)
    finish_figure(figure, destination / "geometry_change_diagnostics.png", "particle 위치는 실제로 바뀌었습니다",
                  "이동의 크기는 유용성 점수가 아닙니다. Planning 단독 효과·객체 의미·성능 개선은 이 그림만으로 판단할 수 없습니다.", pdf=True)


def render_gallery(scenes, destination, aggregate):
    payload = [{"token": scene["current_frame_token"], "scenario": SCENARIO_NAMES[scene["scenario"]],
        "image": encode_image(scene["frozen"][0]["current_rgb"]), "box_ids": scene["box_ids"].tolist(),
        **{method: [{key: snapshot[key].tolist() for key in GEOMETRY_KEYS} for snapshot in scene[method]] for method in RUN_ROOTS}} for scene in scenes]
    page = '''<!doctype html><html lang="ko"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>LPWM 일부 계층·전체 미세조정 particle</title><style>
body{font:16px/1.6 system-ui,sans-serif;color:#18243b;background:#f5f7fb;max-width:1600px;margin:28px auto;padding:0 22px}h1{line-height:1.3}
.card{background:white;padding:20px;margin:18px 0;border:1px solid #dce2ee;border-radius:12px}.controls{display:flex;gap:20px;flex-wrap:wrap;align-items:center}
.panels{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:14px}canvas{width:100%;height:auto;background:#222;image-rendering:pixelated}
#zoom{max-width:650px;display:block}select,button,input{font:inherit}select,button{padding:6px}a{color:#2855a0}.small{color:#526079;font-size:14px}
@media(max-width:950px){.panels{grid-template-columns:repeat(2,minmax(0,1fr))}}@media(max-width:500px){.panels{grid-template-columns:1fr}}</style>
<h1>위치 head도 학습한 LPWM<br>일부 계층·전체 미세조정의 particle 변화</h1>
<p>두 조건은 위치·크기·presence head를 학습했습니다. 실제 checkpoint 가중치 변화도 확인했습니다.
손실은 planning + 0.02 SSL이며, 이 그림은 planning 단독의 인과효과를 분리한 결과가 아닙니다.</p>
<div class="card">학습 전 대비 최종 중심 이동: <b>일부 계층 평균 __PARTIAL__px / 전체 평균 __FULL__px</b> (128×128 입력 기준, 8장면·512 particle).
학습 전 geometry는 두 방법과 고정군이 정확히 같습니다. 전체 미세조정은 원래 실행의 2,095 update checkpoint를 이어받았으며 LR·schedule·FiLM 위치가 일부 계층 조건과 다릅니다.</div>
<div class="card controls"><label>장면 <select id="scene"></select></label><label>시점 <select id="step"><option value="0">학습 전 · 0</option><option value="1">학습 중 · 2,353</option><option value="2" selected>학습 후 · 4,707</option></select></label>
<label><input id="boxes" type="checkbox" checked> 동일 16개 박스</label><label><input id="ids" type="checkbox"> 인덱스 번호</label>
<label><input id="presence" type="checkbox" checked> presence 투명도</label></div>
<div class="card"><div class="panels"><section><h3>원본 입력</h3><canvas id="raw" width="768" height="768"></canvas></section><section><h3>LPWM 고정</h3><canvas id="frozen" width="768" height="768"></canvas></section><section><h3>일부 계층</h3><canvas id="partial" width="768" height="768"></canvas></section><section><h3>전체 미세조정</h3><canvas id="full" width="768" height="768"></canvas></section></div><p id="metrics"></p></div>
<div class="card"><h2>같은 인덱스의 중심 이동 확대</h2><p>청록 원 = 학습 전 / 분홍 + = 선택 시점 / 흰 선 = 이동. 이동량을 인위적으로 키우지 않았습니다.
위의 이미지에서 보고 싶은 위치를 클릭하면 아래 영역이 바뀝니다.</p>
<label>방법 <select id="method"><option value="partial">일부 계층</option><option value="full">전체 미세조정</option></select></label>
<label>영역 확대 <select id="zoomFactor"><option value="1">1×</option><option value="2">2×</option><option value="4" selected>4×</option><option value="8">8×</option></select></label><button id="reset">중앙 보기</button><canvas id="zoom" width="768" height="768"></canvas></div>
<div class="card"><h2>해석 범위</h2><ul><li>점은 전체 64개 particle 중심, 같은 색은 같은 인덱스입니다. 객체 종류나 추적 ID가 아닙니다.</li>
<li>박스는 학습된 공간 범위입니다. 고정군 초기 presence 상위 16개 인덱스를 모든 방법·시점에 동일하게 표시합니다.</li>
<li>관측 4프레임만으로 얻은 현재 particle의 FP32 단일 장면 snapshot입니다. 현재 좌표 계산에 미래 GT를 넣지 않았습니다.</li>
<li>8장면은 기존 사전 선정 개발 장면 전부입니다. 투영 겹침은 실제 가림을 보장하지 않습니다.</li>
<li>점이 이동했다는 사실은 차량·보행자의 정보 보존이나 PDMS 개선을 의미하지 않습니다. 고정군 대비 추가 PDMS 이득은 두 방법 모두 확인되지 않았습니다.</li></ul></div>
<div class="card"><h2>그림과 측정값</h2><a href="geometry_comparison.png">최종 비교</a> · <a href="geometry_centers.png">점만 보기</a> · <a href="partial_particle_evolution.png">일부 계층 전·중·후</a> · <a href="full_particle_evolution.png">전체 전·중·후</a> · <a href="geometry_change_diagnostics.png">이동량 그래프</a> · <a href="summary.json">측정값·가중치 확인</a><div id="links"></div></div>
<script>const scenes=__SCENES__,byId=id=>document.getElementById(id);let currentImage=new Image(),focus=[64,64];
scenes.forEach((scene,index)=>{byId('scene').add(new Option(`${scene.scenario} · ${scene.token}`,index));const link=document.createElement('a');link.href=`scene_${scene.token}.png`;link.textContent=`${scene.scenario} ${scene.token}`;byId('links').append(link,document.createElement('br'))});
function selectedScene(){return scenes[+byId('scene').value]}function snapshot(method,step=+byId('step').value){return selectedScene()[method][step]}
function imageContext(canvas,zoom=1){const context=canvas.getContext('2d');context.setTransform(1,0,0,1,0,0);context.fillStyle='#182031';context.fillRect(0,0,768,768);context.scale(6*zoom,6*zoom);if(zoom!==1)context.translate(64/zoom-focus[0],64/zoom-focus[1]);context.imageSmoothingEnabled=false;context.drawImage(currentImage,0,0,128,128);return context;}
function particles(context,values){const selected=new Set(selectedScene().box_ids);values.centers.forEach((center,index)=>{context.globalAlpha=byId('presence').checked ? .25+.75*values.presence[index] : 1;context.fillStyle=context.strokeStyle=`hsl(${index/64*360} 100% 48%)`;context.lineWidth=.25;const x=center[0]+.5,y=center[1]+.5;if(byId('boxes').checked&&selected.has(index))context.strokeRect(x-values.sizes[index][0]/2,y-values.sizes[index][1]/2,...values.sizes[index]);context.beginPath();context.arc(x,y,.65,0,Math.PI*2);context.fill();if(byId('ids').checked){context.font='2.3px sans-serif';context.strokeStyle='#111';context.lineWidth=.5;context.strokeText(index,x+1,y);context.fillText(index,x+1,y)}});context.globalAlpha=1;}
function shifts(method){return snapshot(method,0).centers.map((origin,index)=>Math.hypot(origin[0]-snapshot(method).centers[index][0],origin[1]-snapshot(method).centers[index][1]))}
function draw(){imageContext(byId('raw'));['frozen','partial','full'].forEach(method=>particles(imageContext(byId(method)),snapshot(method)));const method=byId('method').value,zoom=+byId('zoomFactor').value,context=imageContext(byId('zoom'),zoom);snapshot(method,0).centers.forEach((origin,index)=>{const destination=snapshot(method).centers[index],x=origin[0]+.5,y=origin[1]+.5,dx=destination[0]+.5,dy=destination[1]+.5;context.lineWidth=.18/Math.sqrt(zoom);context.strokeStyle='white';context.beginPath();context.moveTo(x,y);context.lineTo(dx,dy);context.stroke();context.strokeStyle='#00efff';context.beginPath();context.arc(x,y,.7/Math.sqrt(zoom),0,Math.PI*2);context.stroke();context.strokeStyle='#ff35be';context.beginPath();const arm=.6/Math.sqrt(zoom);context.moveTo(dx-arm,dy);context.lineTo(dx+arm,dy);context.moveTo(dx,dy-arm);context.lineTo(dx,dy+arm);context.stroke()});byId('metrics').textContent=['partial','full'].map(method=>{const distances=shifts(method);return `${method==='partial'?'일부 계층':'전체'}: 평균 ${(distances.reduce((sum,value)=>sum+value,0)/64).toFixed(3)}px / 최대 ${Math.max(...distances).toFixed(3)}px`}).join(' | ')}
function loadScene(){focus=[64,64];currentImage.onload=draw;currentImage.src=selectedScene().image}byId('scene').onchange=loadScene;
['step','boxes','ids','presence','method','zoomFactor'].forEach(id=>byId(id).oninput=draw);byId('reset').onclick=()=>{focus=[64,64];draw()};['raw','frozen','partial','full'].forEach(id=>byId(id).onclick=event=>{const bounds=event.target.getBoundingClientRect();focus=[128*(event.clientX-bounds.left)/bounds.width,128*(event.clientY-bounds.top)/bounds.height];draw()});loadScene();</script></html>'''
    page = page.replace("__SCENES__", json.dumps(payload, ensure_ascii=False)).replace("__PARTIAL__", f"{aggregate['partial']['center_shift_mean_pixels']:.3f}").replace("__FULL__", f"{aggregate['full']['center_shift_mean_pixels']:.3f}")
    (destination / "index.html").write_text(page, encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-directory", type=Path, default=PROJECT_ROOT / "results/lpwm_geometry_finetuning_visualization_20261005")
    arguments = parser.parse_args()
    font_path = Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc")
    if font_path.exists():
        font_manager.fontManager.addfont(str(font_path))
        plt.rcParams["font.family"] = font_manager.FontProperties(fname=str(font_path)).get_name()
    plt.rcParams["axes.unicode_minus"] = False
    manifests = {method: json.loads((root / "visualization/latest_snapshot.json").read_text()) for method, root in RUN_ROOTS.items()}
    assert all(manifest["scenes"] == manifests["frozen"]["scenes"] and manifest["optimizer_update"] == 4707 for manifest in manifests.values())
    scenes, provenance = [], []
    for record in manifests["frozen"]["scenes"]:
        scene = dict(record)
        for method in RUN_ROOTS:
            scene[method] = []
            for update in UPDATES:
                path = snapshot_path(method, record["current_frame_token"], update)
                scene[method].append(load_snapshot(path))
                provenance.append({"method": method, "update": update, "path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
            assert all(np.array_equal(snapshot["current_rgb"], scene["frozen"][0]["current_rgb"]) for snapshot in scene[method])
            assert all(np.array_equal(scene[method][0][key], scene["frozen"][0][key]) for key in GEOMETRY_KEYS)
        assert all(np.array_equal(snapshot[key], scene["frozen"][0][key]) for snapshot in scene["frozen"] for key in GEOMETRY_KEYS)
        scene["box_ids"] = np.argsort(-scene["frozen"][0]["presence"], kind="stable")[:16]
        scene["metrics"] = {method: summarize_shifts(*(scene[method][-1][key] - scene[method][0][key] for key in GEOMETRY_KEYS)) for method in RUN_ROOTS}
        scenes.append(scene)
    aggregate = {method: summarize_shifts(*(np.concatenate([scene[method][-1][key] - scene[method][0][key] for scene in scenes]) for key in GEOMETRY_KEYS)) for method in RUN_ROOTS}
    summary = {"scope": "Same eight preselected development scenes; current particles, not future-particle trajectories or semantic labels",
        "updates": list(UPDATES), "input_resolution": [128, 128], "aggregate": aggregate,
        "checks": {"scene_manifests_match": True, "input_rgb_exactly_equal": True, "initial_geometry_exactly_equal": True,
                   "frozen_geometry_exactly_unchanged": True, "all_values_finite": True},
        "geometry_weight_audit": audit_geometry_weights(),
        "full_initial_provenance": "Original full execution update0; final/mid snapshots from continuation of original update2095, not fresh full training",
        "loss": "candidate imitation + six PDM subscore BCE + 0.02 original SSL; no direct object GT auxiliary",
        "comparability_limits": "Full uses conv_in FiLM, world LR1e-6, original20epoch schedule; partial conv_out FiLM, world LR1e-5, one-epoch schedule; physical microbatch histories differ",
        "interpretation": "Geometry changed; planning-only attribution, movement toward useful objects and performance gains not demonstrated",
        "snapshot_protocol": "Saved deterministic FP32 snapshots from four observed images, same ego input; CPU-only rendering; fixed baseline top16 box indices",
        "scenes": [{**{key: scene[key] for key in ("current_frame_token", "scenario", "recording_group", "metrics")}, "box_particle_ids": scene["box_ids"].tolist()} for scene in scenes],
        "source_snapshots": provenance}
    arguments.output_directory.mkdir(parents=True, exist_ok=True)
    render_comparisons(scenes, arguments.output_directory)
    render_gallery(scenes, arguments.output_directory, aggregate)
    (arguments.output_directory / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"directory": str(arguments.output_directory), "aggregate": aggregate, "checks": summary["checks"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
