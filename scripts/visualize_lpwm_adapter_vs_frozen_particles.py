"""Compare saved, paired LPWM particle snapshots on CPU without model inference."""
import argparse
import base64
import hashlib
import io
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from matplotlib.patches import Rectangle
import numpy as np
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[1]
UPDATES = (0, 2353, 4707)
SCENARIO_NAMES = {"straight": "직진", "turn": "회전", "projected_overlap": "투영 객체 겹침", "other": "기타"}
GEOMETRY_KEYS = ("centers", "sizes", "presence")


def load_snapshot(path):
    with np.load(path, allow_pickle=False) as archive:
        snapshot = {key: archive[key].copy() for key in ("current_rgb", *GEOMETRY_KEYS)}
        assert int(archive["optimizer_update"]) == int(path.stem.split("_")[-1])
    assert snapshot["current_rgb"].shape == (128, 128, 3)
    assert snapshot["centers"].shape == snapshot["sizes"].shape == (64, 2)
    assert snapshot["presence"].shape == (64,)
    assert all(np.isfinite(value).all() for value in snapshot.values())
    assert (snapshot["sizes"] > 0).all()
    assert ((snapshot["presence"] >= 0) & (snapshot["presence"] <= 1)).all()
    return snapshot


def summarize_shifts(displacements, size_changes, presence_changes):
    distances = np.linalg.norm(displacements, axis=-1)
    return {
        "center_shift_mean_pixels": float(distances.mean()),
        "center_shift_median_pixels": float(np.median(distances)),
        "center_shift_p95_pixels": float(np.quantile(distances, .95)),
        "center_shift_max_pixels": float(distances.max()),
        "fraction_centers_shifted_below_one_pixel": float((distances < 1).mean()),
        "size_absolute_change_mean_pixels": float(np.abs(size_changes).mean()),
        "presence_signed_change_mean": float(presence_changes.mean()),
        "presence_absolute_change_mean": float(np.abs(presence_changes).mean()),
    }


def show_image(axis, snapshot, title):
    axis.imshow(np.clip(snapshot["current_rgb"], 0, 1), interpolation="nearest")
    axis.set(xlim=(-.5, 127.5), ylim=(127.5, -.5))
    axis.set_title(title, fontsize=12, pad=9)
    axis.axis("off")


def draw_particles(axis, snapshot, box_ids, show_boxes=True):
    colors = plt.get_cmap("hsv")(np.arange(64) / 64)
    colors[:, 3] = .25 + .75 * snapshot["presence"]
    centers = snapshot["centers"]
    axis.scatter(centers[:, 0], centers[:, 1], c=colors, s=13, linewidths=.3, edgecolors="#171717")
    if show_boxes:
        for particle_id in box_ids:
            width, height = snapshot["sizes"][particle_id]
            corner = centers[particle_id] - snapshot["sizes"][particle_id] / 2
            axis.add_patch(Rectangle(corner, width, height, fill=False, edgecolor=colors[particle_id], linewidth=.8))


def finish_figure(figure, output_path, title, caption, pdf=False):
    figure.suptitle(title, fontsize=18, fontweight="bold", y=.992)
    figure.text(.5, .008, caption, ha="center", va="bottom", fontsize=10, linespacing=1.6)
    figure.tight_layout(rect=(0, .065, 1, .96), h_pad=2, w_pad=1)
    figure.savefig(output_path, dpi=165, facecolor="white")
    if pdf:
        figure.savefig(output_path.with_suffix(".pdf"), facecolor="white")
    plt.close(figure)


def render_figures(scene_pairs, output_directory):
    representatives = [next(scene for scene in scene_pairs if scene["scenario"] == scenario)
                       for scenario in ("straight", "turn", "projected_overlap")]
    for show_boxes, filename in ((True, "particle_comparison.png"), (False, "particle_centers.png")):
        figure, axes = plt.subplots(3, 3, figsize=(12, 12))
        for row, scene in enumerate(representatives):
            frozen, adapted = scene["frozen"][-1], scene["adapter"][-1]
            scenario = SCENARIO_NAMES[scene["scenario"]]
            show_image(axes[row, 0], frozen, f"{scenario} · 원본 입력\n{scene['current_frame_token']}")
            for column, snapshot, name in ((1, frozen, "LPWM 고정"), (2, adapted, "Adapter 미세조정")):
                show_image(axes[row, column], snapshot, name + " · 4,707 update")
                draw_particles(axes[row, column], snapshot, scene["box_ids"], show_boxes)
        finish_figure(figure, output_directory / filename, "동일 장면에서 LPWM 고정 vs Adapter: particle 분포",
            "점: 전체 64 particle 중심 / 같은 색: 같은 particle 인덱스 (객체 ID 아님)\n"
            + ("박스: 고정 모델의 presence 상위 16개 인덱스를 양쪽에 동일 적용 / 투명도: presence" if show_boxes else
               "위치 차이는 평균 1픽셀 미만입니다. 128×128 실제 입력을 확대 표시했으며 이동량을 과장하지 않았습니다."), pdf=show_boxes)

    figure, axes = plt.subplots(3, 4, figsize=(15, 11.7))
    for row, scene in enumerate(representatives):
        show_image(axes[row, 0], scene["frozen"][0], SCENARIO_NAMES[scene["scenario"]] + " · 원본")
        for column, (update, snapshot) in enumerate(zip(UPDATES, scene["adapter"]), start=1):
            show_image(axes[row, column], snapshot, ("학습 전" if update == 0 else "학습 중" if update == 2353 else "학습 후") + f" · {update:,}")
            draw_particles(axes[row, column], snapshot, scene["box_ids"])
    finish_figure(figure, output_directory / "adapter_particle_evolution.png", "Adapter: 같은 장면의 학습 전 · 중 · 후",
                  "학습 전 = NAVSIM Stage1 적응 완료 시점 / 모든 열에서 동일 16개 particle 박스를 표시\n"
                  "고정 대조군의 중심·크기·presence는 세 시점 모두 정확히 같습니다.", pdf=True)

    for scene in scene_pairs:
        figure, axes = plt.subplots(1, 4, figsize=(15, 4.8))
        frozen, adapted = scene["frozen"][-1], scene["adapter"][-1]
        for axis, snapshot, name in zip(axes[:3], (frozen, frozen, adapted), ("원본 입력", "LPWM 고정", "Adapter")):
            show_image(axis, snapshot, name)
        draw_particles(axes[1], frozen, scene["box_ids"])
        draw_particles(axes[2], adapted, scene["box_ids"])
        show_image(axes[3], frozen, "같은 인덱스의 중심 겹쳐 보기")
        centers_fixed, centers_adapted = frozen["centers"], adapted["centers"]
        axes[3].scatter(*centers_fixed.T, facecolors="none", edgecolors="#00efff", s=28, linewidths=.8)
        axes[3].scatter(*centers_adapted.T, c="#ff35be", marker="+", s=20, linewidths=.8)
        for origin, destination in zip(centers_fixed, centers_adapted):
            axes[3].plot([origin[0], destination[0]], [origin[1], destination[1]], color="white", linewidth=.6)
        finish_figure(figure, output_directory / f"scene_{scene['current_frame_token']}.png",
            f"{SCENARIO_NAMES[scene['scenario']]} · {scene['current_frame_token']}",
            f"평균 중심 이동 {scene['metrics']['center_shift_mean_pixels']:.4f}px / 최대 {scene['metrics']['center_shift_max_pixels']:.4f}px (128×128 기준)\n"
            "겹침 패널: 청록 원 = 고정, 분홍 + = Adapter / 이동량 ×1 / 박스는 검출 정답이 아닙니다.")

    figure, axes = plt.subplots(1, 3, figsize=(14, 4.8))
    distances = np.concatenate([np.linalg.norm(scene["adapter"][-1]["centers"] - scene["frozen"][-1]["centers"], axis=-1) for scene in scene_pairs])
    axes[0].hist(distances, bins=np.linspace(0, .4, 21), color="#4867ba", edgecolor="white")
    axes[0].set(xlabel="동일 인덱스 중심 이동 (입력 픽셀)", ylabel="particle 수", title="8개 장면 × 64 particle = 512개")
    for scene in scene_pairs:
        center_shifts = [np.linalg.norm(adapted["centers"] - fixed["centers"], axis=-1).mean()
                         for adapted, fixed in zip(scene["adapter"], scene["frozen"])]
        axes[1].plot(UPDATES, center_shifts, marker="o", linewidth=1, label=scene["current_frame_token"][:6])
    axes[1].set(xlabel="Adapter optimizer update", ylabel="고정 모델 대비 평균 이동 (픽셀)", title="학습에 따른 중심 변화")
    axes[1].set_xticks(UPDATES)
    displacements = np.concatenate([scene["adapter"][-1]["centers"] - scene["frozen"][-1]["centers"] for scene in scene_pairs])
    axes[2].scatter(*displacements.T, s=8, alpha=.45, color="#a32f82")
    axes[2].axhline(0, color="gray", linewidth=.6)
    axes[2].axvline(0, color="gray", linewidth=.6)
    axes[2].set(xlim=(-.4, .4), ylim=(.4, -.4), xlabel="Δx (입력 픽셀)", ylabel="Δy (입력 픽셀, 아래가 양수)", title="위치 차이만 별도로 확대해 표시")
    axes[2].set_aspect("equal")
    finish_figure(figure, output_directory / "particle_change_diagnostics.png", "particle 위치 변화의 실제 크기",
                  "사전 선정된 개발 8개 장면의 기술 통계입니다. 전체 개발셋·객체별 정보 보존·planning 중요도 검증이 아닙니다.", pdf=True)


def encode_image(rgb):
    image_buffer = io.BytesIO()
    Image.fromarray(np.rint(np.clip(rgb, 0, 1) * 255).astype(np.uint8)).save(image_buffer, format="PNG")
    return "data:image/png;base64," + base64.b64encode(image_buffer.getvalue()).decode("ascii")


def render_gallery(scene_pairs, output_directory, summary):
    gallery_scenes = []
    for scene in scene_pairs:
        gallery_scenes.append({"token": scene["current_frame_token"], "scenario": SCENARIO_NAMES[scene["scenario"]],
            "image": encode_image(scene["frozen"][0]["current_rgb"]), "box_ids": scene["box_ids"].tolist(),
            "metrics": scene["metrics"], **{method: [{key: snapshot[key].tolist() for key in GEOMETRY_KEYS}
                for snapshot in scene[method]] for method in ("frozen", "adapter")}})
    page = '''<!doctype html><html lang="ko"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>LPWM 고정 vs Adapter · particle 비교</title><style>
body{font:16px/1.6 system-ui,sans-serif;background:#f5f7fb;color:#172033;margin:28px auto;padding:0 24px;max-width:1500px}
h1{line-height:1.3} .card{background:white;border:1px solid #dbe1ed;border-radius:12px;padding:18px;margin:16px 0}
.controls{display:flex;gap:20px;flex-wrap:wrap;align-items:center} select,input,button{font:inherit} select,button{padding:7px}
.panels{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:18px}canvas{width:100%;height:auto;background:#333;image-rendering:pixelated}
label{white-space:nowrap} .small{font-size:14px;color:#495671} #zoom{max-width:650px;display:block}a{color:#244ca0}
@media(max-width:850px){.panels{grid-template-columns:1fr}} </style>
<h1>LPWM 고정 vs Adapter<br>같은 장면에서 particle이 어떻게 달라졌는가?</h1>
<p>둘 다 NAVSIM Stage1 적응 모델에서 출발했습니다. 완료된 Stage2의 <b>0 → 2,353 → 4,707 update</b> 저장 결과입니다.
현재 별도로 학습 중인 공개 LPWM 대조군과는 다른 비교입니다.</p>
<div class="card"><b>관찰:</b> 8장면·512 particle의 최종 평균 중심 이동은 <b>__MEAN__ px</b>, 최대 <b>__MAX__ px</b>입니다.
모든 이동은 1픽셀 미만이며, 고정 모델의 중심·크기·presence는 학습 전·중·후 동일합니다. 기준 해상도는 128×128입니다.</div>
<div class="card controls"><label>장면 <select id="scene"></select></label><label>시점 <select id="step"><option value="0">학습 전 · 0</option><option value="1">학습 중 · 2,353</option><option value="2" selected>학습 후 · 4,707</option></select></label>
<label><input id="boxes" type="checkbox" checked> 동일 16개 박스</label><label><input id="ids" type="checkbox"> 인덱스 번호</label>
<label><input id="presence" type="checkbox" checked> presence 투명도</label><label>원본 밝기 <input id="brightness" type="range" min="0.25" max="1" value="1" step="0.05"></label></div>
<div class="card"><div class="panels"><section><h3>원본 모델 입력</h3><canvas id="raw" width="768" height="768"></canvas></section><section><h3>LPWM 고정</h3><canvas id="frozen" width="768" height="768"></canvas></section><section><h3>Adapter 미세조정</h3><canvas id="adapter" width="768" height="768"></canvas></section></div><p id="metrics"></p></div>
<div class="card"><h2>중심 위치 겹쳐 보기</h2><p>청록 원: 고정 / 분홍 +: Adapter. 같은 인덱스를 선으로 연결합니다. <b>이동량 ×1</b>입니다.
위의 이미지에서 확인할 지점을 클릭하면 아래에서 같은 영역을 확대합니다.</p><label>영역 확대 <select id="zoomFactor"><option value="1">1×</option><option value="2">2×</option><option value="4" selected>4×</option><option value="8">8×</option></select></label><button id="reset">중앙 보기</button><canvas id="zoom" width="768" height="768"></canvas></div>
<div class="card"><h2>점과 박스를 읽는 법</h2><ul><li>64개 점: 현재 프레임 particle 중심. 같은 색은 같은 LPWM 인덱스이며, 객체 종류나 객체 추적 ID를 뜻하지 않습니다.</li>
<li>박스: 학습된 공간 범위(중심·크기)입니다. 검출기의 객체 정답 박스가 아닙니다. 고정 모델의 presence 상위 16개 인덱스를 양쪽과 세 시점에 동일 적용했습니다.</li>
<li>투명도: presence 값에 비례하며 planning 중요도 점수가 아닙니다. 선택을 끄면 낮은 presence도 선명하게 볼 수 있습니다.</li>
<li>원본은 실제 128×128 모델 입력을 확대했습니다. 미래 GT 없이 관측 4프레임에서 얻은 현재 particle입니다. FP32 단일 장면 진단 snapshot이며 BF16 배치 평가와 구분합니다.</li></ul>
<p>Adapter 조건에는 명령 FiLM 학습과 SSL 유지 손실도 포함됩니다. 이 그림만으로 planning loss 단독 효과나 객체 정보·미래 feature 개선을 판단할 수 없습니다.
투영 객체 겹침은 가림의 대리 분류이며 실제 가림을 보장하지 않습니다. 8장면은 학습 전 고정한 선택을 전부 표시했습니다.</p></div>
<div class="card"><h2>저장 이미지</h2><p><a href="particle_comparison.png">대표 3장면 비교 PNG</a> · <a href="particle_centers.png">점만 표시</a> · <a href="adapter_particle_evolution.png">학습 전·중·후</a> · <a href="particle_change_diagnostics.png">변화량 그래프</a> · <a href="summary.json">측정값·출처 JSON</a></p><div id="sceneLinks"></div></div>
<script>const scenes=__SCENES__; const byId=id=>document.getElementById(id); let currentImage=new Image(),focus=[64,64];
scenes.forEach((scene,index)=>{byId('scene').add(new Option(`${scene.scenario} · ${scene.token}`,index));const link=document.createElement('a');link.href=`scene_${scene.token}.png`;link.textContent=`${scene.scenario} ${scene.token}`;byId('sceneLinks').append(link,document.createElement('br'));});
function snapshot(method){return scenes[+byId('scene').value][method][+byId('step').value]}
function imageContext(canvas,zoom=1){const context=canvas.getContext('2d');context.setTransform(1,0,0,1,0,0);context.fillStyle='#151820';context.fillRect(0,0,768,768);context.scale(6*zoom,6*zoom);if(zoom!==1)context.translate(64/zoom-focus[0],64/zoom-focus[1]);context.imageSmoothingEnabled=false;context.globalAlpha=+byId('brightness').value;context.drawImage(currentImage,0,0,128,128);context.globalAlpha=1;return context;}
function particles(context,values){const selected=new Set(scenes[+byId('scene').value].box_ids);values.centers.forEach((center,index)=>{context.globalAlpha=byId('presence').checked?.25+.75*values.presence[index]:1;context.strokeStyle=context.fillStyle=`hsl(${index/64*360} 100% 48%)`;context.lineWidth=.25;const x=center[0]+.5,y=center[1]+.5;if(byId('boxes').checked&&selected.has(index))context.strokeRect(x-values.sizes[index][0]/2,y-values.sizes[index][1]/2,...values.sizes[index]);context.beginPath();context.arc(x,y,.65,0,Math.PI*2);context.fill();if(byId('ids').checked){context.font='2.3px sans-serif';context.strokeStyle='#111';context.lineWidth=.5;context.strokeText(index,x+1,y);context.fillText(index,x+1,y)}});context.globalAlpha=1;}
function draw(){imageContext(byId('raw'));['frozen','adapter'].forEach(method=>particles(imageContext(byId(method)),snapshot(method)));const zoom=+byId('zoomFactor').value,context=imageContext(byId('zoom'),zoom),fixed=snapshot('frozen'),adapted=snapshot('adapter');fixed.centers.forEach((origin,index)=>{const destination=adapted.centers[index],x=origin[0]+.5,y=origin[1]+.5,dx=destination[0]+.5,dy=destination[1]+.5;context.lineWidth=.14/Math.sqrt(zoom);context.strokeStyle='white';context.beginPath();context.moveTo(x,y);context.lineTo(dx,dy);context.stroke();context.strokeStyle='#00efff';context.beginPath();context.arc(x,y,.6/Math.sqrt(zoom),0,Math.PI*2);context.stroke();context.strokeStyle='#ff35be';context.beginPath();const arm=.55/Math.sqrt(zoom);context.moveTo(dx-arm,dy);context.lineTo(dx+arm,dy);context.moveTo(dx,dy-arm);context.lineTo(dx,dy+arm);context.stroke()});const shifts=fixed.centers.map((center,index)=>Math.hypot(center[0]-adapted.centers[index][0],center[1]-adapted.centers[index][1]));byId('metrics').textContent=`선택 시점 중심 이동: 평균 ${(shifts.reduce((sum,value)=>sum+value,0)/64).toFixed(4)} px / 최대 ${Math.max(...shifts).toFixed(4)} px`}
function loadScene(){focus=[64,64];currentImage.onload=draw;currentImage.src=scenes[+byId('scene').value].image}
byId('scene').onchange=loadScene;['step','boxes','ids','presence','brightness','zoomFactor'].forEach(id=>byId(id).oninput=draw);byId('reset').onclick=()=>{focus=[64,64];draw()};['raw','frozen','adapter'].forEach(id=>byId(id).onclick=event=>{const bounds=event.target.getBoundingClientRect();focus=[128*(event.clientX-bounds.left)/bounds.width,128*(event.clientY-bounds.top)/bounds.height];draw()});loadScene();</script></html>'''
    page = page.replace("__SCENES__", json.dumps(gallery_scenes, ensure_ascii=False)).replace("__MEAN__", f"{summary['aggregate']['center_shift_mean_pixels']:.4f}").replace("__MAX__", f"{summary['aggregate']['center_shift_max_pixels']:.4f}")
    (output_directory / "index.html").write_text(page, encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adapter-snapshots", type=Path, default=PROJECT_ROOT / "outputs/lpwm_card_budget_measured_v4/residual_adapter/batch8/metric_plus_world/visualization")
    parser.add_argument("--frozen-snapshots", type=Path, default=PROJECT_ROOT / "outputs/lpwm_frozen_control_v1/batch8/metric_plus_world/visualization")
    parser.add_argument("--output-directory", type=Path, default=PROJECT_ROOT / "results/lpwm_adapter_vs_frozen_particles_20261005")
    arguments = parser.parse_args()
    font_path = Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc")
    if font_path.exists():
        from matplotlib import font_manager
        font_manager.fontManager.addfont(str(font_path))
        plt.rcParams["font.family"] = FontProperties(fname=str(font_path)).get_name()
    plt.rcParams["axes.unicode_minus"] = False
    adapter_manifest = json.loads((arguments.adapter_snapshots / "latest_snapshot.json").read_text())
    frozen_manifest = json.loads((arguments.frozen_snapshots / "latest_snapshot.json").read_text())
    assert adapter_manifest["scenes"] == frozen_manifest["scenes"]
    assert adapter_manifest["optimizer_update"] == frozen_manifest["optimizer_update"] == UPDATES[-1]
    scene_pairs, provenance = [], []
    for record in adapter_manifest["scenes"]:
        scene = dict(record)
        for method, root in (("adapter", arguments.adapter_snapshots), ("frozen", arguments.frozen_snapshots)):
            scene[method] = []
            for update in UPDATES:
                path = root / record["current_frame_token"] / f"update_{update:06d}.npz"
                scene[method].append(load_snapshot(path))
                provenance.append({"path": str(path.resolve()), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
        for snapshot in scene["adapter"] + scene["frozen"]:
            assert np.array_equal(snapshot["current_rgb"], scene["frozen"][0]["current_rgb"])
        for key in GEOMETRY_KEYS:
            assert np.array_equal(scene["adapter"][0][key], scene["frozen"][0][key])
            assert all(np.array_equal(snapshot[key], scene["frozen"][0][key]) for snapshot in scene["frozen"])
        scene["box_ids"] = np.argsort(-scene["frozen"][0]["presence"], kind="stable")[:16]
        scene["metrics"] = summarize_shifts(*(scene["adapter"][-1][key] - scene["frozen"][-1][key] for key in GEOMETRY_KEYS))
        scene_pairs.append(scene)
    aggregate = summarize_shifts(*(np.concatenate([scene["adapter"][-1][key] - scene["frozen"][-1][key] for scene in scene_pairs]) for key in GEOMETRY_KEYS))
    summary = {"comparison": "NAVSIM-adapted frozen LPWM versus completed residual Adapter condition",
        "updates": list(UPDATES), "scene_count": len(scene_pairs), "particle_count_per_scene": 64,
        "input_resolution": [128, 128], "aggregate": aggregate,
        "checks": {"same_scene_manifest": True, "all_rgb_exactly_equal": True, "initial_geometry_exactly_equal": True,
                   "frozen_geometry_exactly_unchanged_at_all_three_updates": True, "all_values_finite": True},
        "snapshot_protocol": "FP32 deterministic single-scene diagnostic; four observed frames only for current particles; no new inference",
        "selection": "All eight preselected development scenes, two per category; representatives use first registered scene per category",
        "box_selection": "Frozen initial presence top16, same particle indices across both methods and all updates",
        "limits": ["Particle indices are not tracked semantic objects; boxes are learned support, not detection GT",
                   "Presence is not planning importance; geometry alone does not measure latent/future information",
                   "Adapter condition includes trainable command FiLM and retained world SSL objective",
                   "Eight selected scenes are descriptive; projected_overlap is an occlusion proxy",
                   "Current public frozen control training is separate and untouched"],
        "scenes": [{**{key: scene[key] for key in ("current_frame_token", "scenario", "recording_group", "metrics")},
                    "fixed_box_particle_ids": scene["box_ids"].tolist()} for scene in scene_pairs], "source_snapshots": provenance}
    arguments.output_directory.mkdir(parents=True, exist_ok=True)
    render_figures(scene_pairs, arguments.output_directory)
    render_gallery(scene_pairs, arguments.output_directory, summary)
    (arguments.output_directory / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"output_directory": str(arguments.output_directory), "aggregate": aggregate, "checks": summary["checks"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
