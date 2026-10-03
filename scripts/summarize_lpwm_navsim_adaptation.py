"""Aggregate fixed LPWM trials and render actual particle decompositions and forecasts."""
import json
import sys
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import Rectangle
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
from planning_aware_future_prediction.object_centric.lpwm_bridge import ARTIFACT_ROOT, extrapolated_rotation_homographies

SHARED_ROOT = PROJECT_ROOT / "results/lpwm_navsim_adaptation_v1"
FIGURE_ROOT = ARTIFACT_ROOT / "visualization"
SCENARIOS = ["straight", "turn", "projected_overlap"]
SCENARIO_NAMES = ["직진", "회전", "객체 겹침·가림 후보"]
SEEDS = [29, 47, 71]


def evaluation_directory(name):
    supplement = ARTIFACT_ROOT / "supplementary_evaluation" / name
    return supplement if (supplement / "metrics.json").exists() else ARTIFACT_ROOT / "evaluation" / name


def mean_metric(rows, metric):
    values = [row["metrics"][metric] for row in rows if row["metrics"].get(metric) is not None]
    return float(np.mean(values)) if values else None


def add_common_area_metrics(result, manifest_lookup):
    for row in result["records"]:
        record = manifest_lookup[row["token"]]
        clip = np.load(PROJECT_ROOT / record["clip_path"])
        predictions = np.load(evaluation_directory(result["name"]) / (row["token"] + ".npz"))
        targets = clip["raw_images"][:8].astype(np.float32) / 255
        reconstruction_masks = np.array([cv2.warpPerspective(np.ones((128, 128), np.float32), np.linalg.inv(transform), (128, 128)) > .999
                                         for transform in clip["observed_rotation_homographies"][:8]])
        extrapolated = extrapolated_rotation_homographies(clip["camera_rotations"][:4], clip["camera_intrinsics"][:4], 4)
        prediction_masks = np.array([cv2.warpPerspective(np.ones((128, 128), np.float32), np.linalg.inv(transform), (128, 128)) > .999 for transform in extrapolated])
        row["metrics"]["common_area_reconstruction_mse"] = float(((predictions["reconstruction"] - targets)**2)[reconstruction_masks].mean())
        row["metrics"]["common_area_forecast_mse"] = float(((predictions["prediction"] - targets[4:])**2)[prediction_masks].mean())
        row["metrics"]["common_area_persistence_mse"] = float(((targets[3] - targets[4:])**2)[prediction_masks].mean())
        row["metrics"]["common_area_reconstruction_fraction"] = float(reconstruction_masks.mean())
        row["metrics"]["common_area_forecast_fraction"] = float(prediction_masks.mean())


def group_summaries(results):
    metrics = sorted(set.union(*(set(row["metrics"]) for result in results for row in result["records"])))
    metrics = [metric for metric in metrics if not metric.startswith("future_intervention")]
    summaries = {}
    for label, names in {
        "zero_shot_raw": ["zero_shot_raw"], "adapted_raw": [f"raw_seed{seed}" for seed in SEEDS],
        "zero_shot_rotation_stabilized": ["zero_shot_rotation_stabilized"],
        "adapted_rotation_stabilized": [f"rotation_stabilized_seed{seed}" for seed in SEEDS],
    }.items():
        selected = [result for result in results if result["name"] in names]
        summaries[label] = {}
        for scenario in ["all", *SCENARIOS]:
            scenario_rows = [[row for row in result["records"] if scenario == "all" or row["scenario"] == scenario] for result in selected]
            summaries[label][scenario] = {metric: float(np.mean(values)) if (values := [value for rows in scenario_rows if (value := mean_metric(rows, metric)) is not None]) else None for metric in metrics}
            summaries[label][scenario]["num_clips"] = len(scenario_rows[0])
            summaries[label][scenario]["num_seeds"] = len(selected)
            summaries[label][scenario]["num_clips_with_projected_objects"] = sum("top16_object_point_coverage" in row["metrics"] for row in scenario_rows[0])
    return summaries


def paired_bootstrap(results, first_names, second_names, metric, second_metric=None):
    def collect(names, selected_metric):
        selected = [result for result in results if result["name"] in names]
        return {row["token"]: np.mean([next(other["metrics"][selected_metric] for other in result["records"] if other["token"] == row["token"]) for result in selected])
                for row in selected[0]["records"] if row["metrics"].get(selected_metric) is not None}
    first = collect(first_names, metric)
    second = collect(second_names, second_metric or metric)
    rows = results[0]["records"]
    recording_groups = sorted({row["recording_group"] for row in rows})
    differences = {recording: np.array([second[row["token"]] - first[row["token"]] for row in rows if row["recording_group"] == recording and row["token"] in first and row["token"] in second]) for recording in recording_groups}
    generator = np.random.default_rng(4381)
    sampled_means = [np.concatenate([differences[recording] for recording in generator.choice(recording_groups, len(recording_groups), replace=True)]).mean() for _ in range(2000)]
    return {"second_minus_first": float(np.concatenate(list(differences.values())).mean()), "recording_bootstrap_95_interval": np.quantile(sampled_means, [.025, .975]).tolist(), "recordings": len(recording_groups)}


def style_plots():
    font_path = Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc")
    if font_path.exists():
        font_manager.fontManager.addfont(str(font_path))
        plt.rcParams["font.family"] = font_manager.FontProperties(fname=str(font_path)).get_name()
    plt.rcParams.update({"font.size": 10, "axes.unicode_minus": False, "pdf.fonttype": 42, "figure.facecolor": "#f5f7fa"})


def draw_image(axis, image):
    axis.imshow(np.clip(image, 0, 1) if image.dtype.kind == "f" else image)
    axis.set_xlim(-.5, 127.5); axis.set_ylim(127.5, -.5); axis.set_xticks([]); axis.set_yticks([])
    for spine in axis.spines.values():
        spine.set_visible(False)


def draw_particles(axis, saved, frame_index=3, budget=16):
    draw_image(axis, saved["raw_images"][frame_index])
    for center, box, particle_id in zip(saved["centers"][frame_index][:budget], saved["boxes"][frame_index][:budget], saved["particle_ids"][frame_index][:budget]):
        color = plt.cm.turbo((int(particle_id) % 64) / 63)
        left, top, right, bottom = np.clip(box, -300, 400)
        axis.add_patch(Rectangle((left, top), right-left, bottom-top, fill=False, edgecolor=color, linewidth=.9, alpha=.8))
        axis.scatter(*center, s=14, c=[color], edgecolor="black", linewidths=.3)


def save_figure(figure, basename, pdf):
    figure.savefig(FIGURE_ROOT / (basename + ".png"), dpi=170, bbox_inches="tight")
    pdf.savefig(figure, bbox_inches="tight")
    plt.close(figure)


def render_figures(summaries, manifest, pdf):
    selected = [next(record for record in manifest["records"] if record["split"] == "development" and record["scenario"] == scenario) for scenario in SCENARIOS]
    # The first manifest entry per scenario was chosen independently of all model outcomes.
    figure, axes = plt.subplots(3, 5, figsize=(15, 10))
    figure.subplots_adjust(top=.89, bottom=.09, wspace=.03, hspace=.19)
    figure.suptitle("LPWM particle은 실제 주행 장면에서 무엇을 표현하는가", fontsize=21, y=.98)
    figure.text(.08, .935, "결과와 무관하게 고른 3개 장면 · 고정 seed 47 · 점/박스는 transparency 상위 16개 · 색은 객체 ID가 아님", fontsize=10)
    for row_index, record in enumerate(selected):
        token = record["current_frame_token"]
        saved = {name: np.load(evaluation_directory(name) / (token + ".npz")) for name in ("zero_shot_raw", "raw_seed47", "rotation_stabilized_seed47")}
        draw_image(axes[row_index, 0], saved["raw_seed47"]["raw_images"][3])
        for obj in record["objects_by_frame"][3]:
            left, top, right, bottom = obj["box"]
            axes[row_index, 0].add_patch(Rectangle((left, top), right-left, bottom-top, fill=False, edgecolor="#00ffff", linewidth=1))
        for column, name in enumerate(("zero_shot_raw", "raw_seed47", "rotation_stabilized_seed47"), 1):
            draw_particles(axes[row_index, column], saved[name])
        mask_values = saved["raw_seed47"]["current_alpha_masks"][:, 0].astype(np.float32)
        mask_ids = saved["raw_seed47"]["decoder_particle_ids"][mask_values.argmax(0)]
        colors = plt.cm.turbo(mask_ids / 63)[..., :3]
        opacity = np.clip(mask_values.sum(0), 0, 1)[..., None]
        overlay = saved["raw_seed47"]["raw_images"][3] / 255 * (1 - .75 * opacity) + colors * .75 * opacity
        draw_image(axes[row_index, 4], overlay)
        axes[row_index, 0].set_ylabel(SCENARIO_NAMES[row_index], fontsize=12)
        axes[row_index, 0].text(0, 1.03, token, transform=axes[row_index, 0].transAxes, fontsize=8)
    for axis, title in zip(axes[0], ["실제 영상 + 평가용 GT 박스", "공식 checkpoint 그대로", "NAVSIM 영상으로 적응", "회전 보정 후 적응", "적응 모델의 실제 합성 마스크"]):
        axis.set_title(title, fontsize=10, pad=21)
    figure.text(.08, .045, "GT 박스는 학습 입력이 아님. 마스크 색칠은 semantic segmentation이 아니라 decoder의 입자별 합성 영역.\n복원 가능한 영역과 planning에 필요한 객체의 대응을 별도로 검증해야 함.", fontsize=10)
    save_figure(figure, "01_particles_and_masks", pdf)

    figure, axes = plt.subplots(3, 5, figsize=(15, 10))
    figure.subplots_adjust(top=.89, bottom=.10, wspace=.03, hspace=.15)
    figure.suptitle("현재 복원과 미래 예측을 구분해서 보기", fontsize=21, y=.98)
    figure.text(.08, .935, "과거 4프레임만 입력 → 2초 후 예측 · 미래 영상·미래 ego pose를 입력하지 않음 · seed 47", fontsize=11)
    for row_index, record in enumerate(selected):
        token = record["current_frame_token"]
        raw_saved = np.load(evaluation_directory("raw_seed47") / (token + ".npz"))
        stabilized_saved = np.load(evaluation_directory("rotation_stabilized_seed47") / (token + ".npz"))
        images = [raw_saved["raw_images"][3], raw_saved["reconstruction"][3], raw_saved["raw_images"][7], raw_saved["prediction"][3], stabilized_saved["prediction"][3]]
        for axis, displayed_image in zip(axes[row_index], images):
            draw_image(axis, displayed_image)
        axes[row_index, 0].set_ylabel(SCENARIO_NAMES[row_index], fontsize=12)
    for axis, title in zip(axes[0], ["실제 현재 영상", "현재 영상 복원", "실제 2초 후 정답", "원영상 모델의 2초 후 예측", "회전 보정 모델의 2초 후 예측"]):
        axis.set_title(title, fontsize=10)
    figure.text(.08, .045, "예측 영상은 실제 모델 출력. 회전 보정의 검은 영역은 시야 손실과 과거 회전에서 외삽한 좌표 변환의 영향을 포함.\n선명한 현재 복원만으로 과거 입력 기반 미래 예측이나 객체 permanence의 성공을 주장하지 않음.", fontsize=10)
    save_figure(figure, "02_reconstruction_and_causal_forecast", pdf)

    figure, axes = plt.subplots(1, 3, figsize=(15, 4.8))
    figure.subplots_adjust(top=.77, bottom=.22, wspace=.35)
    figure.suptitle("작은 NAVSIM 적응 실험의 정량 결과", fontsize=20, y=.98)
    figure.text(.055, .86, "개발 영상 30 clip / 객체 지표 24 clip · 17 recording · 적응 조건 각각 3 seed · 격자는 학습 없는 위치 기준선", fontsize=10)
    conditions = ["zero_shot_raw", "adapted_raw", "adapted_rotation_stabilized"]
    labels = ["공식 모델", "원영상 적응", "회전 보정 적응"]
    colors = ["#929ca8", "#1677a8", "#d28a23"]
    for axis, metric, title in zip(axes[:2], ["top16_object_point_coverage", "top16_object_box_recall_iou_030"], ["객체 박스 안에 점이 있는 비율 ↑", "일대일 박스 대응률 IoU≥0.3 ↑"]):
        values = [summaries[condition]["all"][metric] * 100 for condition in conditions]
        axis.bar(labels, values, color=colors)
        for position, value in enumerate(values):
            axis.text(position, value + 1, f"{value:.1f}%", ha="center")
        grid_metric = "grid16_" + metric.removeprefix("top16_")
        grid_value = summaries["adapted_raw"]["all"][grid_metric] * 100
        axis.axhline(grid_value, color="#9b3547", linestyle="--", label=f"격자 16개: {grid_value:.1f}%")
        axis.set_ylim(0, max(values + [grid_value]) * 1.25 + 3)
        axis.set_title(title, fontsize=11);axis.legend(fontsize=8)
    values = [summaries[condition]["all"]["forecast_mse_full"] for condition in conditions]
    axes[2].bar(labels, values, color=colors)
    persistence = summaries["adapted_raw"]["all"]["persistence_mse_full"]
    axes[2].axhline(persistence, color="#9b3547", linestyle="--", label=f"마지막 영상 유지: {persistence:.4f}")
    for position, value in enumerate(values):
        axes[2].text(position, value + .002, f"{value:.4f}", ha="center")
    axes[2].set_title("과거만 사용한 미래 영상 MSE ↓", fontsize=11); axes[2].legend(fontsize=8)
    axes[2].set_ylim(0, max(values) * 1.2)
    figure.text(.055, .06, "점 포함률은 객체 인식 정확도가 아니며, 낮은 해상도·투영 박스·복잡한 가림의 한계가 있음. 독립 test 및 planning/PDMS 평가 결과가 아님.", fontsize=9)
    save_figure(figure, "03_quantitative_summary", pdf)

    figure, axes = plt.subplots(3, 3, figsize=(10, 10))
    figure.subplots_adjust(top=.88, bottom=.10, wspace=.03, hspace=.12)
    figure.suptitle("가림에 대한 particle의 반응", fontsize=20, y=.98)
    figure.text(.08, .935, "현재 가장 큰 GT 박스 영역을 회색으로 가리는 진단 · 학습에는 사용하지 않음 · seed 47", fontsize=10)
    for row_index, record in enumerate(selected):
        saved = np.load(evaluation_directory("raw_seed47") / (record["current_frame_token"] + ".npz"))
        draw_particles(axes[row_index, 0], saved)
        draw_image(axes[row_index, 1], saved["masked_image"])
        for center, box in zip(saved["occluded_centers"], saved["occluded_boxes"]):
            left, top, right, bottom = box
            axes[row_index, 1].add_patch(Rectangle((left, top), right-left, bottom-top, fill=False, edgecolor="#18dbd2", linewidth=.8))
            axes[row_index, 1].scatter(*center, s=9, color="#18dbd2")
        draw_particles(axes[row_index, 2], saved, 4)
        axes[row_index, 0].set_ylabel(SCENARIO_NAMES[row_index], fontsize=11)
    for axis, title in zip(axes[0], ["원래 현재 영상의 particle", "인위적으로 가린 현재 영상의 particle", "0.5초 뒤 실제 관측의 particle"]):
        axis.set_title(title, fontsize=10)
    figure.text(.08, .035, "회색 영역은 제어된 인공 가림. 박스 안에 점이 남는 현상만으로 객체 기억을 입증할 수 없음.\n현재 encoder는 프레임별 표현을 만들며, 다음 관측에서의 재검출과 가림 동안의 belief 유지는 별도 문제.", fontsize=10)
    save_figure(figure, "04_controlled_occlusion", pdf)

    # Short observed-particle animations show identity handoffs instead of implying persistent object IDs.
    for scenario, record in zip(SCENARIOS, selected):
        saved = {name: np.load(evaluation_directory(name) / (record["current_frame_token"] + ".npz")) for name in ("zero_shot_raw", "raw_seed47", "rotation_stabilized_seed47")}
        animation_frames = []
        for frame_index in range(8):
            figure, axes = plt.subplots(1, 3, figsize=(9, 3.4))
            for axis, name, title in zip(axes, saved, ["공식 모델", "원영상 적응", "회전 보정 적응"]):
                draw_particles(axis, saved[name], frame_index)
                axis.set_title(title)
            figure.suptitle(f"관측 영상의 particle · {frame_index * .5:.1f}초 · 색은 patch ID", fontsize=11)
            figure.tight_layout()
            figure.canvas.draw()
            animation_frames.append(Image.fromarray(np.asarray(figure.canvas.buffer_rgba()).copy()).convert("RGB"))
            plt.close(figure)
        animation_frames[0].save(FIGURE_ROOT / f"particles_{scenario}.gif", save_all=True, append_images=animation_frames[1:], duration=500, loop=0)


def main():
    manifest = json.loads((ARTIFACT_ROOT / "clip_manifest.json").read_text())
    manifest_lookup = {record["current_frame_token"]: record for record in manifest["records"]}
    names = ["zero_shot_raw", "zero_shot_rotation_stabilized", *[f"{variant}_seed{seed}" for variant in ("raw", "rotation_stabilized") for seed in SEEDS]]
    results = [json.loads((evaluation_directory(name) / "metrics.json").read_text()) for name in names]
    for result in results:
        assert len(result["records"]) == 30
        add_common_area_metrics(result, manifest_lookup)
    summaries = group_summaries(results)
    comparisons = {}
    for metric in ("top16_object_point_coverage", "top16_object_box_recall_iou_030", "forecast_mse_full", "common_area_forecast_mse"):
        comparisons[metric] = {
            "raw_adaptation_vs_zero_shot": paired_bootstrap(results, ["zero_shot_raw"], [f"raw_seed{seed}" for seed in SEEDS], metric),
            "rotation_vs_raw_adaptation": paired_bootstrap(results, [f"raw_seed{seed}" for seed in SEEDS], [f"rotation_stabilized_seed{seed}" for seed in SEEDS], metric)}
    training = [json.loads(path.read_text()) for path in sorted((ARTIFACT_ROOT / "runs").glob("*/training_summary.json"))]
    comparisons["raw_forecast_vs_persistence"] = paired_bootstrap(results, [f"raw_seed{seed}" for seed in SEEDS], [f"raw_seed{seed}" for seed in SEEDS], "persistence_mse_full", "forecast_mse_full")
    assert len(training) == 6 and sum(row["updates"] for row in training) == 1800
    summary = {"groups": summaries, "paired_comparisons": comparisons, "training": training,
               "per_run": {result["name"]: {key: mean_metric(result["records"], key) for key in summaries["adapted_raw"]["all"] if not key.startswith("num_")} for result in results},
               "uncertainty_scope": "2000 recording-cluster bootstrap samples of paired, seed-averaged clip metrics; training-seed uncertainty not included; no multiplicity correction"}
    SHARED_ROOT.mkdir(parents=True, exist_ok=True); FIGURE_ROOT.mkdir(parents=True, exist_ok=True)
    (SHARED_ROOT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (SHARED_ROOT / "per_clip_metrics.json").write_text(json.dumps(results, indent=2) + "\n")
    style_plots()
    with PdfPages(SHARED_ROOT / "lpwm_navsim_visualizations.pdf") as pdf:
        render_figures(summaries, manifest, pdf)
    (FIGURE_ROOT / "index.html").write_text('<!doctype html><meta charset="utf-8"><title>LPWM NAVSIM 적응 결과</title><style>body{max-width:1600px;margin:30px auto;font-family:sans-serif}img{width:100%}</style><h1>LPWM NAVSIM 적응 결과</h1>' + ''.join(f'<h2>{path.stem}</h2><img src="{path.name}">' for path in sorted(FIGURE_ROOT.glob("*.png"))) + ''.join(f'<h2>{path.stem}</h2><img src="{path.name}">' for path in sorted(FIGURE_ROOT.glob("*.gif"))))
    print(json.dumps({group: summaries[group]["all"] for group in summaries}, indent=2))


if __name__ == "__main__":
    main()
