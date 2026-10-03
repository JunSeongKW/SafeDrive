"""CPU-only checkpoint inspection; real images are references, NOT predictions.

Reuses verified frozen-encoder caches, strictly restores selector/predictor
submodules, and never invokes an optimizer or reads held-out/navtest inputs.
"""

import argparse
import copy
import hashlib
import html
import json
import math
import pickle
import subprocess
import time
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from matplotlib.patches import Rectangle
from PIL import Image

from planning_aware_future_prediction.models.drive_jepa_adaptive_future import (
    ContextualResidualFuturePredictor,
    EgoQueryPatchSelector,
)
from planning_aware_future_prediction.models.drive_jepa_selective_patch_future import (
    LightweightPatchFuturePredictor,
    PlanningConditionedPatchSelector,
)

WORKSPACE = Path(__file__).resolve().parents[1]
COLORS = ["#00d4ff", "#ffb000", "#ee55ff", "#66ff55"]
HORIZONS = [1, 2, 3, 4]


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def patch_pixel_bounds(patch_id, grid_width=32, grid_height=16):
    if not 0 <= patch_id < grid_width * grid_height:
        raise ValueError("Patch ID outside image grid")
    row, column = divmod(patch_id, grid_width)
    return column * 16, row * 16, 16, 16


def patch_coordinates():
    rows, columns = torch.meshgrid(
        torch.linspace(-1, 1, 16), torch.linspace(-1, 1, 32), indexing="ij"
    )
    return torch.stack((columns, rows), dim=-1).flatten(0, 1)


def selector_scores(selector, current_latents, ego_status, coordinates):
    """Exact production scoring equation, independently checked against hard IDs."""
    if isinstance(selector, EgoQueryPatchSelector):
        keys = selector.key_normalization(
            selector.patch_keys(current_latents)
            + selector.coordinate_keys(coordinates)[None]
        )
        query = selector.query_normalization(
            selector.intent_query(ego_status)
            + selector.context_query(current_latents.mean(dim=1))
        )
        return torch.einsum("bnd,bd->bn", keys, query) / math.sqrt(keys.shape[-1])
    return selector.score_head(
        selector.current_projection(current_latents)
        + selector.context_projection(current_latents.mean(dim=1))[:, None]
        + selector.ego_projection(ego_status)[:, None]
        + selector.coordinate_projection(coordinates)[None]
    ).squeeze(-1)


def predict_selected(predictor, current_latents, ego_status, coordinates, indices):
    selected = current_latents[:, indices]
    selected_coordinates = coordinates[indices][None]
    if isinstance(predictor, ContextualResidualFuturePredictor):
        return predictor(
            selected, selected_coordinates, current_latents, ego_status, coordinates
        )[0]
    return predictor(
        selected, selected_coordinates, current_latents.mean(dim=1), ego_status
    )[0]


def strict_restore_submodule(module, parameters, prefix):
    scoped = {
        name[len(prefix) :]: value
        for name, value in parameters.items()
        if name.startswith(prefix)
    }
    module.load_state_dict(scoped, strict=True)


def relative_parameter_change(before, after):
    initial = torch.cat([value.flatten() for value in before.parameters()])
    trained = torch.cat([value.flatten() for value in after.parameters()])
    return float((trained - initial).norm() / initial.norm().clamp_min(1e-12))


def choose_gallery_records(records, salt, per_command):
    """Current-command stratification; no score/target-validity based selection."""
    chosen, seen_recordings = [], set()
    for command in sorted({row["command_raw_index"] for row in records}):
        candidates = sorted(
            (row for row in records if row["command_raw_index"] == command),
            key=lambda row: hashlib.sha256(
                f"{salt}:{row['current_frame_token']}".encode()
            ).hexdigest(),
        )
        count = 0
        for row in candidates:
            if row["recording_group"] in seen_recordings:
                continue
            chosen.append(row)
            seen_recordings.add(row["recording_group"])
            count += 1
            if count == per_command:
                break
        if count != per_command:
            raise ValueError("Insufficient distinct recordings for gallery")
    return chosen


def finite_nested(values):
    array = np.asarray(values)
    if array.ndim == 0:
        return float(array) if np.isfinite(array) else None
    return [finite_nested(row) for row in array]


def target_metrics(prediction, target, valid):
    squared = (prediction - target).square().mean(dim=-1)
    cosine = F.cosine_similarity(prediction, target, dim=-1)
    return (
        squared.masked_fill(~valid, float("nan")).numpy(),
        cosine.masked_fill(~valid, float("nan")).numpy(),
    )


def read_reference_images(record):
    log_path = WORKSPACE / "dataset/navsim_logs/trainval" / record["segment_filename"]
    with log_path.open("rb") as stream:
        frames = pickle.load(stream)[record["start_index"] : record["start_index"] + 12]
    if len(frames) != 12 or frames[3]["token"] != record["current_frame_token"]:
        raise RuntimeError("Image/cache temporal alignment mismatch")
    images, provenance = {}, []
    for position in [2, 3, 5, 7, 9, 11]:
        relative = next(
            camera["data_path"]
            for name, camera in frames[position]["cams"].items()
            if name.lower() == "cam_f0"
        )
        path = WORKSPACE / "dataset/sensor_blobs/trainval" / relative
        if not path.is_file():
            images[position] = None
            provenance.append(
                {"frame_index": position, "path": str(path), "missing": True}
            )
            continue
        with Image.open(path) as image:
            rgb = np.array(image.convert("RGB"))
        # Exact official front-only builder crop and OpenCV default interpolation.
        images[position] = cv2.resize(rgb[28:-28], (512, 256))
        provenance.append(
            {"frame_index": position, "path": str(path), "sha256": sha256_file(path)}
        )
    return images, provenance


def show_image(axis, image, indices, title):
    axis.set_title(title, fontsize=10)
    axis.set_xlim(0, 512)
    axis.set_ylim(256, 0)
    axis.axis("off")
    if image is None:
        axis.set_facecolor("#eeeeee")
        axis.text(256, 128, "REFERENCE IMAGE MISSING", ha="center", va="center")
        return
    axis.imshow(image, extent=(0, 512, 256, 0))
    for slot, patch_id in enumerate(indices):
        left, top, width, height = patch_pixel_bounds(int(patch_id))
        axis.add_patch(
            Rectangle(
                (left, top), width, height, fill=False, edgecolor=COLORS[slot], lw=2
            )
        )
        axis.text(
            left + 1,
            top - 3,
            f"{slot + 1}",
            color=COLORS[slot],
            fontsize=10,
            weight="bold",
            bbox={"facecolor": "black", "alpha": 0.6, "pad": 1},
        )


def render_scene(output, model_name, record, images, details):
    token = record["current_frame_token"]
    stem = f"{model_name}_{token}"
    figure, axes = plt.subplots(2, 2, figsize=(13, 7.5), constrained_layout=True)
    for column, stage in enumerate(["before", "after"]):
        indices = details[f"{stage}_ids"]
        label = (
            "Before joint training (after auxiliary warmup)"
            if stage == "before"
            else "After 800 joint updates"
        )
        show_image(axes[0, column], images[3], indices, label + f"\nIDs {indices}")
        probabilities = np.array(details[f"{stage}_probabilities"]).reshape(16, 32)
        heat = axes[1, column].imshow(
            probabilities,
            cmap="magma",
            vmin=0,
            vmax=max(
                max(details["before_probabilities"]),
                max(details["after_probabilities"]),
            ),
        )
        axes[1, column].set_title(
            "First-slot softmax probability (not causal importance)"
        )
        axes[1, column].set_xlabel("Patch column")
        axes[1, column].set_ylabel("Patch row")
        figure.colorbar(heat, ax=axes[1, column], fraction=0.025)
    figure.suptitle(
        f"{model_name} | {token} | command ID {record['command_raw_index']} | speed {record['ego_speed_meters_per_second']:.2f} m/s",
        fontsize=13,
    )
    selector_path = output / f"{stem}_selector.png"
    figure.savefig(selector_path, dpi=135)
    plt.close(figure)

    figure, axes = plt.subplots(2, 4, figsize=(17, 7.2), constrained_layout=True)
    for horizon, position in enumerate([5, 7, 9, 11]):
        valid_horizon = any(row[horizon] is not None for row in details["mse_after"])
        show_image(
            axes[0, horizon],
            images[position],
            details["after_ids"],
            f"REAL future reference +{horizon + 1}s\nPair target valid: {valid_horizon}",
        )
    for slot in range(4):
        axis = axes[1, slot]
        for key, label, color, style in [
            ("mse_before", "Before joint (same final IDs)", "#c1872c", "--"),
            ("mse_after", "Trained predictor", "#1769aa", "-"),
            ("mse_copy", "Copy current feature", "#60666c", ":"),
        ]:
            axis.plot(
                HORIZONS,
                details[key][slot],
                marker="o",
                color=color,
                linestyle=style,
                label=label,
            )
        axis.set_title(
            f"Slot {slot + 1} / patch {details['after_ids'][slot]}",
            color=COLORS[slot] if slot == 2 else "#222222",
        )
        axis.set_xlabel("Future tubelet end (s)")
        axis.set_ylabel("Latent MSE (lower is better)")
        axis.set_xticks(HORIZONS)
        axis.grid(alpha=0.2)
        if slot == 0:
            axis.legend(fontsize=7)
    figure.suptitle(
        f"{model_name} | {token}\nTop: ground-truth images, NOT generated predictions. Bottom: 1024-D prediction errors; gaps = unavailable targets.",
        fontsize=12,
    )
    predictor_path = output / f"{stem}_predictor.png"
    figure.savefig(predictor_path, dpi=130)
    plt.close(figure)

    figure, axes = plt.subplots(4, 5, figsize=(11, 8), constrained_layout=True)
    for slot, patch_id in enumerate(details["after_ids"]):
        left, top, width, height = patch_pixel_bounds(patch_id)
        for column, position in enumerate([3, 5, 7, 9, 11]):
            image = images[position]
            axis = axes[slot, column]
            if image is not None:
                axis.imshow(
                    image[top : top + height, left : left + width],
                    interpolation="nearest",
                )
            else:
                axis.text(0.5, 0.5, "Missing", ha="center", transform=axis.transAxes)
            axis.set_xticks([])
            axis.set_yticks([])
            if column == 0:
                axis.set_ylabel(f"Slot {slot + 1}\nID {patch_id}")
            if slot == 0:
                axis.set_title("Current (t=0)" if column == 0 else f"Real +{column}s")
    figure.suptitle(
        "Actual 16x16 patch crops, enlarged without smoothing\nFixed image coordinates, NOT an object track; predictor output is latent, not RGB.",
        fontsize=12,
    )
    crop_path = output / f"{stem}_crops.png"
    figure.savefig(crop_path, dpi=120)
    plt.close(figure)
    return [selector_path.name, predictor_path.name, crop_path.name]


def render_latent_values(output, model_name, token, latent_values):
    """Feature channels are NOT spatial pixels; no RGB decoder is implied."""
    arrays = {key: value.numpy() for key, value in latent_values.items()}
    valid = arrays["valid_mask"]
    target = arrays["target"].copy()
    target[~valid] = np.nan
    streams = [target, arrays["before"], arrays["after"], arrays["copy"]]
    finite_values = np.concatenate([array[np.isfinite(array)] for array in streams])
    color_limit = max(float(np.quantile(np.abs(finite_values), 0.99)), 1e-6)
    figure, axes = plt.subplots(2, 2, figsize=(17, 10), constrained_layout=True)
    labels = [
        f"S{slot + 1} {stage}"
        for slot in range(4)
        for stage in ["GT", "before", "trained", "copy"]
    ]
    color_map = plt.get_cmap("coolwarm").copy()
    color_map.set_bad("#cccccc")
    for horizon, axis in enumerate(axes.flat):
        rows = np.stack(
            [array[slot, horizon] for slot in range(4) for array in streams]
        )
        heat = axis.imshow(
            rows,
            aspect="auto",
            interpolation="nearest",
            cmap=color_map,
            vmin=-color_limit,
            vmax=color_limit,
        )
        axis.set_yticks(range(16), labels, fontsize=8)
        axis.set_xlabel("Latent feature channel (0..1023); NOT image x-coordinate")
        axis.set_title(
            f"+{horizon + 1}s tubelet | valid GT slots {int(valid[:, horizon].sum())}/4"
        )
        for boundary in [3.5, 7.5, 11.5]:
            axis.axhline(boundary, color="black", lw=0.6)
    figure.colorbar(
        heat,
        ax=axes.ravel().tolist(),
        fraction=0.025,
        label="Latent value (shared color scale)",
    )
    figure.suptitle(
        f"{model_name} | {token}: actual 1024-D outputs vs targets\nGT / warmup / trained / current-copy, identical final patch IDs; gray = unavailable GT. Colors clip at shared 99th percentile, metrics do not.",
        fontsize=12,
    )
    filename = f"{model_name}_{token}_latent_values.png"
    figure.savefig(output / filename, dpi=130)
    plt.close(figure)
    np.savez_compressed(output / f"{model_name}_{token}_latent_values.npz", **arrays)
    return filename


def build_modules(model_spec, seed):
    if model_spec["architecture"] == "ego_query_residual":
        selector = EgoQueryPatchSelector(1024, 128, 4)
        predictor = ContextualResidualFuturePredictor(1024, 128, 4)
    else:
        selector = PlanningConditionedPatchSelector(1024, 128, 4)
        predictor = LightweightPatchFuturePredictor(1024, 128, 4)
    warm_path = WORKSPACE / model_spec["warmup_checkpoint"]
    final_path = WORKSPACE / model_spec["final_checkpoint"]
    warm = torch.load(warm_path, map_location="cpu", weights_only=False)
    final = torch.load(final_path, map_location="cpu", weights_only=False)
    if final["completed_update"] != 800 or final["seed"] != seed:
        raise RuntimeError("Expected registered final800/seed checkpoint")
    strict_restore_submodule(selector, warm["trainable_state"], "patch_selector.")
    strict_restore_submodule(predictor, warm["trainable_state"], "future_predictor.")
    before_selector, before_predictor = (
        copy.deepcopy(selector),
        copy.deepcopy(predictor),
    )
    strict_restore_submodule(selector, final["extension_parameters"], "patch_selector.")
    strict_restore_submodule(
        predictor, final["extension_parameters"], "future_predictor."
    )
    for module in [selector, predictor, before_selector, before_predictor]:
        module.eval().requires_grad_(False)
    results_path = final_path.parent / "results.json"
    results = json.loads(results_path.read_text())
    saved_windows = {
        row["token"]: row
        for row in results["evaluations"]["800"]["development"]["windows"]
    }
    metadata = {
        "warmup_path": str(warm_path),
        "warmup_sha256": sha256_file(warm_path),
        "final_path": str(final_path),
        "final_sha256": sha256_file(final_path),
        "strict_submodule_loading": True,
        "selector_relative_parameter_change": relative_parameter_change(
            before_selector, selector
        ),
        "predictor_relative_parameter_change": relative_parameter_change(
            before_predictor, predictor
        ),
        "stage_comparison": "after100_auxiliary_warmup_vs_after800_joint_updates",
    }
    return (
        selector,
        predictor,
        before_selector,
        before_predictor,
        saved_windows,
        metadata,
    )


def render_overview(
    output,
    specification,
    records,
    gallery,
    all_details,
    model_summaries,
    images_by_token,
):
    model_names = list(specification["models"])
    figure, axes = plt.subplots(
        len(gallery),
        len(model_names),
        figsize=(13, 3.2 * len(gallery)),
        constrained_layout=True,
        squeeze=False,
    )
    for row_index, record in enumerate(gallery):
        token = record["current_frame_token"]
        for column, name in enumerate(model_names):
            show_image(
                axes[row_index, column],
                images_by_token[token][3],
                all_details[name][token]["after_ids"],
                f"{name} | command ID {record['command_raw_index']} | {record['ego_speed_meters_per_second']:.1f} m/s\n{token}",
            )
    figure.suptitle(
        "Learned patch selection on real CURRENT images | seed29, final800\nSix hash-selected development windows; colors label four selected slots, not object classes.",
        fontsize=13,
    )
    figure.savefig(output / "selection_contact_sheet.png", dpi=120)
    plt.close(figure)
    figure, axes = plt.subplots(
        2, len(model_names), figsize=(13, 8), constrained_layout=True, squeeze=False
    )
    for column, name in enumerate(model_names):
        summary = model_summaries[name]
        for key, label, color in [
            ("mse_before", "Before joint, final IDs", "#c1872c"),
            ("mse_after", "Trained predictor", "#1769aa"),
            ("mse_copy", "Copy current feature", "#60666c"),
        ]:
            axes[0, column].plot(
                HORIZONS,
                summary[key + "_by_horizon"],
                marker="o",
                label=label,
                color=color,
            )
        axes[0, column].set_title(f"{name}: all {len(records)} development windows")
        axes[0, column].set_xlabel("Future tubelet end (s)")
        axes[0, column].set_ylabel("Latent MSE (valid patch-times)")
        axes[0, column].set_xticks(HORIZONS)
        axes[0, column].legend(fontsize=8)
        axes[0, column].grid(alpha=0.2)
        counts = np.array(summary["selection_counts"]).reshape(16, 32)
        heat = axes[1, column].imshow(counts, cmap="viridis", vmin=0)
        axes[1, column].set_title(
            "Trained selection frequency (not importance ground truth)"
        )
        axes[1, column].set_xlabel("Patch column")
        axes[1, column].set_ylabel("Patch row")
        figure.colorbar(heat, ax=axes[1, column], fraction=0.025)
    figure.suptitle(
        "Selector/predictor diagnostics | same final-selected positions for all prediction baselines\nMissing future supervision stays missing; current candidates are never filtered by it.",
        fontsize=12,
    )
    figure.savefig(output / "module_diagnostics_overview.png", dpi=135)
    plt.close(figure)


def write_gallery_html(output, specification, gallery, model_summaries, figure_links):
    blocks = []
    for name, summary in model_summaries.items():
        blocks.append(
            f"<h2>{html.escape(name)}</h2><p>학습 전후 selector 가중치 상대 변화 {summary['selector_relative_parameter_change']:.3%}, predictor {summary['predictor_relative_parameter_change']:.3%}. 현재 장면당 선택 patch 교체율 {1 - summary['selection_retention_mean']:.1%}. 가중치·선택 변화가 좋은 선택을 뜻하지는 않습니다.</p>"
        )
        for record in gallery:
            token = record["current_frame_token"]
            blocks.append(
                f"<details><summary>명령 ID {record['command_raw_index']} · {record['ego_speed_meters_per_second']:.1f} m/s · {token}</summary>"
            )
            for link in figure_links[name][token]:
                blocks.append(
                    f'<a href="{link}" target="_blank"><img loading="lazy" src="{link}"></a>'
                )
            blocks.append("</details>")
    content = """<!doctype html><html lang="ko"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>학습된 patch selector · predictor 검증</title><style>body{font:16px/1.7 system-ui,sans-serif;max-width:1200px;margin:32px auto;padding:0 20px;color:#172431;background:#f7f9fc}img{width:100%;background:white;border:1px solid #d4dce4}h1,h2{line-height:1.3}details{margin:18px 0;padding:14px;background:white;border:1px solid #d4dce4}summary{cursor:pointer;font-weight:600}.note{border-left:4px solid #167ca6;padding:16px;background:#e9f4fa}a{color:#07648e}li{margin:5px 0}</style><h1>실제 체크포인트: selector와 predictor가 무엇을 배웠나?</h1><div class="note"><b>읽는 법</b><ul><li>Selector: 실제 현재 영상의 색상 사각형 4개가 선택 결과입니다. 같은 색은 같은 slot이지 객체 identity가 아닙니다.</li><li>Before는 완전 무학습이 아니라 미래 보조 warmup 100회 뒤, joint 학습 시작 전입니다. After는 고정 800 update입니다.</li><li>Predictor는 미래 사진이 아닌 1024차원 latent를 출력합니다. 모든 사진·crop은 실제 정답 영상이며 생성된 예측 사진이 아닙니다.</li><li>예측 곡선은 학습 전 predictor / 학습 후 predictor / 현재 feature 복사를 <b>동일한 최종 선택 위치·유효 mask</b>로 비교합니다. 낮은 MSE가 좋습니다.</li><li>현재 입력은 t=-0.5,0초 영상 쌍입니다. 미래 target도 (0.5,1), (1.5,2), (2.5,3), (3.5,4)초의 영상 쌍입니다. 그림에는 각 쌍의 마지막 영상만 표시합니다.</li><li>같은 화면 격자를 예측하며 객체 추적·ego-motion 보정은 없습니다. 국소 crop 밖의 맥락도 encoder feature에 포함될 수 있습니다.</li><li>전체 192 dev 구간·24 recording의 요약과, 명령 ID별 해시 순서로 뽑은 6개 다른 recording을 표시합니다. 성능·미래 유효율로 표본을 고르지 않았습니다. seed29는 첫 사전등록 seed이며 최상 seed를 고른 것이 아닙니다.</li><li>CPU 캐시 추론, 새 학습·GPU 사용·planner 재평가 없음. 정상 로딩·유한값·원래 GPU 결과와 대조를 검사했습니다. 시각화는 planning 개선이나 인과적 중요도의 증명이 아닙니다.</li></ul></div><h2>전체 진단</h2><a href="module_diagnostics_overview.png"><img src="module_diagnostics_overview.png"></a><h2>실제 선택 위치</h2><a href="selection_contact_sheet.png"><img src="selection_contact_sheet.png"></a><h2>장면별 학습 전후 및 미래 예측 오차</h2><p>아래 각 장면을 펼치고, 그림을 누르면 원본 크기로 확인할 수 있습니다.</p>"""
    content += (
        "".join(blocks)
        + '<p><a href="metrics.json">전체 수치·checkpoint hash·원본 대조 결과</a> / <a href="specification.json">고정한 표본·실행 설정</a></p></html>'
    )
    (output / "index.html").write_text(content)


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    started = time.monotonic()
    specification = json.loads(args.config.read_text())
    output = args.output_directory.resolve()
    output.mkdir(parents=True, exist_ok=False)
    records = [
        row
        for row in json.loads((WORKSPACE / specification["cache_index"]).read_text())[
            "records"
        ]
        if row["split"] == "development"
    ]
    gallery = choose_gallery_records(
        records,
        specification["gallery_hash_salt"],
        specification["gallery_windows_per_command"],
    )
    write_json(
        output / "specification.json",
        {
            "config": specification,
            "gallery_records": gallery,
            "source_commit": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], text=True, cwd=WORKSPACE
            ).strip(),
            "runner_sha256": sha256_file(__file__),
        },
    )
    coordinates = patch_coordinates()
    bundles = {
        name: build_modules(model_spec, specification["seed"])
        for name, model_spec in specification["models"].items()
    }
    details_by_model = {name: {} for name in bundles}
    gallery_tokens = {row["current_frame_token"] for row in gallery}
    gallery_latents = {name: {} for name in bundles}
    max_mse_error = {name: 0.0 for name in bundles}
    selected_ids_equal = {name: True for name in bundles}
    for record_index, record in enumerate(records):
        if time.monotonic() - started > specification["maximum_wall_seconds"]:
            raise RuntimeError(
                "Visualization time cap reached; partial output preserved"
            )
        if sha256_file(record["cache_file"]) != record["cache_sha256"]:
            raise RuntimeError("Cache checksum changed")
        cached = torch.load(record["cache_file"], map_location="cpu", weights_only=True)
        current = cached["current_patch_latents"][None]
        ego = cached["current_ego_status"][None]
        for name, (
            selector,
            predictor,
            before_selector,
            before_predictor,
            saved_windows,
            _,
        ) in bundles.items():
            before = before_selector(current, ego, coordinates)
            after = selector(current, ego, coordinates)
            indices = after.selected_patch_indices[0]
            scores_before = selector_scores(before_selector, current, ego, coordinates)
            scores_after = selector_scores(selector, current, ego, coordinates)
            if (
                scores_after.argsort(descending=True)[0, :4].tolist()
                != indices.tolist()
            ):
                raise RuntimeError("Heatmap scoring differs from production selection")
            target = cached["future_target_latents"][:, indices].permute(1, 0, 2)
            valid = cached["future_target_valid_mask"][:, indices].T
            predicted = predict_selected(predictor, current, ego, coordinates, indices)
            predicted_before = predict_selected(
                before_predictor, current, ego, coordinates, indices
            )
            persistence = current[0, indices, None].expand(-1, 4, -1)
            if not all(
                torch.isfinite(tensor).all()
                for tensor in [predicted, predicted_before, scores_before, scores_after]
            ):
                raise RuntimeError("Nonfinite model output")
            details = {
                "before_ids": before.selected_patch_indices[0].tolist(),
                "after_ids": indices.tolist(),
                "before_probabilities": (scores_before[0] / before_selector.temperature)
                .softmax(-1)
                .tolist(),
                "after_probabilities": (scores_after[0] / selector.temperature)
                .softmax(-1)
                .tolist(),
                "command_raw_index": record["command_raw_index"],
                "recording": record["recording_group"],
                "valid_patch_times": int(valid.sum()),
                "target_valid_mask": valid.tolist(),
            }
            for suffix, prediction in [
                ("before", predicted_before),
                ("after", predicted),
                ("copy", persistence),
            ]:
                mse, cosine = target_metrics(prediction, target, valid)
                details["mse_" + suffix] = finite_nested(mse)
                details["cosine_" + suffix] = finite_nested(cosine)
            saved = saved_windows[record["current_frame_token"]]
            selected_ids_equal[name] &= saved["selected_patch_ids"] == indices.tolist()
            if valid.any():
                error = abs(
                    float((predicted - target).square().mean(-1)[valid].mean())
                    - saved["future_mse"]
                )
                max_mse_error[name] = max(max_mse_error[name], error)
            details_by_model[name][record["current_frame_token"]] = details
            if record["current_frame_token"] in gallery_tokens:
                gallery_latents[name][record["current_frame_token"]] = {
                    "target": target.clone(),
                    "before": predicted_before.clone(),
                    "after": predicted.clone(),
                    "copy": persistence.clone(),
                    "valid_mask": valid.clone(),
                    "selected_patch_ids": indices.clone(),
                }
        if record_index % 32 == 0:
            print(f"VISUAL_CHECK {record_index + 1}/{len(records)}", flush=True)
    summaries = {}
    for name, bundle in bundles.items():
        if (
            not selected_ids_equal[name]
            or max_mse_error[name] > specification["cpu_gpu_mse_absolute_tolerance"]
        ):
            write_json(
                output / "validation_failure.json",
                {
                    "selection_ids_equal": selected_ids_equal,
                    "max_cpu_gpu_mse_error": max_mse_error,
                },
            )
            raise RuntimeError(
                "CPU restored modules diverge from stored GPU evaluation"
            )
        details = list(details_by_model[name].values())
        summary = dict(bundle[-1])
        summary.update(
            {
                "cpu_gpu_all_selected_ids_equal": selected_ids_equal[name],
                "maximum_cpu_gpu_window_mse_difference": max_mse_error[name],
                "selection_retention_mean": float(
                    np.mean(
                        [
                            len(set(row["before_ids"]) & set(row["after_ids"])) / 4
                            for row in details
                        ]
                    )
                ),
                "selection_counts": np.bincount(
                    [patch_id for row in details for patch_id in row["after_ids"]],
                    minlength=512,
                ).tolist(),
            }
        )
        for key in [
            "mse_before",
            "mse_after",
            "mse_copy",
            "cosine_before",
            "cosine_after",
            "cosine_copy",
        ]:
            values = np.asarray([row[key] for row in details], dtype=float)
            summary[key + "_by_horizon"] = finite_nested(
                np.nanmean(values, axis=(0, 1))
            )
            summary[key + "_valid_patch_time_mean"] = float(np.nanmean(values))
        summary["valid_patch_times_by_horizon"] = (
            np.isfinite(np.asarray([row["mse_after"] for row in details], dtype=float))
            .sum(axis=(0, 1))
            .tolist()
        )
        summaries[name] = summary
    images_by_token = {}
    figure_links = {name: {} for name in bundles}
    image_sources = {}
    for record in gallery:
        token = record["current_frame_token"]
        images, provenance = read_reference_images(record)
        images_by_token[token] = images
        image_sources[token] = provenance
        for name in bundles:
            figure_links[name][token] = render_scene(
                output, name, record, images, details_by_model[name][token]
            )
            figure_links[name][token].append(
                render_latent_values(output, name, token, gallery_latents[name][token])
            )
    render_overview(
        output,
        specification,
        records,
        gallery,
        details_by_model,
        summaries,
        images_by_token,
    )
    write_gallery_html(output, specification, gallery, summaries, figure_links)
    write_json(
        output / "metrics.json",
        {
            "complete": True,
            "cpu_only": True,
            "training_performed": False,
            "development_windows": len(records),
            "recordings": len({row["recording_group"] for row in records}),
            "metric_aggregation": "valid_patch_time_mean; not scene-macro planning ADE",
            "models": summaries,
            "per_window": details_by_model,
            "reference_images": image_sources,
            "wall_seconds": time.monotonic() - started,
        },
    )
    print(f"VISUALIZATION_COMPLETE {output}", flush=True)


if __name__ == "__main__":
    main()
