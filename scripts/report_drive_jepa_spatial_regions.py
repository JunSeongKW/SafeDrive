"""Summarize fixed-last-checkpoint region runs and render the same six scenes."""

import argparse
import html
import json
import statistics
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle
from summarize_drive_jepa_architecture_followup import paired_recording_comparison
from visualize_drive_jepa_learned_modules import (
    choose_gallery_records,
    read_reference_images,
)

WORKSPACE = Path(__file__).resolve().parents[1]


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def covered_native_cells(region_ids, region_side):
    cells = []
    for region_id in region_ids:
        top, left = (
            (region_id // (32 // region_side)) * region_side,
            (region_id % (32 // region_side)) * region_side,
        )
        cells.extend(
            (top + row) * 32 + left + column
            for row in range(region_side)
            for column in range(region_side)
        )
    return cells


def summarize_selection(windows, initial_windows, region_side):
    counts = np.zeros(512)
    initial = {row["token"]: row for row in initial_windows}
    replacement = []
    for row in windows:
        selected = row["selected_region_ids"]
        counts[covered_native_cells(selected, region_side)] += 1
        before = set(initial[row["token"]]["selected_region_ids"])
        replacement.append(1 - len(before.intersection(selected)) / len(selected))
    probabilities = counts / counts.sum()
    return {
        "native_cell_selection_counts": counts.tolist(),
        "covered_fraction_of_display_grid": float(counts.sum() / (512 * len(windows))),
        "selection_mass_in_lower_half": float(
            counts.reshape(16, 32)[8:].sum() / counts.sum()
        ),
        "across_window_distinct_native_cells": int((counts > 0).sum()),
        "mean_selected_set_replacement_fraction": float(np.mean(replacement)),
        "aggregate_selection_entropy_nats": float(
            -(
                probabilities[probabilities > 0]
                * np.log(probabilities[probabilities > 0])
            ).sum()
        ),
        "interpretation": "Distribution and coverage only; neither semantic nor planning-importance ground truth.",
    }


def image_with_regions(axis, image, region_ids, side, title):
    axis.imshow(image)
    for rank, region_id in enumerate(region_ids):
        column = region_id % (32 // side)
        row = region_id // (32 // side)
        rectangle = Rectangle(
            (column * side * 16, row * side * 16),
            side * 16,
            side * 16,
            facecolor=(1, 0.8, 0, 0.17),
            edgecolor="#ffce32",
            linewidth=1.6,
        )
        axis.add_patch(rectangle)
        axis.text(
            column * side * 16 + 1,
            row * side * 16 + 7,
            str(rank + 1),
            fontsize=6,
            color="black",
            bbox={
                "facecolor": "#ffce32",
                "alpha": 0.8,
                "pad": 0.2,
                "edgecolor": "none",
            },
        )
    axis.set_title(title, fontsize=9)
    axis.axis("off")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-directory", required=True, type=Path)
    parser.add_argument("--share-directory", required=True, type=Path)
    parser.add_argument("--gallery-directory", required=True, type=Path)
    args = parser.parse_args()
    completion = json.loads((args.run_directory / "completion.json").read_text())
    if not completion["complete"]:
        raise RuntimeError("All registered runs must finish before final comparison")
    specification = json.loads((args.run_directory / "specification.json").read_text())
    args.share_directory.mkdir(parents=True, exist_ok=False)
    args.gallery_directory.mkdir(parents=True, exist_ok=False)
    reports, final_windows, aggregates, selection_reports = {}, {}, {}, {}
    for condition, options in specification["conditions"].items():
        reports[condition] = {}
        final_windows[condition] = {}
        selection_reports[condition] = {}
        for seed in specification["seeds"]:
            report = json.loads(
                (
                    args.run_directory / f"{condition}_seed{seed}/results.json"
                ).read_text()
            )
            reports[condition][seed] = report
            final_windows[condition][seed] = report["evaluations"]["800"][
                "development"
            ]["windows"]
            selection_reports[condition][seed] = summarize_selection(
                final_windows[condition][seed],
                report["evaluations"]["0"]["development"]["windows"],
                options["region_side"],
            )
            # Full raw logs/checkpoints remain local. Keep paired dev windows and
            # scalar curves in Git without duplicating every train window at 3 times.
            compact_report = {
                key: value for key, value in report.items() if key != "evaluations"
            }
            compact_report["evaluations"] = {
                update: {
                    split: {
                        "summary": values["summary"],
                        **(
                            {"windows": values["windows"]}
                            if update == "800" and split == "development"
                            else {}
                        ),
                    }
                    for split, values in split_values.items()
                }
                for update, split_values in report["evaluations"].items()
            }
            write_json(
                args.share_directory / f"{condition}_seed{seed}.json", compact_report
            )
        aggregates[condition] = {}
        for split in ("train", "development"):
            summaries = [
                report["evaluations"]["800"][split]["summary"]
                for report in reports[condition].values()
            ]
            aggregates[condition][split] = {
                metric: {
                    "seed_values": [summary[metric] for summary in summaries],
                    "mean": statistics.mean(summary[metric] for summary in summaries),
                    "sample_std": statistics.stdev(
                        summary[metric] for summary in summaries
                    ),
                }
                for metric in (
                    "scene_macro_xy_ade_m",
                    "official_il_loss_whole_split",
                    "retained_current_planner_xy_difference_m",
                    "future_mse",
                    "persistence_mse",
                )
            }
        aggregates[condition]["command_development_ade"] = {
            command: {
                "count": next(iter(reports[condition].values()))["evaluations"]["800"][
                    "development"
                ]["summary"]["command_window_ade"][command]["count"],
                "seed_values": [
                    report["evaluations"]["800"]["development"]["summary"][
                        "command_window_ade"
                    ][command]["mean"]
                    for report in reports[condition].values()
                ],
            }
            for command in ("0", "1", "2")
        }
        aggregates[condition]["wall_seconds_all_seeds"] = sum(
            report["wall_seconds"] for report in reports[condition].values()
        )
        aggregates[condition]["peak_allocated_gib"] = max(
            report["peak_allocated_gib"] for report in reports[condition].values()
        )
        aggregates[condition]["options"] = options
    reference_reports = {
        seed: json.loads(
            (
                WORKSPACE
                / specification["reference_root"]
                / f"ego_lower_learning_rate_seed{seed}/results.json"
            ).read_text()
        )
        for seed in specification["seeds"]
    }
    reference_windows = {
        seed: report["evaluations"]["800"]["development"]["windows"]
        for seed, report in reference_reports.items()
    }
    comparisons = {
        f"{condition}_minus_preserved_patch4": paired_recording_comparison(
            windows, reference_windows
        )
        for condition, windows in final_windows.items()
    }
    for reference_condition in ("region8_random", "region8_planning"):
        comparisons[f"region8_planner_retention_minus_{reference_condition}"] = (
            paired_recording_comparison(
                final_windows["region8_planner_retention"],
                final_windows[reference_condition],
            )
        )
    summary = {
        "basis_commit": specification["basis_commit"],
        "execution_provenance": json.loads(
            (args.run_directory / "provenance.json").read_text()
        ),
        "completion": completion,
        "aggregates": aggregates,
        "paired_comparisons": comparisons,
        "selection_diagnostics": selection_reports,
        "reference_patch4_dev_ade_m": {
            seed: report["evaluations"]["800"]["development"]["summary"][
                "scene_macro_xy_ade_m"
            ]
            for seed, report in reference_reports.items()
        },
        "limitations": [
            "Repeated development split; no independent held-out or official PDMS evaluation.",
            "Frozen encoder features are globally contextual, not isolated local crops.",
            "Pooling changes targets/detail; MSE across region sizes is not like-for-like.",
            "Retention is an optimized current-planner masking proxy, not causal or future usefulness ground truth.",
            "More area/tokens changes cost. Only K8 comparisons control these budgets.",
            "Seed SD and recording bootstrap quantify different variation; exploratory multiple comparisons are not confirmatory evidence.",
        ],
    }
    write_json(args.share_directory / "summary.json", summary)
    cache_index = json.loads((WORKSPACE / specification["reused_cache"]).read_text())
    gallery_config = json.loads(
        (
            WORKSPACE
            / "configs/drive_jepa_selective_future/learned_modules_visualization_v1.json"
        ).read_text()
    )
    gallery_records = choose_gallery_records(
        [
            record
            for record in cache_index["records"]
            if record["split"] == "development"
        ],
        gallery_config["gallery_hash_salt"],
        gallery_config["gallery_windows_per_command"],
    )
    seed = 29
    panels = [
        ("preserved_patch4", 1),
        *[
            (condition, options["region_side"])
            for condition, options in specification["conditions"].items()
        ],
    ]
    file_links, provenance = [], []
    for record in gallery_records:
        images, image_provenance = read_reference_images(record)
        token = record["current_frame_token"]
        original_row = next(
            row for row in reference_windows[seed] if row["token"] == token
        )
        figure, axes = plt.subplots(2, 3, figsize=(16, 6.7), dpi=150)
        for axis, (condition, side) in zip(axes.flat, panels):
            row = (
                original_row
                if condition == "preserved_patch4"
                else next(
                    row
                    for row in final_windows[condition][seed]
                    if row["token"] == token
                )
            )
            ids = (
                row["selected_patch_ids"]
                if condition == "preserved_patch4"
                else row["selected_region_ids"]
            )
            image_with_regions(
                axis,
                images[3],
                ids,
                side,
                f"{condition} | {len(ids)} regions x {side * 16}px\nwindow ADE {row['xy_ade_m']:.3f}m",
            )
        figure.suptitle(
            f"Same current frame | command {record['command_raw_index']} | {token[:12]} | seed29/update800",
            fontsize=11,
        )
        figure.tight_layout()
        filename = f"selection_{token}.png"
        figure.savefig(args.gallery_directory / filename)
        plt.close(figure)
        file_links.append(filename)
        figure, axes = plt.subplots(1, 3, figsize=(16, 3.6), dpi=150)
        before = next(
            row
            for row in reports["region8_planner_retention"][seed]["evaluations"]["0"][
                "development"
            ]["windows"]
            if row["token"] == token
        )
        after = next(
            row
            for row in final_windows["region8_planner_retention"][seed]
            if row["token"] == token
        )
        image_with_regions(
            axes[0],
            images[3],
            original_row["selected_patch_ids"],
            1,
            "Preserved patch4 after training",
        )
        image_with_regions(
            axes[1],
            images[3],
            before["selected_region_ids"],
            2,
            "Region8 retention BEFORE new training",
        )
        image_with_regions(
            axes[2],
            images[3],
            after["selected_region_ids"],
            2,
            "Region8 retention AFTER 800 updates",
        )
        figure.tight_layout()
        before_after = f"retention_before_after_{token}.png"
        figure.savefig(args.gallery_directory / before_after)
        plt.close(figure)
        file_links.append(before_after)
        provenance.append({"record": record, "images": image_provenance})
    figure, axes = plt.subplots(1, 5, figsize=(18, 3.3), dpi=150)
    for axis, condition in zip(axes, specification["conditions"]):
        counts = (
            np.array(
                selection_reports[condition][29]["native_cell_selection_counts"]
            ).reshape(16, 32)
            / 192
        )
        display = axis.imshow(counts, vmin=0, vmax=1, cmap="magma")
        axis.set_title(condition, fontsize=9)
        axis.set_xticks([])
        axis.set_yticks([])
    figure.colorbar(
        display,
        ax=list(axes),
        label="Fraction of 192 dev windows selecting this cell",
        shrink=0.6,
    )
    figure.savefig(args.gallery_directory / "selection_frequency.png")
    plt.close(figure)
    file_links.insert(0, "selection_frequency.png")
    figure, axes = plt.subplots(1, 2, figsize=(11, 4), dpi=150)
    for condition in specification["conditions"]:
        reports_by_seed = reports[condition]
        updates = [0, 200, 800]
        mean_ade = [
            statistics.mean(
                report["evaluations"][str(update)]["development"]["summary"][
                    "scene_macro_xy_ade_m"
                ]
                for report in reports_by_seed.values()
            )
            for update in updates
        ]
        mean_retention = [
            statistics.mean(
                report["evaluations"][str(update)]["development"]["summary"][
                    "retained_current_planner_xy_difference_m"
                ]
                for report in reports_by_seed.values()
            )
            for update in updates
        ]
        axes[0].plot(updates, mean_ade, "o-", label=condition)
        axes[1].plot(updates, mean_retention, "o-", label=condition)
    axes[0].set_title("Dev scene-macro XY ADE (3-seed mean)")
    axes[1].set_title("Current teacher retention proxy (not future utility)")
    for axis in axes:
        axis.set_xlabel("Joint updates")
        axis.set_ylabel("Meters; lower is better")
        axis.grid(alpha=0.25)
    axes[0].legend(fontsize=7)
    figure.tight_layout()
    figure.savefig(args.gallery_directory / "learning_curves.png")
    plt.close(figure)
    file_links.insert(0, "learning_curves.png")
    rows_html = "".join(
        f"<tr><td>{html.escape(condition)}</td><td>{values['development']['scene_macro_xy_ade_m']['mean']:.6f} ± {values['development']['scene_macro_xy_ade_m']['sample_std']:.6f}</td><td>{values['development']['retained_current_planner_xy_difference_m']['mean']:.4f}</td></tr>"
        for condition, values in aggregates.items()
    )
    figures_html = "".join(
        f'<h2>{html.escape(filename)}</h2><a href="{filename}"><img loading="lazy" src="{filename}"></a>'
        for filename in file_links
    )
    document = f"""<!doctype html><html lang="ko"><meta charset="utf-8"><title>선택 영역 확대 비교</title><style>body{{max-width:1500px;margin:30px auto;font:16px system-ui;padding:20px;background:#f6f7fb;color:#182130}}img{{width:100%}}td,th{{padding:10px;border-bottom:1px solid #ccc}}h2{{font-size:16px;overflow-wrap:anywhere}}.note{{background:#fff2cf;padding:16px}}</style><h1>선택 영역 확대·planner 보조 신호 비교</h1><p>5조건 × 3seed × 800update. 이전과 동일한 6개 장면. 황색 사각형은 실제 선택 영역이며 미래 예측 이미지가 아닙니다.</p><p class="note">큰 영역은 2×2 native latent의 평균입니다. Encoder가 이미 전체 장면을 참고하므로 사각형 내부만 읽는 모델은 아닙니다. Retention 점수는 선택기로 학습시킨 보조 목표이며, 중요한 객체를 발견했다는 정답이나 미래 정보의 효용 증거가 아닙니다. 공식 PDMS·안전성은 이번에 평가하지 않았습니다.</p><table><tr><th>조건</th><th>Dev ADE(m), seed 평균 ± 표준편차</th><th>현재 teacher 유지 오차(m)</th></tr>{rows_html}</table>{figures_html}</html>"""
    (args.gallery_directory / "index.html").write_text(document)
    write_json(args.gallery_directory / "gallery_provenance.json", provenance)
    write_json(args.gallery_directory / "summary.json", summary)
    print(
        json.dumps(
            {
                condition: values["development"]["scene_macro_xy_ade_m"]
                for condition, values in aggregates.items()
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
