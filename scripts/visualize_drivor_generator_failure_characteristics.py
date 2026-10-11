#!/usr/bin/env python
"""Show identical generated64 candidates colored by oracle or original scorer."""
import argparse
import copy
import json
import os
from pathlib import Path
import pickle
import sys
import time

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "reference_repositories/DrivoR"))
os.environ.setdefault("NUPLAN_MAPS_ROOT", str(PROJECT_ROOT / "dataset/maps"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D
from navsim.common.dataclasses import Scene, SensorConfig, Trajectory
from navsim.visualization.camera import add_camera_ax
from navsim.visualization.bev import add_configured_bev_on_ax, add_trajectory_to_bev_ax
from navsim.visualization.plots import configure_bev_ax
from navsim.visualization.config import TRAJECTORY_CONFIG


def render(configuration):
    started = time.perf_counter()
    source = PROJECT_ROOT / configuration["source_candidate_directory"]
    destination = PROJECT_ROOT / configuration["results_directory"]
    report = json.loads((destination / "results.json").read_text())
    records = json.loads((source / "registration.json").read_text())["records"]
    candidates = np.load(source / "scorer_cache_and_candidates.npz")
    score_cache = np.load(PROJECT_ROOT / configuration["source_score_directory"] / "distribution_context_scores.npz")
    metrics = np.load(source / "official_metrics.npy")
    token_indices = {record["token"]: scene_index for scene_index, record in enumerate(records)}
    normalization = Normalize(0, 100)
    color_map = plt.get_cmap("viridis")
    rendered = []
    for example_index, example in enumerate(report["examples"]):
        figure, axes = plt.subplots(2, 3, figsize=(18, 11))
        for row_index, (role, row) in enumerate([( "High selection loss", example["failed"]), ("Same-command comparison", example["comparison"])]):
            scene_index = token_indices[row["token"]]
            record = records[scene_index]
            with (PROJECT_ROOT / "dataset/navsim_logs/test" / (record["log_name"] + ".pkl")).open("rb") as stream:
                frames = pickle.load(stream)
            cursor = record["current_frame_index"]
            sensor = SensorConfig.build_no_sensors(); sensor.cam_f0 = [3]
            scene = Scene.from_scene_dict_list(frames[cursor-3:cursor+9], PROJECT_ROOT / "dataset/sensor_blobs/test", 4, 8, sensor)
            assert scene.scene_metadata.initial_token == record["token"]
            current = scene.frames[3]
            paths = candidates["trajectories"][scene_index, :64]
            actual = metrics[scene_index, :64, -1] * 100
            predicted = np.exp(score_cache["separate_log_scores"][scene_index, :64].astype(np.float64)) * 100 / 12
            add_camera_ax(axes[row_index, 0], current.cameras.cam_f0); axes[row_index, 0].axis("off")
            axes[row_index, 0].set_title(f"{role} | {row['command_name']}\nCurrent front camera | ego {row['ego_speed_m_per_s']:.1f} m/s", fontsize=11)
            axes[row_index, 0].text(.01, -.10,
                f"{row['token']}\nDispersion {row['pairwise_rms_distance_median_m']:.1f} m | vehicles <=30m: {row['vehicle_count_within30m']}\nSelected {row['selected_actual_pdms']:.1f} / oracle {row['oracle_pdms']:.1f} | regret {row['selection_regret_pdms_points']:.1f}",
                transform=axes[row_index, 0].transAxes, fontsize=10, va="top")
            for column_index, (scores, title) in enumerate([(actual, "Official actual PDMS"), (predicted, "Original scorer's PDMS estimate")], start=1):
                axis = axes[row_index, column_index]
                add_configured_bev_on_ax(axis, scene.map_api, current)
                for candidate_index in np.argsort(scores, kind="stable"):
                    style = copy.deepcopy(TRAJECTORY_CONFIG["agent"])
                    color = color_map(normalization(scores[candidate_index]))
                    style.update(line_color=color, marker_edge_color=color, line_width=.8, marker_size=0, zorder=10)
                    add_trajectory_to_bev_ax(axis, Trajectory(paths[candidate_index]), style)
                for candidate_index, color, line_style in [(row["selected_candidate_index"], "#e3372b", "-"), (row["oracle_candidate_index"], "#1479ff", "--")]:
                    axis.plot(paths[candidate_index, :, 1], paths[candidate_index, :, 0], color=color, linewidth=2.8, linestyle=line_style, zorder=30)
                configure_bev_ax(axis)
                lateral_limit = max(15, float(np.abs(paths[..., 1]).max()) + 3)
                axis.set_xlim(lateral_limit, -lateral_limit)
                axis.set_ylim(min(-6, float(paths[..., 0].min()) - 3), max(25, float(paths[..., 0].max()) + 4))
                axis.set_xlabel("Ego lateral (m), left positive"); axis.set_ylabel("Ego forward (m)")
                axis.set_title(f"Same 64 generated paths | {title}\nSelected actual NC/DAC/TTC: {row['selected_no_at_fault_collisions']:.1f}/{row['selected_drivable_area_compliance']:.1f}/{row['selected_time_to_collision_within_bound']:.1f}", fontsize=11)
                figure.colorbar(plt.cm.ScalarMappable(norm=normalization, cmap=color_map), ax=axis, shrink=.78, label="PDMS points (0-100)", pad=.015)
        figure.legend(handles=[Line2D([0], [0], color="#e3372b", linewidth=3, label="Scorer-selected path"), Line2D([0], [0], color="#1479ff", linestyle="--", linewidth=3, label="Oracle-best among these 64")], loc="lower center", ncol=2, bbox_to_anchor=(.55, .035))
        figure.suptitle("Generator set shape and scorer selection: error / comparison examples\nNAVSIM current camera, map and GT boxes; path colors change, candidate coordinates do not", fontsize=14)
        figure.text(.5, .012, "Cases selected after inspecting regret. Comparison is approximate, not causal. Oracle PDMS uses simulated tracking + future GT; plotted boxes are current GT.", ha="center", fontsize=9)
        figure.tight_layout(rect=(0, .085, 1, .92))
        image_path = destination / f"generator_error_vs_comparison_case{example_index+1}_navsim.png"
        figure.savefig(image_path, dpi=140); plt.close(figure)
        rendered.append({"image": str(image_path.relative_to(PROJECT_ROOT)), "failed_token": example["failed"]["token"], "comparison_token": example["comparison"]["token"], "selection": example["selection"]})
    manifest = {"examples_selected_after_error_analysis": True, "cases": rendered,
        "backend": "Upstream NAVSIM camera, configured BEV, trajectory overlay", "cpu_wall_seconds": time.perf_counter() - started, "gpu_seconds": 0}
    (destination / "visualization_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(); parser.add_argument("--configuration", type=Path, required=True)
    arguments = parser.parse_args(); render(json.loads(arguments.configuration.read_text()))
