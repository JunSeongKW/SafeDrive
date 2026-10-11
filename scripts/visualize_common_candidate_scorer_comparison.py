#!/usr/bin/env python
"""NAVSIM camera/BEV and candidate scores for explicitly selected error cases."""
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
from matplotlib.lines import Line2D
from navsim.common.dataclasses import Scene, SensorConfig, Trajectory
from navsim.visualization.camera import add_camera_ax
from navsim.visualization.bev import add_configured_bev_on_ax, add_trajectory_to_bev_ax
from navsim.visualization.plots import configure_bev_ax
from navsim.visualization.config import TRAJECTORY_CONFIG

from calibrate_frozen_drivor_scorer import write_json


def render(configuration):
    started = time.perf_counter()
    output = PROJECT_ROOT / configuration["output_directory"]
    destination = PROJECT_ROOT / configuration["results_directory"]
    records = json.loads((output / "registration.json").read_text())["records"]
    arrays = np.load(output / "scorer_cache_and_candidates.npz")
    metrics = np.load(output / "official_metrics.npy")
    selection = []
    used_native_indices = set()
    for pool_name in ("balanced_native128", "nearby64"):
        begin, end = configuration["candidate_pools"][pool_name]
        actual_pdms = metrics[:, begin:end, -1] * 100
        selected = arrays["neural_log_scores"][:, :, begin:end].argmax(axis=2)
        chosen_actual = actual_pdms[np.arange(len(records))[None], selected]
        worst_regret = actual_pdms.max(axis=1) - chosen_actual.min(axis=0)
        ordered = np.argsort(-worst_regret, kind="stable")
        for rank, scene_index in enumerate(ordered[:2] if pool_name == "balanced_native128" else ordered[:1]):
            selection.append((pool_name, int(scene_index), "largest_selection_regret_rank" + str(rank+1)))
    colors = ("#2475c6", "#e98126", "#299348")
    labels = ("DrivoR public", "TOAD-released DrivoR", "Oracle best candidate")
    rendered = []
    for case_index, (pool_name, scene_index, reason) in enumerate(selection):
        record = records[scene_index]
        with (PROJECT_ROOT / "dataset/navsim_logs/test" / (record["log_name"] + ".pkl")).open("rb") as stream:
            frames = pickle.load(stream)
        cursor = record["current_frame_index"]
        assert frames[cursor]["token"] == record["token"]
        sensor = SensorConfig.build_no_sensors()
        sensor.cam_f0 = [3]
        scene = Scene.from_scene_dict_list(frames[cursor-3:cursor+9], PROJECT_ROOT / "dataset/sensor_blobs/test", 4, 8, sensor)
        assert scene.scene_metadata.initial_token == record["token"]
        current = scene.frames[3]
        begin, end = configuration["candidate_pools"][pool_name]
        actual = metrics[scene_index, begin:end, -1] * 100
        log_scores = arrays["neural_log_scores"][:, scene_index, begin:end]
        predicted = np.exp(log_scores.astype(np.float64)) / 12 * 100
        chosen_indices = [int(log_scores[0].argmax()), int(log_scores[1].argmax()), int(actual.argmax())]
        chosen_paths = [arrays["trajectories"][scene_index, begin+index] for index in chosen_indices]
        figure, axes = plt.subplots(1, 3, figsize=(18, 6))
        add_camera_ax(axes[0], current.cameras.cam_f0)
        axes[0].axis("off"); axes[0].set_title("Current front camera\nSame scene for both scorers")
        add_configured_bev_on_ax(axes[1], scene.map_api, current)
        for index, (path, color, label) in enumerate(zip(chosen_paths, colors, labels)):
            style = copy.deepcopy(TRAJECTORY_CONFIG["agent"])
            style.update(line_color=color, marker_edge_color=color, line_width=2.8, marker_size=4,
                zorder=10+index)
            add_trajectory_to_bev_ax(axes[1], Trajectory(path), style)
        configure_bev_ax(axes[1])
        all_poses = np.concatenate(chosen_paths)
        lateral = max(12, float(np.abs(all_poses[:, 1]).max())+4)
        axes[1].set_xlim(lateral, -lateral)
        axes[1].set_ylim(min(-8, float(all_poses[:, 0].min())-3), max(30, float(all_poses[:, 0].max())+5))
        axes[1].set_xlabel("Ego lateral (m), left positive")
        axes[1].set_ylabel("Ego forward (m)")
        axes[1].set_title("NAVSIM map + current GT objects\nPlanned waypoints; overlapping paths may hide")
        representatives = sorted(set(chosen_indices))
        positions = np.arange(len(representatives))
        for model_index in range(2):
            axes[2].bar(positions + (-.24 if model_index == 0 else 0), predicted[model_index, representatives],
                width=.23, color=colors[model_index], label=labels[model_index]+" prediction")
        axes[2].bar(positions+.24, actual[representatives], width=.23, color=colors[2], label="Official actual PDMS")
        axis_labels = []
        for index in representatives:
            roles = [role for role, chosen in zip(("D", "T", "O"), chosen_indices) if chosen == index]
            axis_labels.append("Candidate " + str(index) + "\nchosen by " + "/".join(roles))
        axes[2].set_xticks(positions, axis_labels)
        axes[2].set_ylim(0,110); axes[2].set_ylabel("PDMS estimate / actual PDMS")
        axes[2].set_title("The same candidate can be mis-scored\nD=DrivoR, T=TOAD release, O=oracle")
        axes[2].legend(fontsize=8)
        for x_value, index in zip(positions, representatives):
            axes[2].text(x_value+.24, actual[index]+1, f"{actual[index]:.1f}", ha="center", fontsize=8)
        handles = [Line2D([0], [0], color=color, linewidth=3, label=label+f" | actual {actual[index]:.2f}")
            for color, label, index in zip(colors, labels, chosen_indices)]
        figure.legend(handles=handles, loc="lower center", ncol=3, bbox_to_anchor=(.5,.05), fontsize=10)
        figure.suptitle(pool_name+" | "+record["token"]+" | "+record["command_name"]+"\nError example selected after scoring: "+reason, fontsize=12)
        figure.text(.5,.015,"Oracle = highest official PDMS among these candidates, not the human GT path. PDMS uses simulated tracking and future GT.", ha="center", fontsize=9)
        figure.tight_layout(rect=(0,.14,1,.88))
        path = destination / ("case_" + str(case_index+1) + "_" + pool_name + "_navsim.png")
        figure.savefig(path, dpi=145); plt.close(figure)
        rendered.append({"token": record["token"], "pool": pool_name, "selection_reason": reason,
            "image": str(path.relative_to(PROJECT_ROOT)), "selected_indices": chosen_indices,
            "selected_actual_pdms": [float(actual[index]) for index in chosen_indices],
            "selected_scorer_pdms_estimates": [float(predicted[model_index, chosen_indices[model_index]]) for model_index in range(2)]})
    write_json(destination / "visualization_manifest.json", {"examples_selected_after_inspecting_regret": True,
        "cases": rendered, "visualization_backend": "Upstream NAVSIM camera / configured BEV / trajectory functions",
        "cpu_seconds": time.perf_counter()-started, "gpu_seconds": 0})
    with (destination / "report.md").open("a") as stream:
        stream.write("\n## NAVSIM 오류 사례\n\n")
        for case in rendered:
            stream.write("- ["+case["pool"]+" / "+case["token"]+"]("+Path(case["image"]).name+"): 실제 PDMS(DrivoR/TOAD배포/Oracle) "+" / ".join(f"{value:.2f}" for value in case["selected_actual_pdms"])+".\n")
    write_json(output / "navsim_visualization_complete.json", {"complete": True, "cases": rendered,
        "cpu_seconds": time.perf_counter()-started, "gpu_seconds": 0})
    print(json.dumps({"complete": True, "images": [case["image"] for case in rendered]}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--configuration", type=Path, required=True)
    arguments = parser.parse_args()
    render(json.loads(arguments.configuration.read_text()))
