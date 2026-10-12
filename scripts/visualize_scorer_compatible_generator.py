#!/usr/bin/env python
"""Actual NAVSIM camera/BEV for post-hoc changed-generator success/failure cases."""
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
    started = time.perf_counter(); output = PROJECT_ROOT / configuration["output_directory"]; destination = PROJECT_ROOT / configuration["results_directory"]
    metadata = json.loads((output / "prediction_complete.json").read_text()); records = metadata["records"]
    arrays = np.load(output / "evaluation_candidates.npz"); metrics = np.load(output / "evaluation_metrics.npy")
    model_indices = [0, 3, 4]; model_labels = ["A original", "B quality matching", "C quality + reliability"]
    log_scores = arrays["log_scores"]; selected = log_scores.argmax(axis=-1)
    actual = metrics[..., -1] * 100; chosen_actual = actual[np.arange(5)[:, None], np.arange(len(records))[None], selected]
    development_indices = np.asarray([index for index, record in enumerate(records) if record["pilot_split"] == "locked_development"])
    differences = chosen_actual[4, development_indices] - chosen_actual[0, development_indices]
    examples = [("largest_improvement", int(development_indices[np.argmax(differences)])), ("largest_degradation", int(development_indices[np.argmin(differences)]))]
    normalized = Normalize(0, 100); color_map = plt.get_cmap("viridis"); manifest = []
    for role, scene_index in examples:
        record = records[scene_index]
        with (PROJECT_ROOT / "dataset/navsim_logs/test" / (record["log_name"] + ".pkl")).open("rb") as stream:
            frames = pickle.load(stream)
        cursor = record["current_frame_index"]; sensor = SensorConfig.build_no_sensors(); sensor.cam_f0 = [3]
        scene = Scene.from_scene_dict_list(frames[cursor-3:cursor+9], PROJECT_ROOT / "dataset/sensor_blobs/test", 4, 8, sensor)
        assert scene.scene_metadata.initial_token == record["token"]
        current = scene.frames[3]
        figure, axes = plt.subplots(1, 4, figsize=(22, 7))
        add_camera_ax(axes[0], current.cameras.cam_f0); axes[0].axis("off"); axes[0].set_title("Current front camera\n" + record["command_name"])
        all_paths = arrays["trajectories"][model_indices, scene_index]
        lateral_limit = max(12, float(np.abs(all_paths[..., 1]).max()) + 3)
        forward_lower = min(-6, float(all_paths[..., 0].min()) - 3); forward_upper = max(25, float(all_paths[..., 0].max()) + 4)
        selected_case_scores = []
        for column_index, (model_index, label) in enumerate(zip(model_indices, model_labels), start=1):
            axis = axes[column_index]; add_configured_bev_on_ax(axis, scene.map_api, current)
            paths = arrays["trajectories"][model_index, scene_index]; scores = actual[model_index, scene_index]
            for candidate_index in np.argsort(scores):
                style = copy.deepcopy(TRAJECTORY_CONFIG["agent"]); color = color_map(normalized(scores[candidate_index]))
                style.update(line_color=color, marker_edge_color=color, line_width=.7, marker_size=0, zorder=10)
                add_trajectory_to_bev_ax(axis, Trajectory(paths[candidate_index]), style)
            chosen = int(selected[model_index, scene_index]); oracle = int(scores.argmax())
            axis.plot(paths[chosen, :, 1], paths[chosen, :, 0], color="#e3372b", linewidth=2.7, zorder=30)
            axis.plot(paths[oracle, :, 1], paths[oracle, :, 0], color="#1479ff", linestyle="--", linewidth=2.7, zorder=31)
            configure_bev_ax(axis); axis.set_xlim(lateral_limit, -lateral_limit); axis.set_ylim(forward_lower, forward_upper)
            axis.set_xlabel("Ego lateral (m), left positive"); axis.set_ylabel("Ego forward (m)")
            prediction = float(np.exp(log_scores[model_index, scene_index, chosen]) / 12 * 100)
            submetrics = metrics[model_index, scene_index, chosen]
            axis.set_title(f"{label} | generated64\nSelected actual {scores[chosen]:.2f} / estimate {prediction:.2f}\nOracle {scores[oracle]:.2f}; NC/DAC/TTC {submetrics[0]:.0f}/{submetrics[1]:.0f}/{submetrics[3]:.0f}", fontsize=10)
            selected_case_scores.append({"condition": label, "selected_actual_pdms": float(scores[chosen]), "selected_predicted_pdms": prediction, "oracle_pdms": float(scores[oracle])})
        figure.legend(handles=[Line2D([0], [0], color="#e3372b", linewidth=3, label="Scorer-selected path"), Line2D([0], [0], color="#1479ff", linestyle="--", linewidth=3, label="Oracle best in this set")], loc="lower center", ncol=2, bbox_to_anchor=(.6, .065))
        figure.suptitle(f"Stage 3 | {role} | {record['token']}\nEach model generates its own 64 paths; same scene/scorer/evaluator and no CEM", fontsize=13)
        figure.text(.5, .025, "Post-hoc illustrative case, not representative. Current GT boxes; official PDMS uses tracked rollout and future GT. Colors show actual quality, not scorer attention.", ha="center", fontsize=9)
        figure.subplots_adjust(left=.02, right=.925, bottom=.16, top=.80, wspace=.24)
        colorbar_axis = figure.add_axes([.947, .24, .009, .43])
        figure.colorbar(plt.cm.ScalarMappable(norm=normalized, cmap=color_map), cax=colorbar_axis, label="Official actual PDMS (0-100)")
        path = destination / ("stage3_" + role + "_navsim.png"); figure.savefig(path, dpi=140); plt.close(figure)
        manifest.append({"role": role, "token": record["token"], "image": str(path.relative_to(PROJECT_ROOT)), "models": selected_case_scores})
    metadata = {"post_hoc_examples": manifest, "backend": "Upstream NAVSIM camera / configured BEV / trajectory overlay", "cpu_wall_seconds": time.perf_counter() - started, "gpu_seconds": 0}
    (destination / "visualization_manifest.json").write_text(json.dumps(metadata, indent=2) + "\n")
    with (destination / "report.md").open("a") as stream:
        stream.write("\n## NAVSIM 실제 사례\n\n")
        for case in manifest:
            stream.write(f"- [{case['role']} / {case['token']}]({Path(case['image']).name}): post-hoc C−A 변화 사례.\n")
    print(json.dumps(metadata), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(); parser.add_argument("--configuration", type=Path, required=True)
    arguments = parser.parse_args(); render(json.loads(arguments.configuration.read_text()))
