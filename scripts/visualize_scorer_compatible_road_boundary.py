#!/usr/bin/env python
"""Show the official tracked footprint that flips DAC after head adaptation."""
import argparse
import json
import lzma
import os
from pathlib import Path
import pickle
import sys
import time

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "reference_repositories/DrivoR"))
os.environ.setdefault("NUPLAN_MAPS_ROOT", str(PROJECT_ROOT / "dataset/maps"))


def render(configuration):
    started = time.perf_counter(); output = PROJECT_ROOT / configuration["output_directory"]; destination = PROJECT_ROOT / configuration["results_directory"]
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Polygon as PlotPolygon
    from shapely.ops import unary_union, transform
    from nuplan.common.maps.maps_datatypes import SemanticMapLayer
    from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling
    from navsim.common.dataclasses import Trajectory
    from navsim.evaluate.pdm_score import get_trajectory_as_array, transform_trajectory
    from navsim.planning.simulation.planner.pdm_planner.scoring.pdm_scorer import PDMScorer
    from navsim.planning.simulation.planner.pdm_planner.simulation.pdm_simulator import PDMSimulator
    from navsim.planning.simulation.planner.pdm_planner.utils.pdm_enums import EgoAreaIndex, MultiMetricIndex
    failure = json.loads((destination / "outcome_diagnosis.json").read_text())["development_decomposition"]["C_compatible"]["worst_scene"]
    prediction = json.loads((output / "prediction_complete.json").read_text()); records = prediction["records"]
    scene_index = next(index for index, record in enumerate(records) if record["token"] == failure["token"]); record = records[scene_index]
    with lzma.open(record["metric_cache_file"], "rb") as stream:
        cache = pickle.load(stream)
    arrays = np.load(output / "evaluation_candidates.npz"); selected_index = failure["current_selected_index"]
    trajectories = arrays["trajectories"][[0, 4], scene_index, selected_index]
    sampling = TrajectorySampling(num_poses=40, interval_length=0.1); initial = cache.ego_state
    states = np.stack([get_trajectory_as_array(cache.trajectory, sampling, initial.time_point)] +
        [get_trajectory_as_array(transform_trajectory(Trajectory(path), initial), sampling, initial.time_point) for path in trajectories])
    simulated = PDMSimulator(sampling).simulate_proposals(states, initial)
    scorer = PDMScorer(sampling); scorer.score_proposals(simulated, cache.observation, cache.centerline, cache.route_lane_ids, cache.drivable_area_map)
    offroad = scorer._ego_areas[1:, :, EgoAreaIndex.NON_DRIVABLE_AREA]
    first_failure = int(np.flatnonzero(offroad[1])[0]); assert not offroad[0].any()
    assert np.array_equal(scorer._multi_metrics[MultiMetricIndex.DRIVABLE_AREA, 1:], [1., 0.])
    yaw = float(initial.rear_axle.heading); cosine, sine = np.cos(yaw), np.sin(yaw)
    inverse_rotation = np.array([[cosine, -sine], [sine, cosine]])
    origin = np.array([initial.rear_axle.x, initial.rear_axle.y])
    ego_coordinates = (scorer._ego_coords[1:] - origin) @ inverse_rotation
    drivable_indices = cache.drivable_area_map.get_indices_of_map_type([SemanticMapLayer.ROADBLOCK, SemanticMapLayer.INTERSECTION, SemanticMapLayer.DRIVABLE_AREA, SemanticMapLayer.CARPARK_AREA])
    drivable_global = unary_union([cache.drivable_area_map._geometries[index] for index in drivable_indices])
    def ego_transform(horizontal, vertical, altitude=None):
        horizontal = np.asarray(horizontal) - origin[0]; vertical = np.asarray(vertical) - origin[1]
        return cosine * horizontal + sine * vertical, -sine * horizontal + cosine * vertical
    drivable_ego = transform(ego_transform, drivable_global)
    geometries = list(drivable_ego.geoms) if hasattr(drivable_ego, "geoms") else [drivable_ego]
    figure, axes = plt.subplots(1, 2, figsize=(13, 6))
    for polygon in geometries:
        boundary = np.asarray(polygon.exterior.coords)
        axes[0].fill(boundary[:, 1], boundary[:, 0], color="#eeeeee", zorder=1)
        axes[0].plot(boundary[:, 1], boundary[:, 0], color="#444444", linewidth=1.5, zorder=2)
        for ring in polygon.interiors:
            interior = np.asarray(ring.coords); axes[0].fill(interior[:, 1], interior[:, 0], color="white", zorder=2)
            axes[0].plot(interior[:, 1], interior[:, 0], color="#444444", linewidth=1.5, zorder=3)
    for model_index, label, color in [(0, "A original: DAC=1", "#2676c4"), (1, "C adapted: DAC=0", "#d43432")]:
        footprint = ego_coordinates[model_index, first_failure, :4]
        axes[0].add_patch(PlotPolygon(footprint[:, [1, 0]], fill=False, edgecolor=color, linewidth=2.5, label=label, zorder=5))
        axes[0].scatter(footprint[:, 1], footprint[:, 0], color=color, s=30, zorder=6)
        centers = ego_coordinates[model_index, :, -1]
        axes[0].plot(centers[:, 1], centers[:, 0], color=color, linestyle="--", alpha=.7, linewidth=1, zorder=4)
        axes[1].step(np.arange(offroad.shape[1]) * .1, offroad[model_index].astype(int), where="post", color=color, linewidth=2, label=label)
    center = ego_coordinates[1, first_failure, :4].mean(axis=0)
    axes[0].set_xlim(center[1] + 3.5, center[1] - 3.5); axes[0].set_ylim(center[0] - 3.5, center[0] + 3.5); axes[0].set_aspect("equal")
    axes[0].set_xlabel("Ego lateral (m), left positive"); axes[0].set_ylabel("Ego forward (m)")
    axes[0].set_title(f"Tracked footprint at first off-road instant: t={first_failure*.1:.1f}s\nGray=official drivable union; corners evaluated by PDM"); axes[0].legend(fontsize=8)
    axes[1].axvline(first_failure*.1, color="#555555", linewidth=1, linestyle="--")
    axes[1].set_yticks([0, 1], ["All 4 corners inside", "At least 1 corner outside"]); axes[1].set_ylim(-.1, 1.15)
    axes[1].set_xlabel("Tracked rollout time (s)"); axes[1].set_title("Official non-drivable-area flag over 4 seconds"); axes[1].legend(fontsize=8); axes[1].grid(alpha=.2)
    figure.suptitle(f"Stage 4 | {record['token']} | same selected query37 before/after\nSmall waypoint changes can cross a binary DAC boundary", fontsize=12)
    figure.tight_layout(rect=(0, .055, 1, .89))
    figure.text(.5, .02, "Official DAC is zero if any tracked footprint corner leaves the drivable union at any time. This uses metric-cache maps for diagnosis, not model input.", ha="center", fontsize=9)
    path = destination / "stage4_tracked_footprint_road_boundary.png"; figure.savefig(path, dpi=145); plt.close(figure)
    measurements = {"token": record["token"], "first_C_offroad_time_seconds": first_failure*.1,
        "A_offroad_timestep_count": int(offroad[0].sum()), "C_offroad_timestep_count": int(offroad[1].sum()),
        "tracked_footprint_center_difference_at_failure_m": float(np.linalg.norm(ego_coordinates[1, first_failure, :4].mean(axis=0) - ego_coordinates[0, first_failure, :4].mean(axis=0))),
        "A_C_actual_dac": [1., 0.], "diagnostic_tracked_trajectories_including_reference": 3,
        "cpu_wall_seconds": time.perf_counter()-started, "gpu_seconds": 0, "image": str(path.relative_to(PROJECT_ROOT))}
    (destination / "tracked_footprint_diagnosis.json").write_text(json.dumps(measurements, indent=2) + "\n")
    with (destination / "report.md").open("a") as stream:
        stream.write(f"\n### 실제 tracked footprint와 도로 경계\n\n[확대 시각화](stage4_tracked_footprint_road_boundary.png): 공식tracker rollout에서C는{first_failure*.1:.1f}초에처음drivable영역밖corner가발생했다. A는전구간0회/C는{int(offroad[1].sum())}timestep이다. DAC는중심선만이아니라차량footprint4corner중하나라도주행가능영역밖에나가면0이된다. 앞선BEV는요청waypoint였고이그림은실제trackedfootprint다.\n")
    print(json.dumps(measurements), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(); parser.add_argument("--configuration", type=Path, required=True)
    arguments = parser.parse_args(); render(json.loads(arguments.configuration.read_text()))
