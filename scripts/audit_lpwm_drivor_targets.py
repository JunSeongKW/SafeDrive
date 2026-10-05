"""Compare cached GT and online labels against fresh official DrivoR builders."""
import json
import faulthandler
import os
from pathlib import Path
import pickle
import sys

import numpy as np
from omegaconf import OmegaConf
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
os.environ["NUPLAN_MAPS_ROOT"] = str(PROJECT_ROOT / "outputs/lpwm_drivor_joint_v1/maps")
os.environ["NUPLAN_MAP_VERSION"] = "nuplan-maps-v1.0"
sys.path.insert(0, str(PROJECT_ROOT / "reference_repositories/DrivoR"))
from navsim.common.dataclasses import Scene, SensorConfig
from navsim.agents.drivoR.drivor_features import DrivoRTargetBuilder
from navsim.planning.scenario_builder.navsim_scenario import NavSimScenario
from navsim.planning.metric_caching.train_cache_processor import MetricCacheProcessor
from navsim.agents.drivoR.score_module.compute_navsim_score import get_sub_score
from lpwm_drivor_oracle import initialize_worker, prepare_training_cache


def run():
    faulthandler.dump_traceback_later(60, repeat=True)
    manifest = PROJECT_ROOT / "outputs/lpwm_drivor_joint_v1/engineering_scene_cache/manifest.json"
    output = PROJECT_ROOT / "outputs/lpwm_drivor_joint_v1/engineering_audit"
    output.mkdir(parents=True, exist_ok=True)
    configuration = OmegaConf.create(yaml.safe_load((PROJECT_ROOT / "reference_repositories/DrivoR/navsim/planning/script/config/common/agent/drivoR.yaml").read_text())["config"])
    configuration.trajectory_sampling.num_poses = 8
    configuration.long_trajectory_additional_poses = 2
    builder = DrivoRTargetBuilder(configuration)
    records = json.loads(manifest.read_text())["records"]
    initialize_worker(manifest, PROJECT_ROOT / "outputs/lpwm_drivor_joint_v1/train_metric_cache")
    rows = []
    for record in records[:2]:
        with (PROJECT_ROOT / "dataset/navsim_logs/trainval" / (record["log_name"]+".pkl")).open("rb") as stream:
            frames = pickle.load(stream)
        index = record["current_frame_index"]
        scene = Scene.from_scene_dict_list(frames[index-3:index+11], None, 4, 10, SensorConfig.build_no_sensors())
        official = builder.compute_targets(scene)
        row = {"token": record["token"]}
        for name in ("trajectory", "trajectory_long"):
            stored = np.load(Path(record["cache_directory"]) / (name+".npy"))[record["cache_row"]]
            delta = float(np.max(np.abs(official[name].numpy()-stored)))
            assert delta < 1e-4, (name,delta)
            row[name+"_max_error"] = delta
        status = scene.get_agent_input().ego_statuses[-1]
        expected_status = np.r_[status.ego_pose, status.ego_velocity, status.ego_acceleration, status.driving_command]
        actual_status = np.load(Path(record["cache_directory"]) / "ego.npy")[record["cache_row"]]
        assert np.max(np.abs(expected_status-actual_status)) < 1e-4
        print("GT_PARITY", row, flush=True)
        scenario = NavSimScenario(scene, os.environ["NUPLAN_MAPS_ROOT"], "nuplan-maps-v1.0")
        fresh = MetricCacheProcessor(str(output / "fresh_official_atomic_cache"), False).compute_metric_cache(scenario)
        converted = prepare_training_cache(record["token"])
        poses = official["trajectory"].numpy().astype(np.float32)
        candidates = np.stack([poses, poses*.5, poses*1.5, np.zeros_like(poses)])
        scores = get_sub_score(converted, candidates, test=True)[0]
        fresh_scores = get_sub_score(fresh.file_name, candidates, test=True)[0]
        complete_scores = get_sub_score(converted, candidates, test=False)[0]
        row["converted_vs_fresh_subscores_max_error"] = float(np.max(np.abs(scores-fresh_scores)))
        row["test_flag_subscores_max_error"] = float(np.max(np.abs(scores-complete_scores)))
        assert np.allclose(scores, fresh_scores, atol=1e-6, rtol=1e-6), row
        assert np.array_equal(scores, complete_scores), row
        rows.append(row)
        print(json.dumps(row), flush=True)
    (output / "official_target_oracle_parity.json").write_text(json.dumps({"passed": True, "scenes": rows}, indent=2))


if __name__ == "__main__":
    run()
