"""Compare cached ego inputs to DrivoR's official feature builder, without RGB."""
import json
import os
from pathlib import Path
import pickle
import sys
from unittest.mock import patch

import numpy as np
from omegaconf import OmegaConf
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
os.environ["NUPLAN_MAPS_ROOT"] = str(PROJECT_ROOT / "outputs/lpwm_drivor_joint_v1/maps")
os.environ["NUPLAN_MAP_VERSION"] = "nuplan-maps-v1.0"
sys.path.insert(0, str(PROJECT_ROOT / "reference_repositories/DrivoR"))
from navsim.common.dataclasses import Scene, SensorConfig
from navsim.agents.drivoR.drivor_features import DrivoRFeatureBuilder


def main():
    configuration = OmegaConf.create(yaml.safe_load((PROJECT_ROOT /
        "reference_repositories/DrivoR/navsim/planning/script/config/common/agent/drivoR.yaml").read_text())["config"])
    assert not configuration.full_history_status
    builder = DrivoRFeatureBuilder(configuration)
    manifest = json.loads((PROJECT_ROOT / "outputs/lpwm_drivor_joint_v1/scene_cache/manifest.json").read_text())
    selected_records = []
    for split in ("navtrain", "navval"):
        candidates = [record for record in manifest["records"] if record["split"] == split]
        selected_records.extend(candidates[index] for index in np.linspace(0, len(candidates)-1, 8, dtype=int))
    rows = []
    for record in selected_records:
        with (PROJECT_ROOT / "dataset/navsim_logs/trainval" / (record["log_name"] + ".pkl")).open("rb") as stream:
            frames = pickle.load(stream)
        current_index = record["current_frame_index"]
        scene = Scene.from_scene_dict_list(frames[current_index-3:current_index+9], None, 4, 8,
                                          SensorConfig.build_no_sensors())
        # Skip only RGB preparation; execute the untouched official ego assembly.
        with patch.object(builder, "_get_camera_feature", return_value={}):
            official = builder.compute_features(scene.get_agent_input())["ego_status"][-1].numpy()
        cached = np.load(Path(record["cache_directory"]) / "ego.npy", mmap_mode="r")[record["cache_row"]]
        difference = np.abs(official-cached)
        assert cached.shape == official.shape == (11,)
        assert np.allclose(official, cached, atol=1e-6, rtol=0), difference
        rows.append({"token": record["token"], "split": record["split"],
            "pose_max_absolute_error": float(difference[:3].max()),
            "velocity_acceleration_command_exact": bool(np.array_equal(official[3:], cached[3:])),
            "max_absolute_error": float(difference.max())})
    result = {"passed": True, "scene_count": len(rows), "full_history_status": False,
        "ego_fields": ["pose_x", "pose_y", "pose_heading", "velocity_x", "velocity_y",
                       "acceleration_x", "acceleration_y", "command_0", "command_1", "command_2", "command_3"],
        "planner_ego_injection": "Official DrivoRModel.forward, Linear(11,256); add to proposal tokens and post-attention scorer features",
        "additional_lpwm_path": "Only command[7:11] -> trainable FiLM at particle attribute CNN conv_in; absent in official DrivoR",
        "all_ego_injection_identical": False, "rows": rows}
    output = PROJECT_ROOT / "results/lpwm_drivor_lora_v1/ego_status_audit.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
