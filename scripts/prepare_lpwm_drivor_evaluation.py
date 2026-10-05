"""Use official scene loaders to cache ONLY four current camera/ego inputs."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import sys

import numpy as np
from PIL import Image
from hydra.utils import instantiate
from omegaconf import OmegaConf

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def prepare(arguments):
    version_two = arguments.split != "navtest"
    reference = PROJECT_ROOT / "reference_repositories" / ("NAVSIMOfficialV2LPWMDrivoR" if version_two else "DrivoR")
    sys.path.insert(0, str(reference))
    os.environ["NUPLAN_MAPS_ROOT"] = str(PROJECT_ROOT / "outputs/lpwm_drivor_joint_v1/maps")
    from navsim.common.dataclasses import SensorConfig
    from navsim.common.dataloader import SceneLoader
    split = OmegaConf.load(reference / "navsim/planning/script/config/common/train_test_split/scene_filter" / (arguments.split+".yaml"))
    sensors = SensorConfig(cam_f0=[3], cam_b0=[3], cam_l0=[3], cam_l1=[], cam_l2=[], cam_r0=[3],cam_r1=[],cam_r2=[],lidar_pc=[])
    if version_two:
        loader = SceneLoader(data_path=PROJECT_ROOT / "dataset/navsim_logs/test",
            original_sensor_path=PROJECT_ROOT / "dataset/sensor_blobs/test",
            synthetic_sensor_path=PROJECT_ROOT / "dataset" / arguments.split / "sensor_blobs",
            synthetic_scenes_path=PROJECT_ROOT / "dataset" / arguments.split / "synthetic_scene_pickles",
            scene_filter=instantiate(split), sensor_config=sensors)
    else:
        loader = SceneLoader(data_path=PROJECT_ROOT / "dataset/navsim_logs/test",
            sensor_blobs_path=PROJECT_ROOT / "dataset/sensor_blobs/test", scene_filter=instantiate(split), sensor_config=sensors)
    tokens = sorted(loader.tokens)
    expected = set(split.tokens or [])
    if version_two:
        expected.update(split.get("reactive_synthetic_initial_tokens", []) or [])
        expected.update(split.get("non_reactive_synthetic_initial_tokens", []) or [])
    assert expected == set(tokens), (len(expected-set(tokens)), len(set(tokens)-expected))
    if not version_two:
        assert len(tokens) == 12146
    output = PROJECT_ROOT / "outputs/lpwm_drivor_joint_v1/evaluation_inputs" / arguments.split
    output.mkdir(parents=True,exist_ok=True)
    if (output / "manifest.json").exists():
        previous = json.loads((output / "manifest.json").read_text())
        assert previous["tokens"] == tokens and previous["complete"]
        return
    images = np.lib.format.open_memmap(output / "images.npy",mode="w+",dtype=np.uint8,shape=(len(tokens),4,128,128,3))
    ego = np.lib.format.open_memmap(output / "ego.npy",mode="w+",dtype=np.float32,shape=(len(tokens),11))
    def load(token):
        inputs = loader.get_agent_input_from_token(token)
        current = inputs.cameras[-1]
        rgb = np.stack([np.asarray(Image.fromarray(camera.image).resize((128,128),Image.Resampling.BICUBIC))
            for camera in (current.cam_f0,current.cam_b0,current.cam_l0,current.cam_r0)])
        status = inputs.ego_statuses[-1]
        return rgb, np.r_[status.ego_pose,status.ego_velocity,status.ego_acceleration,status.driving_command]
    with ThreadPoolExecutor(max_workers=arguments.workers) as pool:
        for index, (rgb,status) in enumerate(pool.map(load,tokens)):
            images[index],ego[index] = rgb,status
            if (index+1)%500 == 0:
                print(json.dumps({"split":arguments.split,"completed":index+1,"total":len(tokens)}),flush=True)
    images.flush(); ego.flush()
    (output / "manifest.json").write_text(json.dumps({"complete":True,"split":arguments.split,"tokens":tokens,
        "future_images_used":False,"object_gt_used":False,"source_reference":str(reference)},indent=2))
    print("EVALUATION_INPUTS_READY",arguments.split,len(tokens),flush=True)


if __name__ == "__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--split",choices=["navtest","warmup_two_stage","navhard_two_stage"],required=True)
    parser.add_argument("--workers",type=int,default=4)
    prepare(parser.parse_args())
