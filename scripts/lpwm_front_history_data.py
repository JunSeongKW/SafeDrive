"""Read the preserved four observed front frames and official ego/GT targets."""
from functools import lru_cache
import json
from pathlib import Path

import numpy as np
from torch.utils.data import Dataset

PROJECT_ROOT = Path(__file__).resolve().parents[1]


@lru_cache(maxsize=64)
def observed_frame_cache(cache_path):
    return np.load(cache_path, mmap_mode="r")


@lru_cache(maxsize=64)
def official_target_cache(directory):
    return {name: np.load(Path(directory) / (name + ".npy"), mmap_mode="r")
            for name in ("ego", "trajectory", "trajectory_long")}


class FrontHistorySceneDataset(Dataset):
    def __init__(self, records):
        self.records = records

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        record = self.records[index]
        indices = record["observed_frame_cache_indices"]
        assert len(indices) == 4 and min(indices) >= 0
        frames = observed_frame_cache(str(PROJECT_ROOT / record["observed_frame_cache"]))
        targets = official_target_cache(record["cache_directory"])
        return {"images": np.array(frames[indices]), "token": record["token"],
                **{name: np.array(values[record["cache_row"]]) for name, values in targets.items()}}


def prepare_manifest(destination):
    stage1_root = PROJECT_ROOT / "outputs/lpwm_navsim_full_posttraining_v2"
    old = json.loads((stage1_root / "planning_manifest.json").read_text())
    official = json.loads((PROJECT_ROOT / "outputs/lpwm_drivor_joint_v1/scene_cache/manifest.json").read_text())
    by_token = {record["token"]: record for record in official["records"]}
    protocol = json.loads((PROJECT_ROOT / "outputs/lpwm_partial_planning_v1/evaluation_protocol.json").read_text())
    selected_dev = set(protocol["planning_tokens"])
    records = []
    for old_index, old_record in enumerate(old["records"]):
        token = old_record["current_frame_token"]
        if old_record["split"] != "train" and token not in selected_dev:
            continue
        indices = old_record["frame_cache_indices"][:4]
        assert len(indices) == 4 and min(indices) >= 0
        record = dict(by_token[token])
        record.update({"split": old_record["split"], "stage1_planning_index": old_index,
                       "observed_frame_cache_indices": indices,
                       "observed_frame_cache": "outputs/lpwm_navsim_full_posttraining_v2/rgb_frames.npy"})
        records.append(record)
    train = [row for row in records if row["split"] == "train"]
    dev = [row for row in records if row["split"] == "development"]
    assert len(train) == 75297 and len(dev) == 1024
    assert not {row["recording_group"] for row in train} & {row["recording_group"] for row in dev}
    assert set(row["token"] for row in dev) == selected_dev
    # Keep the previously registered panel's exact order, regardless of manifest order.
    dev_by_token = {row["token"]: row for row in dev}
    ordered_records = train + [dev_by_token[token] for token in protocol["planning_tokens"]]
    content = {"complete": True, "records": ordered_records,
               "counts": {"train": len(train), "development": len(dev)},
               "camera_count": 1, "observed_frames": 4, "image_size": [128, 128],
               "future_images_are_model_inputs": False,
               "dev_panel_source": "outputs/lpwm_partial_planning_v1/evaluation_protocol.json",
               "original_targets": "official DrivoR cache, token matched; original front RGB mmap"}
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        assert json.loads(destination.read_text()) == content
    else:
        destination.write_text(json.dumps(content, indent=2) + "\n")
    return content
