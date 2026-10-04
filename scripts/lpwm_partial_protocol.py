"""Fixed development panels and immutable inputs for the quick Stage2 study."""
from collections import defaultdict
import hashlib
import json
from pathlib import Path

from evaluate_lpwm_full_planning import digest, write_json

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def recording_balanced_indices(records, count, eligible=None):
    groups = defaultdict(list)
    for index, record in enumerate(records):
        if record["split"] == "development" and (eligible is None or record["current_frame_token"] in eligible):
            groups[record["recording_group"]].append(index)
    for indices in groups.values():
        indices.sort(key=lambda index: hashlib.sha256(
            ("lpwm-partial-fixed-panel:" + records[index]["current_frame_token"]).encode()).hexdigest())
    selected = []
    depth = 0
    while len(selected) < count:
        added = False
        for recording in sorted(groups):
            if depth < len(groups[recording]):
                selected.append(groups[recording][depth])
                added = True
                if len(selected) == count:
                    break
        if not added:
            raise ValueError("Requested panel exceeds the eligible development split")
        depth += 1
    return selected


def prepare_protocol(config_path):
    specification = json.loads(config_path.read_text())
    stage1 = json.loads((PROJECT_ROOT / specification["stage1_config"]).read_text())
    stage1_root = PROJECT_ROOT / stage1["output_directory"]
    manifest_path = stage1_root / "planning_manifest.json"
    records = json.loads(manifest_path.read_text())["records"]
    world_records = json.loads((stage1_root / "manifest.json").read_text())["records"]
    world_tokens = {row["current_frame_token"] for row in world_records if row["split"] == "development"}
    planning = recording_balanced_indices(records, specification["trend_evaluation"]["planning_scenes"])
    world = recording_balanced_indices(records, specification["trend_evaluation"]["world_clips"], world_tokens)
    report = {"configuration_sha256": digest(config_path), "planning_manifest_sha256": digest(manifest_path),
        "planning_indices": planning, "world_indices": world,
        "planning_tokens": [records[index]["current_frame_token"] for index in planning],
        "world_tokens": [records[index]["current_frame_token"] for index in world],
        "planning_recordings": len({records[index]["recording_group"] for index in planning}),
        "world_recordings": len({records[index]["recording_group"] for index in world}),
        "selection_uses_model_predictions": False,
        "scope": "Previously exposed development panels for fast trends, not full development or independent test"}
    destination = PROJECT_ROOT / specification["output_directory"] / "evaluation_protocol.json"
    if destination.exists():
        assert json.loads(destination.read_text()) == report
    else:
        write_json(destination, report)
    return report
