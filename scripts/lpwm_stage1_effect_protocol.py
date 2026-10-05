"""Public frozen LPWM control, matched to the completed Stage1 frozen planner."""
import json
from pathlib import Path

import numpy as np

from evaluate_lpwm_full_planning import digest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ALLOWED_CONTRAST_FIELDS = {
    "research_question", "output_directory", "shared_results_directory", "comparison",
    "stage1_gate_required", "stage1_admission_amendment", "stage1_effect", "frozen_control",
}


def verify_stage1_effect_configuration(configuration_path):
    specification = json.loads(Path(configuration_path).read_text())
    contrast = specification["stage1_effect"]
    reference_path = PROJECT_ROOT / contrast["reference_configuration"]
    assert digest(reference_path) == contrast["reference_configuration_sha256"]
    reference = json.loads(reference_path.read_text())
    unchanged = lambda values: {key: value for key, value in values.items() if key not in ALLOWED_CONTRAST_FIELDS}
    assert unchanged(specification) == unchanged(reference), "Unexpected training/evaluation change from frozen Stage1 control"
    assert contrast["representation_condition"] == "public_pretrained"
    expected_freeze = {**reference["frozen_control"], "initialize_from_stage1_only": False}
    assert specification["frozen_control"] == expected_freeze
    assert specification["adaptation_method"] == "frozen_lpwm_planner_control"
    assert not specification["stage1_gate_required"] and specification["stage1_admission_amendment"] is None
    assert specification["lpwm_learning_rate"] == 0
    assert specification["conditions"] == ["metric_plus_world"]
    assert contrast["freeze_all_lpwm_weights_and_buffers"] and contrast["freeze_encoder_command_film"]
    assert contrast["fresh_matched_planner_initialization"] and not contrast["apply_navsim_adaptation_gate"]
    batch_size = specification["microbatch_size_per_gpu"]
    assert batch_size == 8 and specification["gradient_accumulation"] == 1
    assert specification["world_auxiliary_clips_per_gpu_microbatch"] == 4
    assert specification["world_size"] == 2
    assert specification["epochs"] == 1 and specification["seed"] == 47
    assert specification["resource_limits"]["maximum_gpu_used_bytes"] == 48_000_000_000
    checkpoint_path = PROJECT_ROOT / contrast["initial_lpwm_checkpoint"]
    assert digest(checkpoint_path) == contrast["initial_lpwm_checkpoint_sha256"]
    assert digest(PROJECT_ROOT / contrast["official_hyperparameters"]) == contrast["official_hyperparameters_sha256"]
    assert contrast["initial_lpwm_checkpoint"] == contrast["public_checkpoint"]
    assert contrast["initial_lpwm_checkpoint_sha256"] == contrast["public_checkpoint_sha256"]
    assert contrast["public_checkpoint_sha256"] != contrast["adapted_checkpoint_sha256"]
    return specification


def load_stage1_effect_inputs(configuration_path, condition):
    """Keep cached inputs fixed; select only the registered representation weights."""
    specification = verify_stage1_effect_configuration(configuration_path)
    assert condition in specification["conditions"]
    cache_configuration = json.loads((PROJECT_ROOT / specification["stage1_config"]).read_text())
    cache_root = PROJECT_ROOT / cache_configuration["output_directory"]
    contrast = specification["stage1_effect"]
    checkpoint_path = PROJECT_ROOT / contrast["initial_lpwm_checkpoint"]
    teacher = json.loads((PROJECT_ROOT / specification["teacher_directory"] / "completion.json").read_text())
    assert teacher["teacher_gate_passed"]
    manifest = json.loads((cache_root / "planning_manifest.json").read_text())
    records = manifest["records"]
    with np.load(cache_root / "planning_targets.npz") as stored_targets:
        target_arrays = {name: stored_targets[name] for name in stored_targets.files}
    frame_cache = np.load(cache_root / "rgb_frames.npy", mmap_mode="r")
    world_records = json.loads((cache_root / "manifest.json").read_text())["records"]
    index_by_token = {record["current_frame_token"]: index for index, record in enumerate(records)}
    world_indices = [index_by_token[record["current_frame_token"]] for record in world_records if record["split"] == "train"]
    assert sum(record["split"] == "train" for record in records) == 75297
    return specification, cache_configuration, cache_root, checkpoint_path, manifest, target_arrays, frame_cache, world_indices
