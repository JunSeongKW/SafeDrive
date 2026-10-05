"""Validate the registered contrast against the completed Adapter experiment."""
import json
from pathlib import Path

from evaluate_lpwm_full_planning import digest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONTROL_FIELDS = {
    "research_question", "output_directory", "shared_results_directory", "adaptation_method",
    "lpwm_learning_rate", "precision", "encoder_conditioning", "world_loss", "comparison",
    "partial_finetuning", "adapter_finetuning", "execution_amendment", "frozen_control",
    "microbatch_size_per_gpu", "gradient_accumulation", "world_auxiliary_clips_per_gpu_microbatch",
}


def verify_control_configuration(configuration_path):
    specification = json.loads(Path(configuration_path).read_text())
    control = specification["frozen_control"]
    reference_path = PROJECT_ROOT / control["reference_configuration"]
    assert digest(reference_path) == control["reference_configuration_sha256"]
    reference = json.loads(reference_path.read_text())
    unchanged = lambda values: {key: value for key, value in values.items() if key not in CONTROL_FIELDS}
    assert unchanged(specification) == unchanged(reference), "Unregistered scientific setting differs from Adapter"
    assert specification["adaptation_method"] == "frozen_lpwm_planner_control"
    assert specification["lpwm_learning_rate"] == 0
    assert specification["microbatch_size_per_gpu"] * specification["gradient_accumulation"] * specification["world_size"] == 16
    assert specification["world_auxiliary_clips_per_gpu_microbatch"] * specification["gradient_accumulation"] * specification["world_size"] == 8
    assert specification["epochs"] == 1 and specification["seed"] == 47
    assert specification["conditions"] == ["metric_plus_world"]
    assert specification["resource_limits"]["maximum_gpu_used_bytes"] == 48_000_000_000
    assert control["freeze_encoder_command_film"] and control["world_evaluation_mode"]
    assert control["world_objective_stop_gradient"] and control["initialize_from_stage1_only"]
    return specification
