"""Reject confounded or accidentally adapted public-pretraining controls."""
import copy
import json
from pathlib import Path
import sys

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
from lpwm_stage1_effect_protocol import verify_stage1_effect_configuration


@pytest.mark.parametrize("mutation", ["physical_batch", "accumulation", "planner_lr", "world_lr", "encoder_film", "adapted_initialization", "adaptation_gate"])
def test_only_the_public_initial_representation_can_change(tmp_path, mutation):
    specification = json.loads((PROJECT_ROOT / "configs/lpwm_planning/stage1_effect_v1/public_pretrained_batch8.json").read_text())
    specification = copy.deepcopy(specification)
    if mutation == "physical_batch":
        specification["microbatch_size_per_gpu"] = 16
    elif mutation == "accumulation":
        specification["gradient_accumulation"] = 2
    elif mutation == "planner_lr":
        specification["planner_learning_rate"] *= 2
    elif mutation == "world_lr":
        specification["lpwm_learning_rate"] = 1e-5
    elif mutation == "encoder_film":
        specification["frozen_control"]["freeze_encoder_command_film"] = False
    elif mutation == "adapted_initialization":
        specification["stage1_effect"]["initial_lpwm_checkpoint"] = specification["stage1_effect"]["adapted_checkpoint"]
        specification["stage1_effect"]["initial_lpwm_checkpoint_sha256"] = specification["stage1_effect"]["adapted_checkpoint_sha256"]
    else:
        specification["stage1_effect"]["apply_navsim_adaptation_gate"] = True
    configuration = tmp_path / "invalid_configuration.json"
    configuration.write_text(json.dumps(specification))
    with pytest.raises(AssertionError):
        verify_stage1_effect_configuration(configuration)
