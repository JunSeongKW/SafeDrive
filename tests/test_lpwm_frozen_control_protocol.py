"""Negative tests for automatic launch prerequisites and the fixed contrast."""
import json
from pathlib import Path
import sys

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
from lpwm_frozen_control_protocol import verify_control_configuration
from queue_lpwm_frozen_control import predecessor_ready


@pytest.mark.parametrize("mutation", ["planner_lr", "world_lr", "encoder_film", "batch"])
def test_scientific_or_freeze_drift_is_rejected(tmp_path, mutation):
    configuration = json.loads((PROJECT_ROOT / "configs/lpwm_planning/frozen_control_v1/batch8.json").read_text())
    if mutation == "planner_lr":
        configuration["planner_learning_rate"] *= 2
    elif mutation == "world_lr":
        configuration["lpwm_learning_rate"] = 1e-5
    elif mutation == "encoder_film":
        configuration["frozen_control"]["freeze_encoder_command_film"] = False
    else:
        configuration["microbatch_size_per_gpu"] = 4
    path = tmp_path / "invalid_control_configuration.json"
    path.write_text(json.dumps(configuration))
    with pytest.raises(AssertionError):
        verify_control_configuration(path)


def completed_reports():
    return {method: {"engineering_only": False, "checks": {"complete_registered_epoch": True}}
        for method in ("partial_output_layers", "attention_lora", "residual_adapter", "full_low_learning_rate")}


def test_vanished_process_or_marker_alone_does_not_authorize_launch(tmp_path):
    assert not predecessor_ready(tmp_path)
    (tmp_path / "queue_completion.json").write_text('{"complete": true}')
    with pytest.raises(FileNotFoundError):
        predecessor_ready(tmp_path)


def test_all_four_completed_validations_required(tmp_path):
    (tmp_path / "queue_completion.json").write_text('{"complete": true}')
    reports = completed_reports()
    reports["full_low_learning_rate"]["checks"]["complete_registered_epoch"] = False
    path = tmp_path / "four_method_summary.json"
    path.write_text(json.dumps({"complete": True, "reports": reports}))
    with pytest.raises(AssertionError):
        predecessor_ready(tmp_path)
    reports = completed_reports()
    path.write_text(json.dumps({"complete": True, "reports": reports}))
    assert predecessor_ready(tmp_path)
    (tmp_path / "queue_failed.json").write_text('{"error": "evaluation failure"}')
    with pytest.raises(RuntimeError, match="Predecessor failed"):
        predecessor_ready(tmp_path)


def test_scientific_performance_failure_does_not_selectively_skip_control(tmp_path):
    (tmp_path / "queue_completion.json").write_text('{"complete": true}')
    reports = completed_reports()
    reports["partial_output_layers"]["checks"]["forecast_lpips"] = False
    (tmp_path / "four_method_summary.json").write_text(json.dumps({"complete": True, "reports": reports}))
    assert predecessor_ready(tmp_path)
