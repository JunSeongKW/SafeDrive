import importlib.util
from pathlib import Path

import pytest


def load_summary_module():
    path = (
        Path(__file__).resolve().parents[1]
        / "scripts/summarize_drive_jepa_architecture_followup.py"
    )
    specification = importlib.util.spec_from_file_location("architecture_summary", path)
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


def test_paired_recording_comparison_weights_scenes_not_windows():
    module = load_summary_module()
    proposed = [
        {"token": "a", "recording": "log1", "scene_token": "scene1", "xy_ade_m": 2},
        {"token": "b", "recording": "log1", "scene_token": "scene1", "xy_ade_m": 4},
        {"token": "c", "recording": "log2", "scene_token": "scene2", "xy_ade_m": 7},
    ]
    reference = [dict(row, xy_ade_m=1) for row in proposed]
    result = module.paired_recording_comparison(
        {29: proposed, 47: proposed}, {29: reference, 47: reference}
    )
    assert result["mean_difference_m"] == 4
    assert result["recording_differences_m"] == {"log1": 2, "log2": 6}


def test_missing_paired_window_is_rejected():
    module = load_summary_module()
    with pytest.raises(ValueError):
        module.paired_recording_comparison({29: [{"token": "a"}]}, {29: []})
