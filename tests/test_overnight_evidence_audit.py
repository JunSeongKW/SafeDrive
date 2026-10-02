"""Evidence aggregation is scene-weighted, with explicit duplicate rejection."""

import importlib.util
from pathlib import Path

import pytest

SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts/audit_drive_jepa_overnight_evidence.py"
)
SPECIFICATION = importlib.util.spec_from_file_location(
    "overnight_evidence_audit", SCRIPT
)
MODULE = importlib.util.module_from_spec(SPECIFICATION)
SPECIFICATION.loader.exec_module(MODULE)


def test_scene_macro_is_not_window_mean():
    rows = [
        {"token": "a", "recording": "log", "scene_token": "first", "xy_ade_m": 1.0},
        {"token": "b", "recording": "log", "scene_token": "first", "xy_ade_m": 3.0},
        {"token": "c", "recording": "log", "scene_token": "second", "xy_ade_m": 8.0},
    ]
    assert MODULE.scene_macro_ade(rows) == 5.0


def test_duplicate_windows_are_not_silently_weighted_twice():
    row = {"token": "a", "recording": "log", "scene_token": "scene", "xy_ade_m": 1.0}
    with pytest.raises(ValueError, match="Duplicate"):
        MODULE.scene_macro_ade([row, row])


def test_nonfinite_results_cannot_pass_audit():
    row = {
        "token": "a",
        "recording": "log",
        "scene_token": "scene",
        "xy_ade_m": float("nan"),
    }
    with pytest.raises(ValueError, match="Invalid"):
        MODULE.scene_macro_ade([row])


def test_patch_spread_preserves_distinct_selection_budget():
    with pytest.raises(ValueError, match="distinct"):
        MODULE.summarize_patch_spread([{"selected_patch_ids": [1, 1, 2, 3]}])
    summary = MODULE.summarize_patch_spread([{"selected_patch_ids": [0, 31, 480, 511]}])
    assert summary["unique_patch_ids"] == 4
    assert 0 < summary["mean_pairwise_grid_distance_over_diagonal"] <= 1
