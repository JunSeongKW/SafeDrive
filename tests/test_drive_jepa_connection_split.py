"""Protect reserved recording groups before any connection diagnostic update."""

import importlib.util
import json
from pathlib import Path


def selection_function():
    root = Path(__file__).resolve().parents[1]
    specification = importlib.util.spec_from_file_location(
        "drive_connection_split",
        root / "scripts/validate_drive_jepa_selective_future_connection.py",
    )
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module.eligible_train_segments


def test_official_navtrain_does_not_override_project_split():
    choose = selection_function()
    assert choose(
        ["record_b_000_012", "record_a_000_012", "record_c_000_012", "unknown_000_012"],
        {"record_a": "held_out", "record_b": "train", "record_c": "development"},
    ) == ["record_b_000_012"]


def test_previously_exposed_recordings_are_not_admitted():
    root = Path(__file__).resolve().parents[1]
    manifest = json.loads(
        (
            root / "results/pilot_foundation_decision/navtrain_three_way_manifest.json"
        ).read_text()
    )
    assignments = manifest["assignments"]["split_by_recording"]
    assert (
        selection_function()(
            [
                "2021.05.12.19.36.12_veh-35_00215_00405",
                "2021.05.12.22.00.38_veh-35_00005_00118",
            ],
            assignments,
        )
        == []
    )
