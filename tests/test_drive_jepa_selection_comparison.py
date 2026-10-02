"""Small comparison admission: no model outcomes or future validity in sampling."""

import importlib.util
import sys
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
SPEC = importlib.util.spec_from_file_location(
    "selection_comparison", SCRIPTS / "run_drive_jepa_selection_comparison.py"
)
RUNNER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(RUNNER)


def test_recording_split_and_exposure_exclusions():
    assignments = {
        "training": "train",
        "dev": "development",
        "held": "held_out",
        "exposed": "train",
    }
    manifest = {
        "assignments": {"split_by_recording": assignments},
        "records": [
            {
                "recording_group": group,
                "split": split,
                "current_frame_token": f"{group}_{index}",
            }
            for group, split in assignments.items()
            for index in range(3)
        ],
    }
    config = {
        "excluded_recordings": ["exposed"],
        "recording_counts": {"train": 1, "development": 1},
        "windows_per_recording": 2,
    }
    result = RUNNER.select_manifest_records(manifest, config)
    assert len(result) == 4
    assert {row["recording_group"] for row in result} == {"training", "dev"}
    manifest["records"].reverse()
    assert RUNNER.select_manifest_records(manifest, config) == result
    manifest["records"][0]["split"] = "held_out"
    with pytest.raises(ValueError, match="disagrees"):
        RUNNER.select_manifest_records(manifest, config)
