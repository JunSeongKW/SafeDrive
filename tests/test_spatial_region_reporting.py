"""Visualization coordinates must depict actual model region membership."""

import importlib.util
import sys
from pathlib import Path

SCRIPT_DIRECTORY = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPT_DIRECTORY))
SPECIFICATION = importlib.util.spec_from_file_location(
    "region_reporting", SCRIPT_DIRECTORY / "report_drive_jepa_spatial_regions.py"
)
REPORTING = importlib.util.module_from_spec(SPECIFICATION)
SPECIFICATION.loader.exec_module(REPORTING)


def test_region_render_membership():
    assert REPORTING.covered_native_cells([0], 2) == [0, 1, 32, 33]
    assert REPORTING.covered_native_cells([127], 2) == [478, 479, 510, 511]
    assert REPORTING.covered_native_cells([511], 1) == [511]


def test_selection_replacement_does_not_depend_on_rank():
    before = [{"token": "one", "selected_region_ids": [0, 1]}]
    after = [{"token": "one", "selected_region_ids": [1, 0]}]
    report = REPORTING.summarize_selection(after, before, 2)
    assert report["mean_selected_set_replacement_fraction"] == 0
    assert report["covered_fraction_of_display_grid"] == 8 / 512
