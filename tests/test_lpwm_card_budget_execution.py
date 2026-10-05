"""Protect shared-card accounting and the user's batch8 preference."""
import json
from pathlib import Path
import sys

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
from lpwm_card_budget_execution import (
    CARD_LIMIT_BYTES, NON_ALLOCATOR_GROWTH_BYTES, ROUNDING_HEADROOM_BYTES,
    allocator_budget_bytes, choose_largest_safe_profile,
)
from lpwm_48gb_execution import verify_execution_configuration


def test_allowance_includes_other_users_and_our_non_allocator_usage():
    own_reserved = 2 * 1024**3
    card_used = 23 * 1024**3
    allowance = allocator_budget_bytes(card_used, own_reserved)
    assert card_used - own_reserved + allowance + NON_ALLOCATOR_GROWTH_BYTES + ROUNDING_HEADROOM_BYTES == CARD_LIMIT_BYTES
    assert allocator_budget_bytes(card_used + 1024**3, own_reserved) == allowance - 1024**3
    with pytest.raises(RuntimeError):
        allocator_budget_bytes(CARD_LIMIT_BYTES, 0)


def test_batch8_preferred_even_if_smaller_batch_times_are_faster():
    smaller = {"passed": True, "microbatch_size_per_gpu": 4,
        "maximum_sampled_gpu_used_bytes": 36_000_000_000, "steady_update_seconds": 2}
    larger = {"passed": True, "microbatch_size_per_gpu": 8,
        "maximum_sampled_gpu_used_bytes": 47_900_000_000, "steady_update_seconds": 3}
    assert choose_largest_safe_profile([smaller, larger]) is larger
    assert choose_largest_safe_profile([smaller, {**larger, "maximum_sampled_gpu_used_bytes":48_000_000_001}]) is smaller
    with pytest.raises(RuntimeError):
        choose_largest_safe_profile([{**larger, "passed": False}])


@pytest.mark.parametrize("configuration_directory", ["card_budget_batch8_v2", "card_budget_expandable_v3", "card_budget_measured_v4"])
def test_all_future_methods_include_batch8_without_scientific_changes(configuration_directory):
    queue = json.loads((PROJECT_ROOT / "configs/lpwm_planning" / configuration_directory / "queue.json").read_text())
    for method in queue["methods"]:
        if "reuse_selection" in method:
            continue
        configurations = [json.loads((PROJECT_ROOT / path).read_text()) for path in method["profiles"]]
        assert configurations[0]["microbatch_size_per_gpu"] == 8
        for configuration in configurations:
            verify_execution_configuration(configuration, PROJECT_ROOT)
            assert configuration["maximum_reserved_gib"] > 23.2
            assert configuration["minimum_free_gib"] == 0


def test_measured_margin_remains_within_total_card_cap():
    from lpwm_measured_card_budget_execution import allocator_budget_bytes as measured_budget
    before_model = 21_605_908_480
    allowance = measured_budget(before_model, 0)
    # The preceding two real profiles measured about122MiB additional non-allocator usage.
    assert before_model + allowance + 122 * 1024**2 < CARD_LIMIT_BYTES
    assert before_model + allowance + (192 + 64) * 1024**2 == CARD_LIMIT_BYTES
