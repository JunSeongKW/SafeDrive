import pytest
import torch

from planning_aware_future_prediction.patch_selection_controls import control_patch_ids


def test_random_control_is_per_window_stable_and_batch_order_independent():
    original = control_patch_ids(["scene-one", "scene-two"], "seeded_random", 29)
    reversed_order = control_patch_ids(["scene-two", "scene-one"], "seeded_random", 29)
    assert torch.equal(original, reversed_order.flip(0))
    assert all(row.unique().numel() == 4 for row in original)
    assert original.min() >= 0 and original.max() < 512
    assert not torch.equal(
        original,
        control_patch_ids(["scene-one", "scene-two"], "seeded_random", 29, update=1),
    )


def test_fixed_and_learned_controls():
    assert control_patch_ids(["scene-one"], "learned", 29) is None
    assert control_patch_ids(["scene-one"], "fixed_lattice", 29).tolist() == [
        [136, 151, 360, 375]
    ]
    with pytest.raises(ValueError):
        control_patch_ids(["scene-one"], "unregistered", 29)
