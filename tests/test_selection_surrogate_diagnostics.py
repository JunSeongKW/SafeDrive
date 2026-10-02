"""Algorithm checks only; actual-model surrogate usefulness requires execution."""

import numpy as np

from planning_aware_future_prediction.selection_surrogate_diagnostics import (
    correlation_or_none,
    generate_single_slot_substitutions,
)


def test_substitutions_are_deterministic_unique_and_single_slot():
    chosen = [1, 3, 5, 7]
    variants, replacements = generate_single_slot_substitutions(chosen, 16, 3, 29)
    assert (variants, replacements) == generate_single_slot_substitutions(
        chosen, 16, 3, 29
    )
    assert variants[0] == chosen and len(variants) == 13
    for alternative, (slot, previous, replacement) in zip(variants[1:], replacements):
        assert len(set(alternative)) == len(chosen)
        assert sum(left != right for left, right in zip(chosen, alternative)) == 1
        assert previous == chosen[slot] and replacement == alternative[slot]
        assert replacement not in chosen


def test_one_hot_displacement_gradient_is_exact_for_linear_loss():
    gradients = np.arange(32).reshape(4, 8)
    chosen = [1, 3, 5, 7]
    variants, replacements = generate_single_slot_substitutions(chosen, 8, 2, 29)
    base_loss = sum(gradients[slot, patch] for slot, patch in enumerate(chosen))
    for alternative, (slot, previous, replacement) in zip(variants[1:], replacements):
        actual = (
            sum(gradients[index, patch] for index, patch in enumerate(alternative))
            - base_loss
        )
        assert gradients[slot, replacement] - gradients[slot, previous] == actual
    assert correlation_or_none([1, 1], [2, 3]) is None
