"""Deterministic discrete substitutions for read-only routing diagnostics."""

import numpy as np


def generate_single_slot_substitutions(
    selected_ids, candidate_count, alternatives_per_slot, seed
):
    generator = np.random.default_rng(seed)
    available = sorted(set(range(candidate_count)) - set(selected_ids))
    variants, replacements = [list(selected_ids)], []
    for slot, previous_id in enumerate(selected_ids):
        for new_id in generator.choice(available, alternatives_per_slot, replace=False):
            variant = list(selected_ids)
            variant[slot] = int(new_id)
            variants.append(variant)
            replacements.append((slot, previous_id, int(new_id)))
    return variants, replacements


def correlation_or_none(first, second):
    if len(first) < 2 or np.std(first) < 1e-12 or np.std(second) < 1e-12:
        return None
    return float(np.corrcoef(first, second)[0, 1])
