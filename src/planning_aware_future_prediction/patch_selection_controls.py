"""Outcome-independent camera-patch controls with stable per-window randomness."""

import hashlib

import torch


def control_patch_ids(window_tokens, selection_mode, seed, update=None):
    if selection_mode == "learned":
        return None
    if selection_mode == "fixed_lattice":
        return torch.tensor([136, 151, 360, 375], dtype=torch.long).expand(
            len(window_tokens), -1
        )
    if selection_mode != "seeded_random":
        raise ValueError("Unknown patch selection control")
    selected = []
    for token in window_tokens:
        key = (
            f"patch-control|{seed}|{token}|{'evaluation' if update is None else update}"
        )
        derived_seed = int.from_bytes(
            hashlib.sha256(key.encode()).digest()[:8], "big"
        ) % (2**63 - 1)
        generator = torch.Generator().manual_seed(derived_seed)
        selected.append(torch.randperm(512, generator=generator)[:4])
    return torch.stack(selected)
