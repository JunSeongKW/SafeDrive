"""Read-only instrumentation of the production ego-query selector score graph."""

import contextlib
import math

import torch
import torch.nn.functional as F

from .models.drive_jepa_selective_patch_future import PatchSelection


@contextlib.contextmanager
def capture_ego_query_scores(selector):
    """Expose actual pre-top-K logits without changing weights or forward math."""
    original_forward = selector.forward
    captured = {}

    def instrumented(current_region_latents, current_ego_status, region_coordinates):
        keys = selector.key_normalization(
            selector.patch_keys(current_region_latents)
            + selector.coordinate_keys(region_coordinates)[None]
        )
        query = selector.query_normalization(
            selector.intent_query(current_ego_status)
            + selector.context_query(current_region_latents.mean(1))
        )
        scores = torch.einsum("bnd,bd->bn", keys, query) / math.sqrt(keys.shape[-1])
        captured["scores"] = scores
        remaining = torch.ones_like(scores, dtype=torch.bool)
        hard_rows, soft_rows, region_ids = [], [], []
        for _ in range(selector.patch_budget):
            masked = scores.masked_fill(~remaining, -torch.inf)
            soft = (masked / selector.temperature).softmax(-1)
            indices = masked.argmax(-1)
            hard = F.one_hot(indices, scores.shape[-1]).to(scores.dtype)
            hard_rows.append(hard)
            soft_rows.append(soft)
            region_ids.append(indices)
            remaining = remaining & ~hard.bool()
        hard, soft = torch.stack(hard_rows, 1), torch.stack(soft_rows, 1)
        weights = hard + (soft - soft.detach())
        return PatchSelection(
            weights, hard, torch.stack(region_ids, 1), weights @ region_coordinates
        )

    selector.forward = instrumented
    try:
        yield captured
    finally:
        selector.forward = original_forward


def gradient_cosine_or_none(first, second):
    denominator = first.norm() * second.norm()
    if float(denominator) == 0:
        return None
    return float((first.flatten() @ second.flatten()) / denominator)
