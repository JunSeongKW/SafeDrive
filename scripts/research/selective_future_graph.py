"""CPU graph diagnostic, not a NAVSIM agent or a trained visual JEPA model."""

from dataclasses import dataclass

import torch
from torch import Tensor, nn
import torch.nn.functional as F


@dataclass
class Selection:
    weights: Tensor  # [B, K, N], hard forward, optional ST backward
    hard_weights: Tensor
    indices: Tensor  # -1 for inactive slots
    valid: Tensor  # [B, K]


class SequentialSTTopK(nn.Module):
    """Unique hard selection with a conditional, biased soft-backward surrogate.

    Each slot softmaxes over currently remaining valid entities. Previously
    chosen HARD indices are excluded; no gradient through that exclusion.
    Ties use ascending stable entity ID, never incidental tensor position.
    """

    def __init__(self, k: int, temperature: float = 0.7):
        super().__init__()
        if k < 1 or not temperature > 0:
            raise ValueError("k and temperature must be positive")
        self.k, self.temperature = k, temperature

    def forward(self, scores: Tensor, valid: Tensor, entity_ids: Tensor) -> Selection:
        if scores.ndim != 2 or valid.shape != scores.shape or entity_ids.shape != scores.shape:
            raise ValueError("scores, valid and entity_ids must have shape [B, N]")
        if valid.dtype != torch.bool or entity_ids.dtype != torch.long:
            raise ValueError("valid must be bool; entity_ids must be int64")
        if not torch.isfinite(scores[valid]).all():
            raise ValueError("valid scores must be finite")
        for row_ids, row_valid in zip(entity_ids, valid):
            ids = row_ids[row_valid]
            if ids.unique().numel() != ids.numel():
                raise ValueError("valid entity IDs must be unique within each sample")
        batch, n = scores.shape
        if n == 0:
            empty = scores.new_zeros(batch, self.k, 0)
            return Selection(empty, empty, entity_ids.new_full((batch, self.k), -1),
                             valid.new_zeros(batch, self.k))

        order = torch.argsort(entity_ids, dim=-1, stable=True)
        ordered_scores = scores.gather(1, order)
        remaining = valid.gather(1, order)
        soft_rows, hard_rows, indices, slots = [], [], [], []
        for _ in range(self.k):
            active = remaining.any(dim=-1)
            masked = ordered_scores.masked_fill(~remaining, -torch.inf)
            # All-invalid samples must not softmax an all-minus-inf vector.
            safe = torch.where(active[:, None], masked, torch.zeros_like(masked))
            prob = (safe / self.temperature).softmax(dim=-1) * remaining
            hard_index = safe.argmax(dim=-1)
            hard = F.one_hot(hard_index, n).to(scores.dtype) * active[:, None]
            hard_rows.append(torch.zeros_like(hard).scatter(1, order, hard))
            soft_rows.append(torch.zeros_like(prob).scatter(1, order, prob))
            original_index = order.gather(1, hard_index[:, None]).squeeze(1)
            indices.append(torch.where(active, original_index, -1))
            slots.append(active)
            remaining = remaining & ~hard.bool()
        hard = torch.stack(hard_rows, dim=1)
        soft = torch.stack(soft_rows, dim=1)
        # No subset decision when the budget covers every valid entity. Our
        # planner is order invariant: rank gradients here would be spurious.
        soft = soft * (valid.sum(dim=-1) > self.k)[:, None, None]
        # Parentheses ensure exactly hard forward values, not 1+p-p roundoff.
        weights = hard + (soft - soft.detach())
        return Selection(weights, hard, torch.stack(indices, dim=1), torch.stack(slots, dim=1))


class ContextSelector(nn.Module):
    """Shared scoring by entity content and intent; no slot/index embeddings."""

    def __init__(self, entity_dim: int, context_dim: int, intent_dim: int, hidden: int = 32):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(entity_dim + context_dim + intent_dim, hidden),
                                 nn.GELU(), nn.Linear(hidden, 1))

    def forward(self, entities: Tensor, context: Tensor, intent: Tensor) -> Tensor:
        n = entities.shape[1]
        conditioning = torch.cat((context, intent), dim=-1)[:, None].expand(-1, n, -1)
        return self.net(torch.cat((entities, conditioning), dim=-1)).squeeze(-1)


@dataclass
class GraphOutput:
    selection: Selection
    future: Tensor
    trajectory: Tensor


class SelectiveFutureGraph(nn.Module):
    """Minimal differentiable fixture with externally supplied detached targets.

    C is a pooled context vector in this fixture, not an implemented BEV stack.
    Future targets are never arguments of forward(). No target encoder here.
    """

    def __init__(self, entity_dim=6, context_dim=4, intent_dim=3, latent_dim=5,
                 horizon=2, k=2, hidden=24):
        super().__init__()
        self.horizon, self.latent_dim = horizon, latent_dim
        self.selector = ContextSelector(entity_dim, context_dim, intent_dim, hidden)
        self.selection = SequentialSTTopK(k)
        self.predictor = nn.Sequential(
            nn.Linear(entity_dim + context_dim + intent_dim, hidden), nn.GELU(),
            nn.Linear(hidden, horizon * latent_dim))
        self.planner = nn.Sequential(
            nn.Linear(context_dim + intent_dim + horizon * latent_dim, hidden),
            nn.GELU(), nn.Linear(hidden, 3))

    def predict(self, queries: Tensor, context: Tensor, intent: Tensor, valid: Tensor) -> Tensor:
        conditioning = torch.cat((context, intent), dim=-1)[:, None].expand(-1, queries.shape[1], -1)
        values = self.predictor(torch.cat((queries, conditioning), dim=-1))
        values = values.reshape(queries.shape[0], queries.shape[1], self.horizon, self.latent_dim)
        return values * valid[..., None, None]

    def plan(self, context: Tensor, intent: Tensor, future: Tensor, valid: Tensor) -> Tensor:
        count = valid.sum(dim=-1).clamp_min(1).to(future.dtype)
        pooled = (future * valid[..., None, None]).sum(dim=1) / count[:, None, None]
        return self.planner(torch.cat((context, intent, pooled.flatten(1)), dim=-1))

    def forward(self, entities: Tensor, context: Tensor, intent: Tensor, valid: Tensor,
                entity_ids: Tensor, selection_mode="st", detach_future=False) -> GraphOutput:
        if selection_mode not in ("st", "hard", "detach"):
            raise ValueError("unknown selection mode")
        safe_entities = entities.masked_fill(~valid[..., None], 0)
        selection = self.selection(self.selector(safe_entities, context, intent), valid, entity_ids)
        if selection_mode == "hard":
            selection.weights = selection.hard_weights
        elif selection_mode == "detach":
            selection.weights = selection.weights.detach()
        queries = selection.weights @ safe_entities
        future = self.predict(queries, context, intent, selection.valid)
        planning_future = future.detach() if detach_future else future
        trajectory = self.plan(context, intent, planning_future, selection.valid)
        return GraphOutput(selection, future, trajectory)

    def losses(self, output: GraphOutput, entities: Tensor, context: Tensor, intent: Tensor,
               valid: Tensor, future_targets: Tensor, future_valid: Tensor, ego_target: Tensor):
        plan = F.mse_loss(output.trajectory, ego_target)
        # Separate predictor call: auxiliary cannot backprop directly to selector.
        safe_entities = entities.masked_fill(~valid[..., None], 0)
        queries = output.selection.hard_weights.detach() @ safe_entities
        auxiliary = self.predict(queries, context, intent, output.selection.valid)
        batch, slots = output.selection.indices.shape
        if entities.shape[1] == 0:
            target = torch.zeros_like(auxiliary)
            mask = future_valid.new_zeros(batch, slots, self.horizon)
        else:
            indices = output.selection.indices.clamp_min(0)
            target = future_targets.detach().gather(
                1, indices[..., None, None].expand(-1, -1, self.horizon, self.latent_dim))
            mask = future_valid.gather(1, indices[..., None].expand(-1, -1, self.horizon))
        mask = mask & output.selection.valid[..., None]
        # Invalid targets can contain NaN; mask BEFORE forming squared errors.
        target = torch.where(mask[..., None], target, torch.zeros_like(target))
        error = torch.where(mask[..., None], auxiliary - target, torch.zeros_like(auxiliary))
        denom = (mask.sum() * self.latent_dim).clamp_min(1)
        jepa = error.square().sum() / denom
        return {"plan": plan, "jepa": jepa}


def gradient_norm(loss: Tensor, module: nn.Module) -> float:
    gradients = torch.autograd.grad(loss, tuple(module.parameters()), allow_unused=True, retain_graph=True)
    if any(g is not None and not torch.isfinite(g).all() for g in gradients):
        raise AssertionError("non-finite gradient")
    return sum(g.square().sum().item() for g in gradients if g is not None) ** 0.5
