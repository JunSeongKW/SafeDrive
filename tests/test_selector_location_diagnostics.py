import torch

from planning_aware_future_prediction.models.drive_jepa_adaptive_future import (
    EgoQueryPatchSelector,
)
from planning_aware_future_prediction.selector_location_diagnostics import (
    capture_ego_query_scores,
)


def test_score_instrumentation_preserves_forward_and_parameter_gradients():
    torch.manual_seed(11)
    selector = EgoQueryPatchSelector(16, 16, 3)
    current, status, coordinates = (
        torch.randn(2, 12, 16),
        torch.randn(2, 8),
        torch.randn(12, 2),
    )
    original = selector(current, status, coordinates)
    probe = torch.randn_like(original.selection_weights)
    original_gradient = torch.autograd.grad(
        (original.selection_weights * probe).sum(), list(selector.parameters())
    )
    with capture_ego_query_scores(selector) as captured:
        instrumented = selector(current, status, coordinates)
        gradient = torch.autograd.grad(
            (instrumented.selection_weights * probe).sum(),
            [captured["scores"], *selector.parameters()],
        )
        assert gradient[0].norm() > 0
        assert torch.equal(original.selection_weights, instrumented.selection_weights)
        assert all(
            torch.equal(first, second)
            for first, second in zip(original_gradient, gradient[1:])
        )
    restored = selector(current, status, coordinates)
    assert torch.equal(original.selection_weights, restored.selection_weights)
