import pytest
import torch
from torch import nn

from planning_aware_future_prediction.models.future_gradient_routing import (
    freeze_parameter_gradients,
    project_auxiliary_gradient,
    uniform_xy_ade_with_heading,
)


def test_parameter_freeze_preserves_input_gradient_and_restores_auxiliary_training():
    predictor = nn.Sequential(nn.Linear(3, 4), nn.GELU(), nn.Linear(4, 2))
    observed = torch.randn(5, 3, requires_grad=True)
    original = predictor(observed)
    with freeze_parameter_gradients(predictor):
        routed = predictor(observed)
    assert torch.equal(original, routed)
    routed.square().sum().backward()
    assert observed.grad.norm() > 0
    assert all(parameter.grad is None for parameter in predictor.parameters())
    predictor(observed.detach()).square().sum().backward()
    assert all(parameter.grad is not None for parameter in predictor.parameters())


def test_parameter_flags_are_restored_on_error():
    predictor = nn.Linear(3, 2)
    predictor.bias.requires_grad_(False)
    with pytest.raises(RuntimeError), freeze_parameter_gradients(predictor):
        raise RuntimeError("intentional")
    assert predictor.weight.requires_grad
    assert not predictor.bias.requires_grad


def test_one_sided_projection_removes_only_conflicting_auxiliary_component():
    planning = [torch.tensor([1.0, 0.0]), torch.tensor([0.0])]
    auxiliary = [torch.tensor([-2.0, 3.0]), torch.tensor([4.0])]
    combined, report = project_auxiliary_gradient(planning, auxiliary, planning)
    assert report["conflicting"]
    assert torch.equal(combined[0], torch.tensor([1.0, 3.0]))
    assert torch.equal(combined[1], torch.tensor([4.0]))


def test_nonconflicting_and_zero_planning_gradients_remain_unchanged():
    for planning in [torch.tensor([1.0, 2.0]), torch.zeros(2)]:
        auxiliary = torch.tensor([3.0, 4.0])
        combined, report = project_auxiliary_gradient(
            [planning], [auxiliary], [planning]
        )
        assert not report["conflicting"]
        assert torch.equal(combined[0], planning + auxiliary)


def test_heading_wrap_and_zero_error_are_finite():
    prediction = torch.zeros(1, 8, 3, requires_grad=True)
    target = prediction.detach().clone()
    target[..., 2] = 2 * torch.pi
    loss = uniform_xy_ade_with_heading(prediction, target)
    assert loss < 1e-6
    loss.backward()
    assert torch.isfinite(prediction.grad).all()
