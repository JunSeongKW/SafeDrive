import copy
import inspect

import pytest
import torch

from planning_aware_future_prediction.models.direct_metric_visual_probe import (
    DirectMetricVisualProbe, replace_nc_dac_logits,
)
from planning_aware_future_prediction.models.metric_visual_evidence import compute_metric_evidence_losses


def make_inputs():
    torch.manual_seed(17)
    return {
        "visual_tokens": torch.randn(2, 13, 12),
        "valid_token_mask": torch.tensor([[True] * 9 + [False] * 4, [True] * 13]),
        "positions": torch.randn(2, 13, 4),
        "ego_status": torch.randn(2, 11),
        "trajectories": torch.randn(2, 4, 8, 3),
    }


def make_model(condition="metric_specific"):
    return DirectMetricVisualProbe(12, 8, 16, 5, condition)


def test_prediction_has_no_scorer_or_direct_condition_shortcut():
    assert "scored_features" not in inspect.signature(DirectMetricVisualProbe.forward).parameters
    assert "component_logits" not in inspect.signature(DirectMetricVisualProbe.forward).parameters
    inputs = make_inputs()
    model = make_model()
    # With all values identical, changing ego/path/position cannot carry those
    # conditions directly to the output head through the weighted sum.
    constant_visual = torch.randn(1, 1, 12).expand_as(inputs["visual_tokens"])
    original, _ = model(**{**inputs, "visual_tokens": constant_visual})
    changed, _ = model(**{**inputs, "visual_tokens": constant_visual,
                         "ego_status": inputs["ego_status"] * 9,
                         "trajectories": inputs["trajectories"] * -4,
                         "positions": inputs["positions"] * 3})
    assert torch.allclose(original, changed, atol=2e-6, rtol=0)
    assert torch.allclose(original, original[:, :1].expand_as(original), atol=2e-6, rtol=0)


def test_joint_query_metric_relative_odds_depend_on_candidate():
    inputs = make_inputs()
    model = make_model()
    _, attention = model(**inputs, capture_attention=True)
    relative_log_odds = attention[1, :, 0].log() - attention[1, :, 1].log()
    relative_log_odds = relative_log_odds - relative_log_odds[:, :1]
    assert (relative_log_odds - relative_log_odds[:1]).abs().max() > 1e-5


def test_shared_and_specific_match_parameters_and_initialization():
    specific = make_model()
    shared = make_model("shared")
    shared.load_state_dict(copy.deepcopy(specific.state_dict()))
    assert sum(value.numel() for value in specific.parameters()) == sum(value.numel() for value in shared.parameters())
    _, attention = shared(**make_inputs(), capture_attention=True)
    assert torch.equal(attention[:, :, 0], attention[:, :, 1])


def test_loss_gradients_reach_reader_on_first_update():
    inputs = make_inputs()
    inputs["visual_tokens"].requires_grad_(True)
    model = make_model()
    predicted, attention = model(**inputs, capture_attention=True)
    original_logits = torch.randn(2, 4, 6)
    combined = replace_nc_dac_logits(original_logits, predicted)
    targets = torch.rand(2, 4, 7)
    pairs = torch.tensor([[[0, 1], [2, 3]]] * 2)
    weights = {"noc": 1., "dac": 1., "ddc": 0., "ep": 5., "ttc": 5., "comfort": 2.}
    sum(compute_metric_evidence_losses(combined, targets, weights, pairs)).backward()
    assert all(parameter.grad is not None and torch.isfinite(parameter.grad).all() for parameter in model.parameters())
    for module in (model.candidate_metric_query, model.key_projection, model.value_projection, model.metric_prediction_head):
        assert sum(float(parameter.grad.square().sum()) for parameter in module.parameters()) > 0
    assert model.metric_embedding.grad.norm() > 0
    assert inputs["visual_tokens"].grad[0, 9:].count_nonzero() == 0
    assert original_logits.grad is None


def test_padding_nan_excluded_and_empty_scene_rejected():
    inputs = make_inputs()
    model = make_model()
    original, _ = model(**inputs)
    inputs["visual_tokens"][0, 9:] = float("nan")
    inputs["positions"][0, 9:] = float("nan")
    modified, attention = model(**inputs, capture_attention=True)
    assert torch.equal(original, modified)
    assert attention[0, :, :, 9:].count_nonzero() == 0
    assert torch.allclose(attention.sum(-1), torch.ones_like(attention.sum(-1)))
    inputs["valid_token_mask"][:] = False
    with pytest.raises(ValueError, match="valid"):
        model(**inputs)


def test_native_other_metrics_preserved_and_off_exact():
    inputs = make_inputs()
    predicted, _ = make_model()(**inputs)
    original_logits = torch.randn(2, 4, 6)
    combined = replace_nc_dac_logits(original_logits, predicted)
    assert torch.equal(combined[..., 2:], original_logits[..., 2:])
    assert torch.equal(replace_nc_dac_logits(original_logits, predicted, probe_enabled=False), original_logits)
