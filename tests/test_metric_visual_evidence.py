import copy

import pytest
import torch

from planning_aware_future_prediction.models.metric_visual_evidence import (
    MetricVisualScorerCorrection, compute_metric_evidence_losses)


def make_inputs():
    torch.manual_seed(7)
    return {"visual_tokens": torch.randn(2, 9, 12),
        "valid_token_mask": torch.tensor([[True]*6+[False]*3, [True]*9]),
        "positions": torch.randn(9, 4), "ego_status": torch.randn(2, 11),
        "trajectories": torch.randn(2, 4, 8, 3),
        "scored_features": torch.randn(2, 4, 256),
        "component_logits": torch.randn(2, 4, 6)}


def make_model(condition="metric_specific"):
    return MetricVisualScorerCorrection(12, 8, 16, condition=condition)


def test_initialization_off_and_other_metrics_are_exact():
    inputs = make_inputs()
    model = make_model()
    predicted, _ = model(**inputs)
    assert torch.equal(predicted, inputs["component_logits"])
    torch.nn.init.normal_(model.correction_head.weight, std=.1)
    predicted, _ = model(**inputs)
    assert torch.equal(predicted[..., 2:], inputs["component_logits"][..., 2:])
    off, _ = model(**inputs, intervention="correction_off")
    assert torch.equal(off, inputs["component_logits"])


def test_shared_and_specific_have_equal_parameters_and_shared_attention():
    inputs = make_inputs()
    specific = make_model()
    shared = make_model("shared")
    shared.load_state_dict(copy.deepcopy(specific.state_dict()))
    assert sum(parameter.numel() for parameter in shared.parameters()) == sum(parameter.numel() for parameter in specific.parameters())
    _, weights = shared(**inputs, capture_attention=True)
    assert torch.equal(weights[:, :, 0], weights[:, :, 1])
    _, specific_weights = specific(**inputs, capture_attention=True)
    assert (specific_weights[:, :, 0]-specific_weights[:, :, 1]).abs().max() > 1e-6


def test_padding_excluded_even_with_nan_and_large_values():
    inputs = make_inputs()
    model = make_model()
    torch.nn.init.normal_(model.correction_head.weight, std=.1)
    original, weights = model(**inputs, capture_attention=True)
    assert weights[0, :, :, 6:].count_nonzero() == 0
    assert torch.allclose(weights.sum(-1), torch.ones_like(weights.sum(-1)))
    inputs["visual_tokens"][0, 6:] = float("nan")
    modified, _ = model(**inputs)
    assert torch.isfinite(modified).all()
    assert torch.equal(original, modified)


def test_real_loss_gradient_after_first_head_update():
    inputs = make_inputs()
    model = make_model()
    targets = torch.rand(2, 4, 7)
    pairs = torch.tensor([[[0, 1], [2, 3]]]*2)
    score_weights = {"noc": 1., "dac": 1., "ddc": 0., "ep": 5., "ttc": 5., "comfort": 2.}
    optimizer = torch.optim.AdamW(model.parameters(), lr=.01)
    for update in range(2):
        optimizer.zero_grad(set_to_none=True)
        output, _ = model(**inputs)
        losses = compute_metric_evidence_losses(output, targets, score_weights, pairs)
        sum(losses).backward()
        assert all(torch.isfinite(parameter.grad).all() for parameter in model.parameters() if parameter.grad is not None)
        assert model.correction_head.weight.grad.norm() > 0
        if update == 1:
            for module in (model.candidate_query, model.key_projection, model.value_projection, model.residual_adapter):
                assert sum(float(parameter.grad.square().sum()) for parameter in module.parameters()) > 0
            assert model.metric_embedding.grad.norm() > 0
        optimizer.step()
    assert all(value.grad is None for value in inputs.values() if torch.is_tensor(value))


def test_visual_candidate_and_metric_interventions_change_actual_outputs():
    inputs = make_inputs()
    model = make_model()
    torch.nn.init.normal_(model.correction_head.weight, std=.2)
    original, attention = model(**inputs, capture_attention=True)
    zero_visual, _ = model(**{**inputs, "visual_tokens": torch.zeros_like(inputs["visual_tokens"])})
    assert (original-zero_visual).abs().max() > 1e-6
    swapped, _ = model(**inputs, intervention="swap_metric_attention")
    assert (original-swapped).abs().max() > 1e-7
    assert (attention[:, 0]-attention[:, 1]).abs().max() > 1e-6


def test_no_valid_tokens_rejected():
    inputs = make_inputs()
    inputs["valid_token_mask"][:] = False
    with pytest.raises(ValueError, match="valid"):
        make_model()(**inputs)
