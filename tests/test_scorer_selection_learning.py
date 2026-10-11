"""Functional contracts for cached six-subscore correction and acquisition."""
import torch

from planning_aware_future_prediction.models.scorer_selection_learning import (
    CachedScorerCorrection, SelectionErrorFamilySampler, build_sampler_condition,
    compose_drivor_log_score, compute_scorer_correction_losses)


def synthetic_cached_scorer_inputs():
    torch.manual_seed(17)
    return {"scored_features":torch.randn(2,5,256),"scene_features":torch.randn(2,16,256),
        "ego_status":torch.randn(2,11),"trajectories":torch.randn(2,5,8,3),
        "component_logits":torch.randn(2,5,6)}


def test_zero_initialization_and_off_restore_original_logits():
    model=CachedScorerCorrection()
    inputs=synthetic_cached_scorer_inputs()
    assert torch.equal(model(**inputs),inputs["component_logits"])
    torch.nn.init.normal_(model.correction_head.weight,std=.05)
    assert not torch.equal(model(**inputs),inputs["component_logits"])
    assert torch.equal(model(**inputs,correction_enabled=False),inputs["component_logits"])


def test_existing_drivor_composition_matches():
    logits=synthetic_cached_scorer_inputs()["component_logits"].double()
    weights={"noc":1.,"dac":1.,"ddc":0.,"ep":5.,"ttc":5.,"comfort":2.}
    probability=logits.sigmoid()
    original=(probability[...,0].log()+probability[...,1].log()
        +(5*probability[...,2]+5*probability[...,3]+2*probability[...,4]).log())
    torch.testing.assert_close(compose_drivor_log_score(logits,weights),original,rtol=1e-12,atol=1e-12)


def test_finite_head_then_adapter_gradient_without_frozen_input_gradients():
    inputs=synthetic_cached_scorer_inputs()
    model=CachedScorerCorrection()
    optimizer=torch.optim.AdamW(model.parameters(),lr=.001)
    targets=torch.rand(2,5,7)
    pairs=torch.tensor([[[0,1],[2,3],[4,0]],[[0,1],[2,3],[4,0]]])
    weights={"noc":1.,"dac":1.,"ddc":0.,"ep":5.,"ttc":5.,"comfort":2.}
    for step in range(2):
        optimizer.zero_grad(set_to_none=True)
        losses=compute_scorer_correction_losses(model(**inputs),targets,weights,pairs)
        sum(losses).backward()
        assert all(parameter.grad is not None and torch.isfinite(parameter.grad).all() for parameter in model.parameters())
        assert model.correction_head.weight.grad.norm()>0
        if step == 1:
            assert model.feature_adapter[0].weight.grad.norm()>0
        optimizer.step()
    assert all(value.grad is None for value in inputs.values())


def test_ranking_penalizes_reversing_true_order():
    weights={"noc":1.,"dac":1.,"ddc":0.,"ep":5.,"ttc":5.,"comfort":2.}
    targets=torch.ones(1,2,7)
    targets[0,1,6]=0.
    logits=torch.ones(1,2,6)
    logits[0,1,1]=-2.
    pairs=torch.tensor([[[0,1]]])
    correct=compute_scorer_correction_losses(logits,targets,weights,pairs)[1]
    reversed_loss=compute_scorer_correction_losses(logits.flip(1),targets,weights,pairs)[1]
    assert correct<reversed_loss


def test_sampler_gradient_visual_dependence_and_exploration_floor():
    torch.manual_seed(3)
    condition=build_sampler_condition(torch.randn(3,16,256),torch.randn(3,11),torch.randn(3,8,2),torch.ones(3,8,2))
    assert condition.shape==(3,299)
    model=SelectionErrorFamilySampler()
    probabilities=model.probabilities(condition)
    assert torch.isfinite(probabilities).all()
    torch.testing.assert_close(probabilities.sum(-1),torch.ones(3))
    assert probabilities.min()>=.25/8
    loss=(model(condition)-torch.randn(3,8)).square().mean()
    loss.backward()
    assert all(parameter.grad is not None and torch.isfinite(parameter.grad).all() for parameter in model.parameters())
    assert all(parameter.grad.norm()>0 for parameter in model.parameters())
    changed=condition.clone()
    changed[:,:256]=torch.randn(3,256)*3
    assert (model.probabilities(changed)-probabilities).abs().max()>1e-6
