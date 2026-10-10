"""Disposable CPU connection/freeze check for the matched adapter control."""
import json
from pathlib import Path
import sys

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / 'src'))
from planning_aware_future_prediction.object_centric.small_corpus_models import CommonPlannerModel
from planning_aware_future_prediction.object_centric.small_corpus_frozen_adapter import (
    SmallCorpusFrozenAdapterPlanner, adapter_gradient_groups,
)
from planning_aware_future_prediction.object_centric.lpwm_drivor_joint import make_official_loss, oracle_loss_callback
from diagnose_small_corpus_planner import scene_inputs, write_json, DIAGNOSTIC_ROOT, STUDY_ROOT
from train_small_corpus_common_planner import binary_safety_targets
from navsim.agents.drivoR.layers.losses import drivor_loss as official_loss_module
from lpwm_drivor_oracle import DrivoROracleClient


def main():
    torch.set_num_threads(4)
    official_loss_module.three_to_two_classes = binary_safety_targets
    stage1 = STUDY_ROOT / 'lpwm_ssl/latest.pt'
    torch.manual_seed(47)
    reference = CommonPlannerModel('lpwm_sequential', stage1).eval()
    torch.manual_seed(47)
    model = SmallCorpusFrozenAdapterPlanner(stage1).eval()
    records = json.loads((STUDY_ROOT / 'corpus/manifest.json').read_text())['records']
    record = next(row for row in records if row['study_split'] == 'train')
    features, targets = scene_inputs(record, 'cpu')
    with torch.no_grad():
        expected, actual = reference(features), model(features)
        difference = float((expected['proposals'] - actual['proposals']).abs().max())
        assert difference < 1e-5, difference
    initial = torch.load(STUDY_ROOT / 'lpwm_sequential/initial_common_planner.pt', map_location='cpu', weights_only=False)
    assert all(torch.equal(model.planner.state_dict()[name], tensor) for name, tensor in initial.items())
    del reference, expected, actual
    frozen_digest = model.frozen_native_digest()
    optimizer = torch.optim.AdamW(model.optimizer_groups(), weight_decay=.01)
    model.train()
    prediction = model(features)
    oracle = DrivoROracleClient(STUDY_ROOT / 'corpus/manifest.json', DIAGNOSTIC_ROOT / 'adapter_audit', workers=2)
    try:
        labels = torch.from_numpy(oracle.score([record['token']], prediction['proposals'].detach().float().numpy()))
        criterion, configuration = make_official_loss('navsim_v1')
        losses = criterion(targets, prediction, configuration, scoring_function=oracle_loss_callback(labels))
        losses['loss'].backward()
        groups = adapter_gradient_groups(model)
        assert all(value > 0 and __import__('math').isfinite(value) for value in groups.values()), groups
        assert all(parameter.grad is None for parameter in model.native_parameters)
        optimizer.step()
        assert model.frozen_native_digest() == frozen_digest
        world = model.backbone.world_model
        result = {'passed': True, 'device': 'cpu', 'disposable_optimizer_steps': 1,
                  'diagnostic_weights_used_for_training': False, 'initial_proposals_maximum_difference': difference,
                  'initial_common_planner_tensor_count': len(initial), 'native_weights_and_buffers_unchanged': True,
                  'native_gradient_count': 0, 'adapter_regions': model.adapter_region_counts,
                  'adapter_parameter_count': sum(parameter.numel() for parameter in model.adapter_parameters()),
                  'trainable_parameter_count': sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad),
                  'gradient_groups': groups, 'planning_loss': float(losses['loss'].detach()),
                  'module_parameter_counts': {name: sum(parameter.numel() for parameter in module.parameters()) for name,module in
                                              [('encoder',world.encoder_module),('context',world.ctx_module),('dynamics',world.dyn_module)]}}
        write_json(DIAGNOSTIC_ROOT / 'adapter_audit/preflight.json', result)
        print(json.dumps(result), flush=True)
    finally:
        oracle.close()


if __name__ == '__main__':
    main()
