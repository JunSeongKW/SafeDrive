"""Audit native partial updates, frozen weights, intent and future-label isolation."""
import argparse
import hashlib
import json
from pathlib import Path

import torch

import audit_lpwm_planning_finetuning as original_audit
from planning_aware_future_prediction.object_centric.lpwm_partial_finetuning import (
    build_partial_planning_model, parameter_inventory)
from evaluate_lpwm_full_planning import digest, write_json


def frozen_digest(model):
    result = hashlib.sha256()
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            result.update(name.encode())
            result.update(parameter.detach().cpu().contiguous().numpy().tobytes())
    return result.hexdigest()


def run(arguments):
    models, original_frozen = [], []
    def audited_builder(*builder_arguments):
        model = build_partial_planning_model(*builder_arguments)
        original_frozen.append(frozen_digest(model))
        models.append(model)
        return model
    original_audit.build_planning_model = audited_builder
    device = torch.device(arguments.device)
    torch.cuda.set_per_process_memory_fraction(20 * 1024**3 / torch.cuda.get_device_properties(device).total_memory, device)
    original_audit.main(arguments)
    model = models[0]
    assert frozen_digest(model) == original_frozen[0], "A frozen LPWM weight changed"
    assert all(parameter.grad is None for parameter in model.parameters() if not parameter.requires_grad)
    groups = model.optimizer_parameter_groups(1e-5, 3e-4)
    identities = [id(parameter) for group in groups for parameter in group["params"]]
    assert len(identities) == len(set(identities))
    assert all(parameter.requires_grad for group in groups for parameter in group["params"])
    report = json.loads(arguments.output.read_text())
    report.update(parameter_inventory=parameter_inventory(model), frozen_weights_identical_after_optimizer_step=True,
        frozen_sha256=original_frozen[0], configuration_sha256=digest(arguments.config),
        partial_model_source_sha256=digest(Path(__file__).resolve().parents[1] /
            "src/planning_aware_future_prediction/object_centric/lpwm_partial_finetuning.py"))
    write_json(arguments.output, report)
    print("PARTIAL_GRADIENT_AND_FREEZE_AUDIT_PASSED", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--condition", default="metric_plus_world")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--teacher-profile", action="store_true")
    run(parser.parse_args())
