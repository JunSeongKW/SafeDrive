"""Restore the preserved full Stage2 checkpoint without editing its original run."""
from pathlib import Path

import torch

from evaluate_lpwm_full_planning import digest


def restore_preserved_full_checkpoint(model, optimizer, specification, project_root):
    continuation = specification["resume_from_prior_full"]
    checkpoint_path = Path(project_root) / continuation["checkpoint"]
    source_config = Path(project_root) / continuation["source_configuration"]
    assert digest(checkpoint_path) == continuation["checkpoint_sha256"]
    saved = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    assert saved["configuration_sha256"] == digest(source_config)
    assert saved["completed_updates"] == continuation["completed_updates"]
    assert saved["world_size"] == specification["world_size"]
    model.load_state_dict(saved["model"], strict=True)
    optimizer.load_state_dict(saved["optimizer"])
    state_count = 0
    for group in optimizer.param_groups:
        for parameter in group["params"]:
            state = optimizer.state.get(parameter)
            if not state:
                continue
            assert int(state["step"]) == saved["completed_updates"]
            assert state["exp_avg"].shape == parameter.shape == state["exp_avg_sq"].shape
            assert torch.isfinite(state["exp_avg"]).all() and torch.isfinite(state["exp_avg_sq"]).all()
            state_count += 1
    assert state_count == len(saved["optimizer"]["state"]) > 0
    return {"completed_updates": saved["completed_updates"], "elapsed_seconds": saved["elapsed_seconds"],
        "optimizer_states_restored": state_count, "checkpoint_sha256": continuation["checkpoint_sha256"],
        "stage1_checkpoint_sha256": saved["stage1_checkpoint_sha256"],
        "source_configuration_sha256": saved["configuration_sha256"],
        "next_update": saved["completed_updates"] + 1, "new_optimizer_created_from_scratch": False,
        "original_run_modified": False}
