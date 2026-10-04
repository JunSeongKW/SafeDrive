"""Protect optimizer continuity, scientific settings and memory-based selection."""
import copy
import hashlib
import json
from pathlib import Path
import sys

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from lpwm_48gb_execution import choose_profile, fork_execution_checkpoint, verify_execution_configuration


def make_configuration(directory):
    parent = {"microbatch_size_per_gpu": 4, "gradient_accumulation": 2, "world_size": 2,
        "world_auxiliary_clips_per_gpu_microbatch": 2, "workers_per_rank": 0, "lpwm_learning_rate": 1e-5}
    parent_path = directory / "parent.json"
    parent_path.write_text(json.dumps(parent))
    configuration = {**parent, "microbatch_size_per_gpu": 8, "gradient_accumulation": 1,
        "world_auxiliary_clips_per_gpu_microbatch": 4, "resource_limits": {"maximum_gpu_used_bytes": 48_000_000_000},
        "execution_amendment": {"parent_configuration": "parent.json",
            "parent_configuration_sha256": hashlib.sha256(parent_path.read_bytes()).hexdigest()}}
    config_path = directory / "execution.json"
    config_path.write_text(json.dumps(configuration))
    return configuration, config_path


def test_scientific_changes_rejected(tmp_path):
    configuration, _ = make_configuration(tmp_path)
    verify_execution_configuration(configuration, tmp_path)
    for field, value in (("lpwm_learning_rate", 2e-5), ("gradient_accumulation", 2), ("workers_per_rank", 4)):
        changed = copy.deepcopy(configuration)
        changed[field] = value
        with pytest.raises(AssertionError):
            verify_execution_configuration(changed, tmp_path)


def test_checkpoint_fork_preserves_next_adam_update(tmp_path):
    configuration, config_path = make_configuration(tmp_path)
    model = torch.nn.Linear(2, 1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.01)
    for _ in range(3):
        optimizer.zero_grad()
        model(torch.ones(1, 2)).square().sum().backward()
        optimizer.step()
    source = tmp_path / "source.pt"
    torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(), "completed_updates": 3,
        "elapsed_seconds": 15., "configuration_sha256": configuration["execution_amendment"]["parent_configuration_sha256"]}, source)
    source_hash = hashlib.sha256(source.read_bytes()).hexdigest()
    destination = tmp_path / "continuation/latest.pt"
    fork_execution_checkpoint(source, destination, config_path, tmp_path)
    copied = torch.load(destination, weights_only=False)
    restored = torch.nn.Linear(2, 1)
    restored.load_state_dict(copied["model"])
    restored_optimizer = torch.optim.AdamW(restored.parameters(), lr=.01)
    restored_optimizer.load_state_dict(copied["optimizer"])
    for instance, adam in ((model, optimizer), (restored, restored_optimizer)):
        adam.zero_grad()
        instance(torch.ones(1, 2)).square().sum().backward()
        adam.step()
    assert all(torch.equal(value, restored.state_dict()[name]) for name, value in model.state_dict().items())
    assert copied["completed_updates"] == 3 and copied["elapsed_seconds"] == 15.
    profile_path = tmp_path / "profile/latest.pt"
    fork_execution_checkpoint(source, profile_path, config_path, tmp_path, profile_only=True)
    profile = torch.load(profile_path, weights_only=False)
    assert profile["completed_updates"] == 0
    assert {int(value["step"]) for value in profile["optimizer"]["state"].values()} == {3}
    assert hashlib.sha256(source.read_bytes()).hexdigest() == source_hash


def test_fast_but_memory_unsafe_profile_rejected():
    safe = {"passed": True, "minimum_cap_headroom_gib": 8, "minimum_free_gib": 12, "steady_update_seconds": 4}
    risky = {"passed": True, "minimum_cap_headroom_gib": .2, "minimum_free_gib": 3.2, "steady_update_seconds": 3}
    assert choose_profile([safe, risky]) is safe
    with pytest.raises(RuntimeError):
        choose_profile([risky])
