"""Protect the exact optimizer/RNG snapshot across a live latest.pt replacement."""
import importlib.util
import json
from pathlib import Path

import pytest
import torch

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/queue_lpwm_drivor_epoch_review.py"
SPEC = importlib.util.spec_from_file_location("epoch_review_control", SCRIPT)
control = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(control)


def checkpoint_state(update=1614):
    return {"completed_updates": update, "epoch": 1, "next_update_in_epoch": 0, "total_updates": 40350,
            "model": {"weight": torch.tensor([1.])}, "optimizer": {"state": {0: {"step": torch.tensor(update)}}},
            "scheduler": {"last_epoch": update}, "rng_by_rank": [{"torch": [1]}, {"torch": [2]}],
            "configuration_sha256": "preserved", "frozen_native_sha256": "preserved"}


def test_capture_keeps_open_epoch_checkpoint_when_latest_is_replaced(tmp_path, monkeypatch):
    source = tmp_path / "latest.pt"
    torch.save(checkpoint_state(), source)
    expected = control.digest(source)
    copy_original = control.shutil.copyfileobj

    def replace_latest_before_copying(open_source, open_destination, length):
        replacement = tmp_path / "latest.pending.pt"
        torch.save(checkpoint_state(1615), replacement)
        replacement.replace(source)
        copy_original(open_source, open_destination, length)

    monkeypatch.setattr(control.shutil, "copyfileobj", replace_latest_before_copying)
    destination = tmp_path / "review.pt"
    checksum = control.capture_checkpoint_and_request_pause(source, destination, tmp_path / "pause.requested", {"owner": "epoch_review"})
    assert checksum == expected
    assert torch.load(source, weights_only=False)["completed_updates"] == 1615
    assert control.inspect_resume_state(destination, 1, 1614)["remaining_updates"] == 38736


def test_existing_user_pause_is_not_overwritten(tmp_path):
    source = tmp_path / "latest.pt"
    torch.save(checkpoint_state(), source)
    pause = tmp_path / "pause.requested"
    pause.write_text("user pause")
    with pytest.raises(FileExistsError):
        control.capture_checkpoint_and_request_pause(source, tmp_path / "review.pt", pause, {"owner": "epoch_review"})
    assert pause.read_text() == "user pause"
    assert not (tmp_path / "review.pt").exists()


def test_mismatched_epoch_snapshot_cannot_be_reported_as_exact_resume(tmp_path):
    source = tmp_path / "late.pt"
    torch.save(checkpoint_state(1615), source)
    with pytest.raises(AssertionError, match="Missed the exact epoch"):
        control.inspect_resume_state(source, 1, 1614)
