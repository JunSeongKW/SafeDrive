"""Checks for future-region metrics and fixed, recording-aware diagnostics."""
from pathlib import Path
import sys
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from run_lpwm_navsim_posttraining import object_region_mse
from summarize_lpwm_posttraining import paired_recording_interval
from visualize_lpwm_posttraining_progress import select_visualization_records
from train_lpwm_navtrain_distributed import distributed_epoch_order


def test_object_region_metric_excludes_background_and_empty_frames():
    target = np.zeros((2, 128, 128, 3))
    predicted = np.ones_like(target) * 100
    predicted[0, 4:8, 2:6] = .5
    objects = [[{"box": [2., 4., 6., 8.]}], []]
    assert object_region_mse(predicted, target, objects) == .25
    assert object_region_mse(predicted, target, [[], []]) is None


def test_paired_recording_comparison_aligns_tokens_not_list_order():
    original = [{"token": str(index), "recording_group": str(index // 2), "metrics": {"forecast": 10.}}
        for index in range(8)]
    adapted = [{**record, "metrics": {"forecast": 8.}} for record in reversed(original)]
    result = paired_recording_interval(adapted, original, "forecast")
    assert result["mean_difference"] == -2
    assert result["ci95"] == [-2, -2]
    assert result["recordings"] == 4
    missing = [{**record, "metrics": {"forecast": None}} for record in original]
    assert paired_recording_interval(missing, original, "forecast")["ci95"] == [None, None]


def test_visualization_selection_is_fixed_and_excludes_training():
    rows = [{"current_frame_token": str(index), "split": "development", "scenario": "turn"}
        for index in range(6)]
    rows.append({"current_frame_token": "train", "split": "train", "scenario": "turn"})
    first = select_visualization_records({"records": rows})
    second = select_visualization_records({"records": list(reversed(rows))})
    assert first == second
    assert len(first) == 2
    assert all(row["split"] == "development" for row in first)


def test_distributed_epoch_covers_every_clip_without_dropping_last_batch():
    order = distributed_epoch_order(35, 16, 47, 0)
    assert len(order) == 48
    assert set(order) == set(range(35))
    np.testing.assert_array_equal(order, distributed_epoch_order(35, 16, 47, 0))
    assert not np.array_equal(order, distributed_epoch_order(35, 16, 47, 1))


def test_prefetch_preserves_rank_assignment_resume_and_training_rng(tmp_path):
    import torch
    from lpwm_prefetched_video_batches import create_rank_video_loader
    pixels = np.arange(48, dtype=np.uint8).reshape(48, 1, 1, 1)
    cache = tmp_path / "frames.npy"
    np.save(cache, pixels)
    clip_frame_indices = np.arange(48).reshape(48, 1)
    order = distributed_epoch_order(35, 16, 47, 0)
    expected = order.reshape(-1, 2, 8)[1:, 1].reshape(-1)
    torch.manual_seed(47)
    before = torch.get_rng_state().clone()
    loader = create_rank_video_loader(cache, clip_frame_indices, order, rank=1, world_size=2,
        microbatch_size=4, accumulation=2, completed_epoch_updates=1, num_workers=0)
    actual = np.concatenate([batch.numpy().reshape(-1) for batch in loader])
    np.testing.assert_array_equal(actual, expected)
    assert torch.equal(before, torch.get_rng_state())
