"""CPU-only loading comparison for the actual Stage2 RGB cache and batch sizes."""
import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import statistics
import time
from zoneinfo import ZoneInfo

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

PROJECT_ROOT = Path(__file__).resolve().parents[1]


class PlanningAndWorldMicrobatches(Dataset):
    def __init__(self, cache_path, observed_indices, world_indices):
        self.cache_path = str(cache_path)
        self.observed_indices = observed_indices
        self.world_indices = world_indices
        self.frames = None

    def __len__(self):
        return len(self.observed_indices)

    def __getitem__(self, index):
        if self.frames is None:
            self.frames = np.load(self.cache_path, mmap_mode="r")
        observed = torch.from_numpy(np.array(self.frames[self.observed_indices[index]]))
        world = torch.from_numpy(np.array(self.frames[self.world_indices[index]]))
        return {"observed_uint8": observed, "world_uint8": world}


def initialize_worker(_worker_id):
    torch.set_num_threads(1)


def benchmark(arguments):
    torch.set_num_threads(1)
    assert not torch.cuda.is_initialized()
    stage1_root = PROJECT_ROOT / "outputs/lpwm_navsim_full_posttraining_v2"
    records = json.loads((stage1_root / "planning_manifest.json").read_text())["records"]
    world_records = json.loads((stage1_root / "manifest.json").read_text())["records"]
    train_frames = np.asarray([row["frame_cache_indices"][:4] for row in records if row["split"] == "train"])
    frame_indices_by_token = {row["current_frame_token"]: row["frame_cache_indices"] for row in records}
    world_frames = np.asarray([frame_indices_by_token[row["current_frame_token"]] for row in world_records if row["split"] == "train"])
    generator = np.random.default_rng(20261005)
    microbatches = arguments.warmup_microbatches + arguments.measured_microbatches
    observed_indices = train_frames[generator.integers(len(train_frames), size=(microbatches, 4))]
    world_indices = world_frames[generator.integers(len(world_frames), size=(microbatches, 2))]
    del records, world_records, frame_indices_by_token, train_frames, world_frames
    records = []
    for repetition, order in enumerate(((0, 2, 4, 8), (8, 4, 2, 0))):
        for workers in order:
            dataset = PlanningAndWorldMicrobatches(stage1_root / "rgb_frames.npy", observed_indices, world_indices)
            options = dict(batch_size=None, num_workers=workers, pin_memory=False, shuffle=False)
            if workers:
                options.update(multiprocessing_context="spawn", prefetch_factor=2, worker_init_fn=initialize_worker)
            start = time.perf_counter()
            loader = DataLoader(dataset, **options)
            iterator = iter(loader)
            latencies, fingerprint = [], 0
            warmup_seconds = None
            for index in range(microbatches):
                before = time.perf_counter()
                batch = next(iterator)
                elapsed = time.perf_counter() - before
                assert batch["observed_uint8"].shape == (4, 4, 128, 128, 3)
                assert batch["world_uint8"].shape == (2, 12, 128, 128, 3)
                fingerprint += int(batch["observed_uint8"][0, 0, 0, 0, 0]) + int(batch["world_uint8"][0, 0, 0, 0, 0])
                if index + 1 == arguments.warmup_microbatches:
                    warmup_seconds = time.perf_counter() - start
                if index >= arguments.warmup_microbatches:
                    latencies.append(elapsed)
            del iterator, loader, dataset, batch
            record = {"repetition": repetition + 1, "workers": workers, "warmup_seconds": warmup_seconds,
                "mean_microbatch_load_seconds": statistics.mean(latencies),
                "median_microbatch_load_seconds": statistics.median(latencies),
                "p95_microbatch_load_seconds": float(np.quantile(latencies, .95)),
                "equivalent_mean_per_update_seconds": 2 * statistics.mean(latencies),
                "input_fingerprint": fingerprint, "wall_seconds_including_startup_shutdown": time.perf_counter() - start}
            records.append(record)
            print(json.dumps(record), flush=True)
    assert len({row["input_fingerprint"] for row in records}) == 1
    report = {"checked_at_kst": datetime.now(ZoneInfo("Asia/Seoul")).isoformat(), "cpu_only": True,
        "training_interrupted": False, "gpu_work_executed": False,
        "observed_batch": 4, "world_batch": 2, "microbatches_per_update": 2,
        "warmup_microbatches": arguments.warmup_microbatches, "measured_microbatches": arguments.measured_microbatches,
        "records": records, "means_by_workers": {str(workers): statistics.mean(
            row["equivalent_mean_per_update_seconds"] for row in records if row["workers"] == workers)
            for workers in (0, 2, 4, 8)},
        "scope": "One rank, input producer only. Same real cached RGB batches; no H2D, pinning, GPU computation or pacing. Not an end-to-end training speed measurement.",
        "limitations": ["Sequential repetitions can have cache/order effects", "Production has two ranks",
            "A paced GPU consumer could hide some IPC wait; existing logged input preparation is the practical savings bound"]}
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(json.dumps(report, indent=2) + "\n")
    assert not torch.cuda.is_initialized()
    print("STAGE2_INPUT_LOADING_COMPARISON_COMPLETE", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--warmup-microbatches", type=int, default=32)
    parser.add_argument("--measured-microbatches", type=int, default=128)
    arguments = parser.parse_args()
    arguments.output = arguments.output.resolve()
    benchmark(arguments)
