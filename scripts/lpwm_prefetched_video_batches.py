"""Bounded CPU prefetch for already decoded LPWM video memory maps."""
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


class CachedVideoClips(Dataset):
    def __init__(self, cache_path, clip_frame_indices):
        self.cache_path = str(cache_path)
        self.clip_frame_indices = np.asarray(clip_frame_indices, dtype=np.int64)
        self.frame_cache = None

    def __len__(self):
        return len(self.clip_frame_indices)

    def __getitem__(self, clip_index):
        if self.frame_cache is None:
            self.frame_cache = np.load(self.cache_path, mmap_mode="r")
        return torch.from_numpy(np.array(self.frame_cache[self.clip_frame_indices[clip_index]]))


def create_rank_video_loader(cache_path, clip_frame_indices, epoch_order, rank, world_size,
                             microbatch_size, accumulation, completed_epoch_updates, num_workers):
    rank_batch_size = microbatch_size * accumulation
    global_batch_size = rank_batch_size * world_size
    rank_order = np.asarray(epoch_order).reshape(-1, world_size, rank_batch_size)[:, rank]
    remaining_order = rank_order[completed_epoch_updates:].reshape(-1)
    clips = CachedVideoClips(cache_path, clip_frame_indices[remaining_order])
    arguments = dict(batch_size=microbatch_size, shuffle=False, drop_last=False,
                     num_workers=num_workers, pin_memory=True,
                     generator=torch.Generator().manual_seed(20261003 + rank))
    if num_workers:
        arguments.update(multiprocessing_context="spawn", prefetch_factor=2)
    return DataLoader(clips, **arguments)
