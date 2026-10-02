"""Canonical spatial patch-tube IDs for untrained WA-JEPA sensitivity tests.

These are camera/time/spatial tokens, never detected object identities. Selection
uses geometry or an explicit random seed only, and does not access future targets.
"""
import hashlib

import torch


def derive_scene_selection_seed(scene_token: str, selection_seed: int) -> int:
    digest = hashlib.sha256(f"{selection_seed}:{scene_token}".encode()).digest()
    return int.from_bytes(digest[:8], "little") % (2**63 - 1)


def build_canonical_future_token_ids(
    *,
    num_cameras: int,
    num_future_tubelets: int,
    spatial_grid_rows: int,
    spatial_grid_columns: int,
    selection_policy: str,
    selected_spatial_tubes_per_camera: int,
    selection_seed: int = 29,
) -> torch.Tensor:
    dimensions = (num_cameras, num_future_tubelets, spatial_grid_rows, spatial_grid_columns)
    if any(dimension <= 0 for dimension in dimensions):
        raise ValueError("Camera/time/spatial dimensions must be positive")
    num_spatial_tokens = spatial_grid_rows * spatial_grid_columns
    if not 0 < selected_spatial_tubes_per_camera <= num_spatial_tokens:
        raise ValueError("Spatial quota must be between one and the full spatial grid")
    generator = torch.Generator(device="cpu").manual_seed(selection_seed)
    selected_camera_ids = []
    for camera_index in range(num_cameras):
        if selection_policy == "packed_all":
            if selected_spatial_tubes_per_camera != num_spatial_tokens:
                raise ValueError("All-ID selection requires the full spatial quota")
            spatial_ids = torch.arange(num_spatial_tokens, dtype=torch.long)
        elif selection_policy == "fixed_spatial_lattice":
            spatial_ids = torch.tensor([
                row * spatial_grid_columns + column
                for row in range(0, spatial_grid_rows, 2)
                for column in range(0, spatial_grid_columns, 2)
            ], dtype=torch.long)
            if spatial_ids.numel() != selected_spatial_tubes_per_camera:
                raise ValueError("Registered even-row/even-column lattice must exactly match the quota")
        elif selection_policy == "seeded_random_spatial_tubes":
            spatial_ids = torch.randperm(num_spatial_tokens, generator=generator)[:selected_spatial_tubes_per_camera].sort().values
        else:
            raise ValueError(f"Unrecognized untrained selection policy: {selection_policy}")
        camera_offset = camera_index * num_future_tubelets * num_spatial_tokens
        for future_tubelet_index in range(num_future_tubelets):
            selected_camera_ids.append(spatial_ids + camera_offset + future_tubelet_index * num_spatial_tokens)
    return torch.cat(selected_camera_ids).unsqueeze(0)


def decode_canonical_future_token_ids(
    canonical_future_token_ids: torch.Tensor,
    *,
    num_future_tubelets: int,
    spatial_grid_rows: int,
    spatial_grid_columns: int,
) -> dict[str, torch.Tensor]:
    num_spatial_tokens = spatial_grid_rows * spatial_grid_columns
    tokens_per_camera = num_future_tubelets * num_spatial_tokens
    local_ids = canonical_future_token_ids % tokens_per_camera
    spatial_ids = local_ids % num_spatial_tokens
    return {
        "camera_index": canonical_future_token_ids // tokens_per_camera,
        "future_tubelet_index": local_ids // num_spatial_tokens,
        "spatial_row": spatial_ids // spatial_grid_columns,
        "spatial_column": spatial_ids % spatial_grid_columns,
    }
