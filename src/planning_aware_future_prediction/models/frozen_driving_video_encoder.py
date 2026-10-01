"""Import pinned official Drive-JEPA encoder without importing its NAVSIM stack."""

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import torch
from torch import Tensor, nn


class FrozenDrivingVideoEncoder(nn.Module):
    def __init__(self, project_root: Path, device: torch.device):
        super().__init__()
        configuration = json.loads(
            (project_root / "configs/visual_pilot/encoder_checkpoint.json").read_text()
        )
        source_root = project_root / "reference_repositories/Drive-JEPA"
        actual_commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=source_root, text=True
        ).strip()
        if actual_commit != configuration["source_repository_commit"]:
            raise RuntimeError(
                "official source commit differs from pinned configuration"
            )
        if subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=source_root, text=True
        ).strip():
            raise RuntimeError("official reference source is modified")
        checkpoint_path = (
            project_root / "runtime/checkpoints/drive_jepa" / configuration["filename"]
        )
        if checkpoint_path.stat().st_size != configuration["size_bytes"]:
            raise RuntimeError("checkpoint is incomplete")
        with checkpoint_path.open("rb") as checkpoint_file:
            checkpoint_sha = hashlib.file_digest(checkpoint_file, "sha256").hexdigest()
        if checkpoint_sha != configuration["sha256"]:
            raise RuntimeError("checkpoint SHA256 mismatch")
        sys.path.insert(0, str(source_root / "navsim_v1"))
        from vjepa2.src.models.vision_transformer import vit_large

        self.encoder = vit_large(
            img_size=(256, 512),
            num_frames=2,
            patch_size=16,
            tubelet_size=2,
            use_rope=True,
            uniform_power=True,
            use_activation_checkpointing=False,
        )
        # Only trusted, hash-pinned tensors/config accepted; no arbitrary pickle fallback.
        checkpoint = torch.load(
            checkpoint_path, map_location="cpu", weights_only=True, mmap=True
        )
        checkpoint_key = configuration["checkpoint_encoder_key"]
        state_dict = {
            key.removeprefix("module.").removeprefix("backbone."): value
            for key, value in checkpoint[checkpoint_key].items()
        }
        self.encoder.load_state_dict(state_dict, strict=True)
        self.encoder.requires_grad_(False).eval().to(device)
        self.provenance = {
            "checkpoint_sha256": checkpoint_sha,
            "checkpoint_encoder_key": checkpoint_key,
            "strict_weight_loading": True,
            "state_dict_tensor_count": len(state_dict),
            "encoder_parameters": sum(
                parameter.numel() for parameter in self.encoder.parameters()
            ),
            "source_commit": actual_commit,
            "target_encoder_policy": "same_frozen_official_encoder_no_ema_update",
        }

    @torch.no_grad()
    def forward(self, observed_video_clip: Tensor) -> Tensor:
        self.encoder.eval()
        if tuple(observed_video_clip.shape[1:]) != (3, 2, 256, 512):
            raise ValueError("expected [batch, 3, 2, 256, 512] observed video")
        encoded_tokens = self.encoder(observed_video_clip)
        if tuple(encoded_tokens.shape[1:]) != (512, 1024):
            raise RuntimeError(
                f"unexpected official encoder output {encoded_tokens.shape}"
            )
        return encoded_tokens.transpose(1, 2).reshape(-1, 1024, 16, 32)
