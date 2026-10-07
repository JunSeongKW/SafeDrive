"""Keep the original LPWM reconstruction objective with bounded LPIPS activations."""
import torch
from torch import nn
from torch.utils.checkpoint import checkpoint


class CheckpointedFrameReconstructionLoss(nn.Module):
    """Evaluate the same frozen LPIPS/pixel loss in small frame batches.

    LPIPS remains in eval mode. Concatenation preserves per-frame weighting and
    checkpointing retains gradients to the reconstructed RGB frames.
    """

    def __init__(self, original_loss, frames_per_chunk=2):
        super().__init__()
        self.original_loss = original_loss.eval()
        self.frames_per_chunk = frames_per_chunk

    def forward(self, inputs, reconstructions, reduction="mean", split="train", p_loss=True):
        assert inputs.shape == reconstructions.shape and inputs.ndim == 4
        chunks = []
        for offset in range(0, len(inputs), self.frames_per_chunk):
            reference = inputs[offset:offset + self.frames_per_chunk]
            reconstructed = reconstructions[offset:offset + self.frames_per_chunk]
            keywords = {"reduction": "none", "split": split, "p_loss": p_loss}
            if torch.is_grad_enabled() and (reference.requires_grad or reconstructed.requires_grad):
                values = checkpoint(self.original_loss, reference, reconstructed,
                                    use_reentrant=False, **keywords)
            else:
                values = self.original_loss(reference, reconstructed, **keywords)
            chunks.append(values)
        per_frame = torch.cat(chunks, dim=0)
        if reduction == "mean":
            return per_frame.mean()
        if reduction == "sum":
            return per_frame.sum() * inputs[0].numel()
        return per_frame
