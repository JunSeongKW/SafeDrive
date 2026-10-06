"""LoRA along current geometry, visual appearance, and future-particle paths.

All public LPWM weights and buffers remain fixed. RGB reconstruction and the
unused context posterior are excluded from this planning-only condition.
"""
import hashlib
import math

import torch
from torch import nn

from .lpwm_drivor_geometry_lora import LPWMDrivoRGeometryLoRAModel
from .lpwm_lora_finetuning import LowRankLinearAdaptation


class LowRankConvolutionAdaptation(nn.Module):
    """Low-rank update to a flattened Conv2d kernel, preserving its geometry."""

    def __init__(self, pretrained_convolution, rank=8):
        super().__init__()
        assert isinstance(pretrained_convolution, nn.Conv2d)
        assert pretrained_convolution.groups == 1, "Grouped convolutions require a separate rank policy"
        self.pretrained_convolution = pretrained_convolution.requires_grad_(False)
        flattened_inputs = int(pretrained_convolution.weight[0].numel())
        self.rank = min(rank, int(pretrained_convolution.out_channels), flattened_inputs)
        self.scaling = 1.0
        self.lora_input_projection = nn.Parameter(pretrained_convolution.weight.new_empty(self.rank, flattened_inputs))
        self.lora_output_projection = nn.Parameter(pretrained_convolution.weight.new_zeros(pretrained_convolution.out_channels, self.rank))
        nn.init.kaiming_uniform_(self.lora_input_projection, a=math.sqrt(5))

    def forward(self, image_features):
        convolution = self.pretrained_convolution
        # The encoder executes in FP32. Preserve padding mode, dilation, stride,
        # and bias exactly, without materializing a second activation branch.
        kernel_update = (self.lora_output_projection @ self.lora_input_projection).reshape_as(convolution.weight)
        return convolution._conv_forward(image_features, convolution.weight + self.scaling * kernel_update, convolution.bias)


class LPWMDrivoRPlanningPathLoRAModel(LPWMDrivoRGeometryLoRAModel):
    def __init__(self, public_checkpoint, benchmark="navsim_v1", checkpoint_cameras=True):
        super().__init__(public_checkpoint, benchmark, checkpoint_cameras)
        encoder = self.particle_encoder
        world = encoder.world_model
        self.planning_path_targets = []
        self.excluded_planning_targets = []
        # Context appears through three aliases. named_modules deduplicates
        # shared instances, and existing Q/V + geometry adapters are retained.
        candidates = list(world.named_modules())
        encoder._command_hook.remove()
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(13005)
            for name, original in candidates:
                if not isinstance(original, (nn.Linear, nn.Conv2d)):
                    continue
                if ".pretrained_linear" in name:
                    continue
                if name.startswith("decoder_module.") or ".posterior_decoder." in name:
                    self.excluded_planning_targets.append(name)
                    continue
                region = self._region_for_path(name)
                if isinstance(original, nn.Conv2d):
                    adapter = LowRankConvolutionAdaptation(original, rank=8)
                    rank = adapter.rank
                    dimensions = {"kernel_shape": list(original.weight.shape)}
                else:
                    attention_projection = name.rsplit(".", 1)[-1] in ("key", "query", "value", "proj") and ".attn." in name
                    rank = int(min(32 if attention_projection else 8, original.in_features, original.out_features))
                    adapter = LowRankLinearAdaptation(original, rank=rank, alpha=rank)
                    dimensions = {"in_features": int(original.in_features), "out_features": int(original.out_features)}
                parent_name, child_name = name.rsplit(".", 1)
                setattr(world.get_submodule(parent_name), child_name, adapter)
                self._adapted_regions.setdefault(region, []).append(adapter)
                self.lora_target_names.append(name)
                self.planning_path_targets.append({"target": name, "region": region,
                    "type": type(original).__name__, "rank": rank, "scaling": 1.0, **dimensions})
        # FiLM conditions the sum of the native convolution and its LoRA update.
        # Hooking the retained native child would omit the adapter branch.
        attribute_cnn = world.encoder_module.particle_enc.particle_attribute_enc.cnn
        encoder._command_hook = attribute_cnn.conv_in.register_forward_hook(encoder._condition_image_features)
        self.lora_region_counts = {name: len(adapters) for name, adapters in self._adapted_regions.items()}
        assert world.ctx_module is world.encoder_module.ctx_enc is world.dyn_module.context_decoder
        self.assert_native_frozen()

    @staticmethod
    def _region_for_path(name):
        if ".particle_enc.prior_encoder." in name:
            return "current_prior_cnn"
        if ".particle_enc.particle_attribute_enc.cnn." in name:
            return "current_attribute_cnn"
        if ".particle_enc.particle_features_enc." in name:
            return "appearance_encoder"
        if ".bg_encoder." in name:
            return "background_appearance"
        if ".particle_inter_enc." in name:
            return "interaction_visual_and_output"
        if ".ctx_enc." in name or name.startswith("ctx_module."):
            return "context_prior_projection_and_output"
        if name.startswith("dyn_module.particle_decoder."):
            return "future_particle_heads"
        if name.startswith("dyn_module."):
            return "dynamics_projection_and_ffn"
        raise ValueError("Unclassified LoRA target: " + name)

    def frozen_native_digest(self):
        digest = hashlib.sha256()
        for name, tensor in sorted(self.particle_encoder.world_model.state_dict().items()):
            if "lora_input_projection" in name or "lora_output_projection" in name:
                continue
            canonical_name = name.replace(".pretrained_linear.", ".").replace(".pretrained_convolution.", ".")
            digest.update(canonical_name.encode())
            digest.update(tensor.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes())
        return digest.hexdigest()

    def parameter_inventory(self):
        inventory = super().parameter_inventory()
        inventory.update({"condition": "geometry_appearance_future_planning_path_lora",
            "planning_path_targets": self.planning_path_targets,
            "excluded_planning_targets": self.excluded_planning_targets,
            "image_cnn_conv_lora": True,
            "linear_adapter_count": sum(isinstance(adapter, LowRankLinearAdaptation) for adapters in self._adapted_regions.values() for adapter in adapters),
            "convolution_adapter_count": sum(isinstance(adapter, LowRankConvolutionAdaptation) for adapters in self._adapted_regions.values() for adapter in adapters),
            "rgb_decoder_frozen_and_unused": True, "context_posterior_excluded": True})
        return inventory
