"""DrivoR-style rank-32 Q/V LoRA on a frozen public LPWM backbone.

Native CNN/MLP/normalization/attention parameters stay frozen. Existing new
command FiLM, scene projection, and the original DrivoR planner stay trainable.
The same Q/V low-rank parameterization is used; the backbones differ structurally.
"""
import hashlib

import torch
from torch import nn

from .lpwm_drivor_joint import LPWMDrivoRJointModel
from .lpwm_lora_finetuning import LowRankLinearAdaptation


class LPWMDrivoRLoRAModel(LPWMDrivoRJointModel):
    def __init__(self, public_checkpoint, benchmark="navsim_v1", checkpoint_cameras=True):
        super().__init__(public_checkpoint, benchmark, checkpoint_cameras)
        world = self.particle_encoder.world_model
        world.requires_grad_(False)
        self.lora_target_names = []
        self.lora_region_counts = {}
        regions = {
            "particle_interaction": world.encoder_module.particle_inter_enc,
            "context_prior": world.dyn_module.context_decoder,
            "particle_dynamics": world.dyn_module.particle_transformer,
        }
        attention_types = {"ParticleSelfAttention", "CausalParticleSelfAttention", "ParticleCrossAttention"}
        self._adapted_regions = {}
        # LoRA creation must not change planner initialization or dropout RNG.
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(13003)
            for region_name, region in regions.items():
                adapters = []
                for module_name, module in list(region.named_modules()):
                    if type(module).__name__ not in attention_types:
                        continue
                    for projection_name in ("query", "value"):
                        original = getattr(module, projection_name)
                        assert isinstance(original, nn.Linear)
                        adapter = LowRankLinearAdaptation(original, rank=32, alpha=32)
                        setattr(module, projection_name, adapter)
                        adapters.append(adapter)
                        self.lora_target_names.append(f"{region_name}.{module_name}.{projection_name}")
                assert adapters, region_name
                self._adapted_regions[region_name] = adapters
                self.lora_region_counts[region_name] = len(adapters)
        self.assert_native_frozen()

    def train(self, mode=True):
        super().train(mode)
        # Frozen pretrained buffers must not drift. Dropout retains the LPWM
        # train/eval convention; only BatchNorm running-statistics are locked.
        for module in self.particle_encoder.world_model.modules():
            if isinstance(module, nn.modules.batchnorm._BatchNorm):
                module.eval()
        return self

    def assert_native_frozen(self):
        for name, parameter in self.particle_encoder.world_model.named_parameters():
            is_lora = "lora_input_projection" in name or "lora_output_projection" in name
            assert parameter.requires_grad == is_lora, name
            if not is_lora:
                assert parameter.grad is None, name

    def frozen_native_digest(self):
        result = hashlib.sha256()
        for name, tensor in sorted(self.particle_encoder.world_model.state_dict().items()):
            if "lora_input_projection" in name or "lora_output_projection" in name:
                continue
            canonical_name = name.replace(".pretrained_linear.", ".")
            result.update(canonical_name.encode())
            result.update(tensor.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes())
        return result.hexdigest()

    def gradient_groups(self):
        groups = {}
        for region, adapters in self._adapted_regions.items():
            groups[region + "_lora"] = [parameter for adapter in adapters
                for parameter in (adapter.lora_input_projection, adapter.lora_output_projection)]
        groups["encoder_command"] = list(self.particle_encoder.command_modulation.parameters())
        groups["scene_projection"] = list(self.particle_encoder.particle_trajectory_projection.parameters()) + [self.particle_encoder.camera_embedding]
        groups["trajectory_generator"] = list(self.planner.trajectory_decoder.parameters()) + list(self.planner.traj_head.parameters())
        groups["scorer"] = list(self.planner.scorer_attention.parameters()) + list(self.planner.scorer.parameters())
        return groups

    def parameter_inventory(self):
        world = self.particle_encoder.world_model
        return {"rank": 32, "scaling": 1.0, "projections": ["query", "value"],
            "region_counts": self.lora_region_counts, "targets": self.lora_target_names,
            "trainable_lora_parameters": sum(parameter.numel() for parameter in world.parameters() if parameter.requires_grad),
            "frozen_lpwm_parameters": sum(parameter.numel() for parameter in world.parameters() if not parameter.requires_grad),
            "all_trainable_parameters": sum(parameter.numel() for parameter in self.parameters() if parameter.requires_grad),
            "native_backbone_weights_frozen": True, "new_command_film_trainable": True,
            "native_xy_scale_presence_heads_frozen": True}
