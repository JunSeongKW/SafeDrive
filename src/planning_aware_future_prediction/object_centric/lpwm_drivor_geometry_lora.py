"""Planning LoRA on current particle geometry and particle attention.

Keep every public LPWM parameter and buffer fixed. Extend the existing Q/V
condition with zero-initialized adapters in the xy, scale and presence heads.
"""
import torch
from torch import nn

from .lpwm_drivor_lora import LPWMDrivoRLoRAModel
from .lpwm_lora_finetuning import LowRankLinearAdaptation


class LPWMDrivoRGeometryLoRAModel(LPWMDrivoRLoRAModel):
    def __init__(self, public_checkpoint, benchmark="navsim_v1", checkpoint_cameras=True):
        super().__init__(public_checkpoint, benchmark, checkpoint_cameras)
        heads = self.particle_encoder.world_model.encoder_module.particle_enc.particle_attribute_enc
        self.geometry_lora_targets = []
        # Isolate initialization from the original planner, dropout and Q/V LoRA.
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(13004)
            for head_name in ("xy_head", "scale_xy_head", "obj_on_head"):
                head = getattr(heads, head_name)
                adapters = []
                for layer_name, original in list(head.named_children()):
                    if not isinstance(original, nn.Linear):
                        continue
                    # Output sizes are 4, 4 and 1; never exceed a map's dimensions.
                    rank = int(min(8, original.in_features, original.out_features))
                    adapter = LowRankLinearAdaptation(original, rank=rank, alpha=rank)
                    setattr(head, layer_name, adapter)
                    adapters.append(adapter)
                    target = f"particle_geometry.{head_name}.{layer_name}"
                    self.lora_target_names.append(target)
                    self.geometry_lora_targets.append({"target": target, "rank": rank,
                        "in_features": int(original.in_features), "out_features": int(original.out_features),
                        "alpha": rank, "scaling": 1.0})
                assert len(adapters) == 2, head_name
                region = "particle_" + head_name
                self._adapted_regions[region] = adapters
                self.lora_region_counts[region] = len(adapters)
        self.assert_native_frozen()

    def parameter_inventory(self):
        inventory = super().parameter_inventory()
        inventory["attention_rank"] = inventory.pop("rank")
        inventory["attention_projections"] = inventory.pop("projections")
        inventory["geometry_lora_targets"] = self.geometry_lora_targets
        inventory["geometry_lora_parameters"] = sum(parameter.numel()
            for region, adapters in self._adapted_regions.items() if region.startswith("particle_") and region.endswith("_head")
            for adapter in adapters for parameter in (adapter.lora_input_projection, adapter.lora_output_projection))
        inventory["native_xy_scale_presence_heads_frozen"] = True
        inventory["current_xy_scale_presence_head_adapters_trainable"] = True
        inventory["image_cnn_conv_lora"] = False
        return inventory
