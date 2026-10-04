"""LoRA on LPWM particle, context and dynamics attention for planning."""
import math
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .lpwm_partial_finetuning import PartiallyFineTunedParticlePlanner, parameter_inventory


class LowRankLinearAdaptation(nn.Module):
    """Preserve the pretrained linear map and learn an initially zero update."""

    def __init__(self, pretrained_linear, rank, alpha):
        super().__init__()
        if not isinstance(pretrained_linear, nn.Linear):
            raise TypeError("LoRA target must be an existing nn.Linear")
        if not 0 < rank <= min(pretrained_linear.in_features, pretrained_linear.out_features):
            raise ValueError("Invalid LoRA rank")
        self.pretrained_linear = pretrained_linear.requires_grad_(False)
        self.rank = rank
        self.scaling = alpha / rank
        self.lora_input_projection = nn.Parameter(pretrained_linear.weight.new_empty(rank, pretrained_linear.in_features))
        self.lora_output_projection = nn.Parameter(pretrained_linear.weight.new_zeros(pretrained_linear.out_features, rank))
        nn.init.kaiming_uniform_(self.lora_input_projection, a=math.sqrt(5))

    def forward(self, features):
        update = F.linear(F.linear(features, self.lora_input_projection), self.lora_output_projection)
        return self.pretrained_linear(features) + self.scaling * update


class LoRAFineTunedParticlePlanner(PartiallyFineTunedParticlePlanner):
    """Keep native LPWM weights fixed; adapt attention in all three core paths."""

    def __init__(self, pretrained_checkpoint, candidates, configuration):
        super().__init__(pretrained_checkpoint, candidates, configuration)
        self.world_model.requires_grad_(False)
        adaptation = configuration["lora_finetuning"]
        self.lora_configuration = dict(adaptation)
        self.trainable_world_modules = []
        self.lora_region_counts = {}
        attention_types = {"ParticleSelfAttention", "CausalParticleSelfAttention", "ParticleCrossAttention"}
        # The context object is shared with dynamics. named_modules removes
        # duplicate instances, so a shared projection is wrapped exactly once.
        with torch.random.fork_rng(devices=[]):
            torch.default_generator.manual_seed(configuration["seed"] + 13001)
            for region in adaptation["regions"]:
                region_root = self.world_model.get_submodule(region)
                count = 0
                for relative_name, module in list(region_root.named_modules()):
                    if type(module).__name__ not in attention_types:
                        continue
                    for projection_name in adaptation["projections"]:
                        linear = getattr(module, projection_name)
                        if isinstance(linear, LowRankLinearAdaptation):
                            continue
                        replacement = LowRankLinearAdaptation(linear, adaptation["rank"], adaptation["alpha"])
                        setattr(module, projection_name, replacement)
                        self.trainable_world_modules.append(".".join((region, relative_name, projection_name)))
                        count += 1
                if count == 0:
                    raise ValueError(f"No attention projections found in {region}")
                self.lora_region_counts[region] = count
        for name, parameter in self.world_model.named_parameters():
            assert parameter.requires_grad == ("lora_input_projection" in name or "lora_output_projection" in name), name


def build_lora_planning_model(checkpoint_path, specification, condition, project_root):
    if condition != "metric_plus_world":
        raise ValueError("Object auxiliary supervision is deferred in this comparison")
    candidates = np.load(Path(project_root) / specification["teacher_directory"] / "trajectory_vocabulary.npy")
    return LoRAFineTunedParticlePlanner(checkpoint_path, candidates, specification)


def lora_parameter_inventory(model):
    return {**parameter_inventory(model), "lora_region_counts": model.lora_region_counts,
        "lora_configuration": model.lora_configuration, "native_lpwm_parameters_frozen": True,
        "attention_projection_modules": sum(model.lora_region_counts.values())}
