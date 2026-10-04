"""Independent residual-adapter and full-weight LPWM planning comparisons."""
from pathlib import Path

import numpy as np
import torch
from torch import nn

from .lpwm_partial_finetuning import PartiallyFineTunedParticlePlanner, parameter_inventory
from .lpwm_candidate_planner import ParticleCandidatePlanner


class ParticleBlockWithAdapter(nn.Module):
    """Append a zero-initialized nonlinear residual to an unchanged LPWM block."""

    def __init__(self, pretrained_block, feature_dimension, bottleneck_dimension):
        super().__init__()
        self.pretrained_block = pretrained_block.requires_grad_(False)
        self.residual_adapter = nn.Sequential(
            nn.LayerNorm(feature_dimension),
            nn.Linear(feature_dimension, bottleneck_dimension),
            nn.GELU(),
            nn.Linear(bottleneck_dimension, feature_dimension),
        )
        nn.init.zeros_(self.residual_adapter[-1].weight)
        nn.init.zeros_(self.residual_adapter[-1].bias)

    def forward(self, *arguments, **keywords):
        features = self.pretrained_block(*arguments, **keywords)
        return features + self.residual_adapter(features)


class AdapterFineTunedParticlePlanner(PartiallyFineTunedParticlePlanner):
    """Train added adapters, planner and intent input; freeze native LPWM."""

    def __init__(self, pretrained_checkpoint, candidates, configuration):
        super().__init__(pretrained_checkpoint, candidates, configuration)
        self.world_model.requires_grad_(False)
        self.adapter_configuration = dict(configuration["adapter_finetuning"])
        self.trainable_world_modules = []
        self.adapter_region_counts = {}
        with torch.random.fork_rng(devices=[]):
            torch.default_generator.manual_seed(configuration["seed"] + 17001)
            for region in self.adapter_configuration["regions"]:
                transformer = self.world_model.get_submodule(region)
                count = 0
                for index, block in enumerate(list(transformer.blocks)):
                    assert not isinstance(block, ParticleBlockWithAdapter)
                    transformer.blocks[index] = ParticleBlockWithAdapter(block, transformer.n_embed,
                        self.adapter_configuration["bottleneck_dimension"])
                    self.trainable_world_modules.append(f"{region}.blocks.{index}.residual_adapter")
                    count += 1
                assert count > 0
                self.adapter_region_counts[region] = count
        for name, parameter in self.world_model.named_parameters():
            assert parameter.requires_grad == (".residual_adapter." in name), name


class FullyFineTunedParticlePlanner(ParticleCandidatePlanner):
    """Preserve the earlier full model, including its conv_in intent input."""

    def __init__(self, pretrained_checkpoint, candidates, configuration):
        super().__init__(pretrained_checkpoint, candidates,
            imitation_exponent=configuration["inference_imitation_exponent"])
        self.world_model.requires_grad_(True)
        self.trainable_world_modules = ["encoder_module", "dyn_module", "decoder_module"]
        assert all(parameter.requires_grad for parameter in self.world_model.parameters())


def build_adapter_or_full_planning_model(checkpoint_path, specification, condition, project_root):
    if condition != "metric_plus_world":
        raise ValueError("Direct object GT auxiliary supervision remains deferred")
    candidates = np.load(Path(project_root) / specification["teacher_directory"] / "trajectory_vocabulary.npy")
    method = specification["adaptation_method"]
    if method == "residual_adapter":
        return AdapterFineTunedParticlePlanner(checkpoint_path, candidates, specification)
    if method == "full_low_learning_rate":
        return FullyFineTunedParticlePlanner(checkpoint_path, candidates, specification)
    raise ValueError(f"Unregistered adaptation method: {method}")


def adaptation_parameter_inventory(model):
    inventory = parameter_inventory(model)
    if isinstance(model, AdapterFineTunedParticlePlanner):
        inventory.update(adaptation_method="residual_adapter", native_lpwm_parameters_frozen=True,
            adapter_configuration=model.adapter_configuration, adapter_region_counts=model.adapter_region_counts)
    else:
        inventory.update(adaptation_method="full_low_learning_rate", native_lpwm_parameters_frozen=False,
            rgb_decoder_trainable_for_ssl=True)
    inventory["world_parameters_by_region"] = {
        name: {"total": sum(parameter.numel() for parameter in module.parameters()),
            "trainable": sum(parameter.numel() for parameter in module.parameters() if parameter.requires_grad)}
        for name, module in (
            ("image_encoder", model.world_model.encoder_module.particle_enc),
            ("particle_interaction", model.world_model.encoder_module.particle_inter_enc),
            ("context", model.world_model.encoder_module.ctx_enc),
            ("dynamics_including_shared_context", model.world_model.dyn_module),
            ("rgb_decoder", model.world_model.decoder_module),
        )}
    inventory["region_counts_note"] = "Dynamics includes a shared context instance; regional totals must not be summed."
    return inventory
