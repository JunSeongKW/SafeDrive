"""Selective LPWM fine-tuning without changing the preserved full-training model."""
from pathlib import Path

import numpy as np
import torch
from torch.utils.checkpoint import checkpoint

from .lpwm_candidate_planner import ParticleCandidatePlanner
from .lpwm_planning_finetuning import particle_attributes


class PartiallyFineTunedParticlePlanner(ParticleCandidatePlanner):
    """Adapt native output layers; retain gradients through the causal world model."""

    def __init__(self, pretrained_checkpoint, candidates, configuration):
        super().__init__(pretrained_checkpoint, candidates,
            imitation_exponent=configuration["inference_imitation_exponent"])
        self.partial_configuration = configuration["partial_finetuning"]
        # Move the existing zero-initialized command FiLM to the end of the
        # attribute CNN. The frozen prefix then needs no parameter gradients.
        self.world_model.encoder_module.particle_enc.particle_attribute_enc.cnn.conv_out.register_forward_hook(
            self._modulate_output_features)
        self.world_model.requires_grad_(False)
        self.trainable_world_modules = [
            "encoder_module.particle_enc.particle_attribute_enc.cnn.conv_out",
            "encoder_module.particle_enc.particle_attribute_enc.xy_head",
            "encoder_module.particle_enc.particle_attribute_enc.scale_xy_head",
            "encoder_module.particle_enc.particle_attribute_enc.obj_on_head",
            "encoder_module.particle_enc.particle_features_enc.to_mu",
            "encoder_module.particle_enc.particle_features_enc.to_logvar",
            "encoder_module.particle_inter_enc.particle_decoder",
            "encoder_module.ctx_enc.pte.head",
            "encoder_module.ctx_enc.posterior_decoder",
            "encoder_module.ctx_enc.prior_decoder",
            "dyn_module.particle_transformer.head",
            "dyn_module.particle_decoder",
        ]
        if self.partial_configuration["policy"] == "last_block":
            self.trainable_world_modules += [
                "encoder_module.ctx_enc.pte.blocks.3",
                "dyn_module.particle_transformer.blocks.5",
            ]
        else:
            assert self.partial_configuration["policy"] == "output_layers"
        for name in self.trainable_world_modules:
            self.world_model.get_submodule(name).requires_grad_(True)

    def _modulate_encoder_features(self, _module, _arguments, features):
        # The superclass registers this hook at conv_in; keep that prefix frozen.
        return features

    def _modulate_output_features(self, module, arguments, features):
        return super()._modulate_encoder_features(module, arguments, features)

    def optimizer_parameter_groups(self, world_learning_rate, planner_learning_rate):
        return [{**group, "params": [parameter for parameter in group["params"] if parameter.requires_grad]}
            for group in super().optimizer_parameter_groups(world_learning_rate, planner_learning_rate)]

    def forward(self, observed_images, ego_status, auxiliary_world_images=None, auxiliary_ego_status=None,
                reconstruction_loss=None, world_loss_configuration=None, intervention=None):
        assert observed_images.shape[1:] == (4, 3, 128, 128)
        with self.encoder_command(ego_status), torch.autocast(observed_images.device.type, enabled=False):
            images = observed_images.float().contiguous()
            if self.world_model.normalize_rgb:
                images = images * 2 - 1
            encoded = self.world_model.encoder_module(images, deterministic=True)
        arguments = (encoded["z"], encoded["z_scale"], encoded["obj_on"], encoded["z_depth"],
            encoded["z_features"], encoded["z_bg_features"], encoded["z_context"][:, 1:].contiguous(), encoded["z_score"])
        if self.training and self.partial_configuration["checkpoint_future_rollout"]:
            future = checkpoint(self._sample_future, *arguments, use_reentrant=False)
        else:
            future = self._sample_future(*arguments)
        observed_attributes = particle_attributes(encoded, "obj_on")
        future_attributes = particle_attributes(future, "z_obj_on")[:, -8:]
        if intervention == "persistent_future":
            future_attributes = observed_attributes[:, -1:].expand(-1, 8, -1, -1)
        attributes = torch.cat((observed_attributes, future_attributes), 1)
        memory = self.particle_projection(attributes) + self.time_embedding[None, :, None] + self.particle_embedding[None, None]
        output = self.plan_from_memory(memory.flatten(1, 2), ego_status)
        if hasattr(self, "object_state_head"):
            output.update(self.object_state_head(torch.cat((observed_attributes[:, -1:], future_attributes), 1), ego_status))
        objective = output["trajectory"].new_zeros(())
        if auxiliary_world_images is not None:
            def world_objective(videos, status):
                return self._world_objective(videos, status, reconstruction_loss, world_loss_configuration)
            if self.training and self.partial_configuration["checkpoint_world_objective"]:
                objective = checkpoint(world_objective, auxiliary_world_images, auxiliary_ego_status, use_reentrant=False)
            else:
                objective = world_objective(auxiliary_world_images, auxiliary_ego_status)
        return {**output, "world_objective": objective,
            "observed_particle_attributes": observed_attributes, "predicted_particle_attributes": future_attributes}


def build_partial_planning_model(checkpoint_path, specification, condition, project_root):
    candidates = np.load(Path(project_root) / specification["teacher_directory"] / "trajectory_vocabulary.npy")
    model = PartiallyFineTunedParticlePlanner(checkpoint_path, candidates, specification)
    if "object_future" in condition:
        from .lpwm_object_supervision import ParticleObjectStateHead
        configuration = specification["object_auxiliary"]
        model.object_state_head = ParticleObjectStateHead(configuration["auxiliary_head_hidden_dimension"],
            len(configuration["categories"]))
    return model


def parameter_inventory(model):
    world_ids = {id(parameter) for parameter in model.world_model.parameters()}
    return {
        "total_parameters": sum(parameter.numel() for parameter in model.parameters()),
        "trainable_parameters": sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad),
        "trainable_world_parameters": sum(parameter.numel() for parameter in model.world_model.parameters() if parameter.requires_grad),
        "trainable_planner_parameters": sum(parameter.numel() for parameter in model.parameters()
            if parameter.requires_grad and id(parameter) not in world_ids),
        "trainable_world_modules": model.trainable_world_modules,
        "trainable_parameter_names": [name for name, parameter in model.named_parameters() if parameter.requires_grad],
    }
