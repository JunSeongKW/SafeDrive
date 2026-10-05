"""A fixed Stage1 LPWM representation with the existing candidate planner."""
import hashlib
from pathlib import Path

import numpy as np
import torch

from .lpwm_partial_finetuning import PartiallyFineTunedParticlePlanner, parameter_inventory
from .lpwm_planning_finetuning import particle_attributes


class FrozenLPWMPlannerControl(PartiallyFineTunedParticlePlanner):
    """Freeze perception, dynamics and encoder FiLM; learn only the planner.

    Ego commands still enter the original planner's ego projection. Freezing the
    encoder FiLM is necessary to keep representations fixed as training proceeds.
    World modules remain in evaluation mode so buffers and dropout cannot change
    the representation. This mode difference is recorded in the comparison.
    """

    def __init__(self, checkpoint_path, candidates, specification):
        super().__init__(checkpoint_path, candidates, specification)
        self.world_model.requires_grad_(False)
        self.command_feature_modulation.requires_grad_(False)
        self.trainable_world_modules = []
        self.train(True)

    def train(self, mode=True):
        super().train(mode)
        self.world_model.eval()
        self.command_feature_modulation.eval()
        return self

    def frozen_representation_digest(self):
        result = hashlib.sha256()
        for prefix, module in (("world_model", self.world_model),
                               ("command_feature_modulation", self.command_feature_modulation)):
            for name, tensor in module.state_dict().items():
                result.update((prefix + "." + name).encode())
                result.update(tensor.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes())
        return result.hexdigest()

    def forward(self, observed_images, ego_status, auxiliary_world_images=None, auxiliary_ego_status=None,
                reconstruction_loss=None, world_loss_configuration=None, intervention=None):
        assert observed_images.shape[1:] == (4, 3, 128, 128)
        assert not self.world_model.training and not self.command_feature_modulation.training
        with torch.no_grad():
            with self.encoder_command(ego_status), torch.autocast(observed_images.device.type, enabled=False):
                images = observed_images.float().contiguous()
                if self.world_model.normalize_rgb:
                    images = images * 2 - 1
                encoded = self.world_model.encoder_module(images, deterministic=True)
            arguments = (encoded["z"], encoded["z_scale"], encoded["obj_on"], encoded["z_depth"],
                encoded["z_features"], encoded["z_bg_features"], encoded["z_context"][:, 1:].contiguous(), encoded["z_score"])
            future = self._sample_future(*arguments)
            observed_attributes = particle_attributes(encoded, "obj_on")
            future_attributes = particle_attributes(future, "z_obj_on")[:, -8:]
            if intervention == "persistent_future":
                future_attributes = observed_attributes[:, -1:].expand(-1, 8, -1, -1)
            elif intervention is not None:
                raise ValueError(f"Unsupported frozen-control intervention: {intervention}")
            attributes = torch.cat((observed_attributes, future_attributes), 1)
        # Gradients begin at the existing planner's particle projection.
        memory = self.particle_projection(attributes) + self.time_embedding[None, :, None] + self.particle_embedding[None, None]
        output = self.plan_from_memory(memory.flatten(1, 2), ego_status)
        objective = output["trajectory"].new_zeros(())
        if auxiliary_world_images is not None:
            with torch.no_grad():
                objective = self._world_objective(auxiliary_world_images.contiguous(), auxiliary_ego_status,
                    reconstruction_loss, world_loss_configuration)
        assert not objective.requires_grad and not attributes.requires_grad
        return {**output, "world_objective": objective,
            "observed_particle_attributes": observed_attributes, "predicted_particle_attributes": future_attributes}


def build_frozen_control_model(checkpoint_path, specification, condition, project_root):
    assert specification["adaptation_method"] == "frozen_lpwm_planner_control"
    assert condition == "metric_plus_world"
    candidates = np.load(Path(project_root) / specification["teacher_directory"] / "trajectory_vocabulary.npy")
    return FrozenLPWMPlannerControl(checkpoint_path, candidates, specification)


def frozen_control_inventory(model):
    inventory = parameter_inventory(model)
    inventory.update(adaptation_method="frozen_lpwm_planner_control", world_in_evaluation_mode=True,
        encoder_command_film_frozen=True,
        frozen_command_parameters=sum(parameter.numel() for parameter in model.command_feature_modulation.parameters()),
        frozen_representation_sha256=model.frozen_representation_digest(),
        world_objective_role="Detached monitor; no gradients to LPWM or planner")
    assert inventory["trainable_world_parameters"] == 0
    return inventory
