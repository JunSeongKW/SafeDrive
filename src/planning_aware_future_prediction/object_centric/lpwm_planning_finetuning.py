"""Fine-tune a post-trained full LPWM with a learned trajectory planner."""
from contextlib import contextmanager

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from .lpwm_bridge import load_official_lpwm
from .lpwm_planner import normalized_ego_status


def particle_attributes(encoded, presence_name):
    return torch.cat((encoded["z"], encoded["z_scale"].sigmoid(), encoded[presence_name],
        encoded["z_depth"].tanh(), encoded["z_features"].tanh(),
        encoded["z_bg_features"][:, :, None].expand(-1, -1, 64, -1).tanh()), -1)


class PlanningFineTunedLPWM(nn.Module):
    """Planner inputs are observed RGB/status and causal LPWM prior predictions."""
    def __init__(self, pretrained_checkpoint, hidden_dimension=256, condition_encoder_on_command=True):
        super().__init__()
        self.world_model, self.official_configuration = load_official_lpwm("cpu", pretrained_checkpoint)
        self.world_model.timestep_horizon = 11
        self.condition_encoder_on_command = condition_encoder_on_command
        self.command_feature_modulation = nn.Sequential(nn.Linear(4, 64), nn.SiLU(), nn.Linear(64, 64))
        nn.init.zeros_(self.command_feature_modulation[-1].weight)
        nn.init.zeros_(self.command_feature_modulation[-1].bias)
        self._current_command = None
        self.world_model.encoder_module.particle_enc.particle_attribute_enc.cnn.conv_in.register_forward_hook(self._modulate_encoder_features)
        self.particle_projection = nn.Sequential(nn.Linear(14, hidden_dimension), nn.LayerNorm(hidden_dimension), nn.GELU())
        self.time_embedding = nn.Parameter(torch.randn(12, hidden_dimension) * .02)
        self.particle_embedding = nn.Parameter(torch.randn(64, hidden_dimension) * .02)
        self.ego_projection = nn.Sequential(nn.Linear(8, hidden_dimension), nn.GELU(), nn.Linear(hidden_dimension, hidden_dimension))
        self.waypoint_queries = nn.Parameter(torch.randn(8, hidden_dimension) * .02)
        layer = nn.TransformerDecoderLayer(hidden_dimension, 8, hidden_dimension * 4,
            dropout=.1, batch_first=True, norm_first=True)
        self.trajectory_decoder = nn.TransformerDecoder(layer, 4)
        self.trajectory_head = nn.Linear(hidden_dimension, 3)
        # Small nonzero initialization allows planning gradients to reach LPWM on update1.
        nn.init.normal_(self.trajectory_head.weight, std=.001)
        nn.init.zeros_(self.trajectory_head.bias)

    @contextmanager
    def encoder_command(self, ego_status):
        previous_command = self._current_command
        self._current_command = ego_status[:, :4]
        try:
            yield
        finally:
            self._current_command = previous_command

    def _modulate_encoder_features(self, _module, _arguments, features):
        if not self.condition_encoder_on_command or self._current_command is None:
            return features
        modulation = self.command_feature_modulation(self._current_command.float())
        scale, shift = modulation.chunk(2, -1)
        repeats = features.shape[0] // len(modulation)
        assert repeats * len(modulation) == features.shape[0]
        scale = .1 * scale.tanh().repeat_interleave(repeats, 0)[..., None, None]
        shift = .1 * shift.tanh().repeat_interleave(repeats, 0)[..., None, None]
        return features * (1 + scale) + shift

    def _sample_future(self, positions, scales, presence, depth, features, background, context, scores):
        return self.world_model.dyn_module.sample(positions, scales, presence, depth, features, background,
            z_context=context, z_score=scores, steps=8, deterministic=True, return_context_posterior=False)

    def _world_objective(self, videos, ego_status, reconstruction_loss, loss_configuration):
        with self.encoder_command(ego_status), torch.autocast(videos.device.type, enabled=False):
            arguments = {name: value for name, value in loss_configuration.items() if name != "implementation"}
            output = self.world_model(videos.float(), with_loss=True, warmup=False, num_static=1,
                recon_loss_func=reconstruction_loss, **arguments)
            return output["loss_dict"]["loss"]

    def plan_from_memory(self, memory, ego_status):
        queries = self.waypoint_queries[None] + self.ego_projection(normalized_ego_status(ego_status))[:, None]
        decoded = self.trajectory_decoder(queries, memory)
        future_times = torch.arange(1, 9, device=ego_status.device, dtype=ego_status.dtype) * .5
        constant_velocity = ego_status.new_zeros((len(ego_status), 8, 3))
        constant_velocity[..., :2] = ego_status[:, None, 4:6] * future_times[None, :, None]
        return {"trajectory": constant_velocity + self.trajectory_head(decoded) * decoded.new_tensor([10, 10, 1])}

    def forward(self, observed_images, ego_status, auxiliary_world_images=None, auxiliary_ego_status=None,
                reconstruction_loss=None, world_loss_configuration=None, intervention=None):
        assert observed_images.shape[1:] == (4, 3, 128, 128)
        with self.encoder_command(ego_status), torch.autocast(observed_images.device.type, enabled=False):
            encoder_images = observed_images.float().contiguous()
            if self.world_model.normalize_rgb:
                encoder_images = encoder_images * 2 - 1
            encoded = self.world_model.encoder_module(encoder_images, deterministic=True)
        arguments = (encoded["z"], encoded["z_scale"], encoded["obj_on"], encoded["z_depth"],
            encoded["z_features"], encoded["z_bg_features"], encoded["z_context"][:, 1:].contiguous(), encoded["z_score"])
        future = checkpoint(self._sample_future, *arguments, use_reentrant=False) if self.training else self._sample_future(*arguments)
        observed_attributes = particle_attributes(encoded, "obj_on")
        future_attributes = particle_attributes(future, "z_obj_on")[:, -8:]
        if intervention == "persistent_future":
            future_attributes = observed_attributes[:, -1:].expand(-1, 8, -1, -1)
        elif intervention == "shuffled_future":
            future_attributes = future_attributes.roll(1, 0)
        attributes = torch.cat((observed_attributes, future_attributes), 1)
        memory = self.particle_projection(attributes) + self.time_embedding[None, :, None] + self.particle_embedding[None, None]
        planner_output = self.plan_from_memory(memory.flatten(1, 2), ego_status)
        objective = planner_output["trajectory"].new_zeros(())
        if auxiliary_world_images is not None:
            assert auxiliary_ego_status is not None and reconstruction_loss is not None
            def calculate_world_objective(videos, status):
                return self._world_objective(videos, status, reconstruction_loss, world_loss_configuration)
            objective = checkpoint(calculate_world_objective, auxiliary_world_images, auxiliary_ego_status, use_reentrant=False)
        return {**planner_output, "world_objective": objective,
            "observed_particle_attributes": observed_attributes, "predicted_particle_attributes": future_attributes}

    def optimizer_parameter_groups(self, world_learning_rate, planner_learning_rate):
        world_parameters = list(self.world_model.parameters())
        world_ids = {id(parameter) for parameter in world_parameters}
        planner_parameters = [parameter for parameter in self.parameters() if id(parameter) not in world_ids]
        return [{"params": world_parameters, "lr": world_learning_rate, "name": "lpwm"},
            {"params": planner_parameters, "lr": planner_learning_rate, "name": "planner_and_command_input"}]
