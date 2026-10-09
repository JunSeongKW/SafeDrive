"""NAVSIM-adapted LPWM, observed front-camera history, and planning-path LoRA.

The time axis contains four observed frames, never four cameras or future RGB.
Native Stage 1 weights remain fixed. All 64 foreground particles contribute
current and prior-predicted future attributes to the official DrivoR planner.
"""
import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from .lpwm_drivor_joint import ParticleSceneEncoder
from .lpwm_drivor_planning_path_lora import LPWMDrivoRPlanningPathLoRAModel
from .lpwm_planning_finetuning import particle_attributes


class FrontHistoryParticleSceneEncoder(ParticleSceneEncoder):
    def __init__(self, source_encoder):
        nn.Module.__init__(self)
        source_encoder._command_hook.remove()
        self.world_model = source_encoder.world_model
        self.lpwm_configuration = source_encoder.lpwm_configuration
        self.future_steps = source_encoder.future_steps
        self.checkpoint_cameras = source_encoder.checkpoint_cameras
        self.command_modulation = source_encoder.command_modulation
        self.particle_trajectory_projection = source_encoder.particle_trajectory_projection
        self.camera_embedding = nn.Parameter(source_encoder.camera_embedding[:1].detach().clone())
        self.active_command = None
        self.forward_command = None
        self.record_particles = False
        self.latest_particle_attributes = None
        attribute_cnn = self.world_model.encoder_module.particle_enc.particle_attribute_enc.cnn
        self._command_hook = attribute_cnn.conv_in.register_forward_hook(self._condition_image_features)

    def _encode_front_history(self, observed_images, current_command):
        self.active_command = current_command
        try:
            with torch.autocast(observed_images.device.type, enabled=False):
                encoder_images = observed_images.float()
                if self.world_model.normalize_rgb:
                    encoder_images = encoder_images * 2 - 1
                encoded = self.world_model.encoder_module(encoder_images, deterministic=True)
        finally:
            self.active_command = None
        predicted = self.world_model.dyn_module.sample(
            encoded["z"], encoded["z_scale"], encoded["obj_on"], encoded["z_depth"],
            encoded["z_features"], encoded["z_bg_features"],
            z_context=encoded["z_context"][:, 1:].contiguous(),
            z_score=encoded["z_score"], steps=self.future_steps, deterministic=True,
            return_context_posterior=False)
        current_attributes = particle_attributes(encoded, "obj_on")[:, -1:]
        future_attributes = particle_attributes(predicted, "z_obj_on")[:, -self.future_steps:]
        return torch.cat((current_attributes, future_attributes), 1)

    def forward(self, observed_images, _unused_register_seeds):
        assert observed_images.ndim == 5 and observed_images.shape[1:] == (4, 3, 128, 128)
        assert self.forward_command is not None
        if self.training and self.checkpoint_cameras:
            attributes = checkpoint(self._encode_front_history, observed_images,
                                    self.forward_command, use_reentrant=False)
        else:
            attributes = self._encode_front_history(observed_images, self.forward_command)
        particle_states = attributes.permute(0, 2, 1, 3).flatten(2)
        scene_tokens = self.particle_trajectory_projection(particle_states)
        assert scene_tokens.shape[1:] == (64, 256)
        if self.record_particles:
            self.latest_particle_attributes = attributes[:, None].detach().float().cpu()
        return scene_tokens + self.camera_embedding[0]


class FrontHistoryStage1LoRAPlanner(LPWMDrivoRPlanningPathLoRAModel):
    def __init__(self, stage1_checkpoint, benchmark="navsim_v1", checkpoint_cameras=True):
        super().__init__(stage1_checkpoint, benchmark, checkpoint_cameras)
        self.planner.image_backbone = FrontHistoryParticleSceneEncoder(self.particle_encoder)
        self.particle_encoder.world_model.timestep_horizon = 11
        self.planner.num_cams = 1
        self.planner.scene_embeds = torch.zeros(1, 1, 64, 1)

    def parameter_inventory(self):
        inventory = super().parameter_inventory()
        inventory.update({
            "condition": "stage1_adapted_front_history_planning_path_lora",
            "camera_count": 1, "observed_frames": 4, "image_size": [128, 128],
            "foreground_particles": 64, "background_particles": 1,
            "scene_token_count": 64, "particle_pooling": "none",
            "future_steps": 8, "future_rgb_is_input": False,
            "context_source": "four observed frames only; future context uses prior",
            "context_posterior_status": "frozen observed-history context inference; posterior has no LoRA",
            "background_path": "background attributes broadcast into foreground temporal attributes",
        })
        return inventory
