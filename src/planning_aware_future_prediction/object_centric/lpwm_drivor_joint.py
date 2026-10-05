"""Public LPWM perception with the unmodified official DrivoR planning model.

Only the image-backbone interface is replaced. The official forward method,
trajectory generator, scorer, proposal stop-gradient and loss are imported.
No future image, object annotation or NAVSIM-adapted LPWM weight is an input.
"""
from contextlib import redirect_stdout
import copy
import io
from pathlib import Path
import sys

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint
from omegaconf import OmegaConf
import yaml

from .lpwm_bridge import load_official_lpwm
from .lpwm_planning_finetuning import particle_attributes

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DRIVOR_ROOT = PROJECT_ROOT / "reference_repositories/DrivoR"
sys.path.insert(0, str(DRIVOR_ROOT))
import navsim
if not Path(navsim.__file__).resolve().is_relative_to(DRIVOR_ROOT):
    raise RuntimeError("Import DrivoR's NAVSIM before any other NAVSIM checkout")
from navsim.agents.drivoR.drivor_model import DrivoRModel
from navsim.agents.drivoR.layers.losses.drivor_loss import DrivoRLoss


def official_configuration(benchmark="navsim_v1"):
    path = DRIVOR_ROOT / "navsim/planning/script/config/common/agent/drivoR.yaml"
    source = yaml.safe_load(path.read_text())
    configuration = OmegaConf.create(source["config"])
    configuration.long_trajectory_additional_poses = 2 if benchmark == "navsim_v1" else -1
    if benchmark == "navsim_v2":
        for name, value in dict(noc=10., dac=13., ddc=6., ttc=14., ep=15., comfort=2.).items():
            configuration[name] = value
    loss_arguments = {name: value for name, value in source["loss"].items() if name != "_target_"}
    return configuration, loss_arguments


class ParticleSceneEncoder(nn.Module):
    """Four current cameras -> 64 scene tokens, matching DrivoR's input budget.

Each of the 64 native particles retains its current and eight prior-predicted
states before projection. Fixed groups of four particle indices are averaged
to the official 16 scene tokens per camera. This pooling is not object tracking
or an importance selector; all particles receive a differentiable path.
"""
    def __init__(self, checkpoint_path, future_steps=8, checkpoint_cameras=True):
        super().__init__()
        with redirect_stdout(io.StringIO()):
            self.world_model, self.lpwm_configuration = load_official_lpwm("cpu", checkpoint_path)
        self.world_model.requires_grad_(False)
        self.world_model.encoder_module.requires_grad_(True)
        self.world_model.dyn_module.requires_grad_(True)
        self.future_steps = future_steps
        self.checkpoint_cameras = checkpoint_cameras
        self.command_modulation = nn.Sequential(nn.Linear(4, 64), nn.SiLU(), nn.Linear(64, 64))
        nn.init.zeros_(self.command_modulation[-1].weight)
        nn.init.zeros_(self.command_modulation[-1].bias)
        self.particle_trajectory_projection = nn.Sequential(
            nn.Linear((future_steps + 1) * 14, 256), nn.LayerNorm(256), nn.GELU())
        self.camera_embedding = nn.Parameter(torch.randn(4, 1, 256) * .02)
        self.active_command = None
        self.forward_command = None
        self.record_particles = False
        self.latest_particle_attributes = None
        attribute_cnn = self.world_model.encoder_module.particle_enc.particle_attribute_enc.cnn
        self._command_hook = attribute_cnn.conv_in.register_forward_hook(self._condition_image_features)

    def _condition_image_features(self, _module, _arguments, image_features):
        if self.active_command is None:
            return image_features
        scale, shift = self.command_modulation(self.active_command.float()).chunk(2, -1)
        repeats = image_features.shape[0] // len(scale)
        scale = .1 * scale.tanh().repeat_interleave(repeats, 0)[..., None, None]
        shift = .1 * shift.tanh().repeat_interleave(repeats, 0)[..., None, None]
        return image_features * (1 + scale) + shift

    def _encode_camera(self, current_rgb, current_command):
        self.active_command = current_command
        try:
            with torch.autocast(current_rgb.device.type, enabled=False):
                encoder_rgb = current_rgb.float()[:, None]
                if self.world_model.normalize_rgb:
                    encoder_rgb = encoder_rgb * 2 - 1
                encoded = self.world_model.encoder_module(encoder_rgb, deterministic=True)
        finally:
            self.active_command = None
        # Official LPWM supports single observed frame; every context here is a
        # prior prediction. No repeated history and no posterior future input.
        predicted = self.world_model.dyn_module.sample(
            encoded["z"], encoded["z_scale"], encoded["obj_on"], encoded["z_depth"],
            encoded["z_features"], encoded["z_bg_features"], z_context=None,
            z_score=encoded["z_score"], steps=self.future_steps, deterministic=True,
            return_context_posterior=False)
        current_attributes = particle_attributes(encoded, "obj_on")
        future_attributes = particle_attributes(predicted, "z_obj_on")[:, -self.future_steps:]
        return torch.cat((current_attributes, future_attributes), 1)

    def forward(self, current_images, _unused_register_seeds):
        assert current_images.ndim == 5 and current_images.shape[1:] == (4, 3, 128, 128)
        assert self.forward_command is not None
        camera_memories, visualized_particles = [], []
        for camera_index in range(4):
            camera_images = current_images[:, camera_index].contiguous()
            if self.training and self.checkpoint_cameras:
                attributes = checkpoint(self._encode_camera, camera_images, self.forward_command,
                                        use_reentrant=False)
            else:
                attributes = self._encode_camera(camera_images, self.forward_command)
            # (scene,time,particle,attribute) -> per-particle temporal state.
            particle_states = attributes.permute(0, 2, 1, 3).flatten(2)
            particle_features = self.particle_trajectory_projection(particle_states)
            scene_tokens = particle_features.reshape(len(current_images), 16, 4, 256).mean(2)
            camera_memories.append(scene_tokens + self.camera_embedding[camera_index])
            if self.record_particles:
                visualized_particles.append(attributes.detach().float().cpu())
        if self.record_particles:
            self.latest_particle_attributes = torch.stack(visualized_particles, 1)
        return torch.cat(camera_memories, 1)


class LPWMDrivoRJointModel(nn.Module):
    def __init__(self, public_checkpoint, benchmark="navsim_v1", checkpoint_cameras=True):
        super().__init__()
        configuration, _ = official_configuration(benchmark)
        construction_configuration = copy.deepcopy(configuration)
        for name in ("cam_f0", "cam_b0", "cam_l0", "cam_l1", "cam_l2", "cam_r0", "cam_r1", "cam_r2"):
            construction_configuration[name] = []
        # Instantiate the original downstream modules without downloading a
        # DINO backbone that this condition will not use.
        self.planner = DrivoRModel(construction_configuration)
        self.planner._config = configuration
        self.planner.num_cams = 4
        self.planner.register_buffer("scene_embeds", torch.zeros(1, 4, 16, 1))
        self.planner.image_backbone = ParticleSceneEncoder(public_checkpoint,
            checkpoint_cameras=checkpoint_cameras)

    @property
    def particle_encoder(self):
        return self.planner.image_backbone

    def forward(self, features):
        self.particle_encoder.forward_command = features["ego_status"][:, -1, 7:11]
        try:
            return self.planner(features)
        finally:
            self.particle_encoder.forward_command = None

    def optimizer_groups(self, lpwm_learning_rate, planner_learning_rate):
        world_ids = {id(parameter) for parameter in self.particle_encoder.world_model.parameters()}
        world_parameters, task_parameters = [], []
        for parameter in self.parameters():
            if parameter.requires_grad:
                (world_parameters if id(parameter) in world_ids else task_parameters).append(parameter)
        return [{"params": world_parameters, "lr": lpwm_learning_rate, "name": "lpwm_perception"},
                {"params": task_parameters, "lr": planner_learning_rate, "name": "planner_and_representation_projection"}]

    def gradient_groups(self):
        encoder = self.particle_encoder.world_model.encoder_module
        dynamics = self.particle_encoder.world_model.dyn_module
        context_ids = {id(parameter) for parameter in dynamics.context_decoder.parameters()}
        heads = encoder.particle_enc.particle_attribute_enc
        named_groups = {
            "particle_xy_head": list(heads.xy_head.parameters()),
            "particle_scale_head": list(heads.scale_xy_head.parameters()),
            "particle_presence_head": list(heads.obj_on_head.parameters()),
            "image_encoder": [parameter for parameter in encoder.parameters() if id(parameter) not in context_ids],
            "context_prior": list(dynamics.context_decoder.parameters()),
            "particle_dynamics": [parameter for parameter in dynamics.parameters() if id(parameter) not in context_ids],
            "encoder_command": list(self.particle_encoder.command_modulation.parameters()),
            "trajectory_generator": list(self.planner.trajectory_decoder.parameters()) + list(self.planner.traj_head.parameters()),
            "scorer": list(self.planner.scorer_attention.parameters()) + list(self.planner.scorer.parameters()),
        }
        return named_groups


def make_official_loss(benchmark="navsim_v1"):
    configuration, arguments = official_configuration(benchmark)
    return DrivoRLoss(**arguments), configuration


def oracle_loss_callback(subscores):
    def score(_targets, _proposals, test=False):
        assert not test and subscores.shape[-1] == 7
        scores = subscores[..., -1]
        return scores, scores.amax(-1), subscores, None, None, None
    return score
