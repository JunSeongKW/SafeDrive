"""Shared DrivoR planner with explicitly matched front-video interfaces."""
import copy
from contextlib import redirect_stdout
import io
from pathlib import Path
import sys

import torch
from torch import nn
from torch.nn import functional as functional
from torch.utils.checkpoint import checkpoint

from .lpwm_drivor_joint import DrivoRModel, official_configuration
from .lpwm_rectangular import load_rectangular_lpwm

ROOT = Path(__file__).resolve().parents[3]
WEIGHTS = ROOT / 'runtime/checkpoints/four_model_small_corpus'
VJEPA_ROOT = ROOT / 'reference_repositories/Drive-JEPA/navsim_v1/vjepa2'


def normalize_imagenet(video):
    shape = (1, 1, 3, 1, 1)
    return (video - video.new_tensor([.485, .456, .406]).view(shape)) / video.new_tensor(
        [.229, .224, .225]).view(shape)


def make_jepa_networks():
    sys.path.insert(0, str(VJEPA_ROOT.parent))
    sys.path.insert(0, str(VJEPA_ROOT))
    from app.vjepa.utils import init_video_model
    with redirect_stdout(io.StringIO()):
        encoder, predictor = init_video_model(device='cpu', patch_size=16, max_num_frames=8,
            tubelet_size=2, model_name='vit_large', crop_size=(256, 512), pred_depth=12,
            pred_num_heads=12, pred_embed_dim=384, uniform_power=True, use_mask_tokens=True,
            num_mask_tokens=1, zero_init_mask_tokens=True, use_sdpa=True, use_rope=True,
            use_activation_checkpointing=True)
    return encoder, predictor


def initialize_jepa_from_public(encoder, predictor=None, target_encoder=None):
    saved = torch.load(WEIGHTS / 'vjepa2_vitl.pt', map_location='cpu', weights_only=True, mmap=True)
    report = {}
    for label, module in [('encoder', encoder), ('predictor', predictor), ('target_encoder', target_encoder)]:
        if module is None:
            continue
        state = {key.removeprefix('module.'): value for key, value in saved[label].items()}
        # General-video masks can have extra mask tokens; the one driving mask
        # uses token0. All encoder weights must match exactly.
        expected = module.state_dict()
        omitted = [key for key in state if key not in expected]
        if label == 'predictor':
            assert all('mask_tokens.' in key for key in omitted), omitted
            state = {key: value for key, value in state.items() if key in expected}
        module.load_state_dict(state, strict=True)
        report[label] = {'loaded': len(state), 'omitted_extra_mask_tokens': omitted}
    return report


class JEPAFrontEncoder(nn.Module):
    def __init__(self, checkpoint_path):
        super().__init__()
        encoder, predictor = make_jepa_networks()
        del predictor
        saved = torch.load(checkpoint_path, map_location='cpu', weights_only=False, mmap=True)
        encoder.load_state_dict(saved['encoder'], strict=True)
        self.encoder = encoder
        self.projection = nn.Sequential(nn.Linear(1024, 256), nn.LayerNorm(256), nn.GELU())

    def forward(self, images, _scene_tokens):
        assert images.shape[1:] == (2, 3, 256, 512)
        normalized = normalize_imagenet(images).permute(0, 2, 1, 3, 4)
        with torch.autocast(images.device.type, dtype=torch.bfloat16):
            tokens = self.encoder([normalized])[0]
        tokens = functional.adaptive_avg_pool1d(tokens.float().transpose(1, 2), 16).transpose(1, 2)
        return self.projection(tokens)


class RegisterFrontEncoder(nn.Module):
    def __init__(self, configuration):
        super().__init__()
        from navsim.agents.drivoR.layers.image_encoder.dinov2_lora import ImgEncoder
        config = copy.deepcopy(configuration.image_backbone)
        config.image_size = [518, 266]  # Preserve every512x256pixel; boundary padding only.
        config.num_scene_tokens = 16
        config.tf_d_model = 256
        config.model_weights = str(WEIGHTS / 'dinov2_vits_reg4.safetensors')
        assert Path(config.model_weights).exists()
        self.encoder = ImgEncoder(config)
        self.encoder.use_grid_mask = False  # Same unmasked observedRGB for allplanningconditions.
        self.temporal_embeddings = nn.Parameter(torch.zeros(2, 1, self.encoder.num_features))
        self.fusion = nn.Sequential(nn.Linear(512, 256), nn.LayerNorm(256), nn.GELU())

    def forward(self, images, scene_tokens):
        assert images.shape[1:] == (2, 3, 256, 512)
        normalized = functional.pad(normalize_imagenet(images), (0, 6, 0, 10))
        seeds = scene_tokens.expand(-1, 2, -1, -1) + self.temporal_embeddings[None]
        with torch.autocast(images.device.type, dtype=torch.bfloat16):
            encoded = self.encoder(normalized, seeds).reshape(len(images), 2, 16, 256)
        return self.fusion(encoded.float().transpose(1, 2).flatten(2))


class ParticleFrontEncoder(nn.Module):
    def __init__(self, checkpoint_path=None):
        super().__init__()
        self.world_model, self.lpwm_configuration = load_rectangular_lpwm('cpu', foreground_particles=16)
        if checkpoint_path is not None:
            saved = torch.load(checkpoint_path, map_location='cpu', weights_only=False, mmap=True)
            self.world_model.load_state_dict(saved['model'], strict=True)
        self.world_model.requires_grad_(True)
        self.context_dimension = self.world_model.context_dim
        self.command_modulation = nn.Sequential(nn.Linear(4, 64), nn.SiLU(), nn.Linear(64, 64))
        nn.init.zeros_(self.command_modulation[-1].weight)
        nn.init.zeros_(self.command_modulation[-1].bias)
        self.projection = nn.Sequential(nn.Linear(9 * (14 + 2*self.context_dimension), 256),
                                        nn.LayerNorm(256), nn.GELU())
        self.command = None
        self.record_particles = False
        self.latest_attributes = None
        self.future_repeat_current = False

    def attributes(self, states, presence_name, contexts):
        particle_count = states['z'].shape[-2]
        base = torch.cat((states['z'], states['z_scale'].sigmoid(), states[presence_name],
            states['z_depth'].tanh(), states['z_features'].tanh(),
            states['z_bg_features'][:, :, None].expand(-1, -1, particle_count, -1).tanh()), -1)
        assert base.shape[-1] == 14
        local_context = contexts[:, :, :particle_count]
        background_context = contexts[:, :, particle_count:particle_count+1].expand(-1, -1, particle_count, -1)
        return torch.cat((base, local_context, background_context), -1)

    def encode_particles(self, images, command):
        cnn = self.world_model.encoder_module.particle_enc.particle_attribute_enc.cnn
        cnn.rectangular_command_modulation = self.command_modulation(command.float()).repeat_interleave(2*16, 0)
        try:
            observed = images.float()
            if self.world_model.normalize_rgb:
                observed = observed * 2 - 1
            encoded = self.world_model.encode_all(observed, deterministic=True)
        finally:
            cnn.rectangular_command_modulation = None
        predicted = self.world_model.dyn_module.sample(encoded['z'], encoded['z_scale'],
            encoded['obj_on'], encoded['z_depth'], encoded['z_features'], encoded['z_bg_features'],
            z_context=encoded['z_context'][:, 1:].contiguous(), z_score=encoded['z_score'],
            steps=8, deterministic=True, return_context_posterior=False)
        current = self.attributes(encoded, 'obj_on', encoded['z_context'])[:, -1:]
        future_states = {key: value[:, -8:] for key, value in predicted.items()
                         if torch.is_tensor(value) and value.ndim >= 3}
        future = self.attributes(future_states, 'z_obj_on', future_states['z_context'])
        if self.future_repeat_current:
            future = current.expand(-1, 8, -1, -1)
        return torch.cat((current, future), 1)

    def forward(self, images, _scene_tokens):
        assert images.shape[1:] == (2, 3, 256, 512)
        assert self.command is not None
        with torch.autocast(images.device.type, enabled=False):
            if self.training and torch.is_grad_enabled():
                attributes = checkpoint(self.encode_particles, images, self.command, use_reentrant=False)
            else:
                attributes = self.encode_particles(images, self.command)
        if self.record_particles:
            self.latest_attributes = attributes.detach().float().cpu()
        return self.projection(attributes.transpose(1, 2).flatten(2))


class CommonPlannerModel(nn.Module):
    def __init__(self, kind, stage1_checkpoint=None):
        super().__init__()
        config, _ = official_configuration('navsim_v1')
        constructor_config = copy.deepcopy(config)
        for name in ('cam_f0', 'cam_b0', 'cam_l0', 'cam_l1', 'cam_l2', 'cam_r0', 'cam_r1', 'cam_r2'):
            constructor_config[name] = []
        # Shared downstream initialization is independent of backbone construction.
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(4701)
            self.planner = DrivoRModel(constructor_config)
        self.planner._config = config
        self.planner.num_cams = 1
        self.kind = kind
        if kind == 'jepa':
            self.planner.image_backbone = JEPAFrontEncoder(stage1_checkpoint)
            self.planner.register_buffer('scene_embeds', torch.zeros(1, 1, 16, 1))
        elif kind == 'drivor':
            self.planner.image_backbone = RegisterFrontEncoder(config)
            self.planner.scene_embeds = nn.Parameter(torch.randn(1, 1, 16, 384) * 1e-6)
        else:
            assert kind in ('lpwm_sequential', 'lpwm_joint')
            self.planner.image_backbone = ParticleFrontEncoder(stage1_checkpoint)
            self.planner.register_buffer('scene_embeds', torch.zeros(1, 1, 16, 1))

    @property
    def backbone(self):
        return self.planner.image_backbone

    def forward(self, features):
        if self.kind.startswith('lpwm'):
            self.backbone.command = features['ego_status'][:, -1, 7:11]
        try:
            return self.planner(features)
        finally:
            if self.kind.startswith('lpwm'):
                self.backbone.command = None

    def optimizer_groups(self):
        if self.kind.startswith('lpwm'):
            native = list(self.backbone.world_model.parameters())
        else:
            native = list(self.backbone.encoder.parameters())
        native_ids = {id(parameter) for parameter in native}
        return [{'params': [parameter for parameter in native if parameter.requires_grad],
                 'lr': 1e-4 if self.kind == 'drivor' else 1e-5, 'name':'backbone'},
                {'params': [parameter for parameter in self.parameters()
                            if parameter.requires_grad and id(parameter) not in native_ids],
                 'lr':1e-4, 'name':'planner_and_input_interface'}]
