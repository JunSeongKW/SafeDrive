"""Native rectangular LPWM adaptation with the published latent dimensions.

The full RGB image reaches the prior and background CNNs. Foreground glimpses
are sampled at 64x128 for a 256x512 input. Adaptive pooling occurs AFTER the
glimpse/background CNNs, preserving the published fully connected heads. The
decoder upsamples its latent feature maps BEFORE its convolutional rendering.
No full image is resized to 128x128 and no public source file is modified.
"""
from contextlib import redirect_stdout
import io
from types import MethodType

import torch
from torch import nn
from torch.nn import functional as functional
from torch.utils.checkpoint import checkpoint

from .lpwm_bridge import load_official_lpwm


class RectangularImagePatcher(nn.Module):
    def __init__(self, image_height, image_width, grid_rows=8, grid_columns=8):
        super().__init__()
        assert image_height % grid_rows == image_width % grid_columns == 0
        self.image_shape = (image_height, image_width)
        self.patch_shape = (image_height // grid_rows, image_width // grid_columns)
        self.grid_shape = (grid_rows, grid_columns)

    def get_patch_location_idx(self):
        rows = torch.arange(self.grid_shape[0]) * self.patch_shape[0]
        columns = torch.arange(self.grid_shape[1]) * self.patch_shape[1]
        return torch.stack(torch.meshgrid(rows, columns, indexing="ij"), -1).reshape(-1, 2)

    def get_patch_centers(self):
        return self.get_patch_location_idx() + torch.tensor(self.patch_shape) // 2

    def img_to_patches(self, images):
        assert tuple(images.shape[-2:]) == self.image_shape
        patch_height, patch_width = self.patch_shape
        patches = images.unfold(2, patch_height, patch_height).unfold(3, patch_width, patch_width)
        return patches.contiguous().reshape(images.shape[0], images.shape[1], -1, patch_height, patch_width)

    def patches_to_img(self, patches):
        return patches.reshape(*patches.shape[:2], *self.grid_shape, *self.patch_shape).permute(
            0, 1, 2, 4, 3, 5).reshape(*patches.shape[:2], *self.image_shape)

    def forward(self, images, patches=True):
        return self.img_to_patches(images) if patches else self.patches_to_img(images)


def _global_prior_coordinates(prior, local_coordinates):
    origins = prior.patcher.get_patch_location_idx().to(local_coordinates)
    patch_extent = local_coordinates.new_tensor(prior.patcher.patch_shape) - 1
    image_extent = local_coordinates.new_tensor(prior.patcher.image_shape) - 1
    return ((local_coordinates + 1) * 0.5 * patch_extent + origins[:, None]) / image_extent * 2 - 1


def _prior_patch_centers(prior):
    return prior.patcher.get_patch_centers() / (torch.tensor(prior.patcher.image_shape) - 1)


def _encode_rectangular_prior(prior, images, filtering_heuristic="none", k=None):
    assert filtering_heuristic == "none", "This condition preserves every generated patch-origin particle"
    patches = prior.patcher.img_to_patches(images).permute(0, 2, 1, 3, 4)
    patches = patches.reshape(-1, images.shape[1], *prior.patcher.patch_shape)
    heatmaps = _run_cnn_chunks(prior.enc, patches, prior.rectangular_chunk_size)
    local_positions, variances = prior.ssm(heatmaps, probs=False, variance=True)
    local_positions = local_positions.reshape(len(images), -1, prior.n_kp, 2)
    positions = _global_prior_coordinates(prior, local_positions)
    return positions.reshape(len(images), -1, 2), variances.reshape(len(images), -1, 3)


def _run_cnn_chunks(convolution, images, chunk_size, pooled_shape=None):
    modulation = getattr(convolution, "rectangular_command_modulation", None)
    if modulation is not None:
        assert len(modulation) == len(images)
    def encode(chunk, command_modulation=None):
        handle = None
        if command_modulation is not None:
            scale, shift = command_modulation.chunk(2, -1)
            def condition(_module, _arguments, feature_maps):
                return feature_maps * (1 + 0.1 * scale.tanh()[..., None, None]) + 0.1 * shift.tanh()[..., None, None]
            handle = convolution.conv_in.register_forward_hook(condition)
        try:
            features = convolution(chunk)
        finally:
            if handle is not None:
                handle.remove()
        if isinstance(features, tuple):
            features = features[1]
        if pooled_shape is not None:
            features = functional.adaptive_avg_pool2d(features, pooled_shape)
        return features
    chunks = []
    for chunk_index, chunk in enumerate(images.split(chunk_size)):
        command_chunk = None if modulation is None else modulation[chunk_index * chunk_size:chunk_index * chunk_size + len(chunk)]
        if torch.is_grad_enabled() and convolution.training:
            chunks.append(checkpoint(encode, chunk, command_chunk, use_reentrant=False))
        else:
            chunks.append(encode(chunk, command_chunk))
    return torch.cat(chunks, 0)


def crop_particle_glimpses(images, positions, normalized_scales, glimpse_shape):
    """Differentiable per-particle sampling without copying the whole RGB image 64 times."""
    image_count, particle_count = positions.shape[:2]
    glimpse_height, glimpse_width = glimpse_shape
    affine = positions.new_zeros(image_count, particle_count, 2, 3)
    affine[..., 0, 0] = normalized_scales[..., 1]
    affine[..., 1, 1] = normalized_scales[..., 0]
    affine[..., 0, 2] = positions[..., 1]
    affine[..., 1, 2] = positions[..., 0]
    grid = functional.affine_grid(affine.flatten(0, 1),
        (image_count * particle_count, images.shape[1], glimpse_height, glimpse_width), align_corners=False)
    grid = grid.reshape(image_count, particle_count * glimpse_height, glimpse_width, 2)
    glimpses = functional.grid_sample(images, grid, mode="bilinear", padding_mode="border", align_corners=False)
    return glimpses.reshape(image_count, images.shape[1], particle_count, glimpse_height,
                            glimpse_width).permute(0, 2, 1, 3, 4).reshape(
                                image_count * particle_count, images.shape[1], glimpse_height, glimpse_width)


def _attribute_forward(encoder, images, positions, z_scale=None, timesteps=None, deterministic=False):
    image_count, particle_count = positions.shape[:2]
    normalized_scales = torch.full_like(positions, encoder.rectangular_anchor_ratio) if z_scale is None else z_scale.sigmoid()
    glimpses = crop_particle_glimpses(images, positions, normalized_scales, encoder.rectangular_glimpse_shape)
    features = _run_cnn_chunks(encoder.cnn, glimpses, encoder.rectangular_chunk_size,
                              encoder.rectangular_pooled_shape).reshape(image_count, particle_count, -1)
    features = encoder.backbone(features)
    if timesteps is not None and encoder.temp_embed is not None:
        features = (features.reshape(-1, timesteps, *features.shape[1:]) + encoder.temp_embed[:, :timesteps]).reshape_as(features)
    logits = encoder.obj_on_head(features).reshape(image_count, particle_count, 1)
    alpha_gate = logits.sigmoid()
    alpha = ((1 - alpha_gate) * encoder.obj_on_min + alpha_gate * encoder.obj_on_max).exp()
    beta_gate = 1 - alpha_gate
    beta = ((1 - beta_gate) * encoder.obj_on_min + beta_gate * encoder.obj_on_max).exp()
    distribution = torch.distributions.Beta(alpha, beta)
    presence = distribution.mean if deterministic else distribution.rsample()
    position_mean, position_log_variance = encoder.xy_head(features).chunk(2, -1)
    assert encoder.with_scale and encoder.kp_activation == "tanh"
    scale_mean, scale_log_variance = encoder.scale_xy_head(features).chunk(2, -1)
    position_mean = encoder.max_offset * position_mean.tanh()
    depth_mean = depth_log_variance = None
    if encoder.with_depth:
        depth_mean, depth_log_variance = encoder.depth_head(features).chunk(2, -1)
    return {"mu": position_mean, "logvar": position_log_variance, "mu_scale": scale_mean,
        "logvar_scale": scale_log_variance, "lobj_on_a": logits, "lobj_on_b": logits,
        "obj_on": logits, "mu_depth": depth_mean, "logvar_depth": depth_log_variance,
        "obj_on_a": alpha, "obj_on_b": beta, "z_obj_on": presence, "mu_obj_on": distribution.mean}


def _features_forward(encoder, images, positions, z_scale=None, timesteps=None, obj_on=None):
    image_count, particle_count = positions.shape[:2]
    scales = torch.full_like(positions, encoder.rectangular_anchor_ratio) if z_scale is None else z_scale.sigmoid()
    glimpses = crop_particle_glimpses(images, positions, scales, encoder.rectangular_glimpse_shape)
    features = _run_cnn_chunks(encoder.cnn, glimpses, encoder.rectangular_chunk_size, encoder.rectangular_pooled_shape)
    if obj_on is not None:
        features = features * obj_on.reshape(-1, 1, 1, 1)
    assert encoder.temp_embed is None, "Published Sketchy configuration has no glimpse time embedding"
    features = encoder.to_latent(features).reshape(image_count, particle_count, -1)
    mean = encoder.to_mu(features)
    log_variance = encoder.to_logvar(features) if encoder.output_logvar else None
    return {"mu_features": mean, "logvar_features": log_variance,
            "cropped_objects": glimpses.reshape(image_count, particle_count, *glimpses.shape[1:])}


def _background_cnn_hook(module, arguments, features):
    return functional.adaptive_avg_pool2d(features, module.rectangular_pooled_shape)


def _background_decoder_input(module, arguments):
    return (functional.interpolate(arguments[0], size=module.rectangular_seed_shape,
                                   mode="bilinear", align_corners=False),)


def _background_mask(encoder, positions, presence, mask_size, z_scale=None, detach_grad=True):
    height, width = encoder.rectangular_image_shape
    with torch.no_grad():
        rows = (torch.arange(height, device=positions.device) + 0.5) / height * 2 - 1
        columns = (torch.arange(width, device=positions.device) + 0.5) / width * 2 - 1
        scales = torch.full_like(positions, encoder.anchor_s) if z_scale is None else z_scale.sigmoid()
        inside_rows = (rows[None, None, :, None] - positions[..., 0, None, None]).abs() < scales[..., 0, None, None]
        inside_columns = (columns[None, None, None, :] - positions[..., 1, None, None]).abs() < scales[..., 1, None, None]
        visible = presence.reshape(*positions.shape[:2], 1, 1) > 0.2
        return (~(inside_rows & inside_columns & visible).any(1, keepdim=True)).to(positions.dtype)


def _translate_rectangular(decoder, positions, patches, scale=None, translation=None, scale_normalized=False):
    from utils.util_func import spatial_transform
    image_count, particle_count, channels = patches.shape[:3]
    if scale is None:
        scales = positions.new_tensor(patches.shape[-2:]) / positions.new_tensor(decoder.rectangular_image_shape)
        scales = scales.expand_as(positions)
    else:
        scales = scale if scale_normalized else scale.sigmoid()
    output_shape = (image_count * particle_count, channels, *decoder.rectangular_image_shape)
    rendered = spatial_transform(patches.flatten(0, 1), positions.reshape(-1, 2),
                                 scales.reshape(-1, 2), output_shape, inverse=True)
    return rendered.reshape(image_count, particle_count, channels, *decoder.rectangular_image_shape)


PARTICLE_GRIDS = {8: (2, 4), 16: (4, 4), 32: (4, 8), 64: (8, 8)}


def _reduce_particle_embeddings(tensor, particle_axis, grid_shape):
    values = tensor.movedim(particle_axis, 0)
    assert len(values) in (64, 65)
    grid_rows, grid_columns = grid_shape
    foreground = values[:64].reshape(grid_rows, 8 // grid_rows, grid_columns,
                                     8 // grid_columns, *values.shape[1:]).mean((1, 3))
    foreground = foreground.reshape(grid_rows * grid_columns, *values.shape[1:])
    return torch.cat((foreground, values[64:]), 0).movedim(0, particle_axis).contiguous()


def _reduce_particle_count(model, foreground_particles):
    """Average embeddings within spatial grid groups; preserve shared context aliases."""
    for module in model.modules():
        for attribute in ("num_patches", "n_kp_total", "n_kp_prior", "n_kp_enc", "n_particles"):
            if getattr(module, attribute, None) == 64:
                setattr(module, attribute, foreground_particles)
        if hasattr(module, "n_kp_dec"):
            module.n_kp_dec = foreground_particles
        for parameter_name, axis in (("particle_embeddings", 2), ("pos_p_embeddings", None)):
            parameter = module._parameters.get(parameter_name)
            if parameter is None:
                continue
            if axis is None:
                axes = [index for index, size in enumerate(parameter.shape) if size == 65]
                assert len(axes) == 1
                axis = axes[0]
            if parameter.shape[axis] in (64, 65):
                module._parameters[parameter_name] = nn.Parameter(_reduce_particle_embeddings(
                    parameter.detach(), axis, PARTICLE_GRIDS[foreground_particles]))
    model.decoder_module.n_kp_enc = foreground_particles


def load_rectangular_lpwm(device="cpu", checkpoint_path=None, image_height=256, image_width=512,
                          sequence_frames=8, cnn_chunk_size=32, foreground_particles=32):
    """Load published weights, then adapt spatial computation without resizing RGB."""
    with redirect_stdout(io.StringIO()):
        model, configuration = load_official_lpwm("cpu")
    model.timestep_horizon = sequence_frames - 1
    assert model.n_kp_enc == 64 and model.anchor_s == 0.25
    assert foreground_particles in PARTICLE_GRIDS
    if foreground_particles != 64:
        _reduce_particle_count(model, foreground_particles)
    for module in (model, model.encoder_module, model.encoder_module.particle_enc):
        module.n_kp_dec = foreground_particles
    model.decoder_module.n_kp_enc = foreground_particles
    image_shape = (image_height, image_width)
    encoder = model.encoder_module
    prior = encoder.prior_encoder
    grid_rows, grid_columns = PARTICLE_GRIDS[foreground_particles]
    prior.patcher = RectangularImagePatcher(*image_shape, grid_rows=grid_rows, grid_columns=grid_columns)
    prior.rectangular_chunk_size = cnn_chunk_size
    prior.get_global_kp = MethodType(_global_prior_coordinates, prior)
    prior.get_patch_centers = MethodType(_prior_patch_centers, prior)
    prior.encode_prior = MethodType(_encode_rectangular_prior, prior)
    centers = prior.get_patch_centers() * 2 - 1
    with torch.no_grad():
        for module in model.modules():
            for name, buffer in list(module._buffers.items()):
                if name not in ("patch_centers", "particle_anchors", "particles_anchor") or buffer is None:
                    continue
                assert buffer.shape[-1] == 2
                desired = centers
                if buffer.shape[-2] == 65:
                    desired = torch.cat((centers, centers.new_zeros(1, 2)), 0)
                module._buffers[name] = desired.reshape(*buffer.shape[:-2], len(desired), 2).clone()
    for module in (encoder.particle_enc.particle_attribute_enc, encoder.particle_enc.particle_features_enc):
        module.rectangular_anchor_ratio = model.anchor_s
        module.rectangular_glimpse_shape = (image_height // 4, image_width // 4)
        module.rectangular_chunk_size = cnn_chunk_size
        with torch.no_grad():
            old_features = module.cnn(torch.zeros(1, 3, module.patch_size, module.patch_size))
        module.rectangular_pooled_shape = tuple(old_features.shape[-2:])
    encoder.particle_enc.particle_attribute_enc.forward = MethodType(_attribute_forward, encoder.particle_enc.particle_attribute_enc)
    encoder.particle_enc.particle_features_enc.forward = MethodType(_features_forward, encoder.particle_enc.particle_features_enc)
    encoder.rectangular_image_shape = image_shape
    encoder.get_bg_mask_from_particle_glimpses = MethodType(_background_mask, encoder)
    background_encoder = encoder.bg_encoder
    background_encoder.bg_cnn_enc.rectangular_pooled_shape = background_encoder.cnn_out_shape[-2:]
    background_encoder.bg_cnn_enc.register_forward_hook(_background_cnn_hook)
    interaction = encoder.particle_inter_enc
    if interaction is not None and interaction.use_img_input:
        interaction.ctx_cnn_enc.rectangular_pooled_shape = interaction.cnn_out_shape[-2:]
        interaction.ctx_cnn_enc.register_forward_hook(_background_cnn_hook)
    decoder = model.decoder_module
    decoder.rectangular_image_shape = image_shape
    decoder.translate_patches = MethodType(_translate_rectangular, decoder)
    background_cnn = decoder.bg_dec.cnn
    upsample_factor = 2 ** (background_cnn.num_resolutions - 1)
    background_cnn.rectangular_seed_shape = (image_height // upsample_factor, image_width // upsample_factor)
    background_cnn.register_forward_pre_hook(_background_decoder_input)
    if checkpoint_path is not None:
        saved = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        model.load_state_dict(saved.get("model", saved), strict=True)
    configuration = {**configuration, "input_height": image_height, "input_width": image_width,
        "sequence_frames": sequence_frames, "rectangular_adapter_version": 2,
        "foreground_particles": foreground_particles, "background_particles": 1,
        "prior_grid": [grid_rows, grid_columns], "prior_patch_shape": list(prior.patcher.patch_shape),
        "glimpse_shape": [image_height // 4, image_width // 4], "whole_image_downsample_to_128": False}
    return model.to(device), configuration
