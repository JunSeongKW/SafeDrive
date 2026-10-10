"""Native-weight-frozen LPWM control with the existing small-corpus planner."""
import hashlib

import torch

from .small_corpus_models import CommonPlannerModel
from .lpwm_adapter_full_finetuning import ParticleBlockWithAdapter


class SmallCorpusFrozenAdapterPlanner(CommonPlannerModel):
    """Keep native LPWM weights/buffers; adapt interaction/context/dynamics blocks."""

    def __init__(self, stage1_checkpoint, bottleneck_dimension=32):
        super().__init__('lpwm_sequential', stage1_checkpoint)
        world = self.backbone.world_model
        self.native_parameters = tuple(world.parameters())
        self.native_buffers = tuple(world.buffers())
        self.native_initial_digest = self.frozen_native_digest()
        world.requires_grad_(False)
        self.adapter_region_counts = {}
        regions = ('encoder_module.particle_inter_enc.pte', 'ctx_module.pte',
                   'dyn_module.particle_transformer')
        # Shared context aliases point at one module; install adapters once.
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(17048)
            for region in regions:
                transformer = world.get_submodule(region)
                count = 0
                for index, block in enumerate(list(transformer.blocks)):
                    transformer.blocks[index] = ParticleBlockWithAdapter(
                        block, transformer.n_embed, bottleneck_dimension)
                    count += 1
                assert count > 0, region
                self.adapter_region_counts[region] = count
        assert all(not parameter.requires_grad for parameter in self.native_parameters)
        assert not any(isinstance(module, torch.nn.modules.batchnorm._BatchNorm) for module in world.modules()), (
            'Frozen native buffers require an explicit BatchNorm policy')
        assert self.frozen_native_digest() == self.native_initial_digest

    def frozen_native_digest(self):
        checksum = hashlib.sha256()
        # CUDA conversion replaces buffer objects; read the current registered
        # buffers, not the CPU objects captured during construction.
        buffers = tuple(self.backbone.world_model.buffers())
        assert len(buffers) == len(self.native_buffers)
        for tensor in (*self.native_parameters, *buffers):
            checksum.update(tensor.detach().cpu().contiguous().numpy().tobytes())
        return checksum.hexdigest()

    def adapter_parameters(self):
        return [parameter for name, parameter in self.backbone.world_model.named_parameters()
                if '.residual_adapter.' in name]


def adapter_gradient_groups(model):
    groups = {'adapters': model.adapter_parameters(),
              'command': list(model.backbone.command_modulation.parameters()),
              'projection': list(model.backbone.projection.parameters()),
              'generator': list(model.planner.trajectory_decoder.parameters()) + list(model.planner.traj_head.parameters()),
              'scorer': list(model.planner.scorer_attention.parameters()) + list(model.planner.scorer.parameters())}
    return {name: sum(float(parameter.grad.detach().float().square().sum())
                      for parameter in parameters if parameter.grad is not None)**.5
            for name, parameters in groups.items()}
