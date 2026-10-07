"""Execution-only LPWM changes, keeping parameter names and evaluation intact."""
from .lpwm_drivor_planning_path_lora import LPWMDrivoRPlanningPathLoRAModel


def validate_temporal_attention_shape(module, arguments):
    # LPWM folds each particle into the batch before temporal attention. The
    # native particle/time mask and SDPA's token causal mask agree only here.
    if module.torch_attn and arguments[0].shape[1] != 1:
        raise ValueError("SDPA temporal attention requires one particle per attention sequence")


class RuntimeOptimizedLPWMDrivoRModel(LPWMDrivoRPlanningPathLoRAModel):
    def __init__(self, public_checkpoint, benchmark="navsim_v1", *, sdpa_training=False):
        super().__init__(public_checkpoint, benchmark)
        self.sdpa_training = sdpa_training
        self.runtime_attention_modules = []
        for module in self.particle_encoder.world_model.modules():
            if type(module).__name__ not in ("ParticleSelfAttention", "CausalParticleSelfAttention"):
                continue
            if module.positional_bias or module.torch_attn:
                raise ValueError("Unexpected pretrained attention configuration")
            self.runtime_attention_modules.append(module)
            if type(module).__name__ == "CausalParticleSelfAttention":
                module.register_forward_pre_hook(validate_temporal_attention_shape)
        assert len(self.runtime_attention_modules) == 21
        self.train(self.training)

    def train(self, mode=True):
        super().train(mode)
        # Existing inference, particle diagnostics and full benchmark scripts
        # continue using the identical canonical evaluation implementation.
        for module in getattr(self, "runtime_attention_modules", ()):
            module.torch_attn = bool(mode and self.sdpa_training)
        return self

