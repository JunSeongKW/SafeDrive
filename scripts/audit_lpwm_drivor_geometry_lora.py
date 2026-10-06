"""Audit geometry LoRA with actual images and the unchanged official planning loss."""
import argparse
from contextlib import ExitStack
import json
import os
from pathlib import Path
import subprocess
import sys
from unittest.mock import patch

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from planning_aware_future_prediction.object_centric.lpwm_drivor_geometry_lora import LPWMDrivoRGeometryLoRAModel
from planning_aware_future_prediction.object_centric.lpwm_drivor_joint import make_official_loss, oracle_loss_callback
from lpwm_drivor_oracle import DrivoROracleClient
from visualize_lpwm_epoch_particle_geometry import geometry_changes


def card_bytes():
    gpu = os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")[0]
    return int(subprocess.check_output(["nvidia-smi", "--id=" + gpu,
        "--query-gpu=memory.used", "--format=csv,noheader,nounits"], text=True).strip()) * 1024**2


def run(arguments):
    torch.set_num_threads(2)
    torch.manual_seed(2)
    assert card_bytes() < 38_000_000_000
    torch.cuda.set_per_process_memory_fraction(6 * 1024**3 / torch.cuda.get_device_properties(0).total_memory)
    output = arguments.output
    output.mkdir(parents=True, exist_ok=True)
    assert not (output / "passed.json").exists(), "Preserve completed evidence"
    configuration = json.loads((ROOT / "configs/lpwm_drivor_lora/navsim_v1.json").read_text())
    manifest = ROOT / "outputs/lpwm_drivor_joint_v1/engineering_scene_cache/manifest.json"
    records = json.loads(manifest.read_text())["records"][:2]
    def values(name):
        return np.stack([np.load(Path(record["cache_directory"]) / (name + ".npy"), mmap_mode="r")[record["cache_row"]] for record in records])
    features = {"image": torch.from_numpy(values("images")).cuda().permute(0,1,4,2,3).float()/255,
                "ego_status": torch.from_numpy(values("ego")).cuda()[:, None]}
    targets = {"trajectory": torch.from_numpy(values("trajectory")).cuda(),
               "trajectory_long": torch.from_numpy(values("trajectory_long")).cuda()}
    model = LPWMDrivoRGeometryLoRAModel(ROOT / configuration["public_checkpoint"]).cuda().eval()
    model.particle_encoder.record_particles = True
    native_digest = model.frozen_native_digest()
    original_planner = torch.load(ROOT / "outputs/lpwm_drivor_lora_v1/navsim_v1/initial_official_planner.pt", map_location="cpu", weights_only=True)
    assert all(torch.equal(value.cpu(), original_planner[name]) for name, value in model.planner.state_dict().items() if not name.startswith("image_backbone."))
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        initial_proposals = model(features)["proposals"].clone()
        initial_attributes = model.particle_encoder.latest_particle_attributes.numpy().copy()
        with ExitStack() as patches:
            for adapters in model._adapted_regions.values():
                for adapter in adapters:
                    patches.enter_context(patch.object(adapter, "forward", adapter.pretrained_linear.forward))
            native_proposals = model(features)["proposals"]
            assert torch.equal(initial_proposals, native_proposals)
            assert np.array_equal(initial_attributes, model.particle_encoder.latest_particle_attributes.numpy())
    model.particle_encoder.record_particles = False
    # A coordinate-only probe verifies that the newly added upstream adapters
    # are connected to the current attributes before attention or the planner.
    encoder = model.particle_encoder
    encoder.active_command = features["ego_status"][:, -1, 7:11]
    try:
        rgb = features["image"][:, :1].float()
        if encoder.world_model.normalize_rgb:
            rgb = 2*rgb-1
        encoded = encoder.world_model.encoder_module(rgb, deterministic=True)
    finally:
        encoder.active_command = None
    coordinate_routes = {}
    for head_name, attribute in (("xy_head", "z"), ("scale_xy_head", "z_scale"), ("obj_on_head", "obj_on")):
        parameters = model.gradient_groups()["particle_" + head_name + "_lora"]
        probe = torch.linspace(-1, 1, encoded[attribute].numel(), device="cuda").reshape_as(encoded[attribute])
        gradients = torch.autograd.grad((encoded[attribute] * probe).sum(), parameters, retain_graph=True, allow_unused=True)
        norm = sum(float(value.detach().double().square().sum()) for value in gradients if value is not None)**.5
        assert all(value is not None for value in gradients) and norm > 0, head_name
        coordinate_routes[head_name] = {"connected_tensors": len(gradients), "gradient_l2": norm}
    del encoded, gradients
    optimizer = torch.optim.AdamW(model.optimizer_groups(2e-4, 2e-4), weight_decay=.01)
    criterion, official_config = make_official_loss()
    oracle = DrivoROracleClient(manifest, output, workers=2)
    history = []
    model.train()
    try:
        for update in range(2):
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                prediction = model(features)
            scores = torch.from_numpy(oracle.score([record["token"] for record in records], prediction["proposals"].detach().float().cpu().numpy())).cuda()
            with torch.autocast("cuda", dtype=torch.bfloat16):
                losses = criterion(targets, prediction, official_config, scoring_function=oracle_loss_callback(scores))
            assert torch.isfinite(losses["loss"])
            if update == 0:
                assert torch.autograd.grad(losses["final_score_loss"], prediction["proposals"], retain_graph=True, allow_unused=True)[0] is None
            losses["loss"].backward()
            gradients = {name: sum(float(parameter.grad.detach().double().square().sum()) for parameter in parameters if parameter.grad is not None)**.5
                         for name, parameters in model.gradient_groups().items()}
            assert all(np.isfinite(value) and value > 0 for value in gradients.values()), gradients
            model.assert_native_frozen()
            optimizer.step()
            history.append({"engineering_update": update+1, "loss": float(losses["loss"].detach()), "gradient_norms": gradients})
            del prediction, losses, scores
    finally:
        oracle.close()
    model.eval()
    model.particle_encoder.record_particles = True
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        model(features)
        adapted_attributes = model.particle_encoder.latest_particle_attributes.numpy().copy()
        # Remove only geometry adapters, keeping all attention adapters and FiLM.
        with ExitStack() as patches:
            for region, adapters in model._adapted_regions.items():
                if region.endswith("_head"):
                    for adapter in adapters:
                        patches.enter_context(patch.object(adapter, "forward", adapter.pretrained_linear.forward))
            model(features)
            without_geometry = model.particle_encoder.latest_particle_attributes.numpy().copy()
    specific_effect = geometry_changes(without_geometry[:, :, 0], adapted_attributes[:, :, 0])
    assert specific_effect["center_shift_input_pixels"]["maximum"] > 0
    assert specific_effect["size_axis_absolute_change_input_pixels"]["maximum"] > 0
    assert specific_effect["presence_absolute_change"]["maximum"] > 0
    assert model.frozen_native_digest() == native_digest
    assert card_bytes() < 48_000_000_000
    report = {"passed": True, "engineering_updates_discarded": True, "optimizer_steps": 2,
              "initial_zero_lora_outputs_identical": True, "initial_planner_identical": True,
              "native_weights_and_buffers_unchanged": True, "frozen_native_sha256": native_digest,
              "coordinate_routes": coordinate_routes, "history": history,
              "geometry_adapter_only_effect": specific_effect,
              "initial_to_engineering_updated_geometry": geometry_changes(initial_attributes[:, :, 0], adapted_attributes[:, :, 0]),
              "parameter_inventory": model.parameter_inventory(), "maximum_reserved_bytes": torch.cuda.max_memory_reserved(),
              "card_used_bytes": card_bytes(), "not_evidence_of_planning_improvement": True}
    (output / "passed.json").write_text(json.dumps(report, indent=2, allow_nan=False)+"\n")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/lpwm_drivor_geometry_lora_v1/engineering_audit")
    run(parser.parse_args())
