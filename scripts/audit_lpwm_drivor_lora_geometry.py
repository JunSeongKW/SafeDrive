"""Inspect a saved checkpoint without optimizer steps or changes to main training."""
import contextlib
import hashlib
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
from planning_aware_future_prediction.object_centric.lpwm_drivor_lora import LPWMDrivoRLoRAModel


def digest(path):
    checksum = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024**2), b""):
            checksum.update(chunk)
    return checksum.hexdigest()


def card_used_bytes():
    gpu = os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")[0]
    return int(subprocess.check_output(["nvidia-smi", "--id=" + gpu,
        "--query-gpu=memory.used", "--format=csv,noheader,nounits"], text=True).strip()) * 1024**2


def main():
    torch.set_num_threads(2)
    torch.manual_seed(2)
    assert card_used_bytes() < 39_000_000_000, "Main training has memory priority"
    torch.cuda.set_per_process_memory_fraction(4 * 1024**3 / torch.cuda.get_device_properties(0).total_memory)
    monitor = ROOT / "outputs/lpwm_drivor_representation_monitor_v1"
    output = ROOT / "results/lpwm_drivor_lora_geometry_audit_v1"
    assert not (output / "report.json").exists(), "Preserve completed evidence"
    output.mkdir(parents=True, exist_ok=True)
    configuration = json.loads((ROOT / "configs/lpwm_drivor_lora/navsim_v1.json").read_text())
    panel = json.loads((monitor / "panel.json").read_text())
    records = [next(record for record in panel["records"] if record["scene_type"] == scenario)
               for scenario in ("straight", "left_turn", "right_turn")]
    features = []
    for record in records:
        cache = Path(record["cache_directory"])
        image = np.array(np.load(cache / "images.npy", mmap_mode="r")[record["cache_row"]])
        status = np.array(np.load(cache / "ego.npy", mmap_mode="r")[record["cache_row"]])
        features.append({"image": torch.from_numpy(image).cuda().permute(0, 3, 1, 2)[None].float() / 255,
                         "ego_status": torch.from_numpy(status).cuda()[None, None]})
    model = LPWMDrivoRLoRAModel(ROOT / configuration["public_checkpoint"]).cuda().eval()
    model.particle_encoder.record_particles = True
    frozen_digest = model.frozen_native_digest()
    initial_parameters = {name: [parameter.detach().cpu().clone() for parameter in parameters]
                          for name, parameters in model.gradient_groups().items()}

    def infer():
        attributes = []
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            for inputs in features:
                assert card_used_bytes() < 43_000_000_000
                model(inputs)
                attributes.append(model.particle_encoder.latest_particle_attributes[0].numpy().copy())
        return np.stack(attributes)

    initial_attributes = infer()
    checkpoint_path = monitor / "checkpoint_snapshot.pt"
    checkpoint_hash = digest(checkpoint_path)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model"], strict=True)
    completed_updates = checkpoint["completed_updates"]
    assert completed_updates == 100, "This audit refers to the saved 100-update visualization"
    assert frozen_digest == checkpoint["frozen_native_sha256"] == model.frozen_native_digest()
    del checkpoint
    parameter_changes = {}
    for name, parameters in model.gradient_groups().items():
        differences = [parameter.detach().cpu() - previous for parameter, previous in zip(parameters, initial_parameters[name])]
        parameter_changes[name] = {"parameter_change_l2": float(sum(value.double().square().sum() for value in differences)**.5),
                                  "changed_tensors": sum(bool(torch.count_nonzero(value)) for value in differences),
                                  "total_tensors": len(differences)}
        assert parameter_changes[name]["parameter_change_l2"] > 0, name
    del initial_parameters
    trained_attributes = infer()
    variants = {"trained": trained_attributes}
    for name, disable_lora, disable_film in (("lora_disabled", True, False), ("film_disabled", False, True), ("both_disabled", True, True)):
        with contextlib.ExitStack() as replacements:
            if disable_lora:
                for adapters in model._adapted_regions.values():
                    for adapter in adapters:
                        replacements.enter_context(patch.object(adapter, "forward", adapter.pretrained_linear.forward))
            if disable_film:
                replacements.enter_context(patch.object(model.particle_encoder.command_modulation, "forward",
                    side_effect=lambda command: command.new_zeros((len(command), 64))))
            variants[name] = infer()

    def compare(before, after):
        shifts = np.linalg.norm((after[:, :, 0, :, :2] - before[:, :, 0, :, :2]) * 64, axis=-1)
        return {"current_xy_bitwise_equal": bool(np.array_equal(before[:, :, 0, :, :2], after[:, :, 0, :, :2])),
                "current_center_mean_shift_px": float(shifts.mean()), "current_center_max_shift_px": float(shifts.max()),
                "current_appearance_rms_change": float(np.sqrt(np.square(after[:, :, 0, :, 6:10] - before[:, :, 0, :, 6:10]).mean())),
                "future_attribute_rms_change": float(np.sqrt(np.square(after[:, :, 1:] - before[:, :, 1:]).mean())),
                "future_xy_rms_change_px": float(np.sqrt(np.square((after[:, :, 1:, :, :2] - before[:, :, 1:, :, :2])*64).mean()))}

    comparisons = {"initial_to_trained": compare(initial_attributes, trained_attributes)}
    comparisons.update({"trained_to_" + name: compare(trained_attributes, values) for name, values in variants.items() if name != "trained"})
    comparisons["initial_to_film_disabled"] = compare(initial_attributes, variants["film_disabled"])
    comparisons["initial_to_both_disabled"] = compare(initial_attributes, variants["both_disabled"])
    assert comparisons["trained_to_lora_disabled"]["current_xy_bitwise_equal"]
    assert comparisons["initial_to_film_disabled"]["current_xy_bitwise_equal"]
    assert np.array_equal(initial_attributes, variants["both_disabled"])

    # A randomized coordinate projection tests upstream parameter connectivity.
    # This is a graph audit, not a planning objective and not an optimizer step.
    encoder = model.particle_encoder
    encoder.active_command = features[0]["ego_status"][:, -1, 7:11]
    try:
        rgb = features[0]["image"][:, :1].float()
        if encoder.world_model.normalize_rgb:
            rgb = 2 * rgb - 1
        encoded = encoder.world_model.encoder_module(rgb, deterministic=True)
    finally:
        encoder.active_command = None
    parameters = []
    group_slices = {}
    for name, group in model.gradient_groups().items():
        if name.endswith("_lora") or name == "encoder_command":
            start = len(parameters)
            parameters.extend(group)
            group_slices[name] = slice(start, len(parameters))
    probe = torch.linspace(-1, 1, encoded["z"].numel(), device="cuda").reshape_as(encoded["z"])
    gradients = torch.autograd.grad((encoded["z"] * probe).sum(), parameters, allow_unused=True)
    coordinate_routes = {}
    for name, indices in group_slices.items():
        group = gradients[indices]
        coordinate_routes[name] = {"connected_tensors": sum(value is not None for value in group), "total_tensors": len(group),
            "gradient_l2": float(sum(value.detach().double().square().sum() for value in group if value is not None)**.5)}
        if name.endswith("_lora"):
            assert coordinate_routes[name]["connected_tensors"] == 0
        else:
            assert coordinate_routes[name]["gradient_l2"] > 0
    del gradients, encoded
    model.assert_native_frozen()
    assert model.frozen_native_digest() == frozen_digest
    assert digest(checkpoint_path) == checkpoint_hash
    main_gradients = {}
    for rank in (0, 1):
        history = ROOT / f"outputs/lpwm_drivor_lora_v1/navsim_v1/rank{rank}_training.jsonl"
        row = next(json.loads(line) for line in history.read_text().splitlines() if json.loads(line)["completed_updates"] == completed_updates)
        main_gradients[str(rank)] = {"gradient_norms": row["gradient_norms"], "loss": row["loss"], "learning_rates": row["learning_rates"]}
    report = {"passed": True, "checkpoint_updates": completed_updates, "checkpoint_sha256": checkpoint_hash,
        "source_sha256": digest(Path(__file__)), "optimizer_steps": 0, "native_weights_unchanged": True,
        "frozen_native_sha256": frozen_digest, "scene_tokens": [record["token"] for record in records],
        "scope": "Three training-panel scenes for functional path diagnosis, not planning improvement or independent validation",
        "parameter_changes": parameter_changes, "comparisons": comparisons,
        "coordinate_parameter_connectivity": coordinate_routes, "actual_main_training_gradients": main_gradients,
        "peak_diagnostic_reserved_bytes": torch.cuda.max_memory_reserved(), "total_card_used_bytes": card_used_bytes()}
    assert report["total_card_used_bytes"] < 48_000_000_000
    (output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
