"""Read-only speed audit and bounded single-scene LPWM backward probes.

This never changes the live trainer, writes model weights, or steps an optimizer.
The probe has eight future steps, but one camera, batch one and a fixed
particle vector-Jacobian product rather than the official planning loss.
Its timings must not be reported as production DDP throughput.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import statistics
import subprocess
import sys
import time

import numpy as np
import torch
from torch.utils.checkpoint import checkpoint

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from planning_aware_future_prediction.object_centric.lpwm_drivor_planning_path_lora import LPWMDrivoRPlanningPathLoRAModel


def write_json(path, content):
    path.write_text(json.dumps(content, indent=2, allow_nan=False) + "\n")


def card_bytes():
    return int(subprocess.check_output([
        "nvidia-smi", "--id=0", "--query-gpu=memory.used",
        "--format=csv,noheader,nounits"], text=True).strip()) * 1024**2


def log_audit(configuration):
    reports = []
    for rank in (0, 1):
        rows = [json.loads(line) for line in
                (ROOT / configuration["output_directory"] / f"rank{rank}_training.jsonl").read_text().splitlines()][-100:]
        timing = {key: statistics.mean(row["timing_seconds"][key] for row in rows)
                  for key in rows[-1]["timing_seconds"]}
        timing["update"] = statistics.mean(row["seconds_this_update"] for row in rows)
        timing["other_update_work"] = timing["update"] - sum(timing[key] for key in ("forward", "oracle", "backward"))
        reports.append({"rank": rank, "first_update": rows[0]["completed_updates"],
                        "last_update": rows[-1]["completed_updates"], "mean_seconds": timing,
                        "wall_seconds_per_update": (rows[-1]["elapsed_seconds"] - rows[0]["elapsed_seconds"]) / (len(rows)-1)})
    return reports


def main(arguments):
    output = arguments.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(2)
    torch.manual_seed(79)
    configuration = json.loads((ROOT / "configs/lpwm_drivor_planning_path_lora/navsim_v1.json").read_text())
    before = log_audit(configuration)
    write_json(output / "production_timing_before.json", before)
    initial_card_usage = card_bytes()
    allocation_budget = int(arguments.allocator_gib * 1024**3)
    assert initial_card_usage + allocation_budget + 1_000_000_000 < 48_000_000_000, "Live training has priority"
    torch.cuda.set_per_process_memory_fraction(allocation_budget / torch.cuda.get_device_properties(0).total_memory)
    model = LPWMDrivoRPlanningPathLoRAModel(ROOT / configuration["public_checkpoint"]).cuda()
    state = torch.load(arguments.checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(state["model"], strict=True)
    checkpoint_update = state["completed_updates"]
    native_digest = state["frozen_native_sha256"]
    del state
    assert model.frozen_native_digest() == native_digest
    encoder = model.particle_encoder
    panel = json.loads((ROOT / "outputs/lpwm_drivor_planning_path_representation_monitor_v1/panel.json").read_text())
    record = panel["records"][0]
    arrays = {name: np.array(np.load(Path(record["cache_directory"]) / (name + ".npy"), mmap_mode="r")[record["cache_row"]])
              for name in ("images", "ego")}
    images = torch.from_numpy(arrays["images"]).cuda().permute(0,3,1,2)[None].float() / 255
    command = torch.from_numpy(arrays["ego"]).cuda()[None, 7:11]
    encoder.forward_command = command
    probe = torch.randn(1, 9, 64, 14, device="cuda") / (9*64*14)
    attention_modules = [(name, module) for name, module in encoder.world_model.named_modules()
                         if type(module).__name__ in ("CausalParticleSelfAttention", "ParticleSelfAttention")]
    inventory = [{"name": name, "type": type(module).__name__, "torch_attn": module.torch_attn,
                  "positional_bias": module.positional_bias, "attention_dropout": module.attn_pdrop}
                 for name, module in attention_modules]
    assert len(attention_modules) == 21
    assert all(not module.torch_attn and not module.positional_bias for _, module in attention_modules)
    # Temporal blocks flatten particles into the batch and must always have
    # N=1. Otherwise the original temporal-only mask differs from is_causal.
    seen_shapes = Counter()
    def check_attention_shape(module, inputs):
        shape = tuple(inputs[0].shape)
        seen_shapes[(type(module).__name__, shape)] += 1
        if type(module).__name__ == "CausalParticleSelfAttention":
            assert shape[1] == 1, "Do not replace particle-major temporal masks with a token causal mask"
    handles = [module.register_forward_pre_hook(check_attention_shape) for _, module in attention_modules]
    device_high_water = []
    measurements = []
    correctness = []
    original_outputs = original_gradients = None

    def probe_backward(label, training, autocast):
        assert card_bytes() < 47_500_000_000
        model.train(training)
        # Evaluation still checkpoints to keep this read-only VJP inside its
        # allocator budget. Dropout remains off throughout the LPWM modules.
        encoder.training = True
        model.zero_grad(set_to_none=True)
        torch.cuda.reset_peak_memory_stats()
        torch.manual_seed(79)
        torch.cuda.synchronize()
        started = time.perf_counter()
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=autocast):
            tokens = checkpoint(encoder._encode_camera, images[:, 0].contiguous(), command,
                                use_reentrant=False)
            objective = (tokens.float() * probe).sum()
        torch.cuda.synchronize()
        forward_seconds = time.perf_counter() - started
        objective.backward()
        torch.cuda.synchronize()
        total_seconds = time.perf_counter() - started
        used = card_bytes()
        assert used < 48_000_000_000
        device_high_water.append(used)
        gradients = {name: torch.cat([parameter.grad.detach().float().flatten().cpu()
                                     for parameter in parameters if parameter.grad is not None])
                     for name, parameters in model.gradient_groups().items()
                     if any(parameter.grad is not None for parameter in parameters)}
        assert all(torch.isfinite(value).all() for value in gradients.values())
        model.assert_native_frozen()
        row = {"condition": label, "dropout_enabled": training, "bf16_autocast": autocast,
               "forward_seconds": forward_seconds, "forward_backward_seconds": total_seconds,
               "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
               "peak_reserved_bytes": torch.cuda.max_memory_reserved(), "whole_card_bytes": used}
        print(json.dumps(row), flush=True)
        measurements.append(row)
        write_json(output / "progress.json", measurements)
        return tokens.detach().float().cpu(), gradients

    for label, use_sdpa in (("original", False), ("sdpa", True), ("original_repeat", False)):
        for _, module in attention_modules:
            module.torch_attn = use_sdpa
        if label != "original_repeat":
            tokens, gradients = probe_backward(label + "_fp32_equivalence", False, False)
            if original_outputs is None:
                original_outputs, original_gradients = tokens, gradients
            else:
                gradient_comparison = {}
                for name, expected in original_gradients.items():
                    observed = gradients[name]
                    gradient_comparison[name] = {
                        "relative_l2_error": float((observed-expected).norm() / expected.norm().clamp_min(1e-12)),
                        "cosine_similarity": float(torch.nn.functional.cosine_similarity(observed, expected, dim=0)),
                        "finite": bool(torch.isfinite(observed).all())}
                correctness.append({"fp32_no_dropout": True,
                    "token_max_absolute_difference": float((tokens-original_outputs).abs().max()),
                    "token_relative_l2_error": float((tokens-original_outputs).norm() / original_outputs.norm()),
                    "gradient_groups": gradient_comparison})
            del tokens, gradients
        # One warm pass, followed by three timed repetitions. These include
        # training dropout and preserve all geometry/appearance/future routes.
        for repeat in range(arguments.repeats + 1):
            tokens, gradients = probe_backward(label + ("_warmup" if repeat == 0 else "_training"), True, True)
            del tokens, gradients
        model.zero_grad(set_to_none=True)
        torch.cuda.empty_cache()
    for handle in handles:
        handle.remove()
    assert model.frozen_native_digest() == native_digest
    summary = {label: {"median_forward_backward_seconds": statistics.median(row["forward_backward_seconds"] for row in measurements if row["condition"] == label + "_training"),
                       "max_allocated_bytes": max(row["peak_allocated_bytes"] for row in measurements if row["condition"] == label + "_training")}
               for label in ("original", "sdpa", "original_repeat")}
    report = {"checkpoint_update": checkpoint_update,
              "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "scene_token": record["token"], "attention_inventory": inventory,
              "actual_attention_shapes": [{"type": key[0], "shape": list(key[1]), "count": value} for key, value in seen_shapes.items()],
              "measurements": measurements, "summary": summary, "correctness": correctness,
              "maximum_whole_card_bytes": max(device_high_water),
              "native_weights_unchanged": True, "model_or_optimizer_updates": 0,
              "production_timing_before": before, "production_timing_after": log_audit(configuration),
              "camera_count": 1, "future_steps": 8, "allocator_budget_bytes": allocation_budget,
              "limitations": ["Batch-one, one-camera particle VJP, not official loss or full DDP training.",
                              "Runs alongside active GPU0 training; timing is exploratory and perturbs shared throughput.",
                              "FP32 no-dropout VJP validates numerical agreement, not bitwise equivalence in bf16 training.",
                              "SDPA preserves dropout probability but changes sampled masks and floating-point ordering.",
                              "Full batch16 DDP memory, actual loss and resumability benchmark required before adoption.",
                              "Live registered source/configuration, model and queue were not changed."]}
    write_json(output / "report.json", report)
    print("PROFILE_COMPLETE", json.dumps(summary), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--allocator-gib", type=float, default=6.0)
    main(parser.parse_args())
