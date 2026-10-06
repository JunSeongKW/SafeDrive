"""CPU audit of effective LoRA weight changes and the training warmup schedule."""
import argparse
import hashlib
import json
import math
from pathlib import Path

import torch


def file_digest(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024**2), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main(arguments):
    torch.set_num_threads(2)
    assert not arguments.output.exists(), "Preserve existing audit evidence"
    configuration = json.loads(arguments.configuration.read_text())
    checkpoint = torch.load(arguments.checkpoint, map_location="cpu", weights_only=False)
    tensors = checkpoint["model"]
    adapters, seen_storage = [], set()
    for name, output_projection in tensors.items():
        if not name.endswith(".lora_output_projection"):
            continue
        module_name = name.removesuffix(".lora_output_projection")
        input_projection = tensors[module_name + ".lora_input_projection"]
        storage_key = (output_projection.data_ptr(), input_projection.data_ptr(), tuple(output_projection.shape))
        if storage_key in seen_storage:
            continue
        seen_storage.add(storage_key)
        weight_keys = [module_name + ".pretrained_linear.weight", module_name + ".pretrained_convolution.weight"]
        original_weight = tensors[next(key for key in weight_keys if key in tensors)]
        # All adapters in this registered condition have alpha/rank = 1.
        assert configuration["lora"]["all_scalings"] == 1.0
        effective_delta = output_projection.float() @ input_projection.float()
        original_norm = float(original_weight.float().norm())
        delta_norm = float(effective_delta.norm())
        adapters.append({"module": module_name, "rank": input_projection.shape[0],
                         "original_shape": list(original_weight.shape), "original_weight_frobenius_norm": original_norm,
                         "effective_delta_frobenius_norm": delta_norm,
                         "relative_weight_change_percent": 100 * delta_norm / original_norm if original_norm > 0 else None,
                         "nonzero_output_elements": int(torch.count_nonzero(output_projection)),
                         "output_elements": output_projection.numel()})
    assert len(adapters) == 273, len(adapters)
    nonzero = sum(row["effective_delta_frobenius_norm"] > 0 for row in adapters)
    finite_ratios = torch.tensor([row["relative_weight_change_percent"] for row in adapters if row["relative_weight_change_percent"] is not None], dtype=torch.float64)
    geometry = [row for row in adapters if ".particle_attribute_enc." in row["module"]
                and any(f".{head}." in row["module"] for head in ("xy_head", "scale_xy_head", "obj_on_head"))]
    assert len(geometry) == 6
    planned_steps = math.ceil(configuration["scheduler_dataset_size"] / configuration["effective_batch"]) * configuration["epochs"]
    warmup_steps = int(planned_steps * .1)
    records = [json.loads(line) for line in arguments.training_log.read_text().splitlines() if line.strip()]
    snapshot_updates = checkpoint["completed_updates"]
    selected_logs = [{key: row[key] for key in ("completed_updates", "learning_rates", "gradient_norms", "trajectory_loss", "final_score_loss")}
                     for row in records if row["completed_updates"] <= snapshot_updates and row["gradient_norms"] is not None]
    used_learning_rates = [2e-10] + [row["learning_rates"][0] for row in records if row["completed_updates"] < snapshot_updates]
    report = {"checkpoint_sha256": file_digest(arguments.checkpoint), "script_sha256": file_digest(Path(__file__)),
              "completed_updates": snapshot_updates, "unique_adapter_count": len(adapters), "nonzero_effective_adapter_updates": nonzero,
              "all_adapter_relative_weight_change_percent": {"median": float(finite_ratios.median()), "maximum": float(finite_ratios.max()),
                                                             "minimum": float(finite_ratios.min())},
              "geometry_head_adapters": geometry,
              "warmup": {"steps": warmup_steps, "epochs": warmup_steps / 1614, "peak_learning_rate": configuration["lpwm_lr"],
                         "after_snapshot_learning_rate": checkpoint["scheduler"]["_last_lr"][0],
                         "after_snapshot_fraction_of_peak": checkpoint["scheduler"]["_last_lr"][0] / configuration["lpwm_lr"],
                         "mean_used_learning_rate_through_snapshot": sum(used_learning_rates) / len(used_learning_rates),
                         "epoch1_fraction_of_peak": 1e-6 + (1 - 1e-6) * 1614 / warmup_steps},
              "gradient_log_samples": selected_logs, "all_adapters": adapters,
              "interpretation": "Weight norm ratios and gradient norms are diagnostics, not activation contribution or planning benefit; different groups are not directly comparable by raw gradient norm.",
              "training_or_gpu_modified": False}
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({key: report[key] for key in ("completed_updates", "unique_adapter_count", "nonzero_effective_adapter_updates", "all_adapter_relative_weight_change_percent", "geometry_head_adapters", "warmup")}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--configuration", type=Path, required=True)
    parser.add_argument("--training-log", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    main(parser.parse_args())
