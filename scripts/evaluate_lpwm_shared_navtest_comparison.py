"""Identical-scene official PDMS comparison of two preserved LPWM systems."""
import argparse
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
import json
import multiprocessing
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "reference_repositories/DrivoR"))
from prepare_lpwm_shared_navtest_comparison import digest, read_json, write_json


class EvaluationDeferred(Exception):
    """Release only evaluation resources so the existing training has priority."""


def verify_registration(configuration_path, configuration):
    output = ROOT / configuration["output_directory"]
    registration = read_json(output / "registration.json")
    assert digest(configuration_path) == registration["configuration_sha256"]
    for path, checksum in registration["sources"].items():
        assert digest(ROOT / path) == checksum, path
    assert digest(output / "panel.json") == registration["panel_sha256"]
    return registration


def gpu_used_bytes(gpu):
    result = subprocess.check_output(["nvidia-smi", f"--id={gpu}", "--query-gpu=memory.used",
                                      "--format=csv,noheader,nounits"], text=True)
    return int(result.strip()) * 1024**2


def check_resources(configuration):
    output = ROOT / configuration["output_directory"]
    primary = ROOT / configuration["primary_training_root"]
    adapter = ROOT / configuration["adapter_training_root"]
    if any(path.exists() for path in (output / "pause.requested", primary / "pause.requested")):
        raise EvaluationDeferred("User pause or original batch restoration in progress")
    # A new GPU process would be treated as an external allocation by the
    # currently active Adapter queue. Never start until that queue is finished.
    if configuration.get("temporary_adapter_yield"):
        handoff = read_json(output / "adapter_yield_ready.json")
        assert (adapter / "pause.requested").read_text() == handoff["pause_content"]
        assert read_json(adapter / "queue_state.json")["stage"] == "paused"
    else:
        if not (adapter / "complete.json").exists():
            raise EvaluationDeferred("Adapter training/evaluation has priority")
        restored = read_json(ROOT / configuration["restore_primary_root"] / "status.json")
        if restored["stage"] != "restored_original_primary_batch":
            raise EvaluationDeferred("Waiting for the existing batch restoration controller")
    monitor = read_json(ROOT / configuration["primary_monitor_root"] / "status.json")
    progress = read_json(primary / "navsim_v1/progress.json")
    if monitor.get("active_update") is not None or monitor.get("stage") in ("evaluating", "waiting_for_memory"):
        raise EvaluationDeferred("Scheduled particle diagnosis has priority")
    milestone = monitor.get("next_milestone")
    if milestone is not None and progress["completed_updates"] >= milestone - 2:
        raise EvaluationDeferred("Scheduled particle diagnosis boundary is imminent")
    used = gpu_used_bytes(configuration["gpu"])
    if used >= configuration["stop_card_bytes"]:
        raise EvaluationDeferred("Total card memory guard")
    return used


def build_model(configuration, system, checkpoint_override=None):
    import torch
    if system == "primary":
        from planning_aware_future_prediction.object_centric.lpwm_drivor_planning_path_lora import LPWMDrivoRPlanningPathLoRAModel
        model_configuration = read_json(ROOT / configuration["primary_configuration"])
        torch.manual_seed(model_configuration["seed"])
        model = LPWMDrivoRPlanningPathLoRAModel(ROOT / model_configuration["public_checkpoint"])
    else:
        from planning_aware_future_prediction.object_centric.lpwm_adapter_full_finetuning import build_adapter_or_full_planning_model
        model_configuration = read_json(ROOT / configuration["adapter_configuration"])
        stage1 = read_json(ROOT / model_configuration["stage1_config"])
        torch.manual_seed(model_configuration["seed"])
        model = build_adapter_or_full_planning_model(ROOT / stage1["output_directory"] / "stage1/checkpoint.pt",
                                                    model_configuration, "metric_plus_world", ROOT)
    checkpoint = checkpoint_override or ROOT / configuration[f"{system}_checkpoint"]
    saved = torch.load(checkpoint, map_location="cpu", weights_only=False, mmap=True)
    if checkpoint_override is None:
        assert saved["completed_updates"] == configuration[f"{system}_completed_updates"]
    model.load_state_dict(saved["model"], strict=True)
    metadata = {"checkpoint": str(checkpoint), "checkpoint_sha256": digest(checkpoint),
                "completed_updates": saved["completed_updates"], "frozen_native_sha256": saved["frozen_native_sha256"]}
    if system == "primary":
        assert model.frozen_native_digest() == metadata["frozen_native_sha256"]
    else:
        from profile_lpwm_adapter_concurrent_training import native_digest
        assert native_digest(model) == metadata["frozen_native_sha256"]
    del saved
    return model.eval().requires_grad_(False), metadata


def predict_batch(model, system, images, ego, first, last, device):
    import torch
    pixels = torch.from_numpy(np.array(images[first:last])).to(device).permute(0, 1, 4, 2, 3).float() / 255
    status = torch.from_numpy(np.array(ego[first:last])).to(device)
    with torch.inference_mode(), torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
        prediction = model({"image": pixels, "ego_status": status[:, None]}) if system == "primary" else model(pixels, status)
    trajectory = prediction["trajectory"].float().cpu().numpy()
    assert trajectory.shape == (last - first, 8, 3) and np.isfinite(trajectory).all()
    return trajectory


def inputs_for_system(output, system):
    image_name = "primary_current_images.npy" if system == "primary" else "adapter_observed_images.npy"
    return (np.load(output / "inputs" / image_name, mmap_mode="r"),
            np.load(output / "inputs" / f"{system}_ego.npy", mmap_mode="r"))


def cpu_check(configuration):
    """One inference per architecture; no CUDA allocation or metric selection."""
    import torch
    torch.set_num_threads(2)
    output = ROOT / configuration["output_directory"]
    checks = {}
    for system in ("primary", "adapter"):
        model, metadata = build_model(configuration, system)
        images, ego = inputs_for_system(output, system)
        started = time.monotonic()
        prediction = predict_batch(model, system, images, ego, 0, 1, torch.device("cpu"))[0]
        checks[system] = {**metadata, "finite": True, "trajectory_shape": list(prediction.shape),
                          "seconds": time.monotonic() - started,
                          "no_training_updates": True, "metric_computed": False}
        del model
    write_json(output / "cpu_execution_check.json", {"passed": True, "checks": checks,
        "scope": "CPU execution smoke check of the exact comparison checkpoints; no performance result"})
    print(json.dumps(checks), flush=True)


def infer(configuration_path, configuration, system):
    import torch
    registration = verify_registration(configuration_path, configuration)
    output = ROOT / configuration["output_directory"]
    destination = output / system
    destination.mkdir(exist_ok=True)
    if (destination / "inference_complete.json").exists():
        return
    used = check_resources(configuration)
    if used >= configuration["admission_card_bytes"]:
        raise EvaluationDeferred("Insufficient memory for evaluation admission")
    torch.set_num_threads(2)
    torch.cuda.set_device(0)
    device = torch.device("cuda", 0)
    torch.cuda.set_per_process_memory_fraction(configuration["maximum_allocator_bytes"] /
        torch.cuda.get_device_properties(device).total_memory, device)
    model, checkpoint = build_model(configuration, system)
    if system == "primary":
        assert checkpoint["checkpoint_sha256"] == registration["primary_checkpoint_sha256"]
    else:
        epoch = configuration.get("adapter_evaluation_epoch", 3)
        completed = read_json(ROOT / configuration["adapter_training_root"] / f"epoch{epoch:02d}_training_complete.json")
        assert completed["checkpoint_sha256"] == checkpoint["checkpoint_sha256"]
    model = model.to(device)
    images, ego = inputs_for_system(output, system)
    count = configuration["scene_count"]
    progress_path = destination / "inference_progress.json"
    previous = read_json(progress_path) if progress_path.exists() else {"completed": 0}
    if previous["completed"]:
        assert previous["checkpoint_sha256"] == checkpoint["checkpoint_sha256"]
    trajectory_path = destination / "trajectories.npy"
    trajectories = np.lib.format.open_memmap(trajectory_path, mode="r+" if trajectory_path.exists() else "w+",
                                             dtype=np.float32, shape=(count, 8, 3))
    maximum_used = used
    started = time.monotonic()
    for index in range(previous["completed"], count, configuration["inference_batch_size"]):
        maximum_used = max(maximum_used, check_resources(configuration))
        last = min(count, index + configuration["inference_batch_size"])
        trajectories[index:last] = predict_batch(model, system, images, ego, index, last, device)
        trajectories.flush()
        write_json(progress_path, {"completed": last, "scene_count": count,
            "checkpoint_sha256": checkpoint["checkpoint_sha256"], "updated_unix": time.time(),
            "maximum_card_bytes": maximum_used, "elapsed_this_process_seconds": time.monotonic() - started})
    assert np.isfinite(trajectories).all()
    verify_registration(configuration_path, configuration)
    write_json(destination / "inference_complete.json", {"complete": True, **checkpoint,
        "scene_count": count, "trajectory_sha256": digest(trajectory_path), "panel_sha256": registration["panel_sha256"],
        "maximum_card_bytes": maximum_used, "peak_allocator_bytes": torch.cuda.max_memory_reserved(),
        "inference_batch_size": configuration["inference_batch_size"], "future_ground_truth_input": False})


def paired_interval(first_rows, second_rows, metric, repeats, seed):
    first = {row["token"]: row for row in first_rows}
    second = {row["token"]: row for row in second_rows}
    assert set(first) == set(second)
    groups = defaultdict(list)
    for token, row in first.items():
        groups[row["recording_group"]].append(row[metric] - second[token][metric])
    group_values = [np.asarray(groups[group]) for group in sorted(groups)]
    sums = np.array([values.sum() for values in group_values])
    counts = np.array([len(values) for values in group_values])
    generator = np.random.default_rng(seed)
    samples = generator.integers(0, len(group_values), size=(repeats, len(group_values)))
    means = sums[samples].sum(-1) / counts[samples].sum(-1)
    return {"difference": float(sums.sum() / counts.sum()),
            "confidence_interval_95": np.quantile(means, [.025, .975]).tolist(),
            "recordings": len(group_values), "scenes": int(counts.sum())}


def score(configuration_path, configuration):
    from score_lpwm_drivor_navtest import score_scene
    verify_registration(configuration_path, configuration)
    output = ROOT / configuration["output_directory"]
    records = read_json(output / "scene_metadata.json")["records"]
    truth = np.load(output / "inputs/ground_truth_trajectory.npy")
    summaries, all_rows = {}, {}
    for system in ("primary", "adapter"):
        destination = output / system
        completed = read_json(destination / "inference_complete.json")
        assert completed["complete"] and digest(destination / "trajectories.npy") == completed["trajectory_sha256"]
        trajectories = np.load(destination / "trajectories.npy")
        jobs = [(record["token"], trajectory, Path(record["metric_cache_file"]))
                for record, trajectory in zip(records, trajectories)]
        rows = []
        with ProcessPoolExecutor(max_workers=configuration["cpu_workers"], mp_context=multiprocessing.get_context("spawn")) as pool:
            for index, row in enumerate(pool.map(score_scene, jobs, chunksize=4)):
                error = np.linalg.norm(trajectories[index, :, :2] - truth[index, :, :2], axis=-1)
                row.update(recording_group=records[index]["recording_group"], scene_type=records[index]["scene_type"],
                           ade_m=float(error.mean()), fde_m=float(error[-1]), pdms=100 * row["score"])
                rows.append(row)
                if len(rows) % 16 == 0:
                    write_json(output / "score_progress.json", {"system": system, "completed": len(rows), "scene_count": len(records)})
        assert len(rows) == configuration["scene_count"] and all(row["valid"] for row in rows)
        write_json(destination / "official_scores.json", rows)
        numeric = [key for key, value in rows[0].items() if isinstance(value, float)]
        summary = {"checkpoint": completed, "scene_count": len(rows), "failed": 0,
            "means": {key: float(np.mean([row[key] for row in rows])) for key in numeric},
            "by_scene_type": {kind: {"count": sum(row["scene_type"] == kind for row in rows),
                **{metric: float(np.mean([row[metric] for row in rows if row["scene_type"] == kind]))
                   for metric in ("pdms", "ade_m", "fde_m")}} for kind in sorted({row["scene_type"] for row in rows})}}
        summaries[system], all_rows[system] = summary, rows
    comparisons = {metric: paired_interval(all_rows["primary"], all_rows["adapter"], metric,
                    configuration["bootstrap_repeats"], configuration["bootstrap_seed"]) for metric in ("pdms", "ade_m", "fde_m")}
    result = {"complete": True, "scope": configuration["scope"], "planning_epochs": configuration["planning_epochs"],
        "systems": summaries, "primary_minus_adapter": comparisons, "scene_count": len(records),
        "recording_count": len({row["recording_group"] for row in records}), "training_recording_overlap": 0,
        "same_official_scorer": True, "test_driven_model_selection": False,
        "remaining_confounds": configuration["confounds"], "finished_unix": time.time()}
    write_json(output / "evaluation_complete.json", result)
    write_json(ROOT / configuration["shared_results_directory"] / "evaluation_complete.json", result)
    print(json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--mode", choices=["cpu-check", "infer", "score"], required=True)
    parser.add_argument("--system", choices=["primary", "adapter"])
    arguments = parser.parse_args()
    arguments.config = arguments.config.resolve()
    specification = read_json(arguments.config)
    try:
        if arguments.mode == "cpu-check":
            cpu_check(specification)
        elif arguments.mode == "infer":
            assert arguments.system is not None
            infer(arguments.config, specification, arguments.system)
        else:
            score(arguments.config, specification)
    except EvaluationDeferred as error:
        print("EVALUATION_DEFERRED", str(error), flush=True)
        raise SystemExit(75)
