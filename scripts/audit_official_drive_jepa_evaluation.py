"""Audit completeness, smoke-test official inference, and aggregate official CSVs.

This utility does not import the project's pilot models or modify upstream inference.
"""

import argparse
import hashlib
import json
import os
import platform
import sys
import time
from dataclasses import asdict
from pathlib import Path


def write_result(result_path, result):
    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    if set(result) == {"tokens"}:
        print(f"Saved {len(result['tokens'])} expected scene tokens to {result_path}")
    else:
        print(json.dumps(result, indent=2, allow_nan=False))


def official_configuration(workspace):
    from hydra import compose, initialize_config_dir

    specification = json.loads((workspace / "configs/official_drive_jepa/reproduction_v1.json").read_text())
    assets = json.loads((workspace / "results/official_drive_jepa_reproduction/verified_assets.json").read_text())
    official_root = workspace / specification["official_source_worktree"] / "navsim_v1"
    sys.path.insert(0, str(official_root))
    os.chdir(official_root)
    os.environ["OPENSCENE_DATA_ROOT"] = str(workspace / "dataset")
    os.environ["NAVSIM_EXP_ROOT"] = str(workspace / "outputs/official_drive_jepa_reproduction")
    os.environ["NAVSIM_DEVKIT_ROOT"] = str(official_root)
    os.environ["NUPLAN_MAPS_ROOT"] = str(workspace / "dataset/maps")
    os.environ["NUPLAN_MAP_VERSION"] = "nuplan-maps-v1.0"
    checkpoint_path = assets["planning_checkpoint"]["path"]
    encoder_path = assets["initialization_encoder"]["path"]
    overrides = [
        "train_test_split=navtest",
        "agent=drive_jepa_perception_free_agent",
        f"agent.pretrain_pt_path={encoder_path}",
        "agent.image_architecture=vit_large",
        "agent.num_keyval=129",
        "agent.front_only=true",
        "agent.tf_dropout=0.0",
        "agent.freeze_encoder=false",
        "agent.double_image=true",
        f"agent.checkpoint_path={checkpoint_path}",
        f"metric_cache_path={assets['metric_cache_root']}",
    ]
    with initialize_config_dir(config_dir=str(official_root / "navsim/planning/script/config/pdm_scoring"), version_base=None):
        configuration = compose(config_name="default_run_pdm_score", overrides=overrides)
    return specification, assets, configuration, official_root


def create_scene_loader(configuration, sensor_config=None):
    from hydra.utils import instantiate

    from navsim.common.dataclasses import SensorConfig
    from navsim.common.dataloader import SceneLoader

    return SceneLoader(
        sensor_blobs_path=Path(configuration.sensor_blobs_path),
        data_path=Path(configuration.navsim_log_path),
        scene_filter=instantiate(configuration.train_test_split.scene_filter),
        sensor_config=sensor_config if sensor_config is not None else SensorConfig.build_no_sensors(),
    )


def audit_completeness(workspace, configuration):
    from navsim.common.dataloader import MetricCacheLoader

    scene_loader = create_scene_loader(configuration)
    cache_loader = MetricCacheLoader(Path(configuration.metric_cache_path))
    expected_tokens = set(configuration.train_test_split.scene_filter.tokens)
    actual_tokens = set(scene_loader.tokens)
    cache_tokens = set(cache_loader.tokens)
    required_logs = set(configuration.train_test_split.scene_filter.log_names)
    available_logs = {file.stem for file in Path(configuration.navsim_log_path).glob("*.pkl")}
    missing_images = set()
    required_images = set()
    input_cadences_microseconds = set()
    for frames in scene_loader.scene_frames_dicts.values():
        input_cadences_microseconds.add(int(frames[3]["timestamp"]) - int(frames[2]["timestamp"]))
        for frame_index in (2, 3):
            camera_entry = next(value for camera_name, value in frames[frame_index]["cams"].items() if camera_name.lower() == "cam_f0")
            image_path = Path(configuration.sensor_blobs_path) / camera_entry["data_path"]
            required_images.add(str(image_path))
    for image_path in sorted(required_images):
        if not Path(image_path).is_file():
            missing_images.add(image_path)
    missing_cache_files = [str(cache_loader.metric_cache_paths[token]) for token in sorted(actual_tokens & cache_tokens) if not Path(cache_loader.metric_cache_paths[token]).is_file()]
    report = {
        "split": "official_navtest_not_pilot_split",
        "expected_scene_count_from_official_token_list": len(expected_tokens),
        "loaded_scene_count": len(actual_tokens),
        "official_cache_token_count": len(cache_tokens),
        "required_log_count": len(required_logs),
        "missing_logs": sorted(required_logs - available_logs),
        "missing_scene_tokens": sorted(expected_tokens - actual_tokens),
        "unexpected_scene_tokens": sorted(actual_tokens - expected_tokens),
        "missing_metric_cache_tokens": sorted(actual_tokens - cache_tokens),
        "unused_metric_cache_tokens": sorted(cache_tokens - actual_tokens),
        "missing_metric_cache_files": missing_cache_files,
        "unique_required_front_images": len(required_images),
        "missing_front_images": sorted(missing_images),
        "observed_input_cadences_microseconds": sorted(input_cadences_microseconds),
        "future_camera_files_required_for_inference": False,
    }
    report["complete"] = not any(report[key] for key in (
        "missing_logs", "missing_scene_tokens", "unexpected_scene_tokens", "missing_metric_cache_tokens",
        "missing_metric_cache_files", "missing_front_images",
    ))
    result_root = workspace / "results/official_drive_jepa_reproduction"
    write_result(result_root / "evaluation_preflight.json", report)
    write_result(result_root / "expected_scene_tokens.json", {"tokens": sorted(expected_tokens)})
    if not report["complete"]:
        raise RuntimeError("Full official evaluation blocked by incomplete inputs; no silent intersection allowed")


def smoke_official_inference(workspace, specification, configuration, official_root):
    import numpy as np
    import torch
    from hydra.utils import instantiate

    import navsim
    from navsim.common.dataloader import MetricCacheLoader
    from navsim.evaluate.pdm_score import pdm_score

    preflight = json.loads((workspace / "results/official_drive_jepa_reproduction/evaluation_preflight.json").read_text())
    if not preflight["complete"]:
        raise RuntimeError("Preflight incomplete")
    torch.set_num_threads(1)
    torch.manual_seed(0)
    np.random.seed(0)
    if not torch.cuda.is_available():
        raise RuntimeError("Official AbstractAgent.compute_trajectory requires CUDA")
    start_time = time.perf_counter()
    agent = instantiate(configuration.agent)
    planning_checkpoint = torch.load(configuration.agent.checkpoint_path, map_location="cpu")
    official_state = {key.replace("agent.", ""): value for key, value in planning_checkpoint["state_dict"].items()}
    model_state = agent.state_dict()
    missing_keys = sorted(set(model_state) - set(official_state))
    unexpected_keys = sorted(set(official_state) - set(model_state))
    shape_mismatches = [key for key in set(official_state) & set(model_state) if official_state[key].shape != model_state[key].shape]
    if missing_keys or unexpected_keys or shape_mismatches:
        raise RuntimeError(f"Planning checkpoint mismatch: {missing_keys}, {unexpected_keys}, {shape_mismatches}")
    checkpoint_metadata = {key: planning_checkpoint.get(key) for key in ("epoch", "global_step", "pytorch-lightning_version")}
    del planning_checkpoint, official_state, model_state
    # Upstream initialize() uses load_state_dict's strict=True default.
    agent.initialize()
    agent.eval()
    agent.cuda()
    torch.cuda.synchronize()
    initialization_seconds = time.perf_counter() - start_time
    scene_loader = create_scene_loader(configuration, agent.get_sensor_config())
    cache_loader = MetricCacheLoader(Path(configuration.metric_cache_path))
    simulator = instantiate(configuration.simulator)
    scorer = instantiate(configuration.scorer)
    assert simulator.proposal_sampling == scorer.proposal_sampling
    sampled_tokens = [min(tokens) for _, tokens in sorted(scene_loader.get_tokens_list_per_log().items())[:3]]
    scene_records = []
    for scene_token in sampled_tokens:
        scene_start = time.perf_counter()
        agent_input = scene_loader.get_agent_input_from_token(scene_token)
        features = {}
        for builder in agent.get_feature_builders():
            features.update(builder.compute_features(agent_input))
        feature_shapes = {key: list(value.shape) for key, value in features.items()}
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        inference_start = time.perf_counter()
        trajectory = agent.compute_trajectory(agent_input)
        torch.cuda.synchronize()
        inference_seconds = time.perf_counter() - inference_start
        if trajectory.poses.shape != (8, 3) or not np.isfinite(trajectory.poses).all():
            raise RuntimeError(f"Invalid trajectory for scene {scene_token}")
        metric_cache = cache_loader.get_from_token(scene_token)
        scoring_start = time.perf_counter()
        scores = asdict(pdm_score(metric_cache, trajectory, simulator.proposal_sampling, simulator, scorer))
        if not all(np.isfinite(value) for value in scores.values()):
            raise RuntimeError(f"Invalid scores for scene {scene_token}")
        scene_records.append({
            "token": scene_token,
            "valid": True,
            "feature_shapes": feature_shapes,
            "raw_front_image_shapes": [list(agent_input.cameras[index].cam_f0.image.shape) for index in (2, 3)],
            "trajectory_poses_local_xy_meters_heading_radians": trajectory.poses.tolist(),
            "trajectory_interval_seconds": trajectory.trajectory_sampling.interval_length,
            "trajectory_horizon_seconds": trajectory.trajectory_sampling.time_horizon,
            "inference_seconds": inference_seconds,
            "scoring_seconds": time.perf_counter() - scoring_start,
            "scene_total_seconds": time.perf_counter() - scene_start,
            "cuda_peak_allocated_bytes": torch.cuda.max_memory_allocated(),
            "cuda_peak_reserved_bytes": torch.cuda.max_memory_reserved(),
            "scores_fraction_not_percent": scores,
        })
    report = {
        "purpose": "normal_operation_smoke_only_not_paper_reproduction_score",
        "successful_scenes": len(scene_records),
        "failed_scenes": 0,
        "planning_checkpoint_missing_keys": missing_keys,
        "planning_checkpoint_unexpected_keys": unexpected_keys,
        "planning_checkpoint_shape_mismatches": shape_mismatches,
        "strict_full_model_loading": True,
        "strict_initialization_encoder_loading": True,
        "checkpoint_metadata": checkpoint_metadata,
        "source_commit": specification["official_source_commit"],
        "actual_navsim_module_path": navsim.__file__,
        "conda_python_executable": sys.executable,
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "torch_cuda_version": torch.version.cuda,
        "device_name": torch.cuda.get_device_name(),
        "visible_physical_gpus": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "precision": "official float32 inference; no autocast",
        "matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
        "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
        "initialization_seconds": initialization_seconds,
        "scene_records": scene_records,
        "official_source_root": str(official_root),
    }
    write_result(workspace / "results/official_drive_jepa_reproduction/inference_smoke.json", report)


def prepare_evaluation_shards(workspace, configuration):
    """Balance log-level execution shards without filtering any official scene."""
    preflight = json.loads((workspace / "results/official_drive_jepa_reproduction/evaluation_preflight.json").read_text())
    if not preflight["complete"]:
        raise RuntimeError("Preflight incomplete")
    scene_loader = create_scene_loader(configuration)
    token_groups = scene_loader.get_tokens_list_per_log()
    shard_log_names = [[], []]
    shard_tokens = [[], []]
    for log_name, tokens in sorted(token_groups.items(), key=lambda item: (-len(item[1]), item[0])):
        shard_index = min(range(2), key=lambda index: (len(shard_tokens[index]), index))
        shard_log_names[shard_index].append(log_name)
        shard_tokens[shard_index].extend(tokens)
    expected_tokens = set(scene_loader.tokens)
    if set(shard_tokens[0]) & set(shard_tokens[1]) or set(shard_tokens[0] + shard_tokens[1]) != expected_tokens:
        raise RuntimeError("Evaluation shards must be disjoint and cover the complete official split")
    for shard_index in range(2):
        write_result(workspace / f"results/official_drive_jepa_reproduction/evaluation_shard_{shard_index}.json", {
            "execution_shard_index": shard_index,
            "physical_gpu": shard_index,
            "worker_count": 2,
            "scene_count": len(shard_tokens[shard_index]),
            "log_names": sorted(shard_log_names[shard_index]),
            "tokens": sorted(shard_tokens[shard_index]),
            "purpose": "execution parallelism only; not a new evaluation split or selection filter",
        })
    print(f"GPU0/1 execution shard scene counts: {[len(tokens) for tokens in shard_tokens]}")


def summarize_official_evaluation(workspace, specification, csv_paths):
    import numpy as np
    import pandas as pd

    input_tables = [pd.read_csv(csv_path) for csv_path in csv_paths]
    scene_results = pd.concat([table[table["token"] != "average"] for table in input_tables], ignore_index=True)
    expected_tokens = set(json.loads((workspace / "results/official_drive_jepa_reproduction/expected_scene_tokens.json").read_text())["tokens"])
    actual_tokens = set(scene_results["token"])
    paper_scores = specification["paper"]["reported_percent_scores"]
    metric_columns = list(paper_scores)
    if "driving_direction_compliance" in scene_results:
        metric_columns.append("driving_direction_compliance")
    finite_metrics = np.isfinite(scene_results[metric_columns].to_numpy()).all(axis=1)
    successful_rows = scene_results[scene_results["valid"].eq(True) & finite_metrics]
    def server_mean(metric):
        return float(successful_rows[metric].mean() * 100) if len(successful_rows) else None
    comparison = {metric: {
        "paper_percent_score": reported_score,
        "server_percent_score": server_mean(metric),
        "server_minus_paper_percentage_points": server_mean(metric) - reported_score if len(successful_rows) else None,
    } for metric, reported_score in paper_scores.items()}
    report = {
        "official_csv_files": [{"path": str(csv_path.resolve()), "sha256": hashlib.sha256(csv_path.read_bytes()).hexdigest()} for csv_path in csv_paths],
        "expected_scenes": len(expected_tokens),
        "result_rows": len(scene_results),
        "successful_scenes": len(successful_rows),
        "failed_scenes": len(scene_results) - len(successful_rows),
        "failed_tokens": scene_results[~scene_results["valid"].eq(True) | ~finite_metrics]["token"].tolist(),
        "nonfinite_metric_tokens": scene_results[~finite_metrics]["token"].tolist(),
        "missing_tokens": sorted(expected_tokens - actual_tokens),
        "unexpected_tokens": sorted(actual_tokens - expected_tokens),
        "duplicate_tokens": scene_results["token"][scene_results["token"].duplicated()].tolist(),
        "comparison": comparison,
        "all_server_percent_metrics": {metric: server_mean(metric) for metric in metric_columns},
        "judgment": "numeric comparison only; no official tolerance supplied; no arbitrary reproduction-success threshold",
    }
    report["evaluation_complete"] = (
        report["successful_scenes"] == report["expected_scenes"]
        and not any(report[key] for key in ("failed_scenes", "missing_tokens", "unexpected_tokens", "duplicate_tokens"))
    )
    write_result(workspace / "results/official_drive_jepa_reproduction/full_navtest_results.json", report)
    scene_results.sort_values("token").to_csv(workspace / "results/official_drive_jepa_reproduction/official_scene_scores.csv", index=False)
    if not report["evaluation_complete"]:
        raise RuntimeError("Evaluation incomplete; scores are conditional successful-subset statistics, not a full-split reproduction")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("preflight", "smoke", "prepare_shards", "summarize"), required=True)
    parser.add_argument("--workspace", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--csv-path", type=Path, action="append")
    arguments = parser.parse_args()
    workspace = arguments.workspace.resolve()
    specification, _, configuration, official_root = official_configuration(workspace)
    if arguments.stage == "preflight":
        audit_completeness(workspace, configuration)
    elif arguments.stage == "smoke":
        smoke_official_inference(workspace, specification, configuration, official_root)
    elif arguments.stage == "prepare_shards":
        prepare_evaluation_shards(workspace, configuration)
    else:
        if arguments.csv_path is None:
            parser.error("summarize requires --csv-path")
        summarize_official_evaluation(workspace, specification, arguments.csv_path)


if __name__ == "__main__":
    main()
