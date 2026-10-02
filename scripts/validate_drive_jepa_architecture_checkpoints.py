"""Post-training real-image output preservation and bounded inference cost check."""

import argparse
import contextlib
import json
import os
import pickle
import time
from pathlib import Path

import numpy as np
import torch
from run_drive_jepa_architecture_followup import (
    WORKSPACE,
    build_model,
    observed_forward,
)
from run_drive_jepa_selection_comparison import get_training_batch, write_json
from validate_drive_jepa_selective_future_connection import (
    file_sha256,
    parameter_sha256,
)


@torch.no_grad()
def timed_inference(function):
    for _ in range(2):
        function()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    durations = []
    for _ in range(7):
        torch.cuda.synchronize()
        start = time.perf_counter()
        function()
        torch.cuda.synchronize()
        durations.append(time.perf_counter() - start)
    return {
        "wall_seconds_per_two_window_batch": durations,
        "mean_seconds": float(np.mean(durations)),
        "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-directory", type=Path, required=True)
    arguments = parser.parse_args()
    run = arguments.run_directory.resolve()
    destination = run / "trained_checkpoint_postflight.json"
    if destination.exists() or not (run / "completion.json").exists():
        raise RuntimeError("Completed run and fresh postflight required")
    if os.environ.get("CUDA_VISIBLE_DEVICES") not in ("0", "1"):
        raise RuntimeError("One approved GPU required")
    if torch.cuda.mem_get_info()[0] < 16 * 2**30:
        raise RuntimeError("Shared GPU needs 16GiB admission margin")
    torch.set_num_threads(1)
    from audit_official_drive_jepa_evaluation import official_configuration
    from hydra.utils import instantiate

    _, assets, configuration, _ = official_configuration(WORKSPACE)
    execution = json.loads((run / "execution_specification.json").read_text())
    for name in ("planning_checkpoint", "initialization_encoder"):
        if file_sha256(assets[name]["path"]) != assets[name]["sha256"]:
            raise RuntimeError("Official asset mismatch")
    with (
        (run / "postflight_strict_loading.log").open("w") as stream,
        contextlib.redirect_stdout(stream),
    ):
        agent = instantiate(configuration.agent)
        checkpoint = torch.load(
            configuration.agent.checkpoint_path, map_location="cpu", mmap=True
        )
        agent.load_state_dict(
            {
                name.replace("agent.", ""): value
                for name, value in checkpoint["state_dict"].items()
            },
            strict=True,
        )
        del checkpoint
    agent.eval().cuda()
    # Compare the same frozen evaluation configuration as the trained wrapper.
    agent._model.requires_grad_(False)
    original_hash = parameter_sha256(agent._model)
    from navsim.common.dataclasses import AgentInput

    records = json.loads(
        (WORKSPACE / execution["specification"]["reused_cache"]).read_text()
    )["records"][:2]
    clips, cached_rows = [], []
    for record in records:
        if record["split"] != "train":
            raise RuntimeError("Postflight must use training observations")
        with (
            WORKSPACE / "dataset/navsim_logs/trainval" / record["segment_filename"]
        ).open("rb") as stream:
            frames = pickle.load(stream)[
                record["start_index"] : record["start_index"] + 4
            ]
        observed = AgentInput.from_scene_dict_list(
            frames,
            WORKSPACE / "dataset/sensor_blobs/trainval",
            4,
            agent.get_sensor_config(),
        )
        features = agent.get_feature_builders()[0].compute_features(observed)
        clips.append(
            torch.stack(
                (features["camera_feature_2"], features["camera_feature_1"]), dim=1
            )
        )
        cached = torch.load(record["cache_file"], map_location="cpu", weights_only=True)
        cached["current_prefix_latents"] = torch.load(
            run
            / "current_encoder_prefix_cache"
            / (record["current_frame_token"] + ".pt"),
            map_location="cpu",
            weights_only=True,
        )
        cached_rows.append(cached)
    cache = {
        key: torch.stack([row[key] for row in cached_rows]) for key in cached_rows[0]
    }
    batch = get_training_batch(cache, [0, 1])
    clips = torch.stack(clips).cuda()
    report = {
        "seed": 29,
        "scene_tokens": [row["current_frame_token"] for row in records],
        "timing_scope": "two_window_batch_RGB_to_trajectory_excludes_IO_teacher_scorer_shared_GPU_not_pure_kernel",
        "dtype": "float32",
        "warmup": 2,
        "repeats": 7,
        "conditions": {},
    }
    with torch.no_grad():
        for _ in range(2):
            agent._model(clips, batch["current_ego_status"])
        original = agent._model(clips, batch["current_ego_status"])["trajectory"]
        report["original_timing"] = timed_inference(
            lambda: agent._model(clips, batch["current_ego_status"])
        )
        for condition in execution["specification"]["conditions"]:
            model = build_model(agent._model, condition, 29)
            checkpoint = torch.load(
                run / f"{condition}_seed29" / "joint_complete.pt", map_location="cpu"
            )
            parameters = {
                name: parameter
                for name, parameter in model.named_parameters()
                if parameter.requires_grad
            }
            if set(parameters) != set(checkpoint["trainable_state"]):
                raise RuntimeError("Trainable delta checkpoint keys mismatch")
            for name, parameter in parameters.items():
                value = checkpoint["trainable_state"][name]
                if value.shape != parameter.shape:
                    raise RuntimeError("Trainable delta shape mismatch")
                parameter.copy_(value)
            model.eval()
            disabled = model(
                clips, batch["current_ego_status"], enable_future_branch=False
            )["trajectory"]
            enabled = model(clips, batch["current_ego_status"])["trajectory"]
            cached = observed_forward(model, batch)["trajectory"]
            if not torch.equal(original, disabled):
                repeated_original = agent._model(clips, batch["current_ego_status"])[
                    "trajectory"
                ]
                write_json(
                    run / "postflight_equivalence_failure.json",
                    {
                        "condition": condition,
                        "initial_vs_disabled_max_abs_difference": float(
                            (original - disabled).abs().max()
                        ),
                        "repeat_vs_disabled_max_abs_difference": float(
                            (repeated_original - disabled).abs().max()
                        ),
                        "hash_before": original_hash,
                        "hash_after": parameter_sha256(agent._model),
                        "original": original.cpu().tolist(),
                        "disabled": disabled.cpu().tolist(),
                    },
                )
                raise RuntimeError(
                    "Disabling trained branch no longer preserves original"
                )
            torch.testing.assert_close(enabled, cached, atol=1e-5, rtol=1e-5)
            report["conditions"][condition] = {
                "strict_trainable_delta_keys": len(parameters),
                "trained_disabled_bitwise_equal": True,
                "online_vs_cached_trajectory_max_abs_difference": float(
                    (enabled - cached).abs().max()
                ),
                "enabled_timing": timed_inference(
                    lambda model=model: model(clips, batch["current_ego_status"])
                ),
                "cached_branch_timing": timed_inference(
                    lambda model=model: observed_forward(model, batch)
                ),
            }
            del model, parameters, checkpoint
            torch.cuda.empty_cache()
    report["original_model_hash_unchanged"] = (
        parameter_sha256(agent._model) == execution["baseline_hash"]
    )
    if not report["original_model_hash_unchanged"]:
        raise RuntimeError("Original model hash changed")
    write_json(destination, report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
