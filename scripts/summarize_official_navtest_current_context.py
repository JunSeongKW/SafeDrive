"""Describe saved official scores using preregistered CURRENT-input context only.

No inference, target construction, fitting, or metric cache generation is performed.
"""

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from audit_official_drive_jepa_evaluation import (
    create_scene_loader,
    official_configuration,
)


def file_sha256(file_path):
    return hashlib.sha256(file_path.read_bytes()).hexdigest()


def classify_current_speed(current_dynamic_state, speed_bins):
    try:
        velocity = np.asarray(current_dynamic_state, dtype=np.float64).reshape(-1)[:2]
    except (TypeError, ValueError):
        return None, "missing_or_invalid"
    if velocity.size != 2 or not np.isfinite(velocity).all():
        return None, "missing_or_invalid"
    speed = float(np.linalg.norm(velocity))
    for speed_bin in speed_bins:
        upper_limit = speed_bin["upper_exclusive_mps"]
        if speed >= speed_bin["lower_inclusive_mps"] and (upper_limit is None or speed < upper_limit):
            return speed, speed_bin["name"]
    raise ValueError("Preregistered speed bins must cover every nonnegative finite speed")


def classify_stored_command(driving_command):
    try:
        command = np.asarray(driving_command, dtype=np.float64).reshape(-1)
    except (TypeError, ValueError):
        return None, "missing_or_invalid"
    if command.size != 4 or not np.isfinite(command).all():
        return None, "missing_or_invalid"
    if not np.isin(command, (0.0, 1.0)).all() or command.sum() != 1.0:
        return None, "missing_or_invalid"
    command_index = int(np.argmax(command))
    return command_index, f"command_index_{command_index}"


def context_group_summary(joined_rows, axis, category, configuration):
    if axis == "speed_by_command":
        speed_category, command_category = category
        selected_rows = joined_rows[
            joined_rows["speed_bin"].eq(speed_category) & joined_rows["command_category"].eq(command_category)
        ]
        category_name = f"{speed_category}__{command_category}"
    else:
        column_name = "speed_bin" if axis == "speed" else "command_category"
        selected_rows = joined_rows[joined_rows[column_name].eq(category)]
        category_name = category
    scene_count = len(selected_rows)
    recording_count = selected_rows.loc[selected_rows["native_log_token"].ne(""), "native_log_token"].nunique()
    log_count = selected_rows.loc[selected_rows["native_log_name"].ne(""), "native_log_name"].nunique()
    return {
        "axis": axis,
        "category": category_name,
        "scene_count": scene_count,
        "native_recording_count": int(recording_count),
        "exported_log_count": int(log_count),
        "missing_native_recording_id_scenes": int(selected_rows["native_log_token"].eq("").sum()),
        "descriptive_interpretation_allowed": bool(
            scene_count >= configuration["minimum_scenes_for_descriptive_interpretation"]
            and recording_count >= configuration["minimum_native_recordings_for_descriptive_interpretation"]
            and "missing_or_invalid" not in category_name
        ),
        **{
            f"{metric_name}_percent": float(selected_rows[metric_name].mean() * 100) if scene_count else None
            for metric_name in configuration["metric_names"]
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/analysis/official_navtest_current_context_v1.json")
    arguments = parser.parse_args()
    workspace = Path(__file__).resolve().parents[1]
    configuration_path = workspace / arguments.config
    configuration = json.loads(configuration_path.read_text())
    result_root = workspace / configuration["result_directory"]
    if result_root.exists():
        raise RuntimeError("Preserve previous analysis: choose a new result directory in a new config")
    score_path = workspace / configuration["scene_scores_path"]
    score_hash_before = file_sha256(score_path)
    expected_path = workspace / configuration["expected_scene_tokens_path"]
    expected_tokens = set(json.loads(expected_path.read_text())["tokens"])
    start_time = time.perf_counter()
    _, _, official_config, _ = official_configuration(workspace)
    scene_loader = create_scene_loader(official_config)
    metadata_records = []
    current_index = configuration["current_frame_index_in_official_scene"]
    # Build every context label BEFORE reading any scene score or joining it.
    for scene_token in sorted(expected_tokens):
        frames = scene_loader.scene_frames_dicts.get(scene_token)
        current_frame = frames[current_index] if frames is not None and len(frames) > current_index else {}
        if current_frame and current_frame["token"] != scene_token:
            raise RuntimeError("Current frame/token alignment disagrees with the official scene loader")
        speed, speed_bin = classify_current_speed(current_frame.get("ego_dynamic_state"), configuration["speed_bins"])
        command_index, command_category = classify_stored_command(current_frame.get("driving_command"))
        metadata_records.append({
            "token": scene_token,
            "native_log_name": str(current_frame.get("log_name") or ""),
            "native_log_token": str(current_frame.get("log_token") or ""),
            "current_ego_speed_mps": speed,
            "speed_bin": speed_bin,
            "current_command_raw_index": command_index,
            "command_category": command_category,
            "current_frame_metadata_missing": not bool(current_frame),
        })
    metadata = pd.DataFrame(metadata_records)
    score_rows = pd.read_csv(score_path, dtype={"token": str})
    if score_rows["token"].duplicated().any() or set(score_rows["token"]) != expected_tokens:
        raise RuntimeError("Saved score tokens are not the complete unique official split")
    if not score_rows["valid"].eq(True).all() or not np.isfinite(score_rows[configuration["metric_names"]].to_numpy()).all():
        raise RuntimeError("Invalid/nonfinite saved scores: do not silently summarize a success subset")
    joined_rows = metadata.merge(score_rows, on="token", how="left", validate="one_to_one")
    speed_categories = [speed_bin["name"] for speed_bin in configuration["speed_bins"]] + ["missing_or_invalid"]
    command_categories = configuration["command_categories"]
    group_summaries = [
        context_group_summary(joined_rows, "speed", category, configuration) for category in speed_categories
    ] + [
        context_group_summary(joined_rows, "command", category, configuration) for category in command_categories
    ] + [
        context_group_summary(joined_rows, "speed_by_command", (speed_category, command_category), configuration)
        for speed_category in speed_categories for command_category in command_categories
    ]
    for axis in configuration["summaries"]:
        if sum(group["scene_count"] for group in group_summaries if group["axis"] == axis) != len(expected_tokens):
            raise RuntimeError(f"Context bins do not form an exhaustive partition for {axis}")
    score_hash_after = file_sha256(score_path)
    if score_hash_before != score_hash_after:
        raise RuntimeError("Preserved baseline score file changed during analysis")
    report = {
        "purpose": configuration["purpose"],
        "configuration_path": str(configuration_path),
        "configuration_sha256": file_sha256(configuration_path),
        "preserved_score_path": str(score_path),
        "preserved_score_sha256_before": score_hash_before,
        "preserved_score_sha256_after": score_hash_after,
        "metadata_constructed_before_score_join": True,
        "only_current_frame_fields_used_to_define_context": True,
        "expected_scenes": len(expected_tokens),
        "scenes_with_current_metadata": int((~joined_rows["current_frame_metadata_missing"]).sum()),
        "missing_current_metadata_scenes": int(joined_rows["current_frame_metadata_missing"].sum()),
        "missing_or_invalid_speed_scenes": int(joined_rows["speed_bin"].eq("missing_or_invalid").sum()),
        "missing_or_invalid_command_scenes": int(joined_rows["command_category"].eq("missing_or_invalid").sum()),
        "missing_recording_id_scenes": int(joined_rows["native_log_token"].eq("").sum()),
        "native_recording_count": int(joined_rows.loc[joined_rows["native_log_token"].ne(""), "native_log_token"].nunique()),
        "exported_log_count": int(joined_rows.loc[joined_rows["native_log_name"].ne(""), "native_log_name"].nunique()),
        "overall_percent_metrics": {metric: float(joined_rows[metric].mean() * 100) for metric in configuration["metric_names"]},
        "groups": group_summaries,
        "elapsed_cpu_analysis_wall_seconds": time.perf_counter() - start_time,
        "gpu_or_inference_or_training_used": False,
        "conclusions": "Descriptive coverage only, not evidence of context-specific future needs; no navtest-based selector or hyperparameter tuning.",
        "command_semantics": configuration["command_semantic_labels"],
        "command_semantics_evidence": configuration["command_semantics_evidence"],
    }
    result_root.mkdir(parents=True)
    metadata.to_csv(result_root / "current_input_metadata.csv", index=False)
    pd.DataFrame(group_summaries).to_csv(result_root / "context_summary.csv", index=False)
    (result_root / "context_summary.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({key: value for key, value in report.items() if key != "groups"}, indent=2))
    print(pd.DataFrame(group_summaries[:len(speed_categories) + len(command_categories)]).to_string(index=False))


if __name__ == "__main__":
    main()
