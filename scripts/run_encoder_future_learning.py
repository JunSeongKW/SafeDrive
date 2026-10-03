"""Prepare exact prefix caches and run registered planning-facing encoder learning."""

import argparse
import hashlib
import json
import os
import pickle
import signal
import subprocess
import time
from pathlib import Path

import torch

from diagnose_drive_jepa_learning_limitations import load_official_agent
from run_drive_jepa_selection_comparison import summarize_rows, write_json
from validate_drive_jepa_selective_future_connection import file_sha256, parameter_sha256
from planning_aware_future_prediction.models.intent_conditioned_encoder import (
    IntentConditionedEncoderTail, TrainingOnlyFutureHead, compute_selected_latent_loss,
    encode_frozen_prefix, plan_from_encoder_features, pool_spatial_regions,
    select_training_regions, visible_patch_indices,
)

WORKSPACE = Path(__file__).resolve().parents[1]
CONFIGURATION = WORKSPACE / "configs/encoder_future_learning/controlled_comparison_v1.json"
OUTPUT_DIRECTORY = WORKSPACE / "outputs/encoder_future_learning_v1"
STOP_REQUESTED = False


def request_stop(signum, frame):
    global STOP_REQUESTED
    STOP_REQUESTED = True


def guard(specification, started, run_started=None):
    if STOP_REQUESTED:
        raise RuntimeError("Requested stop; completed artifacts preserved")
    if time.perf_counter() - started > specification["maximum_worker_wall_seconds"]:
        raise RuntimeError("Worker wall-time cap")
    if run_started is not None and time.perf_counter() - run_started > specification["maximum_run_wall_seconds"]:
        raise RuntimeError("Registered run wall-time cap")
    if torch.cuda.mem_get_info()[0] < specification["running_reserve_gib"] * 2**30:
        raise RuntimeError("Shared GPU reserve reached; no automatic retry")
    if torch.cuda.max_memory_allocated() > specification["maximum_peak_allocated_gib"] * 2**30:
        raise RuntimeError("Registered process allocation cap")


def build_encoder_and_head(baseline_model, options, seed, specification):
    torch.manual_seed(seed + 61000)
    encoder = IntentConditionedEncoderTail(baseline_model.image_encoder,
        options.get("num_trainable_encoder_blocks", specification["num_trainable_encoder_blocks"]), options["ego_intent"]).cuda().eval()
    torch.manual_seed(seed + 62000)
    prediction_head = TrainingOnlyFutureHead().cuda().eval()
    return encoder, prediction_head


def current_planning_sensitivity(baseline_model, current_features, ego_status):
    current_features = current_features.detach().requires_grad_(True)
    trajectory = plan_from_encoder_features(baseline_model, current_features, ego_status)["trajectory"]
    generator = torch.Generator().manual_seed(314159)
    scores = torch.zeros_like(current_features[..., 0])
    for probe_index in range(2):
        probe_direction = (torch.randint(0, 2, trajectory[..., :2].shape, generator=generator) * 2 - 1).to(trajectory)
        gradient = torch.autograd.grad((trajectory[..., :2] * probe_direction).mean(), current_features,
                                       retain_graph=probe_index == 0)[0]
        scores += (gradient * (current_features - current_features.mean(1, keepdim=True))).abs().mean(-1).detach()
    return pool_spatial_regions(scores[..., None]).squeeze(-1)


def prepare_prefix_cache(agent, specification, output_directory, started):
    from navsim.common.dataclasses import AgentInput

    index = json.loads((WORKSPACE / specification["reused_cache"]).read_text())
    if index["counts"] != specification["expected_counts"]:
        raise RuntimeError("Unexpected source split counts")
    records = index["records"]
    groups = {split: {row["recording_group"] for row in records if row["split"] == split}
              for split in specification["expected_counts"]}
    if groups["train"] & groups["development"]:
        raise RuntimeError("Recording leakage")
    encoder, unused_head = build_encoder_and_head(agent._model, {"ego_intent": False}, 29, specification)
    encoder.requires_grad_(False)
    del unused_head
    cache_directory = output_directory / "prefix_cache"
    cache_directory.mkdir(parents=True, exist_ok=True)
    log_name, log_frames = None, None
    manifest, maximum_difference, cache_bytes = [], 0., 0
    for record_index, record in enumerate(records):
        guard(specification, started)
        token = record["current_frame_token"]
        cache_path = cache_directory / f"{token}.pt"
        metadata_path = cache_path.with_suffix(".json")
        if cache_path.exists() and metadata_path.exists():
            metadata = json.loads(metadata_path.read_text())
            if metadata["sha256"] != file_sha256(cache_path) or metadata["configuration_sha256"] != file_sha256(CONFIGURATION):
                raise RuntimeError("Prepared prefix cache changed")
        else:
            if file_sha256(record["cache_file"]) != record["cache_sha256"]:
                raise RuntimeError("Original teacher cache changed")
            teacher = torch.load(record["cache_file"], map_location="cpu", weights_only=True)
            if log_name != record["segment_filename"]:
                with (WORKSPACE / "dataset/navsim_logs/trainval" / record["segment_filename"]).open("rb") as stream:
                    log_frames = pickle.load(stream)
                log_name = record["segment_filename"]
            observed_frames = log_frames[record["start_index"]:record["start_index"] + 4]
            if observed_frames[-1]["token"] != token:
                raise RuntimeError("Observed input alignment failed")
            agent_input = AgentInput.from_scene_dict_list(observed_frames,
                WORKSPACE / "dataset/sensor_blobs/trainval", 4, agent.get_sensor_config())
            features = agent.get_feature_builders()[0].compute_features(agent_input)
            torch.testing.assert_close(features["status_feature"], teacher["current_ego_status"], atol=0, rtol=0)
            observed_clip = torch.stack((features["camera_feature_2"], features["camera_feature_1"]), 1)[None].cuda()
            ego_status = teacher["current_ego_status"][None].cuda()
            observed_prefix_last6 = encode_frozen_prefix(agent._model, observed_clip, num_trainable_blocks=6)
            with torch.no_grad():
                observed_prefix = observed_prefix_last6
                for block in agent._model.image_encoder.blocks[-6:-2]:
                    observed_prefix = block(observed_prefix, mask=None, attn_mask=None, T=1, H_patches=16, W_patches=32)
            with torch.no_grad():
                reproduced_features = encoder(observed_prefix, ego_status)
                torch.testing.assert_close(reproduced_features[0].cpu(), teacher["current_patch_latents"], atol=1e-5, rtol=1e-5)
                original_trajectory = agent._model(observed_clip, ego_status)["trajectory"]
                reproduced_trajectory = plan_from_encoder_features(agent._model, reproduced_features, ego_status)["trajectory"]
                torch.testing.assert_close(original_trajectory, reproduced_trajectory, atol=1e-5, rtol=1e-5)
            difference = float((reproduced_features[0].cpu() - teacher["current_patch_latents"]).abs().max())
            planning_scores = current_planning_sensitivity(agent._model, reproduced_features, ego_status).cpu()[0]
            history_path = WORKSPACE / specification["reused_history"] / f"{token}.pt"
            history_metadata = json.loads(history_path.with_suffix(".json").read_text())
            if file_sha256(history_path) != history_metadata["cache_sha256"]:
                raise RuntimeError("Original observed motion cache changed")
            history = torch.load(history_path, map_location="cpu", weights_only=True)
            prepared = {
                "observed_prefix_features": observed_prefix[0].cpu(),
                "observed_prefix_features_last6": observed_prefix_last6[0].cpu(),
                "current_teacher_regions": pool_spatial_regions(teacher["current_patch_latents"]),
                "future_teacher_regions": pool_spatial_regions(teacher["future_target_latents"]),
                "future_region_valid_mask": pool_spatial_regions(teacher["future_target_valid_mask"].float()[..., None]).squeeze(-1) == 1,
                "ego_status": teacher["current_ego_status"],
                "ego_trajectory_target": teacher["ego_trajectory_target"],
                "planning_region_scores": planning_scores,
                "motion_region_scores": history["observed_motion_region_scores"],
                "original_trajectory": original_trajectory[0].cpu(),
            }
            for strategy in ("uniform", "planning"):
                masked_prefixes, masked_prefixes_last6, observed_ids, selected_ids = [], [], [], []
                for view_index in range(specification["masked_views_per_strategy"]):
                    digest = hashlib.sha256(f"encoder-mask:{token}:{view_index}".encode()).digest()
                    generator = torch.Generator().manual_seed(int.from_bytes(digest[:8], "little") % (2**63 - 1))
                    selected_regions = select_training_regions(planning_scores[None], strategy,
                        specification["selected_region_budget"], generator).cuda()
                    visible_indices = visible_patch_indices(selected_regions)
                    masked_prefix_last6 = encode_frozen_prefix(agent._model, observed_clip, num_trainable_blocks=6,
                                                               observed_patch_indices=visible_indices)
                    with torch.no_grad():
                        masked_prefix = masked_prefix_last6
                        for block in agent._model.image_encoder.blocks[-6:-2]:
                            masked_prefix = block(masked_prefix, mask=visible_indices, attn_mask=None, T=1, H_patches=16, W_patches=32)
                    if record_index == 0 and strategy == "uniform" and view_index == 0:
                        with torch.no_grad():
                            removed_pixel_mask = torch.ones(1, 512, device="cuda")
                            removed_pixel_mask.scatter_(1, visible_indices, 0)
                            removed_pixel_mask = torch.nn.functional.interpolate(removed_pixel_mask.reshape(1, 1, 16, 32),
                                                                                size=(256, 512), mode="nearest")[:, :, None]
                            changed_clip = observed_clip + removed_pixel_mask * .25
                            changed_prefix = encode_frozen_prefix(agent._model, changed_clip, num_trainable_blocks=6,
                                                                 observed_patch_indices=visible_indices)
                            torch.testing.assert_close(changed_prefix, masked_prefix_last6, atol=0, rtol=0)
                            normalized_clip = agent._model.transform(observed_clip.permute(0, 2, 1, 3, 4).reshape(2, 3, 256, 512))
                            normalized_clip = normalized_clip.reshape(1, 2, 3, 256, 512).permute(0, 2, 1, 3, 4)
                            official_masked_features = agent._model.image_encoder(normalized_clip, masks=visible_indices)
                            reproduced_masked_features = encoder(masked_prefix, ego_status, visible_indices)
                            torch.testing.assert_close(official_masked_features, reproduced_masked_features, atol=1e-5, rtol=1e-5)
                            write_json(output_directory / "input_mask_contract.json", {
                                "changed_removed_pixels_leave_prefix_bitwise_equal": True,
                                "masked_prefix_tail_matches_official_masked_encoder": True,
                                "maximum_masked_feature_difference": float((official_masked_features - reproduced_masked_features).abs().max()),
                                "visible_patch_count": visible_indices.shape[1], "token": token})
                    masked_prefixes.append(masked_prefix[0].cpu())
                    masked_prefixes_last6.append(masked_prefix_last6[0].cpu())
                    observed_ids.append(visible_indices[0].cpu())
                    selected_ids.append(selected_regions[0].cpu())
                prepared[f"{strategy}_masked_prefix_features"] = torch.stack(masked_prefixes)
                prepared[f"{strategy}_masked_prefix_features_last6"] = torch.stack(masked_prefixes_last6)
                prepared[f"{strategy}_visible_patch_indices"] = torch.stack(observed_ids)
                prepared[f"{strategy}_selected_regions"] = torch.stack(selected_ids)
            torch.save(prepared, cache_path)
            metadata = {"token": token, "file": str(cache_path), "sha256": file_sha256(cache_path),
                "configuration_sha256": file_sha256(CONFIGURATION), "original_cache_sha256": record["cache_sha256"],
                "feature_equivalence_max_abs": difference,
                "trajectory_equivalence_max_abs": float((original_trajectory - reproduced_trajectory).abs().max()),
                "masked_before_first_attention": True, "future_input_to_student": False}
            write_json(metadata_path, metadata)
        maximum_difference = max(maximum_difference, metadata["feature_equivalence_max_abs"])
        cache_bytes += cache_path.stat().st_size
        if cache_bytes > specification["maximum_cache_gib"] * 2**30:
            raise RuntimeError("Prefix cache disk budget exceeded")
        manifest.append(metadata)
        if (record_index + 1) % 16 == 0:
            write_json(output_directory / "preparation_progress.json", {"completed": record_index + 1, "total": len(records), "bytes": cache_bytes})
            print(f"PREFIX_PREPARED {record_index + 1}/{len(records)}", flush=True)
    write_json(output_directory / "prefix_cache_manifest.json", {"records": manifest,
        "bytes": cache_bytes, "maximum_feature_difference": maximum_difference,
        "recording_counts": {split: len(names) for split, names in groups.items()},
        "configuration_sha256": file_sha256(CONFIGURATION)})
    print("PREFIX_PREPARATION_COMPLETE", flush=True)


def load_training_cache(specification, output_directory):
    records = json.loads((WORKSPACE / specification["reused_cache"]).read_text())["records"]
    manifest = json.loads((output_directory / "prefix_cache_manifest.json").read_text())
    if manifest["configuration_sha256"] != file_sha256(CONFIGURATION):
        raise RuntimeError("Preparation configuration differs")
    cached_windows = []
    for record, entry in zip(records, manifest["records"]):
        if record["current_frame_token"] != entry["token"] or file_sha256(entry["file"]) != entry["sha256"]:
            raise RuntimeError("Prepared input integrity failure")
        cached_windows.append(torch.load(entry["file"], map_location="cpu", weights_only=True))
    return records, {key: torch.stack([window[key] for window in cached_windows]) for key in cached_windows[0]}


def observed_training_batch(cache, indices, options, update_index, generator, specification):
    training_batch = {key: cache[key][indices].cuda() for key in (
        "ego_status", "ego_trajectory_target", "future_teacher_regions",
        "future_region_valid_mask", "current_teacher_regions")}
    depth_suffix = "_last6" if options.get("num_trainable_encoder_blocks", 2) == 6 else ""
    training_batch["observed_prefix_features"] = cache["observed_prefix_features" + depth_suffix][indices].cuda()
    if options["masked_observation"]:
        strategy = options["targets"]
        view_index = update_index % specification["masked_views_per_strategy"]
        training_batch["auxiliary_prefix_features"] = cache[f"{strategy}_masked_prefix_features{depth_suffix}"][indices, view_index].cuda()
        training_batch["auxiliary_visible_indices"] = cache[f"{strategy}_visible_patch_indices"][indices, view_index].cuda()
        training_batch["selected_regions"] = cache[f"{strategy}_selected_regions"][indices, view_index].cuda()
    else:
        score_key = "motion_region_scores" if options["targets"] == "motion" else "planning_region_scores"
        training_batch["selected_regions"] = select_training_regions(cache[score_key][indices],
            options["targets"], specification["selected_region_budget"], generator).cuda()
    return training_batch


def compute_training_objectives(agent, encoder, prediction_head, training_batch, options):
    encoder_features = encoder(training_batch["observed_prefix_features"], training_batch["ego_status"])
    predictions = plan_from_encoder_features(agent._model, encoder_features, training_batch["ego_status"])
    planning_loss = agent.compute_loss({}, {"trajectory": training_batch["ego_trajectory_target"]}, predictions)
    auxiliary_loss = planning_loss.new_zeros(())
    if options["auxiliary"] != "none":
        if options["masked_observation"]:
            auxiliary_features = encoder(training_batch["auxiliary_prefix_features"], training_batch["ego_status"],
                                         training_batch["auxiliary_visible_indices"])
        else:
            auxiliary_features = encoder_features
        predicted_latents = prediction_head(auxiliary_features, training_batch["selected_regions"],
                                           training_batch.get("auxiliary_visible_indices"))
        teacher_targets = training_batch["future_teacher_regions"]
        if options["auxiliary"] == "current":
            teacher_targets = training_batch["current_teacher_regions"][:, None].expand(-1, 4, -1, -1)
        auxiliary_loss = compute_selected_latent_loss(predicted_latents, teacher_targets,
            training_batch["future_region_valid_mask"], training_batch["selected_regions"])
    return planning_loss, auxiliary_loss


@torch.no_grad()
def evaluate_encoder(agent, encoder, cache, records, split, batch_size=8, collect_features=False):
    indices = [index for index, record in enumerate(records) if record["split"] == split]
    rows, trajectories, targets, compressed_features = [], [], [], []
    # Fixed random projection for the same post-training linear future probe.
    projection = torch.randn(1024, 16, generator=torch.Generator().manual_seed(9173)).cuda() / 32
    for offset in range(0, len(indices), batch_size):
        selected_indices = indices[offset:offset + batch_size]
        ego_status = cache["ego_status"][selected_indices].cuda()
        depth_suffix = "_last6" if len(encoder.blocks) == 6 else ""
        encoder_features = encoder(cache["observed_prefix_features" + depth_suffix][selected_indices].cuda(), ego_status)
        trajectory = plan_from_encoder_features(agent._model, encoder_features, ego_status)["trajectory"].cpu()
        target = cache["ego_trajectory_target"][selected_indices]
        errors = (trajectory[..., :2] - target[..., :2]).norm(dim=-1).mean(-1)
        if collect_features:
            # Pool to4x8 cells after projecting channels: 512 representation dimensions.
            projected = (encoder_features @ projection).reshape(-1, 4, 4, 8, 4, 16).mean((2, 4)).flatten(1)
            compressed_features.append(projected.cpu())
        for position, index in enumerate(selected_indices):
            record = records[index]
            rows.append({"token": record["current_frame_token"], "scene_token": record["scene_token"],
                "recording": record["recording_group"], "command": record["command_raw_index"],
                "ego_speed_meters_per_second": record["ego_speed_meters_per_second"],
                "xy_ade_m": float(errors[position]), "trajectory": trajectory[position].tolist()})
        trajectories.append(trajectory)
        targets.append(target)
    result = {"summary": summarize_rows(rows, torch.cat(trajectories), torch.cat(targets)), "windows": rows}
    return result, torch.cat(compressed_features) if collect_features else None


def gradient_contract(agent, encoder, prediction_head, training_batch, options):
    planning_loss, auxiliary_loss = compute_training_objectives(agent, encoder, prediction_head, training_batch, options)
    named_parameters = [("encoder." + name, parameter) for name, parameter in encoder.named_parameters() if parameter.requires_grad]
    named_parameters += [("prediction_head." + name, parameter) for name, parameter in prediction_head.named_parameters()]
    report = {}
    for objective, loss in (("planning", planning_loss), ("auxiliary", auxiliary_loss)):
        if not loss.requires_grad:
            continue
        gradients = torch.autograd.grad(loss, [parameter for _, parameter in named_parameters],
            allow_unused=True, retain_graph=objective == "planning")
        group_norms = {"encoder_blocks": 0., "encoder_intent": 0., "prediction_head": 0.}
        block_norms = {str(block_index): 0. for block_index in range(len(encoder.blocks))}
        for (name, _), gradient in zip(named_parameters, gradients):
            group = "prediction_head" if name.startswith("prediction_head") else (
                "encoder_intent" if "intent_conditioning" in name else "encoder_blocks")
            if gradient is not None:
                if not torch.isfinite(gradient).all():
                    raise RuntimeError("Non-finite diagnostic gradient")
                group_norms[group] += float(gradient.square().sum())
                if name.startswith("encoder.blocks."):
                    block_norms[name.split(".")[2]] += float(gradient.square().sum())
        report[objective] = {group: squared_norm ** .5 for group, squared_norm in group_norms.items()}
        report[objective]["per_encoder_block_gradient_norm"] = {name: norm ** .5 for name, norm in block_norms.items()}
        if any(norm <= 0 for norm in block_norms.values()):
            raise RuntimeError("An intended trainable encoder block is disconnected")
    if report["planning"]["encoder_blocks"] <= 0 or report["planning"]["prediction_head"] != 0:
        raise RuntimeError("Planner-to-encoder gradient contract failed")
    if options["auxiliary"] != "none" and report["auxiliary"]["encoder_blocks"] <= 0:
        raise RuntimeError("Auxiliary objective does not update encoder")
    if options["ego_intent"] and report["planning"]["encoder_intent"] <= 0:
        raise RuntimeError("Intent conditioning is disconnected")
    return report


def save_checkpoint(path, encoder, prediction_head, optimizer, update_index, schedule_hash, specification):
    checkpoint = {"encoder_state": encoder.state_dict(), "prediction_head_state": prediction_head.state_dict(),
        "optimizer": optimizer.state_dict(), "completed_updates": update_index, "batch_schedule_sha256": schedule_hash,
        "configuration": specification, "torch_rng_state": torch.get_rng_state(),
        "cuda_rng_state": torch.cuda.get_rng_state()}
    staging = path.with_suffix(".partial")
    torch.save(checkpoint, staging)
    staging.replace(path)


def run_training_condition(agent, cache, records, specification, output_directory, condition, seed, started):
    options = specification["conditions"][condition]
    run_directory = output_directory / f"{condition}_seed{seed}"
    if (run_directory / "results.json").exists():
        print(f"PRESERVED_COMPLETE {condition} seed{seed}", flush=True)
        return
    run_directory.mkdir(exist_ok=True)
    if (run_directory / "checkpoint.pt").exists():
        raise RuntimeError("Interrupted run exists; explicit resume implementation required; no overwriting")
    encoder, prediction_head = build_encoder_and_head(agent._model, options, seed, specification)
    original_encoder_hash = parameter_sha256(encoder)
    encoder_parameters = [parameter for name, parameter in encoder.named_parameters()
                          if parameter.requires_grad and not name.startswith("intent_conditioning")]
    parameter_groups = [{"params": encoder_parameters, "lr": specification["encoder_learning_rate"]}]
    if options["ego_intent"]:
        parameter_groups.append({"params": encoder.intent_conditioning.parameters(), "lr": specification["intent_learning_rate"]})
    if options["auxiliary"] != "none":
        parameter_groups.append({"params": prediction_head.parameters(), "lr": specification["prediction_learning_rate"]})
    optimizer = torch.optim.AdamW(parameter_groups, weight_decay=specification["weight_decay"])
    train_indices = [index for index, row in enumerate(records) if row["split"] == "train"]
    schedule_generator = torch.Generator().manual_seed(seed + 10000)
    schedule = [torch.tensor(train_indices)[torch.randperm(len(train_indices), generator=schedule_generator)
                [:specification["batch_size"]]].tolist() for _ in range(specification["joint_updates"])]
    schedule_hash = hashlib.sha256(json.dumps(schedule).encode()).hexdigest()
    target_generator = torch.Generator().manual_seed(seed + 20000)
    diagnostic_indices = [index for index in train_indices if cache["future_region_valid_mask"][index].all()][:specification["batch_size"]]
    diagnostic_batch = observed_training_batch(cache, diagnostic_indices, options, 0,
        torch.Generator().manual_seed(seed + 30000), specification)
    contract = gradient_contract(agent, encoder, prediction_head, diagnostic_batch, options)
    write_json(run_directory / "gradient_contract.json", contract)
    del diagnostic_batch
    torch.cuda.reset_peak_memory_stats()
    run_started = time.perf_counter()
    training_curve = []
    completed_updates = 0
    try:
        for update_index, indices in enumerate(schedule):
            guard(specification, started, run_started)
            training_batch = observed_training_batch(cache, indices, options, update_index, target_generator, specification)
            optimizer.zero_grad(set_to_none=True)
            planning_loss, auxiliary_loss = compute_training_objectives(agent, encoder, prediction_head, training_batch, options)
            total_loss = planning_loss + specification["auxiliary_weight"] * auxiliary_loss
            if not torch.isfinite(total_loss):
                raise RuntimeError("Non-finite training loss")
            total_loss.backward()
            gradient_norm = torch.nn.utils.clip_grad_norm_(
                [parameter for group in optimizer.param_groups for parameter in group["params"]], specification["gradient_clip_norm"],
                error_if_nonfinite=True)
            optimizer.step()
            completed_updates = update_index + 1
            if completed_updates == 1 or completed_updates % 32 == 0:
                row = {"update": completed_updates, "planning_loss": float(planning_loss),
                    "auxiliary_loss": float(auxiliary_loss), "gradient_norm": float(gradient_norm),
                    "elapsed_seconds": time.perf_counter() - run_started}
                training_curve.append(row)
                write_json(run_directory / "progress.json", row)
                print(f"ENCODER_TRAIN {condition} seed{seed} {completed_updates}/{len(schedule)} "
                      f"plan={float(planning_loss):.6f} aux={float(auxiliary_loss):.6f}", flush=True)
            if completed_updates % 128 == 0:
                save_checkpoint(run_directory / "checkpoint.pt", encoder, prediction_head, optimizer,
                                completed_updates, schedule_hash, specification)
    finally:
        save_checkpoint(run_directory / "checkpoint.pt", encoder, prediction_head, optimizer,
                        completed_updates, schedule_hash, specification)
        write_json(run_directory / "training_curve.json", training_curve)
    if completed_updates != specification["joint_updates"]:
        raise RuntimeError("Incomplete run")
    training_seconds = time.perf_counter() - run_started
    train_result, train_features = evaluate_encoder(agent, encoder, cache, records, "train", collect_features=True)
    development_result, development_features = evaluate_encoder(agent, encoder, cache, records, "development", collect_features=True)
    indices = diagnostic_indices
    with torch.no_grad():
        depth_suffix = "_last6" if len(encoder.blocks) == 6 else ""
        observed_prefix = cache["observed_prefix_features" + depth_suffix][indices].cuda()
        original_status = cache["ego_status"][indices].cuda()
        original_features = encoder(observed_prefix, original_status)
        reference_features = observed_prefix
        block_update_rms = []
        for trained_block, reference_block in zip(encoder.blocks, agent._model.image_encoder.blocks[-len(encoder.blocks):]):
            reference_features = reference_block(reference_features, mask=None, attn_mask=None,
                                                  T=1, H_patches=16, W_patches=32)
            squared_update, parameter_count = 0., 0
            for trained_parameter, reference_parameter in zip(trained_block.parameters(), reference_block.parameters()):
                squared_update += float((trained_parameter - reference_parameter).square().sum())
                parameter_count += trained_parameter.numel()
            block_update_rms.append((squared_update / parameter_count) ** .5)
        reference_features = agent._model.image_encoder.norm(reference_features)
        if any(value <= 0 for value in block_update_rms):
            raise RuntimeError("An intended encoder block was not updated")
        changed_status = original_status.clone()
        changed_status[:, :4] = changed_status[:, :4].roll(1, 1)
        changed_features = encoder(observed_prefix, changed_status)
        original_plan = plan_from_encoder_features(agent._model, original_features, original_status)["trajectory"]
        changed_plan = plan_from_encoder_features(agent._model, changed_features, original_status)["trajectory"]
        intervention = {"same_observed_prefix": True, "planner_status_held_fixed": True,
            "encoder_feature_rms_change": float((changed_features - original_features).square().mean().sqrt()),
            "adapted_feature_rms_change_from_original": float((original_features - reference_features).square().mean().sqrt()),
            "per_encoder_block_weight_update_rms": block_update_rms,
            "trajectory_xy_mean_change_m": float((changed_plan[..., :2] - original_plan[..., :2]).norm(dim=-1).mean()),
            "scope": "command sensitivity only; not counterfactual correctness"}
    updated_encoder_hash = parameter_sha256(encoder)
    if original_encoder_hash == updated_encoder_hash:
        raise RuntimeError("Encoder did not change")
    torch.save({"train_features": train_features, "development_features": development_features}, run_directory / "representation_probe_features.pt")
    report = {"condition": condition, "seed": seed, "completed_updates": completed_updates,
        "configuration_sha256": file_sha256(CONFIGURATION), "batch_schedule_sha256": schedule_hash,
        "original_encoder_hash": original_encoder_hash, "updated_encoder_hash": updated_encoder_hash,
        "trainable_encoder_parameters": sum(parameter.numel() for parameter in encoder.parameters() if parameter.requires_grad),
        "gradient_contract": contract, "intent_intervention": intervention,
        "training_seconds": training_seconds, "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
        "evaluations": {str(completed_updates): {"train": train_result, "development": development_result}}}
    write_json(run_directory / "results.json", report)
    print(f"ENCODER_RUN_COMPLETE {condition} seed{seed} seconds={training_seconds:.1f}", flush=True)
    del optimizer, encoder, prediction_head
    torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("prepare", "smoke", "train"))
    parser.add_argument("--worker-index", type=int, default=0)
    parser.add_argument("--worker-count", type=int, default=1)
    args = parser.parse_args()
    specification = json.loads(CONFIGURATION.read_text())
    OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(2)
    if os.environ.get("CUDA_VISIBLE_DEVICES") not in ("0", "1") or not torch.cuda.is_available():
        raise RuntimeError("Exactly one approved physical GPU 0 or1 required")
    if torch.cuda.mem_get_info()[0] < specification["minimum_free_gpu_gib"] * 2**30:
        raise RuntimeError("GPU admission reserve unavailable")
    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)
    started = time.perf_counter()
    worker_directory = OUTPUT_DIRECTORY / f"{args.mode}_worker{args.worker_index}"
    worker_directory.mkdir(exist_ok=True)
    agent, source = load_official_agent(worker_directory)
    original_hash = parameter_sha256(agent._model)
    if original_hash != "05af3ccd8a8e7db1b9b2ae2e77f81ef753446a3309989cef08c161ce0614476a":
        raise RuntimeError("Original model parameters differ")
    write_json(worker_directory / "provenance.json", {"source": source, "configuration": specification,
        "configuration_sha256": file_sha256(CONFIGURATION), "original_model_hash": original_hash,
        "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=WORKSPACE, text=True).strip(),
        "implementation_sha256": {str(path.relative_to(WORKSPACE)): file_sha256(path) for path in (
            Path(__file__).resolve(), WORKSPACE / "src/planning_aware_future_prediction/models/intent_conditioned_encoder.py")},
        "physical_gpu": os.environ["CUDA_VISIBLE_DEVICES"], "torch_version": torch.__version__})
    if args.mode == "prepare":
        prepare_prefix_cache(agent, specification, OUTPUT_DIRECTORY, started)
    else:
        records, cache = load_training_cache(specification, OUTPUT_DIRECTORY)
        initial_encoder, prediction_head = build_encoder_and_head(agent._model, {"ego_intent": False}, 29, specification)
        if args.mode == "smoke":
            smoke_reports = {}
            for condition, options in specification["conditions"].items():
                encoder, head = build_encoder_and_head(agent._model, options, 29, specification)
                selected_indices = [index for index, record in enumerate(records) if record["split"] == "train"
                                    and cache["future_region_valid_mask"][index].all()][:8]
                training_batch = observed_training_batch(cache, selected_indices, options, 0,
                    torch.Generator().manual_seed(29), specification)
                run_started = time.perf_counter()
                contract = gradient_contract(agent, encoder, head, training_batch, options)
                smoke_reports[condition] = {"gradient_contract": contract, "seconds": time.perf_counter() - run_started}
                del encoder, head, training_batch
            write_json(OUTPUT_DIRECTORY / "smoke.json", {"conditions": smoke_reports,
                "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30})
        else:
            if not (OUTPUT_DIRECTORY / "smoke.json").is_file():
                raise RuntimeError("GPU gradient smoke gate not completed")
            if args.worker_index == 0:
                baseline_result, baseline_development_features = evaluate_encoder(agent, initial_encoder, cache, records, "development", collect_features=True)
                _, baseline_train_features = evaluate_encoder(agent, initial_encoder, cache, records, "train", collect_features=True)
                write_json(OUTPUT_DIRECTORY / "original_baseline.json", baseline_result)
                torch.save({"train_features": baseline_train_features, "development_features": baseline_development_features},
                           OUTPUT_DIRECTORY / "original_representation_probe_features.pt")
            del initial_encoder, prediction_head
            runs = [(condition, seed) for condition in specification["conditions"] for seed in specification["seeds"]]
            for run_index, (condition, seed) in enumerate(runs):
                if run_index % args.worker_count == args.worker_index:
                    run_training_condition(agent, cache, records, specification, OUTPUT_DIRECTORY, condition, seed, started)
    if parameter_sha256(agent._model) != original_hash:
        raise RuntimeError("Original reference model changed")
    write_json(worker_directory / "completion.json", {"complete": True, "original_model_preserved": True,
        "wall_seconds": time.perf_counter() - started})
    print(f"ENCODER_{args.mode.upper()}_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
