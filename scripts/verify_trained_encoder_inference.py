"""Replay trained encoders from raw observed images, without a future head."""

import argparse
import hashlib
import json
import os
import pickle
import time
from pathlib import Path

import torch

from diagnose_drive_jepa_learning_limitations import load_official_agent
from evaluate_drive_jepa_region_research_pdm import write_json
from validate_drive_jepa_selective_future_connection import parameter_sha256
from planning_aware_future_prediction.models.intent_conditioned_encoder import (
    IntentConditionedEncoderTail, encode_frozen_prefix, plan_from_encoder_features,
    predict_trajectory_from_observations,
)

WORKSPACE = Path(__file__).resolve().parents[1]
RUN_DIRECTORY = WORKSPACE / "outputs/encoder_future_learning_v1"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--wait-for-training", action="store_true")
    args = parser.parse_args()
    started = time.perf_counter()
    if args.wait_for_training:
        while not all((RUN_DIRECTORY / f"train_worker{worker}/completion.json").exists() for worker in (0, 1)):
            if time.perf_counter() - started > 43800:
                raise RuntimeError("Postflight training wait cap")
            time.sleep(20)
    if os.environ.get("CUDA_VISIBLE_DEVICES") not in ("0", "1"):
        raise RuntimeError("One approved GPU required")
    torch.set_num_threads(2)
    if torch.cuda.mem_get_info()[0] < 26 * 2**30:
        raise RuntimeError("Postflight admission reserve unavailable")
    specification = json.loads((WORKSPACE / "configs/encoder_future_learning/controlled_comparison_v1.json").read_text())
    audit_directory = RUN_DIRECTORY / "trained_inference_audit"
    audit_directory.mkdir(exist_ok=True)
    agent, source = load_official_agent(audit_directory)
    from navsim.common.dataclasses import AgentInput

    original_hash = parameter_sha256(agent._model)
    records = json.loads((WORKSPACE / specification["reused_cache"]).read_text())["records"]
    development_records = [record for record in records if record["split"] == "development"]
    # Sample one real window per observed command, independently of every outcome.
    selected_records = [min((record for record in development_records if record["command_raw_index"] == command),
        key=lambda record: hashlib.sha256(("raw-encoder-audit:" + record["current_frame_token"]).encode()).hexdigest())
        for command in sorted({record["command_raw_index"] for record in development_records})]
    observed_clips, ego_statuses = [], []
    for record in selected_records:
        with (WORKSPACE / "dataset/navsim_logs/trainval" / record["segment_filename"]).open("rb") as stream:
            frames = pickle.load(stream)
        agent_input = AgentInput.from_scene_dict_list(frames[record["start_index"]:record["start_index"] + 4],
            WORKSPACE / "dataset/sensor_blobs/trainval", 4, agent.get_sensor_config())
        features = agent.get_feature_builders()[0].compute_features(agent_input)
        observed_clips.append(torch.stack((features["camera_feature_2"], features["camera_feature_1"]), 1))
        ego_statuses.append(features["status_feature"])
    observed_clips, ego_statuses = torch.stack(observed_clips).cuda(), torch.stack(ego_statuses).cuda()
    results = {}
    with torch.no_grad():
        for condition, options in specification["conditions"].items():
            for seed in specification["seeds"]:
                if torch.cuda.mem_get_info()[0] < 6 * 2**30:
                    raise RuntimeError("Shared GPU reserve reached during postflight")
                name = f"{condition}_seed{seed}"
                encoder = IntentConditionedEncoderTail(agent._model.image_encoder,
                    options.get("num_trainable_encoder_blocks", 2), options["ego_intent"]).cuda().eval()
                checkpoint = torch.load(RUN_DIRECTORY / name / "checkpoint.pt", map_location="cpu", mmap=True)
                encoder.load_state_dict(checkpoint["encoder_state"], strict=True)
                del checkpoint
                replayed = predict_trajectory_from_observations(agent._model, encoder, observed_clips, ego_statuses)["trajectory"]
                report = json.loads((RUN_DIRECTORY / name / "results.json").read_text())
                saved_rows = {row["token"]: row for row in report["evaluations"]["512"]["development"]["windows"]}
                expected = torch.tensor([saved_rows[record["current_frame_token"]]["trajectory"] for record in selected_records], device="cuda")
                torch.testing.assert_close(replayed, expected, atol=1e-4, rtol=1e-4)
                prefix = encode_frozen_prefix(agent._model, observed_clips, num_trainable_blocks=len(encoder.blocks))
                reference_features = encoder(prefix, ego_statuses)
                interventions = []
                for command in (0, 1, 2):
                    changed_status = ego_statuses.clone()
                    changed_status[:, :4] = 0
                    changed_status[:, command] = 1
                    changed_features = encoder(prefix, changed_status)
                    changed_plan = plan_from_encoder_features(agent._model, changed_features, ego_statuses)["trajectory"]
                    feature_changes = (changed_features - reference_features).square().mean((1, 2)).sqrt()
                    trajectory_changes = (changed_plan[..., :2] - replayed[..., :2]).norm(dim=-1).mean(-1)
                    if not options["ego_intent"]:
                        torch.testing.assert_close(changed_features, reference_features, atol=0, rtol=0)
                    interventions.append({"encoder_command_index": command,
                        "feature_rms_changes_by_window": feature_changes.cpu().tolist(),
                        "trajectory_xy_changes_m_by_window": trajectory_changes.cpu().tolist()})
                results[name] = {"raw_image_replay_max_abs_trajectory_error": float((replayed - expected).abs().max()),
                    "no_future_head_constructed": True, "valid_command_interventions": interventions}
                del encoder
                print(f"ENCODER_RAW_REPLAY_COMPLETE {name}", flush=True)
    if parameter_sha256(agent._model) != original_hash:
        raise RuntimeError("Original reference changed during postflight")
    write_json(WORKSPACE / "results/encoder_future_learning_v1/trained_inference_audit.json", {
        "complete": True, "original_model_preserved": True, "methods": results, "source": source,
        "selected_tokens": [record["current_frame_token"] for record in selected_records],
        "valid_command_indices": [0, 1, 2], "planner_status_held_fixed": True,
        "scope": "observed-image deployment equivalence and command sensitivity, not counterfactual correctness",
        "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
        "wall_seconds_including_wait": time.perf_counter() - started})
    print("TRAINED_ENCODER_INFERENCE_AUDIT_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
