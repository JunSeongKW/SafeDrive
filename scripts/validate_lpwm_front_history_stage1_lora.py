"""Fixed development PDMS, future-branch intervention, and Stage1 RGB retention."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import json
import multiprocessing
from pathlib import Path
import sys

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def predict(arguments):
    import contextlib
    import torch
    from torch.utils.data import DataLoader
    sys.path.insert(0, str(PROJECT_ROOT / "src"))
    from planning_aware_future_prediction.object_centric.lpwm_front_history_stage1_lora import FrontHistoryStage1LoRAPlanner
    from lpwm_front_history_data import FrontHistorySceneDataset, observed_frame_cache
    from train_lpwm_drivor_joint import configure_allocator, device_inputs, write_json
    from visualize_lpwm_front_history import snapshot
    from PIL import Image, ImageDraw

    configuration = json.loads(arguments.config.read_text())
    root = PROJECT_ROOT / configuration["output_directory"]
    if arguments.limit_scenes:
        root = arguments.checkpoint.parent
    state = torch.load(arguments.checkpoint, map_location="cpu", weights_only=False)
    output = root / "validation" / f"update_{state['completed_updates']:06d}"
    output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(2)
    torch.cuda.set_device(0)
    configure_allocator(0)
    device = torch.device("cuda", 0)
    model = FrontHistoryStage1LoRAPlanner(PROJECT_ROOT / configuration["stage1_checkpoint"]).to(device)
    model.load_state_dict(state["model"], strict=True)
    model.eval()
    assert model.frozen_native_digest() == state["frozen_native_sha256"]
    records = [record for record in json.loads((PROJECT_ROOT / configuration["manifest"]).read_text())["records"]
               if record["split"] == "development"]
    assert len(records) == 1024
    if arguments.limit_scenes:
        assert 12 <= arguments.limit_scenes <= 1024
        records = records[:arguments.limit_scenes]
    del state
    trajectories, tokens, truth = [], [], []
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        for examples in DataLoader(FrontHistorySceneDataset(records), batch_size=8, num_workers=2):
            features, targets = device_inputs(examples, 0, len(examples["token"]), device, "navsim_v1")
            predictions = model(features)
            trajectories.append(predictions["trajectory"].float().cpu().numpy())
            truth.append(targets["trajectory"].cpu().numpy())
            tokens.extend(examples["token"])
    predictions = np.concatenate(trajectories)
    truth = np.concatenate(truth)
    assert np.isfinite(predictions).all()
    distances = np.linalg.norm(predictions[..., :2] - truth[..., :2], axis=-1)
    np.savez(output / "predictions.npz", tokens=tokens, trajectories=predictions, targets=truth)

    # Change only the future temporal attributes immediately before projection.
    # This tests use of the future branch, not correctness of its predictions.
    def persistent_future(_module, inputs):
        states = inputs[0].reshape(len(inputs[0]), 64, 9, 14)
        return (states[:, :, :1].expand(-1, -1, 9, -1).reshape(len(states), 64, 126),)

    examples = next(iter(DataLoader(FrontHistorySceneDataset(records[:12]), batch_size=12)))
    features, targets = device_inputs(examples, 0, 12, device, "navsim_v1")
    hook = model.particle_encoder.particle_trajectory_projection.register_forward_pre_hook(persistent_future)
    try:
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            persistent = model(features)["trajectory"].float().cpu().numpy()
    finally:
        hook.remove()
    persistent_distance = np.linalg.norm(persistent[..., :2] - truth[:12, ..., :2], axis=-1)
    snapshot(model, records, device, root, output.name, "navsim_v1")

    # RGB reconstruction sees the images being reconstructed. Future forecasts
    # receive only observed frames; future GT is used exclusively for metrics.
    protocol = json.loads((PROJECT_ROOT / "outputs/lpwm_partial_planning_v1/evaluation_protocol.json").read_text())
    old_records = json.loads((PROJECT_ROOT / "outputs/lpwm_navsim_full_posttraining_v2/planning_manifest.json").read_text())["records"]
    old_by_token = {row["current_frame_token"]: row for row in old_records}
    official_records = json.loads((PROJECT_ROOT / "outputs/lpwm_drivor_joint_v1/scene_cache/manifest.json").read_text())["records"]
    official_by_token = {row["token"]: row for row in official_records}
    from lpwm_front_history_data import official_target_cache
    world = model.particle_encoder.world_model
    frames = observed_frame_cache(str(PROJECT_ROOT / "outputs/lpwm_navsim_full_posttraining_v2/rgb_frames.npy"))
    image_rows = []
    world_metrics = []

    @contextlib.contextmanager
    def original_stage1():
        saved = []
        for name, parameter in model.named_parameters():
            if "lora_output_projection" in name or name.startswith("planner.image_backbone.command_modulation.2."):
                saved.append((parameter, parameter.detach().clone()))
                parameter.zero_()
        try:
            yield
        finally:
            for parameter, value in saved:
                parameter.copy_(value)

    with torch.inference_mode():
        for scene_index, token in enumerate(protocol["world_tokens"][:16]):
            reference = old_by_token[token]
            indices = reference["frame_cache_indices"][:12]
            assert min(indices) >= 0
            video = torch.from_numpy(np.array(frames[indices])).to(device).permute(0,3,1,2)[None].float()/255
            target_record = official_by_token[token]
            status = official_target_cache(target_record["cache_directory"])["ego"][target_record["cache_row"]]
            model.particle_encoder.active_command = torch.from_numpy(np.array(status[7:11])).to(device)[None]
            captures = {}
            for label in ("stage1", "stage2"):
                with original_stage1() if label == "stage1" else contextlib.nullcontext():
                    with torch.autocast("cuda", enabled=False):
                        reconstructed = world(video, deterministic=True)["rec_rgb"].reshape_as(video)
                        forecast, _ = world.sample_from_x(video[:, :4].contiguous(), num_steps=8, cond_steps=4,
                            deterministic=True, use_all_ctx=False, return_z=True, n_pred_eq_gt=False)
                captures[label] = (reconstructed[0,3].cpu().numpy(), forecast[0,-1].cpu().numpy())
                world_metrics.append({"token": token, "model": label,
                    "reconstruction_mse": float((reconstructed-video).square().mean()),
                    "future_mse": float((forecast[:, -8:]-video[:, 4:]).square().mean())})
            if scene_index < 4:
                rgb = lambda array: Image.fromarray(np.uint8(np.clip(array.transpose(1,2,0),0,1)*255)).resize((256,256))
                image_rows.append([rgb(video[0,3].cpu().numpy()), rgb(captures["stage1"][0]), rgb(captures["stage2"][0]),
                                   rgb(video[0,-1].cpu().numpy()), rgb(captures["stage1"][1]), rgb(captures["stage2"][1])])
            model.particle_encoder.active_command = None
    canvas = Image.new("RGB", (6*256, 4*280+50), "white")
    drawing = ImageDraw.Draw(canvas)
    for column, caption in enumerate(("Current input", "Stage1 reconstruction", "Stage2 reconstruction", "Future GT (+4s)", "Stage1 forecast", "Stage2 forecast")):
        drawing.text((column*256+4, 5), caption, fill="black")
    for row, images in enumerate(image_rows):
        for column, image in enumerate(images):
            canvas.paste(image, (column*256, row*280+30))
    canvas.save(output / "rgb_stage1_vs_stage2.png")
    world_means = {label: {metric: float(np.mean([row[metric] for row in world_metrics if row["model"] == label]))
                          for metric in ("reconstruction_mse", "future_mse")} for label in ("stage1", "stage2")}
    write_json(output / "prediction_complete.json", {
        "complete": True, "count": len(records), "audit_only": bool(arguments.limit_scenes), "ade_meters": float(distances.mean()), "fde_meters": float(distances[:,-1].mean()),
        "persistent_future_ade_difference_12scenes_meters": float(persistent_distance.mean()-distances[:12].mean()),
        "future_intervention_scope": "future use, not actual future correctness",
        "world_diagnostic_count": 16, "world_means": world_means, "world_records": world_metrics,
        "frozen_native_unchanged": True, "same_dev_panel_as_historical_2stage": True})
    print(str(output), flush=True)


def score(arguments):
    from score_lpwm_drivor_navtest import score_scene
    def write_json(path, content):
        path.write_text(json.dumps(content, indent=2, allow_nan=False) + "\n")
    predictions = np.load(arguments.predictions)
    manifest = json.loads((PROJECT_ROOT / "outputs/lpwm_candidate_teacher_v1/metric_cache_manifest.json").read_text())
    by_token = {row["token"]: row["metric_cache_file"] for row in manifest}
    jobs = [(str(token), trajectory, by_token[str(token)]) for token, trajectory in
            zip(predictions["tokens"], predictions["trajectories"]) if str(token) in by_token]
    assert len(predictions["tokens"]) == 1024 and len(jobs) == 1021
    with ProcessPoolExecutor(max_workers=4, mp_context=multiprocessing.get_context("spawn")) as pool:
        rows = list(pool.map(score_scene, jobs, chunksize=8))
    means = {key: float(np.mean([row[key] for row in rows])) for key in rows[0]
             if key not in ("token", "log_name", "valid")}
    write_json(arguments.predictions.parent / "official_development_scores.json", {"rows": rows})
    inference = json.loads((arguments.predictions.parent / "prediction_complete.json").read_text())
    write_json(arguments.predictions.parent / "validation_complete.json", inference | {
        "passed_execution_and_finiteness": True, "pdms": means["score"]*100,
        "pdms_count": 1021, "failed": 0, "excluded_missing_caches": sorted(set(predictions["tokens"])-set(by_token)),
        "mean_metrics": means, "benchmark_scope": "previously exposed internal navtrain development subset, not full navtest",
        "gate_scope": "execution/finite-output/frozen-native; no arbitrary particle-movement or object-box quality threshold",
        "learning_quality_requires_review": True})


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("predict", "score"), required=True)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--predictions", type=Path)
    parser.add_argument("--limit-scenes", type=int, default=0)
    arguments = parser.parse_args()
    predict(arguments) if arguments.mode == "predict" else score(arguments)
