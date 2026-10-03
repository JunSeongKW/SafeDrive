"""Evaluate LPWM object proxies and strictly past-only rollouts on fixed development clips."""
import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from scipy.optimize import linear_sum_assignment

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
from planning_aware_future_prediction.object_centric.lpwm_bridge import (
    ARTIFACT_ROOT, extrapolated_rotation_homographies, image_tensor, load_official_lpwm,
    particle_geometry, restore_particle_geometry,
)


def object_metrics(centers, particle_boxes, annotated_objects):
    if not annotated_objects:
        return {}, {}
    target_boxes = np.array([obj["box"] for obj in annotated_objects])
    inside = ((centers[:, None] >= target_boxes[None, :, :2]) & (centers[:, None] <= target_boxes[None, :, 2:])).all(-1)
    intersections = np.maximum(0, np.minimum(particle_boxes[:, None, 2:], target_boxes[None, :, 2:]) - np.maximum(particle_boxes[:, None, :2], target_boxes[None, :, :2])).prod(-1)
    unions = np.maximum(0, particle_boxes[:, 2:] - particle_boxes[:, :2]).prod(-1)[:, None] + (target_boxes[:, 2:] - target_boxes[:, :2]).prod(-1)[None] - intersections
    overlap = intersections / np.maximum(unions, 1e-8)
    particle_rows, target_columns = linear_sum_assignment(-overlap)
    assigned_overlaps = overlap[particle_rows, target_columns]
    correspondence = {annotated_objects[column]["track"]: int(row) for row, column, score in zip(particle_rows, target_columns, assigned_overlaps) if score >= .1}
    return {"object_point_coverage": float(inside.any(0).mean()), "particle_center_precision": float(inside.any(1).mean()),
            "object_box_recall_iou_010": float((assigned_overlaps >= .1).sum() / len(target_boxes)),
            "object_box_recall_iou_030": float((assigned_overlaps >= .3).sum() / len(target_boxes)),
            "object_count": len(target_boxes)}, correspondence


def geometry_at_frame(encoded, frame_index, homography, budget):
    centers, sizes, presence, particle_ids = particle_geometry(encoded, frame_index, budget)
    centers, boxes = restore_particle_geometry(centers, sizes, homography)
    return centers, boxes, presence, particle_ids


def evaluate(arguments):
    evaluation_source_sha256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    torch.set_num_threads(4)
    cv2.setNumThreads(1)
    device = f"cuda:{arguments.gpu}"
    assert arguments.gpu in (0, 1)
    torch.cuda.set_device(arguments.gpu)
    os.environ["TORCH_HOME"] = str(ARTIFACT_ROOT / "torch")
    os.chdir(ARTIFACT_ROOT)
    model, _ = load_official_lpwm(device, arguments.checkpoint)
    model.eval()
    from utils.loss_functions import LossLPIPS
    perceptual_loss = LossLPIPS(normalized_rgb=False).to(device).eval().perceptual_loss
    records = [record for record in json.loads((ARTIFACT_ROOT / "clip_manifest.json").read_text())["records"] if record["split"] == "development"]
    evaluation_root = ARTIFACT_ROOT / arguments.evaluation_folder / arguments.name
    evaluation_root.mkdir(parents=True, exist_ok=True)
    if (evaluation_root / "metrics.json").exists():
        raise FileExistsError("Completed evaluation is immutable")
    evaluated_records = []
    with torch.inference_mode():
        for record_index, record in enumerate(records):
            clip = np.load(PROJECT_ROOT / record["clip_path"])
            raw_images = clip["raw_images"][:8]
            stabilized = arguments.variant == "rotation_stabilized"
            model_images = clip["stabilized_images"][:8] if stabilized else raw_images
            geometry_transforms = clip["observed_rotation_homographies"][:8] if stabilized else np.repeat(np.eye(3)[None], 8, axis=0)
            video = image_tensor(model_images, device)[None]
            encoded = model(video, deterministic=True)
            reconstructed = encoded["rec_rgb"].permute(0, 2, 3, 1).cpu().numpy()
            restored_reconstruction = np.array([cv2.warpPerspective(image, np.linalg.inv(transform), (128, 128)) for image, transform in zip(reconstructed, geometry_transforms)])
            # Only observed frames are passed to generation. No future image or future camera pose is accepted.
            observed_video = video[:, :4].contiguous()
            torch.manual_seed(1029)
            predicted_images, future_particles = model.sample_from_x(observed_video, num_steps=4, cond_steps=4,
                deterministic=True, use_all_ctx=False, return_z=True, n_pred_eq_gt=False)
            predicted_images = predicted_images[0, -4:].permute(0, 2, 3, 1).cpu().numpy()
            forecast_transforms = extrapolated_rotation_homographies(clip["camera_rotations"][:4], clip["camera_intrinsics"][:4], 4) if stabilized else np.repeat(np.eye(3)[None], 4, axis=0)
            restored_prediction = np.array([cv2.warpPerspective(image, np.linalg.inv(transform), (128, 128)) for image, transform in zip(predicted_images, forecast_transforms)])
            target_images = raw_images.astype(np.float32) / 255
            forecast_tensor = torch.tensor(restored_prediction, device=device).permute(0, 3, 1, 2)
            target_tensor = image_tensor(raw_images[4:], device)
            metrics = {"reconstruction_mse_full": float(np.mean((restored_reconstruction - target_images)**2)),
                       "forecast_mse_full": float(np.mean((restored_prediction - target_images[4:])**2)),
                       "persistence_mse_full": float(np.mean((target_images[3] - target_images[4:])**2)),
                       "forecast_lpips": float(perceptual_loss(forecast_tensor * 2 - 1, target_tensor * 2 - 1).mean())}
            metrics["reconstruction_lpips"] = float(perceptual_loss(
                torch.tensor(restored_reconstruction, device=device).permute(0, 3, 1, 2) * 2 - 1,
                image_tensor(raw_images, device) * 2 - 1).mean())
            valid_masks = [cv2.warpPerspective(np.ones((128, 128), np.float32), np.linalg.inv(transform), (128, 128))
                           for transform in geometry_transforms]
            metrics["reconstruction_valid_area_fraction"] = float(np.mean(valid_masks))
            all_centers, all_boxes, all_presence, all_ids = [], [], [], []
            correspondence_by_frame = []
            for frame_index in range(8):
                centers, boxes, presence, particle_ids = geometry_at_frame(encoded, frame_index, geometry_transforms[frame_index], 64)
                all_centers.append(centers); all_boxes.append(boxes); all_presence.append(presence); all_ids.append(particle_ids)
                _, correspondence = object_metrics(centers[:16], boxes[:16], record["objects_by_frame"][frame_index])
                correspondence_by_frame.append({track: int(particle_ids[index]) for track, index in correspondence.items()})
                if frame_index == 3:
                    for budget in (8, 16, 32):
                        frame_metrics, _ = object_metrics(centers[:budget], boxes[:budget], record["objects_by_frame"][frame_index])
                        metrics.update({f"top{budget}_{key}": value for key, value in frame_metrics.items()})
            matched_pairs = retained_pairs = 0
            for previous, current in zip(correspondence_by_frame[:-1], correspondence_by_frame[1:]):
                for track in previous.keys() & current.keys():
                    matched_pairs += 1
                    retained_pairs += int(previous[track] == current[track])
            metrics["same_particle_id_fraction"] = retained_pairs / matched_pairs if matched_pairs else None
            metrics["matched_track_pairs"] = matched_pairs
            # Geometry-only controls at the same budget; these cannot establish semantics.
            grid_axis = np.linspace(15.5, 111.5, 4)
            grid_centers = np.array(np.meshgrid(grid_axis, grid_axis)).reshape(2, -1).T
            grid_boxes = np.concatenate([grid_centers - 16, grid_centers + 16], axis=-1)
            grid_metrics, _ = object_metrics(grid_centers, grid_boxes, record["objects_by_frame"][3])
            metrics.update({"grid16_" + key: value for key, value in grid_metrics.items()})
            random_generator = np.random.default_rng(2026)
            random_coverages = []
            for _ in range(100):
                random_centers = random_generator.uniform(0, 127, (16, 2))
                random_metrics, _ = object_metrics(random_centers, np.concatenate([random_centers - 16, random_centers + 16], -1), record["objects_by_frame"][3])
                if random_metrics:
                    random_coverages.append(random_metrics["object_point_coverage"])
            metrics["random16_object_point_coverage"] = float(np.mean(random_coverages)) if random_coverages else None
            masked_raw = raw_images[3].copy()
            occluded_centers = occluded_boxes = np.zeros((0, 2))
            if record["objects_by_frame"][3]:
                masked_object = max(record["objects_by_frame"][3], key=lambda obj: np.prod(np.array(obj["box"])[2:] - np.array(obj["box"])[:2]))
                left, top, right, bottom = np.round(masked_object["box"]).astype(int).clip(0, 128)
                masked_raw[top:bottom, left:right] = 127
                masked_input = cv2.warpPerspective(masked_raw, geometry_transforms[3], (128, 128)) if stabilized else masked_raw
                corrupted_video = observed_video.clone()
                corrupted_video[:, 3] = image_tensor(masked_input[None], device)[0]
                corrupted = model.encode_all(corrupted_video, deterministic=True)
                occluded_centers, occluded_boxes, _, _ = geometry_at_frame(corrupted, 3, geometry_transforms[3], 16)
                occlusion_metrics, _ = object_metrics(occluded_centers, occluded_boxes, [masked_object])
                metrics.update({"controlled_occlusion_" + key: value for key, value in occlusion_metrics.items()})
                feature_delta = (corrupted["mu_features"][0, 3] - encoded["mu_features"][0, 3]).square().mean().sqrt()
                metrics["controlled_occlusion_feature_rms_change"] = float(feature_delta)
            # A synthetic intervention on future frames checks framewise particle causality.
            if record_index == 0:
                altered_video = video.clone()
                altered_video[:, 4:] = 1 - altered_video[:, 4:]
                altered_encoding = model.encode_all(altered_video, deterministic=True)
                leakage_difference = float((altered_encoding["z"][:, :4] - encoded["z"][:, :4]).abs().max())
                assert leakage_difference < 1e-5, leakage_difference
                metrics["future_intervention_observed_particle_max_difference"] = leakage_difference
            np.savez_compressed(evaluation_root / (record["current_frame_token"] + ".npz"),
                                raw_images=raw_images, reconstruction=restored_reconstruction, prediction=restored_prediction,
                                centers=np.array(all_centers), boxes=np.array(all_boxes), presence=np.array(all_presence),
                                particle_ids=np.array(all_ids), masked_image=masked_raw, occluded_centers=occluded_centers,
                                occluded_boxes=occluded_boxes,
                                current_alpha_masks=encoded["alpha_masks"][3].cpu().numpy().astype(np.float16),
                                decoder_particle_ids=torch.topk(encoded["z_base_var"][0, 3].sum(-1), k=model.n_kp_dec, largest=False).indices.cpu().numpy(),
                                current_homography=geometry_transforms[3])
            evaluated_records.append({"token": record["current_frame_token"], "recording_group": record["recording_group"], "scenario": record["scenario"], "metrics": metrics})
            print(f"{arguments.name} {record_index + 1}/{len(records)}", flush=True)
            del encoded, future_particles, predicted_images, video
    summary = {"name": arguments.name, "variant": arguments.variant, "checkpoint": arguments.checkpoint,
               "evaluation_source_sha256": evaluation_source_sha256,
               "forecast_protocol": "deterministic prior rollout, past 4 frames only, no best-of-N, no future-pose input",
               "records": evaluated_records}
    (evaluation_root / "metrics.json").write_text(json.dumps(summary, indent=2) + "\n")
    print("EVAL_DONE", arguments.name, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", required=True)
    parser.add_argument("--variant", choices=("raw", "rotation_stabilized"), required=True)
    parser.add_argument("--checkpoint")
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--evaluation-folder", default="evaluation")
    evaluate(parser.parse_args())
