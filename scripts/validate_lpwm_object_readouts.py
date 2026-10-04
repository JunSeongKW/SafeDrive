"""Frozen LPWM diagnostic queue: observed GT localization, linear probes, paired dev report.

This does not modify Stage1, supervise LPWM, evaluate instance discovery, or launch Stage2.
"""

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import os
from pathlib import Path
import pickle
import subprocess
import sys
import time

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
from planning_aware_future_prediction.object_centric.object_readout_diagnostics import (
    classification_metrics, fit_ridge_readout, paired_recording_interval,
    pool_particle_attributes, predict_ridge_readout, split_probe_recordings,
)


def digest(path):
    checksum = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 ** 2), b""):
            checksum.update(chunk)
    return checksum.hexdigest()


def write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_suffix(path.suffix + ".pending")
    pending.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    pending.replace(path)


def finite_column_means(values):
    return [float(column[np.isfinite(column)].mean()) if np.isfinite(column).any() else None
            for column in np.asarray(values).T]


def load_configuration(path):
    configuration = json.loads(path.read_text())
    return configuration, PROJECT_ROOT / configuration["output_directory"]


def prepare_annotations(configuration, output_root):
    destination = output_root / "observed_object_manifest.json"
    if destination.exists():
        return
    from planning_aware_future_prediction.adapters.navsim_tracked_state import annotation_states_in_current_ego_frame
    from planning_aware_future_prediction.adapters.navsim_visual_entities import project_lidar_box_to_front_roi
    source = PROJECT_ROOT / configuration["stage1_directory"] / "manifest.json"
    manifest = json.loads(source.read_text())
    training_groups = {row["recording_group"] for row in manifest["records"] if row["split"] == "train"}
    fitting_groups, validation_groups = split_probe_recordings(training_groups,
        configuration["probe_validation_recording_fraction"], configuration["split_seed"])
    development_groups = {row["recording_group"] for row in manifest["records"] if row["split"] == "development"}
    assert not training_groups & development_groups
    grouped = defaultdict(list)
    for row in manifest["records"]:
        grouped[row["segment_filename"]].append(row)
    clip_records, object_records = [], []
    started = time.monotonic()
    for segment_index, (filename, rows) in enumerate(sorted(grouped.items())):
        with (PROJECT_ROOT / "dataset/navsim_logs/trainval" / filename).open("rb") as stream:
            log_frames = pickle.load(stream)
        projection_cache = {}
        for row in rows:
            observed = log_frames[row["start_index"]:row["start_index"] + configuration["observed_frames"]]
            assert observed[-1]["token"] == row["current_frame_token"]
            observations_by_track = []
            for frame in observed:
                if frame["token"] not in projection_cache:
                    projected = {}
                    camera = frame["cams"]["CAM_F0"]
                    for annotation_index, (box, category, track) in enumerate(zip(frame["anns"]["gt_boxes"],
                            frame["anns"]["gt_names"], frame["anns"]["track_tokens"])):
                        if category not in configuration["categories"]:
                            continue
                        pixel_box, valid = project_lidar_box_to_front_roi(box, camera)
                        pixel_box = pixel_box * np.array([.25, .5, .25, .5])
                        if not valid or np.min(pixel_box[2:] - pixel_box[:2]) < configuration["minimum_projected_side_pixels"]:
                            continue
                        camera_center = (box[:3] - camera["sensor2lidar_translation"]) @ np.linalg.inv(camera["sensor2lidar_rotation"]).T
                        projected[str(track)] = {"box": pixel_box.tolist(), "category": str(category),
                            "annotation_index": annotation_index, "camera_depth": float(camera_center[2])}
                    projection_cache[frame["token"]] = projected
                observations_by_track.append(projection_cache[frame["token"]])
            current_states = annotation_states_in_current_ego_frame(observed[-1], observed[-1])
            row_start = len(object_records)
            for track, annotation in sorted(observations_by_track[-1].items()):
                state = current_states[annotation["annotation_index"]]
                history_valid = [track in objects for objects in observations_by_track]
                history_boxes = [objects[track]["box"] if valid else [0., 0., 0., 0.]
                                 for objects, valid in zip(observations_by_track, history_valid)]
                state_values = [*state[:2], *state[6:8], annotation["camera_depth"]]
                object_records.append({"clip_index": len(clip_records), "token": row["current_frame_token"],
                    "track": track, "recording_group": row["recording_group"], "split": row["split"],
                    "probe_partition": "development" if row["split"] == "development" else (
                        "probe_validation" if row["recording_group"] in validation_groups else "probe_fit"),
                    "category": configuration["categories"].index(annotation["category"]),
                    "state_target": [float(value) if np.isfinite(value) else None for value in state_values],
                    "observed_boxes": history_boxes, "observed_valid": history_valid,
                    "scenario": row["scenario"],
                    "small": bool(np.prod(np.subtract(annotation["box"][2:], annotation["box"][:2])) < configuration["small_box_area_pixels"]),
                    "far": bool(annotation["camera_depth"] >= configuration["far_depth_meters"])})
            clip_records.append({"token": row["current_frame_token"], "frame_cache_indices": row["frame_cache_indices"][:4],
                                 "object_start": row_start, "object_end": len(object_records)})
        if segment_index % 20 == 0:
            print("ANNOTATIONS", len(clip_records), len(object_records), flush=True)
            write_json(output_root / "annotation_progress.json", {"clips": len(clip_records),
                "objects": len(object_records), "seconds": time.monotonic() - started})
    assert len(clip_records) == len(manifest["records"])
    write_json(destination, {"clips": clip_records, "objects": object_records,
        "source_manifest_sha256": digest(source), "fitting_recordings": sorted(fitting_groups),
        "validation_recordings": sorted(validation_groups), "development_recordings": sorted(development_groups),
        "category_order": configuration["categories"],
        "state_target_order": ["ego_x_m", "ego_y_m", "ego_vx_mps", "ego_vy_mps", "camera_depth_m"],
        "counts": dict(Counter(f"{row['split']}/{configuration['categories'][row['category']]}" for row in object_records)),
        "seconds": time.monotonic() - started,
        "scope": "Observed GT boxes and track correspondences are privileged probe associations, not LPWM predictions."})


def checkpoint_path(configuration, model_name):
    checkpoint = configuration["checkpoints"][model_name]
    path = PROJECT_ROOT / checkpoint["path"]
    if "glob" in checkpoint:
        candidates = list(path.glob(checkpoint["glob"]))
        assert len(candidates) == 1
        path = candidates[0]
    assert digest(path) == checkpoint["sha256"]
    return path


def extract_representations(configuration, output_root, model_name, gpu, smoke=False):
    import torch
    from planning_aware_future_prediction.object_centric.lpwm_bridge import load_official_lpwm
    if gpu not in configuration["resources"]["allowed_gpus"]:
        raise ValueError("Unapproved GPU")
    resource = configuration["resources"]
    torch.set_num_threads(resource["cpu_threads"])
    torch.cuda.set_device(gpu)
    device = torch.device(f"cuda:{gpu}")
    free_bytes, _ = torch.cuda.mem_get_info(device)
    if free_bytes < (resource["minimum_free_gib"] + resource["maximum_allocated_gib"]) * 1024 ** 3:
        raise RuntimeError("Insufficient free GPU memory for guarded read-only diagnostic")
    destination = output_root / (model_name + ("_smoke" if smoke else ""))
    if (destination / "complete.json").exists():
        return
    destination.mkdir(parents=True, exist_ok=True)
    model, _ = load_official_lpwm(device, checkpoint_path(configuration, model_name))
    model.requires_grad_(False).eval()
    model.timestep_horizon = configuration["observed_frames"] - 1
    if smoke:
        # A full-image ROI checks execution/memory only; no semantic score is produced.
        original_manifest = json.loads((PROJECT_ROOT / configuration["stage1_directory"] / "manifest.json").read_text())
        clips = [{"token": row["current_frame_token"], "frame_cache_indices": row["frame_cache_indices"][:4],
                  "object_start": index, "object_end": index + 1}
                 for index, row in enumerate(original_manifest["records"][:resource["clips_per_batch"]])]
        objects = [{"observed_boxes": [[0., 0., 128., 128.]] * 4, "split": "train"} for _ in clips]
    else:
        manifest = json.loads((output_root / "observed_object_manifest.json").read_text())
        objects = manifest["objects"]
        clips = manifest["clips"]
    object_count = clips[-1]["object_end"]
    frames = np.load(PROJECT_ROOT / configuration["stage1_directory"] / "rgb_frames.npy", mmap_mode="r")
    geometry = np.zeros((object_count, 4, 6), np.float32)
    appearance = np.zeros((object_count, 4, model.learned_feature_dim), np.float32)
    background = np.zeros((object_count, 4, model.learned_bg_feature_dim), np.float32)
    support = np.zeros((object_count, 4), np.float32)
    effective = np.zeros_like(support)
    started = time.monotonic()
    gallery_count = 0
    with torch.inference_mode():
        for offset in range(0, len(clips), resource["clips_per_batch"]):
            selected = clips[offset:offset + resource["clips_per_batch"]]
            images = np.stack([frames[row["frame_cache_indices"]] for row in selected])
            video = torch.from_numpy(images).to(device).permute(0, 1, 4, 2, 3).float() / 255
            encoded = model.encode_all(video * 2 - 1 if model.normalize_rgb else video, deterministic=True)
            assert encoded["z"].shape[2] == configuration["expected_particles"]
            decoded = model.decode_all(encoded["z"], encoded["z_scale"], encoded["z_features"],
                encoded["obj_on"], encoded["z_depth"], encoded["z_bg_features"], encoded["z_context"], filter_key=None)
            alpha = decoded["alpha_masks"].reshape(len(selected), 4, configuration["expected_particles"], 128, 128).cpu().numpy()
            particle_geometry = torch.cat([encoded["z"], encoded["z_scale"].sigmoid(), encoded["z_depth"],
                encoded["obj_on"].unsqueeze(-1) if encoded["obj_on"].ndim == 3 else encoded["obj_on"]], -1).cpu().numpy()
            particle_appearance = encoded["z_features"].cpu().numpy()
            background_features = encoded["z_bg_features"].cpu().numpy()
            assert not any(parameter.requires_grad for parameter in model.parameters())
            for batch_index, clip in enumerate(selected):
                first, last = clip["object_start"], clip["object_end"]
                clip_objects = objects[first:last]
                if not clip_objects:
                    continue
                for frame_index in range(4):
                    attributes = np.concatenate([particle_geometry[batch_index, frame_index],
                                                  particle_appearance[batch_index, frame_index]], -1)
                    pooled, coverage, particle_count = pool_particle_attributes(alpha[batch_index, frame_index],
                        attributes, [row["observed_boxes"][frame_index] for row in clip_objects])
                    geometry[first:last, frame_index] = pooled[:, :6]
                    appearance[first:last, frame_index] = pooled[:, 6:]
                    background[first:last, frame_index] = background_features[batch_index, frame_index]
                    support[first:last, frame_index] = coverage
                    effective[first:last, frame_index] = particle_count
                if gallery_count < 8 and clip_objects[0]["split"] == "development":
                    np.savez_compressed(destination / f"alpha_{clip['token']}.npz", observed_images=images[batch_index],
                        alpha_contributions=alpha[batch_index].astype(np.float16),
                        particle_geometry=particle_geometry[batch_index], particle_appearance=particle_appearance[batch_index],
                        observed_object_boxes=np.asarray([row["observed_boxes"] for row in clip_objects]))
                    gallery_count += 1
            del video, encoded, decoded, alpha
            if offset == 0 or offset % 128 == 0 or offset + len(selected) == len(clips):
                torch.cuda.synchronize(device)
                free_bytes, _ = torch.cuda.mem_get_info(device)
                peak = torch.cuda.max_memory_allocated(device) / 1024 ** 3
                if free_bytes < resource["minimum_free_gib"] * 1024 ** 3 or peak > resource["maximum_allocated_gib"]:
                    raise RuntimeError("Readout extraction memory guard reached; no automatic retry")
                progress = {"clips": offset + len(selected), "total_clips": len(clips), "seconds": time.monotonic() - started,
                    "peak_allocated_gib": peak, "free_gib": free_bytes / 1024 ** 3, "model": model_name,
                    "lpwm_optimizer_updates": 0, "observed_frames_only": True}
                write_json(destination / "progress.json", progress)
                print("EXTRACTION", json.dumps(progress), flush=True)
    np.savez_compressed(destination / "features.npz", particle_geometry=geometry, particle_appearance=appearance,
                        background=background, alpha_coverage=support, effective_particles=effective)
    assert digest(checkpoint_path(configuration, model_name)) == configuration["checkpoints"][model_name]["sha256"]
    write_json(destination / "complete.json", {**progress, "objects": object_count,
        "checkpoint_sha256": configuration["checkpoints"][model_name]["sha256"],
        "feature_sha256": digest(destination / "features.npz"), "smoke_only": smoke})


def feature_matrix(features, objects, readout_name, seed):
    valid = np.asarray([row["observed_valid"] for row in objects], dtype=np.float64)
    geometry = features["particle_geometry"].reshape(len(objects), -1)
    appearance = features["particle_appearance"].reshape(len(objects), -1)
    background = features["background"].reshape(len(objects), -1)
    if readout_name == "roi_geometry":
        selected = np.asarray([row["observed_boxes"] for row in objects]).reshape(len(objects), -1) / 128
    elif readout_name == "particle_geometry":
        selected = geometry
    elif readout_name == "particle_appearance":
        selected = appearance
    elif readout_name == "particle_combined":
        selected = np.concatenate([geometry, appearance], -1)
    elif readout_name == "background":
        selected = background
    elif readout_name == "combined_with_background":
        selected = np.concatenate([geometry, appearance, background], -1)
    elif readout_name == "shuffled_appearance":
        selected = appearance.copy()
        generator = np.random.default_rng(seed)
        partitions = np.asarray([row["probe_partition"] for row in objects])
        for partition in sorted(set(partitions)):
            indices = np.flatnonzero(partitions == partition)
            selected[indices] = appearance[generator.permutation(indices)]
    else:
        raise ValueError(readout_name)
    return np.concatenate([selected, valid], -1)


def compute_probe_report(configuration, output_root, model_name):
    manifest = json.loads((output_root / "observed_object_manifest.json").read_text())
    objects = manifest["objects"]
    feature_file = output_root / model_name / "features.npz"
    features = np.load(feature_file)
    partition = np.asarray([row["probe_partition"] for row in objects])
    fitting = partition == "probe_fit"
    validation = partition == "probe_validation"
    development = partition == "development"
    training = fitting | validation
    categories = np.asarray([row["category"] for row in objects])
    targets = np.asarray([row["state_target"] for row in objects], dtype=np.float64)
    number_of_classes = len(configuration["categories"])
    predictions = {"object_indices": np.flatnonzero(development), "categories": categories[development], "state_targets": targets[development]}
    results = {}
    for readout_name in configuration["readouts"]:
        inputs = feature_matrix(features, objects, readout_name, configuration["split_seed"])
        one_hot = np.eye(number_of_classes)[categories]
        fitting_counts = np.bincount(categories[fitting], minlength=number_of_classes)
        train_counts = np.bincount(categories[training], minlength=number_of_classes)
        if (fitting_counts == 0).any():
            raise RuntimeError("A class has no probe fitting examples; report coverage before fitting")
        candidate_scores = []
        for strength in configuration["ridge_regularization_candidates"]:
            readout = fit_ridge_readout(inputs[fitting], one_hot[fitting], strength, 1 / fitting_counts[categories[fitting]])
            predicted = predict_ridge_readout(readout, inputs[validation]).argmax(-1)
            metric = classification_metrics(categories[validation], predicted, number_of_classes)
            candidate_scores.append(metric["macro_f1_present_classes"])
        selected_strength = configuration["ridge_regularization_candidates"][int(np.argmax(candidate_scores))]
        classifier = fit_ridge_readout(inputs[training], one_hot[training], selected_strength, 1 / train_counts[categories[training]])
        class_predictions = predict_ridge_readout(classifier, inputs[development]).argmax(-1)
        predictions[readout_name + "_category"] = class_predictions
        regression_predictions = np.full((int(development.sum()), targets.shape[1]), np.nan)
        regression_strengths = []
        fitted_regressors = []
        for target_index in range(targets.shape[1]):
            finite = np.isfinite(targets[:, target_index])
            if not (fitting & finite).any() or not (validation & finite).any():
                raise RuntimeError("Insufficient finite state supervision for probe")
            errors = []
            for strength in configuration["ridge_regularization_candidates"]:
                regressor = fit_ridge_readout(inputs[fitting & finite], targets[fitting & finite, target_index], strength)
                predicted = predict_ridge_readout(regressor, inputs[validation & finite]).ravel()
                errors.append(float(np.mean(np.abs(predicted - targets[validation & finite, target_index]))))
            strength = configuration["ridge_regularization_candidates"][int(np.argmin(errors))]
            regressor = fit_ridge_readout(inputs[training & finite], targets[training & finite, target_index], strength)
            regression_predictions[:, target_index] = predict_ridge_readout(regressor, inputs[development]).ravel()
            regression_strengths.append(strength)
            fitted_regressors.append(regressor)
        predictions[readout_name + "_state"] = regression_predictions
        class_report = classification_metrics(categories[development], class_predictions, number_of_classes)
        absolute_errors = np.abs(regression_predictions - targets[development])
        strata = {"all": np.ones(int(development.sum()), dtype=bool)}
        development_objects = [row for row in objects if row["split"] == "development"]
        for category_index, name in enumerate(configuration["categories"]):
            strata[name] = categories[development] == category_index
        for key in ("small", "far"):
            strata[key] = np.asarray([row[key] for row in development_objects])
        for scenario in sorted({row["scenario"] for row in development_objects}):
            strata[scenario] = np.asarray([row["scenario"] == scenario for row in development_objects])
        stratified = {}
        for stratum, selected in strata.items():
            stratified[stratum] = {"objects": int(selected.sum()),
                "category": classification_metrics(categories[development][selected], class_predictions[selected], number_of_classes),
                "state_mae": finite_column_means(absolute_errors[selected]) if selected.any() else None}
        results[readout_name] = {"input_dimensions": inputs.shape[1], "category_regularization": selected_strength,
            "regression_regularization": regression_strengths, "category": class_report,
            "state_mae": finite_column_means(absolute_errors), "strata": stratified}
        np.savez_compressed(output_root / model_name / f"readout_{readout_name}.npz",
            **{f"category_{key}": value for key, value in classifier.items()},
            **{f"state_{index}_{key}": value for index, regressor in enumerate(fitted_regressors) for key, value in regressor.items()})
        print("READOUT", model_name, readout_name, class_report["macro_f1_present_classes"], flush=True)
    np.savez_compressed(output_root / model_name / "development_predictions.npz", **predictions)
    report = {"model": model_name, "checkpoint_sha256": configuration["checkpoints"][model_name]["sha256"],
        "counts": manifest["counts"], "probe_records": dict(Counter(partition)),
        "train_recordings": len(manifest["fitting_recordings"]) + len(manifest["validation_recordings"]),
        "development_recordings": len(manifest["development_recordings"]),
        "state_target_order": manifest["state_target_order"], "readouts": results,
        "current_mean_alpha_coverage": float(features["alpha_coverage"][development, -1].mean()),
        "current_no_alpha_support_fraction": float((features["alpha_coverage"][development, -1] <= 1e-12).mean()),
        "current_alpha_coverage_below_001_fraction": float((features["alpha_coverage"][development, -1] < .01).mean()),
        "interpretation": configuration["interpretation"]}
    write_json(output_root / model_name / "probe_report.json", report)
    return report


def summarize(configuration, output_root):
    reports = {name: compute_probe_report(configuration, output_root, name) for name in configuration["checkpoints"]}
    manifest = json.loads((output_root / "observed_object_manifest.json").read_text())
    objects = [row for row in manifest["objects"] if row["split"] == "development"]
    recording_names = np.asarray([row["recording_group"] for row in objects])
    predictions = {name: np.load(output_root / name / "development_predictions.npz") for name in reports}
    np.testing.assert_array_equal(predictions["published"]["object_indices"], predictions["posttrained"]["object_indices"])
    paired = {}
    for readout_name in configuration["readouts"]:
        before = predictions["published"][readout_name + "_state"]
        after = predictions["posttrained"][readout_name + "_state"]
        target = predictions["published"]["state_targets"]
        paired[readout_name] = {target_name: paired_recording_interval(np.abs(before[:, target_index] - target[:, target_index]),
            np.abs(after[:, target_index] - target[:, target_index]), recording_names,
            configuration["bootstrap_replicates"], configuration["bootstrap_seed"])
            for target_index, target_name in enumerate(manifest["state_target_order"])}
    summary = {"status": "completed_current_state_readout_diagnostic", "models": reports,
        "paired_state_error_posttrained_minus_published": paired,
        "classification_uncertainty": "Macro F1 point estimates only; no confidence claim is made for classification differences.",
        "remaining": ["human-reviewed instance masks", "causal future state readouts", "trained-planner interventions"],
        "stage2_started": False, "historical_gate_modified": False}
    write_json(output_root / "summary.json", summary)
    write_json(PROJECT_ROOT / configuration["shared_results_directory"] / "summary.json", summary)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs/lpwm_navsim_adaptation/object_readout_validation_v1.json")
    parser.add_argument("--mode", choices=["prepare", "extract", "summarize", "queue"], required=True)
    parser.add_argument("--model", choices=["published", "posttrained"])
    parser.add_argument("--gpu", type=int)
    parser.add_argument("--smoke", action="store_true")
    arguments = parser.parse_args()
    configuration, output_root = load_configuration(arguments.config)
    output_root.mkdir(parents=True, exist_ok=True)
    if arguments.mode == "prepare":
        prepare_annotations(configuration, output_root)
    elif arguments.mode == "extract":
        extract_representations(configuration, output_root, arguments.model, arguments.gpu, arguments.smoke)
    elif arguments.mode == "summarize":
        summarize(configuration, output_root)
    else:
        source_paths = [Path(__file__), PROJECT_ROOT / "src/planning_aware_future_prediction/object_centric/object_readout_diagnostics.py", arguments.config]
        hashes = {str(path.relative_to(PROJECT_ROOT)): digest(path) for path in source_paths}
        if (output_root / "queue_registration.json").exists():
            raise FileExistsError("Diagnostic queue already registered; inspect state before resuming")
        write_json(output_root / "queue_registration.json", {"source_hashes": hashes,
            "checkpoint_sha256": {name: row["sha256"] for name, row in configuration["checkpoints"].items()},
            "configuration": configuration, "pid": os.getpid()})
        children = []
        try:
            prepare_annotations(configuration, output_root)
            for gpu, name in zip(configuration["resources"]["allowed_gpus"], configuration["checkpoints"]):
                stream = (output_root / f"{name}_extraction.log").open("a")
                process = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), "--config", str(arguments.config.resolve()),
                    "--mode", "extract", "--model", name, "--gpu", str(gpu)], stdout=stream, stderr=subprocess.STDOUT)
                stream.close()
                children.append(process)
            write_json(output_root / "queue_state.json", {"status": "extracting_frozen_representations",
                "child_pids": [process.pid for process in children], "stage2_started": False})
            while any(process.poll() is None for process in children):
                if any(process.poll() not in (None, 0) for process in children):
                    raise RuntimeError("A frozen extraction worker failed")
                if any(digest(PROJECT_ROOT / path) != checksum for path, checksum in hashes.items()):
                    raise RuntimeError("Registered diagnostic source changed")
                time.sleep(5)
            if any(process.returncode != 0 for process in children):
                raise RuntimeError("Frozen extraction failed")
            write_json(output_root / "queue_state.json", {"status": "fitting_cpu_readouts"})
            summarize(configuration, output_root)
            write_json(output_root / "queue_state.json", {"status": "complete", "stage2_started": False})
        except BaseException as error:
            for process in children:
                if process.poll() is None:
                    process.terminate()
                    process.wait()
            write_json(output_root / "queue_failed.json", {"error": repr(error), "stage2_started": False})
            raise


if __name__ == "__main__":
    main()
