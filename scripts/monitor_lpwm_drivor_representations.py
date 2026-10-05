"""Read-only checkpoint monitor; training code, gradients and queues are untouched."""
import argparse
from collections import defaultdict
from contextlib import nullcontext
import ctypes
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from unittest.mock import patch

import numpy as np
import torch
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"src"))
from planning_aware_future_prediction.object_centric.lpwm_drivor_lora import LPWMDrivoRLoRAModel
from planning_aware_future_prediction.object_centric.particle_planning_diagnostics import (
    matched_control_particles, object_particle_weights, particle_geometry, region_distribution)
from planning_aware_future_prediction.object_centric.object_readout_diagnostics import (
    classification_metrics, fit_ridge_readout, predict_ridge_readout, paired_recording_interval)
from lpwm_drivor_oracle import DrivoROracleClient

OUTPUT = ROOT/"outputs/lpwm_drivor_representation_monitor_v1"
TRAINING = ROOT/"outputs/lpwm_drivor_lora_v1/navsim_v1"
CATEGORY_COLORS = {"vehicle": "cyan", "pedestrian": "magenta", "bicycle": "orange"}


def write_json(path, content):
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_suffix(".pending.json")
    pending.write_text(json.dumps(content, indent=2, allow_nan=False)+"\n")
    pending.replace(path)


def digest(path):
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8*1024**2), b""):
            result.update(chunk)
    return result.hexdigest()


def check_registration():
    registration = json.loads((OUTPUT/"registration.json").read_text())
    for name, expected in registration["sources"].items():
        assert digest(ROOT/name) == expected, name
    assert digest(OUTPUT/"panel.json") == registration["panel_sha256"]


def card_bytes():
    gpu = os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")[0]
    return int(subprocess.check_output(["nvidia-smi", "--id="+gpu, "--query-gpu=memory.used",
                                       "--format=csv,noheader,nounits"], text=True).strip()) * 1024**2


def memory_from_attributes(model, attributes):
    encoder = model.particle_encoder
    values = torch.as_tensor(np.array(attributes), device=next(model.parameters()).device, dtype=torch.float32)
    projected = encoder.particle_trajectory_projection(values.permute(0, 2, 1, 3).flatten(2))
    return (projected.reshape(4, 16, 4, 256).mean(2) + encoder.camera_embedding).reshape(1, 64, 256)


def planning_with_memory(model, features, memory):
    with patch.object(model.particle_encoder, "forward", return_value=memory):
        return model.planner(features)


def fixed_candidate_scores(model, features, memory, candidates):
    planner = model.planner
    embedding = planner.pos_embed(candidates.flatten(2).detach())
    context = planner.scorer_attention(embedding, memory) + planner.hist_encoding(features["ego_status"][:, -1])[:, None]
    logits = planner.scorer(candidates, context)[0]
    config = planner._config
    return (config.noc*logits["no_at_fault_collisions"].sigmoid().log()
            + config.dac*logits["drivable_area_compliance"].sigmoid().log()
            + config.ddc*logits["driving_direction_compliance"].sigmoid().log()
            + (config.ttc*logits["time_to_collision_within_bound"].sigmoid()
               + config.ep*logits["ego_progress"].sigmoid()
               + config.comfort*logits["comfort"].sigmoid()).log())


def draw_scene(rgb, attributes, objects, masks, title, destination):
    canvas = Image.new("RGB", (1024, 300), "white")
    for camera in range(4):
        image = Image.fromarray(rgb[camera]).resize((256, 256))
        overlay = np.zeros((128, 128, 4), dtype=np.uint8)
        overlay[masks[camera, 1]] = [0, 230, 0, 35]
        image = Image.alpha_composite(image.convert("RGBA"), Image.fromarray(overlay).resize((256, 256))).convert("RGB")
        draw = ImageDraw.Draw(image)
        centers, sizes, _, presence = particle_geometry(attributes[camera, 0])
        for particle, (center, size) in enumerate(zip(centers, sizes)):
            horizontal, vertical = center*2
            radius = 1+2*presence[particle]
            draw.ellipse((horizontal-radius, vertical-radius, horizontal+radius, vertical+radius), fill="yellow")
        for obj in objects:
            if obj["camera_index"] == camera:
                draw.rectangle(tuple(np.array(obj["box"])*2), outline=CATEGORY_COLORS[obj["category"]], width=1)
        canvas.paste(image, (camera*256, 0))
    draw = ImageDraw.Draw(canvas)
    draw.text((5, 261), title, fill="black")
    draw.text((5, 277), "All 64 particles/camera; size=presence. Boxes=GT projections. Green=planar-map road proxy.", fill="black")
    canvas.save(destination)


def readouts(attributes, panel):
    objects = panel["objects"]
    rows = []
    for scene_index, record in enumerate(panel["records"]):
        for camera in range(4):
            selected = [index for index in range(record["object_start"], record["object_end"]) if objects[index]["camera_index"] == camera]
            if not selected:
                continue
            weights, mass = object_particle_weights(attributes[scene_index, camera, 0], [objects[index]["box"] for index in selected])
            pooled = np.einsum("op,tpf->otf", weights, attributes[scene_index, camera])
            for local_index, index in enumerate(selected):
                rows.append((objects[index], pooled[local_index], mass[local_index]))
    categories = np.array([panel["categories"].index(row[0]["category"]) for row in rows])
    fit = np.array([row[0]["probe_partition"] == "readout_fit" for row in rows])
    pooled = np.stack([row[1] for row in rows])
    boxes = np.array([row[0]["box"] for row in rows])/128
    feature_sets = {"gt_roi_geometry_control": boxes, "particle_current_appearance": pooled[:, 0, 6:10],
                    "particle_current_all": pooled[:, 0], "predicted_2s_all": pooled[:, 4], "predicted_4s_all": pooled[:, 8]}
    result = {"object_view_samples": len(rows), "fit_samples": int(fit.sum()), "evaluation_samples": int((~fit).sum()),
        "empty_support_fraction": float(np.mean([row[2] <= 1e-12 for row in rows])),
        "scope": "Recording-disjoint readout fit/evaluation, both from upstream planning TRAIN distribution. GT association is privileged. No LPWM updates.",
        "classification": {}, "future_displacement": {}}
    for name in ("gt_roi_geometry_control", "particle_current_appearance", "particle_current_all"):
        features = feature_sets[name]
        class_counts = np.bincount(categories[fit], minlength=3)
        sample_weights = 1/np.maximum(class_counts[categories[fit]], 1)
        readout = fit_ridge_readout(features[fit], np.eye(3)[categories[fit]], .1, sample_weights)
        predicted = predict_ridge_readout(readout, features[~fit]).argmax(-1)
        result["classification"][name] = classification_metrics(categories[~fit], predicted, 3)
    for horizon_index, seconds in enumerate((2, 4)):
        targets = np.array([row[0]["future_displacement"][horizon_index] for row in rows])
        valid = np.array([row[0]["future_valid"][horizon_index] for row in rows]) & np.isfinite(targets).all(-1)
        evaluation = ~fit & valid
        horizon = {"valid_evaluation_samples": int(evaluation.sum()), "methods": {}}
        if not evaluation.any() or (fit & valid).sum() < 2:
            result["future_displacement"][str(seconds)] = horizon
            continue
        velocity = np.array([row[0]["state"][2:4] for row in rows])
        predictions = {"zero_displacement": np.zeros_like(targets), "constant_velocity_gt_state_control": velocity*seconds}
        for name in ("gt_roi_geometry_control", "particle_current_appearance", "particle_current_all", f"predicted_{seconds}s_all"):
            features = feature_sets[name]
            readout = fit_ridge_readout(features[fit & valid], targets[fit & valid], .1)
            predictions[name] = predict_ridge_readout(readout, features)
        for name, predicted in predictions.items():
            errors = np.linalg.norm(predicted[evaluation]-targets[evaluation], axis=-1)
            finite = np.isfinite(errors)
            horizon["methods"][name] = {"mean_l2_m": float(errors[finite].mean()) if finite.any() else None, "valid_count": int(finite.sum())}
        current_errors = np.linalg.norm(predictions["particle_current_all"][evaluation]-targets[evaluation], axis=-1)
        future_errors = np.linalg.norm(predictions[f"predicted_{seconds}s_all"][evaluation]-targets[evaluation], axis=-1)
        horizon["predicted_minus_current_error"] = paired_recording_interval(current_errors, future_errors,
            [row[0]["recording_group"] for index, row in enumerate(rows) if evaluation[index]], 300, 71)
        result["future_displacement"][str(seconds)] = horizon
    return result


def summarize_diagnostics(scene_rows, interventions, intents):
    result = {"distribution": {}, "paired_interventions": {}, "intent_sensitivity": {},
              "interpretation": "Location is a proxy; readouts measure accessible information; interventions measure model reliance and may be out of distribution. No automatic success gate."}
    for scenario in ("all", "straight", "left_turn", "right_turn", "projected_overlap"):
        selected = [row for row in scene_rows if scenario == "all" or row["scene_type"] == scenario or (scenario == "projected_overlap" and row["overlap_proxy"])]
        if not selected:
            continue
        table = {"scene_count": len(selected)}
        for region in ("vehicle", "pedestrian", "bicycle", "road_proxy_low", "road_proxy", "road_proxy_high"):
            values = [camera[region] for row in selected for camera in row["cameras"] if not region.startswith("road") or row["road_proxy_available"]]
            table[region] = {"center_fraction": float(np.mean([value["center_fraction"] for value in values])),
                "presence_weighted_fraction": float(np.mean([value["presence_weighted_fraction"] for value in values])),
                "enrichment_ratio_of_sums": sum(value["presence_weighted_fraction"] for value in values)/max(sum(value["image_area_fraction"] for value in values), 1e-12)} if values else None
        table["upper_third_fraction_not_sky_label"] = float(np.mean([camera["upper_third_fraction"] for row in selected for camera in row["cameras"]]))
        result["distribution"][scenario] = table
    baseline = {row["token"]: row for row in interventions if row["variant"] == "baseline"}
    for variant in sorted({row["variant"] for row in interventions} - {"baseline"}):
        changed = [row for row in interventions if row["variant"] == variant]
        paired = [baseline[row["token"]] for row in changed]
        table = {}
        for metric in ("selected_trajectory_oracle_score", "fixed_candidate_oracle_score", "ade_m"):
            table[metric+"_change"] = paired_recording_interval([row[metric] for row in paired], [row[metric] for row in changed],
                [row["recording_group"] for row in changed], 300, 71)
        table["mean_trajectory_change_m"] = float(np.mean([row["trajectory_change_m"] for row in changed]))
        result["paired_interventions"][variant] = table
    changed_intents = [row for row in intents if row["actual_command"][row["command_index"]] != 1]
    for metric in ("current_center_shift_mean_px", "current_foreground_feature_rms_change", "future_attribute_rms_change",
                   "encoder_only_trajectory_change_m", "planner_only_trajectory_change_m", "both_trajectory_change_m"):
        result["intent_sensitivity"][metric] = float(np.mean([row[metric] for row in changed_intents])) if changed_intents else None
    return result


def evaluate_checkpoint(arguments):
    check_registration()
    torch.set_num_threads(2)
    torch.manual_seed(2)
    device = torch.device(arguments.device)
    if device.type == "cuda":
        assert card_bytes() < 40_000_000_000, "Defer diagnostic: training has priority"
        torch.cuda.set_per_process_memory_fraction(4*1024**3/torch.cuda.get_device_properties(0).total_memory)
    configuration = json.loads((ROOT/"configs/lpwm_drivor_lora/navsim_v1.json").read_text())
    model = LPWMDrivoRLoRAModel(ROOT/configuration["public_checkpoint"]).to(device).eval()
    native_digest = model.frozen_native_digest()
    updates, checkpoint_hash = 0, None
    if arguments.checkpoint:
        state = torch.load(arguments.checkpoint, map_location="cpu", weights_only=False)
        model.load_state_dict(state["model"], strict=True)
        updates = state["completed_updates"]
        checkpoint_hash = digest(arguments.checkpoint)
        assert native_digest == model.frozen_native_digest() == state["frozen_native_sha256"]
        del state
    destination = (OUTPUT/"engineering" if arguments.scene_limit else OUTPUT)/f"update_{updates:06d}"
    destination.mkdir(parents=True, exist_ok=True)
    if (destination/"complete.json").exists():
        return
    shutil.copyfile(OUTPUT/"registration.json", destination/"diagnostic_registration.json")
    panel = json.loads((OUTPUT/"panel.json").read_text())
    if arguments.scene_limit:
        panel["records"] = panel["records"][:arguments.scene_limit]
    images = np.load(OUTPUT/"images.npy", mmap_mode="r")
    road = np.load(OUTPUT/"road_proxy_masks.npy", mmap_mode="r")
    model.particle_encoder.record_particles = True
    collected, scene_rows, intervention_rows, intent_rows = [], [], [], []
    oracle = DrivoROracleClient(OUTPUT/"oracle_manifest.json", destination, workers=2)
    started = time.time()
    try:
        for scene_index, record in enumerate(panel["records"]):
            if device.type == "cuda":
                assert card_bytes() < 46_500_000_000, "Stop diagnostics before the 48GB total card limit"
            rgb = np.array(images[scene_index])
            ego = np.array(np.load(Path(record["cache_directory"])/"ego.npy", mmap_mode="r")[record["cache_row"]])
            features = {"image": torch.from_numpy(rgb).to(device).permute(0, 3, 1, 2)[None].float()/255,
                        "ego_status": torch.from_numpy(ego).to(device)[None, None]}
            objects = panel["objects"][record["object_start"]:record["object_end"]]
            precision = torch.autocast("cuda", dtype=torch.bfloat16) if device.type == "cuda" else nullcontext()
            with torch.inference_mode(), precision:
                baseline = model(features)
                attributes = model.particle_encoder.latest_particle_attributes[0].numpy().copy()
                collected.append(attributes)
                memory = memory_from_attributes(model, attributes)
                replay = planning_with_memory(model, features, memory)
                assert torch.equal(replay["proposals"], baseline["proposals"]), "Representation replay must preserve the planner"
                assert torch.equal(fixed_candidate_scores(model, features, memory, baseline["proposals"]), baseline["pdm_score"])
                distributions = []
                importance = np.zeros((4, 64))
                for camera in range(4):
                    masks = {"road_proxy_low": road[scene_index, camera, 0], "road_proxy": road[scene_index, camera, 1], "road_proxy_high": road[scene_index, camera, 2]}
                    for category in panel["categories"]:
                        mask = np.zeros((128, 128), bool)
                        for obj in objects:
                            if obj["camera_index"] == camera and obj["category"] == category:
                                rectangle = np.array(obj["box"])
                                lower, upper = np.floor(rectangle[:2]).astype(int), np.ceil(rectangle[2:]).astype(int)
                                mask[lower[1]:upper[1], lower[0]:upper[0]] = True
                        masks[category] = mask
                    distributions.append(region_distribution(attributes[camera, 0], masks))
                    relevant = [obj for obj in objects if obj["camera_index"] == camera and obj["near_expert_corridor_proxy"]]
                    if relevant:
                        weights, _ = object_particle_weights(attributes[camera, 0], [obj["box"] for obj in relevant])
                        importance[camera] = weights.sum(0)
                scene_rows.append({"token": record["token"], "recording_group": record["recording_group"], "scene_type": record["scene_type"],
                    "command": record["command"], "overlap_proxy": record["projected_overlap_proxy"], "road_proxy_available": record["road_proxy_available"], "cameras": distributions})
                # Keep a fixed small intervention/intent panel; all 96 scenes enter readouts.
                if scene_index % 8 == 0:
                    draw_scene(rgb, attributes, objects, road[scene_index], f"update {updates} | {record['scene_type']} | {record['token']}", destination/f"scene_{scene_index:03d}.png")
                    variants = {"baseline": attributes}
                    repeated = attributes.copy(); repeated[:, 1:] = attributes[:, :1]
                    variants["future_repeat_current"] = repeated
                    reversed_future = attributes.copy(); reversed_future[:, 1:] = attributes[:, 1:][:, ::-1]
                    variants["future_reverse_time"] = reversed_future
                    selected = np.argsort(-importance.ravel(), kind="stable")[:8]
                    selected = selected[importance.ravel()[selected] > 0]
                    if len(selected):
                        controls = matched_control_particles(attributes, selected)
                        for name, indices in (("relevant_particle_mean", selected), ("matched_control_mean", controls)):
                            changed = attributes.copy()
                            for index in indices:
                                camera, particle = divmod(int(index), 64)
                                changed[camera, :, particle] = attributes[camera].mean(1)
                            variants[name] = changed
                    candidates = baseline["proposals"].float().cpu().numpy()[0]
                    trajectories, fixed_indices, names = [], [], []
                    for name, changed in variants.items():
                        changed_memory = memory_from_attributes(model, changed)
                        prediction = planning_with_memory(model, features, changed_memory)
                        trajectories.append(prediction["trajectory"].float().cpu().numpy()[0])
                        fixed_indices.append(int(fixed_candidate_scores(model, features, changed_memory, baseline["proposals"]).argmax(-1).item()))
                        names.append(name)
                    scores = oracle.score([record["token"]], np.concatenate((candidates, trajectories), 0)[None])[0]
                    expert = np.array(np.load(Path(record["cache_directory"])/"trajectory.npy", mmap_mode="r")[record["cache_row"]])
                    for index, name in enumerate(names):
                        intervention_rows.append({"token": record["token"], "recording_group": record["recording_group"], "variant": name,
                            "selected_trajectory_oracle_score": float(scores[64+index, -1]),
                            "fixed_candidate_oracle_score": float(scores[fixed_indices[index], -1]),
                            "fixed_candidate_oracle_regret": float(scores[:64, -1].max()-scores[fixed_indices[index], -1]),
                            "ade_m": float(np.linalg.norm(trajectories[index][:, :2]-expert[:, :2], axis=-1).mean()),
                            "trajectory_change_m": float(np.linalg.norm(trajectories[index][:, :2]-trajectories[0][:, :2], axis=-1).mean()),
                            "affected_particles": len(selected) if "particle" in name or "control" in name else None})
                    for command_index in range(3):
                        alternative = {name: value.clone() for name, value in features.items()}
                        alternative["ego_status"][:, :, 7:11] = 0
                        alternative["ego_status"][:, :, 7+command_index] = 1
                        both = model(alternative)
                        alternative_attributes = model.particle_encoder.latest_particle_attributes[0].numpy().copy()
                        alternative_memory = memory_from_attributes(model, alternative_attributes)
                        encoder_only = planning_with_memory(model, features, alternative_memory)
                        planner_only = planning_with_memory(model, alternative, memory)
                        shifts = np.linalg.norm((alternative_attributes[:, 0, :, :2]-attributes[:, 0, :, :2])*64, axis=-1)
                        row = {"token": record["token"], "command_index": command_index, "actual_command": record["command"],
                            "current_center_shift_mean_px": float(shifts.mean()), "current_center_shift_max_px": float(shifts.max()),
                            "current_foreground_feature_rms_change": float(np.sqrt(np.square(alternative_attributes[:, 0, :, 6:10]-attributes[:, 0, :, 6:10]).mean())),
                            "future_attribute_rms_change": float(np.sqrt(np.square(alternative_attributes[:, 1:]-attributes[:, 1:]).mean()))}
                        for name, prediction in (("encoder_only", encoder_only), ("planner_only", planner_only), ("both", both)):
                            row[name+"_trajectory_change_m"] = float(torch.linalg.vector_norm(prediction["trajectory"][..., :2]-baseline["trajectory"][..., :2], dim=-1).mean())
                        intent_rows.append(row)
                        draw_scene(rgb, alternative_attributes, objects, road[scene_index], f"update {updates} | command index {command_index} (sensitivity, not a valid GT alternative)", destination/f"scene_{scene_index:03d}_command_{command_index}.png")
            write_json(destination/"progress.json", {"completed_scenes": scene_index+1, "total_scenes": len(panel["records"]), "seconds": time.time()-started})
            if scene_index % 16 == 0:
                print("MONITOR", updates, scene_index+1, len(panel["records"]), flush=True)
    finally:
        oracle.close()
    attributes = np.asarray(collected)
    np.save(destination/"particle_attributes.npy", attributes)
    write_json(destination/"distribution.json", scene_rows)
    write_json(destination/"interventions.json", intervention_rows)
    write_json(destination/"intent_sensitivity.json", intent_rows)
    write_json(destination/"summary.json", summarize_diagnostics(scene_rows, intervention_rows, intent_rows))
    if not arguments.scene_limit:
        write_json(destination/"readouts.json", readouts(attributes, panel))
    initial = OUTPUT/"update_000000/particle_attributes.npy"
    change = None
    if updates and initial.exists():
        before = np.load(initial)
        if before.shape == attributes.shape:
            center_shift = np.linalg.norm((attributes[:, :, 0, :, :2]-before[:, :, 0, :, :2])*64, axis=-1)
            change = {"current_center_mean_shift_px": float(center_shift.mean()), "current_center_max_shift_px": float(center_shift.max()),
                      "foreground_feature_rms_change": float(np.sqrt(np.square(attributes[..., 6:10]-before[..., 6:10]).mean()))}
    write_json(destination/"complete.json", {"complete": True, "completed_updates": updates, "checkpoint_sha256": checkpoint_hash,
        "scene_count": len(panel["records"]), "change_from_initial": change, "seconds": time.time()-started,
        "not_full_navtest": True, "not_independent_planning_validation": True,
        "counterfactual_commands_have_no_alternative_gt": True, "interventions_can_cause_distribution_shift": True,
        "original_weights_frozen_sha256": native_digest, "max_diagnostic_reserved_bytes": torch.cuda.max_memory_reserved() if device.type == "cuda" else 0})
    render_index()


def render_index():
    runs = sorted(OUTPUT.glob("update_*/complete.json"))
    body = ["<html><meta charset='utf-8'><title>LPWM planning representation monitor</title><body>",
        "<h1>LPWM 학습 중 표현 진단</h1><p>같은 학습 장면을 반복 확인합니다. 독립 검증이나 full navtest 성능이 아닙니다. "
        "GT 박스는 평가용 투영이며 녹색 영역은 평면 가정 지도 proxy입니다. 점의 위치만으로 의미적 이해를 판단하지 않습니다.</p>"]
    for result in runs:
        folder = result.parent
        metadata = json.loads(result.read_text())
        body.append(f"<h2>Update {metadata['completed_updates']}</h2><a href='{folder.name}/summary.json'>요약</a> · <a href='{folder.name}/complete.json'>실행 결과</a> · "
                    f"<a href='{folder.name}/distribution.json'>분포</a> · <a href='{folder.name}/readouts.json'>readout</a> · "
                    f"<a href='{folder.name}/interventions.json'>planning 개입</a> · <a href='{folder.name}/intent_sensitivity.json'>의도 경로 분해</a>")
        for picture in sorted(folder.glob("scene_*.png")):
            body.append(f"<div><a href='{folder.name}/{picture.name}'>{picture.name}</a><br><img width='1024' src='{folder.name}/{picture.name}'></div>")
    (OUTPUT/"index.html").write_text("\n".join(body)+"</body></html>")


def watch():
    ctypes.CDLL(None).prctl(15, b"kjs-lpwm-watch", 0, 0, 0)
    import fcntl
    lock = (OUTPUT/"watch.lock").open("w")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    configuration = json.loads((OUTPUT/"registration.json").read_text())
    assert configuration["sealed_for_watch"], "Seal the read-only diagnostic protocol before watching"
    milestones = configuration["milestones"]
    while True:
        check_registration()
        if (OUTPUT/"pause.requested").exists():
            write_json(OUTPUT/"watch_status.json", {"stage": "paused"}); return
        if any(path.exists() for path in (TRAINING.parent/"pause.requested", TRAINING/"pause.requested", TRAINING/"paused.json")):
            write_json(OUTPUT/"watch_status.json", {"stage": "paused_with_training"}); return
        progress = json.loads((TRAINING/"progress.json").read_text())
        completed = progress["completed_updates"]
        existing = [json.loads(path.read_text())["completed_updates"] for path in OUTPUT.glob("update_*/complete.json")]
        latest_evaluated = max(existing, default=-1)
        pending = [milestone for milestone in milestones if latest_evaluated < milestone <= completed]
        needs_initial = 0 not in existing
        if needs_initial or pending:
            if card_bytes() >= 40_000_000_000:
                write_json(OUTPUT/"watch_status.json", {"stage": "waiting_for_memory", "completed_training_updates": completed})
                time.sleep(30); continue
            command = [sys.executable, "-u", str(Path(__file__).resolve()), "--device", "cuda"]
            if not needs_initial:
                snapshot = OUTPUT/"checkpoint_snapshot.pt"
                shutil.copyfile(TRAINING/"latest.pt", snapshot)
                # Read metadata on CPU; latest.pt is atomically replaced by training.
                checkpoint = torch.load(snapshot, map_location="cpu", weights_only=False)
                update = checkpoint["completed_updates"]
                del checkpoint
                if update <= latest_evaluated or update < min(pending):
                    time.sleep(30); continue
                command += ["--checkpoint", str(snapshot)]
            write_json(OUTPUT/"watch_status.json", {"stage": "evaluating", "training_updates_at_start": completed, "initial": needs_initial})
            with (OUTPUT/"monitor.log").open("a") as stream:
                result = subprocess.run(command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT)
            if result.returncode:
                write_json(OUTPUT/"watch_status.json", {"stage": "diagnostic_failed_training_unaffected", "returncode": result.returncode})
                return
        elif (TRAINING/"training_complete.json").exists() and latest_evaluated >= completed:
            write_json(OUTPUT/"watch_status.json", {"stage": "complete"}); return
        else:
            write_json(OUTPUT/"watch_status.json", {"stage": "waiting_for_checkpoint", "training_updates": completed,
                "last_evaluated_update": latest_evaluated, "next_milestone": next((value for value in milestones if value > latest_evaluated), None)})
        time.sleep(30)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--device", default="cuda", choices=["cpu", "cuda"])
    parser.add_argument("--scene-limit", type=int, default=0)
    parser.add_argument("--watch", action="store_true")
    arguments = parser.parse_args()
    if arguments.watch:
        watch()
    else:
        evaluate_checkpoint(arguments)
