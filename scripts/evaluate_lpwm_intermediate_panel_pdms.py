"""Replay saved particle representations and score a fixed training panel with official PDMS."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import csv
import hashlib
import json
import multiprocessing
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
MONITOR = ROOT / "outputs/lpwm_drivor_planning_path_representation_monitor_v1"
CONTROLLER = ROOT / "outputs/lpwm_drivor_particle_trends_every500_v1"


def read_json(path):
    return json.loads(path.read_text())


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_suffix(".pending.json")
    pending.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    pending.replace(path)


def digest(path):
    checksum = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024**2), b""):
            checksum.update(chunk)
    return checksum.hexdigest()


def checkpoint_for_update(update):
    if update == 0:
        return None
    preserved = CONTROLLER / f"checkpoints/update_{update:06d}.pt"
    return preserved if preserved.exists() else MONITOR / "checkpoint_snapshot.pt"


def predict_panel(arguments):
    import torch
    import monitor_lpwm_drivor_planning_path_representations as diagnostic
    diagnostic.check_registration()
    output = arguments.output.resolve()
    assert not (output / "evaluation_complete.json").exists(), "Preserve the completed evaluation"
    panel = read_json(MONITOR / "panel.json")
    cache_rows = read_json(ROOT / "outputs/lpwm_candidate_teacher_v1/metric_cache_manifest.json")
    cache_by_token = {row["token"]: row["metric_cache_file"] for row in cache_rows}
    indices = [index for index, record in enumerate(panel["records"])
               if record["token"] in cache_by_token and Path(cache_by_token[record["token"]]).exists()]
    excluded = [record["token"] for index, record in enumerate(panel["records"]) if index not in indices]
    assert len(indices) == 95 and len(excluded) == 1
    checkpoints = {update: checkpoint_for_update(update) for update in arguments.updates}
    evidence = {}
    for update, checkpoint in checkpoints.items():
        complete = read_json(MONITOR / f"update_{update:06d}/complete.json")
        assert complete["complete"] and complete["completed_updates"] == update
        checksum = digest(checkpoint) if checkpoint else None
        assert checksum == complete["checkpoint_sha256"], update
        evidence[str(update)] = {"checkpoint": str(checkpoint) if checkpoint else "public initialization + seed2 planner",
                                 "checkpoint_sha256": checksum, "attributes_sha256": digest(MONITOR / f"update_{update:06d}/particle_attributes.npy")}
    write_json(output / "registration.json", {"source_sha256": digest(Path(__file__)), "panel_sha256": digest(MONITOR / "panel.json"),
        "updates": arguments.updates, "checkpoints": evidence, "scene_count": len(indices), "panel_count": 96,
        "excluded_missing_official_cache": excluded, "training_distribution": True, "full_navtest": False,
        "scoring": "Unmodified official NAVSIM v1 pdm_score with standard MetricCache, 40 x 0.1s simulator sampling",
        "cached_representation_replay": True, "model_updates": False, "started_unix": time.time()})
    manifest_records = [{**panel["records"][index], "metric_cache_file": cache_by_token[panel["records"][index]["token"]]}
                        for index in indices]
    write_json(output / "manifest.json", {"records": manifest_records, "excluded_tokens": excluded})
    assert diagnostic.card_bytes() < 42_000_000_000, "Training has priority; defer evaluation"
    torch.set_num_threads(2)
    torch.cuda.set_per_process_memory_fraction(4 * 1024**3 / torch.cuda.get_device_properties(0).total_memory)
    device = torch.device("cuda")
    configuration = read_json(ROOT / "configs/lpwm_drivor_planning_path_lora/navsim_v1.json")
    images = np.load(MONITOR / "images.npy", mmap_mode="r")
    predictions = []
    max_card_used = 0
    replay_checks = []
    for update in arguments.updates:
        torch.manual_seed(2)
        model = diagnostic.LPWMDrivoRPlanningPathLoRAModel(ROOT / configuration["public_checkpoint"]).to(device).eval()
        native_digest = model.frozen_native_digest()
        if checkpoints[update]:
            checkpoint = torch.load(checkpoints[update], map_location="cpu", weights_only=False)
            assert checkpoint["completed_updates"] == update
            model.load_state_dict(checkpoint["model"], strict=True)
            assert native_digest == checkpoint["frozen_native_sha256"] == model.frozen_native_digest()
            del checkpoint
        attributes = np.load(MONITOR / f"update_{update:06d}/particle_attributes.npy", mmap_mode="r")
        baseline = {row["token"]: row for row in read_json(MONITOR / f"update_{update:06d}/interventions.json") if row["variant"] == "baseline"}
        trajectories, expert_errors = [], []
        for completed, index in enumerate(indices, 1):
            used = diagnostic.card_bytes()
            max_card_used = max(max_card_used, used)
            assert used < 46_500_000_000, "Stop evaluation before the total 48GB card limit"
            record = panel["records"][index]
            ego = np.array(np.load(Path(record["cache_directory"]) / "ego.npy", mmap_mode="r")[record["cache_row"]])
            features = {"image": torch.from_numpy(np.array(images[index])).to(device).permute(0, 3, 1, 2)[None].float() / 255,
                        "ego_status": torch.from_numpy(ego).to(device)[None, None]}
            with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                memory = diagnostic.memory_from_attributes(model, np.array(attributes[index]))
                prediction = diagnostic.planning_with_memory(model, features, memory)
                trajectory = prediction["trajectory"].float().cpu().numpy()[0]
            assert trajectory.shape == (8, 3) and np.isfinite(trajectory).all()
            expert = np.array(np.load(Path(record["cache_directory"]) / "trajectory.npy", mmap_mode="r")[record["cache_row"]])
            error = float(np.linalg.norm(trajectory[:, :2] - expert[:, :2], axis=-1).mean())
            if record["token"] in baseline:
                difference = abs(error - baseline[record["token"]]["ade_m"])
                assert difference < 1e-6, (update, record["token"], difference)
                replay_checks.append({"update": update, "token": record["token"], "ade_difference": difference})
            trajectories.append(trajectory)
            expert_errors.append(error)
            write_json(output / "progress.json", {"stage": "trajectory_replay", "update": update,
                       "completed_scenes": completed, "scene_count": len(indices), "max_card_used_bytes": max_card_used})
            del features, memory, prediction
        destination = output / f"update_{update:06d}/predictions.npz"
        destination.parent.mkdir(parents=True, exist_ok=True)
        np.savez(destination, tokens=np.asarray([record["token"] for record in manifest_records]),
                 trajectories=np.asarray(trajectories), expert_ade_m=np.asarray(expert_errors))
        predictions.append(str(destination))
        del model, attributes
        torch.cuda.empty_cache()
    write_json(output / "replay_checks.json", {"checks": replay_checks, "maximum_ade_difference": max(row["ade_difference"] for row in replay_checks),
               "max_card_used_bytes": max_card_used, "original_weights_frozen_sha256": native_digest})
    diagnostic.check_registration()
    environment_root = ROOT / "runtime/environments/drive_jepa_official_evaluation"
    environment = dict(os.environ, CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1",
                       LD_LIBRARY_PATH=str(environment_root / "lib"), PYTHONPATH=str(ROOT / "reference_repositories/DrivoR"))
    subprocess.run([str(environment_root / "bin/python"), "-u", str(Path(__file__).resolve()), "--score-only",
                    "--output", str(output), "--workers", str(arguments.workers), "--updates", *map(str, arguments.updates)],
                   cwd=ROOT, env=environment, check=True)


def score_panel(arguments):
    from score_lpwm_drivor_navtest import score_scene
    output = arguments.output.resolve()
    records = read_json(output / "manifest.json")["records"]
    summaries = []
    with ProcessPoolExecutor(max_workers=arguments.workers, mp_context=multiprocessing.get_context("spawn")) as pool:
        for update in arguments.updates:
            predictions = np.load(output / f"update_{update:06d}/predictions.npz")
            assert predictions["tokens"].tolist() == [record["token"] for record in records]
            jobs = [(record["token"], trajectory, Path(record["metric_cache_file"])) for record, trajectory in zip(records, predictions["trajectories"])]
            rows = []
            for completed, row in enumerate(pool.map(score_scene, jobs, chunksize=1), 1):
                record = records[completed - 1]
                row.update(recording_group=record["recording_group"], scene_type=record["scene_type"], expert_ade_m=float(predictions["expert_ade_m"][completed - 1]))
                rows.append(row)
                write_json(output / "progress.json", {"stage": "official_pdms", "update": update, "completed_scenes": completed, "scene_count": len(records)})
            folder = output / f"update_{update:06d}"
            write_json(folder / "scores.json", rows)
            with (folder / "scores.csv").open("w", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
            numeric_keys = [key for key, value in rows[0].items() if isinstance(value, float)]
            summary = {"completed_updates": update, "scene_count": len(rows), "failed": 0,
                       "pdms": float(np.mean([row["score"] for row in rows]) * 100),
                       "zero_score_count": sum(row["score"] == 0 for row in rows),
                       "mean_metrics": {key: float(np.mean([row[key] for row in rows])) for key in numeric_keys},
                       "by_scene_type": {scenario: {"count": sum(row["scene_type"] == scenario for row in rows),
                          "pdms": float(np.mean([row["score"] for row in rows if row["scene_type"] == scenario]) * 100)}
                          for scenario in sorted(set(row["scene_type"] for row in rows))}}
            write_json(folder / "summary.json", summary)
            summaries.append(summary)
    result = {"complete": True, "scope": "Official NAVSIM v1 PDMS on a fixed 95-scene subset of the planning training distribution; not navtest",
              "scene_count": len(records), "recording_count": len(set(record["recording_group"] for record in records)),
              "excluded_missing_cache_tokens": read_json(output / "manifest.json")["excluded_tokens"],
              "rows": summaries, "finished_unix": time.time()}
    write_json(output / "evaluation_complete.json", result)
    print(json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/lpwm_drivor_intermediate_pdms_v1")
    parser.add_argument("--updates", type=int, nargs="+", default=[0, 500, 1000])
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--score-only", action="store_true")
    arguments = parser.parse_args()
    if arguments.score_only:
        score_panel(arguments)
    else:
        predict_panel(arguments)
