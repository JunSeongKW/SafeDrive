"""Official PDM scoring on the preregistered DEVELOPMENT windows only.

Runs on CPU. Preparing state-based metric caches never feeds labels to the model.
Can wait for training completion, then score every registered final checkpoint.
"""

import argparse
import hashlib
import json
import lzma
import os
import pickle
import subprocess
import sys
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np

from audit_official_drive_jepa_evaluation import official_configuration

WORKSPACE = Path(__file__).resolve().parents[1]


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(path.suffix + ".partial")
    partial.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    partial.replace(path)


def file_sha256(path):
    with Path(path).open("rb") as stream:
        digest = hashlib.sha256()
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def prepare_cache(records, output):
    from navsim.common.dataclasses import Scene, SensorConfig
    from navsim.planning.metric_caching.metric_cache_processor import MetricCacheProcessor
    from navsim.planning.scenario_builder.navsim_scenario import NavSimScenario

    processor = MetricCacheProcessor(str(output / "metric_cache"), False)
    previous_segment, log_frames = None, None
    manifest = []
    for index, record in enumerate(records):
        if previous_segment != record["segment_filename"]:
            log_path = WORKSPACE / "dataset/navsim_logs/trainval" / record["segment_filename"]
            with log_path.open("rb") as stream:
                log_frames = pickle.load(stream)
            previous_segment = record["segment_filename"]
            log_hash = file_sha256(log_path)
        # Ten future 0.5-second states are needed by the official metric processor.
        frames = log_frames[record["start_index"]:record["start_index"] + 14]
        if len(frames) != 14 or frames[3]["token"] != record["current_frame_token"]:
            raise RuntimeError("PDM state history/current/future alignment mismatch")
        scene = Scene.from_scene_dict_list(
            frames, WORKSPACE / "dataset/sensor_blobs/trainval", 4, 10,
            SensorConfig.build_no_sensors())
        scenario = NavSimScenario(scene, str(WORKSPACE / "dataset/maps"), "nuplan-maps-v1.0")
        if scenario.token != record["current_frame_token"]:
            raise RuntimeError("PDM scenario token mismatch")
        candidate_path = output / "metric_cache" / scenario.log_name / scenario.scenario_type / scenario.token / "metric_cache.pkl"
        if candidate_path.exists() and candidate_path.stat().st_size == 0:
            # Preserve interrupted sandbox writes; never accept an empty cache.
            candidate_path.rename(candidate_path.with_name("metric_cache.interrupted_empty.pkl"))
        metadata = processor.compute_metric_cache(scenario)
        cache_path = Path(metadata.file_name)
        manifest.append({"token": scenario.token, "recording": record["recording_group"],
                         "metric_cache_file": str(cache_path), "cache_sha256": file_sha256(cache_path),
                         "source_log_file": str(log_path), "source_log_sha256": log_hash,
                         "current_timestamp_us": int(frames[3]["timestamp"]),
                         "last_state_timestamp_us": int(frames[-1]["timestamp"])})
        if (index + 1) % 8 == 0:
            write_json(output / "cache_progress.json", {"completed": index + 1, "total": len(records)})
            print(f"PDM_DEVELOPMENT_CACHED {index + 1}/{len(records)}", flush=True)
    write_json(output / "metric_cache_manifest.json", manifest)
    return manifest


def score_runs(run_directory, specification, manifest, configuration, output):
    from hydra.utils import instantiate
    from navsim.common.dataclasses import Trajectory
    from navsim.evaluate.pdm_score import pdm_score

    simulator = instantiate(configuration.simulator)
    scorer = instantiate(configuration.scorer)
    if simulator.proposal_sampling != scorer.proposal_sampling:
        raise RuntimeError("Official scorer sampling differs from simulator")
    predictions = {}
    baseline_rows = None
    for condition in specification["conditions"]:
        for seed in specification["seeds"]:
            result_path = run_directory / f"{condition}_seed{seed}/results.json"
            report = json.loads(result_path.read_text())
            initial_rows = report["evaluations"]["0"]["development"]["windows"]
            initial = {row["token"]: row["trajectory"] for row in initial_rows}
            if baseline_rows is None:
                baseline_rows = initial_rows
            elif initial != {row["token"]: row["trajectory"] for row in baseline_rows}:
                raise RuntimeError("Initial baseline predictions differ across registered runs")
            predictions[f"{condition}_seed{seed}"] = {
                row["token"]: row for row in report["evaluations"][str(specification["joint_updates"])]["development"]["windows"]}
    predictions["original_frozen"] = {row["token"]: row for row in baseline_rows}
    expected_tokens = {row["token"] for row in manifest}
    if any(set(rows) != expected_tokens for rows in predictions.values()):
        raise RuntimeError("PDM development token sets differ")
    scores = {name: [] for name in predictions}
    for index, entry in enumerate(manifest):
        token = entry["token"]
        if file_sha256(entry["metric_cache_file"]) != entry["cache_sha256"]:
            raise RuntimeError("PDM cache hash differs")
        with lzma.open(entry["metric_cache_file"], "rb") as stream:
            metric_cache = pickle.load(stream)
        per_scene_path = output / "scene_scores" / f"{token}.json"
        if per_scene_path.exists():
            per_scene = json.loads(per_scene_path.read_text())
            if set(per_scene) != set(predictions):
                raise RuntimeError("Resume PDM conditions differ")
        else:
            per_scene = {}
            for name, rows in predictions.items():
                predicted_trajectory = Trajectory(np.asarray(rows[token]["trajectory"], dtype=np.float32))
                metrics = asdict(pdm_score(metric_cache, predicted_trajectory,
                                          simulator.proposal_sampling, simulator, scorer))
                if not all(np.isfinite(value) for value in metrics.values()):
                    raise RuntimeError("Non-finite official PDM result")
                per_scene[name] = {"token": token, "recording": entry["recording"], **metrics}
            write_json(per_scene_path, per_scene)
        for name, row in per_scene.items():
            scores[name].append(row)
        if (index + 1) % 8 == 0:
            write_json(output / "score_progress.json", {"completed_scenes": index + 1, "total_scenes": len(manifest),
                                                        "predictions_per_scene": len(predictions)})
            print(f"PDM_DEVELOPMENT_SCORED {index + 1}/{len(manifest)} x {len(predictions)}", flush=True)
    summary = {name: {key: float(np.mean([row[key] for row in rows]))
                       for key in rows[0] if key not in ("token", "recording")}
               for name, rows in scores.items()}
    write_json(output / "results.json", {"scope": "192 preregistered development windows, not official navtest",
                                        "summary": summary, "windows": scores})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-directory", required=True, type=Path)
    parser.add_argument("--output-directory", required=True, type=Path)
    parser.add_argument("--wait-for-training", action="store_true")
    parser.add_argument("--detach", action="store_true")
    args = parser.parse_args()
    args.run_directory = args.run_directory.resolve()
    args.output_directory = args.output_directory.resolve()
    if args.detach:
        command = ["nohup", sys.executable, "-u", str(Path(__file__).resolve()),
                   "--run-directory", str(args.run_directory),
                   "--output-directory", str(args.output_directory)]
        if args.wait_for_training:
            command.append("--wait-for-training")
        with args.output_directory.with_suffix(".log").open("a") as stream:
            process = subprocess.Popen(command, cwd=WORKSPACE, stdout=stream,
                stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, start_new_session=True)
        print(json.dumps({"pid": process.pid, "output": str(args.output_directory)}), flush=True)
        return
    args.output_directory.mkdir(parents=True, exist_ok=True)
    specification = json.loads((WORKSPACE / "configs/drive_jepa_selective_future/region_research_v1.json").read_text())
    records = [record for record in json.loads((WORKSPACE / specification["reused_cache"]).read_text())["records"]
               if record["split"] == "development"]
    if len(records) != 192 or len({record["recording_group"] for record in records}) != 24:
        raise RuntimeError("Unexpected development split")
    _, _, configuration, official_root = official_configuration(WORKSPACE)
    write_json(args.output_directory / "provenance.json", {"scorer_source": str(official_root),
               "evaluation_scope": "development only", "gpu_used": False,
               "config_sha256": file_sha256(WORKSPACE / "configs/drive_jepa_selective_future/region_research_v1.json")})
    started = time.perf_counter()
    manifest_path = args.output_directory / "metric_cache_manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else prepare_cache(records, args.output_directory)
    if args.wait_for_training:
        while not (args.run_directory / "completion.json").is_file():
            if time.perf_counter() - started > specification["maximum_wall_seconds"] + 600:
                raise RuntimeError("Training completion wait cap reached; cache preserved")
            time.sleep(30)
    score_runs(args.run_directory, specification, manifest, configuration, args.output_directory)
    write_json(args.output_directory / "completion.json", {"complete": True, "wall_seconds": time.perf_counter() - started})
    print("DEVELOPMENT_PDM_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
