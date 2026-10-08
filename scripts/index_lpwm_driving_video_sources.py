"""Index actual front-camera sequences; preserve official planning split identities."""
import argparse
from concurrent.futures import ProcessPoolExecutor
from collections import Counter
import hashlib
import json
from pathlib import Path
import pickle
import time

ROOT = Path(__file__).resolve().parents[1]


def recording_identity(log_name):
    return "_".join(log_name.split("_")[:-2])


def index_recording(arguments):
    path, validation_recordings, excluded_recordings = arguments
    path = Path(path)
    recording = recording_identity(path.stem)
    if recording in excluded_recordings:
        return {"log": path.stem, "excluded_evaluation_recording": True, "episodes": [], "planning_inputs": {}}
    raw = path.read_bytes()
    frames = pickle.loads(raw)
    paths = [ROOT / "dataset/sensor_blobs/trainval" / frame["cams"]["CAM_F0"]["data_path"] for frame in frames]
    present = [path.is_file() for path in paths]
    episodes, consecutive = [], []
    def finish():
        if len(consecutive) >= 8:
            split = "validation" if recording in validation_recordings else "train"
            episodes.append({"dataset": "OpenScene", "recording": recording, "log": path.stem,
                "split": split, "storage": "image_paths", "frame_paths": [str(paths[index]) for index in consecutive],
                "timestamps_microseconds": [int(frames[index]["timestamp"]) for index in consecutive],
                "frame_count": len(consecutive), "duration_seconds": len(consecutive) / 2,
                "metadata_sha256": hashlib.sha256(raw).hexdigest()})
    for index, exists in enumerate(present):
        adjacent = not consecutive or 400000 < frames[index]["timestamp"] - frames[consecutive[-1]]["timestamp"] < 600000
        if not exists or not adjacent:
            finish()
            consecutive = []
        if exists:
            consecutive.append(index)
    finish()
    planning_inputs = {frame["token"]: {"observed_front_paths": [str(paths[index - 1]), str(paths[index])],
        "history_complete": bool(index > 0 and present[index] and present[index - 1]),
        "interval_seconds": (frame["timestamp"] - frames[index - 1]["timestamp"]) / 1e6}
        for index, frame in enumerate(frames) if index > 0}
    return {"log": path.stem, "metadata_frames": len(frames), "available_frames": sum(present),
            "episodes": episodes, "planning_inputs": planning_inputs}


def main(arguments):
    import yaml
    output = arguments.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    planning_source = ROOT / "outputs/lpwm_drivor_joint_v1/scene_cache/manifest.json"
    planning = json.loads(planning_source.read_text())
    validation_recordings = {row["recording_group"] for row in planning["records"] if row["split"] == "navval"}
    training_recordings = {row["recording_group"] for row in planning["records"] if row["split"] == "navtrain"}
    assert not validation_recordings & training_recordings
    exclusion_files = [ROOT / "reference_repositories/Drive-JEPA" / benchmark / "navsim/planning/script/config/common/train_test_split/scene_filter/navtest.yaml"
                       for benchmark in ("navsim_v1", "navsim_v2")]
    excluded_recordings = set()
    for path in exclusion_files:
        content = yaml.safe_load(path.read_text())
        excluded_recordings.update(recording_identity(name) for name in content.get("log_names", []))
    assert not training_recordings & excluded_recordings
    by_log = {}
    for row in planning["records"]:
        by_log.setdefault(row["log_name"], []).append(row)
    started = time.time()
    scenes, summaries, counts, missing_history, cadence_exceptions = [], [], Counter(), [], []
    sources = sorted((ROOT / "dataset/navsim_logs/trainval").glob("*.pkl"))
    with (output / "local_openscene_episodes.jsonl").open("w") as stream, ProcessPoolExecutor(max_workers=arguments.workers) as executor:
        jobs = [(str(path), validation_recordings, excluded_recordings) for path in sources]
        for result in executor.map(index_recording, jobs, chunksize=1):
            for episode in result["episodes"]:
                stream.write(json.dumps(episode) + "\n")
                counts[episode["split"] + "_frames"] += episode["frame_count"]
                counts[episode["split"] + "_clips_stride8"] += episode["frame_count"] // 8
            for record in by_log.get(result["log"], []):
                observed = result["planning_inputs"][record["token"]]
                if not observed["history_complete"]:
                    missing_history.append({"token": record["token"], "log": result["log"], **observed})
                if not .4 < observed["interval_seconds"] < .6:
                    cadence_exceptions.append({"token": record["token"], "log": result["log"], **observed})
                scenes.append(record | observed)
            summaries.append({key: value for key, value in result.items() if key not in ("episodes", "planning_inputs")})
            progress = {"completed_logs": len(summaries), "total_logs": len(sources), "counts": dict(counts), "elapsed_seconds": time.time() - started}
            (output / "index_progress.json").write_text(json.dumps(progress, indent=2) + "\n")
            if len(summaries) % 50 == 0:
                print(json.dumps(progress), flush=True)
    assert len(scenes) == len(planning["records"])
    scenes.sort(key=lambda row: row["index"])
    manifest = {"complete": not missing_history, "records": scenes, "counts": dict(Counter(row["split"] for row in scenes)),
        "front_camera_only": True, "observations": 2, "observation_order": "previous,current",
        "image_width": 512, "image_height": 256, "crop_top_bottom_pixels": 28,
        "resize": "cv2.INTER_LINEAR", "source_manifest_sha256": hashlib.sha256(planning_source.read_bytes()).hexdigest(),
        "train_validation_recording_overlap": 0, "history_cadence_exceptions": len(cadence_exceptions),
        "actual_timestamps_preserved": True}
    (output / "planning_manifest.json").write_text(json.dumps(manifest) + "\n")
    (output / "missing_planning_history.json").write_text(json.dumps(missing_history, indent=2) + "\n")
    (output / "planning_history_cadence_exceptions.json").write_text(json.dumps(cadence_exceptions, indent=2) + "\n")
    report = {"complete": True, "counts": dict(counts), "hours": {split: counts[split + "_frames"] / 7200 for split in ("train", "validation")},
        "validation_recordings": sorted(validation_recordings), "excluded_recordings": sorted(excluded_recordings),
        "unique_duration_not_overlapping_clip_sum": True, "missing_planning_history_count": len(missing_history),
        "planning_history_cadence_exceptions": len(cadence_exceptions),
        "planning_ready": not missing_history, "logs": summaries, "elapsed_seconds": time.time() - started}
    (output / "index_complete.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: value for key, value in report.items() if key != "logs"}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/lpwm_driving_video_512x256_v1/corpus")
    parser.add_argument("--workers", type=int, default=4)
    main(parser.parse_args())
