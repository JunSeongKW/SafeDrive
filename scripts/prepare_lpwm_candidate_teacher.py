"""Train-only trajectory vocabulary and NAVSIM-v1 metric distillation targets.

Run with the official NAVSIM environment on CPU. Future scene states are labels,
never planner inputs. Batch progress is renormalized against the PDM reference
individually, then checked against unchanged official single-trajectory scoring.
"""
import argparse
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
import json
import lzma
import multiprocessing
import os
from pathlib import Path
import pickle
import time

import numpy as np

from evaluate_lpwm_full_planning import digest, write_json

PROJECT_ROOT = Path(__file__).resolve().parents[1]
METRIC_NAMES = ["no_at_fault_collisions", "drivable_area_compliance", "ego_progress",
                "time_to_collision_within_bound", "comfort", "driving_direction_compliance", "score"]
WORKER_STATE = {}


def score_candidate_trajectories(metric_cache, candidates, simulator, scorer):
    from navsim.common.dataclasses import Trajectory
    from navsim.evaluate.pdm_score import transform_trajectory, get_trajectory_as_array
    from navsim.planning.simulation.planner.pdm_planner.utils.pdm_enums import MultiMetricIndex, WeightedMetricIndex
    initial = metric_cache.ego_state
    sampling = simulator.proposal_sampling
    trajectories = [metric_cache.trajectory] + [transform_trajectory(Trajectory(candidate), initial) for candidate in candidates]
    states = np.stack([get_trajectory_as_array(trajectory, sampling, initial.time_point) for trajectory in trajectories])
    simulated = simulator.simulate_proposals(states, initial)
    scorer.score_proposals(simulated, metric_cache.observation, metric_cache.centerline,
                           metric_cache.route_lane_ids, metric_cache.drivable_area_map)
    multiplicative = scorer._multi_metrics.prod(0)
    raw_progress = scorer._progress_raw * multiplicative
    pair_maximum = np.maximum(raw_progress[0], raw_progress[1:])
    progress = np.where(pair_maximum > scorer._config.progress_distance_threshold,
                        raw_progress[1:] / np.maximum(pair_maximum, 1e-12), (multiplicative[1:] != 0).astype(float))
    weighted = scorer._weighted_metrics[:, 1:].copy()
    weighted[WeightedMetricIndex.PROGRESS] = progress
    weights = scorer._config.weighted_metrics_array
    scores = multiplicative[1:] * (weighted * weights[:, None]).sum(0) / weights.sum()
    return np.stack([scorer._multi_metrics[MultiMetricIndex.NO_COLLISION, 1:],
        scorer._multi_metrics[MultiMetricIndex.DRIVABLE_AREA, 1:], progress,
        weighted[WeightedMetricIndex.TTC], weighted[WeightedMetricIndex.COMFORTABLE],
        weighted[WeightedMetricIndex.DRIVING_DIRECTION], scores], -1).astype(np.float32)


def initialize_worker(output_root):
    import faulthandler
    faulthandler.dump_traceback_later(180, repeat=True)
    from audit_official_drive_jepa_evaluation import official_configuration
    from hydra.utils import instantiate
    _, _, configuration, _ = official_configuration(PROJECT_ROOT)
    from navsim.planning.metric_caching.metric_cache_processor import MetricCacheProcessor
    WORKER_STATE.update(root=Path(output_root), candidates=np.load(Path(output_root) / "trajectory_vocabulary.npy"),
        processor=MetricCacheProcessor(str(Path(output_root) / "metric_cache"), False),
        simulator=instantiate(configuration.simulator), scorer=instantiate(configuration.scorer))


def prepare_segment(segment_and_records):
    from navsim.common.dataclasses import Scene, SensorConfig, Trajectory
    from navsim.planning.scenario_builder.navsim_scenario import NavSimScenario
    from navsim.evaluate.pdm_score import pdm_score
    filename, indexed_records = segment_and_records
    available_kib = next(int(line.split()[1]) for line in Path("/proc/meminfo").read_text().splitlines() if line.startswith("MemAvailable:"))
    if available_kib < 64 * 1024**2:
        raise MemoryError("CPU teacher requires at least 64 GiB host memory reserve; no automatic retry")
    root = WORKER_STATE["root"]
    destination = root / "segments" / (Path(filename).stem + ".npz")
    if destination.exists():
        return str(destination)
    source_path = PROJECT_ROOT / "dataset/navsim_logs/trainval" / filename
    with source_path.open("rb") as stream:
        log_frames = pickle.load(stream)
    results, valid_indices, entries, unavailable, parity = [], [], [], [], []
    started = time.monotonic()
    for record_index, record in indexed_records:
        import faulthandler
        faulthandler.cancel_dump_traceback_later()
        faulthandler.dump_traceback_later(180, repeat=True)
        frames = log_frames[record["start_index"]:record["start_index"] + 14]
        token = record["current_frame_token"]
        if len(frames) != 14 or not np.allclose(np.diff([frame["timestamp"] for frame in frames]) / 1e6, .5, atol=.1):
            unavailable.append({"index": record_index, "token": token, "reason": "missing_or_irregular_5s_future"})
            continue
        assert frames[3]["token"] == token
        scene = Scene.from_scene_dict_list(frames, PROJECT_ROOT / "dataset/sensor_blobs/trainval", 4, 10, SensorConfig.build_no_sensors())
        scenario = NavSimScenario(scene, str(PROJECT_ROOT / "dataset/maps"), "nuplan-maps-v1.0")
        cache_path = root / "metric_cache" / scenario.log_name / scenario.scenario_type / scenario.token / "metric_cache.pkl"
        if cache_path.exists() and cache_path.stat().st_size == 0:
            cache_path.rename(cache_path.with_name("metric_cache.interrupted_empty.pkl"))
        metadata = WORKER_STATE["processor"].compute_metric_cache(scenario)
        with lzma.open(metadata.file_name, "rb") as stream:
            metric_cache = pickle.load(stream)
        labels = score_candidate_trajectories(metric_cache, WORKER_STATE["candidates"], WORKER_STATE["simulator"], WORKER_STATE["scorer"])
        assert np.isfinite(labels).all() and labels.min() >= 0 and labels.max() <= 1.000001
        # Audit diverse candidates for every segment, including the oracle winner.
        if not results:
            chosen = sorted(set([0, len(labels) // 2, len(labels) - 1, int(labels[:, -1].argmax())]))
            maximum_difference = 0.
            for candidate_index in chosen:
                official = asdict(pdm_score(metric_cache, Trajectory(WORKER_STATE["candidates"][candidate_index]),
                    WORKER_STATE["simulator"].proposal_sampling, WORKER_STATE["simulator"], WORKER_STATE["scorer"]))
                difference = float(np.abs(labels[candidate_index] - np.array([official[name] for name in METRIC_NAMES])).max())
                maximum_difference = max(maximum_difference, difference)
                np.testing.assert_allclose(labels[candidate_index], [official[name] for name in METRIC_NAMES], atol=2e-5, rtol=0)
            parity.append({"token": token, "candidate_indices": chosen, "maximum_difference": maximum_difference})
        results.append(labels)
        valid_indices.append(record_index)
        entries.append({"token": token, "index": record_index, "split": record["split"],
            "recording_group": record["recording_group"], "metric_cache_file": str(metadata.file_name)})
        if len(entries) == 1 or len(entries) % 8 == 0:
            write_json(root / "worker_progress" / (Path(filename).stem + ".json"), {"segment": filename,
                "completed": len(entries), "total": len(indexed_records), "seconds": time.monotonic() - started,
                "worker_pid": os.getpid(), "heartbeat_unix": time.time()})
    destination.parent.mkdir(parents=True, exist_ok=True)
    pending = destination.with_suffix(".pending.npz")
    labels = np.stack(results) if results else np.empty((0, len(WORKER_STATE["candidates"]), 7), np.float32)
    np.savez(pending, indices=np.array(valid_indices, dtype=np.int64), labels=labels)
    write_json(destination.with_suffix(".json"), {"entries": entries, "unavailable": unavailable,
        "parity": parity, "seconds": time.monotonic() - started, "source_log_sha256": digest(source_path)})
    pending.replace(destination)
    faulthandler.cancel_dump_traceback_later()
    return str(destination)


def prepare_vocabulary(root, records, targets, specification):
    path = root / "trajectory_vocabulary.npy"
    if path.exists():
        return
    from sklearn.cluster import MiniBatchKMeans
    from sklearn.metrics import pairwise_distances_argmin_min
    train_indices = np.array([index for index, record in enumerate(records) if record["split"] == "train"])
    descriptors = targets[train_indices, :, :2].reshape(len(train_indices), -1)
    clustering = MiniBatchKMeans(n_clusters=specification["candidate_count"], random_state=specification["seed"],
        batch_size=4096, n_init=3, max_iter=200, reassignment_ratio=.005).fit(descriptors)
    nearest, _ = pairwise_distances_argmin_min(clustering.cluster_centers_, descriptors)
    source_indices = train_indices[nearest]
    assert len(set(source_indices.tolist())) == specification["candidate_count"], "Duplicate vocabulary medoids"
    candidates = targets[source_indices]
    np.save(path, candidates)
    diagnostics = {}
    for split in ("train", "development"):
        indices = np.array([index for index, record in enumerate(records) if record["split"] == split])
        minimum_errors = []
        for offset in range(0, len(indices), 128):
            distances = np.linalg.norm(targets[indices[offset:offset + 128], None, :, :2] - candidates[None, :, :, :2], axis=-1).mean(-1)
            minimum_errors.extend(distances.min(-1).tolist())
        diagnostics[split] = {"scenes": len(indices), "oracle_ade_meters": float(np.mean(minimum_errors)),
            "oracle_ade_p95_meters": float(np.quantile(minimum_errors, .95))}
    write_json(root / "vocabulary.json", {"train_only": True, "count": len(candidates), "source_training_indices": source_indices.tolist(),
        "source_training_tokens": [records[index]["current_frame_token"] for index in source_indices],
        "sha256": digest(path), "coverage": diagnostics, "construction": "train-only XY MiniBatchKMeans, real training trajectory medoids"})


def main(arguments):
    specification = json.loads(arguments.config.read_text())
    root = PROJECT_ROOT / specification["teacher_directory"]
    root.mkdir(parents=True, exist_ok=True)
    import fcntl
    lock = (root / "preparation.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    stage1_config = json.loads((PROJECT_ROOT / specification["stage1_config"]).read_text())
    stage1_root = PROJECT_ROOT / stage1_config["output_directory"]
    manifest_path = stage1_root / "planning_manifest.json"
    records = json.loads(manifest_path.read_text())["records"]
    with np.load(stage1_root / "planning_targets.npz") as stored:
        targets = stored["ego_trajectory_target"]
    identity = {"configuration_sha256": digest(arguments.config), "manifest_sha256": digest(manifest_path),
        "targets_sha256": digest(stage1_root / "planning_targets.npz"), "teacher_source_sha256": digest(__file__)}
    official_root = PROJECT_ROOT / "reference_repositories/DriveJEPAOfficialEvaluation/navsim_v1/navsim"
    identity["official_oracle_source_sha256"] = {name: digest(official_root / name) for name in (
        "evaluate/pdm_score.py", "planning/simulation/planner/pdm_planner/scoring/pdm_scorer.py",
        "planning/simulation/planner/pdm_planner/simulation/pdm_simulator.py")}
    if (root / "identity.json").exists():
        assert json.loads((root / "identity.json").read_text()) == identity
    else:
        write_json(root / "identity.json", identity)
    if (root / "completion.json").exists() and not arguments.profile:
        return
    prepare_vocabulary(root, records, targets, specification)
    grouped = defaultdict(list)
    selected = range(len(records))
    if arguments.profile:
        selected = [next(index for index, record in enumerate(records) if record["split"] == split) for split in ("train", "development")]
        profile_root = root / "profile"
        profile_root.mkdir(exist_ok=True)
        vocabulary_link = profile_root / "trajectory_vocabulary.npy"
        if not vocabulary_link.exists():
            vocabulary_link.symlink_to(root / "trajectory_vocabulary.npy")
        root = profile_root
    for index in selected:
        grouped[records[index]["segment_filename"]].append((index, records[index]))
    started = time.monotonic()
    paths = []
    with ProcessPoolExecutor(max_workers=min(arguments.workers, len(grouped)), mp_context=multiprocessing.get_context("spawn"),
                             initializer=initialize_worker, initargs=(str(root),)) as executor:
        futures = [executor.submit(prepare_segment, item) for item in sorted(grouped.items())]
        for future in as_completed(futures):
            paths.append(future.result())
            write_json(root / "progress.json", {"segments_completed": len(paths), "segments_total": len(grouped),
                "seconds": time.monotonic() - started, "workers": arguments.workers, "supervisor_pid": os.getpid()})
    if arguments.profile:
        write_json(root / "completion.json", {"profile_only": True, "segments": [json.loads(Path(path).with_suffix(".json").read_text()) for path in paths]})
        return
    labels_path = root / "candidate_metrics.npy"
    labels = np.lib.format.open_memmap(labels_path.with_suffix(".pending.npy"), mode="w+", dtype=np.float32,
        shape=(len(records), specification["candidate_count"], len(METRIC_NAMES)))
    labels[:] = np.nan
    entries, unavailable, parity = [], [], []
    for path in paths:
        with np.load(path) as segment:
            labels[segment["indices"]] = segment["labels"]
        segment_report = json.loads(Path(path).with_suffix(".json").read_text())
        entries.extend(segment_report["entries"])
        unavailable.extend(segment_report["unavailable"])
        parity.extend(segment_report["parity"])
    assert len(entries) + len(unavailable) == len(records)
    assert len({entry["index"] for entry in entries}) == len(entries)
    labels.flush()
    del labels
    labels_path.with_suffix(".pending.npy").replace(labels_path)
    counts = Counter(entry["split"] for entry in entries)
    total_counts = Counter(record["split"] for record in records)
    write_json(root / "metric_cache_manifest.json", entries)
    coverage = {split: counts[split] / total_counts[split] for split in total_counts}
    write_json(root / "completion.json", {"complete": True, "teacher_gate_passed": all(value >= .95 for value in coverage.values()),
        "coverage": coverage, "valid_counts": counts, "unavailable": unavailable,
        "official_pair_scoring_parity": parity, "metric_names": METRIC_NAMES,
        "labels_sha256": digest(labels_path), "vocabulary_sha256": digest(root / "trajectory_vocabulary.npy"),
        "identity": identity, "seconds_this_invocation": time.monotonic() - started,
        "privileged_future_is_training_supervision_only": True})
    print("CANDIDATE_TEACHER_READY", dict(counts), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--profile", action="store_true")
    main(parser.parse_args())
