"""Persistent CPU service using DrivoR's unmodified online subscore oracle.

Each request scores the CURRENT generated coordinates. Existing standard metric
caches are converted to DrivoR's training-cache format in a new directory.
There is no fixed-vocabulary score lookup and no oracle gradient at inference.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
import contextlib
from functools import lru_cache
import json
import lzma
import multiprocessing
import os
from pathlib import Path
import pickle
import select
import subprocess
import sys

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
STATE = {}


def atomic_training_cache_dump(cache):
    """Same official pickle/LZMA bytes, atomic local I/O instead of aiofiles."""
    destination = Path(cache.file_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    pending = destination.with_suffix(f".{os.getpid()}.pending.pkl")
    pending.write_bytes(lzma.compress(pickle.dumps(cache, protocol=pickle.HIGHEST_PROTOCOL), preset=0))
    pending.replace(destination)


def initialize_worker(manifest_path, cache_root):
    sys.stdout = sys.stderr
    sys.path.insert(0, str(PROJECT_ROOT / "reference_repositories/DrivoR"))
    os.environ["NUPLAN_MAPS_ROOT"] = str(PROJECT_ROOT / "outputs/lpwm_drivor_joint_v1/maps")
    os.environ["NUPLAN_MAP_VERSION"] = "nuplan-maps-v1.0"
    from navsim.planning.metric_caching.train_metric_chache import MetricCache
    MetricCache.dump = atomic_training_cache_dump
    records = json.loads(Path(manifest_path).read_text())["records"]
    existing = json.loads((PROJECT_ROOT / "outputs/lpwm_candidate_teacher_v1/metric_cache_manifest.json").read_text())
    STATE.update(records={record["token"]: record for record in records},
                 existing={record["token"]: record["metric_cache_file"] for record in existing},
                 cache_root=Path(cache_root))


@lru_cache(maxsize=4)
def load_log(log_name):
    with (PROJECT_ROOT / "dataset/navsim_logs/trainval" / (log_name + ".pkl")).open("rb") as stream:
        return pickle.load(stream)


def prepare_training_cache(token):
    from navsim.planning.metric_caching.train_metric_chache import MetricCache
    from navsim.planning.metric_caching import train_cache_processor as official_cache
    record = STATE["records"][token]
    destination = STATE["cache_root"] / record["log_name"] / "unknown" / token / "metric_cache.pkl"
    if destination.exists():
        return destination
    if token in STATE["existing"]:
        with lzma.open(STATE["existing"][token], "rb") as stream:
            original = pickle.load(stream)
        # These are the exact operations in DrivoR's train_cache_processor.
        predicted_states = official_cache.get_trajectory_as_array(original.trajectory,
            official_cache.proposal_sampling, original.ego_state.time_point)
        simulated = official_cache.simulator.simulate_proposals(predicted_states[None], original.ego_state)
        scorer = official_cache.scorer
        scorer._reset(simulated, original.observation, original.centerline,
                      original.route_lane_ids, original.drivable_area_map)
        scorer._calculate_no_at_fault_collision()
        scorer._calculate_drivable_area_compliance()
        scorer._calculate_progress()
        progress = scorer._progress_raw * scorer._multi_metrics.prod(axis=0).cumprod(axis=-1)
        original.drivable_area_map._geometries = [
            np.asarray(polygon.__geo_interface__["coordinates"][0], dtype=np.float32)
            for polygon in original.drivable_area_map._geometries]
        converted = MetricCache(destination, original.ego_state, original.observation,
            original.centerline._linestring, original.route_lane_ids, original.drivable_area_map, progress)
        destination.parent.mkdir(parents=True, exist_ok=True)
        pending = destination.with_suffix(f".{os.getpid()}.pending.pkl")
        converted.file_path = destination
        with lzma.open(pending, "wb", preset=0) as stream:
            pickle.dump(converted, stream, protocol=pickle.HIGHEST_PROTOCOL)
        pending.replace(destination)
    else:
        from navsim.common.dataclasses import Scene, SensorConfig
        from navsim.planning.scenario_builder.navsim_scenario import NavSimScenario
        frames = load_log(record["log_name"])
        current_index = record["current_frame_index"]
        scene = Scene.from_scene_dict_list(frames[current_index - 3:current_index + 11], None,
                                           4, 10, SensorConfig.build_no_sensors())
        scenario = NavSimScenario(scene, os.environ["NUPLAN_MAPS_ROOT"], "nuplan-maps-v1.0")
        processor = official_cache.MetricCacheProcessor(str(STATE["cache_root"]), False)
        generated = processor.compute_metric_cache(scenario)
        assert Path(generated.file_name) == destination
    return destination


def score_scene(job):
    from navsim.agents.drivoR.score_module.compute_navsim_score import get_sub_score
    token, proposals = job
    path = prepare_training_cache(token)
    # Auxiliary object outputs are disabled in official configuration; test=True
    # skips their construction, without changing any of the seven subscores.
    scores = get_sub_score(path, np.asarray(proposals, np.float32), test=True)[0]
    assert scores.shape == (len(proposals), 7) and np.isfinite(scores).all(), token
    return scores.astype(np.float32)


class DrivoROracleClient:
    def __init__(self, manifest_path, output_directory, rank=0, workers=4):
        self.directory = Path(output_directory).resolve() / "oracle_requests" / f"rank{rank}"
        self.directory.mkdir(parents=True, exist_ok=True)
        self.log = (self.directory / "server.log").open("a")
        oracle_environment = PROJECT_ROOT / "runtime/environments/drive_jepa_official_evaluation"
        environment = {**os.environ, "CUDA_VISIBLE_DEVICES": "", "OMP_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
            "LD_LIBRARY_PATH": str(oracle_environment / "lib"),
            "PYTHONPATH": str(PROJECT_ROOT / "reference_repositories/DrivoR")}
        self.process = subprocess.Popen([str(oracle_environment / "bin/python"), "-u", str(Path(__file__).resolve()),
            "--manifest", str(Path(manifest_path).resolve()), "--cache-root",
            str(PROJECT_ROOT / "outputs/lpwm_drivor_joint_v1/train_metric_cache"), "--workers", str(workers)],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=self.log, text=True, bufsize=1,
            cwd=PROJECT_ROOT, env=environment)

    def score(self, tokens, proposals):
        request, response = self.directory / "request.npz", self.directory / "response.npy"
        np.savez(request, tokens=np.asarray(tokens), proposals=np.asarray(proposals, np.float32))
        self.process.stdin.write(json.dumps({"request": str(request), "response": str(response)}) + "\n")
        self.process.stdin.flush()
        ready, _, _ = select.select([self.process.stdout], [], [], 600)
        if not ready:
            raise TimeoutError("DrivoR online oracle timed out; labels are never substituted")
        message = self.process.stdout.readline()
        if not message:
            raise RuntimeError("DrivoR oracle exited: " + str(self.directory / "server.log"))
        assert json.loads(message)["response"] == str(response)
        scores = np.load(response)
        assert scores.shape == (len(tokens), proposals.shape[1], 7)
        return scores

    def close(self):
        if self.process.poll() is None:
            self.process.stdin.close()
            try:
                self.process.wait(timeout=20)
            except subprocess.TimeoutExpired:
                self.process.terminate()
        self.log.close()


def serve(arguments):
    protocol_stream = sys.stdout
    with contextlib.redirect_stdout(sys.stderr), ProcessPoolExecutor(max_workers=arguments.workers,
            mp_context=multiprocessing.get_context("spawn"), initializer=initialize_worker,
            initargs=(str(arguments.manifest), str(arguments.cache_root))) as pool:
        for line in sys.stdin:
            message = json.loads(line)
            with np.load(message["request"]) as request:
                jobs = list(zip(request["tokens"].tolist(), request["proposals"]))
            scores = np.stack(list(pool.map(score_scene, jobs)))
            np.save(message["response"], scores)
            protocol_stream.write(json.dumps({"response": message["response"]}) + "\n")
            protocol_stream.flush()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    serve(parser.parse_args())
