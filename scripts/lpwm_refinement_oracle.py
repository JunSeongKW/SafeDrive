"""CPU-only official NAVSIM oracle for the model's actual refined trajectories.

The client runs in the LPWM environment; its persistent subprocess and worker
pool run in the official NAVSIM environment. Only detached candidate poses go
to this training-label service. Nothing here is called by planner inference.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
import contextlib
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
ORACLE_STATE = {}


def initialize_oracle(manifest_path):
    sys.stdout = sys.stderr
    from audit_official_drive_jepa_evaluation import official_configuration
    from hydra.utils import instantiate
    _, _, configuration, _ = official_configuration(PROJECT_ROOT)
    entries = json.loads(Path(manifest_path).read_text())
    ORACLE_STATE.update(entries={entry["index"]: entry for entry in entries},
        simulator=instantiate(configuration.simulator), scorer=instantiate(configuration.scorer))


def score_refined_scene(index_and_candidates):
    from prepare_lpwm_candidate_teacher import score_candidate_trajectories
    from navsim.planning.simulation.planner.pdm_planner.utils.pdm_enums import EgoAreaIndex
    index, candidates = index_and_candidates
    entry = ORACLE_STATE["entries"].get(int(index))
    if entry is None:
        return np.full((len(candidates), 7), np.nan, np.float32), np.full((len(candidates), 8, 2), np.nan, np.float32)
    with lzma.open(entry["metric_cache_file"], "rb") as stream:
        metric_cache = pickle.load(stream)
    scorer, simulator = ORACLE_STATE["scorer"], ORACLE_STATE["simulator"]
    metrics = score_candidate_trajectories(metric_cache, np.asarray(candidates, dtype=np.float32), simulator, scorer)
    time_indices = np.arange(5, 41, 5)
    no_collision_by_time = time_indices[None] < scorer._collision_time_idcs[1:, None]
    offroad_prefix = np.maximum.accumulate(scorer._ego_areas[1:, :, EgoAreaIndex.NON_DRIVABLE_AREA], axis=1)
    drivable_by_time = ~offroad_prefix[:, time_indices]
    temporal = np.stack((no_collision_by_time, drivable_by_time), -1).astype(np.float32)
    assert metrics.shape == (len(candidates), 7) and temporal.shape == (len(candidates), 8, 2)
    return metrics, temporal


class RefinementOracleClient:
    def __init__(self, manifest_path, output_root, rank, workers=4):
        self.output_root = Path(output_root).resolve() / "online_oracle" / f"rank{rank}"
        self.output_root.mkdir(parents=True, exist_ok=True)
        self.log = (self.output_root / "server.log").open("a")
        environment = {**os.environ, "CUDA_VISIBLE_DEVICES": "", "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1"}
        self.process = subprocess.Popen([str(PROJECT_ROOT / "runtime/environments/drive_jepa_official_evaluation/bin/python"),
            str(Path(__file__).resolve()), "--manifest", str(manifest_path), "--workers", str(workers)],
            cwd=PROJECT_ROOT, env=environment, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=self.log,
            text=True, bufsize=1)

    def score(self, record_indices, candidates):
        # Fixed per-rank filenames are safe: at most one outstanding request.
        request = self.output_root / "request.npz"
        response = self.output_root / "response.npz"
        np.savez(request, record_indices=np.asarray(record_indices), candidates=np.asarray(candidates, dtype=np.float32))
        self.process.stdin.write(json.dumps({"request": str(request), "response": str(response)}) + "\n")
        self.process.stdin.flush()
        readable, _, _ = select.select([self.process.stdout], [], [], 180)
        if not readable:
            raise TimeoutError("Official refinement oracle exceeded 180 seconds; training stops, never reuses stale labels")
        line = self.process.stdout.readline()
        if not line:
            raise RuntimeError("Refinement oracle exited; see " + str(self.output_root / "server.log"))
        reply = json.loads(line)
        assert reply["response"] == str(response)
        with np.load(response) as stored:
            return stored["metrics"].copy(), stored["temporal_safety"].copy()

    def close(self):
        if self.process.poll() is None:
            self.process.stdin.close()
            try:
                self.process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                self.process.terminate()
        self.log.close()


def serve(arguments):
    protocol_output = sys.stdout
    # Library progress text cannot corrupt the request/response protocol.
    with contextlib.redirect_stdout(sys.stderr), ProcessPoolExecutor(max_workers=arguments.workers,
            mp_context=multiprocessing.get_context("spawn"), initializer=initialize_oracle,
            initargs=(str(arguments.manifest.resolve()),)) as executor:
        for line in sys.stdin:
            message = json.loads(line)
            with np.load(message["request"]) as stored:
                jobs = list(zip(stored["record_indices"].tolist(), stored["candidates"]))
            scored = list(executor.map(score_refined_scene, jobs))
            np.savez(message["response"], metrics=np.stack([row[0] for row in scored]),
                temporal_safety=np.stack([row[1] for row in scored]))
            protocol_output.write(json.dumps({"response": message["response"]}) + "\n")
            protocol_output.flush()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    serve(parser.parse_args())
