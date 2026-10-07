"""Compare CPU oracle workers without changing or loading the active model."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import statistics
import time

import numpy as np

from lpwm_drivor_oracle import DrivoROracleClient, PROJECT_ROOT


def read_current_request(path):
    for attempt in range(20):
        try:
            with np.load(path) as request:
                return request["tokens"].tolist(), request["proposals"].copy()
        except (OSError, ValueError, EOFError):
            if attempt == 19:
                raise
            time.sleep(.1)


def score_pair(clients, requests, executor):
    started = time.perf_counter()
    futures = [executor.submit(client.score, *request)
               for client, request in zip(clients, requests)]
    scores = [future.result() for future in futures]
    return time.perf_counter() - started, scores


def main(arguments):
    output = arguments.output_directory.resolve()
    output.mkdir(parents=True, exist_ok=False)
    configuration = json.loads(arguments.config.read_text())
    training_root = PROJECT_ROOT / configuration["output_directory"]
    manifest = PROJECT_ROOT / configuration["manifest"]
    requests = [read_current_request(training_root / "oracle_requests" /
                                     f"rank{rank}" / "request.npz")
                for rank in range(2)]
    for rank, (tokens, proposals) in enumerate(requests):
        np.savez(output / f"rank{rank}_fixed_request.npz", tokens=tokens, proposals=proposals)
    # Two equal microbatches form each rank's effective batch of 32 scenes.
    expected_scores = None
    measurements = []
    for round_index, workers in enumerate((4, 8, 6, 4)):
        directory = output / f"round_{round_index}_workers_{workers}"
        clients = [DrivoROracleClient(manifest, directory, rank=rank, workers=workers)
                   for rank in range(2)]
        try:
            with ThreadPoolExecutor(max_workers=2) as executor:
                startup_seconds, scores = score_pair(clients, requests, executor)
                if expected_scores is None:
                    expected_scores = scores
                for reference, observed in zip(expected_scores, scores):
                    np.testing.assert_array_equal(observed, reference)
                durations = []
                for _ in range(arguments.repeats):
                    duration, scores = score_pair(clients, requests, executor)
                    for reference, observed in zip(expected_scores, scores):
                        np.testing.assert_array_equal(observed, reference)
                    durations.append(duration)
        finally:
            for client in clients:
                client.close()
        measurement = {"workers_per_rank": workers, "startup_seconds": startup_seconds,
                       "paired_request_seconds": durations,
                       "median_seconds": statistics.median(durations),
                       "mean_seconds": statistics.mean(durations),
                       "all_seven_subscores_exactly_equal": True}
        measurements.append(measurement)
        print(json.dumps(measurement), flush=True)
        (output / "progress.json").write_text(json.dumps(measurements, indent=2) + "\n")
    report = {"cpu_only": True, "training_interrupted": False,
              "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "scenes_per_rank_request": [len(request[0]) for request in requests],
              "proposals_per_scene": [request[1].shape[1] for request in requests],
              "measurements": measurements,
              "limitations": ["Fixed current proposals from two ranks; warm metric caches.",
                               "Runs alongside active training on a shared CPU.",
                               "Pair wall time estimates CPU-only oracle speed, not full training throughput."]}
    (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=5)
    main(parser.parse_args())
