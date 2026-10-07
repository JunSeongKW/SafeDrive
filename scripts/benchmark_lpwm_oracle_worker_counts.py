"""Compare higher oracle parallelism with identical cached training proposals."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import statistics

import numpy as np

from benchmark_lpwm_drivor_oracle_parallelism import read_current_request, score_pair
from lpwm_drivor_oracle import DrivoROracleClient, PROJECT_ROOT


def main(arguments):
    output = arguments.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    configuration = json.loads((PROJECT_ROOT / "configs/lpwm_drivor_planning_path_lora/navsim_v1.json").read_text())
    manifest = PROJECT_ROOT / configuration["manifest"]
    training = PROJECT_ROOT / configuration["output_directory"]
    requests = [read_current_request(training / f"oracle_requests/rank{rank}/request.npz") for rank in (0, 1)]
    for rank, (tokens, proposals) in enumerate(requests):
        np.savez(output / f"rank{rank}_requests.npz", tokens=tokens, proposals=proposals)
    baseline = None
    measurements = []
    for round_index, workers in enumerate((8, 12, 16, 8)):
        clients = [DrivoROracleClient(manifest, output / f"round{round_index}_workers{workers}", rank, workers) for rank in (0, 1)]
        try:
            with ThreadPoolExecutor(max_workers=2) as executor:
                startup_seconds, scores = score_pair(clients, requests, executor)
                if baseline is None:
                    baseline = scores
                for expected, observed in zip(baseline, scores):
                    np.testing.assert_array_equal(expected, observed)
                durations = []
                for _ in range(4):
                    seconds, scores = score_pair(clients, requests, executor)
                    durations.append(seconds)
                    for expected, observed in zip(baseline, scores):
                        np.testing.assert_array_equal(expected, observed)
        finally:
            for client in clients:
                client.close()
        measurements.append({"workers_per_rank": workers, "startup_seconds": startup_seconds,
                             "paired_request_seconds": durations, "median_seconds": statistics.median(durations),
                             "all_scores_bitwise_equal": True})
        (output / "report.json").write_text(json.dumps({"measurements": measurements,
            "fixed_requests_on_shared_cpus": True, "full_training_speedup_not_measured": True}, indent=2) + "\n")
        print(json.dumps(measurements[-1]), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    main(parser.parse_args())
