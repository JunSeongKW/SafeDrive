"""Official NAVSIM v1 CPU scoring of completed LPWM planning trajectories."""
import argparse
import hashlib
import json
import lzma
import pickle
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np

from audit_official_drive_jepa_evaluation import official_configuration
from evaluate_drive_jepa_region_research_pdm import file_sha256, write_json

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = PROJECT_ROOT / "outputs/lpwm_planning_v1"
RESULT_ROOT = PROJECT_ROOT / "results/lpwm_planning_v1"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--wait", action="store_true")
    arguments = parser.parse_args()
    specification = json.loads((PROJECT_ROOT / "configs/lpwm_planning/controlled_v1.json").read_text())
    _, _, configuration, official_root = official_configuration(PROJECT_ROOT)
    from hydra.utils import instantiate
    from navsim.common.dataclasses import Trajectory
    from navsim.evaluate.pdm_score import pdm_score
    simulator, scorer = instantiate(configuration.simulator), instantiate(configuration.scorer)
    assert simulator.proposal_sampling == scorer.proposal_sampling
    manifest = json.loads((PROJECT_ROOT / specification["metric_manifest"]).read_text())
    assert len(manifest) == 192 and len({entry["token"] for entry in manifest}) == 192
    for entry in manifest:
        assert file_sha256(entry["metric_cache_file"]) == entry["cache_sha256"]
    baseline = json.loads((PROJECT_ROOT / "results/encoder_future_learning_v1/development_pdm_results.json").read_text())
    write_json(RESULT_ROOT / "pdm" / "drive_jepa_reference.json", {"windows": baseline["windows"]["original_frozen"], "summary": baseline["summary"]["original_frozen"],
               "source": "Reused unchanged official Drive-JEPA scores on identical development tokens"})
    pending = [f"{condition}_seed{seed}" for condition in specification["conditions"] for seed in specification["seeds"]]
    started = time.monotonic()
    while pending:
        ready = [name for name in pending if (OUTPUT_ROOT / "runs" / name / "results.json").exists()]
        if not ready:
            if not arguments.wait or time.monotonic() - started > 25200:
                raise RuntimeError("Unfinished registered runs: " + str(pending))
            time.sleep(15)
            continue
        for name in ready:
            report = json.loads((OUTPUT_ROOT / "runs" / name / "results.json").read_text())
            assert report["completed_updates"] == specification["updates"]
            assert report["configuration_sha256"] == file_sha256(PROJECT_ROOT / "configs/lpwm_planning/controlled_v1.json")
            variants = {name: report["evaluations"]["development"]["windows"]}
            if report["condition"] == "object_future_risk":
                for intervention in ("zero_predicted_futures", "shuffle_scene_particles"):
                    variants[name + "__" + intervention] = report["evaluations"]["development"][intervention]
            for variant_name, predictions in variants.items():
                destination = RESULT_ROOT / "pdm" / f"{variant_name}.json"
                digest = hashlib.sha256(json.dumps(predictions, sort_keys=True).encode()).hexdigest()
                if destination.exists():
                    assert json.loads(destination.read_text())["prediction_sha256"] == digest
                    continue
                by_token = {row["token"]: row for row in predictions}
                assert set(by_token) == {entry["token"] for entry in manifest}
                scores = []
                for entry in manifest:
                    with lzma.open(entry["metric_cache_file"], "rb") as stream:
                        metric_cache = pickle.load(stream)
                    trajectory = Trajectory(np.asarray(by_token[entry["token"]]["trajectory"], dtype=np.float32))
                    metrics = asdict(pdm_score(metric_cache, trajectory, simulator.proposal_sampling, simulator, scorer))
                    assert all(np.isfinite(value) for value in metrics.values())
                    scores.append({"token": entry["token"], "recording": entry["recording"], **metrics})
                summary = {key: float(np.mean([row[key] for row in scores])) for key in scores[0] if key not in ("token", "recording")}
                write_json(destination, {"windows": scores, "summary": summary, "prediction_sha256": digest, "scorer_source": str(official_root), "scope": "192 development scenes; not independent test"})
                print("LPWM_PDM_DONE", variant_name, summary, flush=True)
            pending.remove(name)
            write_json(OUTPUT_ROOT / "pdm_progress.json", {"remaining_runs": pending, "wall_seconds": time.monotonic() - started})
    write_json(OUTPUT_ROOT / "pdm_complete.json", {"complete": True, "scored_model_trajectories": 24 * 192, "reused_baseline_scenes": 192, "wall_seconds": time.monotonic() - started})
    print("LPWM_PDM_ALL_DONE", flush=True)


if __name__ == "__main__":
    main()
