"""Finite authorized pilot; validation gates, common profile, A/B/C, evaluation."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TRAIN_PYTHON = "/rhome/junseong/envs/kjs-drive-jepa-extension/bin/python"
EVALUATION_PYTHON = str(PROJECT_ROOT / "runtime/environments/drive_jepa_official_evaluation/bin/python")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/soft_token_reweighting/pilot_v1.json")
    parser.add_argument("--reuse-complete-cache", action="store_true")
    parser.add_argument("--replay-id", help="Create a separate output/config/results set without overwriting the completed pilot")
    arguments = parser.parse_args()
    config_path = (PROJECT_ROOT / arguments.config).resolve()
    configuration = json.loads(config_path.read_text())
    if arguments.replay_id:
        if not arguments.replay_id.replace("_", "").replace("-", "").isalnum():
            raise ValueError("Replay identifier must use letters, digits, underscores or dashes")
        configuration["output_directory"] += "_" + arguments.replay_id
        configuration["results_directory"] += "_" + arguments.replay_id
        replay_root = PROJECT_ROOT / configuration["output_directory"]
        replay_root.mkdir(parents=True, exist_ok=False)
        config_path = replay_root / "replay_configuration.json"
        config_path.write_text(json.dumps(configuration, indent=2)+"\n")
    output = PROJECT_ROOT / configuration["output_directory"]
    output.mkdir(parents=True, exist_ok=True)
    environment = dict(os.environ, OMP_NUM_THREADS="2", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1",
        LD_LIBRARY_PATH=str(PROJECT_ROOT / "runtime/environments/drive_jepa_official_evaluation/lib"))
    stages = [] if arguments.reuse_complete_cache else [("prepare", None), ("cache", None)]
    if arguments.reuse_complete_cache:
        assert json.loads((output / "cache/metadata.json").read_text())["complete"]
    stages += [("verify", None), ("profile", None)]
    for condition in configuration["conditions"]:
        stages += [("train", condition), ("predict", condition), ("score", condition)]
    stages += [("latency", None), ("diagnose", None), ("report", None)]
    started = time.time()
    for stage, condition in stages:
        state = {"status": "running", "stage": stage, "condition": condition, "started_unix": started}
        (output / "suite_state.json").write_text(json.dumps(state, indent=2)+"\n")
        child_environment = environment.copy()
        child_environment["CUDA_VISIBLE_DEVICES"] = str(configuration["physical_gpu"]) if stage in {"cache", "profile", "train", "predict", "latency", "diagnose"} else ""
        interpreter = EVALUATION_PYTHON if stage == "score" else TRAIN_PYTHON
        command = [interpreter, "-u", str(PROJECT_ROOT / "scripts/run_soft_token_reweighting.py"), stage,
                   "--config", str(config_path)]
        if stage == "diagnose":
            command = [interpreter, "-u", str(PROJECT_ROOT / "scripts/diagnose_soft_token_reweighting.py"), "--config", str(config_path)]
        if condition:
            command += ["--condition", condition]
        log_path = output / ("stage_" + stage + ("_"+condition if condition else "") + ".log")
        print(json.dumps(state), flush=True)
        with log_path.open("w") as stream:
            child = subprocess.Popen(command, cwd=PROJECT_ROOT, env=child_environment,
                                     stdout=stream, stderr=subprocess.STDOUT)
            state["child_pid"] = child.pid
            (output / "suite_state.json").write_text(json.dumps(state, indent=2)+"\n")
            return_code = child.wait()
        if return_code:
            state.update(status="failed", return_code=return_code, log=str(log_path))
            (output / "suite_state.json").write_text(json.dumps(state, indent=2)+"\n")
            raise RuntimeError(state)
    (output / "suite_state.json").write_text(json.dumps({"status": "complete", "wall_seconds": time.time()-started}, indent=2)+"\n")


if __name__ == "__main__":
    main()
