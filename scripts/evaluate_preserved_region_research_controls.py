"""Read-only PDM evaluation of the preserved global K8 learned/random controls."""

import argparse
import gc
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
import torch

from diagnose_drive_jepa_learning_limitations import load_official_agent
from run_drive_jepa_architecture_followup import WORKSPACE, build_model, module_hashes, observed_forward
from run_drive_jepa_selection_comparison import write_json
from train_drive_jepa_spatial_regions import set_selection_control
from validate_drive_jepa_selective_future_connection import parameter_sha256, file_sha256
from planning_aware_future_prediction.models.spatial_region_future import SpatialRegionSelector


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-directory", type=Path, required=True)
    parser.add_argument("--pdm-directory", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--wait-training-pid", type=int)
    parser.add_argument("--detach", action="store_true")
    args = parser.parse_args()
    for name in ("run_directory", "pdm_directory", "output_directory"):
        setattr(args, name, getattr(args, name).resolve())
    if args.detach:
        command = ["nohup", sys.executable, "-u", str(Path(__file__).resolve()),
                   "--run-directory", str(args.run_directory), "--pdm-directory", str(args.pdm_directory),
                   "--output-directory", str(args.output_directory)]
        if args.wait_training_pid:
            command += ["--wait-training-pid", str(args.wait_training_pid)]
        with args.output_directory.with_suffix(".log").open("x") as stream:
            process = subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT,
                stdin=subprocess.DEVNULL, start_new_session=True, cwd=WORKSPACE)
        print(json.dumps({"pid": process.pid, "output": str(args.output_directory)}), flush=True)
        return
    if os.environ.get("CUDA_VISIBLE_DEVICES") not in ("0", "1"):
        raise RuntimeError("Exactly one approved physical GPU required")
    started = time.perf_counter()
    if args.wait_training_pid:
        while not (args.run_directory / "completion.json").exists():
            if time.perf_counter() - started > 11400:
                raise RuntimeError("Training completion wait cap")
            time.sleep(30)
        while Path(f"/proc/{args.wait_training_pid}/cmdline").exists():
            command_line = Path(f"/proc/{args.wait_training_pid}/cmdline").read_bytes()
            if b"train_drive_jepa_region_research.py" not in command_line:
                break
            time.sleep(5)
    torch.set_num_threads(1)
    if torch.cuda.mem_get_info()[0] < 16 * 2**30:
        raise RuntimeError("GPU admission reserve insufficient; no retry")
    args.output_directory.mkdir(parents=True, exist_ok=False)
    agent, source = load_official_agent(args.output_directory)
    original_hash = parameter_sha256(agent._model)
    if original_hash != "05af3ccd8a8e7db1b9b2ae2e77f81ef753446a3309989cef08c161ce0614476a":
        raise RuntimeError("Original planner hash differs")
    specification = json.loads((args.run_directory / "specification.json").read_text())
    index = json.loads((WORKSPACE / specification["reused_cache"]).read_text())
    records = [row for row in index["records"] if row["split"] == "development"]
    cached = []
    for record in records:
        if file_sha256(record["cache_file"]) != record["cache_sha256"]:
            raise RuntimeError("Preserved feature cache changed")
        loaded = torch.load(record["cache_file"], map_location="cpu", weights_only=True)
        cached.append({name: loaded[name] for name in ("current_patch_latents", "current_ego_status", "ego_trajectory_target")})
    cache = {name: torch.stack([row[name] for row in cached]) for name in cached[0]}
    del cached, loaded
    preserved_root = WORKSPACE / "outputs/drive_jepa_selective_future/spatial_region_selection_v1_20261003"
    predictions = {}
    for label, condition, selection_mode in (("preserved_global_learned", "region8_planning", "learned"),
                                              ("preserved_global_random", "region8_random", "random")):
        for seed in specification["seeds"]:
            model = build_model(agent._model, "ego_query_residual", seed)
            model.patch_selector = SpatialRegionSelector(model.patch_selector, 16, 32, 2, 8).cuda()
            checkpoint_path = preserved_root / f"{condition}_seed{seed}/complete.pt"
            checkpoint = torch.load(checkpoint_path, map_location="cpu")
            parameters = {name: parameter for name, parameter in model.named_parameters() if not name.startswith("baseline_model.")}
            if set(parameters) != set(checkpoint["extension_parameters"]):
                raise RuntimeError("Preserved extension checkpoint key mismatch")
            with torch.no_grad():
                for name, parameter in parameters.items():
                    parameter.copy_(checkpoint["extension_parameters"][name])
            del checkpoint
            preserved_report = json.loads((preserved_root / f"{condition}_seed{seed}/results.json").read_text())
            if module_hashes(model) != preserved_report["final_module_hashes"]:
                raise RuntimeError("Preserved module hash mismatch")
            preserved_rows = {row["token"]: row for row in preserved_report["evaluations"]["800"]["development"]["windows"]}
            model.eval()
            rows = []
            with torch.no_grad():
                for start in range(0, len(records), 8):
                    indices = list(range(start, min(start + 8, len(records))))
                    if torch.cuda.mem_get_info()[0] < 6 * 2**30:
                        raise RuntimeError("Shared GPU reserve reached")
                    observed = {name: value[indices].cuda() for name, value in cache.items()}
                    set_selection_control(model, records, indices, {"selection_mode": selection_mode, "region_budget": 8}, seed)
                    outputs = observed_forward(model, observed)
                    ade = (outputs["trajectory"][..., :2] - observed["ego_trajectory_target"][..., :2]).norm(dim=-1).mean(-1)
                    for position, index_value in enumerate(indices):
                        token = records[index_value]["current_frame_token"]
                        if abs(float(ade[position]) - preserved_rows[token]["xy_ade_m"]) > 1e-6:
                            raise RuntimeError("Read-only reconstruction differs from preserved ADE")
                        rows.append({"token": token, "recording": records[index_value]["recording_group"],
                                     "trajectory": outputs["trajectory"][position].cpu().tolist()})
            predictions[f"{label}_seed{seed}"] = rows
            print(f"PRESERVED_TRAJECTORIES {label} seed={seed} windows={len(rows)}", flush=True)
            del model, parameters, outputs
    if parameter_sha256(agent._model) != original_hash:
        raise RuntimeError("Original planner weights changed")
    del agent, observed, cache
    gc.collect()
    torch.cuda.empty_cache()
    write_json(args.output_directory / "predictions.json", predictions)
    from audit_official_drive_jepa_evaluation import official_configuration
    from hydra.utils import instantiate
    from navsim.common.dataclasses import Trajectory
    from navsim.evaluate.pdm_score import pdm_score
    _, _, configuration, _ = official_configuration(WORKSPACE)
    simulator, scorer = instantiate(configuration.simulator), instantiate(configuration.scorer)
    manifest = json.loads((args.pdm_directory / "metric_cache_manifest.json").read_text())
    indexed = {name: {row["token"]: row for row in rows} for name, rows in predictions.items()}
    scores = {name: [] for name in predictions}
    for entry in manifest:
        if file_sha256(entry["metric_cache_file"]) != entry["cache_sha256"]:
            raise RuntimeError("Development metric cache changed")
        with lzma.open(entry["metric_cache_file"], "rb") as stream:
            metric_cache = pickle.load(stream)
        for name, rows in indexed.items():
            trajectory = Trajectory(np.asarray(rows[entry["token"]]["trajectory"], dtype=np.float32))
            metrics = asdict(pdm_score(metric_cache, trajectory, simulator.proposal_sampling, simulator, scorer))
            if not all(np.isfinite(value) for value in metrics.values()):
                raise RuntimeError("Non-finite preserved-control PDM result")
            scores[name].append({"token": entry["token"], "recording": entry["recording"], **metrics})
    summary = {name: {key: float(np.mean([row[key] for row in rows])) for key in rows[0]
                      if key not in ("token", "recording")} for name, rows in scores.items()}
    write_json(args.output_directory / "results.json", {"windows": scores, "summary": summary,
               "scope": "preserved global controls, same development192, no optimizer steps"})
    write_json(args.output_directory / "completion.json", {"complete": True, "baseline_hash": original_hash,
               "wall_seconds_including_wait": time.perf_counter() - started, "source": source})
    print("PRESERVED_CONTROL_PDM_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
