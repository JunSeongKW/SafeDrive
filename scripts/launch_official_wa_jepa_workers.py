"""Launch only registered, disjoint official WA-JEPA workers in dedicated tmux."""
import argparse
import json
import os
import shlex
import subprocess
from pathlib import Path


def prepare_owner_labeled_python_entrypoint(workspace, environment, relative_alias):
    """Label future launches without moving or modifying the live Conda prefix."""
    # Normalize "../envs" without resolving the alias back to its Conda target.
    environment_alias = Path(os.path.abspath(workspace / relative_alias))
    if not environment_alias.is_symlink() and not environment_alias.exists():
        environment_alias.parent.mkdir(parents=True, exist_ok=True)
        environment_alias.symlink_to(environment.resolve(), target_is_directory=True)
    if environment_alias.resolve() != environment.resolve():
        raise RuntimeError(f"Process-label path points to a different environment: {environment_alias}")
    python_entrypoint = environment_alias / "bin/python"
    if not python_entrypoint.is_file():
        raise RuntimeError(f"Missing environment Python entrypoint: {python_entrypoint}")
    return python_entrypoint


def acknowledge_user_pause_for_explicit_resume(pause_marker, explicit_user_resume):
    if not pause_marker.exists():
        return
    if not explicit_user_resume:
        raise RuntimeError("Evaluation is explicitly user-paused. Do not launch until user requests resume; then use --resume-user-paused after checking GPU allocation.")
    pause_metadata = json.loads(pause_marker.read_text())
    acknowledged_marker = pause_marker.with_name(f"evaluation_pause_acknowledged_{pause_metadata['requested_at_utc'].replace(':', '').replace('+', '_')}.json")
    if acknowledged_marker.exists():
        raise RuntimeError("Pause acknowledgement already exists; preserve audit before resume")
    pause_marker.rename(acknowledged_marker)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, choices=[0, 1], required=True)
    parser.add_argument("--execution-shard", type=int, help="Launch only this existing partition; never change the partition count")
    parser.add_argument("--resume-user-paused", action="store_true", help="Only after explicit user resume and renewed GPU availability check")
    arguments = parser.parse_args()
    workspace = Path(__file__).resolve().parents[1]
    pause_marker = workspace / "outputs/official_wa_jepa_reproduction/evaluation_pause.json"
    specification = json.loads((workspace / "configs/official_wa_jepa/reproduction_v1.json").read_text())
    if arguments.execution_shard is not None:
        assignments = specification["parallel_evaluation"]["physical_gpu_by_shard"]
        if not 0 <= arguments.execution_shard < len(assignments) or assignments[arguments.execution_shard] != arguments.gpu:
            raise ValueError("Requested shard does not belong to the requested physical GPU")
    acknowledge_user_pause_for_explicit_resume(pause_marker, arguments.resume_user_paused)
    environment = workspace / specification["conda_environment"]
    python_entrypoint = prepare_owner_labeled_python_entrypoint(workspace, environment, specification["process_environment_alias"])
    socket_name = "planning-aware-wa-jepa"
    manifest_path = workspace / "outputs/official_wa_jepa_reproduction/parallel_worker_manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {"registered_workers": [], "execution_shards": specification["parallel_evaluation"]["execution_shards"]}
    if manifest["execution_shards"] != specification["parallel_evaluation"]["execution_shards"]:
        raise RuntimeError("Existing live worker manifest uses a different partition")
    for shard_index, physical_gpu in enumerate(specification["parallel_evaluation"]["physical_gpu_by_shard"]):
        if physical_gpu != arguments.gpu:
            continue
        if arguments.execution_shard is not None and shard_index != arguments.execution_shard:
            continue
        session_name = f"official_wa_jepa_gpu{physical_gpu}_worker{shard_index}"
        if subprocess.run(["tmux", "-L", socket_name, "has-session", "-t", session_name], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False).returncode == 0:
            print(f"Already active: {session_name}")
            continue
        log_path = workspace / f"outputs/official_wa_jepa_reproduction/dense_full_workers14_shard{shard_index}.log"
        command = ["env", f"CUDA_VISIBLE_DEVICES={physical_gpu}", "OMP_NUM_THREADS=1", "OPENBLAS_NUM_THREADS=1", f"LD_LIBRARY_PATH={environment / 'lib'}", str(python_entrypoint), "-u", str(workspace / "scripts/evaluate_official_wa_jepa.py"), "full", "--execution-shard", str(shard_index)]
        shell_command = f"cd {shlex.quote(str(workspace))} && exec {shlex.join(command)} >> {shlex.quote(str(log_path))} 2>&1"
        subprocess.run(["tmux", "-L", socket_name, "new-session", "-d", "-s", session_name, shell_command], check=True)
        pane_pid = int(subprocess.check_output(["tmux", "-L", socket_name, "display-message", "-p", "-t", session_name, "#{pane_pid}"], text=True).strip())
        manifest["registered_workers"].append({"shard": shard_index, "physical_gpu": physical_gpu, "session": session_name, "pane_pid": pane_pid, "log": str(log_path), "python_entrypoint": str(python_entrypoint), "process_owner_initials": specification["process_owner_initials"]})
        staged_manifest = manifest_path.with_name(f"{manifest_path.name}.{os.getpid()}.pending")
        staged_manifest.write_text(json.dumps(manifest, indent=2) + "\n")
        staged_manifest.replace(manifest_path)
        print(f"Started {session_name}, pid={pane_pid}", flush=True)


if __name__ == "__main__":
    main()
