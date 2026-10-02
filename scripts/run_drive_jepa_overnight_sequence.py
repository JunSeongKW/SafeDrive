"""Durable, serial GPU1 experiment queue with a fixed morning deadline.

No training choices depend on scores. Stops on any failure or memory guard.
Only children created by this queue can receive its termination signal.
"""

import argparse
import json
import os
import signal
import subprocess
import time
from datetime import datetime
from pathlib import Path

WORKSPACE = Path(__file__).resolve().parents[1]
PYTHON = Path("/rhome/junseong/envs/kjs-drive-jepa-extension/bin/python")
ROOT = WORKSPACE / "outputs/drive_jepa_selective_future"
DEADLINE = datetime.fromisoformat("2026-10-03T09:00:00+09:00")
STOP_REQUESTED = False


def request_stop(signum, frame):
    global STOP_REQUESTED
    STOP_REQUESTED = True


def save(path, payload):
    temporary = path.with_suffix(".partial")
    temporary.write_text(json.dumps(payload, indent=2) + "\n")
    temporary.replace(path)


def must_stop(directory):
    return (
        STOP_REQUESTED
        or (directory / "pause.json").exists()
        or (ROOT / "overnight_sequence_20261003/pause.json").exists()
        or datetime.now().astimezone() >= DEADLINE
    )


def run_stage(directory, name, script, arguments, gpu=True):
    if must_stop(directory):
        raise RuntimeError("Deadline or explicit stop before next stage")
    environment = dict(os.environ)
    environment.update(
        {
            "CUDA_VISIBLE_DEVICES": "1" if gpu else "",
            "LD_LIBRARY_PATH": str(
                WORKSPACE / "runtime/environments/drive_jepa_official_evaluation/lib"
            ),
            "OMP_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "PYTHONPATH": str(WORKSPACE / "src"),
        }
    )
    command = [str(PYTHON), "-u", str(WORKSPACE / "scripts" / script), *arguments]
    started = time.monotonic()
    with (directory / f"{name}.log").open("x") as log:
        child = subprocess.Popen(
            command,
            cwd=WORKSPACE,
            env=environment,
            stdout=log,
            stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,
        )
        save(
            directory / "active_stage.json",
            {
                "stage": name,
                "pid": child.pid,
                "command": command,
                "started_at": datetime.now().astimezone().isoformat(),
            },
        )
        stop_sent = False
        while child.poll() is None:
            if must_stop(directory) and not stop_sent:
                # SIGTERM handler in the training runner saves optimizer/RNG.
                child.send_signal(signal.SIGTERM)
                stop_sent = True
            time.sleep(10)
    result = {
        "stage": name,
        "pid": child.pid,
        "returncode": child.returncode,
        "wall_seconds": time.monotonic() - started,
        "stop_sent": stop_sent,
    }
    save(directory / f"{name}_completion.json", result)
    if child.returncode != 0 or stop_sent:
        raise RuntimeError(f"Stage stopped: {name}, returncode={child.returncode}")
    return result


def run_projection_series(directory):
    stages = []
    predecessor = ROOT / "overnight_sequence_20261003"
    try:
        while not (predecessor / "completion.json").exists():
            if must_stop(directory) or (predecessor / "stopped.json").exists():
                raise RuntimeError("Deadline/pause/predecessor stopped; no GPU launch")
            save(
                directory / "active_stage.json",
                {
                    "stage": "waiting_for_coverage_sequence",
                    "checked_at": datetime.now().astimezone().isoformat(),
                },
            )
            time.sleep(10)
        output = ROOT / "overnight_projection_transfer_v1_20261003"
        stages.append(
            run_stage(
                directory,
                "projection_transfer_training",
                "run_drive_jepa_causal_followup.py",
                [
                    "--config",
                    str(
                        WORKSPACE
                        / "configs/drive_jepa_selective_future/overnight_projection_transfer_v1.json"
                    ),
                    "--output-directory",
                    str(output),
                ],
            )
        )
        stages.append(
            run_stage(
                directory,
                "export_projection_results",
                "summarize_drive_jepa_causal_followup.py",
                [
                    "--run-directory",
                    str(output),
                    "--share-directory",
                    str(
                        WORKSPACE
                        / "results/drive_jepa_selective_future/overnight_projection_transfer_v1_20261003"
                    ),
                ],
                gpu=False,
            )
        )
        conservative_output = ROOT / "overnight_conservative_adaptation_v1_20261003"
        stages.append(
            run_stage(
                directory,
                "conservative_adaptation_training",
                "run_drive_jepa_causal_followup.py",
                [
                    "--config",
                    str(
                        WORKSPACE
                        / "configs/drive_jepa_selective_future/overnight_conservative_adaptation_v1.json"
                    ),
                    "--output-directory",
                    str(conservative_output),
                ],
            )
        )
        stages.append(
            run_stage(
                directory,
                "export_conservative_results",
                "summarize_drive_jepa_causal_followup.py",
                [
                    "--run-directory",
                    str(conservative_output),
                    "--share-directory",
                    str(
                        WORKSPACE
                        / "results/drive_jepa_selective_future/overnight_conservative_adaptation_v1_20261003"
                    ),
                ],
                gpu=False,
            )
        )
        sharing = commit_completed_results(directory)
        save(
            directory / "completion.json",
            {
                "complete": True,
                "stages": stages,
                "sharing": sharing,
                "ended_at": datetime.now().astimezone().isoformat(),
            },
        )
        print("PROJECTION_TRANSFER_SEQUENCE_COMPLETE", flush=True)
    except BaseException as error:
        save(
            directory / "stopped.json",
            {
                "reason": repr(error),
                "completed_stages": stages,
                "automatic_retry": False,
            },
        )
        raise


def commit_completed_results(directory):
    """Commit only known generated evidence, never somebody else's staged work."""
    branch = subprocess.check_output(
        ["git", "branch", "--show-current"], cwd=WORKSPACE, text=True
    ).strip()
    if branch != "junseong/main":
        return {"committed": False, "reason": "Unexpected branch; preserve all results"}
    paths = []
    for run_name in (
        "overnight_coverage_training_v1_20261003",
        "overnight_projection_transfer_v1_20261003",
        "overnight_conservative_adaptation_v1_20261003",
    ):
        for filename in (
            "summary.json",
            "source.json",
            "gradient_and_dependence.json",
            "window_results.csv",
        ):
            path = Path("results/drive_jepa_selective_future") / run_name / filename
            if not (WORKSPACE / path).is_file():
                raise RuntimeError("Missing generated evidence before scoped commit")
            paths.append(str(path))
    subprocess.run(["git", "add", "--", *paths], cwd=WORKSPACE, check=True)
    subprocess.run(
        [
            "git",
            "commit",
            "--only",
            "-m",
            "[exp] preserve completed overnight coverage projection and adaptation results",
            "--",
            *paths,
        ],
        cwd=WORKSPACE,
        check=True,
    )
    commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=WORKSPACE, text=True
    ).strip()
    environment = {**os.environ, "GIT_TERMINAL_PROMPT": "0"}
    try:
        pushed = subprocess.run(
            ["git", "push", "mine", "junseong/main"],
            cwd=WORKSPACE,
            env=environment,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=45,
            check=False,
        )
        (directory / "push.log").write_text(pushed.stdout)
        return {
            "committed": True,
            "commit": commit,
            "pushed": pushed.returncode == 0,
            "push_returncode": pushed.returncode,
        }
    except subprocess.TimeoutExpired:
        return {
            "committed": True,
            "commit": commit,
            "pushed": False,
            "reason": "Push timed out; local results preserved",
        }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--detach", action="store_true")
    parser.add_argument(
        "--series", choices=["coverage", "projection"], default="coverage"
    )
    args = parser.parse_args()
    directory = ROOT / (
        "overnight_sequence_20261003"
        if args.series == "coverage"
        else "overnight_projection_sequence_20261003"
    )
    directory.mkdir(exist_ok=True)
    if args.detach:
        import fcntl

        # Lock inherited by the child prevents accidental duplicate queues.
        lock = (directory / "queue.lock").open("w")
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with (directory / "supervisor.log").open("a") as log:
            child = subprocess.Popen(
                [
                    str(PYTHON),
                    "-u",
                    str(Path(__file__).resolve()),
                    "--series",
                    args.series,
                ],
                cwd=WORKSPACE,
                stdout=log,
                stderr=subprocess.STDOUT,
                stdin=subprocess.DEVNULL,
                start_new_session=True,
                pass_fds=(lock.fileno(),),
            )
        save(
            directory / "supervisor.json",
            {
                "pid": child.pid,
                "deadline": DEADLINE.isoformat(),
                "source_commit": subprocess.check_output(
                    ["git", "rev-parse", "HEAD"], cwd=WORKSPACE, text=True
                ).strip(),
            },
        )
        print(f"DURABLE_OVERNIGHT_QUEUE_PID={child.pid}", flush=True)
        return
    for signum in (signal.SIGINT, signal.SIGTERM):
        signal.signal(signum, request_stop)
    if args.series == "projection":
        return run_projection_series(directory)
    first_run = ROOT / "overnight_causal_followup_v1_20261003"
    stages = []
    try:
        while not (first_run / "completion.json").exists():
            if must_stop(directory):
                raise RuntimeError("Deadline or pause while waiting for existing run")
            if list(first_run.glob("*/stopped.json")):
                raise RuntimeError(
                    "Existing run stopped; preserve and inspect, do not duplicate it"
                )
            save(
                directory / "active_stage.json",
                {
                    "stage": "waiting_for_existing_causal_comparison",
                    "checked_at": datetime.now().astimezone().isoformat(),
                },
            )
            time.sleep(10)
        stages.append(
            run_stage(
                directory,
                "export_causal_results",
                "summarize_drive_jepa_causal_followup.py",
                [
                    "--run-directory",
                    str(first_run),
                    "--share-directory",
                    str(
                        WORKSPACE
                        / "results/drive_jepa_selective_future/overnight_causal_followup_v1_20261003"
                    ),
                ],
                gpu=False,
            )
        )
        stages.append(
            run_stage(
                directory,
                "recording_coverage_cache",
                "prepare_drive_jepa_recording_coverage.py",
                [
                    "--config",
                    str(
                        WORKSPACE
                        / "configs/drive_jepa_selective_future/overnight_recording_coverage_v1.json"
                    ),
                    "--output-directory",
                    str(ROOT / "overnight_recording_coverage_v1_20261003"),
                    "--build",
                ],
            )
        )
        training_output = ROOT / "overnight_coverage_training_v1_20261003"
        stages.append(
            run_stage(
                directory,
                "coverage_and_selection_training",
                "run_drive_jepa_causal_followup.py",
                [
                    "--config",
                    str(
                        WORKSPACE
                        / "configs/drive_jepa_selective_future/overnight_coverage_training_v1.json"
                    ),
                    "--output-directory",
                    str(training_output),
                ],
            )
        )
        stages.append(
            run_stage(
                directory,
                "export_coverage_results",
                "summarize_drive_jepa_causal_followup.py",
                [
                    "--run-directory",
                    str(training_output),
                    "--share-directory",
                    str(
                        WORKSPACE
                        / "results/drive_jepa_selective_future/overnight_coverage_training_v1_20261003"
                    ),
                ],
                gpu=False,
            )
        )
        save(
            directory / "completion.json",
            {
                "complete": True,
                "stages": stages,
                "ended_at": datetime.now().astimezone().isoformat(),
                "further_unregistered_training": False,
            },
        )
        print("REGISTERED_OVERNIGHT_SEQUENCE_COMPLETE", flush=True)
    except BaseException as error:
        save(
            directory / "stopped.json",
            {
                "reason": repr(error),
                "completed_stages": stages,
                "ended_at": datetime.now().astimezone().isoformat(),
                "automatic_retry": False,
            },
        )
        raise


if __name__ == "__main__":
    main()
