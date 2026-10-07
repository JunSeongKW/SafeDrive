"""Run the fixed common comparison after existing Adapter/restore work finishes."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from prepare_lpwm_shared_navtest_comparison import digest, read_json, write_json
from evaluate_lpwm_shared_navtest_comparison import check_resources, EvaluationDeferred, gpu_used_bytes, verify_registration

ROOT = Path(__file__).resolve().parents[1]
GPU_PYTHON = ROOT / "runtime/environments/kjs-lpwm-drivor-joint/bin/python"
CPU_PYTHON = ROOT / "runtime/environments/drive_jepa_official_evaluation/bin/python"


def process_alive(pid):
    try:
        return Path(f"/proc/{pid}/stat").read_text().split()[2] != "Z"
    except FileNotFoundError:
        return False


def yield_adapter(configuration, output, status):
    """Use the existing graceful checkpoint hook, never kill in-flight updates."""
    adapter = ROOT / configuration["adapter_training_root"]
    primary = ROOT / configuration["primary_training_root"]
    pause = adapter / "pause.requested"
    assert not pause.exists() and not (primary / "pause.requested").exists(), "Do not override a previous user pause"
    before = read_json(adapter / "queue_state.json")
    assert before["stage"] == "epoch03_training" and process_alive(before["queue_pid"])
    request = {"owner": str(output.relative_to(ROOT)), "purpose": "User-requested same-scene Adapter epoch2 versus current LoRA comparison",
               "created_unix": time.time(), "resume_after_gpu_inference": True}
    content = json.dumps(request, sort_keys=True) + "\n"
    write_json(output / "adapter_yield_request.json", {**request, "pause_content": content, "previous_queue_state": before})
    with pause.open("x") as stream:
        stream.write(content)
    status("requesting_adapter_checkpoint", previous_child_pid=before["child_pid"])
    deadline = time.monotonic() + 240
    while process_alive(before["queue_pid"]) or process_alive(before["child_pid"]):
        assert time.monotonic() < deadline, "Adapter graceful checkpoint timeout; preserve pause for inspection"
        assert pause.read_text() == content
        time.sleep(2)
    assert read_json(adapter / "queue_state.json")["stage"] == "paused"
    saved = read_json(adapter / "saved_state.json")
    assert saved["reason"] == "resource_or_user_pause"
    preserved = output / "adapter_resume_before_evaluation.pt"
    os.link(adapter / "latest.pt", preserved)
    assert digest(preserved) == saved["checkpoint_sha256"]
    write_json(output / "adapter_yield_ready.json", {"pause_content": content, "saved_state": saved,
               "preserved_resume": str(preserved), "no_partial_update_discarded": True,
               "previous_queue_pid": before["queue_pid"], "previous_child_pid": before["child_pid"]})
    status("adapter_checkpoint_preserved", completed_updates=saved["completed_updates"])


def resume_adapter(configuration, output):
    adapter = ROOT / configuration["adapter_training_root"]
    request_path = output / "adapter_yield_request.json"
    if not request_path.exists() or (output / "adapter_resumed.json").exists():
        return
    request = read_json(request_path)
    pause = adapter / "pause.requested"
    if not pause.exists() or pause.read_text() != request["pause_content"]:
        write_json(output / "adapter_resume_pending.json", {"reason": "Pause ownership changed; user control respected"})
        return
    if (ROOT / configuration["primary_training_root"] / "pause.requested").exists() or (output / "pause.requested").exists():
        write_json(output / "adapter_resume_pending.json", {"reason": "A newer user pause is active"})
        return
    assert read_json(adapter / "queue_state.json")["stage"] == "paused"
    pause.rename(output / "completed_adapter_yield.requested")
    command = [str(ROOT / "runtime/environments/kjs-lpwm-stage2/bin/python"), "-u",
        "scripts/queue_lpwm_adapter_original_batch_shared.py", "--config",
        "configs/lpwm_adapter_original_batch_rebalance/adapter_batch8_shared.json", "--detach"]
    subprocess.run(command, cwd=ROOT, check=True)
    launch = read_json(adapter / "launch.json")
    assert process_alive(launch["queue_pid"])
    write_json(output / "adapter_resumed.json", {"launch": launch, "unchanged_configuration": command[-2],
               "resumed_from": read_json(adapter / "saved_state.json"), "resumed_unix": time.time()})


def run(configuration_path, detach):
    configuration_path = configuration_path.resolve()
    configuration = read_json(configuration_path)
    output = ROOT / configuration["output_directory"]
    if detach:
        with (output / "queue.log").open("a") as stream:
            process = subprocess.Popen([str(CPU_PYTHON), "-u", str(Path(__file__).resolve()), "--config", str(configuration_path)],
                cwd=ROOT, stdin=subprocess.DEVNULL, stdout=stream, stderr=subprocess.STDOUT, start_new_session=True,
                env={**os.environ, "CUDA_VISIBLE_DEVICES": "", "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1"})
        launch = {"queue_pid": process.pid, "started_unix": time.time(), "training_modified": False}
        write_json(output / "launch.json", launch)
        print(json.dumps(launch), flush=True)
        return
    lock = (output / "queue.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)

    def status(stage, **values):
        write_json(output / "queue_state.json", {"stage": stage, "queue_pid": os.getpid(),
                                                  "updated_unix": time.time(), **values})

    adapter_yielded = False
    try:
        verify_registration(configuration_path, configuration)
        assert read_json(output / "inputs_complete.json")["complete"]
        assert read_json(output / "cpu_execution_check.json")["passed"]
        if (output / "evaluation_complete.json").exists():
            status("complete")
            return
        if configuration.get("temporary_adapter_yield"):
            yield_adapter(configuration, output, status)
            adapter_yielded = True
        for system in ("primary", "adapter"):
            while not (output / system / "inference_complete.json").exists():
                try:
                    check_resources(configuration)
                    if gpu_used_bytes(configuration["gpu"]) >= configuration["admission_card_bytes"]:
                        raise EvaluationDeferred("Waiting for card memory")
                except EvaluationDeferred as error:
                    status("waiting_for_existing_work", reason=str(error), pending_system=system)
                    time.sleep(30)
                    continue
                verify_registration(configuration_path, configuration)
                environment = {**os.environ, "CUDA_VISIBLE_DEVICES": str(configuration["gpu"]),
                    "PYTHONPATH": str(ROOT / "src"), "OMP_NUM_THREADS": "2", "OPENBLAS_NUM_THREADS": "1",
                    "MKL_NUM_THREADS": "1", "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"}
                with (output / f"{system}_inference.log").open("a") as stream:
                    process = subprocess.Popen([str(GPU_PYTHON), "-u", "scripts/evaluate_lpwm_shared_navtest_comparison.py",
                        "--config", str(configuration_path), "--mode", "infer", "--system", system],
                        cwd=ROOT, env=environment, stdout=stream, stderr=subprocess.STDOUT)
                    status("inference", system=system, child_pid=process.pid)
                    result = process.wait()
                if result == 75:
                    continue
                assert result == 0, f"{system} inference failed; see its preserved log"
        # Scoring is CPU-only. Release the temporary pause as soon as GPU
        # predictions finish, using the unchanged original Adapter queue.
        if adapter_yielded:
            resume_adapter(configuration, output)
            adapter_yielded = False
        environment = {**os.environ, "CUDA_VISIBLE_DEVICES": "", "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1", "LD_LIBRARY_PATH": str(CPU_PYTHON.parent.parent / "lib")}
        with (output / "official_scoring.log").open("a") as stream:
            process = subprocess.Popen([str(CPU_PYTHON), "-u", "scripts/evaluate_lpwm_shared_navtest_comparison.py",
                "--config", str(configuration_path), "--mode", "score"], cwd=ROOT, env=environment, stdout=stream, stderr=subprocess.STDOUT)
            status("official_scoring", child_pid=process.pid)
            assert process.wait() == 0, "Official scorer failed; see its preserved log"
        assert read_json(output / "evaluation_complete.json")["complete"]
        status("complete")
    except Exception as error:
        status("failed", error=repr(error))
        raise
    finally:
        if adapter_yielded and (output / "adapter_yield_ready.json").exists():
            resume_adapter(configuration, output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--detach", action="store_true")
    arguments = parser.parse_args()
    run(arguments.config, arguments.detach)
