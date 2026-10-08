"""Pause/save JEPA, measure serial repeatability, and resume the study queue.

All test weights are disposable. No scientific trainer, batch, optimizer,
dataset, objective or seed is changed. A failed audit retains serial execution.
"""
from datetime import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import time

import numpy as np

import queue_four_model_small_corpus_overlap_launch_fix as scheduler_base

ROOT = scheduler_base.ROOT
OUTPUT = scheduler_base.OUTPUT
DIRECTORY = OUTPUT / "scheduling_v3_calibrated"
ORIGINAL_PROFILES = OUTPUT / "scheduling_v2/profiles"
PYTHON = scheduler_base.original_queue.GPU_PYTHON


def load_rows(directory, rank):
    return [json.loads(line) for line in (directory / f"training_rank{rank}.jsonl").read_text().splitlines()]


def relative_error(left, right):
    return abs(left - right) / max(abs(left), abs(right), 1e-12)


def maximum_training_difference(first, second, field):
    differences = []
    for rank in (0, 1):
        left, right = load_rows(first, rank), load_rows(second, rank)
        assert len(left) == len(right) == 8
        assert [row["completed_updates"] for row in left] == [row["completed_updates"] for row in right]
        assert all(math.isfinite(row[field]) for row in left + right)
        differences.extend(relative_error(a[field], b[field]) for a, b in zip(left, right))
    return max(differences)


def planner_prediction_difference(first, second):
    with np.load(first / "validation/profile_after.npz") as left, np.load(second / "validation/profile_after.npz") as right:
        assert np.array_equal(left["tokens"], right["tokens"])
        differences = left["trajectories"] - right["trajectories"]
    displacement = np.linalg.norm(differences[..., :2], axis=-1)
    heading = np.abs(np.arctan2(np.sin(differences[..., 2]), np.cos(differences[..., 2])))
    return dict(mean_displacement_m=float(displacement.mean()), maximum_displacement_m=float(displacement.max()),
                maximum_heading_rad=float(heading.max()))


def assess(protocol):
    prior = json.loads((OUTPUT / "scheduling_v2/jepa_ssl_and_drivor.admission.json").read_text())
    checks = {name: prior["checks"][name] for name in ("faster", "elapsed_training_faster", "actual_overlap", "memory_safe")}
    details = {}
    for kind in ("jepa_ssl", "drivor"):
        original = ORIGINAL_PROFILES / ("jepa_ssl_and_drivor_serial_" + kind)
        parallel = ORIGINAL_PROFILES / ("jepa_ssl_and_drivor_parallel1_" + kind)
        repeated = DIRECTORY / "profiles" / ("serial_repeat_" + kind)
        registration_original = json.loads((original / "registration.json").read_text())
        registration_repeated = json.loads((repeated / "registration.json").read_text())
        assert registration_original == registration_repeated
        values = dict(serial_repeat_loss_relative=maximum_training_difference(original, repeated, "loss"),
                      cross_mode_loss_relative=maximum_training_difference(original, parallel, "loss"),
                      serial_repeat_gradient_relative=maximum_training_difference(original, repeated, "gradient_norm"),
                      cross_mode_gradient_relative=maximum_training_difference(original, parallel, "gradient_norm"))
        tolerance = max(protocol["minimum_repeatability_envelope"],
                        protocol["repeatability_envelope_multiplier"] * values["serial_repeat_loss_relative"])
        checks[kind + "_loss_bounded"] = max(values["serial_repeat_loss_relative"], values["cross_mode_loss_relative"]) <= protocol["loss_relative_cap"]
        checks[kind + "_within_repeatability_envelope"] = values["cross_mode_loss_relative"] <= tolerance
        checks[kind + "_gradient_bounded"] = max(values["serial_repeat_gradient_relative"], values["cross_mode_gradient_relative"]) <= protocol["gradient_relative_cap"]
        if kind == "jepa_ssl":
            validations = {name: json.loads((path / "profile_after.json").read_text())
                           for name, path in (("original", original), ("parallel", parallel), ("repeated", repeated))}
            values["validation"] = validations
            for metric, cap in (("masked_latent_l1", protocol["validation_loss_relative_cap"]),
                                ("spatial_feature_std", protocol["feature_std_relative_cap"])):
                checks[metric + "_bounded"] = all(relative_error(validations["original"][metric], validations[name][metric]) <= cap
                                                     for name in ("parallel", "repeated"))
        else:
            values["prediction_differences"] = {name: planner_prediction_difference(original, path)
                                                for name, path in (("parallel", parallel), ("repeated", repeated))}
            checks["drivor_predictions_bounded"] = all(
                value["mean_displacement_m"] <= protocol["trajectory_mean_displacement_cap_m"]
                and value["maximum_displacement_m"] <= protocol["trajectory_maximum_displacement_cap_m"]
                and value["maximum_heading_rad"] <= protocol["trajectory_maximum_heading_cap_rad"]
                for value in values["prediction_differences"].values())
        details[kind] = values
    return dict(parallel_approved=all(checks.values()), checks=checks, repeatability=details,
                previous_admission=prior, protocol=protocol,
                scientific_protocol_changed=False, previous_failed_gate_preserved=True,
                conclusion_scope="Eight-update execution repeatability and three fixed validation clips; not proof of equal final PDMS")


def run_profile(kind):
    directory = DIRECTORY / "profiles" / ("serial_repeat_" + kind)
    directory.mkdir(parents=True, exist_ok=True)
    assert not (directory / "complete.json").exists()
    helper = scheduler_base.StudyScheduler({})
    command = helper.command(kind, directory, None, 8)
    environment = os.environ.copy()
    environment.update(CUDA_VISIBLE_DEVICES="0,1", OMP_NUM_THREADS="2", OPENBLAS_NUM_THREADS="1",
                       MKL_NUM_THREADS="1", NCCL_P2P_DISABLE="1", PYTHONUNBUFFERED="1")
    with (DIRECTORY / ("serial_repeat_" + kind + ".log")).open("a") as log:
        child = subprocess.Popen([str(value) for value in command], cwd=ROOT, env=environment,
                                 stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        scheduler_base.atomic_json(DIRECTORY / "audit_state.json", dict(phase="serial_repeat_" + kind,
                                     pid=child.pid, command=[str(value) for value in command], started_unix=time.time()))
        try:
            result = child.wait(timeout=600)
        except subprocess.TimeoutExpired:
            # This process group was created above solely for this disposable profile.
            os.killpg(child.pid, signal.SIGTERM)
            try:
                child.wait(timeout=60)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid, signal.SIGKILL)
                child.wait(timeout=30)
            raise
    assert result == 0 and (directory / "complete.json").exists(), (kind, result)


def resume_study():
    if (OUTPUT / "pause.requested").exists():
        raise RuntimeError("User study pause exists; do not restart")
    marker = OUTPUT / "jepa_ssl/pause.requested"
    if marker.exists():
        value = json.loads(marker.read_text())
        assert value.get("owner") == "four_model_overlap_v2" and value.get("scheduler_pid") == 1084697
        scheduler_base.atomic_json(DIRECTORY / "archived_jepa_operational_pause.json", value)
        marker.unlink()
    command = [str(PYTHON), "-u", str(ROOT / "scripts/queue_small_corpus_calibrated_parallel.py")]
    with (DIRECTORY / "console.log").open("a") as log:
        child = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                                 stdin=subprocess.DEVNULL, start_new_session=True)
    scheduler_base.atomic_json(DIRECTORY / "launch.json", dict(pid=child.pid, command=command, started_unix=time.time()))
    print(json.dumps(dict(resumed_scheduler_pid=child.pid)), flush=True)


def main():
    DIRECTORY.mkdir(parents=True, exist_ok=True)
    protocol = dict(loss_relative_cap=.001, gradient_relative_cap=.10,
                    minimum_repeatability_envelope=.0001, repeatability_envelope_multiplier=2.,
                    validation_loss_relative_cap=.005, feature_std_relative_cap=.02,
                    trajectory_mean_displacement_cap_m=.05, trajectory_maximum_displacement_cap_m=.2,
                    trajectory_maximum_heading_cap_rad=.02, updates=8,
                    rationale="Calibrate same-configuration numerical repeatability instead of requiring bitwise agreement across independent nondeterministic runs")
    protocol_path = DIRECTORY / "audit_protocol.json"
    assert not protocol_path.exists(), "This transition is a one-time operation"
    scheduler_base.atomic_json(protocol_path, protocol)
    scheduler_base.verify_original_sources()
    state = json.loads((OUTPUT / "queue_state.json").read_text())
    assert state["scheduler_pid"] == 1084697 and state["pid"] == 1099274 and state["status"] == "running"
    controller = scheduler_base.process_identity(1084697)
    training = scheduler_base.process_identity(1099274)
    assert controller and controller["uid"] == os.getuid() and "queue_four_model_small_corpus_overlap_launch_fix.py" in controller["command"]
    assert training and training["uid"] == os.getuid() and "train_small_corpus_jepa_ssl.py" in training["command"]
    scheduler_base.atomic_json(DIRECTORY / "transition_before.json", dict(controller=controller, training=training,
                                 queue=state, progress=json.loads((OUTPUT / "jepa_ssl/progress.json").read_text())))
    os.kill(controller["pid"], signal.SIGTERM)
    deadline = time.monotonic() + 240
    while scheduler_base.still_alive(controller) or scheduler_base.still_alive(training):
        assert time.monotonic() < deadline, "Existing scheduler did not finish saving"
        time.sleep(2)
    paused = json.loads((OUTPUT / "jepa_ssl/paused.json").read_text())
    assert (OUTPUT / "jepa_ssl/latest.pt").exists()
    scheduler_base.atomic_json(DIRECTORY / "saved_transition.json", dict(paused=paused,
                                 checkpoint_bytes=(OUTPUT / "jepa_ssl/latest.pt").stat().st_size,
                                 no_original_scientific_source_changes=True))
    print(json.dumps(dict(saved_jepa_updates=paused["completed_updates"])), flush=True)
    try:
        for kind in ("jepa_ssl", "drivor"):
            run_profile(kind)
        admission = assess(protocol)
    except Exception as error:
        admission = dict(parallel_approved=False, reason=repr(error), preserve_serial_on_audit_failure=True)
        print(json.dumps(admission), flush=True)
    scheduler_base.atomic_json(DIRECTORY / "repeatability_admission.json", admission)
    scheduler_base.verify_original_sources()
    resume_study()
    scheduler_base.atomic_json(DIRECTORY / "audit_complete.json", dict(parallel_approved=admission["parallel_approved"],
                                 time=datetime.now().astimezone().isoformat()))
    print(json.dumps(admission), flush=True)


if __name__ == "__main__":
    main()
