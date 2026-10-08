"""Measure LPWM physical-batch growth now using the observed overlap peak."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import threading
import time

import queue_small_corpus_batch_growth as growth

base = growth.base
DIRECTORY = base.OUTPUT / "scheduling_v11_measured_batch_growth"


def measured_other_budget(card_peak, target_allocator_peak, companion_allocator_peaks):
    """Subtract only target allocated bytes, leaving CUDA/context overhead."""
    return max(card_peak - target_allocator_peak,
               2_000_000_000 + sum(companion_allocator_peaks.values()))


class MeasuredBatchGrowthScheduler(growth.BatchGrowthScheduler):
    def attempt_growth(self, job):
        kind = job.label.removeprefix("train_")
        companions = [other.label.removeprefix("train_") for other in self.gpu_jobs if other is not job]
        target = json.loads((job.directory / "progress.json").read_text())
        companion_peaks = {name: json.loads((base.OUTPUT / name / "progress.json").read_text())["peak_allocated_bytes"]
                           for name in companions}
        # The inherited monitor peak was measured with exactly the three current
        # batches. After a changed batch/peer completion, reservations revert to
        # v10's conservative policy instead of reusing an obsolete subtraction.
        initial_overlap = (self.current_microbatch[kind] == 2
                           and set(companions + [kind]) == {"jepa", "lpwm_sequential", "lpwm_joint"}
                           and all(self.current_microbatch[name] == 2 for name in companions))
        if not initial_overlap:
            return super().attempt_growth(job)
        budget = measured_other_budget(max(self.memory_peak_bytes), target["peak_allocated_bytes"], companion_peaks)
        old_reservations = growth.recovery.RESERVATIONS
        # v10 adds2GB to the sum of reservations. Allocate the observed budget's
        # remainder proportionally; the scalar sum is what gates the profile.
        updated = dict(old_reservations)
        total_allocated = sum(companion_peaks.values())
        remaining = max(0, budget - 2_000_000_000)
        for index, name in enumerate(companions):
            allocation = (remaining if index == len(companions) - 1 else
                          int((budget - 2_000_000_000) * companion_peaks[name] / total_allocated))
            updated[name] = allocation
            remaining -= allocation
        base.atomic_json(DIRECTORY / f"measured_admission_{kind}.json", dict(
            whole_card_peak_bytes=max(self.memory_peak_bytes), target_allocator_peak_bytes=target["peak_allocated_bytes"],
            companion_allocator_peak_bytes=companion_peaks, protected_other_bytes=budget,
            user_card_limit_bytes=growth.CARD_LIMIT, time=time.time()))
        growth.recovery.RESERVATIONS = updated
        try:
            return super().attempt_growth(job)
        finally:
            growth.recovery.RESERVATIONS = old_reservations

    def publish(self, status="running"):
        super().publish(status)
        value = json.loads((DIRECTORY / "state.json").read_text())
        value.update(scheduler="measured_batch_growth_v11")
        value["batch_growth"].update(trigger="User requested immediate measured headroom trial, then peer completions",
                                     admission_peak_basis="Measured full-card overlap peak plus allocator peaks")
        base.atomic_json(DIRECTORY / "state.json", value)
        base.atomic_json(base.OUTPUT / "queue_state.json", value)


def main(arguments):
    DIRECTORY.mkdir(parents=True, exist_ok=True)
    growth.DIRECTORY = DIRECTORY
    growth.previous.DIRECTORY = growth.guarded.DIRECTORY = growth.guarded.previous.DIRECTORY = DIRECTORY
    base.SCHEDULING = growth.recovery.DIRECTORY = DIRECTORY
    growth.guarded.JEPA_SAVE_PAUSE_BYTES = growth.guarded.TRIAL_ADMISSION_BYTES = growth.CARD_LIMIT
    growth.recovery.can_admit = growth.guarded.guarded_admission
    growth.guarded.previous.priority_defers = lambda *_arguments: False
    original = base.verify_original_sources()
    registration = json.loads((base.OUTPUT / "scheduling_v10_batch_growth/registration.json").read_text())
    for name, expected in registration["source_sha256"].items():
        assert hashlib.sha256((base.ROOT / name).read_bytes()).hexdigest() == expected
    own_lock = (DIRECTORY / "controller.lock").open("w")
    fcntl.flock(own_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    assert not (base.OUTPUT / "pause.requested").exists()
    state = json.loads((base.OUTPUT / "queue_state.json").read_text())
    assert state["scheduler"] == "batch_growth_v10" and state["scheduler_pid"] == arguments.take_over_pid
    assert state["status"] == "running"
    controller = base.process_identity(arguments.take_over_pid)
    assert controller and controller["uid"] == os.getuid()
    assert "queue_small_corpus_batch_growth.py" in controller["command"]
    scheduler = MeasuredBatchGrowthScheduler(registration["configuration"].copy())
    scheduler.memory_peak_bytes = state["monitored_peak_card_bytes"]
    scheduler.seen_complete = {name for name in base.PLANNING_KINDS if (base.OUTPUT / name / "complete.json").exists()}
    scheduler.retry_below_count = state.get("jepa_retry_below_active_lpwm_count")
    assert not state["batch_growth"]["trials"]
    adopted = []
    for entry in state["active_jobs"]:
        identity = base.process_identity(entry["pid"])
        assert identity and identity["uid"] == os.getuid() and base.still_alive(identity)
        directory = Path(entry["directory"])
        assert directory.parent == base.OUTPUT and not (directory / "pause.requested").exists()
        if entry["cpu"]:
            assert "score_four_model_small_corpus.py" in identity["command"]
            prediction = Path(entry["command"][entry["command"].index("--predictions") + 1])
            expected = prediction.with_suffix(".pdms.json")
        else:
            assert "resume_small_corpus_user_memory_limit.py" in identity["command"]
            expected = directory / "complete.json"
        job = base.Job(entry["label"], directory, entry["command"], expected, identity, cpu=entry["cpu"])
        adopted.append(identity)
        if entry["cpu"]:
            scheduler.cpu_job = job
            scheduler.evaluation_started[job.label] = time.time()
        else:
            scheduler.gpu_jobs.append(job)
    assert len(scheduler.gpu_jobs) == 3
    sources = dict(registration["source_sha256"])
    sources[str(Path(__file__).relative_to(base.ROOT))] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    base.atomic_json(DIRECTORY / "registration.json", dict(configuration=scheduler.configuration,
        original_registration=original, previous_registration=registration, source_sha256=sources,
        user_request="Check larger batches now while about20GB VRAM is free perGPU",
        allocator_budget_basis="Measured worst whole-card overlap, minus target allocated memory",
        effective_batch=16, card_limit_bytes=growth.CARD_LIMIT))
    base.atomic_json(DIRECTORY / "handover_before.json", dict(controller=controller, queue=state, adopted=adopted))
    os.kill(controller["pid"], signal.SIGKILL)
    deadline = time.monotonic() + 15
    while base.still_alive(controller):
        assert time.monotonic() < deadline
        time.sleep(.1)
    study_lock = (base.OUTPUT / "queue.lock").open("w")
    fcntl.flock(study_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    assert all(job.running() for job in scheduler.gpu_jobs)
    base.atomic_json(DIRECTORY / "handover_verified.json", dict(controller_pid=os.getpid(),
        training_restart=False, adopted=adopted, immediate_profile_requested=True))
    signal.signal(signal.SIGTERM, scheduler.request_stop)
    signal.signal(signal.SIGINT, scheduler.request_stop)
    trial_monitor = threading.Thread(target=scheduler.monitor_memory, daemon=True)
    trial_monitor.start()
    try:
        # Only the sequential candidate has enough predicted worst-case room
        # during three-job overlap. Preserve other jobs through this trial.
        candidate = next(job for job in scheduler.gpu_jobs if job.label == "train_lpwm_sequential")
        scheduler.phase = "wait_safe_boundary_for_immediate_batch_trial"
        while not scheduler.safe_resize_point(candidate):
            scheduler.check_user_pause()
            scheduler.service_cpu_scoring()
            scheduler.publish("waiting_existing_epoch_validation")
            time.sleep(2)
        scheduler.attempt_growth(candidate)
        scheduler.monitor_stop.set()
        trial_monitor.join(timeout=5)
        assert not trial_monitor.is_alive(), "Trial telemetry did not finish"
        scheduler.monitor_stop.clear()
        scheduler.execute()
    except base.QueuePaused as error:
        scheduler.mark_training_pause()
        while any(job.running() for job in scheduler.gpu_jobs):
            time.sleep(2)
        base.atomic_json(DIRECTORY / "paused.json", dict(reason=str(error), time=time.time()))
        scheduler.publish("paused")
    except Exception as error:
        scheduler.mark_training_pause()
        base.atomic_json(DIRECTORY / "failed.json", dict(reason=repr(error), time=time.time()))
        scheduler.publish("failed_saving_active_jobs")
        while any(job.running() for job in scheduler.gpu_jobs):
            time.sleep(2)
        scheduler.publish("failed")
        raise
    finally:
        scheduler.monitor_stop.set()
        trial_monitor.join(timeout=5)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--take-over-pid", type=int, required=True)
    main(parser.parse_args())
