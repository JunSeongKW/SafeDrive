"""Run capped JEPA alongside both LPWM jobs with continuous memory monitoring."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import threading
import time

import queue_small_corpus_joint_after_drivor as previous

base = previous.base
recovery = previous.recovery
DIRECTORY = base.OUTPUT / "scheduling_v8_guarded_jepa_overlap"
BASELINE_TWO_JOB_PEAK_BYTES = 36_500_000_000  # Includes the observed external GPU job.
JEPA_RESERVATION_BYTES = 9_000_000_000  # 8.5GB allocator plus CUDA overhead.
TRIAL_ADMISSION_BYTES = 45_500_000_000
JEPA_SAVE_PAUSE_BYTES = 45_000_000_000
original_admission = recovery.can_admit


def guarded_admission(kind, active_kinds, observed_peak, maximum_jobs):
    if len(active_kinds) >= maximum_jobs:
        return False
    if kind == "jepa" and "lpwm_joint" in active_kinds:
        baseline = BASELINE_TWO_JOB_PEAK_BYTES if "lpwm_sequential" in active_kinds else 33_000_000_000
        return max(observed_peak, baseline) + JEPA_RESERVATION_BYTES <= TRIAL_ADMISSION_BYTES
    return original_admission(kind, active_kinds, observed_peak, maximum_jobs)


def defer_jepa(active_lpwm_count, retry_below_count):
    return retry_below_count is not None and active_lpwm_count >= retry_below_count


class GuardedJepaOverlapScheduler(previous.JointAfterDrivorScheduler):
    def __init__(self, configuration):
        super().__init__(configuration)
        self.retry_below_count = None
        self.memory_peak_bytes = [0, 0]
        self.monitor_error = None
        self.monitor_stop = threading.Event()

    def publish(self, status="running"):
        super().publish(status)
        value = json.loads((DIRECTORY / "state.json").read_text())
        value.update(scheduler="guarded_jepa_overlap_v8", phase="guarded_three_job_overlap",
            stage="guarded_three_job_overlap", temporarily_deferred_kind="jepa" if self.retry_below_count else None,
            monitored_peak_card_bytes=self.memory_peak_bytes, jepa_allocator_cap_bytes=8_500_000_000,
            jepa_retry_below_active_lpwm_count=self.retry_below_count,
            jepa_save_pause_bytes=JEPA_SAVE_PAUSE_BYTES,
            admission_ceiling_bytes=TRIAL_ADMISSION_BYTES,
            admission_note="JEPA uses measured LPWM+external peak plus9GB; other admissions keep original reservations")
        base.atomic_json(DIRECTORY / "state.json", value)
        base.atomic_json(base.OUTPUT / "queue_state.json", value)

    def monitor_memory(self):
        failures = 0
        try:
            with (DIRECTORY / "memory_samples.jsonl").open("a") as stream:
                while not self.monitor_stop.is_set():
                    try:
                        values = recovery.current_card_bytes()
                    except Exception as error:
                        failures += 1
                        if failures >= 3:
                            raise RuntimeError("Repeated GPU telemetry failure") from error
                        self.monitor_stop.wait(.5)
                        continue
                    failures = 0
                    self.memory_peak_bytes = [max(old, new) for old, new in zip(self.memory_peak_bytes, values)]
                    jobs = list(self.gpu_jobs)
                    live = [job for job in jobs if job.running()]
                    stream.write(json.dumps(dict(time=time.time(), card_bytes=values,
                        active=[job.label for job in live])) + "\n")
                    stream.flush()
                    jepa = next((job for job in live if job.label == "train_jepa"), None)
                    lpwm_count = sum(job.label.startswith("train_lpwm_") for job in live)
                    if jepa and lpwm_count and max(values) >= JEPA_SAVE_PAUSE_BYTES:
                        self.retry_below_count = lpwm_count
                        marker = jepa.directory / "pause.requested"
                        if not marker.exists():
                            base.atomic_json(marker, dict(owner="guarded_jepa_overlap_v8",
                                scheduler_pid=os.getpid(), reason="jepa_memory_headroom",
                                retry_below_active_lpwm_count=lpwm_count, observed_card_bytes=values,
                                requested_unix=time.time()))
                    self.monitor_stop.wait(.5)
        except Exception as error:
            self.monitor_error = error

    def reap_finished(self):
        if self.monitor_error:
            raise self.monitor_error
        for job in list(self.gpu_jobs):
            if job.label != "train_jepa" or job.running() or job.expected.exists():
                continue
            marker = job.directory / "pause.requested"
            if not marker.exists():
                continue
            request = json.loads(marker.read_text())
            if request.get("scheduler_pid") != os.getpid() or request.get("reason") != "jepa_memory_headroom":
                continue
            paused = job.directory / "paused.json"
            assert job.returncode() in (None, 0) and paused.exists(), "JEPA memory yield did not complete"
            update = json.loads(paused.read_text())["completed_updates"]
            assert (job.directory / "latest.pt").exists()
            recovery.archive_markers(job.directory, DIRECTORY / "memory_yields" / str(update), {os.getpid()})
            if job.log:
                job.log.close()
            self.gpu_jobs.remove(job)
            base.atomic_json(DIRECTORY / "last_jepa_memory_yield.json", dict(update=update,
                retry_below_active_lpwm_count=self.retry_below_count, other_jobs_continue=True))
        super().reap_finished()

    def launch_resume(self, kind):
        if kind != "jepa":
            return recovery.MemoryRecoveryScheduler.launch_resume(self, kind)
        lpwm_count = sum(job.label.startswith("train_lpwm_") for job in self.gpu_jobs)
        if defer_jepa(lpwm_count, self.retry_below_count):
            return None
        self.retry_below_count = None
        directory = base.OUTPUT / "jepa"
        arguments = ["--kind", "jepa", "--output", directory, "--micro-batch", "2",
                     "--stage1-checkpoint", base.OUTPUT / "jepa_ssl/latest.pt"]
        before = os.environ.get("PYTORCH_CUDA_ALLOC_CONF")
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
        try:
            return self.launch("train_jepa", base.original_queue.distributed_command(
                "resume_small_corpus_jepa_memory_cap.py", arguments), directory)
        finally:
            if before is None:
                os.environ.pop("PYTORCH_CUDA_ALLOC_CONF", None)
            else:
                os.environ["PYTORCH_CUDA_ALLOC_CONF"] = before

    def execute(self):
        monitor = threading.Thread(target=self.monitor_memory, daemon=True)
        monitor.start()
        try:
            super().execute()
            path = base.OUTPUT / "comparison_complete.json"
            if path.exists():
                result = json.loads(path.read_text())
                result["scheduler"] = "guarded_jepa_overlap_v8"
                base.atomic_json(path, result)
        finally:
            self.monitor_stop.set()
            monitor.join(timeout=5)


def main(arguments):
    DIRECTORY.mkdir(parents=True, exist_ok=True)
    previous.DIRECTORY = DIRECTORY
    base.SCHEDULING = DIRECTORY
    recovery.DIRECTORY = DIRECTORY
    recovery.can_admit = guarded_admission
    previous.priority_defers = lambda *_arguments: False
    original = base.verify_original_sources()
    old_directory = base.OUTPUT / "scheduling_v7_joint_after_drivor"
    old_registration = json.loads((old_directory / "registration.json").read_text())
    for filename, expected in old_registration["source_sha256"].items():
        assert hashlib.sha256((base.ROOT / filename).read_bytes()).hexdigest() == expected
    controller_lock = (DIRECTORY / "controller.lock").open("w")
    fcntl.flock(controller_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    assert not (base.OUTPUT / "pause.requested").exists()
    assert not (base.OUTPUT / "jepa/pause.requested").exists()
    state = json.loads((base.OUTPUT / "queue_state.json").read_text())
    assert state["status"] == "running" and state["scheduler_pid"] == arguments.take_over_pid
    controller = base.process_identity(arguments.take_over_pid)
    assert controller and controller["uid"] == os.getuid()
    assert "queue_small_corpus_joint_after_drivor.py" in controller["command"]
    configuration = old_registration["configuration"].copy()
    configuration["maximum_concurrent_gpu_jobs"] = 3
    scheduler = GuardedJepaOverlapScheduler(configuration)
    identities = []
    for entry in state["active_jobs"]:
        identity = base.process_identity(entry["pid"])
        assert identity and base.still_alive(identity) and identity["uid"] == os.getuid()
        directory = Path(entry["directory"])
        assert directory.parent == base.OUTPUT
        if entry["cpu"]:
            assert "score_four_model_small_corpus.py" in identity["command"]
            prediction = Path(entry["command"][entry["command"].index("--predictions") + 1])
            expected = prediction.with_suffix(".pdms.json")
        else:
            assert entry["label"] in ("train_lpwm_joint", "train_lpwm_sequential")
            assert "resume_small_corpus_planner_full_state.py" in identity["command"]
            expected = directory / "complete.json"
        job = base.Job(entry["label"], directory, entry["command"], expected, identity, cpu=entry["cpu"])
        identities.append(identity)
        if entry["cpu"]:
            scheduler.cpu_job = job
            scheduler.evaluation_started[job.label] = time.time()
        else:
            scheduler.gpu_jobs.append(job)
    assert len(scheduler.gpu_jobs) == 2
    assert json.loads((base.OUTPUT / "jepa/progress.json").read_text())["completed_updates"] == 1428
    backup = DIRECTORY / "before_jepa_resume"
    backup.mkdir()
    for name in ("latest.pt", "registration.json", "progress.json"):
        os.link(base.OUTPUT / "jepa" / name, backup / name)
    paths = [Path(__file__), Path(previous.__file__), Path(recovery.__file__), Path(base.__file__),
             base.ROOT / "scripts/resume_small_corpus_jepa_memory_cap.py",
             base.ROOT / "scripts/resume_small_corpus_planner_full_state.py"]
    base.atomic_json(DIRECTORY / "registration.json", dict(configuration=configuration,
        original_registration=original, previous_registration=old_registration,
        user_requested="Resume JEPA concurrently if feasible under48GB",
        baseline_peak_including_external_bytes=BASELINE_TWO_JOB_PEAK_BYTES,
        jepa_additional_reservation_bytes=JEPA_RESERVATION_BYTES,
        continuous_save_pause_bytes=JEPA_SAVE_PAUSE_BYTES,
        source_sha256={str(path.relative_to(base.ROOT)):hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}))
    base.atomic_json(DIRECTORY / "handover_before.json", dict(controller=controller, queue=state, adopted=identities))
    os.kill(controller["pid"], signal.SIGKILL)
    deadline = time.monotonic() + 15
    while base.still_alive(controller):
        assert time.monotonic() < deadline
        time.sleep(.1)
    study_lock = (base.OUTPUT / "queue.lock").open("w")
    fcntl.flock(study_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    assert all(job.running() for job in scheduler.gpu_jobs)
    base.atomic_json(DIRECTORY / "handover_verified.json", dict(controller_pid=os.getpid(),
        adopted_jobs_alive=True, lpwm_training_restart=False, adopted=identities))
    signal.signal(signal.SIGTERM, scheduler.request_stop)
    signal.signal(signal.SIGINT, scheduler.request_stop)
    try:
        scheduler.execute()
    except base.QueuePaused as error:
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--take-over-pid", type=int, required=True)
    main(parser.parse_args())
