"""Resume saved planners with headroom and recover memory pauses as queued work."""
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import time

import queue_four_model_small_corpus_overlap_launch_fix as base
from queue_small_corpus_four_planners import FourPlannerScheduler

DIRECTORY = base.OUTPUT / "scheduling_v6_memory_recovery"
NATIVE_MARKER = base.OUTPUT / "scheduling_v5_joint_overlap/native_joint_allowed.json"
RESERVATIONS = dict(drivor=3_000_000_000, jepa=12_000_000_000,
                    lpwm_sequential=10_000_000_000, lpwm_joint=31_000_000_000)
ADMISSION_CEILING = 44_000_000_000
OVERHEAD = 2_000_000_000


def current_card_bytes():
    result = subprocess.check_output(["nvidia-smi", "--id=0,1", "--query-gpu=memory.used",
                                      "--format=csv,noheader,nounits"], text=True)
    return [int(value) * 1024**2 for value in result.splitlines()]


def can_admit(kind, active_kinds, observed_peak, maximum_jobs):
    if len(active_kinds) >= maximum_jobs:
        return False
    protected_usage = max(observed_peak, OVERHEAD + sum(RESERVATIONS[value] for value in active_kinds))
    return protected_usage + RESERVATIONS[kind] <= ADMISSION_CEILING


def archive_markers(directory, destination, allowed_scheduler_pids):
    marker = directory / "pause.requested"
    if marker.exists():
        value = json.loads(marker.read_text())
        assert value.get("scheduler_pid") in allowed_scheduler_pids, "Preserve unowned pause request"
    destination.mkdir(parents=True, exist_ok=True)
    for name in ("pause.requested", "paused.json"):
        path = directory / name
        if path.exists():
            assert not (destination / name).exists()
            path.rename(destination / name)


class MemoryRecoveryScheduler(FourPlannerScheduler):
    def service_cpu_scoring(self):
        # Every evaluation worker in this recovery queue is a direct child.
        return base.StudyScheduler.service_cpu_scoring(self)

    def publish(self, status="running"):
        # Avoid the v5 class's fixed directory while preserving the base schema.
        base.StudyScheduler.publish(self, status)
        state = json.loads((DIRECTORY / "state.json").read_text())
        state.update(scheduler="memory_recovery_v6", maximum_gpu_jobs=self.configuration["maximum_concurrent_gpu_jobs"],
                     reservation_bytes=RESERVATIONS, admission_ceiling_bytes=ADMISSION_CEILING)
        base.atomic_json(DIRECTORY / "state.json", state)
        base.atomic_json(base.OUTPUT / "queue_state.json", state)

    def reap_finished(self):
        for job in list(self.gpu_jobs):
            if job.running():
                continue
            if job.log:
                job.log.close()
            self.gpu_jobs.remove(job)
            if job.expected.exists() and job.returncode() in (None, 0):
                continue
            paused = job.directory / "paused.json"
            marker = job.directory / "pause.requested"
            rank_rows = [base.timing_rows(job.directory, rank)[-1] for rank in (0, 1)]
            high_memory = max(row["card_used_bytes"] for row in rank_rows) > 46_500_000_000
            owned_marker = marker.exists() and json.loads(marker.read_text()).get("scheduler_pid") == os.getpid()
            if paused.exists() and job.returncode() in (None, 0) and (high_memory or owned_marker):
                update = json.loads(paused.read_text())["completed_updates"]
                archive_markers(job.directory, DIRECTORY / "memory_pauses" / job.label / str(update), {os.getpid()})
                self.configuration["maximum_concurrent_gpu_jobs"] = max(1, min(
                    self.configuration["maximum_concurrent_gpu_jobs"] - 1, len(self.gpu_jobs)))
                base.atomic_json(DIRECTORY / "last_memory_requeue.json", dict(label=job.label, update=update,
                    reason="Saved memory pause requeued without stopping healthy companions",
                    maximum_gpu_jobs=self.configuration["maximum_concurrent_gpu_jobs"], time=time.time()))
                continue
            raise RuntimeError("Unrecognized training failure: " + job.label)

    def launch_resume(self, kind):
        directory = base.OUTPUT / kind
        arguments = ["--kind", kind, "--output", directory, "--micro-batch", "2"]
        if kind in ("jepa", "lpwm_sequential"):
            arguments += ["--stage1-checkpoint", base.OUTPUT / ("jepa_ssl" if kind == "jepa" else "lpwm_ssl") / "latest.pt"]
        if kind == "lpwm_joint":
            arguments += ["--native-marker", NATIVE_MARKER]
        environment_before = os.environ.get("PYTORCH_CUDA_ALLOC_CONF")
        if kind == "lpwm_joint":
            os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
        try:
            return self.launch("train_" + kind, base.original_queue.distributed_command(
                "resume_small_corpus_planner_full_state.py", arguments), directory)
        finally:
            if environment_before is None:
                os.environ.pop("PYTORCH_CUDA_ALLOC_CONF", None)
            else:
                os.environ["PYTORCH_CUDA_ALLOC_CONF"] = environment_before

    def execute(self):
        self.phase = "resume_with_reserved_memory_headroom"
        while True:
            if self.stop_requested or (base.OUTPUT / "pause.requested").exists():
                self.stop_requested = True
                self.mark_training_pause()
                self.wait_jobs(list(self.gpu_jobs))
                raise base.QueuePaused("User pause preserved")
            self.reap_finished()
            remaining = [kind for kind in base.PLANNING_KINDS if not (base.OUTPUT / kind / "complete.json").exists()]
            if not remaining:
                break
            for kind in ("drivor", "jepa", "lpwm_sequential", "lpwm_joint"):
                active = [job.label.removeprefix("train_") for job in self.gpu_jobs]
                if kind not in remaining or kind in active:
                    continue
                if can_admit(kind, active, max(current_card_bytes()), self.configuration["maximum_concurrent_gpu_jobs"]):
                    self.launch_resume(kind)
            if remaining == ["lpwm_joint"] and self.gpu_jobs and not NATIVE_MARKER.exists():
                base.atomic_json(NATIVE_MARKER, dict(time=time.time(), scheduler_pid=os.getpid(),
                    all_other_registered_gpu_jobs_complete=True, per_rank_external_memory_guard_bytes=2_000_000_000))
            self.service_cpu_scoring()
            self.publish()
            time.sleep(2)
        self.phase = "finish_cpu_scoring"
        while True:
            self.service_cpu_scoring()
            pending = [kind for kind in base.PLANNING_KINDS if any(
                not (base.OUTPUT / kind / "validation" / f"pass{epoch}.pdms.json").exists() for epoch in (1, 3, 5))]
            if not pending and self.cpu_job is None:
                break
            if self.stop_requested or (base.OUTPUT / "pause.requested").exists():
                raise base.QueuePaused("User evaluation pause")
            if pending and self.cpu_job is None:
                raise RuntimeError("Missing prediction artifacts: " + repr(pending))
            self.publish()
            time.sleep(2)
        base.atomic_json(base.OUTPUT / "comparison_complete.json", dict(
            results={kind: json.loads((base.OUTPUT / kind / "validation/pass5.pdms.json").read_text()) for kind in base.PLANNING_KINDS},
            full_navtest=False, seed_count=1, scheduler="memory_recovery_v6"))
        self.publish("complete")


def main():
    DIRECTORY.mkdir(parents=True, exist_ok=True)
    base.SCHEDULING = DIRECTORY
    original = base.verify_original_sources()
    assert not (base.OUTPUT / "pause.requested").exists()
    state = json.loads((base.OUTPUT / "queue_state.json").read_text())
    assert state["status"] == "failed" and state["scheduler_pid"] == 2114751
    assert not state["active_jobs"]
    for pid in (2114751, 2114768, 1567970, 1580030, 1891906):
        identity = base.process_identity(pid)
        assert identity is None or not base.still_alive(identity), "An old job is still alive"
    lock = (base.OUTPUT / "queue.lock").open("w")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    for kind in base.PLANNING_KINDS:
        directory = base.OUTPUT / kind
        history = DIRECTORY / "before_recovery" / kind
        history.mkdir(parents=True, exist_ok=True)
        assert (directory / "latest.pt").exists()
        for name in ("latest.pt", "registration.json", "progress.json"):
            assert not (history / name).exists()
            os.link(directory / name, history / name)
        archive_markers(directory, history / "markers", {2114751})
    configuration = json.loads(base.CONFIGURATION.read_text())
    configuration["maximum_concurrent_gpu_jobs"] = 3
    paths = [Path(__file__), base.ROOT / "scripts/resume_small_corpus_planner_full_state.py",
             base.ROOT / "scripts/train_small_corpus_joint_adaptive_checkpointing.py",
             base.ROOT / "scripts/queue_small_corpus_four_planners.py", Path(base.__file__), base.CONFIGURATION]
    base.atomic_json(DIRECTORY / "registration.json", dict(original_registration=original,
        configuration=configuration, reservation_bytes=RESERVATIONS, admission_ceiling_bytes=ADMISSION_CEILING,
        source_sha256={str(path.relative_to(base.ROOT)): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}))
    scheduler = MemoryRecoveryScheduler(configuration)
    signal.signal(signal.SIGINT, scheduler.request_stop)
    signal.signal(signal.SIGTERM, scheduler.request_stop)
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
    main()
