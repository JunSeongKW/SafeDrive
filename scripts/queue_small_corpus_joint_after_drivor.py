"""Prioritize saved joint LPWM after DrivoR's final development evaluation."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import time

import queue_small_corpus_memory_recovery as recovery

base = recovery.base
DIRECTORY = base.OUTPUT / "scheduling_v7_joint_after_drivor"


def drivor_evaluation_complete(study_directory):
    directory = study_directory / "drivor"
    try:
        training = json.loads((directory / "complete.json").read_text())
        evaluation = json.loads((directory / "validation/pass5.pdms.json").read_text())
        return (training["completed_updates"] == 3200
                and evaluation["count"] == 1024 and evaluation["failed"] == 0)
    except (FileNotFoundError, KeyError, json.JSONDecodeError):
        return False


def joint_priority_active(study_directory):
    return (drivor_evaluation_complete(study_directory)
            and not (study_directory / "lpwm_joint/complete.json").exists())


def priority_defers(kind, active, maximum_jobs):
    return active and (kind == "jepa" or (kind == "lpwm_sequential" and maximum_jobs == 1))


class JointAfterDrivorScheduler(recovery.MemoryRecoveryScheduler):
    def execute(self):
        super().execute()
        comparison = base.OUTPUT / "comparison_complete.json"
        if comparison.exists():
            result = json.loads(comparison.read_text())
            result["scheduler"] = "joint_after_drivor_v7"
            base.atomic_json(comparison, result)

    def publish(self, status="running"):
        jobs = self.gpu_jobs + ([self.cpu_job] if self.cpu_job else [])
        phase = ("joint_priority_after_drivor" if joint_priority_active(base.OUTPUT)
                 else "wait_drivor_final_evaluation" if not drivor_evaluation_complete(base.OUTPUT)
                 else "finish_remaining_training_and_evaluation")
        value = dict(status=status, scheduler_pid=os.getpid(), scheduler="joint_after_drivor_v7",
            phase=phase, stage=phase, execution_phase=self.phase,
            pid=self.gpu_jobs[0].identity["pid"] if self.gpu_jobs else None,
            updated_unix=time.time(), held_conditions=self.held_conditions,
            active_jobs=[dict(label=job.label, pid=job.identity["pid"], cpu=job.cpu,
                directory=str(job.directory), command=job.command) for job in jobs if job.running()],
            maximum_gpu_jobs=self.configuration["maximum_concurrent_gpu_jobs"],
            reservation_bytes=recovery.RESERVATIONS, admission_ceiling_bytes=recovery.ADMISSION_CEILING,
            transition_trigger="DrivoR3200 complete and pass5 PDMS1024 complete with zero failures",
            joint_resume_update=87, temporarily_deferred_kind="jepa" if joint_priority_active(base.OUTPUT) else None)
        base.atomic_json(DIRECTORY / "state.json", value)
        base.atomic_json(base.OUTPUT / "queue_state.json", value)

    def service_cpu_scoring(self):
        # Existing scoring workers may be adopted rather than direct children.
        if self.cpu_job and not self.cpu_job.running():
            job = self.cpu_job
            if job.log:
                job.log.close()
            self.cpu_job = None
            if job.returncode() not in (None, 0) or not job.expected.exists():
                raise RuntimeError("Official CPU scoring failed: " + job.label)
            base.atomic_json(DIRECTORY / (job.label + ".timing.json"), dict(
                wall_seconds=time.time() - self.evaluation_started[job.label],
                completed=True, adopted=job.process is None))
        base.StudyScheduler.service_cpu_scoring(self)

    def reap_finished(self):
        # A requested priority yield is a normal full-state checkpoint boundary.
        # It must not be classified as a memory failure or reduce concurrency.
        for job in list(self.gpu_jobs):
            if job.running() or job.expected.exists():
                continue
            marker = job.directory / "pause.requested"
            if not marker.exists():
                continue
            request = json.loads(marker.read_text())
            if request.get("scheduler_pid") != os.getpid() or request.get("reason") != "joint_after_drivor_priority":
                continue
            paused = job.directory / "paused.json"
            if job.returncode() not in (None, 0) or not paused.exists():
                raise RuntimeError("Priority yield failed to save: " + job.label)
            saved_update = json.loads(paused.read_text())["completed_updates"]
            assert (job.directory / "latest.pt").exists()
            assert json.loads((job.directory / "progress.json").read_text())["completed_updates"] == saved_update
            destination = DIRECTORY / "priority_yields" / job.label / str(saved_update)
            recovery.archive_markers(job.directory, destination, {os.getpid()})
            if job.log:
                job.log.close()
            self.gpu_jobs.remove(job)
            base.atomic_json(destination / "saved.json", dict(update=saved_update,
                preserved_full_state=True, automatically_resume_after_joint=True, time=time.time()))
        super().reap_finished()
        if not joint_priority_active(base.OUTPUT):
            return
        for job in self.gpu_jobs:
            kind = job.label.removeprefix("train_")
            if not priority_defers(kind, True, self.configuration["maximum_concurrent_gpu_jobs"]):
                continue
            marker = job.directory / "pause.requested"
            if job.running() and not marker.exists():
                base.atomic_json(marker, dict(owner="joint_after_drivor_v7", scheduler_pid=os.getpid(),
                    reason="joint_after_drivor_priority", resume_automatically=True,
                    requested_unix=time.time()))

    def launch_resume(self, kind):
        active = joint_priority_active(base.OUTPUT)
        if kind == "lpwm_joint" and not drivor_evaluation_complete(base.OUTPUT):
            return None
        if priority_defers(kind, active, self.configuration["maximum_concurrent_gpu_jobs"]):
            return None
        return super().launch_resume(kind)


def adopt_running_jobs(scheduler, state):
    identities = []
    for entry in state["active_jobs"]:
        identity = base.process_identity(entry["pid"])
        if not identity or not base.still_alive(identity):
            continue
        assert identity["uid"] == os.getuid()
        directory = Path(entry["directory"])
        assert directory.parent == base.OUTPUT
        if entry["cpu"]:
            assert "score_four_model_small_corpus.py" in identity["command"]
            prediction = Path(entry["command"][entry["command"].index("--predictions") + 1])
            expected = prediction.with_suffix(".pdms.json")
        else:
            assert "resume_small_corpus_planner_full_state.py" in identity["command"]
            assert entry["label"] in ("train_drivor", "train_jepa", "train_lpwm_sequential")
            expected = directory / "complete.json"
        job = base.Job(entry["label"], directory, entry["command"], expected, identity, cpu=entry["cpu"])
        identities.append(identity)
        if entry["cpu"]:
            assert scheduler.cpu_job is None
            scheduler.cpu_job = job
            scheduler.evaluation_started[job.label] = time.time()
        else:
            scheduler.gpu_jobs.append(job)
    return identities


def main(arguments):
    DIRECTORY.mkdir(parents=True, exist_ok=True)
    base.SCHEDULING = DIRECTORY
    recovery.DIRECTORY = DIRECTORY
    original = base.verify_original_sources()
    previous = base.OUTPUT / "scheduling_v6_memory_recovery"
    old_registration = json.loads((previous / "registration.json").read_text())
    for filename, expected in old_registration["source_sha256"].items():
        assert hashlib.sha256((base.ROOT / filename).read_bytes()).hexdigest() == expected
    assert not (base.OUTPUT / "pause.requested").exists()
    controller_lock = (DIRECTORY / "controller.lock").open("w")
    fcntl.flock(controller_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    state = json.loads((base.OUTPUT / "queue_state.json").read_text())
    assert state["status"] == "running" and state["scheduler_pid"] == arguments.take_over_pid
    controller = base.process_identity(arguments.take_over_pid)
    assert controller and controller["uid"] == os.getuid()
    assert "queue_small_corpus_memory_recovery.py" in controller["command"]
    configuration = old_registration["configuration"].copy()
    configuration["maximum_concurrent_gpu_jobs"] = state.get("maximum_gpu_jobs", 3)
    scheduler = JointAfterDrivorScheduler(configuration)
    scheduler.held_conditions = state.get("held_conditions", {})
    identities = adopt_running_jobs(scheduler, state)
    for kind in ("drivor", "jepa", "lpwm_sequential"):
        assert (base.OUTPUT / kind / "complete.json").exists() or any(
            job.label == "train_" + kind for job in scheduler.gpu_jobs), "Missing trainer: " + kind
    assert not (base.OUTPUT / "lpwm_joint/complete.json").exists()
    assert json.loads((base.OUTPUT / "lpwm_joint/progress.json").read_text())["completed_updates"] == 87
    paths = [Path(__file__), Path(recovery.__file__), Path(base.__file__),
             base.ROOT / "scripts/resume_small_corpus_planner_full_state.py",
             base.ROOT / "scripts/train_small_corpus_joint_adaptive_checkpointing.py"]
    base.atomic_json(DIRECTORY / "registration.json", dict(configuration=configuration,
        original_registration=original, previous_execution_registration=old_registration,
        user_requested="Immediately prioritize joint LPWM after DrivoR finishes",
        source_sha256={str(path.relative_to(base.ROOT)): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}))
    base.atomic_json(DIRECTORY / "handover_before.json", dict(controller=controller, queue=state, adopted=identities))
    # SIGTERM requests trainer pauses in the old controller. Only this verified
    # coordinator is killed; each adopted trainer has an independent session.
    assert base.still_alive(controller)
    os.kill(controller["pid"], signal.SIGKILL)
    deadline = time.monotonic() + 15
    while base.still_alive(controller):
        assert time.monotonic() < deadline
        time.sleep(.1)
    study_lock = (base.OUTPUT / "queue.lock").open("w")
    fcntl.flock(study_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    assert all(job.running() or job.expected.exists() for job in scheduler.gpu_jobs +
               ([scheduler.cpu_job] if scheduler.cpu_job else []))
    base.atomic_json(DIRECTORY / "handover_verified.json", dict(controller_pid=os.getpid(),
        adopted_jobs_alive=True, training_restart=False, adopted=identities))
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
