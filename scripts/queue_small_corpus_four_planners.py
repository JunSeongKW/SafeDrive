"""Adopt the three live planners and add the validated memory-bounded joint job."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import time

import queue_four_model_small_corpus_overlap_launch_fix as base

DIRECTORY = base.OUTPUT / "scheduling_v5_joint_overlap"
NATIVE_MARKER = DIRECTORY / "native_joint_allowed.json"


class FourPlannerScheduler(base.StudyScheduler):
    def publish(self, status="running"):
        super().publish(status)
        state = json.loads((DIRECTORY / "state.json").read_text())
        state["scheduler"] = "four_planners_v5"
        base.atomic_json(DIRECTORY / "state.json", state)
        base.atomic_json(base.OUTPUT / "queue_state.json", state)

    def service_cpu_scoring(self):
        # Adopted workers are not children of this process, so their exit status
        # is unavailable. A complete score artifact is required in either case.
        if self.cpu_job and not self.cpu_job.running():
            job = self.cpu_job
            if job.log:
                job.log.close()
            self.cpu_job = None
            if job.returncode() not in (None, 0) or not job.expected.exists():
                raise RuntimeError("Official CPU scoring failed: " + job.label)
            base.atomic_json(DIRECTORY / (job.label + ".timing.json"),
                dict(completed=True, adopted=job.process is None))
        super().service_cpu_scoring()

    def reap_finished(self):
        for job in list(self.gpu_jobs):
            if job.running():
                continue
            if job.log:
                job.log.close()
            self.gpu_jobs.remove(job)
            if job.returncode() not in (None, 0) or not job.expected.exists():
                raise RuntimeError("Training failed or paused: " + job.label)

    def execute(self):
        self.phase = "four_planners_parallel"
        environment_before = os.environ.get("PYTORCH_CUDA_ALLOC_CONF")
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
        command = base.original_queue.distributed_command("train_small_corpus_joint_adaptive_checkpointing.py",
            ["--kind", "lpwm_joint", "--output", base.OUTPUT / "lpwm_joint", "--micro-batch", "2",
             "--native-marker", NATIVE_MARKER])
        try:
            self.launch("train_lpwm_joint", command, base.OUTPUT / "lpwm_joint")
        finally:
            if environment_before is None:
                os.environ.pop("PYTORCH_CUDA_ALLOC_CONF", None)
            else:
                os.environ["PYTORCH_CUDA_ALLOC_CONF"] = environment_before
        while self.gpu_jobs:
            if self.stop_requested or (base.OUTPUT / "pause.requested").exists():
                self.stop_requested = True
                self.mark_training_pause()
                self.wait_jobs(list(self.gpu_jobs))
                raise base.QueuePaused("Requested pause completed")
            self.reap_finished()
            if self.gpu_jobs and all(job.label == "train_lpwm_joint" for job in self.gpu_jobs):
                self.phase = "joint_exclusive_native_storage"
                if not NATIVE_MARKER.exists():
                    base.atomic_json(NATIVE_MARKER, dict(time=time.time(), scheduler_pid=os.getpid(),
                        all_other_registered_gpu_jobs_complete=True,
                        additional_guard="Each rank must see external/context memory below2GB"))
            self.service_cpu_scoring()
            self.publish()
            time.sleep(2)
        self.phase = "finish_cpu_scoring"
        while True:
            self.service_cpu_scoring()
            remaining = [kind for kind in base.PLANNING_KINDS if kind not in self.held_conditions
                and any(not (base.OUTPUT / kind / "validation" / f"pass{epoch}.pdms.json").exists() for epoch in (1, 3, 5))]
            if not remaining and self.cpu_job is None:
                break
            if self.stop_requested or (base.OUTPUT / "pause.requested").exists():
                raise base.QueuePaused("Scoring queue paused")
            if remaining and self.cpu_job is None:
                raise RuntimeError("Missing prediction artifacts: " + repr(remaining))
            self.publish()
            time.sleep(2)
        results = {kind: json.loads((base.OUTPUT / kind / "validation/pass5.pdms.json").read_text())
                   for kind in base.PLANNING_KINDS if kind not in self.held_conditions}
        base.atomic_json(base.OUTPUT / ("comparison_partial.json" if self.held_conditions else "comparison_complete.json"),
            dict(results=results, held_conditions=self.held_conditions, full_navtest=False, seed_count=1,
                 scheduler="four_planners_v5", conclusion_scope="Small-data system trends; upstream pretraining differs"))
        self.publish("held_for_stage1_review" if self.held_conditions else "complete")


def main(arguments):
    base.SCHEDULING = DIRECTORY
    original = base.verify_original_sources()
    admission = json.loads((DIRECTORY / "admission.json").read_text())
    assert admission["approved"]
    assert not NATIVE_MARKER.exists()
    assert not (base.OUTPUT / "pause.requested").exists()
    assert not (base.OUTPUT / "lpwm_joint/registration.json").exists()
    controller_lock = (DIRECTORY / "controller.lock").open("w")
    fcntl.flock(controller_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    state = json.loads((base.OUTPUT / "queue_state.json").read_text())
    assert state["status"] == "running" and state["scheduler_pid"] == arguments.take_over_pid
    controller = base.process_identity(arguments.take_over_pid)
    assert controller and controller["uid"] == os.getuid() and "queue_small_corpus_three_jobs.py" in controller["command"]
    configuration = json.loads(base.CONFIGURATION.read_text())
    configuration["maximum_concurrent_gpu_jobs"] = 4
    scheduler = FourPlannerScheduler(configuration)
    scheduler.held_conditions = state.get("held_conditions", {})
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
            assert "train_small_corpus_common_planner.py" in identity["command"]
            assert entry["label"] in ("train_drivor", "train_jepa", "train_lpwm_sequential")
            expected = directory / "complete.json"
        job = base.Job(entry["label"], directory, entry["command"], expected, identity, cpu=entry["cpu"])
        identities.append(identity)
        if entry["cpu"]:
            assert scheduler.cpu_job is None
            scheduler.cpu_job = job
        else:
            scheduler.gpu_jobs.append(job)
    assert scheduler.gpu_jobs
    for kind in ("drivor", "jepa", "lpwm_sequential"):
        assert (base.OUTPUT / kind / "complete.json").exists() or any(
            job.label == "train_" + kind for job in scheduler.gpu_jobs), "Missing existing trainer: " + kind
    base.atomic_json(DIRECTORY / "handover_before.json", dict(controller=controller, queue=state, adopted=identities))
    # The old controller's SIGTERM handler pauses its jobs. Terminate only the
    # verified controller; all adopted jobs have independent sessions.
    os.kill(controller["pid"], signal.SIGKILL)
    deadline = time.monotonic() + 15
    while base.still_alive(controller):
        assert time.monotonic() < deadline
        time.sleep(.1)
    study_lock = (base.OUTPUT / "queue.lock").open("w")
    fcntl.flock(study_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    assert all(job.running() or job.expected.exists() for job in scheduler.gpu_jobs +
               ([scheduler.cpu_job] if scheduler.cpu_job else []))
    paths = [Path(__file__), Path(base.__file__), base.CONFIGURATION,
             base.ROOT / "scripts/train_small_corpus_joint_adaptive_checkpointing.py",
             DIRECTORY / "protocol.json", DIRECTORY / "admission.json"]
    base.atomic_json(DIRECTORY / "registration.json", dict(configuration=configuration, original_registration=original,
        source_sha256={str(path.relative_to(base.ROOT)): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}))
    base.atomic_json(DIRECTORY / "handover_verified.json", dict(adopted_jobs_alive=True, training_restart=False,
        controller_pid=os.getpid(), adopted=identities))
    signal.signal(signal.SIGTERM, scheduler.request_stop)
    signal.signal(signal.SIGINT, scheduler.request_stop)
    try:
        scheduler.execute()
    except base.QueuePaused as error:
        scheduler.publish("paused")
        base.atomic_json(DIRECTORY / "paused.json", dict(reason=str(error), time=time.time()))
    except Exception as error:
        scheduler.mark_training_pause()
        base.atomic_json(DIRECTORY / "failed.json", dict(error=repr(error), time=time.time()))
        scheduler.publish("failed_saving_active_jobs")
        while any(job.running() for job in scheduler.gpu_jobs):
            time.sleep(2)
        scheduler.publish("failed")
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--take-over-pid", type=int, required=True)
    main(parser.parse_args())
