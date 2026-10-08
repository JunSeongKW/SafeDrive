"""Apply the user's 48GB policy once and continue all three saved trainings."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import time

import queue_small_corpus_guarded_jepa_overlap as guarded

base = guarded.base
recovery = guarded.recovery
DIRECTORY = base.OUTPUT / "scheduling_v9_user_48gb"
USER_CARD_LIMIT = 48_000_000_000


class UserMemoryLimitScheduler(guarded.GuardedJepaOverlapScheduler):
    def publish(self, status="running"):
        super().publish(status)
        value = json.loads((DIRECTORY / "state.json").read_text())
        value.update(scheduler="user_48gb_v9", trainer_card_guard_bytes=USER_CARD_LIMIT,
            user_card_limit_bytes=USER_CARD_LIMIT,
            temporary_checkpoint_restart=self.phase == "apply_user_memory_limit")
        base.atomic_json(DIRECTORY / "state.json", value)
        base.atomic_json(base.OUTPUT / "queue_state.json", value)

    def launch_resume(self, kind):
        if kind == "drivor":
            assert (base.OUTPUT / "drivor/complete.json").exists()
            return None
        if kind == "jepa":
            lpwm_count = sum(job.label.startswith("train_lpwm_") for job in self.gpu_jobs)
            if guarded.defer_jepa(lpwm_count, self.retry_below_count):
                return None
            self.retry_below_count = None
        directory = base.OUTPUT / kind
        arguments = ["--kind", kind, "--output", directory, "--micro-batch", "2"]
        if kind in ("jepa", "lpwm_sequential"):
            arguments += ["--stage1-checkpoint", base.OUTPUT / ("jepa_ssl" if kind == "jepa" else "lpwm_ssl") / "latest.pt"]
        if kind == "lpwm_joint":
            arguments += ["--native-marker", recovery.NATIVE_MARKER]
        before = os.environ.get("PYTORCH_CUDA_ALLOC_CONF")
        if kind in ("jepa", "lpwm_joint"):
            os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
        try:
            return self.launch("train_" + kind, base.original_queue.distributed_command(
                "resume_small_corpus_user_memory_limit.py", arguments), directory)
        finally:
            if before is None:
                os.environ.pop("PYTORCH_CUDA_ALLOC_CONF", None)
            else:
                os.environ["PYTORCH_CUDA_ALLOC_CONF"] = before

    def apply_trainer_policy(self):
        self.phase = "apply_user_memory_limit"
        self.mark_training_pause()
        while any(job.running() for job in self.gpu_jobs):
            self.service_cpu_scoring()
            self.publish("saving_for_memory_policy_update")
            time.sleep(2)
        for job in self.gpu_jobs:
            paused = job.directory / "paused.json"
            assert job.returncode() in (None, 0) and paused.exists()
            update = json.loads(paused.read_text())["completed_updates"]
            history = DIRECTORY / "saved_before_policy_update" / job.label
            history.mkdir(parents=True)
            for name in ("latest.pt", "registration.json", "progress.json"):
                os.link(job.directory / name, history / name)
            recovery.archive_markers(job.directory, history / "markers", {os.getpid()})
            base.atomic_json(history / "saved.json", dict(completed_updates=update,
                automatic_immediate_resume=True, time=time.time()))
        self.gpu_jobs.clear()
        if self.stop_requested or (base.OUTPUT / "pause.requested").exists():
            raise base.QueuePaused("User pause honored after full-state save")
        # Start the two LPWM jobs before capped JEPA, as in the validated v8
        # overlap; this avoids the earlier conservative inverse-order gate.
        self.phase = "resume_with_user_48gb_limit"
        for kind in ("lpwm_sequential", "lpwm_joint", "jepa"):
            active = [job.label.removeprefix("train_") for job in self.gpu_jobs]
            assert guarded.guarded_admission(kind, active, max(recovery.current_card_bytes()), 3)
            self.launch_resume(kind)

    def execute(self):
        self.apply_trainer_policy()
        super().execute()
        path = base.OUTPUT / "comparison_complete.json"
        if path.exists():
            result = json.loads(path.read_text())
            result["scheduler"] = "user_48gb_v9"
            base.atomic_json(path, result)


def main(arguments):
    DIRECTORY.mkdir(parents=True, exist_ok=True)
    guarded.DIRECTORY = DIRECTORY
    guarded.previous.DIRECTORY = DIRECTORY
    base.SCHEDULING = DIRECTORY
    recovery.DIRECTORY = DIRECTORY
    guarded.JEPA_SAVE_PAUSE_BYTES = USER_CARD_LIMIT
    guarded.TRIAL_ADMISSION_BYTES = USER_CARD_LIMIT
    recovery.can_admit = guarded.guarded_admission
    guarded.previous.priority_defers = lambda *_arguments: False
    original = base.verify_original_sources()
    previous_directory = base.OUTPUT / "scheduling_v8_guarded_jepa_overlap"
    registration = json.loads((previous_directory / "registration.json").read_text())
    for filename, expected in registration["source_sha256"].items():
        assert hashlib.sha256((base.ROOT / filename).read_bytes()).hexdigest() == expected
    controller_lock = (DIRECTORY / "controller.lock").open("w")
    fcntl.flock(controller_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    assert not (base.OUTPUT / "pause.requested").exists()
    state = json.loads((base.OUTPUT / "queue_state.json").read_text())
    assert state["status"] == "running" and state["scheduler_pid"] == arguments.take_over_pid
    controller = base.process_identity(arguments.take_over_pid)
    assert controller and controller["uid"] == os.getuid()
    assert "queue_small_corpus_guarded_jepa_overlap.py" in controller["command"]
    scheduler = UserMemoryLimitScheduler(registration["configuration"].copy())
    adopted = []
    for entry in state["active_jobs"]:
        identity = base.process_identity(entry["pid"])
        assert identity and base.still_alive(identity) and identity["uid"] == os.getuid()
        directory = Path(entry["directory"])
        assert directory.parent == base.OUTPUT
        if entry["cpu"]:
            assert "score_four_model_small_corpus.py" in identity["command"]
            prediction = Path(entry["command"][entry["command"].index("--predictions")+1])
            expected = prediction.with_suffix(".pdms.json")
        else:
            assert entry["label"] in ("train_jepa", "train_lpwm_sequential", "train_lpwm_joint")
            assert "resume_small_corpus_" in identity["command"]
            assert not (directory / "pause.requested").exists()
            expected = directory / "complete.json"
        job = base.Job(entry["label"], directory, entry["command"], expected, identity, cpu=entry["cpu"])
        adopted.append(identity)
        if entry["cpu"]:
            scheduler.cpu_job = job
            scheduler.evaluation_started[job.label] = time.time()
        else:
            scheduler.gpu_jobs.append(job)
    assert len(scheduler.gpu_jobs) == 3
    paths = [Path(__file__), Path(guarded.__file__), Path(guarded.previous.__file__), Path(recovery.__file__),
        base.ROOT / "scripts/resume_small_corpus_user_memory_limit.py",
        base.ROOT / "scripts/resume_small_corpus_jepa_memory_cap.py",
        base.ROOT / "scripts/resume_small_corpus_planner_full_state.py"]
    base.atomic_json(DIRECTORY / "registration.json", dict(configuration=scheduler.configuration,
        original_registration=original, previous_registration=registration,
        user_request="Continue up to48GB; remove45GB and46.5GB voluntary stops",
        active_card_limit_bytes=USER_CARD_LIMIT,
        source_sha256={str(path.relative_to(base.ROOT)):hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}))
    base.atomic_json(DIRECTORY / "handover_before.json", dict(controller=controller, queue=state, adopted=adopted))
    os.kill(controller["pid"], signal.SIGKILL)
    deadline = time.monotonic()+15
    while base.still_alive(controller):
        assert time.monotonic()<deadline
        time.sleep(.1)
    study_lock = (base.OUTPUT / "queue.lock").open("w")
    fcntl.flock(study_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
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
    parser=argparse.ArgumentParser()
    parser.add_argument("--take-over-pid",type=int,required=True)
    main(parser.parse_args())
