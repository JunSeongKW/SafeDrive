"""Use a measured repeatability audit for the first pair; preserve trainers."""
import fcntl
import hashlib
import json
from pathlib import Path
import signal
import time

import queue_four_model_small_corpus_overlap_launch_fix as scheduler_base

ROOT = scheduler_base.ROOT
OUTPUT = scheduler_base.OUTPUT
SCHEDULING = OUTPUT / "scheduling_v3_calibrated"


class CalibratedStudyScheduler(scheduler_base.StudyScheduler):
    def pair_admission(self, name, conditions):
        if name == "jepa_ssl_and_drivor":
            audit = json.loads((SCHEDULING / "repeatability_admission.json").read_text())
            scheduler_base.atomic_json(SCHEDULING / (name + ".admission.json"), audit)
            return audit["parallel_approved"]
        return super().pair_admission(name, conditions)

    def publish(self, status="running"):
        super().publish(status)
        state = json.loads((SCHEDULING / "state.json").read_text())
        state["scheduler"] = "overlap_v3_repeatability"
        scheduler_base.atomic_json(SCHEDULING / "state.json", state)
        scheduler_base.atomic_json(OUTPUT / "queue_state.json", state)


def main():
    SCHEDULING.mkdir(parents=True, exist_ok=True)
    scheduler_base.SCHEDULING = SCHEDULING
    original = scheduler_base.verify_original_sources()
    configuration = json.loads(scheduler_base.CONFIGURATION.read_text())
    controller_lock = (SCHEDULING / "controller.lock").open("w")
    fcntl.flock(controller_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    study_lock = (OUTPUT / "queue.lock").open("w")
    fcntl.flock(study_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    paths = [Path(__file__), Path(scheduler_base.__file__), scheduler_base.CONFIGURATION,
             ROOT / "scripts/reassess_small_corpus_parallel_execution.py",
             SCHEDULING / "repeatability_admission.json", SCHEDULING / "audit_protocol.json"]
    registration = dict(configuration=configuration, original_registration=original,
                        source_sha256={str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
                                       for path in paths})
    registration_path = SCHEDULING / "registration.json"
    if registration_path.exists():
        assert json.loads(registration_path.read_text()) == registration
    else:
        scheduler_base.atomic_json(registration_path, registration)
    scheduler = CalibratedStudyScheduler(configuration)
    signal.signal(signal.SIGTERM, scheduler.request_stop)
    signal.signal(signal.SIGINT, scheduler.request_stop)
    try:
        scheduler.execute(None)
    except scheduler_base.QueuePaused as error:
        if scheduler.cpu_job is not None:
            scheduler.cpu_job.process.wait()
            scheduler.cpu_job.log.close()
            scheduler.cpu_job = None
        scheduler_base.atomic_json(SCHEDULING / "paused.json", dict(reason=str(error), time=time.time()))
        scheduler.publish("paused")
    except Exception as error:
        scheduler.mark_training_pause()
        scheduler_base.atomic_json(SCHEDULING / "failed.json", dict(error=repr(error), time=time.time()))
        scheduler.publish("failed_saving_active_jobs")
        while any(job.running() for job in scheduler.gpu_jobs):
            time.sleep(2)
        if scheduler.cpu_job is not None:
            scheduler.cpu_job.process.wait()
            scheduler.cpu_job.log.close()
            scheduler.cpu_job = None
        scheduler.publish("failed")
        raise


if __name__ == "__main__":
    main()
