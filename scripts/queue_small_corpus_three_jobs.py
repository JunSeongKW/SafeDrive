"""Run dependency-ready study jobs after bounded numerical/concurrency audits."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import statistics
import time

import queue_four_model_small_corpus_overlap_launch_fix as scheduler_base
from reassess_small_corpus_parallel_execution import maximum_training_difference, planner_prediction_difference

ROOT = scheduler_base.ROOT
OUTPUT = scheduler_base.OUTPUT
DIRECTORY = OUTPUT / "scheduling_v4_three_jobs"


def elapsed_rows(directory):
    return [json.loads(line) for line in (directory / "training_rank0.jsonl").read_text().splitlines()]


def median_step(directory):
    return max(statistics.median(row["seconds"] for row in scheduler_base.timing_rows(directory, rank)[1:])
               for rank in (0, 1))


class ThreeJobScheduler(scheduler_base.StudyScheduler):
    def publish(self, status="running"):
        super().publish(status)
        state = json.loads((DIRECTORY / "state.json").read_text())
        state["scheduler"] = "three_jobs_v4"
        scheduler_base.atomic_json(DIRECTORY / "state.json", state)
        scheduler_base.atomic_json(OUTPUT / "queue_state.json", state)

    def reap_finished(self):
        for job in list(self.gpu_jobs):
            if job.running():
                continue
            if job.log:
                job.log.close()
            self.gpu_jobs.remove(job)
            if job.returncode() not in (None, 0) or not job.expected.exists():
                raise RuntimeError("Training did not complete: " + job.label)

    def add_training(self, kind, checkpoint=None):
        directory = OUTPUT / kind
        return self.launch("train_" + kind, self.command(kind, directory, checkpoint), directory)

    def assess_third_job(self, baseline, concurrent, completed, other_job_start_rows):
        if not completed:
            return dict(third_job_approved=False, reason="Concurrent disposable profile did not complete")
        protocol = json.loads((DIRECTORY / "protocol.json").read_text())
        loss_difference = maximum_training_difference(baseline, concurrent, "loss")
        gradient_difference = maximum_training_difference(baseline, concurrent, "gradient_norm")
        predictions = planner_prediction_difference(baseline, concurrent)
        baseline_step, concurrent_step = median_step(baseline), median_step(concurrent)
        other_speeds = {}
        for kind, start_rows in other_job_start_rows.items():
            rows = elapsed_rows(OUTPUT / kind)[start_rows:]
            # Skip startup for a newly launched main training job.
            if len(rows) > 2:
                rows = rows[1:]
            other_speeds[kind] = dict(updates=len(rows), median_seconds=statistics.median(row["seconds"] for row in rows))
        first_pair = json.loads((OUTPUT / "scheduling_v2/jepa_ssl_and_drivor.admission.json").read_text())
        conservative_previous_block = max(first_pair["parallel_step_seconds"]) + baseline_step
        concurrent_block = max(concurrent_step, *(value["median_seconds"] for value in other_speeds.values()))
        block_speedup = conservative_previous_block / concurrent_block
        peak = max(row["card_used_bytes"] for rank in (0, 1) for row in scheduler_base.timing_rows(concurrent, rank))
        checks = dict(loss_bounded=loss_difference <= protocol["loss_relative_cap"],
                      gradient_bounded=gradient_difference <= protocol["gradient_relative_cap"],
                      predictions_bounded=predictions["mean_displacement_m"] <= protocol["trajectory_mean_cap_m"]
                        and predictions["maximum_displacement_m"] <= protocol["trajectory_maximum_cap_m"]
                        and predictions["maximum_heading_rad"] <= protocol["trajectory_heading_cap_rad"],
                      memory_safe=peak <= protocol["profile_card_ceiling_bytes"],
                      lpwm_not_excessively_slowed=concurrent_step <= baseline_step * protocol["lpwm_slowdown_cap"],
                      work_block_faster=block_speedup >= protocol["minimum_block_speedup"],
                      other_jobs_progressing=all(value["updates"] >= 3 for value in other_speeds.values()))
        return dict(third_job_approved=all(checks.values()), checks=checks, loss_relative_difference=loss_difference,
                    gradient_relative_difference=gradient_difference, prediction_difference=predictions,
                    lpwm_baseline_seconds=baseline_step, lpwm_three_job_seconds=concurrent_step,
                    other_job_speeds=other_speeds, estimated_equal_update_block_speedup=block_speedup,
                    peak_card_bytes=peak,
                    scope="LPWM baseline already overlaps JEPA; current three-job throughput versus earlier pair then LPWM. Short scheduling estimate, not guaranteed full-run speedup or equal final PDMS")

    def execute_with_adopted_jepa(self, adopted):
        assert scheduler_base.original_queue.gate_lpwm()
        self.gpu_jobs.append(adopted)
        checkpoint = OUTPUT / "lpwm_ssl/latest.pt"
        self.phase = "profile_lpwm_with_jepa"
        baseline = DIRECTORY / "profiles/lpwm_with_jepa"
        baseline_job = self.launch("profile_lpwm_with_jepa", self.command("lpwm_sequential", baseline, checkpoint, 8), baseline)
        baseline_completed = self.wait_jobs([baseline_job], allow_profile_failure=True)
        self.phase = "jepa_ssl_and_drivor_parallel"
        self.add_training("drivor")
        if baseline_completed:
            self.phase = "profile_three_jobs"
            other_job_start_rows = {kind: len(elapsed_rows(OUTPUT / kind)) if (OUTPUT / kind / "training_rank0.jsonl").exists() else 0
                                    for kind in ("jepa_ssl", "drivor")}
            concurrent = DIRECTORY / "profiles/lpwm_with_jepa_and_drivor"
            profile = self.launch("profile_three_jobs", self.command("lpwm_sequential", concurrent, checkpoint, 8), concurrent)
            completed = self.wait_jobs([profile], allow_profile_failure=True)
            admission = self.assess_third_job(baseline, concurrent, completed, other_job_start_rows)
        else:
            admission = dict(third_job_approved=False, reason="LPWM disposable profile failed")
        scheduler_base.atomic_json(DIRECTORY / "third_job_admission.json", admission)
        if admission["third_job_approved"]:
            self.add_training("lpwm_sequential", checkpoint)
        self.phase = "dependency_ready_planning"
        jepa_gate_checked = False
        while True:
            if self.stop_requested or (OUTPUT / "pause.requested").exists():
                self.stop_requested = True
                self.mark_training_pause()
                self.wait_jobs(list(self.gpu_jobs))
                raise scheduler_base.QueuePaused("Requested pause completed")
            self.reap_finished()
            jepa_ready = (OUTPUT / "jepa_ssl/complete.json").exists()
            if jepa_ready and not jepa_gate_checked:
                jepa_gate_checked = True
                if not scheduler_base.original_queue.gate_jepa():
                    self.held_conditions["jepa"] = "Stage1 quality gate failed"
            desired = [("lpwm_sequential", checkpoint)]
            if jepa_ready and "jepa" not in self.held_conditions:
                desired.append(("jepa", OUTPUT / "jepa_ssl/latest.pt"))
            allowed = 3 if admission["third_job_approved"] else 2
            for kind, weights in desired:
                if len(self.gpu_jobs) >= allowed:
                    break
                if not (OUTPUT / kind / "complete.json").exists() and not any(job.label == "train_" + kind for job in self.gpu_jobs):
                    self.add_training(kind, weights)
            self.service_cpu_scoring()
            self.publish()
            if not self.gpu_jobs:
                break
            time.sleep(2)
        self.phase = "lpwm_joint_exclusive"
        self.wait_jobs([self.add_training("lpwm_joint")])
        self.phase = "finish_cpu_scoring"
        while True:
            self.service_cpu_scoring()
            remaining = [kind for kind in scheduler_base.PLANNING_KINDS if kind not in self.held_conditions
                         and any(not (OUTPUT / kind / "validation" / f"pass{epoch}.pdms.json").exists() for epoch in (1, 3, 5))]
            if not remaining and self.cpu_job is None:
                break
            if self.stop_requested or (OUTPUT / "pause.requested").exists():
                raise scheduler_base.QueuePaused("Scoring queue paused")
            if remaining and self.cpu_job is None:
                raise RuntimeError("Missing prediction artifacts after training: " + repr(remaining))
            self.publish()
            time.sleep(2)
        results = {kind: json.loads((OUTPUT / kind / "validation/pass5.pdms.json").read_text())
                   for kind in scheduler_base.PLANNING_KINDS if kind not in self.held_conditions}
        scheduler_base.atomic_json(OUTPUT / ("comparison_partial.json" if self.held_conditions else "comparison_complete.json"),
            dict(results=results, held_conditions=self.held_conditions, full_navtest=False, seed_count=1,
                 scheduler="three_jobs_v4", conclusion_scope="Small-data system trends; upstream pretraining differs"))
        self.publish("held_for_stage1_review" if self.held_conditions else "complete")


def main(arguments):
    DIRECTORY.mkdir(parents=True, exist_ok=True)
    scheduler_base.SCHEDULING = DIRECTORY
    original = scheduler_base.verify_original_sources()
    configuration = json.loads(scheduler_base.CONFIGURATION.read_text())
    configuration["maximum_concurrent_gpu_jobs"] = 3
    lock = (DIRECTORY / "controller.lock").open("w")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    revision = json.loads((DIRECTORY / "first_pair_admission_revision.json").read_text())
    assert revision["parallel_approved"]
    state = json.loads((OUTPUT / "queue_state.json").read_text())
    assert state["status"] == "running" and state["scheduler_pid"] == arguments.take_over_pid
    assert len(state["active_jobs"]) == 1 and state["active_jobs"][0]["label"] == "train_jepa_ssl"
    controller = scheduler_base.process_identity(arguments.take_over_pid)
    job_state = state["active_jobs"][0]
    training = scheduler_base.process_identity(job_state["pid"])
    assert controller and controller["uid"] == os.getuid() and "queue_small_corpus_calibrated_parallel.py" in controller["command"]
    assert training and training["uid"] == os.getuid() and "train_small_corpus_jepa_ssl.py" in training["command"]
    scheduler_base.atomic_json(DIRECTORY / "handover_before.json", dict(controller=controller, training=training,
                                 queue=state, jepa_progress=json.loads((OUTPUT / "jepa_ssl/progress.json").read_text())))
    # The old scheduler's SIGTERM handler pauses trainers. Kill only this verified
    # controller to retain its independently sessioned torchrun child unchanged.
    os.kill(controller["pid"], signal.SIGKILL)
    deadline = time.monotonic() + 15
    while scheduler_base.still_alive(controller):
        assert time.monotonic() < deadline
        time.sleep(.1)
    assert scheduler_base.still_alive(training)
    study_lock = (OUTPUT / "queue.lock").open("w")
    fcntl.flock(study_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    adopted = scheduler_base.Job("train_jepa_ssl", Path(job_state["directory"]), job_state["command"],
                                 OUTPUT / "jepa_ssl/complete.json", training)
    paths = [Path(__file__), Path(scheduler_base.__file__), ROOT / "scripts/reassess_small_corpus_parallel_execution.py",
             scheduler_base.CONFIGURATION, DIRECTORY / "protocol.json", DIRECTORY / "first_pair_admission_revision.json"]
    scheduler_base.atomic_json(DIRECTORY / "registration.json", dict(configuration=configuration,
        original_registration=original, source_sha256={str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}))
    scheduler = ThreeJobScheduler(configuration)
    signal.signal(signal.SIGTERM, scheduler.request_stop)
    signal.signal(signal.SIGINT, scheduler.request_stop)
    try:
        scheduler.execute_with_adopted_jepa(adopted)
    except scheduler_base.QueuePaused as error:
        if scheduler.cpu_job:
            scheduler.cpu_job.process.wait()
            scheduler.cpu_job.log.close()
            scheduler.cpu_job = None
        scheduler_base.atomic_json(DIRECTORY / "paused.json", dict(reason=str(error), time=time.time()))
        scheduler.publish("paused")
    except Exception as error:
        scheduler.mark_training_pause()
        scheduler_base.atomic_json(DIRECTORY / "failed.json", dict(error=repr(error), time=time.time()))
        scheduler.publish("failed_saving_active_jobs")
        while any(job.running() for job in scheduler.gpu_jobs):
            time.sleep(2)
        if scheduler.cpu_job:
            scheduler.cpu_job.process.wait()
            scheduler.cpu_job.log.close()
            scheduler.cpu_job = None
        scheduler.publish("failed")
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--take-over-pid", type=int, required=True)
    main(parser.parse_args())
