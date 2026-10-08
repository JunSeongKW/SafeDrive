"""Adopt live jobs; consider larger physical batches after a peer completes."""
import argparse
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import statistics
import threading
import time

import queue_small_corpus_user_48gb as previous

base = previous.base
guarded = previous.guarded
recovery = previous.recovery
DIRECTORY = base.OUTPUT / "scheduling_v10_batch_growth"
CARD_LIMIT = 48_000_000_000
PROFILE_UPDATES = 8


def candidate_peak_bytes(previous_peak, previous_microbatch, candidate_microbatch):
    # Conservative linear activation estimate; reserve an extra10% for shape
    # workspaces. Profiles still measure actual memory, including doubled SSL.
    fixed_bytes = 2_000_000_000
    return int((fixed_bytes + max(0, previous_peak - fixed_bytes)
                * candidate_microbatch / previous_microbatch) * 1.10)


def profile_summary(directory, expected_stop):
    completion = directory / "complete.json"
    if not completion.exists():
        return dict(passed=False, reason="Profile did not complete")
    result = json.loads(completion.read_text())
    if result.get("completed_updates") != expected_stop or not result.get("disposable_profile"):
        return dict(passed=False, reason="Incorrect disposable completion")
    per_rank = []
    for rank in (0, 1):
        rows = base.timing_rows(directory, rank)
        if len(rows) != PROFILE_UPDATES or rows[-1]["completed_updates"] != expected_stop:
            return dict(passed=False, reason="Missing rank updates")
        if any(not math.isfinite(row[key]) for row in rows for key in ("loss", "gradient_norm", "total_objective")):
            return dict(passed=False, reason="Nonfinite loss or gradient")
        per_rank.append(dict(seconds=statistics.median(row["seconds"] for row in rows[1:-1]),
            peak_allocated_bytes=max(row["peak_allocated_bytes"] for row in rows),
            card_peak_bytes=max(row["card_used_bytes"] for row in rows)))
    peak = max(row["card_peak_bytes"] for row in per_rank)
    return dict(passed=peak < CARD_LIMIT, seconds=max(row["seconds"] for row in per_rank),
        card_peak_bytes=peak, allocator_peak_bytes=max(row["peak_allocated_bytes"] for row in per_rank),
        final_profile_update_exercises_doubled_ssl=True, weights_discarded=True)


class BatchGrowthScheduler(previous.UserMemoryLimitScheduler):
    def __init__(self, configuration):
        super().__init__(configuration)
        self.seen_complete = set()
        self.pending_growth = False
        self.current_microbatch = {kind: 2 for kind in base.PLANNING_KINDS}
        self.batch_cap = {}
        self.scaled_resume_kinds = set()
        self.growth_trials = []
        self.automatic_native_allowed = False

    def publish(self, status="running"):
        super().publish(status)
        value = json.loads((DIRECTORY / "state.json").read_text())
        value.update(scheduler="batch_growth_v10", phase=self.phase, stage=self.phase,
            batch_growth=dict(trigger="A previously active experiment finishes",
                pending=self.pending_growth, physical_microbatch_per_gpu=self.current_microbatch,
                accumulation={kind: 8 // micro for kind, micro in self.current_microbatch.items()},
                effective_batch=16, candidates=[4, 8], trials=self.growth_trials,
                original_science_registration_kept=True,
                numerical_equivalence_claimed=False), temporary_checkpoint_restart=False)
        base.atomic_json(DIRECTORY / "state.json", value)
        base.atomic_json(base.OUTPUT / "queue_state.json", value)

    def scaled_arguments(self, kind, destination, micro, allocator_cap, stop=0):
        arguments = ["--kind", kind, "--output", destination, "--micro-batch", "2",
                     "--execution-microbatch", str(micro), "--allocator-cap-bytes", str(allocator_cap)]
        if kind == "lpwm_sequential":
            arguments += ["--stage1-checkpoint", base.OUTPUT / "lpwm_ssl/latest.pt"]
        if stop:
            arguments += ["--stop-after-update", str(stop)]
        return arguments

    def launch_scaled(self, label, kind, directory, micro, allocator_cap, stop=0):
        before = os.environ.get("PYTORCH_CUDA_ALLOC_CONF")
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
        log_path = DIRECTORY / (label + ".log")
        log_start = log_path.stat().st_size if log_path.exists() else 0
        try:
            job = self.launch(label, base.original_queue.distributed_command(
                "resume_small_corpus_scaled_batch.py",
                self.scaled_arguments(kind, directory, micro, allocator_cap, stop)), directory)
            if job is not None:
                job.execution_log_start = log_start
            return job
        finally:
            if before is None:
                os.environ.pop("PYTORCH_CUDA_ALLOC_CONF", None)
            else:
                os.environ["PYTORCH_CUDA_ALLOC_CONF"] = before

    def launch_resume(self, kind):
        micro = self.current_microbatch[kind]
        if micro > 2 or kind in self.scaled_resume_kinds:
            return self.launch_scaled("train_" + kind, kind, base.OUTPUT / kind, micro,
                                      self.batch_cap[kind])
        return super().launch_resume(kind)

    def check_user_pause(self):
        if self.stop_requested or (base.OUTPUT / "pause.requested").exists():
            self.stop_requested = True
            self.mark_training_pause()
            raise base.QueuePaused("User pause honored during batch selection")

    def finish_profile(self, job, stop_update):
        deadline = time.monotonic() + 1800
        while job.running():
            self.check_user_pause()
            if time.monotonic() >= deadline:
                self.stop_disposable_job(job)
                return dict(passed=False, reason="Disposable profile exceeded30minutes")
            self.service_cpu_scoring()
            self.publish("profiling_batch_growth")
            time.sleep(2)
        if job.log:
            job.log.close()
        self.gpu_jobs.remove(job)
        if job.returncode() not in (None, 0):
            return dict(passed=False, reason="Disposable profile failed; original checkpoint untouched")
        result = profile_summary(job.directory, stop_update)
        telemetry = DIRECTORY / "memory_samples.jsonl"
        if telemetry.exists():
            observed = [max(row["card_bytes"]) for line in telemetry.read_text().splitlines()
                        if line.strip() for row in [json.loads(line)] if job.label in row["active"]]
            if observed:
                result["continuous_card_peak_bytes"] = max(observed)
                result["passed"] = result["passed"] and max(observed) < CARD_LIMIT
        return result

    def stop_disposable_job(self, job):
        assert job.label.startswith("profile_") and job.directory.is_relative_to(DIRECTORY / "batch_trials")
        if job.running():
            base.atomic_json(job.directory / "pause.requested", dict(scheduler_pid=os.getpid(),
                owner="batch_growth_v10", reason="End disposable timing probe"))
            deadline = time.monotonic() + 45
            while job.running() and time.monotonic() < deadline:
                time.sleep(1)
            if job.running():
                identity = base.process_identity(job.identity["pid"])
                assert identity and identity["uid"] == os.getuid() and base.still_alive(job.identity)
                assert "resume_small_corpus_scaled_batch.py" in identity["command"]
                os.killpg(job.identity["pid"], signal.SIGKILL)
                job.process.wait(timeout=30)
        if job.log:
            job.log.close()
        if job in self.gpu_jobs:
            self.gpu_jobs.remove(job)

    def profile_from_snapshot(self, kind, snapshot, destination, micro, cap, completed):
        destination.mkdir(parents=True)
        for name in ("latest.pt", "registration.json"):
            os.link(snapshot / name, destination / name)
        job = self.launch_scaled(f"profile_{kind}_{completed}_micro{micro}", kind, destination,
                                 micro, cap, completed + PROFILE_UPDATES)
        return self.finish_profile(job, completed + PROFILE_UPDATES)

    def attempt_growth(self, job):
        kind = job.label.removeprefix("train_")
        current = self.current_microbatch[kind]
        progress = json.loads((job.directory / "progress.json").read_text())
        candidates = [micro for micro in (4, 8) if micro > current]
        # Protect registered companion peaks, rather than treating an idle
        # CUDA allocation trough as permanently available VRAM.
        companions = [other.label.removeprefix("train_") for other in self.gpu_jobs if other is not job]
        companion_reservation = 2_000_000_000 + sum(recovery.RESERVATIONS[name] for name in companions)
        predicted = {micro: candidate_peak_bytes(progress["peak_allocated_bytes"], current, micro)
                     for micro in candidates}
        feasible = [micro for micro in candidates if predicted[micro] + companion_reservation < CARD_LIMIT]
        trial = dict(kind=kind, from_microbatch=current, predicted_peak_bytes=predicted,
            companion_reservation_bytes=companion_reservation, candidate_microbatches=feasible,
            time=time.time(), completed_updates_before_request=progress["completed_updates"])
        self.growth_trials.append(trial)
        if not feasible:
            trial.update(result="No larger batch fits the protected memory budget; continue current batch")
            return
        self.check_user_pause()
        marker = job.directory / "pause.requested"
        assert not marker.exists(), "Preserve external pause request"
        base.atomic_json(marker, dict(owner="batch_growth_v10", scheduler_pid=os.getpid(),
            reason="Save full state for user-authorized batch throughput measurement"))
        self.phase = "save_candidate_for_batch_profile"
        while job.running():
            self.check_user_pause()
            self.service_cpu_scoring()
            self.publish("saving_batch_candidate")
            time.sleep(2)
        assert job.returncode() in (None, 0) and (job.directory / "paused.json").exists()
        completed = json.loads((job.directory / "paused.json").read_text())["completed_updates"]
        history = DIRECTORY / "batch_trials" / kind / str(completed)
        snapshot = history / "original_saved_state"
        snapshot.mkdir(parents=True)
        for name in ("latest.pt", "registration.json", "progress.json"):
            os.link(job.directory / name, snapshot / name)
        recovery.archive_markers(job.directory, snapshot / "markers", {os.getpid()})
        self.gpu_jobs.remove(job)
        if job.log:
            job.log.close()
        # A concurrently finishing peer may have freed more memory, but no new
        # companion is admitted until this sequential selection returns.
        observed_other = max(recovery.current_card_bytes())
        protected_other = max(observed_other, companion_reservation)
        allocator_cap = min(44_000_000_000, CARD_LIMIT - protected_other - 500_000_000)
        selected = current
        measurements = {}
        try:
            self.phase = "disposable_batch_throughput_profile"
            measurements[current] = self.profile_from_snapshot(kind, snapshot, history / f"micro{current}",
                                                               current, allocator_cap, completed)
            if measurements[current]["passed"]:
                baseline = measurements[current]["seconds"]
                best_seconds = baseline
                for candidate in feasible:
                    if predicted[candidate] > allocator_cap:
                        measurements[candidate] = dict(passed=False, reason="Protected allocator budget too small")
                        continue
                    self.check_user_pause()
                    result = self.profile_from_snapshot(kind, snapshot, history / f"micro{candidate}",
                                                        candidate, allocator_cap, completed)
                    measurements[candidate] = result
                    if result["passed"] and result["seconds"] < best_seconds / 1.05:
                        selected = candidate
                        best_seconds = result["seconds"]
        except base.QueuePaused:
            raise
        except Exception as error:
            selected = current
            trial["profile_error"] = repr(error)
            for disposable in list(self.gpu_jobs):
                if disposable.label.startswith("profile_"):
                    self.stop_disposable_job(disposable)
        finally:
            # Profiles load hard-linked snapshots and write only to their own
            # directories. Main latest.pt/optimizer/RNG remain exactly saved.
            assert os.path.samefile(job.directory / "latest.pt", snapshot / "latest.pt")
            self.current_microbatch[kind] = selected
            self.batch_cap[kind] = allocator_cap
            if selected > 2:
                self.scaled_resume_kinds.add(kind)
            trial.update(saved_update=completed, selected_microbatch=selected, measurements=measurements,
                effective_batch=16, resume_original_checkpoint=True, profile_weights_discarded=True,
                numerical_or_pdms_equivalence_claimed=False)
            base.atomic_json(history / "decision.json", trial)
            self.check_user_pause()
            self.phase = "continue_after_batch_selection"
            self.launch_resume(kind)

    def reap_finished(self):
        for job in list(self.gpu_jobs):
            kind = job.label.removeprefix("train_")
            if kind not in self.scaled_resume_kinds or job.running() or job.expected.exists():
                continue
            paused = job.directory / "paused.json"
            log_path = DIRECTORY / (job.label + ".log")
            if log_path.exists():
                with log_path.open() as stream:
                    stream.seek(getattr(job, "execution_log_start", 0))
                    tail = stream.read()[-12000:]
            else:
                tail = ""
            memory_failure = "out of memory" in tail.lower() or "OutOfMemoryError" in tail
            if paused.exists():
                memory_failure |= json.loads(paused.read_text())["card_used_bytes"] > CARD_LIMIT
            if not memory_failure:
                continue
            history = DIRECTORY / "batch_rollbacks" / kind / str(time.time_ns())
            history.mkdir(parents=True)
            recovery.archive_markers(job.directory, history / "markers", {os.getpid()})
            if job.log:
                job.log.close()
            self.gpu_jobs.remove(job)
            previous_micro = self.current_microbatch[kind]
            self.current_microbatch[kind] = 2
            self.batch_cap[kind] = min(self.batch_cap[kind], 27_000_000_000 if kind == "lpwm_joint" else 12_000_000_000)
            base.atomic_json(history / "rollback.json", dict(from_microbatch=previous_micro,
                to_microbatch=2, reason="Memory pressure after batch growth; resume latest full state",
                companions_not_stopped=True, time=time.time()))
        super().reap_finished()

    def safe_resize_point(self, job):
        progress = json.loads((job.directory / "progress.json").read_text())
        completed = progress["completed_updates"]
        # Do not save/exit between an epoch checkpoint and its prediction pass.
        if completed > 3000 or not 16 <= completed % 640 <= 600:
            return False
        return all(base.prediction_ready(job.directory, epoch)
                   for epoch in (1, 3, 5) if epoch * 640 <= completed)

    def execute(self):
        monitor = threading.Thread(target=self.monitor_memory, daemon=True)
        monitor.start()
        try:
            while True:
                self.check_user_pause()
                self.reap_finished()
                completed = {kind for kind in base.PLANNING_KINDS if (base.OUTPUT / kind / "complete.json").exists()}
                newly_complete = completed - self.seen_complete
                if newly_complete:
                    self.pending_growth = True
                    self.seen_complete = completed
                remaining = [kind for kind in base.PLANNING_KINDS if kind not in completed]
                if not remaining:
                    break
                self.phase = "train_and_wait_for_peer_completion"
                if self.pending_growth and "jepa" in completed:
                    eligible = [job for job in list(self.gpu_jobs)
                                if job.label in ("train_lpwm_sequential", "train_lpwm_joint")]
                    if eligible and all(self.safe_resize_point(job) for job in eligible):
                        self.pending_growth = False
                        for job in eligible:
                            self.attempt_growth(job)
                        self.automatic_native_allowed = remaining == ["lpwm_joint"]
                for kind in ("jepa", "lpwm_sequential", "lpwm_joint"):
                    active = [job.label.removeprefix("train_") for job in self.gpu_jobs]
                    if kind in remaining and kind not in active and guarded.guarded_admission(
                            kind, active, max(recovery.current_card_bytes()), self.configuration["maximum_concurrent_gpu_jobs"]):
                        self.launch_resume(kind)
                if (remaining == ["lpwm_joint"] and self.automatic_native_allowed
                        and self.current_microbatch["lpwm_joint"] == 2 and not recovery.NATIVE_MARKER.exists()):
                    base.atomic_json(recovery.NATIVE_MARKER, dict(time=time.time(), scheduler_pid=os.getpid(),
                        all_other_registered_gpu_jobs_complete=True, per_rank_external_memory_guard_bytes=2_000_000_000))
                self.service_cpu_scoring()
                self.publish()
                time.sleep(2)
            self.phase = "finish_cpu_scoring"
            while True:
                self.check_user_pause()
                self.service_cpu_scoring()
                pending = [kind for kind in base.PLANNING_KINDS if any(
                    not (base.OUTPUT / kind / "validation" / f"pass{epoch}.pdms.json").exists() for epoch in (1, 3, 5))]
                if not pending and self.cpu_job is None:
                    break
                if pending and self.cpu_job is None:
                    raise RuntimeError("Missing prediction artifacts: " + repr(pending))
                self.publish()
                time.sleep(2)
            base.atomic_json(base.OUTPUT / "comparison_complete.json", dict(
                results={kind: json.loads((base.OUTPUT / kind / "validation/pass5.pdms.json").read_text())
                         for kind in base.PLANNING_KINDS}, full_navtest=False, seed_count=1,
                scheduler="batch_growth_v10", batch_growth_trials=self.growth_trials))
            self.publish("complete")
        finally:
            self.monitor_stop.set()
            monitor.join(timeout=5)


def main(arguments):
    DIRECTORY.mkdir(parents=True, exist_ok=True)
    previous.DIRECTORY = guarded.DIRECTORY = guarded.previous.DIRECTORY = DIRECTORY
    base.SCHEDULING = recovery.DIRECTORY = DIRECTORY
    guarded.JEPA_SAVE_PAUSE_BYTES = guarded.TRIAL_ADMISSION_BYTES = CARD_LIMIT
    recovery.can_admit = guarded.guarded_admission
    guarded.previous.priority_defers = lambda *_arguments: False
    original = base.verify_original_sources()
    old_registration = json.loads((base.OUTPUT / "scheduling_v9_user_48gb/registration.json").read_text())
    for name, expected in old_registration["source_sha256"].items():
        assert hashlib.sha256((base.ROOT / name).read_bytes()).hexdigest() == expected
    own_lock = (DIRECTORY / "controller.lock").open("w")
    fcntl.flock(own_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    assert not (base.OUTPUT / "pause.requested").exists()
    state = json.loads((base.OUTPUT / "queue_state.json").read_text())
    assert state["scheduler_pid"] == arguments.take_over_pid and state["status"] == "running"
    controller = base.process_identity(arguments.take_over_pid)
    assert controller and controller["uid"] == os.getuid()
    assert "queue_small_corpus_user_48gb.py" in controller["command"]
    scheduler = BatchGrowthScheduler(old_registration["configuration"].copy())
    scheduler.seen_complete = {kind for kind in base.PLANNING_KINDS if (base.OUTPUT / kind / "complete.json").exists()}
    scheduler.memory_peak_bytes = state.get("monitored_peak_card_bytes", [0, 0])
    scheduler.retry_below_count = state.get("jepa_retry_below_active_lpwm_count")
    adopted = []
    for entry in state["active_jobs"]:
        identity = base.process_identity(entry["pid"])
        assert identity and identity["uid"] == os.getuid() and base.still_alive(identity)
        directory = Path(entry["directory"])
        assert directory.parent == base.OUTPUT
        assert not (directory / "pause.requested").exists()
        if entry["cpu"]:
            assert "score_four_model_small_corpus.py" in identity["command"]
            prediction = Path(entry["command"][entry["command"].index("--predictions") + 1])
            expected = prediction.with_suffix(".pdms.json")
        else:
            assert "resume_small_corpus_user_memory_limit.py" in identity["command"]
            assert entry["label"] in ("train_jepa", "train_lpwm_sequential", "train_lpwm_joint")
            expected = directory / "complete.json"
        job = base.Job(entry["label"], directory, entry["command"], expected, identity, cpu=entry["cpu"])
        adopted.append(identity)
        if entry["cpu"]:
            scheduler.cpu_job = job
            scheduler.evaluation_started[job.label] = time.time()
        else:
            scheduler.gpu_jobs.append(job)
    assert len(scheduler.gpu_jobs) == 3
    paths = [Path(__file__), base.ROOT / "scripts/resume_small_corpus_scaled_batch.py"]
    sources = dict(old_registration["source_sha256"])
    sources.update({str(path.relative_to(base.ROOT)): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths})
    base.atomic_json(DIRECTORY / "registration.json", dict(configuration=scheduler.configuration,
        original_registration=original, previous_registration=old_registration, source_sha256=sources,
        user_request="Use freed VRAM after experiments finish to increase physical batch and speed up remaining training",
        effective_batch=16, candidate_physical_batches=[4, 8], minimum_measured_speedup=1.05,
        card_limit_bytes=CARD_LIMIT, disposable_profiles=True, stochastic_equivalence_claimed=False))
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
        adopted_jobs_alive=True, training_restart=False, adopted=adopted))
    signal.signal(signal.SIGTERM, scheduler.request_stop)
    signal.signal(signal.SIGINT, scheduler.request_stop)
    try:
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--take-over-pid", type=int, required=True)
    main(parser.parse_args())
