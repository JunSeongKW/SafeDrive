"""Resume the registered overlap scheduler after a launch-metadata failure.

Identical scheduling and scientific commands to overlap_v2. Only launch
metadata merges and this runner's separate registration filename differ.
Original source, registration, failed.json and profile artifacts are preserved.
"""
import argparse
from dataclasses import dataclass
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import statistics
import subprocess
import time

import queue_four_model_small_corpus as original_queue

ROOT = original_queue.ROOT
OUTPUT = original_queue.OUTPUT
SCHEDULING = OUTPUT / 'scheduling_v2'
CONFIGURATION = ROOT / 'configs/four_model_small_corpus/scheduling_overlap_v2.json'
PLANNING_KINDS = ('drivor', 'lpwm_sequential', 'jepa', 'lpwm_joint')


def atomic_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_suffix(path.suffix + '.pending')
    pending.write_text(json.dumps(value, indent=2) + '\n')
    pending.replace(path)


def process_identity(pid):
    directory = Path('/proc') / str(pid)
    try:
        status = (directory / 'stat').read_text().rsplit(')', 1)[1].split()
        return dict(pid=pid, start_ticks=status[19], state=status[0],
                    uid=directory.stat().st_uid,
                    command=(directory / 'cmdline').read_bytes().replace(b'\0', b' ').decode())
    except (FileNotFoundError, ProcessLookupError):
        return None


def still_alive(identity):
    current = process_identity(identity['pid'])
    return current is not None and current['start_ticks'] == identity['start_ticks'] and current['state'] != 'Z'


def verify_original_sources():
    registration = json.loads((OUTPUT / 'queue_registration.json').read_text())
    changed = [name for name, expected in registration['source_sha256'].items()
               if hashlib.sha256((ROOT / name).read_bytes()).hexdigest() != expected]
    if changed:
        raise RuntimeError('Registered scientific sources changed: ' + repr(changed))
    return registration


def prediction_ready(directory, epoch):
    path = directory / 'validation' / f'pass{epoch}.npz'
    try:
        # The trainer writes this metadata AFTER closing the NPZ file.
        metadata = json.loads(path.with_suffix('.json').read_text())
        return path.is_file() and metadata['count'] == 1024
    except (FileNotFoundError, KeyError, json.JSONDecodeError):
        return False


def timing_rows(directory, rank=0):
    return [json.loads(line) for line in (directory / f'training_rank{rank}.jsonl').read_text().splitlines()]


def compare_pair_profiles(serial_paths, parallel_paths, configuration):
    serial = [timing_rows(path, rank) for path in serial_paths for rank in (0, 1)]
    parallel = [timing_rows(path, rank) for path in parallel_paths for rank in (0, 1)]
    steps = configuration['pair_profile_updates']
    complete = all(len(rows) == steps for rows in serial + parallel)
    if not complete:
        return dict(parallel_approved=False, reason='Incomplete disposable profile')
    serial_ranks = [statistics.median(row['seconds'] for row in rows[1:]) for rows in serial]
    parallel_ranks = [statistics.median(row['seconds'] for row in rows[1:]) for rows in parallel]
    serial_seconds = [max(serial_ranks[index:index+2]) for index in (0, 2)]
    parallel_seconds = [max(parallel_ranks[index:index+2]) for index in (0, 2)]
    speedup = sum(serial_seconds) / max(parallel_seconds)
    windows = []
    for path, rows in zip(parallel_paths, (parallel[0], parallel[2])):
        finished = (path / 'progress.json').stat().st_mtime
        windows.append((finished-rows[-1]['elapsed_seconds'], finished))
    overlap_seconds = max(0., min(value[1] for value in windows)-max(value[0] for value in windows))
    overlap_fraction = overlap_seconds/min(value[1]-value[0] for value in windows)
    elapsed_speedup = sum(serial[index][-1]['elapsed_seconds'] for index in (0, 2)) / (
        max(value[1] for value in windows)-min(value[0] for value in windows))
    loss_matches = all(math.isclose(left['loss'], right['loss'],
        rel_tol=configuration['parallel_loss_relative_tolerance'],
        abs_tol=configuration['parallel_loss_absolute_tolerance'])
        for serial_rows, parallel_rows in zip(serial, parallel)
        for left, right in zip(serial_rows, parallel_rows))
    peak_bytes = max(row['card_used_bytes'] for rows in parallel for row in rows)
    checks = dict(faster=speedup >= configuration['minimum_parallel_speedup'],
                  elapsed_training_faster=elapsed_speedup > 1.,
                  actual_overlap=overlap_fraction >= configuration['minimum_profile_training_overlap'],
                  loss_matches=loss_matches,
                  memory_safe=peak_bytes <= configuration['profile_card_ceiling_bytes'])
    return dict(parallel_approved=all(checks.values()), checks=checks,
        projected_pair_speedup=speedup, serial_step_seconds=serial_seconds,
        measured_training_window_speedup=elapsed_speedup, training_overlap_fraction=overlap_fraction,
        parallel_step_seconds=parallel_seconds, parallel_card_peak_bytes=peak_bytes,
        measurement_scope='Discard first update startup; eight-update pilot, not guaranteed full-run speedup')


@dataclass
class Job:
    label: str
    directory: Path
    command: list
    expected: Path
    identity: dict
    process: object = None
    log: object = None
    cpu: bool = False

    def running(self):
        return self.process.poll() is None if self.process is not None else still_alive(self.identity)

    def returncode(self):
        return self.process.poll() if self.process is not None else None


class QueuePaused(Exception):
    pass


class StudyScheduler:
    def __init__(self, configuration):
        self.configuration = configuration
        self.phase = 'initializing'
        self.gpu_jobs = []
        self.cpu_job = None
        self.stop_requested = False
        self.evaluation_started = {}
        self.held_conditions = {}

    def request_stop(self, *_arguments):
        self.stop_requested = True

    def publish(self, status='running'):
        jobs = self.gpu_jobs + ([self.cpu_job] if self.cpu_job else [])
        value = dict(status=status, scheduler_pid=os.getpid(), scheduler='overlap_v2',
            phase=self.phase, stage=self.phase,
            pid=self.gpu_jobs[0].identity['pid'] if self.gpu_jobs else None,
            updated_unix=time.time(), held_conditions=self.held_conditions,
            active_jobs=[dict(label=job.label, pid=job.identity['pid'], cpu=job.cpu,
                directory=str(job.directory), command=job.command) for job in jobs if job.running()])
        atomic_json(SCHEDULING / 'state.json', value)
        atomic_json(OUTPUT / 'queue_state.json', value)

    def launch(self, label, command, directory, expected=None, cpu=False):
        verify_original_sources()
        expected = expected or directory / 'complete.json'
        if expected.exists():
            return None
        if self.stop_requested or (OUTPUT / 'pause.requested').exists():
            raise QueuePaused('Study pause requested')
        if not cpu and (directory / 'pause.requested').exists():
            raise QueuePaused('Job pause marker exists: ' + str(directory))
        environment = os.environ.copy()
        environment.update(CUDA_VISIBLE_DEVICES='' if cpu else '0,1', OMP_NUM_THREADS='2',
            OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1', NCCL_P2P_DISABLE='1', PYTHONUNBUFFERED='1')
        if cpu:
            environment['LD_LIBRARY_PATH'] = str(original_queue.CPU_PYTHON.parent.parent / 'lib')
        directory.mkdir(parents=True, exist_ok=True)
        log = (SCHEDULING / (label + '.log')).open('a')
        child = subprocess.Popen([str(value) for value in command], cwd=ROOT, env=environment,
                                 stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        identity = process_identity(child.pid)
        assert identity is not None
        job = Job(label, directory, [str(value) for value in command], expected, identity, child, log, cpu)
        atomic_json(SCHEDULING / (label + '.launch.json'), {**identity, 'process_command': identity['command'], 'command': job.command,
            'expected': str(expected), 'started_unix': time.time()})
        if cpu:
            assert self.cpu_job is None
            self.cpu_job = job
        else:
            assert len(self.gpu_jobs) < self.configuration['maximum_concurrent_gpu_jobs']
            self.gpu_jobs.append(job)
        self.publish()
        return job

    def service_cpu_scoring(self):
        if self.cpu_job and not self.cpu_job.running():
            job = self.cpu_job
            job.log.close()
            self.cpu_job = None
            if job.returncode() != 0 or not job.expected.exists():
                raise RuntimeError('Official CPU scoring failed: ' + job.label)
            atomic_json(SCHEDULING / (job.label + '.timing.json'), dict(
                wall_seconds=time.time()-self.evaluation_started[job.label], completed=True))
        if self.cpu_job or self.stop_requested or (OUTPUT / 'pause.requested').exists():
            return
        for epoch in (1, 3, 5):
            for kind in PLANNING_KINDS:
                directory = OUTPUT / kind
                prediction = directory / 'validation' / f'pass{epoch}.npz'
                expected = prediction.with_suffix('.pdms.json')
                if prediction_ready(directory, epoch) and not expected.exists():
                    label = f'score_{kind}_pass{epoch}'
                    self.evaluation_started[label] = time.time()
                    self.launch(label, ['nice', '-n', str(self.configuration['cpu_evaluation_nice']),
                        original_queue.CPU_PYTHON, '-u', ROOT / 'scripts/score_four_model_small_corpus.py',
                        '--predictions', prediction, '--workers', str(self.configuration['evaluation_workers'])],
                        directory, expected, cpu=True)
                    return

    def mark_training_pause(self):
        for job in self.gpu_jobs:
            marker = job.directory / 'pause.requested'
            if job.running() and not marker.exists():
                atomic_json(marker, dict(owner='four_model_overlap_v2', scheduler_pid=os.getpid(),
                    reason='Scheduler pause or failure: save optimizer and RNG before exiting'))

    def wait_jobs(self, jobs, allow_profile_failure=False):
        jobs = [job for job in jobs if job is not None]
        while any(job.running() for job in jobs):
            if self.stop_requested or (OUTPUT / 'pause.requested').exists():
                self.stop_requested = True
                self.mark_training_pause()
            failed = [job for job in jobs if not job.running()
                      and (job.returncode() not in (None, 0) or not job.expected.exists())]
            if failed and not allow_profile_failure and not self.stop_requested:
                self.mark_training_pause()
                raise RuntimeError('Job failed; saving active companions: ' + ', '.join(job.label for job in failed))
            self.service_cpu_scoring()
            self.publish('pausing' if self.stop_requested else 'running')
            time.sleep(2)
        passed = True
        for job in jobs:
            if job.log:
                job.log.close()
            if job in self.gpu_jobs:
                self.gpu_jobs.remove(job)
            passed &= job.returncode() in (None, 0) and job.expected.exists()
        if self.stop_requested:
            raise QueuePaused('All running training jobs saved and paused')
        if not passed and not allow_profile_failure:
            raise RuntimeError('Training failed or paused: ' + ', '.join(job.label for job in jobs))
        return passed

    def command(self, kind, destination, checkpoint_path=None, profile_updates=0):
        if kind == 'jepa_ssl':
            arguments = ['--output', destination, '--micro-batch', '2']
            if profile_updates:
                arguments += ['--profile', '--updates', str(profile_updates)]
            return original_queue.distributed_command('train_small_corpus_jepa_ssl.py', arguments)
        arguments = ['--kind', kind, '--output', destination, '--micro-batch', '2']
        if checkpoint_path is not None:
            arguments += ['--stage1-checkpoint', checkpoint_path]
        if profile_updates:
            arguments += ['--profile-updates', str(profile_updates)]
        return original_queue.distributed_command('train_small_corpus_common_planner.py', arguments)

    def pair_admission(self, name, conditions):
        decision_path = SCHEDULING / (name + '.admission.json')
        if decision_path.exists():
            return json.loads(decision_path.read_text())['parallel_approved']
        self.phase = 'profile_' + name
        serial_paths, parallel_paths = [], []
        steps = self.configuration['pair_profile_updates']
        for kind, checkpoint_path in conditions:
            destination = SCHEDULING / 'profiles' / (name + '_serial_' + kind)
            serial_paths.append(destination)
            self.wait_jobs([self.launch(name+'_serial_'+kind,
                self.command(kind, destination, checkpoint_path, steps), destination)])
        jobs = []
        # After an interrupted profiling attempt, both jobs must run together
        # in fresh directories; a cached completed companion cannot prove overlap.
        attempt = 1
        while any((SCHEDULING / 'profiles' / f'{name}_parallel{attempt}_{kind}').exists()
                  for kind, _ in conditions):
            attempt += 1
        for kind, checkpoint_path in conditions:
            destination = SCHEDULING / 'profiles' / f'{name}_parallel{attempt}_{kind}'
            parallel_paths.append(destination)
            jobs.append(self.launch(name+'_parallel_'+kind,
                self.command(kind, destination, checkpoint_path, steps), destination))
        parallel_completed = self.wait_jobs(jobs, allow_profile_failure=True)
        decision = compare_pair_profiles(serial_paths, parallel_paths, self.configuration) if parallel_completed else dict(
            parallel_approved=False, reason='Concurrent profile failed; isolated profiles passed; retain serial execution')
        decision.update(serial_profiles=[str(path) for path in serial_paths],
                        parallel_profiles=[str(path) for path in parallel_paths])
        atomic_json(decision_path, decision)
        return decision['parallel_approved']

    def run_pair_or_serial(self, name, conditions):
        pending = [(kind, checkpoint_path) for kind, checkpoint_path in conditions
                   if not (OUTPUT / kind / 'complete.json').exists()]
        parallel = len(pending) == 2 and self.pair_admission(name, pending)
        self.phase = name + ('_parallel' if parallel else '_serial')
        jobs = []
        for kind, checkpoint_path in pending:
            directory = OUTPUT / kind
            job = self.launch('train_'+kind, self.command(kind, directory, checkpoint_path), directory)
            if parallel:
                jobs.append(job)
            else:
                self.wait_jobs([job])
        self.wait_jobs(jobs)

    def execute(self, adopted):
        self.phase = 'lpwm_ssl'
        if adopted is not None:
            self.gpu_jobs.append(adopted)
            self.wait_jobs([adopted])
        else:
            directory = OUTPUT / 'lpwm_ssl'
            command = original_queue.distributed_command('train_small_corpus_lpwm_ssl.py', [
                '--output', directory, '--micro-batch', '4', '--accumulation', '2', '--epochs', '5', '--workers', '4'],
                original_queue.SSL_PYTHON)
            self.wait_jobs([self.launch('lpwm_ssl', command, directory)])
        lpwm_ready = original_queue.gate_lpwm()
        if not lpwm_ready:
            self.held_conditions['lpwm_sequential'] = 'Stage1 quality gate failed'
        self.run_pair_or_serial('jepa_ssl_and_drivor', [('jepa_ssl', None), ('drivor', None)])
        jepa_ready = original_queue.gate_jepa()
        if not jepa_ready:
            self.held_conditions['jepa'] = 'Stage1 quality gate failed'
        conditions = []
        if lpwm_ready:
            conditions.append(('lpwm_sequential', OUTPUT / 'lpwm_ssl/latest.pt'))
        if jepa_ready:
            conditions.append(('jepa', OUTPUT / 'jepa_ssl/latest.pt'))
        self.run_pair_or_serial('lpwm_and_jepa_planning', conditions)
        self.phase = 'lpwm_joint_exclusive'
        directory = OUTPUT / 'lpwm_joint'
        self.wait_jobs([self.launch('train_lpwm_joint', self.command('lpwm_joint', directory), directory)])
        self.phase = 'finish_cpu_scoring'
        while True:
            self.service_cpu_scoring()
            remaining = [kind for kind in PLANNING_KINDS if kind not in self.held_conditions
                         and any(not (OUTPUT / kind / 'validation' / f'pass{epoch}.pdms.json').exists()
                                 for epoch in (1, 3, 5))]
            if not remaining and self.cpu_job is None:
                break
            if self.stop_requested or (OUTPUT / 'pause.requested').exists():
                raise QueuePaused('Scoring queue paused')
            if remaining and self.cpu_job is None:
                raise RuntimeError('Missing prediction artifacts after training: ' + repr(remaining))
            self.publish()
            time.sleep(2)
        results = {kind: json.loads((OUTPUT / kind / 'validation/pass5.pdms.json').read_text())
                   for kind in PLANNING_KINDS if kind not in self.held_conditions}
        atomic_json(OUTPUT / ('comparison_partial.json' if self.held_conditions else 'comparison_complete.json'),
            dict(results=results, held_conditions=self.held_conditions, full_navtest=False, seed_count=1,
                 scheduler='overlap_v2', conclusion_scope='Small-data system trends; upstream pretraining differs'))
        self.publish('held_for_stage1_review' if self.held_conditions else 'complete')


def takeover(previous_pid):
    previous = process_identity(previous_pid)
    assert previous and previous['uid'] == os.getuid()
    assert 'scripts/queue_four_model_small_corpus.py' in previous['command'], previous
    state = json.loads((OUTPUT / 'queue_state.json').read_text())
    assert state['stage'] == 'lpwm_ssl', state
    child = process_identity(state['pid'])
    assert child and child['uid'] == os.getuid() and 'train_small_corpus_lpwm_ssl.py' in child['command'], child
    atomic_json(SCHEDULING / 'previous_controller.json', dict(controller=previous, active_training=child, state=state))
    # Signal only the verified controller PID. Its torchrun child keeps training.
    os.kill(previous_pid, signal.SIGTERM)
    deadline = time.monotonic() + 15
    while still_alive(previous):
        assert time.monotonic() < deadline, 'Old controller did not exit'
        time.sleep(.1)
    assert still_alive(child), 'Training unexpectedly stopped during controller handover'
    return Job('lpwm_ssl', OUTPUT / 'lpwm_ssl', state['command'], OUTPUT / 'lpwm_ssl/complete.json', child)


def main(arguments):
    SCHEDULING.mkdir(parents=True, exist_ok=True)
    configuration = json.loads(CONFIGURATION.read_text())
    original = verify_original_sources()
    # A separate v2 lock prevents two controllers trying to take over together.
    controller_lock = (SCHEDULING / 'controller.lock').open('w')
    fcntl.flock(controller_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    registration = dict(configuration=configuration, original_registration=original,
        scheduler_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        configuration_sha256=hashlib.sha256(CONFIGURATION.read_bytes()).hexdigest())
    registration_path = SCHEDULING / 'registration_launch_fix.json'
    if registration_path.exists():
        assert json.loads(registration_path.read_text()) == registration
    else:
        atomic_json(registration_path, registration)
    adopted = takeover(arguments.take_over_pid) if arguments.take_over_pid else None
    study_lock = (OUTPUT / 'queue.lock').open('w')
    fcntl.flock(study_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    scheduler = StudyScheduler(configuration)
    signal.signal(signal.SIGTERM, scheduler.request_stop)
    signal.signal(signal.SIGINT, scheduler.request_stop)
    try:
        scheduler.execute(adopted)
    except QueuePaused as error:
        if scheduler.cpu_job is not None:
            scheduler.cpu_job.process.wait()
            scheduler.cpu_job.log.close()
            scheduler.cpu_job = None
        atomic_json(SCHEDULING / 'paused.json', dict(reason=str(error), time=time.time()))
        scheduler.publish('paused')
    except Exception as error:
        scheduler.mark_training_pause()
        atomic_json(SCHEDULING / 'failed.json', dict(error=repr(error), time=time.time()))
        scheduler.publish('failed_saving_active_jobs')
        while any(job.running() for job in scheduler.gpu_jobs):
            time.sleep(2)
        if scheduler.cpu_job is not None:
            scheduler.cpu_job.process.wait()
            scheduler.cpu_job.log.close()
            scheduler.cpu_job = None
        scheduler.publish('failed')
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--take-over-pid', type=int)
    main(parser.parse_args())
