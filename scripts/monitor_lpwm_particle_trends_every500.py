"""Schedule exact checkpoint diagnostics and publish particle trends every 500 updates."""
import argparse
import csv
import ctypes
import fcntl
import html
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

import numpy as np
import torch

import monitor_lpwm_drivor_planning_path_representations as diagnostic
from visualize_lpwm_planning_path_geometry import geometry_changes

ROOT = Path(__file__).resolve().parents[1]


def read_json(path):
    return json.loads(path.read_text())


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_suffix(".pending.json")
    pending.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    pending.replace(path)


def load_configuration(path):
    configuration = read_json(path)
    for key in ("training_directory", "monitor_directory", "output_directory"):
        configuration[key] = ROOT / configuration[key]
    previous = read_json(configuration["monitor_directory"] / "registration.json")
    interval = configuration["update_interval"]
    assert interval > 0 and configuration["preserve_existing_epoch_diagnostics"]
    configuration["milestones"] = sorted(set(previous["milestones"]) | set(range(interval, configuration["total_updates"] + 1, interval)))
    return configuration


def verify_configuration(configuration, path):
    assert configuration["monitor_directory"] == diagnostic.OUTPUT
    assert configuration["training_directory"] == diagnostic.TRAINING
    registration = read_json(configuration["output_directory"] / "registration.json")
    assert diagnostic.digest(path) == registration["configuration_sha256"]
    for name, expected in registration["sources"].items():
        assert diagnostic.digest(ROOT / name) == expected, name
    diagnostic.check_registration()


def completed_updates(configuration):
    return {int(read_json(path)["completed_updates"]) for path in configuration["monitor_directory"].glob("update_*/complete.json")
            if read_json(path).get("complete")}


def preserve_exact_checkpoint(source, directory, expected_updates):
    """A hard link keeps the old inode even when the trainer replaces latest.pt."""
    directory.mkdir(parents=True, exist_ok=True)
    candidate = directory / "capture_candidate.pt"
    if candidate.exists():
        candidate.unlink()
    os.link(source, candidate)
    state = torch.load(candidate, map_location="cpu", weights_only=False)
    actual_updates = int(state["completed_updates"])
    del state
    if actual_updates != expected_updates:
        candidate.unlink()
        return actual_updates, None
    destination = directory / f"update_{actual_updates:06d}.pt"
    if destination.exists():
        candidate.unlink()
    else:
        candidate.replace(destination)
        write_json(destination.with_suffix(".json"), {"completed_updates": actual_updates,
                   "checkpoint_sha256": diagnostic.digest(destination), "captured_unix": time.time(),
                   "source": str(source), "preserved_read_only_hard_link": True})
    return actual_updates, destination


def capture_due_checkpoint(configuration, completed, captured_signature):
    training = configuration["training_directory"]
    snapshots = configuration["output_directory"] / "checkpoints"
    existing = completed_updates(configuration)
    due = [update for update in configuration["milestones"] if update <= completed and update not in existing
           and not (snapshots / f"update_{update:06d}.pt").exists()]
    if not due:
        return captured_signature
    source = training / "latest.pt"
    stat = source.stat()
    signature = (stat.st_ino, stat.st_mtime_ns, stat.st_size)
    if signature == captured_signature:
        return captured_signature
    actual, saved = preserve_exact_checkpoint(source, snapshots, min(due))
    if actual > min(due) and saved is None:
        # Epoch checkpoints remain available after latest.pt advances. Wait
        # until at least one later update, so its non-atomic write has finished.
        target = min(due)
        epoch_source = training / f"epoch_{target // configuration['updates_per_epoch']:02d}.pt"
        if target % configuration["updates_per_epoch"] == 0 and epoch_source.exists() and completed > target:
            found, saved = preserve_exact_checkpoint(epoch_source, snapshots, target)
            assert found == target and saved is not None
        else:
            raise RuntimeError(f"Exact checkpoint {target} was missed; latest is {actual}. No later-checkpoint substitution.")
    return signature


def ensure_geometry(configuration, update, stream):
    if update == 0:
        return
    folder = configuration["monitor_directory"] / f"update_{update:06d}"
    if (folder / "geometry/geometry_report.json").exists():
        return
    subprocess.run([sys.executable, "-u", str(ROOT / "scripts/visualize_lpwm_planning_path_geometry.py"),
                    "--monitor", str(configuration["monitor_directory"]), "--updates", str(update),
                    "--output", str(folder / "geometry")], cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT, check=True)


def publish_trends(configuration):
    """CPU only: summarize completed diagnostics; never infer planning gains from movement."""
    output = configuration["output_directory"]
    monitor = configuration["monitor_directory"]
    output.mkdir(parents=True, exist_ok=True)
    records = read_json(monitor / "panel.json")["records"]
    initial = np.array(np.load(monitor / "update_000000/particle_attributes.npy", mmap_mode="r")[:, :, 0])
    prior = initial
    previous_update = 0
    rows, reports = [], []
    for update in sorted(completed_updates(configuration)):
        folder = monitor / f"update_{update:06d}"
        current = np.array(np.load(folder / "particle_attributes.npy", mmap_mode="r")[:, :, 0])
        assert current.shape == initial.shape == (96, 4, 64, 14) and np.isfinite(current).all()
        initial_change = geometry_changes(initial, current)
        previous_change = geometry_changes(prior, current)
        readouts = read_json(folder / "readouts.json")
        row = {"completed_updates": update, "previous_diagnostic_updates": previous_update,
               "center_mean_from_initial_px": initial_change["center_shift_input_pixels"]["mean"],
               "center_max_from_initial_px": initial_change["center_shift_input_pixels"]["maximum"],
               "size_mean_from_initial_percent": initial_change["size_axis_relative_change_percent"]["mean"],
               "presence_mean_from_initial": initial_change["presence_absolute_change"]["mean"],
               "center_mean_from_previous_px": previous_change["center_shift_input_pixels"]["mean"],
               "size_mean_from_previous_percent": previous_change["size_axis_relative_change_percent"]["mean"],
               "object_readout_macro_f1": readouts["classification"]["particle_current_all"]["macro_f1_present_classes"]}
        rows.append(row)
        report = {"metrics": row, "from_initial": initial_change, "from_previous_diagnostic": previous_change,
                  "by_camera": {camera: geometry_changes(initial[:, index], current[:, index]) for index, camera in enumerate(("front", "back", "left", "right"))},
                  "by_scene_type": {scenario: geometry_changes(initial[[record['scene_type'] == scenario for record in records]], current[[record['scene_type'] == scenario for record in records]]) for scenario in ("straight", "left_turn", "right_turn")},
                  "representation_diagnostics": read_json(folder / "summary.json"),
                  "readouts": readouts, "checkpoint": read_json(folder / "complete.json"),
                  "interpretation": "Fixed training-distribution panel. Geometry changes, projected-box association and model sensitivity do not establish planning improvement.",
                  "chat_notification_sent": False}
        report_directory = output / "reports"
        report_directory.mkdir(exist_ok=True)
        destination = report_directory / f"update_{update:06d}.json"
        if not destination.exists():
            write_json(destination, report)
            with (output / "report_events.jsonl").open("a") as stream:
                stream.write(json.dumps({"completed_updates": update, "created_unix": time.time(), "report": str(destination), "delivery": "local_report_only", "chat_notification_sent": False}) + "\n")
        reports.append(report)
        prior, previous_update = current, update
    if not rows:
        return
    write_json(output / "trend_summary.json", {"rows": rows, "latest_updates": rows[-1]["completed_updates"], "scene_count": 96, "not_independent_planning_validation": True})
    with (output / "trend.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    figure, axes = plt.subplots(2, 2, figsize=(12, 7), constrained_layout=True)
    for axis, key, title in zip(axes.flat,
                               ("center_mean_from_initial_px", "size_mean_from_initial_percent", "presence_mean_from_initial", "object_readout_macro_f1"),
                               ("Mean center shift from initial (input px)", "Mean width/height change from initial (%)", "Mean absolute presence change", "Object readout macro F1 (training panel)")):
        axis.plot([row["completed_updates"] for row in rows], [row[key] for row in rows], marker="o")
        axis.set(title=title, xlabel="Optimizer updates")
        axis.grid(alpha=.25)
    figure.suptitle("LPWM particle trends | 96 fixed scenes x 4 cameras | movement is not planning benefit")
    figure.savefig(output / "trend.png", dpi=140)
    plt.close(figure)
    latest = rows[-1]
    message = (f"LPWM 중간 결과: {latest['completed_updates']} update\n"
               f"초기 대비 중심 이동 평균 {latest['center_mean_from_initial_px']:.5f}px, 최대 {latest['center_max_from_initial_px']:.5f}px\n"
               f"초기 대비 크기 축별 평균 변화 {latest['size_mean_from_initial_percent']:.4f}%\n"
               f"Presence 절대 변화 평균 {latest['presence_mean_from_initial']:.6f}\n"
               f"직전 진단({latest['previous_diagnostic_updates']} update) 대비 중심 평균 이동 {latest['center_mean_from_previous_px']:.5f}px\n"
               f"객체 상태 readout macro F1 {latest['object_readout_macro_f1']:.4f}\n"
               "해석: 변화 추세 진단이며, 이동 증가 자체가 planning 성능 개선을 뜻하지 않습니다.\n"
               "동일 학습 분포 패널이며 독립 navtest 평가가 아닙니다. 채팅 푸시 알림은 발송되지 않았습니다.\n")
    (output / "latest_report.txt").write_text(message)
    links = []
    for row in reversed(rows):
        update = row["completed_updates"]
        geometry = monitor / f"update_{update:06d}/geometry/before_after_particle_geometry.png"
        relative = os.path.relpath(geometry, output)
        image_link = f'<a href="{html.escape(relative)}">전후 이미지</a>' if geometry.exists() else "초기 기준"
        links.append(f'<tr><td>{update}</td><td>{row["center_mean_from_initial_px"]:.5f}</td><td>{row["size_mean_from_initial_percent"]:.4f}</td><td>{row["presence_mean_from_initial"]:.6f}</td><td>{row["object_readout_macro_f1"]:.4f}</td><td>{image_link} · <a href="reports/update_{update:06d}.json">상세</a></td></tr>')
    (output / "index.html").write_text('<!doctype html><html lang="ko"><meta charset="utf-8"><meta http-equiv="refresh" content="60"><title>LPWM 500 update 중간 보고</title><style>body{font-family:sans-serif;max-width:1250px;margin:28px auto}img{max-width:100%}table{border-collapse:collapse}td,th{border:1px solid #bbb;padding:8px}pre{white-space:pre-wrap}</style><h1>LPWM particle 변화 추세</h1><p>500 update 간격 + 기존 epoch 진단. 페이지는 60초마다 새로고침됩니다.</p><pre>' + html.escape(message) + '</pre><img src="trend.png"><p><a href="trend.csv">CSV</a> · <a href="latest_report.txt">최신 보고</a></p><table><tr><th>Update</th><th>중심 평균 이동(px)</th><th>크기 평균 변화(%)</th><th>Presence 변화</th><th>객체 readout F1</th><th>결과</th></tr>' + ''.join(links) + '</table></html>')
    write_json(output / "publication_status.json", {"latest_updates": latest["completed_updates"], "published_unix": time.time(), "chat_notification_sent": False})


def watch(configuration, path):
    output, monitor, training = (configuration[key] for key in ("output_directory", "monitor_directory", "training_directory"))
    output.mkdir(parents=True, exist_ok=True)
    ctypes.CDLL(None).prctl(15, b"kjs-lpwm-500", 0, 0, 0)
    lock = (monitor / "watch.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    verify_configuration(configuration, path)
    stop_requested = []
    for signum in (signal.SIGINT, signal.SIGTERM):
        signal.signal(signum, lambda *_: stop_requested.append(True))
    child, active_update, signature = None, None, None
    last_published = set()
    log = (output / "diagnostics.log").open("a")
    try:
        while True:
            verify_configuration(configuration, path)
            paused = any(candidate.exists() for candidate in (output / "pause.requested", monitor / "pause.requested", training.parent / "pause.requested", training / "pause.requested", training / "paused.json"))
            if stop_requested or paused:
                write_json(output / "status.json", {"stage": "paused", "active_update": active_update})
                write_json(monitor / "watch_status.json", {"stage": "paused_with_training" if paused else "paused"})
                return
            progress = read_json(training / "progress.json")
            completed = int(progress["completed_updates"])
            signature = capture_due_checkpoint(configuration, completed, signature)
            if child is not None and child.poll() is not None:
                assert child.returncode == 0, f"Diagnostic update {active_update} failed with {child.returncode}"
                assert active_update in completed_updates(configuration)
                child, active_update = None, None
            existing = completed_updates(configuration)
            if child is None and existing != last_published:
                for update in sorted(existing):
                    ensure_geometry(configuration, update, log)
                publish_trends(configuration)
                last_published = existing
            pending = [update for update in configuration["milestones"] if update not in existing]
            if not pending and (training / "training_complete.json").exists():
                write_json(output / "status.json", {"stage": "complete"})
                write_json(monitor / "watch_status.json", {"stage": "complete"})
                return
            status = {"stage": "evaluating" if child is not None else "waiting_for_checkpoint", "training_updates": completed,
                      "last_evaluated_update": max(existing, default=0), "next_milestone": min(pending) if pending else None,
                      "active_update": active_update, "updated_unix": time.time(), "controller": str(Path(__file__).relative_to(ROOT)),
                      "interval_updates": configuration["update_interval"], "chat_notification_sent": False}
            if child is None and pending:
                target = min(pending)
                snapshot = output / f"checkpoints/update_{target:06d}.pt"
                if snapshot.exists():
                    if diagnostic.card_bytes() < configuration["admission_total_card_bytes"]:
                        environment = dict(os.environ, CUDA_VISIBLE_DEVICES=str(configuration["diagnostic_gpu"]), OMP_NUM_THREADS="2", OPENBLAS_NUM_THREADS="1")
                        child = subprocess.Popen([sys.executable, "-u", str(ROOT / "scripts/monitor_lpwm_drivor_planning_path_representations.py"),
                                                  "--checkpoint", str(snapshot), "--device", "cuda"], cwd=ROOT, env=environment,
                                                 stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                        active_update = target
                        status.update(stage="evaluating", active_update=target, diagnostic_pid=child.pid)
                    else:
                        status["stage"] = "waiting_for_memory"
            write_json(output / "status.json", status)
            write_json(monitor / "watch_status.json", status)
            time.sleep(configuration["poll_seconds"])
    except Exception as error:
        write_json(output / "status.json", {"stage": "diagnostic_failed_training_unaffected", "error": repr(error), "active_update": active_update})
        write_json(monitor / "watch_status.json", {"stage": "diagnostic_failed_training_unaffected", "error": repr(error)})
        raise
    finally:
        if child is not None and child.poll() is None:
            child.send_signal(signal.SIGINT)
            try:
                child.wait(timeout=30)
            except subprocess.TimeoutExpired:
                child.terminate()
                child.wait(timeout=10)
        log.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=ROOT / "configs/lpwm_drivor_review/particle_trends_every500.json")
    parser.add_argument("--publish-only", action="store_true")
    arguments = parser.parse_args()
    configuration = load_configuration(arguments.config)
    if arguments.publish_only:
        publish_trends(configuration)
    else:
        watch(configuration, arguments.config)
