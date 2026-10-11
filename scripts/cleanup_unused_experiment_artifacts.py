"""Audited, user-authorized removal of retired profiling weights, October 11.

Only explicit historical output families are eligible. Scientific checkpoints,
records, image galleries, caches and the active LPWM experiment are preserved.
The default action inventories files. Cleanup quarantines before unlinking.
"""

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import errno
import hashlib
import json
import os
from pathlib import Path
import pwd
import stat
import subprocess


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ACTIVE_EXPERIMENT = PROJECT_ROOT / "outputs/lpwm_front_history_stage1_lora_v1"
LOCAL_REPORT = PROJECT_ROOT / "outputs/unused_result_cleanup_20261011"
QUARANTINE_ROOT = PROJECT_ROOT / ".trash-unused-experiment-artifacts-20261011"
PROFILE_FAMILIES = {
    "lpwm_stage1_effect_v1", "lpwm_card_budget_planning_v2",
    "lpwm_card_budget_expandable_v3", "lpwm_drivor_planning_path_lora_v1",
    "lpwm_drivor_geometry_lora_v1", "lpwm_frozen_control_v1",
    "lpwm_48gb_planning_v1", "lpwm_navsim_posttraining_v1",
    "lpwm_drivor_joint_v1", "lpwm_drivor_lora_v1",
    "lpwm_object_future_planning_v3", "lpwm_card_budget_measured_v4",
}


def write_json(path, content):
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_suffix(path.suffix + ".pending")
    pending.write_text(json.dumps(content, indent=2) + "\n")
    pending.replace(path)


def read_json(path):
    return json.loads(path.read_text())


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def selection_reason(relative_path):
    parts = relative_path.parts
    if len(parts) < 3 or parts[0] != "outputs" or relative_path.suffix != ".pt":
        return None
    family = parts[1]
    if family in PROFILE_FAMILIES and any(
        name in parts for name in ("profile", "engineering_profiles", "ddp_engineering")
    ):
        return "Discarded engineering/profile weights; scientific outputs retained"
    if family == "lpwm_partial_planning_profiles":
        return "Retired microbatch throughput profile weights"
    if family == "lpwm_drivor_optimized_execution_v1" and parts[2] == "benchmarks":
        return "Retired oracle/SDPA throughput benchmark weights"
    if family == "lpwm_adapter_original_batch_rebalance_v1" and parts[2].endswith("_profile"):
        return "Retired parallel microbatch/offload profile weights or gradients"
    if family == "lpwm_driving_video_512x256_v1" and parts[2] == "engineering_check_particles8":
        return "Disposable pre-admission particle-count engineering weights"
    if family == "four_model_small_corpus_v1":
        if "profiles" in parts:
            return "Disposable concurrency/admission profiles; common initialization retained"
        if any(name in parts for name in (
            "before_recovery", "saved_before_policy_update", "before_jepa_resume", "batch_trials"
        )):
            return "Superseded recovery/profile snapshot; later full-state checkpoint retained"
    return None


def file_identity(path):
    metadata = path.lstat()
    assert stat.S_ISREG(metadata.st_mode) and metadata.st_uid == os.getuid(), str(path)
    assert path.resolve(strict=True) == path, "Symlink/indirect target: " + str(path)
    assert path.is_relative_to(PROJECT_ROOT), str(path)
    return dict(device=metadata.st_dev, inode=metadata.st_ino, links=metadata.st_nlink,
                logical_bytes=metadata.st_size, allocated_bytes=metadata.st_blocks * 512,
                modified_ns=metadata.st_mtime_ns, uid=metadata.st_uid)


def retained_dependencies():
    registration = read_json(ACTIVE_EXPERIMENT / "registration.json")
    expected = {}
    for key in ("sources", "configurations", "immutable_inputs"):
        expected.update(registration[key])
    for path, digest in expected.items():
        assert sha256_file(PROJECT_ROOT / path) == digest, "Protected hash changed: " + path
    manifest = read_json(ACTIVE_EXPERIMENT / "manifest.json")
    frame_cache_paths = sorted({record["observed_frame_cache"] for record in manifest["records"]})
    for path in frame_cache_paths:
        assert (PROJECT_ROOT / path).is_file(), path
    # Current four-camera scene cache is also used by the live one-camera trainer.
    for path in {record["cache_directory"] for record in manifest["records"]}:
        assert Path(path).is_dir(), path
    retained_weights = [
        "outputs/lpwm_navsim_full_posttraining_v2/stage1/checkpoint.pt",
        "outputs/lpwm_navsim_full_posttraining_v2/stage1/latest.pt",
        "outputs/lpwm_adapter_original_batch_shared_v4/latest.pt",
        "outputs/lpwm_drivor_planning_path_lora_v1/navsim_v1/latest.pt",
    ]
    small_corpus_root = PROJECT_ROOT / "outputs/four_model_small_corpus_v1"
    assert (small_corpus_root / "pause.requested").exists(), "Historical run resumed"
    completed = {}
    for condition in ("drivor", "jepa", "lpwm_sequential", "lpwm_joint"):
        directory = small_corpus_root / condition
        completed[condition] = read_json(directory / "progress.json")["completed_updates"]
        retained_weights.append(str((directory / "latest.pt").relative_to(PROJECT_ROOT)))
        for path in directory.glob("pass*.pt"):
            retained_weights.append(str(path.relative_to(PROJECT_ROOT)))
    assert completed == dict(drivor=3200, jepa=3200, lpwm_sequential=3200, lpwm_joint=2264)
    retained = {path: file_identity(PROJECT_ROOT / path) for path in retained_weights}
    gallery_catalog = PROJECT_ROOT / "results/lpwm_particle_movement_review_20261011/all_joint_lora_particle_png_paths.txt"
    galleries = gallery_catalog.read_text().splitlines()
    assert galleries and all(Path(path).is_file() for path in galleries)
    return dict(registered_hashes_verified=len(expected), registered_hashes=expected,
                retained_checkpoint_identity=retained, original_frame_caches=frame_cache_paths,
                historical_completed_updates=completed,
                requested_particle_image_paths_preserved=len(galleries),
                live_progress=read_json(ACTIVE_EXPERIMENT / "stage2/progress.json"))


def superseded_snapshot_evidence(relative_path, retained):
    parts = relative_path.parts
    if parts[1] != "four_model_small_corpus_v1" or "Superseded" not in selection_reason(relative_path):
        return None
    if "before_jepa_resume" in parts:
        condition = "jepa"
    elif "batch_trials" in parts:
        condition = parts[parts.index("batch_trials") + 1]
    else:
        marker = "before_recovery" if "before_recovery" in parts else "saved_before_policy_update"
        condition = parts[parts.index(marker) + 1].removeprefix("train_")
    earlier_progress = read_json((PROJECT_ROOT / relative_path).parent / "progress.json")
    earlier_update = earlier_progress["completed_updates"]
    later_update = retained["historical_completed_updates"][condition]
    assert earlier_update < later_update, str(relative_path)
    return dict(condition=condition, superseded_update=earlier_update, retained_update=later_update,
                retained_path=f"outputs/four_model_small_corpus_v1/{condition}/latest.pt")


def process_references(candidate_rows, dependency_report, quarantined=False):
    queue = read_json(ACTIVE_EXPERIMENT / "stage2/queue_state.json")
    # The default sandbox process namespace does not expose host trainers.
    # Fail closed: run this audit in the approved host namespace if necessary.
    assert (Path("/proc") / str(queue["queue_pid"])).is_dir(), "Host process namespace required"
    identities = {(row["identity"]["device"], row["identity"]["inode"]) for row in candidate_rows}
    target_paths = {str((QUARANTINE_ROOT if quarantined else PROJECT_ROOT) / row["path"])
                    for row in candidate_rows}
    target_directories = {str(Path(path).parent) for path in target_paths}
    references, scanned_processes, excluded_system_sessions, stale_unrelated_handles = [], [], [], []
    for process in Path("/proc").iterdir():
        if not process.name.isdigit() or int(process.name) == os.getpid():
            continue
        try:
            if process.stat().st_uid != os.getuid():
                continue
            command = (process / "cmdline").read_bytes().replace(b"\0", b" ").decode(errors="replace")
            fields = (process / "stat").read_text().split(") ", 1)[1].split()
            if fields[0] == "Z":
                continue
            pid = int(process.name)
            # PAM session managers are privileged, non-dumpable OS services,
            # not experiment jobs. Their fds are intentionally inaccessible.
            # All other owned processes must pass the actual fd/maps audit.
            process_name = (process / "comm").read_text().strip()
            if process_name == "(sd-pam)" and command.strip() == "(sd-pam)":
                excluded_system_sessions.append(dict(pid=pid, name="sd-pam", reason="OS authentication session"))
                continue
            if process_name == "sshd" and command.startswith("sshd: " + pwd.getpwuid(os.getuid()).pw_name + "@"):
                excluded_system_sessions.append(dict(pid=pid, name="sshd", reason="SSH transport; children inspected separately"))
                continue
            # No arbitrary commands/environment variables are written to the report.
            scanned_processes.append(pid)
            if any(path in command for path in target_paths | target_directories):
                references.append(dict(pid=pid, kind="command_reference"))
            try:
                working_directory = os.readlink(process / "cwd")
                if any(working_directory == path or working_directory.startswith(path + "/")
                       for path in target_directories):
                    references.append(dict(pid=pid, kind="working_directory_reference"))
            except FileNotFoundError:
                continue
            for handle in (process / "fd").iterdir():
                try:
                    handle_target = os.readlink(handle)
                    if handle_target.removesuffix(" (deleted)") in target_paths:
                        references.append(dict(pid=pid, kind="open_file_path", path=handle_target))
                    metadata = handle.stat()
                    if (metadata.st_dev, metadata.st_ino) in identities:
                        references.append(dict(pid=pid, kind="open_file", path=handle_target))
                except FileNotFoundError:
                    continue
                except OSError as error:
                    # NFS may retain an already-stale fd for an unrelated file.
                    # All candidate hardlink names are known; matching paths
                    # above remain blocking even when their stat is unavailable.
                    if error.errno != errno.ESTALE:
                        raise
                    assert handle_target.removesuffix(" (deleted)") not in target_paths
                    stale_unrelated_handles.append(dict(pid=pid, descriptor=handle.name))
            # mmap users may have closed their file descriptors.
            for line in (process / "maps").read_text().splitlines():
                fields = line.split(None, 5)
                if len(fields) == 6 and fields[5] in target_paths:
                    references.append(dict(pid=pid, kind="mapped_file", path=fields[5]))
        except (FileNotFoundError, ProcessLookupError):
            continue
        except PermissionError as error:
            raise RuntimeError(f"Cannot inspect owned process {process.name}; abort cleanup") from error
    assert queue["queue_pid"] in scanned_processes, "Live queue was not inspected"
    assert not references, references
    return dict(owned_process_count=len(scanned_processes), candidate_references=references,
                excluded_system_sessions=excluded_system_sessions,
                stale_unrelated_file_descriptors=stale_unrelated_handles,
                preserved_queue_pid=queue["queue_pid"], active_pid=queue.get("active_pid"),
                active_experiment=str(ACTIVE_EXPERIMENT), host_namespace_verified=True)


def audit():
    dependency_report = retained_dependencies()
    tracked = set(subprocess.check_output(["git", "ls-files"], cwd=PROJECT_ROOT, text=True).splitlines())
    weight_paths = subprocess.check_output(
        ["rg", "--files", "--hidden", "--no-ignore", "outputs", "-g", "*.pt"],
        cwd=PROJECT_ROOT, text=True).splitlines()
    candidates = []
    for relative_name in sorted(weight_paths):
        relative_path = Path(relative_name)
        reason = selection_reason(relative_path)
        if reason is None:
            continue
        assert relative_name not in tracked
        assert not (PROJECT_ROOT / relative_path).is_relative_to(ACTIVE_EXPERIMENT)
        assert relative_name not in dependency_report["registered_hashes"]
        candidates.append(dict(path=relative_name, reason=reason,
                               identity=file_identity(PROJECT_ROOT / relative_path),
                               superseded_by=superseded_snapshot_evidence(relative_path, dependency_report)))
    assert candidates, "No candidates; cleanup might already have completed"
    groups = defaultdict(list)
    for row in candidates:
        identity = row["identity"]
        groups[(identity["device"], identity["inode"])].append(row)
    released_bytes, family_bytes = 0, Counter()
    for rows in groups.values():
        identity = rows[0]["identity"]
        assert len(rows) == identity["links"], "A candidate inode also has a retained hardlink"
        released_bytes += identity["allocated_bytes"]
        family_bytes[Path(rows[0]["path"]).parts[1]] += identity["allocated_bytes"]
    protected_file_hashes = {name: sha256_file(PROJECT_ROOT / name) for name in tracked
                            if (PROJECT_ROOT / name).is_file() and not (PROJECT_ROOT / name).is_symlink()}
    result = dict(audited_at=datetime.now(timezone.utc).isoformat(), user_authorized=True,
                  selection_scope="Retired profile weights and superseded recovery snapshots only",
                  files=candidates, file_count=len(candidates), unique_inode_count=len(groups),
                  logical_bytes=sum(row["identity"]["logical_bytes"] for row in candidates),
                  expected_released_allocated_bytes=released_bytes,
                  released_bytes_by_family=dict(family_bytes), dependency_check=dependency_report,
                  tracked_file_hashes=protected_file_hashes)
    result["host_process_check"] = process_references(candidates, dependency_report)
    write_json(LOCAL_REPORT / "audit.json", result)
    print(json.dumps(dict(audited_files=len(candidates), expected_released_allocated_bytes=released_bytes,
                          host_process_check=result["host_process_check"])), flush=True)


def cleanup():
    assert not (LOCAL_REPORT / "cleanup_complete.json").exists(), "Cleanup already completed"
    plan = read_json(LOCAL_REPORT / "audit.json")
    assert plan["user_authorized"] and not QUARANTINE_ROOT.exists()
    before = retained_dependencies()
    assert before["retained_checkpoint_identity"] == plan["dependency_check"]["retained_checkpoint_identity"]
    process_references(plan["files"], before)
    for row in plan["files"]:
        assert selection_reason(Path(row["path"])) == row["reason"]
        assert file_identity(PROJECT_ROOT / row["path"]) == row["identity"], row["path"]
    QUARANTINE_ROOT.mkdir()
    moved = []
    try:
        for row in plan["files"]:
            destination = QUARANTINE_ROOT / row["path"]
            destination.parent.mkdir(parents=True, exist_ok=True)
            (PROJECT_ROOT / row["path"]).rename(destination)
            moved.append(row)
        after_quarantine = retained_dependencies()
        assert after_quarantine["retained_checkpoint_identity"] == before["retained_checkpoint_identity"]
        host_check = process_references(plan["files"], after_quarantine, quarantined=True)
        for row in moved:
            assert file_identity(QUARANTINE_ROOT / row["path"]) == row["identity"]
    except BaseException:
        # No deletion has happened yet. Restore all moved paths on a failed guard.
        for row in reversed(moved):
            (QUARANTINE_ROOT / row["path"]).rename(PROJECT_ROOT / row["path"])
        raise
    write_json(LOCAL_REPORT / "quarantine_verified.json", dict(
        file_count=len(moved), dependency_check=after_quarantine, host_process_check=host_check))
    removed = []
    for row in moved:
        (QUARANTINE_ROOT / row["path"]).unlink()
        removed.append(row["path"])
    for directory, _, _ in os.walk(QUARANTINE_ROOT, topdown=False, followlinks=False):
        Path(directory).rmdir()
    after = retained_dependencies()
    assert after["retained_checkpoint_identity"] == before["retained_checkpoint_identity"]
    for name, expected in plan["tracked_file_hashes"].items():
        assert sha256_file(PROJECT_ROOT / name) == expected, name
    assert all(not (PROJECT_ROOT / path).exists() for path in removed)
    result = dict(complete=True, completed_at=datetime.now(timezone.utc).isoformat(),
                  removed_files=removed, removed_file_count=len(removed),
                  released_allocated_bytes=plan["expected_released_allocated_bytes"],
                  released_bytes_by_family=plan["released_bytes_by_family"],
                  before_progress=before["live_progress"], after_progress=after["live_progress"],
                  retained_dependency_check=after, host_process_check=host_check,
                  tracked_files_unchanged=len(plan["tracked_file_hashes"]),
                  no_shared_originals_modified=True, scientific_final_and_epoch_checkpoints_preserved=True,
                  all_requested_particle_image_paths_preserved=True,
                  note="Allocated bytes from unique last-linked inodes; concurrent training changes global free space.")
    write_json(LOCAL_REPORT / "cleanup_complete.json", result)
    print(json.dumps(dict(complete=True, removed_files=len(removed),
        released_allocated_bytes=result["released_allocated_bytes"],
        live_updates_before=before["live_progress"]["completed_updates"],
        live_updates_after=after["live_progress"]["completed_updates"])), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--action", choices=("audit", "cleanup"), default="audit")
    arguments = parser.parse_args()
    {"audit": audit, "cleanup": cleanup}[arguments.action]()
