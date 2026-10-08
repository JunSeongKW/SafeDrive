"""Download and transcode authorized CoVLA/DrivingDojo candidates on CPU.

Retain lossless 512x256, 2Hz RGB videos and provenance in the workspace. Original
video transfer buffers live only in the dedicated processed_dataset spool.
Converted files are corpus candidates; a separate manifest audit admits them
to the 330-hour train split after deduplication and evaluation-overlap checks.
"""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import sys
import tarfile
import time

import cv2
from huggingface_hub import get_token

from lpwm_driving_video_io import ResumableVerifiedReader

ROOT = Path(__file__).resolve().parents[1]
RAW_ROOT = Path("/home/user/data/processed_dataset/junseong/lpwm_driving_video_512x256_v1")


def write_json(path, content):
    pending = path.with_suffix(path.suffix + ".pending")
    pending.write_text(json.dumps(content, indent=2, allow_nan=False) + "\n")
    pending.replace(path)


def convert_video(source, destination):
    """Sample each .5s from the source time axis, with exact Drive-JEPA resize settings."""
    reader = cv2.VideoCapture(str(source))
    assert reader.isOpened(), f"Cannot decode {source.name}"
    fps = reader.get(cv2.CAP_PROP_FPS)
    declared_frames = int(reader.get(cv2.CAP_PROP_FRAME_COUNT))
    assert 2 <= fps <= 240 and declared_frames > 0, (fps, declared_frames)
    pending = destination.with_suffix(".pending.avi")
    writer = cv2.VideoWriter(str(pending), cv2.VideoWriter_fourcc(*"FFV1"), 2., (512, 256))
    assert writer.isOpened(), "Lossless FFV1 encoder unavailable"
    source_frames, timestamps = 0, []
    next_sample_time = 0.
    previous_timestamp = -1.
    maximum_interval_error = 0.
    first_preprocessed = None
    source_shape = None
    try:
        while reader.grab():
            timestamp = reader.get(cv2.CAP_PROP_POS_MSEC) / 1000.
            if timestamp <= previous_timestamp and source_frames:
                raise RuntimeError("Source timestamps are not increasing; manual timestamp audit required")
            previous_timestamp = timestamp
            source_frames += 1
            if timestamp + 1e-6 < next_sample_time:
                continue
            success, image = reader.retrieve()
            assert success and image is not None
            assert image.shape[0] > 56
            source_shape = list(image.shape)
            resized = cv2.resize(image[28:-28], (512, 256), interpolation=cv2.INTER_LINEAR)
            if first_preprocessed is None:
                first_preprocessed = resized.copy()
            writer.write(resized)
            timestamps.append(timestamp)
            maximum_interval_error = max(maximum_interval_error, abs(timestamp - next_sample_time))
            next_sample_time += .5
    finally:
        reader.release()
        writer.release()
    assert source_frames == declared_frames, f"Truncated source: decoded {source_frames} of {declared_frames}"
    assert len(timestamps) >= 8, "Fewer than eight sampled frames"
    assert maximum_interval_error < .1, maximum_interval_error
    validation = cv2.VideoCapture(str(pending))
    assert int(validation.get(cv2.CAP_PROP_FRAME_COUNT)) == len(timestamps)
    success, decoded_first = validation.read()
    validation.release()
    assert success and (decoded_first == first_preprocessed).all(), "Lossless encode validation failed"
    pending.replace(destination)
    digest = hashlib.sha256()
    with destination.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return {"storage": "ffv1_video", "video_path": str(destination), "video_sha256": digest.hexdigest(),
        "frame_count": len(timestamps), "sample_timestamps_seconds": timestamps,
        "nominal_sampled_duration_seconds": len(timestamps) / 2.,
        "source_duration_seconds": declared_frames / fps, "source_frames": source_frames,
        "source_fps": fps, "source_shape": source_shape, "crop_top_bottom_pixels": 28,
        "width": 512, "height": 256, "sample_fps": 2., "color_storage": "lossless BGR; convert to RGB on load",
        "maximum_sample_timestamp_error_seconds": maximum_interval_error,
        "eligible_for_training": False, "admission_pending": "curated manifest identity, split and duplicate audit"}


def catalog_url(repository, revision, filename):
    return f"https://huggingface.co/datasets/{repository}/resolve/{revision}/{filename}"


def main(arguments):
    cv2.setNumThreads(1)
    control = arguments.control.resolve() / arguments.dataset
    control.mkdir(parents=True, exist_ok=True)
    lock = (control / "prepare.lock").open("w")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    owner = json.loads((RAW_ROOT / "owner.json").read_text())
    assert owner["project"] == str(ROOT)
    spool = RAW_ROOT / "video_transfer_spool" / arguments.dataset
    spool.mkdir(parents=True, exist_ok=True)
    video_output = arguments.videos.resolve() / arguments.dataset
    video_output.mkdir(parents=True, exist_ok=True)
    manifest_output = control / "converted_videos"
    manifest_output.mkdir(exist_ok=True)
    catalog = json.loads(arguments.catalog.read_text())
    assert catalog["all_sources_accessible"]
    records = [row for row in catalog["records"] if
        ("CoVLA" in row["repository"] if arguments.dataset == "covla" else "DrivingDojo" in row["repository"])]
    token = get_token()
    headers = {"Authorization": "Bearer " + token}
    existing = [json.loads(path.read_text()) for path in manifest_output.glob("*.json")]
    total_seconds = sum(row["nominal_sampled_duration_seconds"] for row in existing)
    completed_videos = len(existing)
    source_registration = {"dataset": arguments.dataset, "maximum_hours": arguments.maximum_hours,
        "catalog_sha256": hashlib.sha256(arguments.catalog.read_bytes()).hexdigest(),
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "io_sha256": hashlib.sha256((ROOT / "scripts/lpwm_driving_video_io.py").read_bytes()).hexdigest(),
        "raw_spool_limit_per_video_bytes": 2_000_000_000, "full_corpus_admitted": False}
    if (control / "registration.json").exists():
        assert json.loads((control / "registration.json").read_text()) == source_registration
    else:
        write_json(control / "registration.json", source_registration)

    def status(phase, **extra):
        write_json(control / "status.json", {"status": phase, "dataset": arguments.dataset,
            "completed_videos": completed_videos, "converted_nominal_hours": total_seconds / 3600,
            "updated_unix": time.time(), "full_330h_manifest_ready": False, **extra})

    def ingest(source_stream, source_bytes, identity, source_hash=None):
        nonlocal total_seconds, completed_videos
        key = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:24]
        marker = manifest_output / (key + ".json")
        if marker.exists():
            return
        assert 0 < source_bytes <= 2_000_000_000, "Transfer spool is limited to 2GB per video"
        assert shutil.disk_usage(video_output).free > 100_000_000_000, "Leave 100GB workspace headroom"
        raw_video = spool / (key + ".mp4")
        status("receiving_video", source=identity, source_bytes=source_bytes)
        digest, received = hashlib.sha256(), 0
        with raw_video.open("wb") as stream:
            while received < source_bytes:
                content = source_stream.read(min(1024 * 1024, source_bytes - received))
                assert content, "Unexpected video EOF"
                stream.write(content)
                digest.update(content)
                received += len(content)
        if source_hash:
            assert digest.hexdigest() == source_hash, "Video checksum mismatch"
        status("transcoding_video", source=identity, source_bytes=source_bytes)
        prepared = convert_video(raw_video, video_output / (key + ".avi"))
        prepared.update({"source": identity, "source_video_sha256": digest.hexdigest(),
            "source_bytes": source_bytes, "source_checksum_verified": source_hash is not None})
        write_json(marker, prepared)
        total_seconds += prepared["nominal_sampled_duration_seconds"]
        completed_videos += 1
        # Remove only this process's newly created, verified transfer buffer.
        assert raw_video.resolve().is_relative_to(spool.resolve())
        raw_video.unlink()
        status("video_converted", last_video=key)
        if completed_videos % 10 == 0:
            print(json.dumps({"dataset": arguments.dataset, "videos": completed_videos,
                "hours": total_seconds / 3600}), flush=True)

    for record in records:
        files = record["files"]
        if arguments.dataset == "drivingdojo":
            files = sorted(files, key=lambda item: int(re.findall(r"\d+", Path(item["filename"]).name)[0]))
        for source in files:
            if (control / "pause.requested").exists():
                status("paused")
                return
            if total_seconds >= arguments.maximum_hours * 3600:
                status("candidate_duration_target_reached_not_admitted")
                return
            identity = {"repository": record["repository"], "revision": record["revision"], "filename": source["filename"]}
            url = catalog_url(record["repository"], record["revision"], source["filename"])
            def network_progress(downloaded, reconnections):
                status("receiving_source", source=identity, received_bytes=downloaded,
                    expected_bytes=source["bytes"], reconnections=reconnections)
            if arguments.dataset == "covla":
                key = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:24]
                if (manifest_output / (key + ".json")).exists():
                    continue
                with ResumableVerifiedReader(url, headers, source["bytes"], source["lfs_sha256"], network_progress) as reader:
                    ingest(reader, source["bytes"], identity, source["lfs_sha256"])
                    reader.verify()
            else:
                archive_key = record["repository"].split("/")[-1] + "_" + Path(source["filename"]).name
                marker = control / (archive_key + ".verified.json")
                if marker.exists():
                    continue
                members_seen, videos_seen = 0, 0
                with ResumableVerifiedReader(url, headers, source["bytes"], source["lfs_sha256"], network_progress) as reader:
                    with tarfile.open(fileobj=reader, mode="r|gz") as archive:
                        for member in archive:
                            members_seen += 1
                            if members_seen <= 5:
                                print(json.dumps({"archive": archive_key, "member": member.name, "bytes": member.size}), flush=True)
                            if not member.isfile() or not member.name.lower().endswith((".mp4", ".avi", ".mov", ".mkv")):
                                continue
                            videos_seen += 1
                            if total_seconds >= arguments.maximum_hours * 3600:
                                continue
                            video_identity = {**identity, "archive_member": member.name,
                                              "archive_verification_marker": str(marker)}
                            ingest(archive.extractfile(member), member.size, video_identity)
                    reader.verify()
                assert videos_seen > 0, "Unexpected archive format: no video files"
                write_json(marker, {"source": identity, "sha256": source["lfs_sha256"],
                    "verified": True, "video_members": videos_seen, "completed_unix": time.time()})
    status("source_files_processed_not_admitted")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=("covla", "drivingdojo"), required=True)
    parser.add_argument("--maximum-hours", type=float, required=True)
    parser.add_argument("--catalog", type=Path, default=ROOT / "outputs/lpwm_driving_video_512x256_v1/download_access_after_user_approval.json")
    parser.add_argument("--control", type=Path, default=ROOT / "outputs/lpwm_driving_video_512x256_v1/external_video_preparation")
    parser.add_argument("--videos", type=Path, default=ROOT / "outputs/lpwm_driving_video_512x256_v1/converted_external_videos")
    arguments = parser.parse_args()
    try:
        main(arguments)
    except Exception as error:
        output = arguments.control.resolve() / arguments.dataset
        output.mkdir(parents=True, exist_ok=True)
        write_json(output / "failed.json", {"type": type(error).__name__, "error": str(error), "time": time.time()})
        raise
