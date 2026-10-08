"""Prepare DrivingDojo's JPEG sequences, as documented in the official paper.

Source nominal rate is 5Hz. Select the nearest source image at a nominal 2Hz
grid, retaining actual relative source times (.4/.6s intervals can alternate).
Source archives must pass their pinned SHA before any derived clip is admitted.
"""
import argparse
import fcntl
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import shutil
import tarfile
import time

import cv2
import numpy as np
from huggingface_hub import get_token
from lpwm_driving_video_io import ResumableVerifiedReader

ROOT = Path(__file__).resolve().parents[1]
FPS_SOURCE = "https://arxiv.org/html/2410.10738v1#S3.SS4"


def write_json(path, content):
    pending = path.with_suffix(path.suffix + ".pending")
    pending.write_text(json.dumps(content, indent=2, allow_nan=False) + "\n")
    pending.replace(path)


def resampling_indices(frame_ids):
    identifiers = np.asarray(frame_ids)
    gaps = np.diff(identifiers)
    assert len(identifiers) >= 20 and (gaps > 0).all()
    nominal_stride = int(np.median(gaps))
    assert nominal_stride > 0 and (gaps % nominal_stride == 0).all()
    observed_times = (identifiers - identifiers[0]) / (5. * nominal_stride)
    targets = np.arange(0., observed_times[-1] + .2 - 1e-7, .5)
    indices = np.abs(observed_times[:, None] - targets[None]).argmin(0)
    assert len(np.unique(indices)) == len(indices)
    assert np.abs(observed_times[indices] - targets).max() <= .10001, "Sequence gap exceeds nominal resampling tolerance"
    return indices, observed_times, targets, nominal_stride


def main(arguments):
    cv2.setNumThreads(1)
    output = arguments.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    lock = (output / "prepare.lock").open("w")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    videos = ROOT / "outputs/lpwm_driving_video_512x256_v1/converted_external_videos/drivingdojo"
    videos.mkdir(parents=True, exist_ok=True)
    manifest = output / "converted_clips"
    manifest.mkdir(exist_ok=True)
    catalog = json.loads(arguments.catalog.read_text())
    source_files = [(record, item) for record in catalog["records"] if "DrivingDojo" in record["repository"]
                    for item in record["files"]]
    source_files.sort(key=lambda pair: int(re.findall(r"\d+", Path(pair[1]["filename"]).name)[0]))
    registration = {"catalog_sha256": hashlib.sha256(arguments.catalog.read_bytes()).hexdigest(),
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "io_sha256": hashlib.sha256((ROOT / "scripts/lpwm_driving_video_io.py").read_bytes()).hexdigest(),
        "fps_source": FPS_SOURCE, "source_fps": 5., "target_nominal_fps": 2.,
        "maximum_sample_time_error_seconds": .10001, "maximum_hours": arguments.maximum_hours}
    if (output / "registration.json").exists():
        assert json.loads((output / "registration.json").read_text()) == registration
    else:
        write_json(output / "registration.json", registration)
    existing = [json.loads(path.read_text()) for path in manifest.glob("*.json")]
    completed_clips = len(existing)
    total_seconds = sum(row["nominal_sampled_duration_seconds"] for row in existing)
    headers = {"Authorization": "Bearer " + get_token()}
    archives_verified = 0
    for record, source in source_files:
        if (output / "pause.requested").exists():
            return
        if total_seconds >= arguments.maximum_hours * 3600:
            break
        archive_key = record["repository"].split("/")[-1] + "_" + Path(source["filename"]).name
        archive_marker = output / (archive_key + ".verified.json")
        if archive_marker.exists():
            archives_verified += 1
            continue
        source_identity = {"repository": record["repository"], "revision": record["revision"],
                           "archive": source["filename"], "archive_verification_marker": str(archive_marker)}
        url = f"https://huggingface.co/datasets/{record['repository']}/resolve/{record['revision']}/{source['filename']}"
        current_clip, current_frames, current_hashes = None, {}, {}
        seen_clip_ids = set()
        def status(downloaded=0, reconnections=0):
            write_json(output / "status.json", {"status": "streaming_and_converting", "archive": archive_key,
                "received_bytes": downloaded, "archive_bytes": source["bytes"], "reconnections": reconnections,
                "completed_clips": completed_clips, "converted_nominal_hours": total_seconds / 3600,
                "archives_verified": archives_verified, "updated_unix": time.time(), "full_330h_manifest_ready": False})
        def finish_clip():
            nonlocal completed_clips, total_seconds
            if not current_frames:
                return
            identity = {**source_identity, "clip_id": current_clip}
            key = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:24]
            marker = manifest / (key + ".json")
            if marker.exists():
                return
            ids = sorted(current_frames)
            selection, source_times, targets, stride = resampling_indices(ids)
            assert shutil.disk_usage(videos).free > 100_000_000_000
            destination = videos / (key + ".avi")
            pending = destination.with_suffix(".pending.avi")
            writer = cv2.VideoWriter(str(pending), cv2.VideoWriter_fourcc(*"FFV1"), 2., (512, 256))
            assert writer.isOpened()
            for index in selection:
                writer.write(current_frames[ids[index]])
            writer.release()
            check = cv2.VideoCapture(str(pending))
            assert int(check.get(cv2.CAP_PROP_FRAME_COUNT)) == len(selection)
            success, first = check.read()
            check.release()
            assert success and np.array_equal(first, current_frames[ids[selection[0]]])
            pending.replace(destination)
            prepared = {"source": identity, "frame_count": len(selection), "source_frame_count": len(ids),
                "source_frame_ids": ids, "source_frame_stride": stride,
                "selected_source_frame_ids": [ids[index] for index in selection],
                "source_nominal_fps": 5., "source_fps_reference": FPS_SOURCE,
                "sample_timestamps_seconds": source_times[selection].tolist(), "nominal_target_timestamps": targets.tolist(),
                "source_image_sha256": current_hashes, "nominal_sampled_duration_seconds": len(selection) / 2.,
                "source_duration_seconds": float(source_times[-1] + .2),
                "storage": "ffv1_video", "video_path": str(destination),
                "video_sha256": hashlib.sha256(destination.read_bytes()).hexdigest(),
                "crop_top_bottom_pixels": 28, "width": 512, "height": 256, "target_nominal_fps": 2.,
                "exact_half_second_source_intervals": False, "eligible_for_training": False,
                "admission_pending": "archive SHA plus recording/split/deduplication audit"}
            write_json(marker, prepared)
            completed_clips += 1
            total_seconds += len(selection) / 2.
            if completed_clips % 10 == 0:
                print(json.dumps({"completed_clips": completed_clips, "hours": total_seconds / 3600}), flush=True)
        with ResumableVerifiedReader(url, headers, source["bytes"], source["lfs_sha256"], status) as reader:
            with tarfile.open(fileobj=reader, mode="r|gz") as archive:
                for member in archive:
                    if not member.isfile() or not member.name.endswith("_CameraFpgaP0H120.jpg"):
                        continue
                    components = PurePosixPath(member.name).parts
                    assert len(components) >= 2
                    clip_id = components[-2]
                    if clip_id != current_clip:
                        finish_clip()
                        assert clip_id not in seen_clip_ids, "Archive clip groups must be contiguous"
                        seen_clip_ids.add(clip_id)
                        current_clip, current_frames, current_hashes = clip_id, {}, {}
                    if total_seconds >= arguments.maximum_hours * 3600:
                        continue
                    frame_id = int(components[-1].split("_", 1)[0])
                    assert frame_id not in current_frames and member.size < 25_000_000
                    content = archive.extractfile(member).read()
                    assert len(content) == member.size
                    image = cv2.imdecode(np.frombuffer(content, dtype=np.uint8), cv2.IMREAD_COLOR)
                    assert image is not None and image.shape[0] > 56
                    current_frames[frame_id] = cv2.resize(image[28:-28], (512, 256), interpolation=cv2.INTER_LINEAR)
                    current_hashes[frame_id] = hashlib.sha256(content).hexdigest()
                    assert sum(frame.nbytes for frame in current_frames.values()) < 1_000_000_000
                finish_clip()
            reader.verify()
        assert seen_clip_ids, "No documented front-camera JPEG groups found"
        write_json(archive_marker, {"source": source_identity, "sha256_verified": True,
            "sha256": source["lfs_sha256"], "clip_count": len(seen_clip_ids), "completed_unix": time.time()})
        archives_verified += 1
    write_json(output / "status.json", {"status": "sources_prepared_not_admitted", "completed_clips": completed_clips,
        "converted_nominal_hours": total_seconds / 3600, "archives_verified": archives_verified,
        "full_330h_manifest_ready": False})


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--catalog", type=Path, default=ROOT / "outputs/lpwm_driving_video_512x256_v1/download_access_after_user_approval.json")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/lpwm_driving_video_512x256_v1/external_video_preparation/drivingdojo_frames_v2")
    parser.add_argument("--maximum-hours", type=float, default=180.)
    arguments = parser.parse_args()
    try:
        main(arguments)
    except Exception as error:
        write_json(arguments.output / "failed.json", {"type": type(error).__name__, "error": str(error), "time": time.time()})
        raise
