"""Stream official archives, retaining only missing front images under an owned raw root.

Compressed archives are never stored in full. Shared originals are read only.
Each finished archive is SHA256 checked against the pinned Hugging Face catalog.
Only completed archives may be admitted to a subsequent expanded corpus manifest.
"""
import argparse
import hashlib
import io
import json
import os
from pathlib import Path, PurePosixPath
import pickle
import tarfile
import time

from huggingface_hub import get_token
import requests
from lpwm_driving_video_io import ResumableVerifiedReader

ROOT = Path(__file__).resolve().parents[1]
RAW_PARENT = Path("/home/user/data/processed_dataset/junseong/lpwm_driving_video_512x256_v1")


class HashingArchiveReader(io.RawIOBase):
    def __init__(self, response):
        self.response = response
        self.digest = hashlib.sha256()
        self.byte_count = 0

    def readable(self):
        return True

    def read(self, size=-1):
        content = self.response.read(size)
        self.digest.update(content)
        self.byte_count += len(content)
        return content


def write_json(path, content):
    pending = path.with_suffix(".pending.json")
    pending.write_text(json.dumps(content, indent=2) + "\n")
    pending.replace(path)


def main(arguments):
    control = arguments.control.resolve()
    control.mkdir(parents=True, exist_ok=True)
    raw_root = RAW_PARENT / "openscene_front_trainval"
    raw_root.mkdir(parents=True, exist_ok=True)
    owner = RAW_PARENT / "owner.json"
    if owner.exists():
        assert json.loads(owner.read_text())["project"] == str(ROOT)
    else:
        write_json(owner, {"project": str(ROOT), "user_requested_cleanup_after_training": True,
            "shared_originals_must_never_be_deleted": True, "raw_storage_cap_bytes": 1_000_000_000_000})
    shared_root = ROOT / "dataset/sensor_blobs/trainval"
    needed = set()
    for metadata in sorted((ROOT / "dataset/navsim_logs/trainval").glob("*.pkl")):
        with metadata.open("rb") as stream:
            frames = pickle.load(stream)
        for frame in frames:
            relative = frame["cams"]["CAM_F0"]["data_path"]
            if not (shared_root / relative).is_file():
                needed.add(relative)
    write_json(control / "required_images.json", {"count": len(needed), "raw_root": str(raw_root)})
    catalog = json.loads(arguments.catalog.read_text())
    source = next(row for row in catalog["records"] if row["repository"] == "OpenDriveLab/OpenScene")
    archives = sorted(source["files"], key=lambda row: int(Path(row["filename"]).stem.rsplit("_", 1)[1]))
    token = get_token()
    headers = {"Authorization": "Bearer " + token} if token else {}
    total_retained = sum(path.stat().st_size for path in raw_root.rglob("*.jpg"))
    retained_images, completed_archives = 0, 0
    for archive in archives[:arguments.max_archives]:
        if (control / "pause.requested").exists():
            write_json(control / "status.json", {"status": "paused", "raw_root": str(raw_root)})
            return
        archive_id = Path(archive["filename"]).name
        marker = control / (archive_id + ".complete.json")
        if marker.exists():
            completed_archives += 1
            continue
        url = f"https://huggingface.co/datasets/{source['repository']}/resolve/{source['revision']}/{archive['filename']}"
        start = time.time()
        write_json(control / "status.json", {"status": "streaming", "archive": archive_id,
            "completed_archives": completed_archives, "retained_bytes": total_retained, "started_unix": start})
        def progress(network_bytes, reconnections):
            write_json(control / "status.json", {"status": "streaming", "archive": archive_id,
                "completed_archives": completed_archives, "retained_bytes": total_retained,
                "network_bytes_this_archive": network_bytes, "archive_bytes": archive["bytes"],
                "reconnections": reconnections, "elapsed_seconds_this_archive": time.time() - start})
        with ResumableVerifiedReader(url, headers, archive["bytes"], archive["lfs_sha256"], progress) as reader:
            image_ledger = []
            with tarfile.open(fileobj=reader, mode="r|gz") as members:
                for member in members:
                    components = PurePosixPath(member.name).parts
                    if not member.isfile() or len(components) < 3 or components[-2] != "CAM_F0":
                        continue
                    relative = "/".join(components[-3:])
                    if relative not in needed:
                        continue
                    destination = raw_root / relative
                    assert destination.resolve().is_relative_to(raw_root.resolve())
                    assert 0 < member.size < 25_000_000, "Unexpected image payload size"
                    if total_retained + member.size > 450_000_000_000:
                        raise RuntimeError("Owned raw dataset would approach the 1TB cap")
                    content = members.extractfile(member).read()
                    assert len(content) == member.size
                    image_hash = hashlib.sha256(content).hexdigest()
                    if destination.exists():
                        assert hashlib.sha256(destination.read_bytes()).hexdigest() == image_hash
                    else:
                        destination.parent.mkdir(parents=True, exist_ok=True)
                        pending = destination.with_suffix(".jpg.pending")
                        pending.write_bytes(content)
                        pending.replace(destination)
                        total_retained += len(content)
                    image_ledger.append({"relative_path": relative, "bytes": len(content), "sha256": image_hash})
                    retained_images += 1
                    if retained_images % 500 == 0:
                        write_json(control / "status.json", {"status": "streaming", "archive": archive_id,
                            "completed_archives": completed_archives, "retained_images_this_process": retained_images,
                            "retained_bytes": total_retained, "network_bytes_this_archive": reader.byte_count,
                            "elapsed_seconds_this_archive": time.time() - start})
            reader.verify()
        write_json(marker, {"archive": archive, "repository": source["repository"], "revision": source["revision"],
            "archive_sha256_verified": True, "images": image_ledger, "elapsed_seconds": time.time() - start})
        completed_archives += 1
        print(json.dumps({"archive": archive_id, "images_retained": len(image_ledger),
            "retained_bytes": total_retained, "seconds": time.time() - start}), flush=True)
    write_json(control / "status.json", {"status": "requested_archives_complete", "completed_archives": completed_archives,
        "retained_bytes": total_retained, "raw_root": str(raw_root), "full_330h_corpus_complete": False})


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--catalog", type=Path, default=ROOT / "outputs/lpwm_driving_video_512x256_v1/download_catalog.json")
    parser.add_argument("--control", type=Path, default=ROOT / "outputs/lpwm_driving_video_512x256_v1/openscene_download")
    parser.add_argument("--max-archives", type=int, default=200)
    arguments = parser.parse_args()
    try:
        main(arguments)
    except Exception as error:
        arguments.control.mkdir(parents=True, exist_ok=True)
        write_json(arguments.control / "failed.json", {"error_type": type(error).__name__, "message": str(error), "time": time.time()})
        raise
