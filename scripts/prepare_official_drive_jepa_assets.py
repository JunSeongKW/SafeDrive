"""Verify official assets and relocate only the downloaded cache's metadata paths."""

import argparse
import hashlib
import json
import shutil
import tarfile
from pathlib import Path


def compute_sha256(file_path):
    digest = hashlib.sha256()
    with file_path.open("rb") as source:
        for chunk in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_asset(file_path, specification):
    actual_size = file_path.stat().st_size
    actual_hash = compute_sha256(file_path)
    if actual_size != specification["size_bytes"] or actual_hash != specification["sha256"]:
        raise RuntimeError(f"Official asset verification failed: {file_path}")
    return {"path": str(file_path), "size_bytes": actual_size, "sha256": actual_hash}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, default=Path(__file__).resolve().parents[1])
    arguments = parser.parse_args()
    workspace = arguments.workspace.resolve()
    specification = json.loads((workspace / "configs/official_drive_jepa/reproduction_v1.json").read_text())
    output_root = workspace / "outputs/official_drive_jepa_reproduction"
    planning_path = workspace / "runtime/checkpoints/drive_jepa_official_evaluation" / specification["planning_checkpoint"]["filename"]
    encoder_path = workspace / specification["initialization_encoder_checkpoint"]["reuse_path"]
    archive_path = output_root / "downloads/metric_cache.tar"
    if not archive_path.exists():
        partial_path = archive_path.with_suffix(".tar.partial")
        verify_asset(partial_path, specification["metric_cache_archive"])
        partial_path.rename(archive_path)
    assets = {
        "planning_checkpoint": verify_asset(planning_path, specification["planning_checkpoint"]),
        "initialization_encoder": verify_asset(encoder_path, specification["initialization_encoder_checkpoint"]),
        "metric_cache_archive": verify_asset(archive_path, specification["metric_cache_archive"]),
    }
    extraction_root = output_root / "downloaded_metric_cache"
    completion_marker = extraction_root / "extraction_complete.json"
    if not completion_marker.exists():
        extraction_root.mkdir(parents=True, exist_ok=True)
        with tarfile.open(archive_path) as archive:
            members = archive.getmembers()
            for member in members:
                member_path = (extraction_root / member.name).resolve()
                if extraction_root.resolve() not in member_path.parents or not (member.isfile() or member.isdir()):
                    raise RuntimeError(f"Unsafe archive member: {member.name}")
            archive.extractall(extraction_root)
        completion_marker.write_text(json.dumps({"archive_sha256": assets["metric_cache_archive"]["sha256"]}, indent=2) + "\n")
    metadata_files = sorted(extraction_root.glob("**/metadata/*.csv"))
    if len(metadata_files) != 1:
        raise RuntimeError(f"Expected exactly one official metadata CSV, found {metadata_files}")
    metadata_path = metadata_files[0]
    preserved_metadata = output_root / "downloads/official_metadata_original.csv"
    if not preserved_metadata.exists():
        shutil.copy2(metadata_path, preserved_metadata)
    cache_root = metadata_path.parent.parent
    cache_paths_by_token = {}
    for cache_path in cache_root.glob("**/metric_cache.pkl"):
        token = cache_path.parent.name
        if token in cache_paths_by_token:
            raise RuntimeError(f"Duplicate metric cache token: {token}")
        cache_paths_by_token[token] = cache_path.resolve()
    original_lines = preserved_metadata.read_text().splitlines()
    relocated_lines = [original_lines[0]]
    for original_path in original_lines[1:]:
        token = original_path.split("/")[-2]
        if token not in cache_paths_by_token:
            raise RuntimeError(f"Missing extracted metric cache token: {token}")
        relocated_lines.append(str(cache_paths_by_token[token]))
    metadata_path.write_text("\n".join(relocated_lines) + "\n")
    assets["metric_cache_root"] = str(cache_root)
    assets["metric_cache_tokens"] = len(relocated_lines) - 1
    assets["metadata_relocation"] = {
        "original_preserved_at": str(preserved_metadata),
        "local_metadata_path": str(metadata_path),
        "only_change": "absolute file paths in a downloaded derivative, not cache contents or shared originals",
    }
    result_path = workspace / "results/official_drive_jepa_reproduction/verified_assets.json"
    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.write_text(json.dumps(assets, indent=2) + "\n")
    print(json.dumps(assets, indent=2))


if __name__ == "__main__":
    main()
