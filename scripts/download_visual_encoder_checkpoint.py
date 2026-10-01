"""Download exactly one revision-pinned official encoder checkpoint, verify SHA256.

Only project runtime/checkpoints/ is writable. No dataset/cache bundle downloads.
An incomplete transfer remains resumable; existing verified bytes are not replaced.
"""

import hashlib
import json
import subprocess
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def main():
    config_path = PROJECT_ROOT / "configs/visual_pilot/encoder_checkpoint.json"
    config = json.loads(config_path.read_text())
    destination = PROJECT_ROOT / "runtime/checkpoints/drive_jepa" / config["filename"]
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and destination.stat().st_size > config["size_bytes"]:
        raise ValueError(
            "existing checkpoint is larger than the pinned artifact; refusing overwrite"
        )
    url = f"https://huggingface.co/datasets/{config['repository_id']}/resolve/{config['revision']}/{config['filename']}"
    if not destination.exists() or destination.stat().st_size < config["size_bytes"]:
        print(
            f"Downloading one official file: {config['filename']} ({config['size_bytes']} bytes)",
            flush=True,
        )
        subprocess.run(
            [
                "curl",
                "--fail",
                "--location",
                "--silent",
                "--show-error",
                "--retry",
                "3",
                "--continue-at",
                "-",
                "--output",
                str(destination),
                "--write-out",
                "downloaded_bytes=%{size_download} elapsed_seconds=%{time_total} bytes_per_second=%{speed_download}\n",
                url,
            ],
            check=True,
        )
    if destination.stat().st_size != config["size_bytes"]:
        raise ValueError("download size does not match pinned metadata")
    digest = hashlib.sha256()
    with destination.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    actual_sha256 = digest.hexdigest()
    if actual_sha256 != config["sha256"]:
        raise ValueError(
            "checkpoint SHA256 mismatch; file retained for diagnosis, not loaded"
        )
    verification = {
        **config,
        "local_path": str(destination),
        "verified_sha256": actual_sha256,
    }
    (destination.parent / "verified_checkpoint_manifest.json").write_text(
        json.dumps(verification, indent=2) + "\n"
    )
    print(json.dumps(verification, indent=2))


if __name__ == "__main__":
    main()
