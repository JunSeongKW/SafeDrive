"""Acquire the separately distributed DrivingDojo archive 35, with source SHA.

The existing tar-stream converter handles archives 1-34 and 36-45. This ZIP
requires seekable local storage; acquiring it does not admit its frames to SSL.
"""
import argparse
import fcntl
import hashlib
import json
from pathlib import Path
import shutil
import time
import zipfile

from huggingface_hub import get_token
from lpwm_driving_video_io import ResumableVerifiedReader

ROOT = Path(__file__).resolve().parents[1]


def write_json(path, value):
    pending = path.with_suffix(path.suffix + '.pending')
    pending.write_text(json.dumps(value, indent=2) + '\n')
    pending.replace(path)


def main(arguments):
    control = arguments.output.resolve()
    control.mkdir(parents=True, exist_ok=True)
    lock = (control / 'download.lock').open('w')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    raw_root = Path('/home/user/data/processed_dataset/junseong/lpwm_driving_video_512x256_v1')
    owner = json.loads((raw_root / 'owner.json').read_text())
    assert str(ROOT) in json.dumps(owner), 'Owned raw root identity mismatch'
    catalog = json.loads(arguments.catalog.read_text())
    sources = [(record, item) for record in catalog['records'] for item in record['files']
               if 'DrivingDojo' in record['repository'] and item['filename'].endswith('.zip')]
    assert len(sources) == 1
    record, source = sources[0]
    assert source['filename'] == 'videos_35.zip' and source['bytes'] < 40_000_000_000
    registration = dict(source=source, repository=record['repository'], revision=record['revision'],
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        io_sha256=hashlib.sha256((ROOT / 'scripts/lpwm_driving_video_io.py').read_bytes()).hexdigest(),
        catalog_sha256=hashlib.sha256(arguments.catalog.read_bytes()).hexdigest())
    if (control / 'registration.json').exists():
        assert json.loads((control / 'registration.json').read_text()) == registration
    else:
        write_json(control / 'registration.json', registration)
    if (control / 'verified.json').exists():
        return
    destination = raw_root / 'drivingdojo_zip' / source['filename']
    destination.parent.mkdir(exist_ok=True)
    assert not destination.exists(), 'Inspect existing unregistered ZIP before replacement'
    # Other registered source storage is bounded at 450GB plus 2GB spools.
    assert shutil.disk_usage(raw_root).free > source['bytes'] + 100_000_000_000
    pending = destination.with_suffix('.zip.pending')
    url = f"https://huggingface.co/datasets/{record['repository']}/resolve/{record['revision']}/{source['filename']}"
    def progress(received=0, reconnections=0):
        write_json(control / 'status.json', dict(status='downloading_zip35', received_bytes=received,
            expected_bytes=source['bytes'], reconnections=reconnections, updated_unix=time.time(),
            source_archive_verified=False, converted=False, eligible_for_training=False))
    progress()
    with ResumableVerifiedReader(url, {'Authorization': 'Bearer ' + get_token()},
            source['bytes'], source['lfs_sha256'], progress) as reader, pending.open('wb') as stream:
        while content := reader.read(8 * 1024**2):
            if (control / 'pause.requested').exists():
                write_json(control / 'paused.json', dict(saved_partial_bytes=stream.tell()))
                return
            assert shutil.disk_usage(raw_root).free > 100_000_000_000
            stream.write(content)
        reader.verify()
    assert pending.stat().st_size == source['bytes']
    with zipfile.ZipFile(pending) as archive:
        names = archive.namelist()
        front_frames = sum(name.endswith('_CameraFpgaP0H120.jpg') for name in names)
        assert front_frames > 0
    pending.replace(destination)
    result = dict(status='zip35_verified_conversion_pending', source_archive_verified=True,
        sha256=source['lfs_sha256'], path=str(destination), bytes=source['bytes'],
        front_jpeg_count=front_frames, converted=False, eligible_for_training=False,
        next='convert ZIP JPEG sequences with timestamp provenance before corpus admission')
    write_json(control / 'verified.json', result)
    write_json(control / 'status.json', result)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--catalog', type=Path, required=True)
    parser.add_argument('--output', type=Path, default=ROOT / 'outputs/lpwm_driving_video_512x256_v1/drivingdojo_zip35_download')
    args = parser.parse_args()
    try:
        main(args)
    except Exception as error:
        args.output.mkdir(parents=True, exist_ok=True)
        write_json(args.output / 'failed.json', dict(error=repr(error), time=time.time()))
        raise
