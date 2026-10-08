"""User-requested cleanup of this project's October8 driving downloads only.

Default is a read-only inventory. Quarantine and purge require explicit actions.
Shared originals, current manifests, checkpoints, and provenance are preserved.
"""
import argparse
from collections import defaultdict
from datetime import datetime, timezone, timedelta
import hashlib
import json
import os
from pathlib import Path
import shutil
import stat

ROOT = Path(__file__).resolve().parents[1]
OWNED_RAW = Path('/home/user/data/processed_dataset/junseong/lpwm_driving_video_512x256_v1')
CONVERTED = ROOT / 'outputs/lpwm_driving_video_512x256_v1/converted_external_videos'
REPORT = ROOT / 'outputs/recent_driving_download_cleanup_20261008'
SSL_MANIFEST = ROOT / 'outputs/lpwm_driving_video_512x256_v1/corpus/local_openscene_episodes.jsonl'
PLANNING_MANIFEST = ROOT / 'outputs/four_model_small_corpus_v1/corpus/manifest.json'
CUTOFF = datetime(2026, 10, 8, 0, tzinfo=timezone(timedelta(hours=9))).timestamp()
TARGETS = {
    'recent_covla_converted': CONVERTED / 'covla',
    'recent_drivingdojo_converted': CONVERTED / 'drivingdojo',
    'recent_drivingdojo_zip': OWNED_RAW / 'drivingdojo_zip',
    'recent_covla_transfer_buffer': OWNED_RAW / 'video_transfer_spool/covla',
    'recent_drivingdojo_transfer_buffer': OWNED_RAW / 'video_transfer_spool/drivingdojo',
    'recent_openscene_supplement': OWNED_RAW / 'openscene_front_trainval',
}


def write(name, value):
    REPORT.mkdir(parents=True, exist_ok=True)
    path = REPORT / name
    pending = path.with_suffix(path.suffix + '.pending')
    pending.write_text(json.dumps(value, indent=2) + '\n')
    pending.replace(path)


def is_under(path, directory):
    return path == directory or directory in path.parents


def preserved_sources():
    registration = json.loads((ROOT / 'outputs/four_model_small_corpus_v1/queue_registration.json').read_text())
    for name, expected in registration['source_sha256'].items():
        assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == expected, name
    return len(registration['source_sha256'])


def protected_paths():
    paths = set()
    for line in SSL_MANIFEST.read_text().splitlines():
        paths.update(json.loads(line)['frame_paths'])
    planning = json.loads(PLANNING_MANIFEST.read_text())
    for record in planning['records']:
        paths.update(record['observed_front_paths'])
        if 'metric_cache_file' in record:
            paths.add(record['metric_cache_file'])
    corpus = PLANNING_MANIFEST.parent
    paths.update(str(corpus / (name + '.npy')) for name in ('images', 'ego', 'trajectory', 'trajectory_long'))
    paths.update([str(SSL_MANIFEST), str(PLANNING_MANIFEST)])
    return paths


def check_dependencies():
    candidates = [path.resolve() for path in TARGETS.values()]
    # Resolve each directory once; inspect actual directory entries to detect
    # file-level symlinks without performing100k repeated ancestor lookups.
    by_parent = defaultdict(set)
    for value in protected_paths():
        path = Path(value)
        by_parent[path.parent].add(path.name)
    checked = 0
    resolved_roots = set()
    for parent, names in by_parent.items():
        real_parent = parent.resolve(strict=True)
        resolved_roots.add(str(real_parent))
        assert not any(is_under(real_parent, target) for target in candidates), str(parent)
        with os.scandir(real_parent) as entries:
            found = {entry.name: entry for entry in entries if entry.name in names}
        assert names == set(found), (str(parent), sorted(names-set(found))[:5])
        for name, entry in found.items():
            if entry.is_symlink():
                actual = Path(entry.path).resolve(strict=True)
                assert not any(is_under(actual, target) for target in candidates), str(actual)
            assert entry.is_file(), entry.path
            checked += 1
    return dict(protected_file_count=checked, protected_directory_count=len(by_parent),
        deletion_target_overlap=0, all_references_exist=True,
        manifest_sha256={str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
                         for path in (SSL_MANIFEST, PLANNING_MANIFEST)},
        scientific_source_hashes_verified=preserved_sources())


def tree_inventory(path):
    assert path.is_dir() and not path.is_symlink(), str(path)
    files, directories, logical_bytes, allocated_bytes = 0, 0, 0, 0
    oldest, newest = float('inf'), 0.
    for parent, child_directories, child_files in os.walk(path, followlinks=False):
        for name in child_directories + child_files:
            child = Path(parent) / name
            metadata = child.lstat()
            assert not stat.S_ISLNK(metadata.st_mode), 'Unexpected link in recent downloads: ' + str(child)
            assert metadata.st_uid == os.getuid(), 'Unexpected owner: ' + str(child)
            assert metadata.st_mtime >= CUTOFF, 'Pre-existing file outside requested time range: ' + str(child)
            assert stat.S_ISREG(metadata.st_mode) or stat.S_ISDIR(metadata.st_mode), str(child)
            allocated_bytes += metadata.st_blocks*512
            if stat.S_ISREG(metadata.st_mode):
                files += 1
                logical_bytes += metadata.st_size
                oldest = min(oldest, metadata.st_mtime)
                newest = max(newest, metadata.st_mtime)
            else:
                directories += 1
    root_stat = path.stat()
    assert root_stat.st_uid == os.getuid() and root_stat.st_mtime >= CUTOFF
    allocated_bytes += root_stat.st_blocks*512
    return dict(files=files, directories=directories+1, logical_bytes=logical_bytes,
                allocated_bytes=allocated_bytes, oldest_file_unix=oldest if files else None,
                newest_file_unix=newest if files else None, root_inode=root_stat.st_ino,
                root_device=root_stat.st_dev)


def active_download_users(paths):
    references = []
    canonical = [str(path.resolve()) for path in paths]
    download_scripts = ('download_lpwm_openscene_front.py', 'prepare_lpwm_external_driving_videos.py',
                        'prepare_lpwm_drivingdojo_frames.py', 'download_lpwm_drivingdojo_zip.py')
    for process in Path('/proc').iterdir():
        if not process.name.isdigit() or int(process.name) == os.getpid():
            continue
        try:
            if process.stat().st_uid != os.getuid():
                continue
            command = (process / 'cmdline').read_bytes().replace(b'\0', b' ').decode()
            if any(script in command for script in download_scripts):
                references.append(dict(pid=int(process.name), kind='download_process', command=command))
            for handle in (process / 'fd').iterdir():
                try:
                    target = os.readlink(handle)
                except (FileNotFoundError, PermissionError):
                    continue
                if any(target == prefix or target.startswith(prefix+'/') for prefix in canonical):
                    references.append(dict(pid=int(process.name), kind='open_file', path=target))
        except (FileNotFoundError, ProcessLookupError, PermissionError):
            continue
    assert not references, references
    return dict(owned_download_processes=0, owned_open_file_references=0)


def audit():
    assert json.loads((OWNED_RAW / 'owner.json').read_text())['project'] == str(ROOT)
    evidence = check_dependencies()
    targets = []
    for label, path in TARGETS.items():
        actual = path.resolve(strict=True)
        assert not is_under(actual, Path('/home/user/data/Dataset').resolve())
        assert is_under(actual, OWNED_RAW.resolve()) or is_under(actual, CONVERTED.resolve())
        parent = OWNED_RAW if is_under(path, OWNED_RAW) else CONVERTED
        quarantine = parent / '.trash-recent-unused-downloads-20261008' / label
        targets.append(dict(label=label, path=str(path), canonical_path=str(actual),
                            quarantine=str(quarantine), **tree_inventory(path)))
    result = dict(user_authorized=True, scope='Only newly acquired October8 driving datasets',
                  targets=targets, dependency_check=evidence,
                  total_allocated_bytes=sum(row['allocated_bytes'] for row in targets))
    write('audit.json', result)
    print(json.dumps({key: value for key, value in result.items() if key != 'dependency_check'}), flush=True)


def quarantine():
    plan = json.loads((REPORT / 'audit.json').read_text())
    dependencies = check_dependencies()
    assert dependencies['manifest_sha256'] == plan['dependency_check']['manifest_sha256']
    handles = active_download_users(list(TARGETS.values()))
    moved = []
    for row in plan['targets']:
        source, destination = Path(row['path']), Path(row['quarantine'])
        assert source == TARGETS[row['label']]
        assert source.resolve(strict=True) == Path(row['canonical_path'])
        current = tree_inventory(source)
        assert current == {key: row[key] for key in current}, 'Files changed since audit: ' + str(source)
        assert not destination.exists()
        destination.parent.mkdir(exist_ok=True)
        source.rename(destination)
        moved.append(row)
        write('quarantine.json', dict(targets=moved, complete=len(moved)==len(plan['targets']),
             dependency_check=dependencies, open_handles_check=handles))
    after = check_dependencies()
    write('post_quarantine_dependencies.json', after)
    print(json.dumps(dict(quarantined=len(moved), dependencies_preserved=after['all_references_exist'])), flush=True)


def purge():
    plan = json.loads((REPORT / 'quarantine.json').read_text())
    assert plan['complete']
    dependencies = check_dependencies()
    assert dependencies['manifest_sha256'] == plan['dependency_check']['manifest_sha256']
    active_download_users([Path(row['quarantine']) for row in plan['targets']])
    removed = []
    for row in plan['targets']:
        original, path = Path(row['path']), Path(row['quarantine'])
        assert original == TARGETS[row['label']] and not original.exists()
        assert path.name == row['label'] and path.parent.name == '.trash-recent-unused-downloads-20261008'
        assert path.parent.parent in (OWNED_RAW, CONVERTED)
        current = tree_inventory(path)
        assert current == {key: row[key] for key in current}, str(path)
        shutil.rmtree(path)
        removed.append(row)
        write('purge_progress.json', dict(removed=removed, complete=len(removed)==len(plan['targets'])))
    for parent in (OWNED_RAW, CONVERTED):
        trash = parent / '.trash-recent-unused-downloads-20261008'
        if trash.exists():
            trash.rmdir()
    after = check_dependencies()
    result = dict(complete=True, removed=removed,
        deleted_logical_bytes=sum(row['logical_bytes'] for row in removed),
        deleted_allocated_bytes=sum(row['allocated_bytes'] for row in removed),
        protected_dependencies=after, current_training_queue_preserved=True,
        note='File allocation bytes; concurrent activity can change filesystem free-space readings')
    write('cleanup_complete.json', result)
    print(json.dumps(dict(complete=True, deleted_allocated_bytes=result['deleted_allocated_bytes'],
                         protected_file_count=after['protected_file_count'])), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--action', choices=['audit', 'quarantine', 'purge'], default='audit')
    arguments = parser.parse_args()
    {'audit': audit, 'quarantine': quarantine, 'purge': purge}[arguments.action]()
