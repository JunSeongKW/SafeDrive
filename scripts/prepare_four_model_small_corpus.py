"""Freeze shared planning scenes; cache identical current/previous front images."""
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
import json
import random
import pickle
from pathlib import Path
import hashlib
import numpy as np
import cv2

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / 'outputs/four_model_small_corpus_v1/corpus'


@lru_cache(maxsize=80)
def frames_for_log(log_name):
    with (ROOT / 'dataset/navsim_logs/trainval' / (log_name + '.pkl')).open('rb') as stream:
        return pickle.load(stream)


def image_pair(record):
    paths = [Path(path) for path in record['observed_front_paths']]
    images = []
    for path in paths:
        rgb = cv2.imread(str(path))
        if rgb is None:
            raise FileNotFoundError(path)
        images.append(cv2.cvtColor(cv2.resize(rgb[28:-28], (512, 256)), cv2.COLOR_BGR2RGB))
    return np.stack(images), [str(path) for path in paths]


def main():
    cv2.setNumThreads(1)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    assert not (OUTPUT / 'manifest.json').exists(), 'Do not overwrite a frozen study manifest'
    source_path = ROOT / 'outputs/lpwm_drivor_joint_v1/scene_cache/manifest.json'
    source = json.loads(source_path.read_text())['records']
    ssl_path = ROOT / 'outputs/lpwm_driving_video_512x256_v1/corpus/local_openscene_episodes.jsonl'
    episodes = [json.loads(line) for line in ssl_path.read_text().splitlines()]
    train_groups = {row['recording'] for row in episodes if row['split'] == 'train'}
    validation_groups = {row['recording'] for row in episodes if row['split'] != 'train'}
    assert not train_groups & validation_groups
    previous_by_current = {}
    for episode in episodes:
        paths, times = episode['frame_paths'], episode['timestamps_microseconds']
        for index in range(1, len(paths)):
            if 350000 <= times[index] - times[index-1] <= 650000:
                previous_by_current[paths[index]] = paths[index-1]
    caches = json.loads((ROOT / 'outputs/lpwm_candidate_teacher_v1/metric_cache_manifest.json').read_text())
    cache_by_token = {row['token']: row['metric_cache_file'] for row in caches}
    generator = random.Random(47)
    selected = []
    for split, groups, count in [('train', train_groups, 10240), ('dev', validation_groups, 1024)]:
        candidates = [dict(row) for row in source if row['recording_group'] in groups
                      and (split == 'train' or row['token'] in cache_by_token)]
        generator.shuffle(candidates)
        picked = []
        for record in candidates:
            current = record['current_camera_paths'][0]
            previous = previous_by_current.get(current)
            if previous is None:
                continue
            record['observed_front_paths'] = [previous, current]
            record['study_split'] = split
            if record['token'] in cache_by_token:
                record['metric_cache_file'] = cache_by_token[record['token']]
            picked.append(record)
            if len(picked) == count:
                break
        assert len(picked) == count, (split, len(picked), count)
        selected.extend(picked)
    assert not {r['recording_group'] for r in selected if r['study_split']=='train'} & {
        r['recording_group'] for r in selected if r['study_split']=='dev'}
    image_cache = np.lib.format.open_memmap(OUTPUT / 'images.npy', mode='w+', dtype=np.uint8,
                                          shape=(len(selected), 2, 256, 512, 3))
    targets = {name: [] for name in ('ego', 'trajectory', 'trajectory_long')}
    with ThreadPoolExecutor(max_workers=8) as pool:
        for index, (images, paths) in enumerate(pool.map(image_pair, selected)):
            record = selected[index]
            image_cache[index] = images
            original_cache = Path(record['cache_directory'])
            for name in targets:
                values = np.load(original_cache / (name + '.npy'), mmap_mode='r')
                targets[name].append(np.array(values[record['cache_row']]))
            record.update(cache_directory=str(OUTPUT), cache_row=index, observed_front_paths=paths)
            if index % 500 == 0:
                print(json.dumps({'cached': index, 'total': len(selected)}), flush=True)
    image_cache.flush()
    for name, values in targets.items():
        np.save(OUTPUT / (name + '.npy'), np.asarray(values))
    manifest = dict(records=selected, counts={'train':10240, 'dev':1024}, seed=47,
        camera_count=1, observed_frames=2, image_width=512, image_height=256,
        order='previous,current', preprocessing='crop28top_bottom_cv2linear_resize',
        train_dev_recording_overlap=0, ssl_train_dev_recording_overlap=0,
        ssl_manifest=str(ssl_path), ssl_manifest_sha256=hashlib.sha256(ssl_path.read_bytes()).hexdigest(),
        source_scene_manifest_sha256=hashlib.sha256(source_path.read_bytes()).hexdigest(),
        selection='seeded uniform scene sampling, all feasible eligible official navtrain/navval records',
        full_navtest=False, future_images_in_planning_inputs=False)
    (OUTPUT / 'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    print('COMMON_CORPUS_READY', flush=True)


if __name__ == '__main__':
    main()
