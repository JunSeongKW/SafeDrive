"""Evaluate preserved LPWM systems on identical panels without training.

Keep each checkpoint's original input recipe. Explicitly report historical
training overlap; the reduced development panel is not held out for old models.
"""
import argparse
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from contextlib import nullcontext, redirect_stdout
import gc
import fcntl
import io
import json
import multiprocessing
import os
from pathlib import Path
import pickle
import sys
import time

import cv2
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from prepare_lpwm_shared_navtest_comparison import digest, read_json, write_json

OUTPUT = ROOT / 'outputs/lpwm_preserved_checkpoint_same_panels_20261009'
RESULTS = ROOT / 'results/lpwm_preserved_checkpoint_same_panels_20261009'
SMALL_STUDY = ROOT / 'outputs/four_model_small_corpus_v1'
SHARED_PANEL = ROOT / 'outputs/lpwm_shared_navtest_adapter2_primary5400_v1'
BASE_CONFIGURATION = ROOT / 'configs/lpwm_shared_navtest_comparison/adapter_epoch2_vs_primary_update5400.json'
CHECKPOINT_SOURCES = {
    'historical_adapter': ROOT / 'outputs/lpwm_adapter_original_batch_shared_v4/latest.pt',
    'historical_joint': ROOT / 'outputs/lpwm_drivor_planning_path_lora_v1/navsim_v1/latest.pt',
    'rectangular_sequential': SMALL_STUDY / 'lpwm_sequential/latest.pt',
    'rectangular_joint': SMALL_STUDY / 'lpwm_joint/latest.pt',
}
LABELS = {'historical_adapter': '128×128 | 2-stage Adapter',
          'historical_joint': '128×128 | 공동학습 LoRA (planning only)',
          'rectangular_sequential': '512×256 | 2-stage',
          'rectangular_joint': '512×256 | 공동학습 (planning + SSL)'}


def prepare_evaluation():
    import torch
    OUTPUT.mkdir(parents=True, exist_ok=True)
    RESULTS.mkdir(parents=True, exist_ok=True)
    assert read_json(SMALL_STUDY / 'queue_state.json')['status'] == 'paused'
    paused = read_json(SMALL_STUDY / 'lpwm_joint/paused.json')
    assert (SMALL_STUDY / 'pause.requested').exists()
    checkpoints = OUTPUT / 'checkpoints'
    checkpoints.mkdir(exist_ok=True)
    checkpoint_metadata = {}
    for condition, source in CHECKPOINT_SOURCES.items():
        saved = torch.load(source, map_location='cpu', weights_only=False, mmap=True)
        snapshot = checkpoints / (condition + '.pt')
        if not snapshot.exists():
            os.link(source, snapshot)
        assert digest(snapshot) == digest(source)
        checkpoint_metadata[condition] = dict(path=str(snapshot), sha256=digest(snapshot),
            source=str(source), completed_updates=saved['completed_updates'])
        if condition == 'rectangular_joint':
            assert saved['completed_updates'] == paused['completed_updates'] == 2264
            assert saved['scheduler']['last_epoch'] == 2264
            assert len(saved['rank_rng_states']) == 2
            assert all(int(state['step']) == 2264 for state in saved['optimizer']['state'].values())
        del saved
    configuration = read_json(BASE_CONFIGURATION)
    source_manifest = read_json(ROOT / 'outputs/lpwm_drivor_joint_v1/scene_cache/manifest.json')
    historical_lookup = {record['token']: record for record in source_manifest['records']}
    historical_joint_tokens = set(historical_lookup)
    historical_joint_recordings = {record['recording_group'] for record in source_manifest['records']}
    adapter_records = read_json(ROOT / 'outputs/lpwm_navsim_full_posttraining_v2/planning_manifest.json')['records']
    adapter_records = [record for record in adapter_records if record['split'] == 'train']
    historical_adapter_tokens = {record['current_frame_token'] for record in adapter_records}
    historical_adapter_recordings = {record['recording_group'] for record in adapter_records}
    stage1_records = read_json(ROOT / 'outputs/lpwm_navsim_full_posttraining_v2/manifest.json')['records']
    stage1_recordings = {record['recording_group'] for record in stage1_records if record['split'] == 'train'}
    small_manifest = read_json(SMALL_STUDY / 'corpus/manifest.json')
    small_records = [dict(record) for record in small_manifest['records'] if record['study_split'] == 'dev']
    shared_records = [dict(record) for record in read_json(SHARED_PANEL / 'scene_metadata.json')['records']]
    panels = {'reduced_dev': small_records, 'shared_navtest': shared_records}
    all_metadata = {}
    for panel_name, records in panels.items():
        assert len(records) == 1024 and len({record['token'] for record in records}) == 1024
        folder = OUTPUT / panel_name
        inputs = folder / 'inputs'
        inputs.mkdir(parents=True, exist_ok=True)
        overlap = {
            'historical_joint_planning_tokens': sum(record['token'] in historical_joint_tokens for record in records),
            'historical_joint_planning_recording_scenes': sum(record['recording_group'] in historical_joint_recordings for record in records),
            'historical_adapter_planning_tokens': sum(record['token'] in historical_adapter_tokens for record in records),
            'historical_adapter_planning_recording_scenes': sum(record['recording_group'] in historical_adapter_recordings for record in records),
            'historical_adapter_stage1_recording_scenes': sum(record['recording_group'] in stage1_recordings for record in records),
        }
        if panel_name == 'shared_navtest':
            assert all(value == 0 for value in overlap.values())
        if not (folder / 'inputs_complete.json').exists():
            if panel_name == 'shared_navtest':
                for source_name, target_name in [('primary_current_images.npy','historical_joint_images.npy'),
                                                ('primary_ego.npy','historical_joint_ego.npy'),
                                                ('adapter_observed_images.npy','historical_adapter_images.npy'),
                                                ('adapter_ego.npy','historical_adapter_ego.npy'),
                                                ('ground_truth_trajectory.npy','trajectory.npy')]:
                    source = SHARED_PANEL / 'inputs' / source_name
                    os.symlink(source, inputs / target_name)
            else:
                primary_images = np.lib.format.open_memmap(inputs/'historical_joint_images.npy', mode='w+', dtype=np.uint8,
                    shape=(1024,4,128,128,3))
                primary_ego = np.empty((1024,11), np.float32)
                source_arrays = {}
                for index, record in enumerate(records):
                    source_record = historical_lookup[record['token']]
                    cache = Path(source_record['cache_directory'])
                    if cache not in source_arrays:
                        source_arrays[cache] = (np.load(cache/'images.npy', mmap_mode='r'), np.load(cache/'ego.npy', mmap_mode='r'))
                    image_array, status_array = source_arrays[cache]
                    primary_images[index] = image_array[source_record['cache_row']]
                    primary_ego[index] = status_array[source_record['cache_row']]
                primary_images.flush()
                np.save(inputs/'historical_joint_ego.npy', primary_ego)
                rows = [record['cache_row'] for record in records]
                np.save(inputs/'trajectory.npy', np.load(SMALL_STUDY/'corpus/trajectory.npy',mmap_mode='r')[rows])
                np.save(inputs/'rectangular_images.npy', np.load(SMALL_STUDY/'corpus/images.npy',mmap_mode='r')[rows])
                np.save(inputs/'rectangular_ego.npy', primary_ego)
                del source_arrays, primary_images
            rectangular = None
            if panel_name == 'shared_navtest':
                rectangular = np.lib.format.open_memmap(inputs/'rectangular_images.npy',mode='w+',dtype=np.uint8,
                    shape=(1024,2,256,512,3))
                np.save(inputs/'rectangular_ego.npy', np.load(inputs/'historical_joint_ego.npy'))
            adapter_images = None
            if panel_name == 'reduced_dev':
                adapter_images = np.lib.format.open_memmap(inputs/'historical_adapter_images.npy',mode='w+',dtype=np.uint8,
                    shape=(1024,4,128,128,3))
                adapter_ego = np.empty((1024,8),np.float32)
            by_log = defaultdict(list)
            for index, record in enumerate(records):
                by_log[record['log_name']].append((index,record))
            def prepare_log(item):
                log_name, selected = item
                split = 'trainval' if panel_name == 'reduced_dev' else 'test'
                with (ROOT/'dataset/navsim_logs'/split/(log_name+'.pkl')).open('rb') as stream:
                    frames = pickle.load(stream)
                lookup = {frame['token']:index for index,frame in enumerate(frames)}
                for index,record in selected:
                    current_index = lookup[record['token']]
                    assert current_index >= 3
                    observed = frames[current_index-3:current_index+1]
                    if adapter_images is not None:
                        for frame_index,frame in enumerate(observed):
                            with Image.open(ROOT/'dataset/sensor_blobs'/split/frame['cams']['CAM_F0']['data_path']) as image:
                                adapter_images[index,frame_index] = cv2.resize(np.asarray(image.convert('RGB'))[28:-28],
                                    (128,128),interpolation=cv2.INTER_AREA)
                        adapter_ego[index] = np.r_[observed[-1]['driving_command'],observed[-1]['ego_dynamic_state']]
                    if rectangular is not None:
                        for frame_index,frame in enumerate(observed[-2:]):
                            with Image.open(ROOT/'dataset/sensor_blobs'/split/frame['cams']['CAM_F0']['data_path']) as image:
                                rectangular[index,frame_index] = cv2.resize(np.asarray(image.convert('RGB'))[28:-28],
                                    (512,256),interpolation=cv2.INTER_LINEAR)
                return len(selected)
            with ThreadPoolExecutor(max_workers=4) as pool:
                prepared = sum(pool.map(prepare_log, sorted(by_log.items())))
            assert prepared == 1024
            if adapter_images is not None:
                adapter_images.flush(); np.save(inputs/'historical_adapter_ego.npy',adapter_ego)
            if rectangular is not None:
                rectangular.flush()
            primary_ego = np.load(inputs/'historical_joint_ego.npy')
            adapter_ego = np.load(inputs/'historical_adapter_ego.npy')
            np.testing.assert_allclose(adapter_ego,np.c_[primary_ego[:,7:11],primary_ego[:,3:7]],atol=1e-6)
            write_json(folder/'inputs_complete.json',dict(count=1024,files_sha256={path.name:digest(path) for path in inputs.glob('*.npy')}))
        all_metadata[panel_name] = dict(count=1024,records=records,overlap=overlap,full_navtest=False,
            interpretation='Historical training overlap; diagnostic only' if panel_name=='reduced_dev' else 'Common held-out NAVSIM v1 navtest subset')
        write_json(folder/'panel.json',all_metadata[panel_name])
    registration = dict(checkpoints=checkpoint_metadata,panels={name:dict(count=metadata['count'],overlap=metadata['overlap'],
        panel_sha256=digest(OUTPUT/name/'panel.json')) for name,metadata in all_metadata.items()},
        constructor_configuration=configuration, no_training_updates=True, current_training_paused_at=2264,
        input_recipes={'historical_adapter':'1 front camera, 4 observed frames, crop28 top/bottom, INTER_AREA128, 8D ego',
                      'historical_joint':'4 current cameras, whole-image BICUBIC128, original11D ego',
                      'rectangular':'1 front camera, 2 observed frames, crop28 top/bottom, INTER_LINEAR512x256, 11D ego'},
        scope='Preserved-system comparison. Resolution, camera/history, particles, planner, training data and budget differ.',
        registered_unix=time.time())
    write_json(OUTPUT/'registration.json',registration)
    write_json(RESULTS/'registration.json',registration)
    print(json.dumps({name:metadata['overlap'] for name,metadata in all_metadata.items()}),flush=True)


def create_model(condition, trained=True):
    import torch
    registration = read_json(OUTPUT/'registration.json')
    checkpoint = Path(registration['checkpoints'][condition]['path'])
    assert digest(checkpoint) == registration['checkpoints'][condition]['sha256']
    with redirect_stdout(io.StringIO()):
        if condition.startswith('historical_'):
            configuration = registration['constructor_configuration']
            if trained:
                from evaluate_lpwm_shared_navtest_comparison import build_model
                system = 'adapter' if condition=='historical_adapter' else 'primary'
                model,_ = build_model(configuration,system,checkpoint_override=checkpoint)
            elif condition=='historical_adapter':
                from planning_aware_future_prediction.object_centric.lpwm_adapter_full_finetuning import build_adapter_or_full_planning_model
                specification=read_json(ROOT/configuration['adapter_configuration'])
                stage1=read_json(ROOT/specification['stage1_config'])
                torch.manual_seed(specification['seed'])
                model=build_adapter_or_full_planning_model(ROOT/stage1['output_directory']/'stage1/checkpoint.pt',specification,'metric_plus_world',ROOT)
            else:
                from planning_aware_future_prediction.object_centric.lpwm_drivor_planning_path_lora import LPWMDrivoRPlanningPathLoRAModel
                specification=read_json(ROOT/configuration['primary_configuration'])
                torch.manual_seed(specification['seed'])
                model=LPWMDrivoRPlanningPathLoRAModel(ROOT/specification['public_checkpoint'])
        else:
            from planning_aware_future_prediction.object_centric.small_corpus_models import CommonPlannerModel
            kind='lpwm_sequential' if condition=='rectangular_sequential' else 'lpwm_joint'
            stage1=SMALL_STUDY/'lpwm_ssl/latest.pt' if condition=='rectangular_sequential' and not trained else None
            torch.manual_seed(47)
            model=CommonPlannerModel(kind,stage1)
            if trained:
                saved=torch.load(checkpoint,map_location='cpu',weights_only=False,mmap=True)
                model.load_state_dict(saved['model'],strict=True)
                del saved
    return model.eval().requires_grad_(False)


def card_used_bytes():
    import subprocess
    physical_gpu=os.environ.get('CUDA_VISIBLE_DEVICES','0').split(',')[0]
    value=subprocess.check_output(['nvidia-smi',f'--id={physical_gpu}','--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True)
    return int(value.strip())*1024**2


def infer(condition, batch_size):
    import torch
    torch.set_num_threads(2)
    torch.cuda.set_device(0)
    device=torch.device('cuda',0)
    torch.cuda.set_per_process_memory_fraction(12*1024**3/torch.cuda.get_device_properties(device).total_memory)
    panels=('shared_navtest',) if condition.startswith('rectangular_') else ('reduced_dev','shared_navtest')
    if all((OUTPUT/panel/condition/'inference_complete.json').exists() for panel in panels):
        for panel in panels:
            metadata=read_json(OUTPUT/panel/condition/'inference_complete.json')
            assert digest(OUTPUT/panel/condition/'predictions.npz')==metadata['prediction_sha256']
        return
    model=create_model(condition).to(device)
    started=time.monotonic()
    for panel_name in ('reduced_dev','shared_navtest'):
        if condition.startswith('rectangular_') and panel_name=='reduced_dev':
            continue
        folder=OUTPUT/panel_name
        destination=folder/condition
        destination.mkdir(exist_ok=True)
        if (destination/'inference_complete.json').exists():
            continue
        prefix=condition if condition.startswith('historical_') else 'rectangular'
        images=np.load(folder/'inputs'/(prefix+'_images.npy'),mmap_mode='r')
        status=np.load(folder/'inputs'/(prefix+'_ego.npy'),mmap_mode='r')
        records=read_json(folder/'panel.json')['records']
        predictions=np.empty((1024,8,3),np.float32)
        for first in range(0,1024,batch_size):
            assert card_used_bytes()<48_000_000_000
            last=min(1024,first+batch_size)
            pixels=torch.from_numpy(np.array(images[first:last])).to(device).permute(0,1,4,2,3).float()/255
            ego=torch.from_numpy(np.array(status[first:last])).to(device)
            with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
                output=model(pixels,ego) if condition=='historical_adapter' else model({'image':pixels,'ego_status':ego[:,None]})
            prediction=output['trajectory'].float().cpu().numpy()
            assert prediction.shape==(last-first,8,3) and np.isfinite(prediction).all()
            predictions[first:last]=prediction
            write_json(destination/'progress.json',dict(completed_scenes=last,total_scenes=1024,
                elapsed_seconds=time.monotonic()-started,card_used_bytes=card_used_bytes()))
        np.savez_compressed(destination/'predictions.npz',tokens=np.asarray([r['token'] for r in records]),trajectories=predictions)
        truth=np.load(folder/'inputs/trajectory.npy')
        errors=np.linalg.norm(predictions[:,:,:2]-truth[:,:,:2],axis=-1)
        write_json(destination/'inference_complete.json',dict(count=1024,ade_m=float(errors.mean()),fde_m=float(errors[:,-1].mean()),
            checkpoint=read_json(OUTPUT/'registration.json')['checkpoints'][condition],prediction_sha256=digest(destination/'predictions.npz'),
            no_training_updates=True, future_ground_truth_input=False, peak_allocator_bytes=torch.cuda.max_memory_reserved()))
        print(json.dumps(dict(condition=condition,panel=panel_name,complete=True)),flush=True)


def score(condition, panel_name, workers):
    from score_lpwm_drivor_navtest import score_scene
    import pandas as pd
    folder=OUTPUT/panel_name
    destination=folder/condition
    if (destination/'pdms.json').exists():
        return
    panel=read_json(folder/'panel.json')
    completed=read_json(destination/'inference_complete.json')
    path=destination/'predictions.npz'
    assert digest(path)==completed['prediction_sha256']
    with np.load(path,allow_pickle=False) as prediction:
        assert prediction['tokens'].tolist()==[row['token'] for row in panel['records']]
        jobs=[(record['token'],trajectory,Path(record['metric_cache_file'])) for record,trajectory in zip(panel['records'],prediction['trajectories'])]
    with ProcessPoolExecutor(max_workers=workers,mp_context=multiprocessing.get_context('spawn')) as pool:
        rows=list(pool.map(score_scene,jobs,chunksize=8))
    assert len(rows)==1024 and all(row['valid'] for row in rows)
    assert [row['token'] for row in rows]==[row['token'] for row in panel['records']]
    frame=pd.DataFrame(rows)
    frame.to_csv(destination/'scores.csv',index=False)
    summary=dict(condition=condition,panel=panel_name,count=1024,failed=0,pdms=float(frame['score'].mean()*100),
        ade_m=completed['ade_m'],fde_m=completed['fde_m'],
        non_drivable_trajectory_count=int((frame['drivable_area_compliance']==0).sum()),
        subscores=frame.select_dtypes('number').mean().to_dict(),
        training_overlap=panel['overlap'],interpretation=panel['interpretation'],full_navtest=False,
        checkpoint=completed['checkpoint'])
    write_json(destination/'pdms.json',summary)
    write_json(RESULTS/panel_name/(condition+'.pdms.json'),summary)
    print(json.dumps(summary),flush=True)


def capture_visuals(condition):
    import torch
    torch.set_num_threads(2)
    torch.cuda.set_device(0)
    device=torch.device('cuda',0)
    torch.cuda.set_per_process_memory_fraction(12*1024**3/torch.cuda.get_device_properties(device).total_memory)
    completed_path=OUTPUT/(condition+'_visuals.json')
    if completed_path.exists():
        completed=read_json(completed_path)
        assert digest(OUTPUT/(condition+'_visuals.npz'))==completed['array_sha256']
        assert completed['checkpoint']==read_json(OUTPUT/'registration.json')['checkpoints'][condition]
        return
    folder=OUTPUT/'reduced_dev'
    prefix=condition if condition.startswith('historical_') else 'rectangular'
    pixels=np.load(folder/'inputs'/(prefix+'_images.npy'),mmap_mode='r')
    ego_array=np.load(folder/'inputs'/(prefix+'_ego.npy'),mmap_mode='r')
    captures={}
    for label,trained in [('before',False),('after',True)]:
        model=create_model(condition,trained).to(device)
        for scene in range(3):
            assert card_used_bytes()<48_000_000_000
            observed=torch.from_numpy(np.array(pixels[scene:scene+1])).to(device).permute(0,1,4,2,3).float()/255
            ego=torch.from_numpy(np.array(ego_array[scene:scene+1])).to(device)
            if condition=='historical_joint':
                world=model.particle_encoder.world_model
                observed=observed[:,:1]
                model.particle_encoder.active_command=ego[:,7:11]
                command_context=nullcontext()
            elif condition=='historical_adapter':
                world=model.world_model
                command_context=model.encoder_command(ego)
            else:
                world=model.backbone.world_model
                cnn=world.encoder_module.particle_enc.particle_attribute_enc.cnn
                cnn.rectangular_command_modulation=model.backbone.command_modulation(ego[:,7:11]).repeat_interleave(2*16,0)
                command_context=nullcontext()
            with torch.inference_mode(),torch.autocast('cuda',enabled=False),command_context:
                inputs=observed*2-1 if world.normalize_rgb else observed
                encoded=world.encode_all(inputs,deterministic=True)
                states=[encoded[key] for key in ('z','z_scale','z_features','obj_on','z_depth','z_bg_features')]
                filter_key=encoded['z_base_var'].sum(-1) if world.filter_particles_in_decoder and world.n_kp_enc!=world.n_kp_dec else None
                decoded=world.decode_all(*states,z_ctx=encoded['z_context'],filter_key=filter_key)['rec_rgb'].reshape_as(observed)
                geometry=torch.cat((encoded['z'],encoded['z_scale'].sigmoid(),encoded['obj_on']),-1)[0,-1].cpu().numpy()
                captures[f'scene{scene}_{label}_geometry']=geometry
                captures[f'scene{scene}_{label}_reconstruction']=decoded[0,-1].cpu().numpy()
                captures[f'scene{scene}_input']=observed[0,-1].cpu().numpy()
                assert np.isfinite(geometry).all() and torch.isfinite(decoded).all()
                if label=='after' and scene==0:
                    captures['encoded_particles']=np.asarray(world.n_kp_enc)
                    captures['decoded_particles']=np.asarray(world.n_kp_dec)
            if condition=='historical_joint':
                model.particle_encoder.active_command=None
            elif condition.startswith('rectangular_'):
                cnn.rectangular_command_modulation=None
        del model,world,encoded,decoded,states
        gc.collect(); torch.cuda.empty_cache()
    path=OUTPUT/(condition+'_visuals.npz')
    np.savez_compressed(path,**captures)
    write_json(OUTPUT/(condition+'_visuals.json'),dict(condition=condition,count=3,
        selection='First3 saved reduced development scenes, fixed before training',no_training_updates=True,
        before='Stage1-adapted LPWM before planning' if condition.endswith(('adapter','sequential')) else 'Public LPWM before local planning',
        array_sha256=digest(path), checkpoint=read_json(OUTPUT/'registration.json')['checkpoints'][condition],
        decoded_particles=int(captures['decoded_particles']),encoded_particles=int(captures['encoded_particles'])))
    print(json.dumps(dict(condition=condition,visuals_complete=True)),flush=True)


def render_visuals():
    from PIL import ImageDraw
    from visualize_small_corpus_planning_particles import compose, BEFORE_COLOR, AFTER_COLOR
    bundles={condition:np.load(OUTPUT/(condition+'_visuals.npz'),allow_pickle=False) for condition in LABELS}
    panel=read_json(OUTPUT/'reduced_dev/panel.json')
    registration=read_json(OUTPUT/'registration.json')
    metrics={}
    def make_rgb(array):
        return Image.fromarray(np.rint(array.transpose(1,2,0).clip(0,1)*255).astype(np.uint8))
    def draw_snapshot(rgb, geometry, selected, color):
        image=rgb.convert('RGBA')
        height,width=geometry.shape[0],rgb.width
        native_height=rgb.height
        centers=(geometry[:,:2][:,::-1]+1)*[width/2,native_height/2]-.5
        sizes=geometry[:,2:4][:,::-1]*[width,native_height]
        layer=Image.new('RGBA',image.size)
        draw=ImageDraw.Draw(layer)
        for index,(center,size,presence) in enumerate(zip(centers,sizes,geometry[:,4])):
            horizontal,vertical=center; box_width,box_height=size
            radius=(1+1.4*presence) if width==128 else (2+3*presence)
            draw.ellipse((horizontal-radius,vertical-radius,horizontal+radius,vertical+radius),fill=(*color,210))
            if index in selected:
                draw.rectangle((horizontal-box_width/2,vertical-box_height/2,horizontal+box_width/2,vertical+box_height/2),outline=(*color,170),width=1)
        return Image.alpha_composite(image,layer).convert('RGB')
    for scene in range(3):
        particle_rows,reconstruction_rows=[],[]
        for condition,bundle in bundles.items():
            actual=make_rgb(bundle[f'scene{scene}_input'])
            before=bundle[f'scene{scene}_before_geometry']; after=bundle[f'scene{scene}_after_geometry']
            selected=set(np.argsort(-before[:,4],kind='stable')[:16].tolist())
            before_image=draw_snapshot(actual,before,selected,BEFORE_COLOR)
            after_image=draw_snapshot(actual,after,selected,AFTER_COLOR)
            overlay=draw_snapshot(before_image,after,selected,AFTER_COLOR)
            dimensions=np.asarray([actual.height/2,actual.width/2])
            shift=np.linalg.norm((after[:,:2]-before[:,:2])*dimensions,axis=-1)
            normalized_shift=np.linalg.norm((after[:,:2]-before[:,:2])/2,axis=-1)*100
            size_change=np.abs(after[:,2:4]-before[:,2:4])/before[:,2:4]*100
            metrics.setdefault(condition,[]).append(dict(scene=scene+1,center_shift_native_input_pixels=float(shift.mean()),
                center_shift_normalized_coordinates_percent=float(normalized_shift.mean()),
                mean_absolute_width_height_change_percent=float(size_change.mean()),
                reconstruction_mse_before=float(np.mean((bundle[f'scene{scene}_before_reconstruction']-bundle[f'scene{scene}_input'])**2)),
                reconstruction_mse_after=float(np.mean((bundle[f'scene{scene}_after_reconstruction']-bundle[f'scene{scene}_input'])**2))))
            steps=registration['checkpoints'][condition]['completed_updates']
            caption=f"{LABELS[condition]} | {steps:,} update | {before.shape[0]} FG | 중심이동 {shift.mean():.2f} native px / 폭높이 {size_change.mean():.1f}%"
            expand=lambda image:image.resize((512,256),Image.Resampling.NEAREST)
            particle_rows.append((caption,[expand(actual),expand(before_image),expand(after_image),expand(overlay)]))
            reconstruction_rows.append((caption,[expand(actual),expand(make_rgb(bundle[f'scene{scene}_before_reconstruction'])),
                expand(make_rgb(bundle[f'scene{scene}_after_reconstruction']))]))
        footer='청록 = planning 전, 주황 = planning 후. 점 = 중심/presence, 박스 = 초기 top16 glimpse. 검출/attention 아님.\n128 출력은 nearest로 표시 확장했습니다. Crop·관측·particle 수·학습량·planner가 서로 다릅니다.\n과거 학습 장면과 겹치는 보조 진단입니다. 좌표 변화만으로 객체 정보 보존/성능 향상을 입증하지 않습니다.'
        compose(RESULTS/f'scene{scene+1}_particle_before_after_overlay.png',f'같은 장면 {scene+1}: 이전128 모델과 현재512 모델의 particle 변화',
            f"token {panel['records'][scene]['token']} | 각 checkpoint의 실제 학습 입력을 유지했습니다.",
            ['현재 전방 입력','planning 전','planning 후','전후 겹침'],particle_rows,footer)
        compose(RESULTS/f'scene{scene+1}_reconstruction_before_after.png',f'같은 장면 {scene+1}: 현재 영상의 reconstruction 학습 전후',
            '각 모델의 관측 프레임만 인코딩해 복원했습니다. 미래 GT 영상 입력/선명화 없음.',
            ['각 모델의 실제 입력','planning 전 복원','planning 후 복원'],reconstruction_rows,footer)
    write_json(RESULTS/'visualization_report.json',dict(metrics=metrics,source_sha256={str(OUTPUT/(condition+'_visuals.npz')):digest(OUTPUT/(condition+'_visuals.npz')) for condition in LABELS},
        scope='First3 fixed reduced-dev scenes; historical training overlap. Representation comparison, not resolution-only causal ablation.',
        files=[str(path) for path in sorted(RESULTS.glob('*.png'))]))
    print(json.dumps(dict(visualization_complete=True,output=str(RESULTS))),flush=True)


def summarize_results():
    """Verify common scenes and retain scenario scores and paired intervals."""
    import pandas as pd
    from evaluate_lpwm_shared_navtest_comparison import paired_interval

    registration=read_json(OUTPUT/'registration.json')
    checkpoint_checks={}
    for condition,metadata in registration['checkpoints'].items():
        checkpoint_checks[condition]=dict(
            snapshot_unchanged=digest(Path(metadata['path']))==metadata['sha256'],
            source_unchanged=digest(Path(metadata['source']))==metadata['sha256'])
        assert all(checkpoint_checks[condition].values())
    assert (SMALL_STUDY/'pause.requested').exists()
    assert read_json(SMALL_STUDY/'queue_state.json')['status']=='paused'
    assert not read_json(SMALL_STUDY/'queue_state.json')['active_jobs']
    summaries={}
    score_rows={}
    source_hashes={}
    for panel_name in ('shared_navtest','reduced_dev'):
        panel=read_json(OUTPUT/panel_name/'panel.json')
        records_by_token={record['token']:record for record in panel['records']}
        ground_truth=np.load(OUTPUT/panel_name/'inputs/trajectory.npy',mmap_mode='r')
        scene_types={record['token']:record.get('scene_type',
            'left_turn' if ground_truth[index,-1,2]>.25 else
            'right_turn' if ground_truth[index,-1,2]<-.25 else 'straight')
            for index,record in enumerate(panel['records'])}
        summaries[panel_name]={}
        for condition in LABELS:
            score_path=OUTPUT/panel_name/condition/'scores.csv'
            if not score_path.exists():
                continue
            summary=read_json(score_path.parent/'pdms.json')
            frame=pd.read_csv(score_path,dtype={'token':str})
            assert frame['token'].tolist()==[record['token'] for record in panel['records']]
            assert len(frame)==1024 and frame['valid'].all()
            assert np.isclose(frame['score'].mean()*100,summary['pdms'],atol=1e-10)
            frame['recording_group']=frame['token'].map(lambda token:records_by_token[token]['recording_group'])
            frame['scene_type']=frame['token'].map(scene_types)
            frame['pdms_points']=frame['score']*100
            scenario_scores={}
            for scene_type,selected in frame.groupby('scene_type'):
                scenario_scores[scene_type]=dict(count=len(selected),pdms=float(selected['pdms_points'].mean()),
                    drivable_area_compliance=float(selected['drivable_area_compliance'].mean()),
                    ego_progress=float(selected['ego_progress'].mean()))
            summaries[panel_name][condition]={**summary,'scenario_scores':scenario_scores}
            score_rows[(panel_name,condition)]=frame.to_dict('records')
            source_hashes[str(score_path)]=digest(score_path)
    differences={}
    for first,second in [('historical_adapter','rectangular_sequential'),
                         ('historical_joint','rectangular_joint'),
                         ('historical_adapter','historical_joint')]:
        comparison_name=first+'_minus_'+second
        differences[comparison_name]=paired_interval(score_rows[('shared_navtest',first)],
            score_rows[('shared_navtest',second)],'pdms_points',5000,71)
    visual_metrics=read_json(RESULTS/'visualization_report.json')['metrics']
    mean_visual_metrics={condition:{metric:float(np.mean([row[metric] for row in rows]))
        for metric in rows[0] if metric!='scene'} for condition,rows in visual_metrics.items()}
    result=dict(summaries=summaries,paired_recording_bootstrap=differences,
        bootstrap_repeats=5000,bootstrap_seed=71,mean_visual_metrics_three_scenes=mean_visual_metrics,
        checkpoint_integrity=checkpoint_checks,source_sha256=source_hashes,
        current_training_remains_paused=True,
        scene_type_definition='Expert terminal yaw above+0.25rad: left; below-0.25rad: right; otherwise straight. Evaluation stratification only.',
        interpretation='Common held-out navtest subset of1024scenes; original inputs and unequal training budgets retained. Not a resolution-only comparison.',
        visualization_scope='Three fixed reduced-dev scenes with historical training overlap; current reconstruction, not future prediction.')
    write_json(RESULTS/'comparison_summary.json',result)
    print(json.dumps(dict(pdms={condition:summary['pdms'] for condition,summary in summaries['shared_navtest'].items()},
        paired_recording_bootstrap=differences,current_training_remains_paused=True)),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--phase',choices=['prepare','infer','score','capture','render','summarize'],required=True)
    parser.add_argument('--condition',choices=list(LABELS))
    parser.add_argument('--panel',choices=['reduced_dev','shared_navtest'],default='reduced_dev')
    parser.add_argument('--batch-size',type=int,default=4)
    parser.add_argument('--workers',type=int,default=4)
    arguments=parser.parse_args()
    # CUDA/LPWM and the official NAVSIM scorer have separate preserved envs.
    # Re-exec scoring in the same environment used for all earlier PDMS runs.
    if arguments.phase=='score' and sys.version_info[:2] != (3,9):
        scorer_python=ROOT/'runtime/environments/drive_jepa_official_evaluation/bin/python'
        scorer_environment=dict(os.environ,LD_LIBRARY_PATH=str(scorer_python.parent.parent/'lib'),
                                OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
        os.execve(str(scorer_python),[str(scorer_python),'-u',str(Path(__file__).resolve()),*sys.argv[1:]],scorer_environment)
    cv2.setNumThreads(1)
    if arguments.phase=='prepare': prepare_evaluation()
    elif arguments.phase=='infer':
        with (OUTPUT/(arguments.condition+'.infer.lock')).open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX)
            infer(arguments.condition,arguments.batch_size)
    elif arguments.phase=='score': score(arguments.condition,arguments.panel,arguments.workers)
    elif arguments.phase=='capture':
        with (OUTPUT/(arguments.condition+'.capture.lock')).open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX)
            capture_visuals(arguments.condition)
    elif arguments.phase=='render': render_visuals()
    else: summarize_results()
