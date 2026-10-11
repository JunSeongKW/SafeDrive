"""Real saved trajectories with upstream NAVSIM maps, annotations and projection."""
import argparse
import copy
import csv
import json
import os
from pathlib import Path
import pickle
import shutil
import sys
import time

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
NAVSIM_ROOT = PROJECT_ROOT / 'reference_repositories/DrivoR'
sys.path.insert(0,str(NAVSIM_ROOT))
os.environ.setdefault('NUPLAN_MAPS_ROOT',str(PROJECT_ROOT/'dataset/maps'))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from PIL import Image
from navsim.common.dataclasses import Scene, SensorConfig, Trajectory
from navsim.visualization.plots import plot_bev_with_agent, configure_bev_ax
from navsim.visualization.bev import add_configured_bev_on_ax, add_trajectory_to_bev_ax
from navsim.visualization.camera import add_camera_ax, _transform_pcs_to_images
from navsim.visualization.config import TRAJECTORY_CONFIG

from compare_lpwm_front_history_epoch1 import read_json, write_json, sha256, verify_registration

COLORS={'human':'#2ca02c','lpwm':'#f28e2b','drivor':'#1f77b4'}


class SavedTrajectoryAgent:
    """Replay exactly the trajectory from the evaluated checkpoint."""
    def __init__(self, poses):
        self.trajectory=Trajectory(poses)

    def compute_trajectory(self, agent_input):
        return self.trajectory


def trajectory_style(condition):
    style=copy.deepcopy(TRAJECTORY_CONFIG['human' if condition=='human' else 'agent'])
    style.update(line_color=COLORS[condition],marker_edge_color=COLORS[condition],
        line_width=2.5,marker_size=4,zorder=10)
    return style


def project_trajectory(axis,camera,poses,lidar_to_ego,color):
    # Flat ground in current ego coordinates; the road's elevation is unknown.
    # NAVSIM's native camera projection handles extrinsics/intrinsics/FOV.
    ego_points=np.concatenate((poses[:,:2],np.zeros((len(poses),1)),np.ones((len(poses),1))),axis=-1)
    lidar_points=(np.linalg.inv(lidar_to_ego)@ego_points.T)[:3]
    projected,valid=_transform_pcs_to_images(lidar_points,camera.sensor2lidar_rotation,
        camera.sensor2lidar_translation,camera.intrinsics,img_shape=camera.image.shape[:2])
    visible=projected.copy();visible[~valid]=np.nan
    axis.plot(visible[:,0],visible[:,1],color=color,linewidth=2.3,marker='o',markersize=3)
    return int(valid.sum())


def draw_scene_comparison(scene,frame_dict,trajectories,title,output_path):
    figure,axes=plt.subplots(1,4,figsize=(20,5))
    current=scene.frames[3]
    add_camera_ax(axes[0],current.cameras.cam_f0)
    projection_counts={}
    for condition,poses in trajectories.items():
        projection_counts[condition]=project_trajectory(axes[0],current.cameras.cam_f0,poses,
            np.asarray(frame_dict['lidar2ego']),COLORS[condition])
    axes[0].set_title('Current front camera\nGround-plane trajectory projection (approximate)')
    for index,condition in ((1,'lpwm'),(2,'drivor'),(3,None)):
        axis=axes[index]
        add_configured_bev_on_ax(axis,scene.map_api,current)
        add_trajectory_to_bev_ax(axis,Trajectory(trajectories['human']),trajectory_style('human'))
        for name in ([condition] if condition else ['lpwm','drivor']):
            add_trajectory_to_bev_ax(axis,Trajectory(trajectories[name]),trajectory_style(name))
        configure_bev_ax(axis)
        # Show the complete predicted horizon with the same metric scale.
        all_poses=np.concatenate(list(trajectories.values()))
        forward_max=max(32,float(all_poses[:,0].max())+5)
        forward_min=min(-8,float(all_poses[:,0].min())-3)
        lateral_extent=max(16,float(np.abs(all_poses[:,1]).max())+5)
        axis.set_xlim(lateral_extent,-lateral_extent);axis.set_ylim(forward_min,forward_max)
        axis.set_xlabel('Ego lateral (m), left positive');axis.set_ylabel('Ego forward (m)')
        axis.set_title('Both predictions + GT' if condition is None else ('LPWM epoch 9' if condition=='lpwm' else 'DrivoR epoch 5'))
        axis.grid(alpha=.12)
    axes[0].axis('off')
    legend=[Line2D([0],[0],color=COLORS[name],lw=3,label=label) for name,label in
        [('human','Human GT'),('lpwm','LPWM epoch 9'),('drivor','DrivoR epoch 5')]]
    figure.legend(handles=legend,loc='lower center',ncol=3,bbox_to_anchor=(.5,.015))
    figure.suptitle(title,fontsize=13)
    figure.tight_layout(rect=(0,.15,1,.90));figure.savefig(output_path,dpi=150);plt.close(figure)
    return projection_counts


def render(configuration_path):
    started=time.perf_counter()
    configuration=read_json(configuration_path)
    output=PROJECT_ROOT/configuration['output_directory']
    destination=PROJECT_ROOT/configuration['visualization_directory']
    destination.mkdir(parents=True,exist_ok=True)
    registration=verify_registration(configuration_path)
    records=read_json(PROJECT_ROOT/configuration['panel'])['records']
    predictions={};scores={};evaluations={}
    for condition in ('lpwm','drivor'):
        saved=np.load(output/condition/'predictions.npz')
        assert saved['tokens'].tolist()==[row['token'] for row in records]
        predictions[condition]=saved['trajectories']
        evaluations[condition]=read_json(output/condition/'evaluation_complete.json')
        assert evaluations[condition]['count']==1024 and evaluations[condition]['failed']==0
        with (output/condition/'scores.csv').open() as stream:
            rows=list(csv.DictReader(stream))
        assert [row['token'] for row in rows]==[row['token'] for row in records]
        scores[condition]=rows
        shutil.copy2(output/condition/'scores.csv',destination/(condition+'_scores.csv'))
    # Rank crowded scenes from current GT labels, without inspecting scores.
    raw_logs={};current_frames={};vehicle_counts={}
    for index,record in enumerate(records):
        if record['log_name'] not in raw_logs:
            with (PROJECT_ROOT/'dataset/navsim_logs/test'/(record['log_name']+'.pkl')).open('rb') as stream:
                frames=pickle.load(stream)
            raw_logs[record['log_name']]=(frames,{frame['token']:i for i,frame in enumerate(frames)})
        frames,by_token=raw_logs[record['log_name']]
        frame=frames[by_token[record['token']]]
        current_frames[index]=frame
        names=np.asarray(frame['anns']['gt_names'],dtype=str)
        boxes=np.asarray(frame['anns']['gt_boxes'])
        nearby=(np.linalg.norm(boxes[:,:2],axis=-1)<=40)&(boxes[:,0]>-5)
        vehicle_counts[index]=int(np.sum((names=='vehicle')&nearby))
    selected=[]
    for scene_type in ('straight','left_turn','right_turn'):
        indices=[index for index,row in enumerate(records) if row['scene_type']==scene_type]
        selected.append((indices[0],'first_'+scene_type))
        ranked=sorted(indices,key=lambda index:(-vehicle_counts[index],index))
        crowded=next(index for index in ranked if index!=indices[0])
        selected.append((crowded,'vehicle_rich_'+scene_type))
    assert len(selected)==configuration['visualized_scene_count']
    selection_manifest=[]
    plot_files=[]
    for index,label in selected:
        record=records[index]
        frames,by_token=raw_logs[record['log_name']]
        cursor=by_token[record['token']]
        sensor_configuration=SensorConfig.build_no_sensors()
        sensor_configuration.cam_f0=[3]
        scene=Scene.from_scene_dict_list(frames[cursor-3:cursor+9],
            PROJECT_ROOT/'dataset/sensor_blobs/test',4,8,sensor_configuration)
        assert scene.scene_metadata.initial_token==record['token']
        human=scene.get_future_trajectory().poses
        for condition in predictions:
            cached_target=np.load(output/condition/'predictions.npz')['targets'][index]
            assert np.allclose(human,cached_target,atol=1e-5)
        scene_trajectories={'human':human,**{name:values[index] for name,values in predictions.items()}}
        title=(label+' | '+record['token']+' | nearby forward vehicles '+str(vehicle_counts[index])+
            ' | PDMS: LPWM '+f"{float(scores['lpwm'][index]['score'])*100:.2f}"+
            ', DrivoR '+f"{float(scores['drivor'][index]['score'])*100:.2f}")
        path=destination/(label+'_navsim.png')
        visible=draw_scene_comparison(scene,current_frames[index],scene_trajectories,title,path)
        plot_files.append(path)
        # Also execute the high-level upstream agent visualization directly.
        # The agent replays the already evaluated trajectory; no extra inference.
        native_paths={}
        for condition in predictions:
            figure,axis=plot_bev_with_agent(scene,SavedTrajectoryAgent(predictions[condition][index]))
            axis.set_title(label+' | '+condition+' | '+record['token'])
            native_path=destination/(label+'_'+condition+'_native_navsim_bev.png')
            figure.savefig(native_path,dpi=150,bbox_inches='tight');plt.close(figure)
            native_paths[condition]=str(native_path.relative_to(PROJECT_ROOT))
        selection_manifest.append({'token':record['token'],'scene_type':record['scene_type'],
            'selection':label,'current_forward_vehicles_within40m':vehicle_counts[index],
            'log_name':record['log_name'],'predictions_cache_index':index,
            'pdms':{name:float(values[index]['score'])*100 for name,values in scores.items()},
            'camera_visible_waypoints':visible,'comparison_png':str(path.relative_to(PROJECT_ROOT)),
            'native_navsim_bev_png':native_paths})
        print(json.dumps({'rendered':label,'token':record['token']}),flush=True)
    # Overview uses the first sample per command; detailed scenes retain full size.
    overview_images=[Image.open(plot_files[index]).convert('RGB') for index in (0,2,4)]
    overview=Image.new('RGB',(overview_images[0].width,sum(image.height for image in overview_images)), 'white')
    vertical=0
    for image in overview_images:
        overview.paste(image,(0,vertical));vertical+=image.height
    overview.save(destination/'planning_comparison_overview.png')
    # Share the latest generated particle snapshot from the same checkpoint.
    stage2=PROJECT_ROOT/'outputs/lpwm_front_history_stage1_lora_v1/stage2'
    particle_folder=stage2/'particles/update_010593'
    if not particle_folder.exists():
        particle_folder=stage2/'particles/update_010500'
    if not (particle_folder/'particles_before_after_overlay.png').exists():
        particle_folder=stage2/'particles/update_010000'
    shutil.copy2(particle_folder/'particles_before_after_overlay.png',destination/'lpwm_particles_before_after_overlay.png')
    particle_diagnostic=read_json(particle_folder/'diagnostics.json')
    write_json(destination/'particle_diagnostics.json',particle_diagnostic)
    # Plot existing internal development history, explicitly separate from NAVTEST.
    history=[]
    for validation in sorted(stage2.glob('validation/*/validation_complete.json')):
        values=read_json(validation)
        updates=int(validation.parent.name.split('_')[-1])
        if updates<=10593:
            history.append({'epoch':updates//1177,'updates':updates,'pdms':values['pdms'],
                'count':values['pdms_count'],'ade_meters':values['ade_meters'],'fde_meters':values['fde_meters']})
    figure,axes=plt.subplots(1,2,figsize=(11,4))
    axes[0].plot([row['epoch'] for row in history],[row['pdms'] for row in history],marker='o',color=COLORS['lpwm'])
    axes[0].set(title='LPWM internal development history',xlabel='Completed planning epoch',ylabel='PDMS (1,021 exposed dev scenes)')
    axes[1].bar(['LPWM epoch9','DrivoR epoch5'],[evaluations[name]['pdms'] for name in ('lpwm','drivor')],color=[COLORS['lpwm'],COLORS['drivor']])
    axes[1].set(title='Same NAVTEST subset (1,024 scenes)',ylabel='PDMS (0-100)',ylim=(0,100))
    for index,name in enumerate(('lpwm','drivor')):
        axes[1].text(index,evaluations[name]['pdms']+1,f"{evaluations[name]['pdms']:.2f}",ha='center')
    figure.tight_layout();figure.savefig(destination/'training_progress_and_common_navtest.png',dpi=160);plt.close(figure)
    result={'complete':True,'count':1024,'full_navtest':False,'results':evaluations,
        'planning_epochs_by_condition':{'lpwm':9,'drivor':5},'conditions':{name:configuration['conditions'][name] for name in ('lpwm','drivor')},
        'same_panel_input_and_score_token_order_verified':True,'overlap':registration['overlap'],
        'visualized_scenes':selection_manifest,'selection_scope':'First scene and most nearby forward GT vehicles per actual command; no score-based selection',
        'camera_projection':'Native NAVSIM projection; ego ground z=0 approximation, not road elevation estimation',
        'native_visualization_functions':['navsim.visualization.plots.plot_bev_with_agent',
            'navsim.visualization.bev.add_configured_bev_on_ax','navsim.visualization.bev.add_trajectory_to_bev_ax',
            'navsim.visualization.camera.add_camera_ax','navsim.visualization.camera._transform_pcs_to_images'],
        'navsim_source':str(NAVSIM_ROOT),'internal_development_history':history,'particle_snapshot':particle_diagnostic,
        'interpretation':configuration['interpretation'],'render_seconds':time.perf_counter()-started,
        'original_training_progress_at_render':read_json(stage2/'progress.json'),
        'native_source_sha256':{name:sha256(NAVSIM_ROOT/name) for name in
            ('navsim/visualization/plots.py','navsim/visualization/bev.py','navsim/visualization/camera.py','navsim/common/dataclasses.py')},
        'source_sha256':{str(Path(__file__).relative_to(PROJECT_ROOT)):sha256(Path(__file__))}}
    write_json(destination/'visualization_report.json',result)
    shutil.copy2(configuration_path,destination/'configuration.json')
    with (destination/'summary.csv').open('w') as stream:
        writer=csv.DictWriter(stream,fieldnames=['model','planning_epoch','pdms','ade_meters','fde_meters','count'])
        writer.writeheader()
        for name,values in evaluations.items():
            writer.writerow({'model':name,'planning_epoch':result['planning_epochs_by_condition'][name],
                **{key:values[key] for key in ('pdms','ade_meters','fde_meters','count')}})
    lines=['# LPWM and DrivoR: native NAVSIM planning visualization','',
        'Same fixed NAVTEST subset, 1,024 scenes. Latest fixed LPWM checkpoint is planning epoch9; DrivoR is preserved final epoch5.','',
        '| Model | Epoch | PDMS | ADE m | FDE m |','|---|---:|---:|---:|---:|']
    for name,values in evaluations.items():
        lines.append(f"| {name} | {result['planning_epochs_by_condition'][name]} | {values['pdms']:.4f} | {values['ade_meters']:.4f} | {values['fde_meters']:.4f} |")
    lines += ['','| Actual driving command | Scenes | LPWM PDMS | DrivoR PDMS |',
        '|---|---:|---:|---:|']
    for scene_type in ('straight','left_turn','right_turn'):
        lpwm_result=evaluations['lpwm']['by_scene_type'][scene_type]
        drivor_result=evaluations['drivor']['by_scene_type'][scene_type]
        assert lpwm_result['count']==drivor_result['count']
        lines.append(f"| {scene_type} | {lpwm_result['count']} | {lpwm_result['pdms']:.4f} | {drivor_result['pdms']:.4f} |")
    lines += ['','| Official mean submetric (0-1) | LPWM | DrivoR |','|---|---:|---:|']
    for metric in ('no_at_fault_collisions','drivable_area_compliance','ego_progress',
            'time_to_collision_within_bound','comfort','driving_direction_compliance'):
        lines.append(f"| {metric} | {evaluations['lpwm']['mean_metrics'][metric]:.6f} | {evaluations['drivor']['mean_metrics'][metric]:.6f} |")
    lines += ['',configuration['interpretation'],'',
        'LPWM planning: 75,297 scenes x9 passes, 10,593 updates, effective batch64, seed2; front1/history4/128x128; reused Stage1 23,126 clips x20 epochs. DrivoR adaptation: 10,240 scenes x5 passes, 3,200 updates, effective batch16, seed47; front1/history2/512x256, public DINOv2 plus official q/v LoRA and common planner. These are different training budgets.',
        f"LPWM GPU inference: {evaluations['lpwm']['seconds']:.2f}s, peak reserved {evaluations['lpwm']['peak_reserved_bytes']/1e9:.3f}GB, largest sampled whole-card use {evaluations['lpwm']['largest_sampled_total_card_bytes']/1e9:.3f}GB <48GB. Official PDMS failures: 0/1024. Existing DrivoR predictions/scores reused byte-for-byte.",
        'Original training continues without changes to batch, optimizer, learning rate, checkpoint or queue. Current ongoing epoch is separate from the fixed epoch9 checkpoint shown here.','',
        'Comparison colors: green human GT, orange LPWM epoch9, blue DrivoR epoch5. Dots are 0.5s waypoint intervals over a 4s horizon. Camera projection assumes a flat ego ground plane; BEV positions use the saved ego-frame trajectory directly.',
        'Map and current GT object boxes come from native NAVSIM plotting. Box colors denote object types. Each network uses front camera only; GT map/boxes are visualization context, not model input.',
        'High-level native plot_bev_with_agent was also executed for both models and all six scenes. In those individual native PNGs the original library colors apply.','',
        '![Planning overview](planning_comparison_overview.png)','',
        '![Training and evaluation](training_progress_and_common_navtest.png)','',
        '![Particle change](lpwm_particles_before_after_overlay.png)','',
        'Particle panel uses separate fixed internal development scenes. Cyan is after Stage1/before Stage2; orange is the snapshot identified in particle_diagnostics.json. Dot center/size represent particle geometry/presence; rectangles are LPWM glimpses rather than detected objects.',
        '', '## Detailed scenes','']
    for scene in selection_manifest:
        lines.append('['+scene['selection']+']('+Path(scene['comparison_png']).name+') — '+scene['token'])
    lines += ['', '## Replay', '', '```bash', 'cd /rhome/junseong/PlanningAwareFuturePrediction',
        'runtime/environments/kjs-lpwm-drivor-joint/bin/python scripts/run_lpwm_drivor_navsim_visualization.py --config configs/lpwm_front_history_stage1_lora/epoch9_navsim_visualization.json', '```','']
    (destination/'report.md').write_text('\n'.join(lines))
    print(json.dumps({'complete':True,'results':str(destination),'pdms':{name:row['pdms'] for name,row in evaluations.items()}}),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--config',type=Path,required=True)
    render(parser.parse_args().config.resolve())
