"""Generate sealed trajectories before the official CPU metric evaluators run."""
import argparse
import json
from pathlib import Path
import sys
import numpy as np
import torch

PROJECT_ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(PROJECT_ROOT/'src'))
from planning_aware_future_prediction.object_centric.lpwm_drivor_joint import LPWMDrivoRJointModel
from train_lpwm_drivor_joint import configure_allocator, digest, write_json


def run(arguments):
    torch.set_num_threads(2)
    torch.cuda.set_device(0)
    configure_allocator(0)
    configuration=json.loads(arguments.config.read_text())
    output=PROJECT_ROOT/configuration['output_directory']/'evaluation'/arguments.split
    output.mkdir(parents=True,exist_ok=True)
    training_root=PROJECT_ROOT/configuration['output_directory']
    completion=json.loads((training_root/'training_complete.json').read_text())
    checkpoint=training_root/'latest.pt'
    assert digest(checkpoint)==completion['checkpoint_sha256']
    assert completion['completed_updates']==completion['total_updates']
    inputs_root=PROJECT_ROOT/'outputs/lpwm_drivor_joint_v1/evaluation_inputs'/arguments.split
    manifest=json.loads((inputs_root/'manifest.json').read_text())
    tokens=manifest['tokens']
    assert manifest['complete']
    images=np.load(inputs_root/'images.npy',mmap_mode='r')
    ego=np.load(inputs_root/'ego.npy',mmap_mode='r')
    state=torch.load(checkpoint,map_location='cpu',weights_only=False)
    model=LPWMDrivoRJointModel(PROJECT_ROOT/configuration['public_checkpoint'],configuration['benchmark']).cuda().eval()
    model.load_state_dict(state['model'],strict=True)
    del state
    trajectories=[]
    for first in range(0,len(tokens),8):
        features={'image':torch.from_numpy(np.array(images[first:first+8])).cuda().permute(0,1,4,2,3).float()/255,
                  'ego_status':torch.from_numpy(np.array(ego[first:first+8])).cuda()[:,None]}
        with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
            prediction=model(features)['trajectory'].float().cpu().numpy()
        assert np.isfinite(prediction).all()
        trajectories.extend(prediction)
        if first%256==0:
            write_json(output/'inference_progress.json',{'completed':len(trajectories),'total':len(tokens)})
    np.savez(output/'predictions.npz',tokens=np.asarray(tokens),trajectories=np.asarray(trajectories))
    write_json(output/'inference_complete.json',{'complete':True,'count':len(tokens),
        'checkpoint_sha256':completion['checkpoint_sha256'],'prediction_sha256':digest(output/'predictions.npz')})


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--config',type=Path,required=True)
    parser.add_argument('--split',required=True,choices=['navtest','warmup_two_stage','navhard_two_stage'])
    run(parser.parse_args())
