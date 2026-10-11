#!/usr/bin/env python
"""Read-only attention evolution on the same preregistered examples."""
import argparse
import json
from pathlib import Path
import sys

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plotting
import numpy as np
import torch

PROJECT_ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(PROJECT_ROOT/"scripts"))
from run_metric_visual_evidence_pilot import cache_tensors,initialize_reader,load_artifacts,verify_registration
from diagnose_metric_visual_evidence import blend_attention,map_attention_to_original
from calibrate_frozen_drivor_scorer import file_sha256,write_json


def main(configuration_path):
    configuration=json.loads(configuration_path.read_text())
    verify_registration(configuration)
    torch.set_num_threads(configuration["cpu_threads"])
    _,records,_,_,output,result=load_artifacts(configuration)
    diagnostic_rows=json.loads((result/"diagnostic_subset.json").read_text())["rows"]
    examples=json.loads((result/"visualization_metadata.json").read_text())["examples"]
    traces=[]
    maps={}
    for step in (0,*configuration["checkpoint_steps"]):
        model=initialize_reader(configuration,"metric_specific").eval()
        path=output/("common_initialization.pt" if step==0 else "metric_specific_step"+str(step)+".pt")
        saved=torch.load(path,map_location="cpu",weights_only=False)
        model.load_state_dict(saved if step==0 else saved["model"],strict=True)
        metric_distances=[]
        entropies=[]
        for row in diagnostic_rows:
            with torch.no_grad():
                _,attention=model(**cache_tensors(configuration,[row],torch.device("cpu")),capture_attention=True)
            metric_distances.append(float((attention[:,:,0]-attention[:,:,1]).abs().sum(-1).mul(.5).mean()))
            entropies.append(float(-(attention*attention.clamp_min(1e-30).log()).sum(-1).mean()/np.log(1406)))
            example=next((example for example in examples if example["pilot_row"]==row),None)
            if example is not None:
                maps[(row,step)]=attention[0,example["candidate"],:,703:].numpy()*1406
        traces.append({"step":step,"checkpoint_sha256":file_sha256(path),"mean_NC_DAC_attention_total_variation":float(np.mean(metric_distances)),
            "mean_normalized_attention_entropy":float(np.mean(entropies)),"diagnostic_scenes":len(diagnostic_rows)})
    display_maximum=max(1.,float(np.percentile(np.concatenate([values.flatten() for values in maps.values()]),99.5)))
    figure,axes=plotting.subplots(len(examples),5,figsize=(20,3.3*len(examples)),squeeze=False)
    panel_settings=((0,0),(100,0),(500,0),(500,1))
    for panel,example in enumerate(examples):
        record=records[example["pilot_row"]]
        raw=cv2.cvtColor(cv2.imread(record["observed_front_paths"][-1]),cv2.COLOR_BGR2RGB)
        axes[panel,0].imshow(raw);axes[panel,0].axis("off");axes[panel,0].set_title(f"Original / command {example['command_index']}")
        for column,(step,metric_index) in enumerate(panel_settings,1):
            projected=map_attention_to_original(maps[(example["pilot_row"],step)][metric_index],raw.shape)
            axes[panel,column].imshow(blend_attention(raw,projected,[255,0,0] if metric_index==0 else [0,70,255],display_maximum))
            axes[panel,column].axis("off");axes[panel,column].set_title(f"{'NC' if metric_index==0 else 'DAC'} / step {step}")
    figure.suptitle(f"Same image and same candidate: initialization -> selected100 -> last500\nAttention density uniform=1; all panels share display max={display_maximum:.3f}. Last500 is a representation diagnostic, not the selected evaluation checkpoint.")
    figure.tight_layout(rect=(0,0,1,.93));figure.savefig(result/"nc_dac_attention_training_evolution.png",dpi=160);plotting.close(figure)
    write_json(result/"attention_training_evolution.json",{"complete":True,"traces":traces,"display_maximum":display_maximum,
        "same_registered_examples":True,"training_steps_added":0,"gpu_seconds":0,
        "source_sha256":file_sha256(Path(__file__)),"interpretation":"No object semantics established by these attention maps"})


if __name__=="__main__":
    parser=argparse.ArgumentParser();parser.add_argument("--configuration",type=Path,required=True)
    main(parser.parse_args().configuration)
