#!/usr/bin/env python
"""CPU diagnostic interventions and actual attention overlays, no training.

The image overlays show reading weights, not segmentation or causal risk GT.
Important-token replacement is an internal representation intervention.
"""
import argparse
import json
from pathlib import Path
import sys
import time

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plotting
import numpy as np
import torch

PROJECT_ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(PROJECT_ROOT/"scripts"))
from run_metric_visual_evidence_pilot import (cache_tensors, load_artifacts,
    load_trained_reader, native_labels, verify_registration)
from calibrate_frozen_drivor_scorer import protected_training_audit,write_json


def map_attention_to_original(attention,original_shape):
    padded=cv2.resize(attention.reshape(19,37).astype("float32"),(518,266),interpolation=cv2.INTER_LINEAR)
    cropped=padded[:256,:512]
    original_height,original_width=original_shape[:2]
    reconstructed=np.zeros((original_height,original_width),dtype="float32")
    reconstructed[28:-28]=cv2.resize(cropped,(original_width,original_height-56),interpolation=cv2.INTER_LINEAR)
    return reconstructed


def blend_attention(image,weights,color,display_maximum):
    alpha=np.clip(weights/display_maximum,0,1)[...,None]*.6
    # Excluded original top/bottom crop receives no overlay.
    return np.clip((1-alpha)*image.astype("float32")+alpha*np.array(color,dtype="float32"),0,255).astype("uint8")


def main(configuration_path):
    configuration=json.loads(configuration_path.read_text())
    verify_registration(configuration)
    torch.set_num_threads(configuration["cpu_threads"])
    _,records,original,_,output,result=load_artifacts(configuration)
    targets=native_labels(configuration)
    device=torch.device("cpu")
    specific=load_trained_reader(configuration,"metric_specific",device)
    shared=load_trained_reader(configuration,"shared",device)
    initial=specific.__class__(384,configuration["reader_dimension"],configuration["hidden_dimension"],"metric_specific",configuration["maximum_logit_correction"])
    initial.load_state_dict(torch.load(output/"common_initialization.pt",map_location="cpu",weights_only=True))
    development=[index for index,record in enumerate(records) if record["pilot_split"]=="locked_development"]
    # Fix examples by recording/sample order and actual command before looking
    # at attention quality or gain. Four per command for functional diagnostics.
    fixed_diagnostic_rows=[]
    for command in (0,1,2):
        fixed_diagnostic_rows.extend([index for index in development if records[index]["command_index"]==command][:4])
    write_json(result/"diagnostic_subset.json",{"selection_rule":"First four separate-development scenes per logged command, not selected by attention or score improvement",
        "rows":fixed_diagnostic_rows,"tokens":[records[index]["token"] for index in fixed_diagnostic_rows]})
    patches=np.load(output/"spatial_tokens.npy",mmap_mode="r")
    training_mean=np.zeros(384,dtype="float64")
    for first in range(0,configuration["training_scenes"],32):
        training_mean+=np.array(patches[first:first+32],dtype="float32").sum((0,1),dtype="float64")
    training_mean=torch.tensor(training_mean/(configuration["training_scenes"]*patches.shape[1]),dtype=torch.float32)
    started=time.perf_counter()
    functional=[]
    geometry_examples=[]
    initial_actual=[]
    for row in fixed_diagnostic_rows:
        inputs=cache_tensors(configuration,[row],device)
        with torch.no_grad():
            current_logits,attention=specific(**inputs,capture_attention=True)
            _,shared_attention=shared(**inputs,capture_attention=True)
            _,initial_attention=initial(**inputs,capture_attention=True)
        valid_count=inputs["valid_token_mask"].sum(1).float()
        coefficient=(attention.std(-1,unbiased=False)*valid_count[:,None,None]).mean().item()
        normalized_entropy=(-(attention*attention.clamp_min(1e-30).log()).sum(-1)/valid_count.log()[:,None,None]).mean().item()
        metric_distance=(attention[:,:,0]-attention[:,:,1]).abs().sum(-1).mul(.5).mean().item()
        candidate_distance=(attention-attention.mean(1,keepdim=True)).abs().sum(-1).mul(.5).mean().item()
        initial_change=(attention-initial_attention).abs().sum(-1).mul(.5).mean().item()
        shared_metric_error=(shared_attention[:,:,0]-shared_attention[:,:,1]).abs().max().item()
        assert shared_metric_error==0.
        replacements={}
        for metric_index,metric_name in enumerate(("NC","DAC")):
            candidate=int(targets[row,:,metric_index].argmin())
            important=attention[0,candidate,metric_index].argsort(descending=True)[:max(1,patches.shape[1]//10)]
            rng=torch.Generator().manual_seed(7000+row+metric_index)
            random_indices=torch.randperm(patches.shape[1],generator=rng)[:len(important)]
            values=[]
            for indices in (important,random_indices):
                altered=inputs["visual_tokens"].clone()
                altered[:,indices]=training_mean
                with torch.no_grad():
                    modified,_=specific(**{**inputs,"visual_tokens":altered})
                values.append(abs(float(modified[0,candidate,metric_index]-current_logits[0,candidate,metric_index])))
            replacements[metric_name]={"candidate":candidate,"official_target":float(targets[row,candidate,metric_index]),
                "important_token_count":len(important),"important_replacement_abs_logit_change":values[0],"random_replacement_abs_logit_change":values[1]}
        token_gradients={}
        for metric_index,metric_name in enumerate(("NC","DAC")):
            candidate=int(targets[row,:,metric_index].argmin())
            tokens=inputs["visual_tokens"].clone().requires_grad_(True)
            prediction,_=specific(**{**inputs,"visual_tokens":tokens})
            gradient=torch.autograd.grad(prediction[0,candidate,metric_index],tokens)[0]
            assert torch.isfinite(gradient).all()
            token_gradients[metric_name]={"feature_gradient_norm":float(gradient.norm()),
                "absolute_gradient_times_feature_sum":float((gradient*tokens).abs().sum())}
        functional.append({"token":records[row]["token"],"pilot_row":row,"command_index":records[row]["command_index"],
            "attention_coefficient_of_variation":coefficient,"normalized_attention_entropy":normalized_entropy,
            "NC_DAC_attention_total_variation":metric_distance,"candidate_attention_mean_total_variation":candidate_distance,
            "before_after_attention_total_variation":initial_change,"shared_metric_attention_max_error":shared_metric_error,
            "feature_replacement":replacements,"patch_branch_gradients":token_gradients})
        initial_actual.append((initial_attention.numpy(),attention.numpy()))
        # Use one fixed example per real command for the main overlay gallery.
        if sum(item["command_index"]==records[row]["command_index"] for item in geometry_examples)==0:
            frozen_scores=np.load(result/"evaluation_predictions.npz")["unchanged_scores"]
            dev_index=development.index(row)
            selected=int(frozen_scores[dev_index].argmax())
            geometry_examples.append({"pilot_row":row,"token":records[row]["token"],"command_index":records[row]["command_index"],
                "candidate":selected,"initial_attention":initial_attention[0,selected].numpy(),"attention":attention[0,selected].numpy()})
    summary={"complete":True,"diagnostic_scenes":len(functional),"records":functional,
        "mean_NC_DAC_attention_total_variation":float(np.mean([item["NC_DAC_attention_total_variation"] for item in functional])),
        "mean_candidate_attention_total_variation":float(np.mean([item["candidate_attention_mean_total_variation"] for item in functional])),
        "mean_before_after_attention_total_variation":float(np.mean([item["before_after_attention_total_variation"] for item in functional])),
        "mean_normalized_attention_entropy":float(np.mean([item["normalized_attention_entropy"] for item in functional])),
        "mean_attention_coefficient_of_variation":float(np.mean([item["attention_coefficient_of_variation"] for item in functional])),
        "feature_replacement_summary":{name:{"mean_important_abs_logit_change":float(np.mean([item["feature_replacement"][name]["important_replacement_abs_logit_change"] for item in functional])),
            "mean_random_abs_logit_change":float(np.mean([item["feature_replacement"][name]["random_replacement_abs_logit_change"] for item in functional]))} for name in ("NC","DAC")},
        "interpretation":"Attention reading and frozen-token interventions; no segmentation, no image/object causal removal, no semantic specialization proof",
        "cpu_wall_seconds":time.perf_counter()-started,"gpu_seconds":0.,"protected_training":protected_training_audit(configuration)}
    write_json(result/"functional_diagnostics.json",summary)
    densities=np.concatenate([example[key].flatten()*1406 for example in geometry_examples for key in ("initial_attention","attention")])
    display_maximum=max(1.,float(np.percentile(densities,99.5)))
    figure,axes=plotting.subplots(len(geometry_examples),5,figsize=(20,3.3*len(geometry_examples)),squeeze=False)
    exported_examples=[]
    for panel,example in enumerate(geometry_examples):
        record=records[example["pilot_row"]]
        raw=cv2.cvtColor(cv2.imread(record["observed_front_paths"][-1]),cv2.COLOR_BGR2RGB)
        current_attention=example["attention"][:,703:]*1406
        before_attention=example["initial_attention"][:,703:]*1406
        maps=[map_attention_to_original(values,raw.shape) for values in current_attention]
        before_maps=[map_attention_to_original(values,raw.shape) for values in before_attention]
        images=[raw,blend_attention(raw,before_maps[0],[255,0,0],display_maximum),
            blend_attention(raw,maps[0],[255,0,0],display_maximum),blend_attention(raw,maps[1],[0,70,255],display_maximum)]
        red_alpha=np.clip(maps[0]/display_maximum,0,1)*.45
        blue_alpha=np.clip(maps[1]/display_maximum,0,1)*.45
        overlay=(1-red_alpha[...,None]-blue_alpha[...,None])*raw+red_alpha[...,None]*np.array([255,0,0])+blue_alpha[...,None]*np.array([0,70,255])
        images.append(overlay.astype("uint8"))
        for column,title in enumerate(("Original observed image","NC before fitting","NC after fitting","DAC after fitting","NC red + DAC blue")):
            axes[panel,column].imshow(images[column]);axes[panel,column].axis("off")
            axes[panel,column].set_title(title+f"\ncommand {record['command_index']} / candidate {example['candidate']}",fontsize=10)
        exported_examples.append({key:value for key,value in example.items() if key not in ("initial_attention","attention")})
    figure.suptitle(f"Actual metric-specific attention / fixed first example per command\nShared color scaling across all panels: density=1406 x attention, uniform=1, display max={display_maximum:.3f}\nOnly current-frame mass overlaid; cropped top/bottom excluded. Reading weights are not segmentation or causal risk.",fontsize=12)
    figure.tight_layout(rect=(0,0,1,.91));figure.savefig(result/"nc_dac_attention_overlays.png",dpi=160);plotting.close(figure)
    # Raw weights retain a common scale; no independent per-image min/max.
    figure,axes=plotting.subplots(len(geometry_examples),3,figsize=(14,3.5*len(geometry_examples)),squeeze=False)
    for panel,example in enumerate(geometry_examples):
        for column,(metric,title) in enumerate(((0,"NC attention density"),(1,"DAC attention density"))):
            rendered=example["attention"][metric,703:].reshape(19,37)*1406
            artist=axes[panel,column].imshow(rendered,vmin=0,vmax=display_maximum,cmap="Reds" if metric==0 else "Blues")
            axes[panel,column].set_title(title);figure.colorbar(artist,ax=axes[panel,column],fraction=.035)
        axis=axes[panel,2]
        for metric,label,color in ((0,"NC","red"),(1,"DAC","blue")):
            axis.plot(example["attention"][metric]*1406,color=color,label=label,lw=.8)
        axis.axvline(703,color="gray",ls="--");axis.axhline(1,color="black",ls=":",label="uniform")
        axis.set_title("Past tokens | Current tokens");axis.set_ylabel("Relative density");axis.legend()
    figure.suptitle("Actual reading weights; shared scaling preserves weak or uniform attention",fontsize=12)
    figure.tight_layout(rect=(0,0,1,.95));figure.savefig(result/"nc_dac_attention_density.png",dpi=160);plotting.close(figure)
    write_json(result/"visualization_metadata.json",{"examples":exported_examples,"display_maximum":display_maximum,
        "mapping":"Patch grid19x37 -> padded518x266 -> crop512x256 -> original image resize into rows28:-28",
        "same_scale_all_panels":True,"future_input":False,"semantic_supervision_used":False})
    results=json.loads((result/"results.json").read_text())
    figure,axes=plotting.subplots(1,2,figsize=(10,4.5))
    names=("unchanged","shared","metric_specific")
    axes[0].bar(names,[results["summaries"][name]["selected_pdms"] for name in names])
    axes[0].axhline(results["summaries"]["unchanged"]["oracle_pdms"],color="black",ls="--",label="Fixed64 oracle")
    axes[0].set_ylabel("PDMS points");axes[0].legend()
    axes[1].bar(names,[results["summaries"][name]["selection_regret"] for name in names]);axes[1].set_ylabel("Selection regret points")
    figure.tight_layout();figure.savefig(result/"score_comparison.png",dpi=160);plotting.close(figure)
    write_json(output/"diagnostics_complete.json",{"complete":True})
    print(json.dumps({"complete":True,"functional_diagnostics":str(result/"functional_diagnostics.json")}),flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser();parser.add_argument("--configuration",type=Path,required=True)
    main(parser.parse_args().configuration)
