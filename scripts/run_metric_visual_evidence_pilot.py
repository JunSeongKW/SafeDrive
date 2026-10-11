#!/usr/bin/env python
"""Frozen-patch NC/DAC evidence pilot using existing official labels.

No original model file is modified. Spatial tokens are hooked from the same
frozen DrivoR forward that produces the registered native64 candidates.
"""
import argparse
from collections import Counter
import copy
import csv
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import platform
import subprocess
import sys
import time

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT/"src"))
from calibrate_frozen_drivor_scorer import (
    METRIC_NAMES, file_sha256, protected_training_audit, recording_bootstrap, write_json)

CONDITIONS = ("shared", "metric_specific")


def load_artifacts(configuration):
    source = PROJECT_ROOT/configuration["source_pilot_directory"]
    source_registration = json.loads((source/"registration.json").read_text())
    original_cache = {path.stem: np.load(path, mmap_mode="r") for path in (source/"cache").glob("*.npy")}
    weights = json.loads((source/"cache_complete.json").read_text())["score_weights"]
    output = PROJECT_ROOT/configuration["output_directory"]
    result = PROJECT_ROOT/configuration["results_directory"]
    return source, source_registration["records"], original_cache, weights, output, result


def native_labels(configuration):
    source, records, _, _, _, _ = load_artifacts(configuration)
    targets = np.zeros((len(records), configuration["native_candidates"], 7), dtype="float32")
    scout = np.load(source/"scout_official_metrics.npy", mmap_mode="r")
    remaining = np.load(source/"training_remainder_official_metrics.npy", mmap_mode="r")
    development = np.load(source/"development_official_metrics.npy", mmap_mode="r")
    scout_count = sum(record["common_scout"] for record in records)
    targets[:scout_count] = scout[:, :64]
    targets[scout_count:configuration["training_scenes"]] = remaining[:, :64]
    targets[configuration["training_scenes"]:] = development[:, :64]
    assert np.isfinite(targets).all()
    return targets


def verify_registration(configuration):
    _, _, _, _, output, _ = load_artifacts(configuration)
    registered = json.loads((output/"registration.json").read_text())
    assert registered["configuration"] == configuration
    assert all(file_sha256(PROJECT_ROOT/name) == digest for name, digest in registered["sources"].items())
    assert all(file_sha256(PROJECT_ROOT/name) == digest for name, digest in registered["inputs"].items())
    return registered


def prepare(configuration, configuration_path):
    source, records, _, _, output, result = load_artifacts(configuration)
    output.mkdir(parents=True, exist_ok=True)
    result.mkdir(parents=True, exist_ok=True)
    source_files = [configuration_path, Path(__file__),
        PROJECT_ROOT/"src/planning_aware_future_prediction/models/metric_visual_evidence.py",
        PROJECT_ROOT/"tests/test_metric_visual_evidence.py",
        PROJECT_ROOT/"scripts/diagnose_metric_visual_evidence.py",
        PROJECT_ROOT/"scripts/run_metric_visual_evidence_pilot.sh"]
    inputs = [source/"registration.json", source/"cache_complete.json", source/"scout_official_metrics.npy",
        source/"training_remainder_official_metrics.npy", source/"development_official_metrics.npy",
        PROJECT_ROOT/configuration["checkpoint"]]
    train_groups = {record["recording_group"] for record in records if record["pilot_split"] == "train"}
    tune_groups = {record["recording_group"] for record in records if record["pilot_split"] == "tuning"}
    eval_groups = {record["recording_group"] for record in records if record["pilot_split"] == "locked_development"}
    assert not train_groups.intersection(tune_groups|eval_groups)
    assert not tune_groups.intersection(eval_groups)
    assert Counter(record["pilot_split"] for record in records) == {"train":1024,"tuning":64,"locked_development":192}
    content = {"registered_utc": datetime.now(timezone.utc).isoformat(), "configuration": configuration,
        "sources": {str(path.relative_to(PROJECT_ROOT)): file_sha256(path) for path in source_files},
        "inputs": {str(path.relative_to(PROJECT_ROOT)): file_sha256(path) for path in inputs},
        "records": records, "recording_overlap":0, "independent_final_test":False,
        "protected_training":protected_training_audit(configuration)}
    if (output/"registration.json").exists():
        verify_registration(configuration)
    else:
        write_json(output/"registration.json", content)
        write_json(result/"subset_manifest.json", content)
    targets = native_labels(configuration)
    summary = {subset: {"scenes":len(indices),
        "recordings":len({records[index]["recording_group"] for index in indices}),
        "NC_negative_fraction":float((targets[indices,:,0]<1).mean()),
        "DAC_negative_fraction":float((targets[indices,:,1]<1).mean()),
        "scenes_with_NC_variation":int((targets[indices,:,0].max(1)>targets[indices,:,0].min(1)).sum()),
        "scenes_with_DAC_variation":int((targets[indices,:,1].max(1)>targets[indices,:,1].min(1)).sum())}
        for subset in ("train","tuning","locked_development")
        for indices in [[index for index,record in enumerate(records) if record["pilot_split"] == subset]]}
    write_json(result/"label_distribution.json", summary)
    unit_command=[str(PROJECT_ROOT/"runtime/environments/kjs-scorer-selection-pilot/bin/python"),"-m","pytest","-q", "tests/test_metric_visual_evidence.py"]
    started=time.perf_counter()
    unit = subprocess.run(unit_command,cwd=PROJECT_ROOT,env={**os.environ,"CUDA_VISIBLE_DEVICES":"","PYTHONPATH":str(PROJECT_ROOT/"src")},capture_output=True,text=True)
    write_json(result/"unit_validation.json",{"passed":unit.returncode==0,"stdout":unit.stdout,"stderr":unit.stderr,"wall_seconds":time.perf_counter()-started})
    assert unit.returncode==0,unit.stdout+unit.stderr
    write_json(output/"prepare_complete.json",{"complete":True,"counts":{key:value["scenes"] for key,value in summary.items()}})
    print(json.dumps({"phase":"prepare","labels":summary}),flush=True)


def require_small_gpu(configuration):
    import torch
    assert os.environ.get("CUDA_VISIBLE_DEVICES") == str(configuration["physical_gpu"])
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable in this execution context; rerun this phase with approved host access. No CPU encoder fallback is started.")
    torch.cuda.set_per_process_memory_fraction(configuration["allocator_limit_gib"]*1024**3/torch.cuda.get_device_properties(0).total_memory)
    return torch.device("cuda:0")


def whole_card_memory(configuration):
    output=subprocess.check_output(["nvidia-smi","--query-gpu=index,memory.used,memory.free,utilization.gpu","--format=csv,noheader"],text=True)
    selected=next(line for line in output.splitlines() if int(line.split(",")[0])==configuration["physical_gpu"])
    used=int(selected.split(",")[1].split()[0])*1024**2
    assert used<configuration["whole_card_limit_decimal_gb"]*1e9,selected
    return used


def cache_patches(configuration):
    import torch
    from diagnose_residual_control_sampler import load_reference_helpers, module_digest
    verify_registration(configuration)
    _, records, original, _, output, result=load_artifacts(configuration)
    if (output/"patch_cache_complete.json").exists():
        return
    torch.set_num_threads(configuration["cpu_threads"])
    torch.manual_seed(configuration["seed"])
    torch.use_deterministic_algorithms(True)
    device=require_small_gpu(configuration)
    before_card=whole_card_memory(configuration)
    constructor,_=load_reference_helpers(configuration)
    model=constructor("drivor",None)
    saved=torch.load(PROJECT_ROOT/configuration["checkpoint"],map_location="cpu",weights_only=False,mmap=True)
    assert saved["completed_updates"]==configuration["expected_checkpoint_updates"]
    model.load_state_dict(saved["model"],strict=True)
    del saved
    model.eval().requires_grad_(False)
    digest=module_digest(model)
    model.to(device)
    encoder=model.backbone.encoder
    vision_transformer=encoder.model.lora_vit if encoder.use_lora else encoder.model
    rows_count,columns_count=vision_transformer.patch_embed.grid_size
    feature_dimension=vision_transformer.num_features
    prefix_count=encoder.num_prefix_tokens+vision_transformer.num_prefix_tokens
    frame_token_count=rows_count*columns_count
    assert (rows_count,columns_count,feature_dimension)==(19,37,384)
    extracted=[]
    hook=vision_transformer.norm.register_forward_hook(lambda _module,_arguments,tensor: extracted.append(tensor))
    image_cache=np.load(PROJECT_ROOT/configuration["corpus_directory"]/"images.npy",mmap_mode="r")
    tokens=np.lib.format.open_memmap(output/"spatial_tokens.npy",mode="w+",dtype="float16",
        shape=(len(records),2*frame_token_count,feature_dimension))
    centers_x=(np.arange(columns_count)+.5)*14
    centers_y=(np.arange(rows_count)+.5)*14
    coordinates=np.stack(np.meshgrid(centers_x,centers_y),-1).reshape(-1,2)
    position_arrays=[]
    for offset in (-.5,0.):
        position_arrays.append(np.concatenate((coordinates/np.array([512.,256.])*2-1,
            np.zeros((frame_token_count,1)),np.full((frame_token_count,1),offset)),1))
    positions=np.concatenate(position_arrays).astype("float32")
    # Every boundary patch intersects observed pixels; no fully padded token.
    valid_mask=np.ones((len(records),2*frame_token_count),dtype="bool")
    np.save(output/"token_positions.npy",positions)
    np.save(output/"valid_token_mask.npy",valid_mask)
    maximum_errors={"scene_features":0.,"native_trajectories":0.,"component_scores":0.}
    storage_absolute_error=0.
    storage_relative_error=0.
    maximum_card=before_card
    started=time.perf_counter()
    for first in range(0,len(records),configuration["cache_batch_size"]):
        chosen=records[first:first+configuration["cache_batch_size"]]
        selected_rows=[record["cache_row"] for record in chosen]
        observed=torch.from_numpy(np.array(image_cache[selected_rows])).permute(0,1,4,2,3).to(device).float()/255
        ego=torch.from_numpy(np.array(original["ego_status"][first:first+len(chosen)])).to(device)
        extracted.clear()
        with torch.no_grad():
            predicted=model({"image":observed,"ego_status":ego[:,None]})
            assert len(extracted)==1 and extracted[0].shape==(2*len(chosen),prefix_count+frame_token_count,feature_dimension)
            live_patch=extracted[0][:,prefix_count:].reshape(len(chosen),2*frame_token_count,feature_dimension).float()
            stored=live_patch.half()
            tokens[first:first+len(chosen)]=stored.cpu().numpy()
            storage_absolute_error=max(storage_absolute_error,float((live_patch-stored.float()).abs().max()))
            storage_relative_error=max(storage_relative_error,float((live_patch-stored.float()).norm()/live_patch.norm().clamp_min(1e-8)))
            # Same hooked norm output also reconstructs the unchanged register route.
            with torch.autocast(device.type,dtype=torch.bfloat16):
                register=encoder.neck(extracted[0][:,:encoder.num_prefix_tokens])
            register=register.reshape(len(chosen),2,16,256)
            scene=model.backbone.fusion(register.float().transpose(1,2).flatten(2))
            for name,actual,expected in (("scene_features",scene,original["scene_features"][first:first+len(chosen)]),
                ("native_trajectories",predicted["proposals"],original["native_trajectories"][first:first+len(chosen)]),
                ("component_scores",predicted["pdm_score"],None)):
                if expected is None:
                    from planning_aware_future_prediction.models.scorer_selection_learning import compose_drivor_log_score
                    expected=compose_drivor_log_score(torch.from_numpy(np.array(original["native_component_logits"][first:first+len(chosen)])).to(device),
                        json.loads((PROJECT_ROOT/configuration["source_pilot_directory"]/"cache_complete.json").read_text())["score_weights"])
                    error=float((actual-expected).abs().max())
                else:
                    error=float((actual-torch.from_numpy(np.array(expected)).to(device)).abs().max())
                maximum_errors[name]=max(maximum_errors[name],error)
                assert error<2e-5,(name,error)
        if first==0 or (first+len(chosen))%128==0:
            maximum_card=max(maximum_card,whole_card_memory(configuration))
            write_json(output/"patch_cache_progress.json",{"completed_scenes":first+len(chosen),"total_scenes":len(records),"elapsed_seconds":time.perf_counter()-started})
            print(json.dumps({"phase":"cache","scenes":first+len(chosen),"total":len(records)}),flush=True)
    torch.cuda.synchronize()
    elapsed=time.perf_counter()-started
    hook.remove()
    tokens.flush()
    assert all(parameter.grad is None for parameter in model.parameters())
    peak_allocated=torch.cuda.max_memory_allocated()
    peak_reserved=torch.cuda.max_memory_reserved()
    model.cpu()
    assert module_digest(model)==digest
    from calibrate_frozen_drivor_scorer import ego_boundary
    boundary_checks=[ego_boundary(records[index],original["ego_status"][index]) for index in (0,1023,1024,1088)]
    metadata={"complete":True,"shape":list(tokens.shape),"dtype":"float16","grid_shape_per_frame":[rows_count,columns_count],
        "observed_frames":2,"time_offsets_seconds":[-.5,0.],"camera":"CAM_F0","future_input":False,
        "extraction":"Final DINO norm, after original learned registers/LoRA attention and before prefix slicing/neck; hook only",
        "spatial_mapping":"14px patches of right6/bottom10 padded512x256 crop28 input; partially padded boundary patches retained",
        "token_position_columns":["normalized_x","normalized_y","camera_index","time_offset_seconds"],
        "same_forward_original_route_errors":maximum_errors,"float16_maximum_absolute_storage_error":storage_absolute_error,
        "float16_maximum_relative_l2_storage_error":storage_relative_error,"frozen_parameters_unchanged":True,
        "frozen_gradients_absent":True,"encoder_calls":math.ceil(len(records)/configuration["cache_batch_size"]),
        "encoder_calls_during_training":0,"backbone_checkpoint_sha256":file_sha256(PROJECT_ROOT/configuration["checkpoint"]),
        "cache_loop_wall_seconds":elapsed,"gpu_loop_seconds":elapsed,"physical_gpu":configuration["physical_gpu"],
        "peak_allocated_gpu_bytes":peak_allocated,"peak_reserved_gpu_bytes":peak_reserved,"maximum_whole_card_bytes":maximum_card,
        "observed_sample_ego_boundary_checks":boundary_checks,
        "cache_sha256":{name:file_sha256(output/name) for name in ("spatial_tokens.npy","token_positions.npy","valid_token_mask.npy")},
        "protected_training":protected_training_audit(configuration)}
    write_json(output/"patch_cache_complete.json",metadata)
    write_json(result/"spatial_cache_metadata.json",metadata)


def initialize_reader(configuration,condition):
    from planning_aware_future_prediction.models.metric_visual_evidence import MetricVisualScorerCorrection
    return MetricVisualScorerCorrection(384,configuration["reader_dimension"],configuration["hidden_dimension"],condition,configuration["maximum_logit_correction"])


def cache_tensors(configuration,selected_rows,device,visual_replacement=None):
    import torch
    _,_,original,_,output,_=load_artifacts(configuration)
    patches=np.load(output/"spatial_tokens.npy",mmap_mode="r")
    features=torch.from_numpy(np.array(patches[selected_rows])).to(device).float()
    if visual_replacement is not None:
        features=visual_replacement[None,None].expand_as(features)
    return {"visual_tokens":features,
        "valid_token_mask":torch.from_numpy(np.array(np.load(output/"valid_token_mask.npy",mmap_mode="r")[selected_rows])).to(device),
        "positions":torch.from_numpy(np.load(output/"token_positions.npy")).to(device),
        "ego_status":torch.from_numpy(np.array(original["ego_status"][selected_rows])).to(device),
        "trajectories":torch.from_numpy(np.array(original["native_trajectories"][selected_rows])).to(device),
        "scored_features":torch.from_numpy(np.array(original["native_scored_features"][selected_rows])).to(device),
        "component_logits":torch.from_numpy(np.array(original["native_component_logits"][selected_rows])).to(device)}


def validate_actual_cache(configuration):
    import torch
    from planning_aware_future_prediction.models.metric_visual_evidence import compute_metric_evidence_losses
    verify_registration(configuration)
    torch.set_num_threads(configuration["cpu_threads"])
    _,_,_,weights,output,result=load_artifacts(configuration)
    metadata=json.loads((output/"patch_cache_complete.json").read_text())
    assert all(file_sha256(output/name)==digest for name,digest in metadata["cache_sha256"].items())
    inputs=cache_tensors(configuration,[0,1],torch.device("cpu"))
    targets=torch.from_numpy(native_labels(configuration)[:2])
    torch.manual_seed(configuration["seed"])
    specific=initialize_reader(configuration,"metric_specific")
    shared=initialize_reader(configuration,"shared")
    shared.load_state_dict(copy.deepcopy(specific.state_dict()))
    assert sum(parameter.numel() for parameter in specific.parameters())==sum(parameter.numel() for parameter in shared.parameters())
    pairs=torch.randint(0,64,(2,128,2),generator=torch.Generator().manual_seed(43))
    gradient_checks=[]
    optimizer=torch.optim.AdamW(specific.parameters(),lr=configuration["learning_rate"])
    for step in range(2):
        optimizer.zero_grad(set_to_none=True)
        corrected,attention=specific(**inputs,capture_attention=True)
        if step==0:
            assert torch.equal(corrected,inputs["component_logits"])
        component,ranking=compute_metric_evidence_losses(corrected,targets,weights,pairs)
        (component+ranking).backward()
        assert all(torch.isfinite(parameter.grad).all() for parameter in specific.parameters() if parameter.grad is not None)
        norms={name:float(parameter.grad.norm()) for name,parameter in specific.named_parameters() if parameter.grad is not None}
        if step==1:
            assert all(norms[name]>0 for name in ("key_projection.weight","value_projection.weight","candidate_query.0.weight","metric_embedding","correction_head.weight"))
        gradient_checks.append({"step":step+1,"loss":float(component+ranking),"parameter_gradient_norms":norms})
        optimizer.step()
    assert all(tensor.grad is None for tensor in inputs.values())
    with torch.no_grad():
        off,_=specific(**inputs,intervention="correction_off")
        assert torch.equal(off,inputs["component_logits"])
        unchanged_components,_=specific(**inputs)
        assert torch.equal(unchanged_components[...,2:],inputs["component_logits"][...,2:])
        _,shared_attention=shared(**inputs,capture_attention=True)
        assert torch.equal(shared_attention[:,:,0],shared_attention[:,:,1])
    write_json(result/"actual_cache_validation.json",{"passed":True,"spatial_cache_hashes_match":True,
        "initial_and_off_logits_exact":True,"other_four_metrics_exact":True,"equal_parameter_count":True,
        "actual_cache_two_step_gradients":gradient_checks,"cache_input_gradients_absent":True,
        "encoder_and_scorer_modules_present_in_training":False,"encoder_calls_in_cached_steps":0,
        "future_input":False,"protected_training":protected_training_audit(configuration)})
    write_json(output/"validation_complete.json",{"complete":True})


def predict(configuration,model,selected_rows,device,intervention=None,visual_replacement=None):
    import torch
    from planning_aware_future_prediction.models.scorer_selection_learning import compose_drivor_log_score
    _,_,_,weights,_,_=load_artifacts(configuration)
    scores=[]
    logits=[]
    with torch.no_grad():
        for first in range(0,len(selected_rows),configuration["batch_size"]):
            inputs=cache_tensors(configuration,selected_rows[first:first+configuration["batch_size"]],device,visual_replacement)
            corrected,_=model(**inputs,intervention=intervention)
            logits.append(corrected.cpu().numpy())
            scores.append(compose_drivor_log_score(corrected,weights).cpu().numpy())
    return np.concatenate(scores),np.concatenate(logits)


def profile(configuration):
    import torch
    from planning_aware_future_prediction.models.metric_visual_evidence import compute_metric_evidence_losses
    verify_registration(configuration)
    device=require_small_gpu(configuration)
    _,_,_,weights,output,result=load_artifacts(configuration)
    torch.set_num_threads(configuration["cpu_threads"])
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(configuration["seed"])
    model=initialize_reader(configuration,"metric_specific").to(device)
    optimizer=torch.optim.AdamW(model.parameters(),lr=configuration["learning_rate"])
    targets=native_labels(configuration)
    timings=[]
    maximum_card=whole_card_memory(configuration)
    for step in range(configuration["profile_steps"]):
        selected=np.arange(configuration["batch_size"])
        torch.cuda.synchronize()
        started=time.perf_counter()
        inputs=cache_tensors(configuration,selected,device)
        output_logits,_=model(**inputs)
        metrics=torch.from_numpy(targets[selected]).to(device)
        pairs=torch.randint(0,64,(len(selected),128,2),device=device)
        loss=sum(compute_metric_evidence_losses(output_logits,metrics,weights,pairs))
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        torch.cuda.synchronize()
        timings.append(time.perf_counter()-started)
        maximum_card=max(maximum_card,whole_card_memory(configuration))
    median=float(np.median(timings[2:]))
    estimated=2*configuration["training_steps"]*median
    assert estimated<configuration["maximum_combined_training_gpu_hours"]*3600,(estimated,median)
    summary={"complete":True,"batch_size":configuration["batch_size"],"candidate_count":64,
        "spatial_tokens":1406,"profile_steps":len(timings),"step_seconds":timings,"steady_median_step_seconds":median,
        "estimated_combined_training_seconds":estimated,"measured_profile_gpu_seconds":sum(timings),
        "peak_allocated_bytes":torch.cuda.max_memory_allocated(),"peak_reserved_bytes":torch.cuda.max_memory_reserved(),
        "maximum_whole_card_bytes":maximum_card,"trainable_parameters":sum(parameter.numel() for parameter in model.parameters()),
        "profile_weights_used_for_training":False,"physical_gpu":configuration["physical_gpu"]}
    write_json(output/"profile_complete.json",summary)
    write_json(result/"profiling.json",summary)
    print(json.dumps({"phase":"profile","step_seconds":median,"estimated_train_seconds":estimated}),flush=True)


def train_readers(configuration):
    import torch
    from planning_aware_future_prediction.models.metric_visual_evidence import compute_metric_evidence_losses
    from planning_aware_future_prediction.models.scorer_selection_learning import compose_drivor_log_score
    verify_registration(configuration)
    device=require_small_gpu(configuration)
    _,records,_,weights,output,result=load_artifacts(configuration)
    torch.set_num_threads(configuration["cpu_threads"])
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(configuration["seed"])
    targets=native_labels(configuration)
    tuning_rows=[index for index,record in enumerate(records) if record["pilot_split"]=="tuning"]
    initial=initialize_reader(configuration,"metric_specific")
    initialization=copy.deepcopy(initial.state_dict())
    torch.save(initialization,output/"common_initialization.pt")
    generator=torch.Generator().manual_seed(configuration["seed"]+41)
    batches=torch.stack([torch.randperm(configuration["training_scenes"],generator=generator)[:configuration["batch_size"]]
        for _ in range(configuration["training_steps"])]).numpy()
    pairs=torch.randint(0,64,(configuration["training_steps"],configuration["batch_size"],configuration["pairs_per_scene"],2),generator=generator)
    np.save(output/"batch_order.npy",batches)
    np.save(output/"pair_indices.npy",pairs.numpy())
    summaries={}
    for condition in CONDITIONS:
        model=initialize_reader(configuration,condition).to(device)
        model.load_state_dict(initialization,strict=True)
        optimizer=torch.optim.AdamW(model.parameters(),lr=configuration["learning_rate"])
        logs=[]
        training_step_seconds=[]
        best_step=None
        best_pdms=-float("inf")
        maximum_card=whole_card_memory(configuration)
        started=time.perf_counter()
        for step,selected_rows in enumerate(batches,1):
            torch.cuda.synchronize()
            step_started=time.perf_counter()
            inputs=cache_tensors(configuration,selected_rows,device)
            corrected,_=model(**inputs)
            current_targets=torch.from_numpy(targets[selected_rows]).to(device)
            current_pairs=pairs[step-1].to(device).clone()
            current_pairs[:,0,0]=compose_drivor_log_score(inputs["component_logits"],weights).argmax(1)
            current_pairs[:,0,1]=current_targets[:,:,6].argmax(1)
            component,ranking=compute_metric_evidence_losses(corrected,current_targets,weights,current_pairs,configuration["ranking_temperature"])
            loss=configuration["component_loss_weight"]*component+configuration["ranking_loss_weight"]*ranking
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            assert all(torch.isfinite(parameter.grad).all() for parameter in model.parameters() if parameter.grad is not None)
            gradient=float(torch.nn.utils.clip_grad_norm_(model.parameters(),5.))
            assert torch.isfinite(loss)
            optimizer.step()
            torch.cuda.synchronize()
            training_step_seconds.append(time.perf_counter()-step_started)
            if step==1 or step%25==0:
                reader_gradient=math.sqrt(sum(float(parameter.grad.square().sum()) for name,parameter in model.named_parameters()
                    if parameter.grad is not None and any(fragment in name for fragment in ("candidate_query","metric_embedding","key_projection","value_projection"))))
                logs.append({"step":step,"loss":float(loss),"component_loss":float(component),"ranking_loss":float(ranking),
                    "gradient_norm_before_clip":gradient,"reader_gradient_norm":reader_gradient,"elapsed_seconds":time.perf_counter()-started})
                write_json(output/(condition+"_progress.json"),logs[-1])
            if step in configuration["checkpoint_steps"]:
                maximum_card=max(maximum_card,whole_card_memory(configuration))
                model.eval()
                tuning_scores,_=predict(configuration,model,tuning_rows,device)
                selection=tuning_scores.argmax(1)
                tuning_pdms=float(targets[tuning_rows,selection,6].mean()*100)
                torch.save({"model":model.state_dict(),"optimizer":optimizer.state_dict(),"completed_steps":step},output/(condition+"_step"+str(step)+".pt"))
                if tuning_pdms>best_pdms:
                    best_pdms=tuning_pdms
                    best_step=step
                print(json.dumps({"phase":"train","condition":condition,"step":step,"tuning_pdms":tuning_pdms}),flush=True)
                model.train()
        summaries[condition]={"completed_steps":configuration["training_steps"],"trainable_parameters":sum(parameter.numel() for parameter in model.parameters()),
            "best_tuning_step":best_step,"best_tuning_pdms":best_pdms,"gpu_training_step_seconds":sum(training_step_seconds),
            "median_training_step_seconds":float(np.median(training_step_seconds)),"phase_wall_seconds":time.perf_counter()-started,
            "maximum_whole_card_bytes":maximum_card,"peak_allocated_bytes":torch.cuda.max_memory_allocated(),
            "peak_reserved_bytes":torch.cuda.max_memory_reserved(),"logs":logs}
        write_json(output/(condition+"_training.json"),summaries[condition])
        del optimizer,model
        torch.cuda.empty_cache()
    assert sum(summary["gpu_training_step_seconds"] for summary in summaries.values())<configuration["maximum_combined_training_gpu_hours"]*3600
    write_json(output/"training_complete.json",{"complete":True,"conditions":summaries,"encoder_calls":0,"original_scorer_calls":0,
        "common_initialization_sha256":file_sha256(output/"common_initialization.pt"),"common_batch_order_sha256":file_sha256(output/"batch_order.npy"),
        "common_pair_order_sha256":file_sha256(output/"pair_indices.npy"),"physical_gpu":configuration["physical_gpu"],
        "protected_training":protected_training_audit(configuration)})


def load_trained_reader(configuration,condition,device):
    import torch
    _,_,_,_,output,_=load_artifacts(configuration)
    metadata=json.loads((output/"training_complete.json").read_text())["conditions"][condition]
    saved=torch.load(output/(condition+"_step"+str(metadata["best_tuning_step"])+".pt"),map_location="cpu",weights_only=False)
    model=initialize_reader(configuration,condition)
    model.load_state_dict(saved["model"],strict=True)
    return model.to(device).eval()


def ranking_diagnostics(scores,logits,targets):
    from scipy.stats import spearmanr
    candidate_pairs=np.triu_indices(scores.shape[1],1)
    true_gap=targets[:,candidate_pairs[0],6]-targets[:,candidate_pairs[1],6]
    predicted_gap=scores[:,candidate_pairs[0]]-scores[:,candidate_pairs[1]]
    gaps=np.abs(true_gap)
    weighted_accuracy=float((gaps*((true_gap*predicted_gap)>0)).sum()/max(float(gaps.sum()),1e-8))
    correlations=[float(spearmanr(row_scores,row_targets[:,6]).statistic) for row_scores,row_targets in zip(scores,targets) if np.ptp(row_targets[:,6])>1e-8]
    probability=1/(1+np.exp(-np.clip(logits[...,:2],-60,60)))
    calibration={}
    for metric_index,metric_name in enumerate(("NC","DAC")):
        predicted=probability[...,metric_index]
        actual=targets[...,metric_index]
        clipped=np.clip(predicted,1e-7,1-1e-7)
        calibration[metric_name]={"brier":float(np.square(predicted-actual).mean()),
            "BCE":float(-(actual*np.log(clipped)+(1-actual)*np.log(1-clipped)).mean()),
            "negative_fraction":float((actual<1).mean())}
    return {"gap_weighted_pair_accuracy":weighted_accuracy,"mean_within_scene_spearman":float(np.mean(correlations)),"calibration":calibration}


def metric_summary(scores,labels,records):
    selected=scores.argmax(1)
    chosen=labels[np.arange(len(records)),selected]
    oracle=labels[:,:,6].max(1)*100
    values=chosen[:,6]*100
    summary={"selected_pdms":float(values.mean()),"oracle_pdms":float(oracle.mean()),"selection_regret":float((oracle-values).mean()),
        "collision_or_offroad_rate":float(((chosen[:,0]<1)|(chosen[:,1]<1)).mean()),
        "NC_failure_rate":float((chosen[:,0]<1).mean()),"DAC_failure_rate":float((chosen[:,1]<1).mean()),
        "submetrics":{metric:float(chosen[:,index].mean()) for index,metric in enumerate(METRIC_NAMES[:6])},
        "by_command":{str(command):{"scenes":int(mask.sum()),"selected_pdms":float(values[mask].mean())}
            for command in sorted({record["command_index"] for record in records})
            for mask in [np.array([record["command_index"]==command for record in records])]}}
    return summary,values,selected


def report(configuration):
    import torch
    from planning_aware_future_prediction.models.scorer_selection_learning import compose_drivor_log_score
    verify_registration(configuration)
    device=require_small_gpu(configuration)
    torch.set_num_threads(configuration["cpu_threads"])
    _,records,original,weights,output,result=load_artifacts(configuration)
    selected_rows=[index for index,record in enumerate(records) if record["pilot_split"]=="locked_development"]
    evaluation_records=[records[index] for index in selected_rows]
    groups=[record["recording_group"] for record in evaluation_records]
    labels=native_labels(configuration)[selected_rows]
    base_logits=np.array(original["native_component_logits"][selected_rows])
    predictions={"unchanged":compose_drivor_log_score(torch.from_numpy(base_logits),weights).numpy()}
    all_logits={"unchanged":base_logits}
    summaries={}
    intervention_results={}
    diagnostic_costs={}
    started=time.perf_counter()
    # Training-only reference feature; development does not determine replacement.
    patches=np.load(output/"spatial_tokens.npy",mmap_mode="r")
    patch_mean=np.zeros(384,dtype=np.float64)
    for first in range(0,configuration["training_scenes"],32):
        patch_mean+=np.array(patches[first:first+32],dtype="float32").sum((0,1),dtype="float64")
    patch_mean=torch.from_numpy((patch_mean/(configuration["training_scenes"]*patches.shape[1])).astype("float32")).to(device)
    for condition in CONDITIONS:
        model=load_trained_reader(configuration,condition,device)
        predictions[condition],all_logits[condition]=predict(configuration,model,selected_rows,device)
        assert np.array_equal(all_logits[condition][...,2:],base_logits[...,2:])
        off,_=predict(configuration,model,selected_rows,device,"correction_off")
        assert np.allclose(off,predictions["unchanged"],atol=2e-6,rtol=0)
        assert np.array_equal(off.argmax(1),predictions["unchanged"].argmax(1))
        intervention_results[condition]={"correction_off_maximum_score_error":float(np.abs(off-predictions["unchanged"]).max()),"off_selection_restored":True}
        if condition=="metric_specific":
            for intervention in ("uniform_attention","swap_metric_attention","patch_mean"):
                scores,logits=predict(configuration,model,selected_rows,device,
                    None if intervention=="patch_mean" else intervention,patch_mean if intervention=="patch_mean" else None)
                predictions[condition+"_"+intervention]=scores
                all_logits[condition+"_"+intervention]=logits
                intervention_results[condition][intervention]={"mean_absolute_log_score_change":float(np.abs(scores-predictions[condition]).mean()),
                    "selected_index_changed_fraction":float((scores.argmax(1)!=predictions[condition].argmax(1)).mean())}
        # Same real cached minibatch; excludes original encoder/scorer and file IO.
        inputs=cache_tensors(configuration,selected_rows[:configuration["batch_size"]],device)
        with torch.no_grad():
            for _ in range(5):
                model(**inputs)
            torch.cuda.synchronize()
            latency_started=time.perf_counter()
            for _ in range(30):
                model(**inputs)
            torch.cuda.synchronize()
        diagnostic_costs[condition]={"cached_adapter_gpu_forward_ms_per_scene_batched8":1000*(time.perf_counter()-latency_started)/(30*len(inputs["visual_tokens"]))}
    scene_rows=[]
    selected_values={}
    for condition,scores in predictions.items():
        summaries[condition],selected_values[condition],choices=metric_summary(scores,labels,evaluation_records)
        summaries[condition]["ranking"]=ranking_diagnostics(scores,all_logits[condition],labels)
        if condition!="unchanged":
            difference=selected_values[condition]-selected_values["unchanged"]
            summaries[condition]["minus_unchanged_pdms"]=float(difference.mean())
            summaries[condition]["minus_unchanged_recording_95ci"]=recording_bootstrap(difference,groups,73,configuration["bootstrap_repetitions"])
            summaries[condition]["selected_index_changed_from_original_fraction"]=float((choices!=predictions["unchanged"].argmax(1)).mean())
        for index,record in enumerate(evaluation_records):
            scene_rows.append({"condition":condition,"token":record["token"],"recording_group":record["recording_group"],"command_index":record["command_index"],
                "selected_candidate":int(choices[index]),"selected_pdms":float(selected_values[condition][index]),
                "oracle_pdms":float(labels[index,:,6].max()*100),"NC":float(labels[index,choices[index],0]),"DAC":float(labels[index,choices[index],1])})
    difference=selected_values["metric_specific"]-selected_values["shared"]
    interval=recording_bootstrap(difference,groups,73,configuration["bootstrap_repetitions"])
    gate=interval[0]>0 and summaries["metric_specific"]["collision_or_offroad_rate"]<=min(summaries[name]["collision_or_offroad_rate"] for name in ("shared","unchanged"))+.01
    with (result/"scene_results.csv").open("w") as stream:
        writer=csv.DictWriter(stream,fieldnames=list(scene_rows[0])); writer.writeheader(); writer.writerows(scene_rows)
    with (result/"comparison.csv").open("w") as stream:
        writer=csv.DictWriter(stream,fieldnames=["condition","selected_pdms","oracle_pdms","selection_regret","NC_failure_rate","DAC_failure_rate"])
        writer.writeheader()
        for condition in ("unchanged",*CONDITIONS):
            writer.writerow({"condition":condition,**{name:summaries[condition][name] for name in writer.fieldnames[1:]}})
    np.savez_compressed(result/"evaluation_predictions.npz",tokens=np.array([record["token"] for record in evaluation_records]),official_metrics=labels,
        **{condition+"_scores":scores for condition,scores in predictions.items()},**{condition+"_logits":logits for condition,logits in all_logits.items()})
    train=json.loads((output/"training_complete.json").read_text())
    execution={"complete":True,"completed_utc":datetime.now(timezone.utc).isoformat(),"configuration":configuration,
        "summaries":summaries,"metric_specific_minus_shared_pdms":float(difference.mean()),"metric_specific_minus_shared_recording_95ci":interval,
        "replication_gate_passed":bool(gate),"full_scale_started":False,"independent_final_test":False,
        "training":train,"interventions":intervention_results,"adapter_latency":diagnostic_costs,
        "spatial_cache":json.loads((output/"patch_cache_complete.json").read_text()),"profile":json.loads((output/"profile_complete.json").read_text()),
        "evaluation_gpu_phase_seconds":time.perf_counter()-started,"maximum_whole_card_bytes":whole_card_memory(configuration),
        "official_labels_reused":True,"new_official_candidate_evaluations":0,"training_official_candidates":1024*64,
        "environment":{"python":platform.python_version(),"torch":torch.__version__,"physical_gpu":configuration["physical_gpu"],
            "CUBLAS_WORKSPACE_CONFIG":os.environ.get("CUBLAS_WORKSPACE_CONFIG"),"deterministic_algorithms":torch.are_deterministic_algorithms_enabled()},
        "protected_training":protected_training_audit(configuration)}
    write_json(result/"results.json",execution)
    write_json(output/"report_complete.json",{"complete":True,"replication_gate_passed":bool(gate)})
    print(json.dumps({"phase":"report","scores":{name:summaries[name]["selected_pdms"] for name in ("unchanged",*CONDITIONS)},"gate":bool(gate)}),flush=True)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--configuration",type=Path,required=True)
    parser.add_argument("--phase",choices=("prepare","cache","validate","profile","train","report"),required=True)
    arguments=parser.parse_args()
    configuration=json.loads(arguments.configuration.read_text())
    if arguments.phase=="prepare":
        prepare(configuration,arguments.configuration.resolve())
    else:
        {"cache":cache_patches,"validate":validate_actual_cache,"profile":profile,"train":train_readers,"report":report}[arguments.phase](configuration)


if __name__=="__main__":
    main()
