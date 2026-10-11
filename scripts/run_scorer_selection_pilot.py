#!/usr/bin/env python
"""Equal-label-budget acquisition and cached scorer-correction pilot.

Phases are resumable and never change the running LPWM experiment. Official
labels are computed in its established separate environment. All learning is
CPU-only over frozen caches; no evaluator/argmax gradients are claimed.
"""
import argparse
import ast
from concurrent.futures import ProcessPoolExecutor
import copy
import csv
from datetime import datetime, timezone
import json
import lzma
import multiprocessing
import os
from pathlib import Path
import pickle
import subprocess
import sys
import time

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
from calibrate_frozen_drivor_scorer import (METRIC_NAMES, file_sha256, official_scene_scores,
    protected_training_audit, recording_bootstrap, write_json)

STRATEGIES = ("A_uniform", "B_boundary", "C_selection_error")


def load_registration(configuration):
    output = PROJECT_ROOT / configuration["output_directory"]
    registration = json.loads((output/"registration.json").read_text())
    assert registration["configuration"] == configuration
    return output, registration


def load_feature_cache(output):
    return {path.stem: np.load(path, mmap_mode="r") for path in (output/"cache").glob("*.npy")}


def validate_cache(configuration):
    import torch
    from calibrate_frozen_drivor_scorer import ego_boundary
    from planning_aware_future_prediction.models.scorer_selection_learning import CachedScorerCorrection,compose_drivor_log_score
    output,registration=load_registration(configuration)
    cache=load_feature_cache(output)
    metadata=json.loads((output/"cache_complete.json").read_text())
    assert all(file_sha256(output/"cache"/(name+".npy"))==digest for name,digest in metadata["cache_files_sha256"].items())
    training_groups={record["recording_group"] for record in registration["records"] if record["pilot_split"] == "train"}
    tuning_groups={record["recording_group"] for record in registration["records"] if record["pilot_split"] == "tuning"}
    locked_groups={record["recording_group"] for record in registration["records"] if record["pilot_split"] == "locked_development"}
    assert not training_groups.intersection(tuning_groups|locked_groups)
    assert not tuning_groups.intersection(locked_groups)
    boundary_checks=[ego_boundary(record,cache["ego_status"][record["pilot_row"]])
        for record in (registration["records"][0],registration["records"][-1])]
    model=CachedScorerCorrection(configuration["hidden_dimension"],configuration["maximum_logit_correction"])
    inputs={"scored_features":torch.from_numpy(np.array(cache["native_scored_features"][:2])),
        "scene_features":torch.from_numpy(np.array(cache["scene_features"][:2])),
        "ego_status":torch.from_numpy(np.array(cache["ego_status"][:2])),
        "trajectories":torch.from_numpy(np.array(cache["native_trajectories"][:2])),
        "component_logits":torch.from_numpy(np.array(cache["native_component_logits"][:2]))}
    assert torch.equal(model(**inputs),inputs["component_logits"])
    scores=compose_drivor_log_score(model(**inputs),metadata["score_weights"])
    assert torch.isfinite(scores).all()
    write_json(output/"cache_validation_complete.json",{"complete":True,"all_cache_hashes_match":True,
        "train_tuning_locked_recording_overlap":0,"raw_previous_current_ego_and_sample_matching_checks":boundary_checks,
        "zero_initialized_actual_cache_correction_error":0.,"future_observations_in_input":False,
        "protected_training":protected_training_audit(configuration)})


def numpy_log_scores(logits, weights):
    probability = 1/(1+np.exp(-logits))
    log_probability = -np.logaddexp(0., -logits)
    return (weights["noc"]*log_probability[...,0] + weights["dac"]*log_probability[...,1]
            + weights["ddc"]*log_probability[...,5] + np.log(weights["ep"]*probability[...,2]
            + weights["ttc"]*probability[...,3] + weights["comfort"]*probability[...,4]))


def evaluate_pool_job(job):
    record, trajectories, neural_rankings, native_count = job
    remaining_count = len(trajectories)-native_count
    # The helper restores reference-plus-individual-candidate progress; the
    # neural arrays here only pick additional official equivalence checks.
    row = official_scene_scores((record, trajectories, neural_rankings[:native_count],
        neural_rankings[native_count:], neural_rankings[native_count:native_count+min(16,remaining_count)]))
    return row


def evaluate_jobs(configuration, jobs, destination, progress_path):
    started = time.perf_counter()
    values, check_count, maximum_error = [], 0, 0.
    with ProcessPoolExecutor(max_workers=configuration["official_evaluation_workers"],
                             mp_context=multiprocessing.get_context("spawn")) as pool:
        for index, row in enumerate(pool.map(evaluate_pool_job, jobs, chunksize=1)):
            values.append(np.asarray(row["official_metrics"], dtype=np.float64))
            check_count += row["checked_candidate_count"]
            maximum_error = max(maximum_error,row["pair_evaluator_maximum_error"])
            if index == 0 or (index+1)%64 == 0:
                write_json(progress_path,{"scored_scenes":index+1,"total_scenes":len(jobs),"elapsed_seconds":time.perf_counter()-started})
                print(json.dumps({"phase":"official_labels","complete_scenes":index+1,"total":len(jobs)}),flush=True)
    assert all(np.isfinite(value).all() for value in values)
    if len({value.shape for value in values}) == 1:
        np.save(destination,np.stack(values))
    else:
        raise ValueError("Use one homogeneous candidate-count group per label phase")
    return {"scenes":len(jobs),"official_candidates":sum(len(job[1]) for job in jobs),
        "unmodified_pair_checks":check_count,"maximum_pair_evaluator_error":maximum_error,
        "cpu_wall_seconds":time.perf_counter()-started,"gpu_seconds":0.}


def control_family_candidates(reference, base_controls, reference_deviations, speed, family_indices, seed):
    import torch
    generator = torch.Generator().manual_seed(seed)
    candidate_count = len(family_indices)
    noise = torch.randn(candidate_count,8,2,generator=generator)
    # Temporal correlation reduces needlessly jagged independent control noise.
    noise = (noise + noise.mean(1,keepdim=True))/np.sqrt(1.375)
    controls = base_controls[None].expand(candidate_count,-1,-1).clone()
    raw = controls.clone()
    for candidate_index,family in enumerate(family_indices):
        scale = 0.15 if family == 0 else (0.65 if family == 1 else 0.3)
        residual = scale*noise[candidate_index]
        if family == 2:
            residual[:,0] += 0.75
        elif family == 3:
            residual[:,0] -= 0.75
        elif family == 4:
            residual[:,1] += 0.75
        elif family == 5:
            residual[:,1] -= 0.75
        elif family == 6:
            residual[:,0] -= torch.linspace(1.25,0.,8)
        elif family == 7:
            residual[:,1] += torch.linspace(0.,1.25,8)*(1 if noise[candidate_index,0,1]>=0 else -1)
        raw[candidate_index] += reference_deviations*residual
    controls = reference.clamp_controls(raw)
    trajectories,speeds = reference.bicycle_rollout(controls[None],speed.reshape(1),.5)
    return trajectories[0], {"clipped_control_fraction":float((controls!=raw).float().mean()),
        "maximum_abs_acceleration":float(controls[...,0].abs().max()),
        "maximum_abs_yaw_rate":float(controls[...,1].abs().max())}


def boundary_candidates(record, ground_truth_trajectory, candidate_count):
    """Training-only geometric boundary acquisition, before simulator labels.

    Inspired by Versatile Samples. It uses untracked GT footprints and a coarse
    geometry scan, not that paper's simulator-in-the-loop margin solver.
    Geometry probes are counted separately from official trajectory labels.
    """
    from shapely import union_all
    from shapely.geometry import Polygon
    sys.path.insert(0,str(PROJECT_ROOT/"reference_repositories/DrivoR"))
    with lzma.open(record["metric_cache_file"],"rb") as stream:
        metric_cache=pickle.load(stream)
    drivable_geometry=union_all(metric_cache.drivable_area_map._geometries)
    rear_axle=metric_cache.ego_state.rear_axle
    ego_heading=float(rear_axle.heading)
    rotation=np.array([[np.cos(ego_heading),-np.sin(ego_heading)],[np.sin(ego_heading),np.cos(ego_heading)]])
    origin=np.array([rear_axle.x,rear_axle.y])
    parameters=metric_cache.ego_state.car_footprint.vehicle_parameters
    half_width=parameters.width/2
    longitudinal_offsets=np.array([parameters.rear_axle_to_center+parameters.length/2,
                                   parameters.rear_axle_to_center-parameters.length/2])
    path=np.asarray(ground_truth_trajectory,dtype=float)
    positions=np.concatenate((np.zeros((1,2)),path[:,:2]),0)
    distance=np.linalg.norm(np.diff(positions,axis=0),axis=-1)
    stations=np.concatenate(([0.],np.cumsum(distance)))
    ramp=np.clip(stations[1:]/10.,0.,1.)
    ramp=ramp*ramp*(3-2*ramp)
    geometry_queries=0

    def footprint(trajectory,index):
        center=trajectory[index,:2]
        heading=trajectory[index,2]
        direction=np.array([np.cos(heading),np.sin(heading)])
        normal=np.array([-direction[1],direction[0]])
        corners=np.stack([center+longitudinal_offsets[0]*direction+half_width*normal,
            center+longitudinal_offsets[0]*direction-half_width*normal,
            center+longitudinal_offsets[1]*direction-half_width*normal,
            center+longitudinal_offsets[1]*direction+half_width*normal])
        return Polygon(corners@rotation.T+origin)

    def recompute_headings(trajectory):
        differences=np.diff(np.concatenate((np.zeros((1,2)),trajectory[:,:2]),0),axis=0)
        moving=np.linalg.norm(differences,axis=-1)>1e-5
        trajectory[moving,2]=np.arctan2(differences[moving,1],differences[moving,0])
        return trajectory

    def lateral(offset):
        result=path.copy()
        normal=np.stack((-np.sin(path[:,2]),np.cos(path[:,2])),-1)
        result[:,:2]+=offset*ramp[:,None]*normal
        return recompute_headings(result)

    def longitudinal(offset):
        result=path.copy()
        targets=stations[1:]+offset*ramp
        for coordinate in range(2):
            result[:,coordinate]=np.interp(targets,stations,positions[:,coordinate])
        past_end=(targets-stations[-1]).clip(min=0.)
        result[:,:2]+=past_end[:,None]*np.array([np.cos(path[-1,2]),np.sin(path[-1,2])])
        return recompute_headings(result)

    lateral_thresholds=[]
    for sign in (-1,1):
        threshold=8.
        for offset in np.linspace(.25,8.,24):
            proposed=lateral(sign*offset)
            violated=False
            for step in range(8):
                geometry_queries+=1
                if not drivable_geometry.covers(footprint(proposed,step)):
                    violated=True
                    break
            if violated:
                threshold=float(offset)
                break
        lateral_thresholds.append(sign*threshold)
    contact_threshold=12.
    contact_found=False
    for offset in np.linspace(.25,12.,24):
        proposed=longitudinal(offset)
        for step in range(8):
            geometry_queries+=1
            intersected=metric_cache.observation[(step+1)*5].intersects(footprint(proposed,step))
            relevant=[token for token in intersected if "red_light" not in token
                      and token not in metric_cache.observation.collided_track_ids]
            if relevant:
                contact_threshold=float(offset)
                contact_found=True
                break
        if contact_found:
            break
    candidates=[]
    factors=np.linspace(.65,1.35,candidate_count//4)
    for threshold in lateral_thresholds:
        candidates.extend(lateral(threshold*factor) for factor in factors)
    candidates.extend(longitudinal(contact_threshold*factor) for factor in np.linspace(.65,1.35,candidate_count//2))
    return np.stack(candidates).astype("float32"), {"geometry_queries":geometry_queries,
        "lateral_thresholds_m":lateral_thresholds,"contact_threshold_m":contact_threshold,
        "contact_found":contact_found,"official_labels_used_to_construct":0,
        "input": "training GT/map/future occupancy only; never development/inference input"}


def boundary_geometry_job(job):
    record,trajectory=job
    paths,diagnostic=boundary_candidates(record,trajectory,32)
    return record["pilot_row"],paths,diagnostic


def prepare_boundary_geometry(configuration):
    output,registration=load_registration(configuration)
    cache=load_feature_cache(output)
    rows=registration["records"][configuration["common_scout_scenes"]:configuration["training_scenes"]]
    destination=np.lib.format.open_memmap(output/"boundary_geometry_candidates.npy",mode="w+",dtype="float32",shape=(configuration["training_scenes"],32,8,3))
    started=time.perf_counter()
    diagnostics={}
    jobs=[(record,np.array(cache["ground_truth_trajectory"][record["pilot_row"]])) for record in rows]
    with ProcessPoolExecutor(max_workers=configuration["official_evaluation_workers"],mp_context=multiprocessing.get_context("spawn")) as pool:
        for index,(row,paths,diagnostic) in enumerate(pool.map(boundary_geometry_job,jobs,chunksize=1)):
            assert np.isfinite(paths).all()
            destination[row]=paths
            diagnostics[str(row)]=diagnostic
            if (index+1)%64 == 0:
                print(json.dumps({"phase":"boundary_geometry","completed_scenes":index+1,"total":len(rows)}),flush=True)
    destination.flush()
    write_json(output/"boundary_geometry_complete.json",{"complete":True,"cpu_wall_seconds":time.perf_counter()-started,
        "gpu_seconds":0.,"official_label_queries":0,"geometry_queries":sum(value["geometry_queries"] for value in diagnostics.values()),
        "diagnostics":diagnostics})


def load_frozen_model(configuration):
    import torch
    from diagnose_residual_control_sampler import load_reference_helpers
    constructor,reference=load_reference_helpers(configuration)
    model=constructor("drivor",None)
    checkpoint=torch.load(PROJECT_ROOT/configuration["checkpoint"],map_location="cpu",mmap=True,weights_only=False)
    model.load_state_dict(checkpoint["model"],strict=True)
    model.eval().requires_grad_(False)
    def reject_encoder(_module,_inputs):
        raise RuntimeError("Encoder called in cached acquisition/training")
    model.backbone.register_forward_pre_hook(reject_encoder)
    return model,reference


def allocate_acquisition_arrays(configuration,output,scene_count):
    destination=output/"acquisition"
    destination.mkdir(exist_ok=True)
    specifications={"trajectories":(32,8,3),"scored_features":(32,256),"component_logits":(32,6),"family_indices":(32,)}
    return {strategy:{name:np.lib.format.open_memmap(destination/(strategy+"_"+name+".npy"),mode="w+",
        dtype="int64" if name=="family_indices" else "float32",shape=(scene_count,*shape))
        for name,shape in specifications.items()} for strategy in STRATEGIES}


def acquire_scout(configuration):
    import torch
    from build_scorer_selection_cache import scorer_arrays
    torch.set_num_threads(configuration["cpu_threads"])
    torch.use_deterministic_algorithms(True)
    output,registration=load_registration(configuration)
    cache=load_feature_cache(output)
    model,reference=load_frozen_model(configuration)
    acquired=allocate_acquisition_arrays(configuration,output,configuration["training_scenes"])
    started=time.perf_counter()
    diagnostics=[]
    for record in registration["records"][:configuration["common_scout_scenes"]]:
        row=record["pilot_row"]
        families=np.repeat(np.arange(8),4)
        trajectories,diagnostic=control_family_candidates(reference,torch.from_numpy(np.array(cache["base_controls"][row])),
            torch.from_numpy(np.array(cache["reference_standard_deviations"][row])),
            torch.tensor(float(cache["ego_status"][row,3])).clamp_min(0),families,10000+row)
        with torch.no_grad():
            scored,logits=scorer_arrays(model,trajectories[None],torch.from_numpy(np.array(cache["scene_features"][row]))[None],
                torch.from_numpy(np.array(cache["ego_tokens"][row]))[None])
        for strategy in STRATEGIES:
            for name,value in (("trajectories",trajectories.numpy()),("scored_features",scored[0].numpy()),
                               ("component_logits",logits[0].numpy()),("family_indices",families)):
                acquired[strategy][name][row]=value
        diagnostics.append(diagnostic)
    for strategy in STRATEGIES:
        for array in acquired[strategy].values():
            array.flush()
    write_json(output/"scout_candidates_complete.json",{"complete":True,"scene_count":configuration["common_scout_scenes"],
        "identical_scout_candidates_all_strategies":True,"candidate_count":32,"encoder_cached_calls":0,
        "frozen_parameter_gradients_absent":all(parameter.grad is None for parameter in model.parameters()),
        "cpu_wall_seconds":time.perf_counter()-started,"gpu_seconds":0.,"control_diagnostics":diagnostics})


def labels_scout(configuration):
    output,registration=load_registration(configuration)
    cache=load_feature_cache(output)
    weights=json.loads((output/"cache_complete.json").read_text())["score_weights"]
    trajectories=np.load(output/"acquisition/A_uniform_trajectories.npy",mmap_mode="r")
    logits=np.load(output/"acquisition/A_uniform_component_logits.npy",mmap_mode="r")
    jobs=[]
    for record in registration["records"][:configuration["common_scout_scenes"]]:
        row=record["pilot_row"]
        combined=np.concatenate((cache["native_trajectories"][row],trajectories[row]),0)
        scores=numpy_log_scores(np.concatenate((cache["native_component_logits"][row],logits[row]),0),weights)
        jobs.append((record,combined,scores,64))
    summary=evaluate_jobs(configuration,jobs,output/"scout_official_metrics.npy",output/"scout_label_progress.json")
    write_json(output/"scout_labels_complete.json",{"complete":True,**summary})


def learn_sampler(configuration):
    import torch
    from planning_aware_future_prediction.models.scorer_selection_learning import SelectionErrorFamilySampler,build_sampler_condition
    torch.set_num_threads(configuration["cpu_threads"])
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(configuration["seed"])
    output,_=load_registration(configuration)
    cache=load_feature_cache(output)
    scout_count=configuration["common_scout_scenes"]
    condition=build_sampler_condition(*(torch.from_numpy(np.array(cache[name][:scout_count]))
        for name in ("scene_features","ego_status","base_controls","reference_standard_deviations")))
    labels=np.load(output/"scout_official_metrics.npy")
    logits=np.load(output/"acquisition/A_uniform_component_logits.npy",mmap_mode="r")[:scout_count]
    weights=json.loads((output/"cache_complete.json").read_text())["score_weights"]
    scores=numpy_log_scores(logits,weights)
    probabilities=np.exp((scores-scores.max(1,keepdims=True))/.25)
    probabilities/=probabilities.sum(1,keepdims=True)
    oracle=labels[:,:,6].max(1,keepdims=True)
    rewards=(probabilities*32*(oracle-labels[:,64:,6]).clip(min=0)).reshape(scout_count,8,4).mean(2)
    reward_targets=torch.from_numpy(rewards.astype("float32"))
    sampler=SelectionErrorFamilySampler(condition_dimension=condition.shape[-1],hidden_dimension=configuration["hidden_dimension"])
    optimizer=torch.optim.AdamW(sampler.parameters(),lr=configuration["learning_rate"])
    generator=torch.Generator().manual_seed(configuration["seed"]+71)
    started=time.perf_counter()
    first_loss=last_loss=None
    for step in range(configuration["sampler_training_steps"]):
        indices=torch.randperm(scout_count,generator=generator)[:32]
        prediction=sampler(condition[indices])
        loss=torch.nn.functional.mse_loss(prediction,reward_targets[indices])
        assert torch.isfinite(loss)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        assert all(parameter.grad is not None and torch.isfinite(parameter.grad).all() for parameter in sampler.parameters())
        torch.nn.utils.clip_grad_norm_(sampler.parameters(),5.)
        optimizer.step()
        first_loss=float(loss) if first_loss is None else first_loss
        last_loss=float(loss)
    sampler.eval()
    with torch.no_grad():
        predicted=sampler.probabilities(condition,configuration["sampler_temperature"],configuration["sampler_exploration_probability"])
        blanked=condition.clone()
        blanked[:,:256]=0
        blanked_probability=sampler.probabilities(blanked,configuration["sampler_temperature"],configuration["sampler_exploration_probability"])
    torch.save({"model":sampler.state_dict(),"condition_dimension":condition.shape[-1],"optimizer":optimizer.state_dict()},output/"sampler.pt")
    write_json(output/"sampler_complete.json",{"complete":True,"train_only_scout_scenes":scout_count,
        "optimizer_steps":configuration["sampler_training_steps"],"trainable_parameters":sum(p.numel() for p in sampler.parameters()),
        "first_minibatch_loss":first_loss,"last_minibatch_loss":last_loss,"cpu_wall_seconds":time.perf_counter()-started,
        "gpu_seconds":0.,"mean_family_probabilities":predicted.mean(0).tolist(),
        "visual_zero_probability_mean_absolute_change":float((predicted-blanked_probability).abs().mean()),
        "family_probability_scene_standard_deviation":predicted.std(0).tolist(),
        "reward_definition":"soft scorer selection probability times actual oracle-minus-candidate utility; averaged by acquisition family",
        "reward_targets_minimum":float(reward_targets.min()),"reward_targets_maximum":float(reward_targets.max()),
        "no_development_labels_used":True,"no_official_gradient":True})


def acquire_remaining(configuration):
    import torch
    from build_scorer_selection_cache import scorer_arrays
    from diagnose_residual_control_sampler import module_digest
    from planning_aware_future_prediction.models.scorer_selection_learning import SelectionErrorFamilySampler,build_sampler_condition
    torch.set_num_threads(configuration["cpu_threads"])
    torch.use_deterministic_algorithms(True)
    output,registration=load_registration(configuration)
    cache=load_feature_cache(output)
    model,reference=load_frozen_model(configuration)
    initial_digest=module_digest(model)
    saved=torch.load(output/"sampler.pt",map_location="cpu",weights_only=False)
    sampler=SelectionErrorFamilySampler(saved["condition_dimension"],hidden_dimension=configuration["hidden_dimension"])
    sampler.load_state_dict(saved["model"],strict=True)
    sampler.eval().requires_grad_(False)
    acquired={strategy:{name:np.load(output/"acquisition"/(strategy+"_"+name+".npy"),mmap_mode="r+")
        for name in ("trajectories","scored_features","component_logits","family_indices")} for strategy in STRATEGIES}
    diagnostics=[]
    started=time.perf_counter()
    boundary_cache=np.load(output/"boundary_geometry_candidates.npy",mmap_mode="r")
    boundary_metadata=json.loads((output/"boundary_geometry_complete.json").read_text())
    for record in registration["records"][configuration["common_scout_scenes"]:configuration["training_scenes"]]:
        row=record["pilot_row"]
        condition=build_sampler_condition(*(torch.from_numpy(np.array(cache[name][row]))[None]
            for name in ("scene_features","ego_status","base_controls","reference_standard_deviations")))
        with torch.no_grad():
            probabilities=sampler.probabilities(condition,configuration["sampler_temperature"],configuration["sampler_exploration_probability"])[0].numpy()
        sampling_rng=np.random.default_rng(configuration["seed"]+20000+row)
        uniform_variates=sampling_rng.random(32)
        learned_families=np.searchsorted(probabilities.cumsum(),uniform_variates).clip(max=7)
        uniform_families=(uniform_variates*8).astype("int64")
        boundary_paths=boundary_cache[row]
        boundary_diagnostic=boundary_metadata["diagnostics"][str(row)]
        scene=torch.from_numpy(np.array(cache["scene_features"][row]))[None]
        ego_token=torch.from_numpy(np.array(cache["ego_tokens"][row]))[None]
        for strategy in STRATEGIES:
            if strategy == "B_boundary":
                trajectories=torch.from_numpy(boundary_paths)
                families=np.full(32,-1,dtype="int64")
                diagnostic=boundary_diagnostic
            else:
                families=uniform_families if strategy == "A_uniform" else learned_families
                trajectories,diagnostic=control_family_candidates(reference,torch.from_numpy(np.array(cache["base_controls"][row])),
                    torch.from_numpy(np.array(cache["reference_standard_deviations"][row])),
                    torch.tensor(float(cache["ego_status"][row,3])).clamp_min(0),families,30000+row)
            with torch.no_grad():
                scored,logits=scorer_arrays(model,trajectories[None],scene,ego_token)
            for name,value in (("trajectories",trajectories.numpy()),("scored_features",scored[0].numpy()),
                               ("component_logits",logits[0].numpy()),("family_indices",families)):
                assert np.isfinite(value).all(),(strategy,row,name)
                acquired[strategy][name][row]=value
            diagnostics.append({"row":row,"strategy":strategy,**diagnostic})
        if (row+1)%64 == 0:
            write_json(output/"acquisition_progress.json",{"completed_training_scenes":row+1,"total_training_scenes":configuration["training_scenes"],"elapsed_seconds":time.perf_counter()-started})
            print(json.dumps({"phase":"acquisition","scenes":row+1,"elapsed_seconds":time.perf_counter()-started}),flush=True)
    for strategy in STRATEGIES:
        for array in acquired[strategy].values():
            array.flush()
    assert initial_digest == module_digest(model)
    assert all(parameter.grad is None for parameter in model.parameters())
    write_json(output/"acquisition_complete.json",{"complete":True,"cpu_wall_seconds":time.perf_counter()-started,
        "gpu_seconds":0.,"encoder_cached_calls":0,"frozen_weights_unchanged":True,"diagnostics":diagnostics,
        "sampler_updated":False,"sampler_used_for_acquisition_only":True,"inference_candidate_count_unchanged":True})


def labels_remaining(configuration):
    output,registration=load_registration(configuration)
    cache=load_feature_cache(output)
    weights=json.loads((output/"cache_complete.json").read_text())["score_weights"]
    acquisitions={strategy:{name:np.load(output/"acquisition"/(strategy+"_"+name+".npy"),mmap_mode="r")
        for name in ("trajectories","component_logits")} for strategy in STRATEGIES}
    training_jobs,development_jobs=[],[]
    for record in registration["records"]:
        row=record["pilot_row"]
        if record["common_scout"]:
            continue
        trajectories=[cache["native_trajectories"][row]]
        logits=[cache["native_component_logits"][row]]
        if record["pilot_split"] == "train":
            for strategy in STRATEGIES:
                trajectories.append(acquisitions[strategy]["trajectories"][row])
                logits.append(acquisitions[strategy]["component_logits"][row])
        else:
            trajectories.append(cache["residual_trajectories"][row])
            logits.append(cache["residual_component_logits"][row])
        job=(record,np.concatenate(trajectories,0),numpy_log_scores(np.concatenate(logits,0),weights),64)
        (training_jobs if record["pilot_split"] == "train" else development_jobs).append(job)
    training=evaluate_jobs(configuration,training_jobs,output/"training_remainder_official_metrics.npy",output/"training_label_progress.json")
    development=evaluate_jobs(configuration,development_jobs,output/"development_official_metrics.npy",output/"development_label_progress.json")
    write_json(output/"labels_complete.json",{"complete":True,"training":training,"development":development,
        "metric_names":METRIC_NAMES,"training_official_labels_per_strategy_per_scene":96,
        "development_official_labels_per_scene":128,"shared_labels_physically_computed_once":True,
        "all_privileged_map_future_information_is_training_label_only":True})


def train_rankers(configuration):
    import torch
    from planning_aware_future_prediction.models.scorer_selection_learning import CachedScorerCorrection,compute_scorer_correction_losses,compose_drivor_log_score
    torch.set_num_threads(configuration["cpu_threads"])
    torch.use_deterministic_algorithms(True)
    output,registration=load_registration(configuration)
    cache=load_feature_cache(output)
    weights=json.loads((output/"cache_complete.json").read_text())["score_weights"]
    scout=np.load(output/"scout_official_metrics.npy",mmap_mode="r")
    remaining=np.load(output/"training_remainder_official_metrics.npy",mmap_mode="r")
    development=np.load(output/"development_official_metrics.npy",mmap_mode="r")
    tuning_rows=[record["pilot_row"] for record in registration["records"] if record["pilot_split"] == "tuning"]
    torch.manual_seed(configuration["seed"])
    common_model=CachedScorerCorrection(configuration["hidden_dimension"],configuration["maximum_logit_correction"])
    common_state=copy.deepcopy(common_model.state_dict())
    torch.save(common_state,output/"common_correction_initialization.pt")
    generator=torch.Generator().manual_seed(configuration["seed"]+41)
    batch_order=torch.stack([torch.randperm(configuration["training_scenes"],generator=generator)[:configuration["scorer_batch_size"]]
        for _ in range(configuration["scorer_training_steps"])]).numpy()
    pairs=torch.randint(0,96,(configuration["scorer_training_steps"],configuration["scorer_batch_size"],configuration["pairs_per_scene"],2),generator=generator)
    np.save(output/"training_batch_order.npy",batch_order)
    np.save(output/"training_pair_indices.npy",pairs.numpy())
    summaries={}
    for strategy_index,strategy in enumerate(STRATEGIES):
        model=CachedScorerCorrection(configuration["hidden_dimension"],configuration["maximum_logit_correction"])
        model.load_state_dict(common_state,strict=True)
        optimizer=torch.optim.AdamW(model.parameters(),lr=configuration["learning_rate"])
        acquired={name:np.load(output/"acquisition"/(strategy+"_"+name+".npy"),mmap_mode="r")
            for name in ("trajectories","scored_features","component_logits")}
        full_targets=np.zeros((configuration["training_scenes"],96,7),dtype="float32")
        scout_count=configuration["common_scout_scenes"]
        full_targets[:scout_count]=scout
        full_targets[scout_count:,:64]=remaining[:,:64]
        full_targets[scout_count:,64:]=remaining[:,64+strategy_index*32:64+(strategy_index+1)*32]
        logs=[]
        started=time.perf_counter()
        best_tuning=-float("inf")
        best_step=None
        maximum_gradient=0.
        for step,selected_rows in enumerate(batch_order,1):
            inputs={"scored_features":torch.from_numpy(np.concatenate((cache["native_scored_features"][selected_rows],acquired["scored_features"][selected_rows]),1)),
                "scene_features":torch.from_numpy(np.array(cache["scene_features"][selected_rows])),
                "ego_status":torch.from_numpy(np.array(cache["ego_status"][selected_rows])),
                "trajectories":torch.from_numpy(np.concatenate((cache["native_trajectories"][selected_rows],acquired["trajectories"][selected_rows]),1)),
                "component_logits":torch.from_numpy(np.concatenate((cache["native_component_logits"][selected_rows],acquired["component_logits"][selected_rows]),1))}
            corrected=model(**inputs)
            targets=torch.from_numpy(full_targets[selected_rows])
            current_pairs=pairs[step-1].clone()
            # Ensure winner/oracle pairs are represented, in the same rule for
            # each method; additional pairs are fixed common random indices.
            current_pairs[:,0,0]=compose_drivor_log_score(inputs["component_logits"],weights).argmax(1)
            current_pairs[:,0,1]=targets[:,:,6].argmax(1)
            component_loss,ranking_loss=compute_scorer_correction_losses(corrected,targets,weights,current_pairs,configuration["ranking_temperature"])
            loss=configuration["component_loss_weight"]*component_loss+configuration["ranking_loss_weight"]*ranking_loss
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            gradients=[parameter.grad for parameter in model.parameters() if parameter.grad is not None]
            assert gradients and all(torch.isfinite(gradient).all() for gradient in gradients)
            gradient_norm=float(torch.nn.utils.clip_grad_norm_(model.parameters(),5.))
            maximum_gradient=max(maximum_gradient,gradient_norm)
            assert torch.isfinite(loss)
            optimizer.step()
            if step == 1 or step%25 == 0:
                logs.append({"step":step,"loss":float(loss),"component_loss":float(component_loss),
                    "ranking_loss":float(ranking_loss),"gradient_norm_before_clip":gradient_norm,
                    "elapsed_seconds":time.perf_counter()-started})
            if step in configuration["checkpoint_steps"]:
                model.eval()
                scores=predict_cached(model,cache,tuning_rows,"native",weights)
                selected=scores.argmax(1)
                tuning_metric=development[:len(tuning_rows),:64,6][np.arange(len(tuning_rows)),selected].mean()*100
                torch.save({"model":model.state_dict(),"optimizer":optimizer.state_dict(),"completed_steps":step},output/(strategy+"_step"+str(step)+".pt"))
                if tuning_metric>best_tuning:
                    best_tuning=float(tuning_metric)
                    best_step=step
                print(json.dumps({"phase":"train","strategy":strategy,"step":step,"tuning_pdms":float(tuning_metric)}),flush=True)
                model.train()
        summaries[strategy]={"completed_steps":configuration["scorer_training_steps"],
            "trainable_parameters":sum(parameter.numel() for parameter in model.parameters()),
            "best_tuning_step":best_step,"best_tuning_pdms":best_tuning,
            "cpu_wall_seconds":time.perf_counter()-started,"gpu_seconds":0.,
            "maximum_gradient_norm_before_clip":maximum_gradient,"logs":logs}
        write_json(output/(strategy+"_training.json"),summaries[strategy])
    write_json(output/"training_complete.json",{"complete":True,"strategies":summaries,
        "common_initialization_sha256":file_sha256(output/"common_correction_initialization.pt"),
        "batch_order_sha256":file_sha256(output/"training_batch_order.npy"),
        "pair_indices_sha256":file_sha256(output/"training_pair_indices.npy"),
        "cache_only_encoder_calls":0,"cache_only_frozen_scorer_calls":0,
        "protected_training":protected_training_audit(configuration)})


def predict_cached(model,cache,rows,pool_name,score_weights,correction_enabled=True):
    import torch
    from planning_aware_future_prediction.models.scorer_selection_learning import compose_drivor_log_score
    scores=[]
    with torch.no_grad():
        for start in range(0,len(rows),16):
            selected=rows[start:start+16]
            corrected=model(torch.from_numpy(np.array(cache[pool_name+"_scored_features"][selected])),
                torch.from_numpy(np.array(cache["scene_features"][selected])),
                torch.from_numpy(np.array(cache["ego_status"][selected])),
                torch.from_numpy(np.array(cache[pool_name+"_trajectories"][selected])),
                torch.from_numpy(np.array(cache[pool_name+"_component_logits"][selected])),correction_enabled=correction_enabled)
            scores.append(compose_drivor_log_score(corrected,score_weights).numpy())
    return np.concatenate(scores,0)


def report(configuration):
    import torch
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plotting
    from planning_aware_future_prediction.models.scorer_selection_learning import CachedScorerCorrection
    torch.set_num_threads(configuration["cpu_threads"])
    output,registration=load_registration(configuration)
    result_directory=PROJECT_ROOT/configuration["results_directory"]
    result_directory.mkdir(exist_ok=True,parents=True)
    cache=load_feature_cache(output)
    weights=json.loads((output/"cache_complete.json").read_text())["score_weights"]
    records=[record for record in registration["records"] if record["pilot_split"] == "locked_development"]
    rows=[record["pilot_row"] for record in records]
    groups=[record["recording_group"] for record in records]
    all_dev_records=[record for record in registration["records"] if record["pilot_split"] != "train"]
    relative_rows=[all_dev_records.index(record) for record in records]
    labels=np.load(output/"development_official_metrics.npy")[relative_rows]
    train=json.loads((output/"training_complete.json").read_text())
    predictions={"unchanged":{pool:numpy_log_scores(cache[pool+"_component_logits"][rows],weights) for pool in ("native","residual")}}
    costs={}
    correction_checks={}
    for strategy in STRATEGIES:
        saved=torch.load(output/(strategy+"_step"+str(train["strategies"][strategy]["best_tuning_step"])+".pt"),map_location="cpu",weights_only=False)
        model=CachedScorerCorrection(configuration["hidden_dimension"],configuration["maximum_logit_correction"])
        model.load_state_dict(saved["model"],strict=True)
        model.eval()
        started=time.perf_counter()
        predictions[strategy]={pool:predict_cached(model,cache,rows,pool,weights) for pool in ("native","residual")}
        inference_seconds=time.perf_counter()-started
        off=predict_cached(model,cache,rows,"native",weights,False)
        correction_checks[strategy]={"off_baseline_max_error":float(np.abs(off-predictions["unchanged"]["native"]).max()),
            "off_baseline_selected_index_match":bool(np.array_equal(off.argmax(1),predictions["unchanged"]["native"].argmax(1))),
            "on_baseline_score_mean_absolute_change":float(np.abs(predictions[strategy]["native"]-off).mean())}
        assert correction_checks[strategy]["off_baseline_max_error"]<2e-6
        costs[strategy]={"cached_correction_inference_ms_per_scene_batched16":1000*inference_seconds/(2*len(rows)),
                         "training":train["strategies"][strategy]}
    summaries={}
    per_scene_rows=[]
    selected_values={}
    for name,pools in predictions.items():
        summaries[name]={}
        for pool_index,pool_name in enumerate(("native","residual")):
            current_labels=labels[:,pool_index*64:(pool_index+1)*64]
            selected=pools[pool_name].argmax(1)
            chosen=current_labels[np.arange(len(records)),selected]
            oracle=current_labels[:,:,6].max(1)*100
            selected_pdms=chosen[:,6]*100
            regret=oracle-selected_pdms
            selected_values[(name,pool_name)]=selected_pdms
            summary={"scenes":len(records),"recordings":len(set(groups)),"candidate_count":64,
                "selected_pdms":float(selected_pdms.mean()),"oracle_pdms":float(oracle.mean()),
                "selection_regret":float(regret.mean()),"regret_recording_bootstrap_95ci":recording_bootstrap(regret,groups,73,configuration["bootstrap_repetitions"]),
                "collision_or_offroad_rate":float(((chosen[:,0]<1)|(chosen[:,1]<1)).mean()),
                "selected_submetrics":{metric:float(chosen[:,index].mean()) for index,metric in enumerate(METRIC_NAMES[:6])},
                "by_command":{str(command):{"scenes":sum(record["command_index"]==command for record in records),
                    "selected_pdms":float(selected_pdms[[record["command_index"]==command for record in records]].mean()),
                    "regret":float(regret[[record["command_index"]==command for record in records]].mean())}
                    for command in sorted({record["command_index"] for record in records})}}
            if name != "unchanged":
                difference=selected_pdms-selected_values[("unchanged",pool_name)]
                summary["difference_from_unchanged"]=float(difference.mean())
                summary["difference_recording_bootstrap_95ci"]=recording_bootstrap(difference,groups,73,configuration["bootstrap_repetitions"])
            summaries[name][pool_name]=summary
            for index,record in enumerate(records):
                per_scene_rows.append({"strategy":name,"pool":pool_name,"token":record["token"],"recording_group":record["recording_group"],
                    "command_index":record["command_index"],"selected_candidate_index":int(selected[index]),
                    "selected_pdms":float(selected_pdms[index]),"oracle_pdms":float(oracle[index]),"selection_regret":float(regret[index]),
                    **{metric:float(chosen[index,metric_index]) for metric_index,metric in enumerate(METRIC_NAMES[:6])}})
    difference=selected_values[("C_selection_error","native")]-selected_values[("A_uniform","native")]
    comparison_ci=recording_bootstrap(difference,groups,73,configuration["bootstrap_repetitions"])
    promotion=(comparison_ci[0]>0 and summaries["C_selection_error"]["native"]["collision_or_offroad_rate"] <= summaries["A_uniform"]["native"]["collision_or_offroad_rate"]+.01)
    with (result_directory/"scene_results.csv").open("w") as stream:
        writer=csv.DictWriter(stream,fieldnames=list(per_scene_rows[0]))
        writer.writeheader()
        writer.writerows(per_scene_rows)
    columns=["strategy","selected_pdms","oracle_pdms","selection_regret","collision_or_offroad_rate"]
    with (result_directory/"comparison.csv").open("w") as stream:
        writer=csv.DictWriter(stream,fieldnames=columns)
        writer.writeheader()
        for name in summaries:
            writer.writerow({"strategy":name,**{column:summaries[name]["native"][column] for column in columns[1:]}})
    sampler=json.loads((output/"sampler_complete.json").read_text())
    cache_metadata=json.loads((output/"cache_complete.json").read_text())
    scout_labels=json.loads((output/"scout_labels_complete.json").read_text())
    labels_metadata=json.loads((output/"labels_complete.json").read_text())
    official_candidates=scout_labels["official_candidates"]+labels_metadata["training"]["official_candidates"]+labels_metadata["development"]["official_candidates"]
    results={"complete":True,"completed_utc":datetime.now(timezone.utc).isoformat(),"configuration":configuration,
        "summaries":summaries,"costs":costs,"sampler":sampler,"cache":cache_metadata,
        "C_minus_A_selected_pdms":float(difference.mean()),"C_minus_A_recording_bootstrap_95ci":comparison_ci,
        "replication_gate_passed":promotion,"full_scale_started":False,"independent_final_test":False,
        "actual_official_candidates_evaluated":official_candidates,"effective_training_label_budget_per_strategy":configuration["training_scenes"]*96,
        "official_label_cost_cpu_seconds":scout_labels["cpu_wall_seconds"]+labels_metadata["training"]["cpu_wall_seconds"]+labels_metadata["development"]["cpu_wall_seconds"],
        "correction_off_checks":correction_checks,"protected_training":protected_training_audit(configuration)}
    write_json(result_directory/"results.json",results)
    write_json(result_directory/"validation.json",{"cache":cache_metadata,"off_checks":correction_checks,
        "official_pair_equivalence_max_error":max(scout_labels["maximum_pair_evaluator_error"],labels_metadata["training"]["maximum_pair_evaluator_error"],labels_metadata["development"]["maximum_pair_evaluator_error"]),
        "no_encoder_or_frozen_scorer_calls_during_training":True,"common_batch_order":True,
        "tuning_locked_recording_overlap":0,"train_development_recording_overlap":0})
    np.savez_compressed(result_directory/"fixed_pool_predictions.npz",tokens=np.array([record["token"] for record in records]),
        official_metrics=labels,**{name+"_"+pool:scores for name,pools in predictions.items() for pool,scores in pools.items()})
    figure,axes=plotting.subplots(1,3,figsize=(15,4.5))
    names=list(summaries)
    display_names=["Frozen baseline","Uniform","Boundary","Selection error"]
    axes[0].bar(display_names,[summaries[name]["native"]["selected_pdms"] for name in names])
    axes[0].axhline(summaries["unchanged"]["native"]["oracle_pdms"],color="black",linestyle="--",label="Fixed-pool oracle")
    axes[0].set_ylabel("PDMS (points)")
    axes[0].legend()
    axes[1].bar(display_names,[summaries[name]["native"]["selection_regret"] for name in names])
    axes[1].set_ylabel("Selection regret (points)")
    axes[2].bar(configuration["acquisition_families"],sampler["mean_family_probabilities"])
    axes[2].set_ylabel("Learned acquisition probability")
    for axis in axes:
        axis.tick_params(axis="x",rotation=35)
    figure.suptitle("Fixed native64 pool / locked NAVVAL development / one seed")
    figure.tight_layout()
    figure.savefig(result_directory/"selection_regret_comparison.png",dpi=160)
    plotting.close(figure)
    markdown=["# Selection-error acquisition and frozen-scorer correction pilot", "",
        "## Question", "", configuration["research_question"], "",
        "## Actual locked development results", "",
        "192 previously used NAVVAL development scenes; recording-disjoint from fitting and tuning. One seed, not independent NAVTEST.", "",
        "| Method | Selected PDMS | Fixed oracle | Selection regret | Collision/offroad |", "|---|---:|---:|---:|---:|"]
    for name in names:
        summary=summaries[name]["native"]
        markdown.append(f"| {name} | {summary['selected_pdms']:.4f} | {summary['oracle_pdms']:.4f} | {summary['selection_regret']:.4f} | {100*summary['collision_or_offroad_rate']:.2f}% |")
    markdown.extend(["",f"C minus A: {difference.mean():+.4f} points; paired recording 95% interval {comparison_ci}. Replication gate passed: {promotion}.",
        "", "## Controls and interpretation", "",
        "All rankers use identical frozen encoder/generator/scorer weights and native64 candidate contexts. The six corrected logits use the original scalar composition. A/B/C have identical correction initialization, parameter count, optimizer, batch order, pairs, 500 fitting steps and 96 official labels per training scene. Each method uses 64 native and 32 acquired candidates, with the first128 common scouts identical.",
        "", "The first128 training scenes supervise an eight-family sampler using true utility gaps weighted by soft frozen-scorer selection probability. Its300 CPU updates use no development labels. C chooses later acquisition families using scene/ego/base controls/spread. The sampler is used during data acquisition only; inference uses the same frozen native proposals.",
        "", "Boundary baseline uses training GT, map and future occupancy for coarse geometric boundary scans. It does not reproduce the Versatile Samples simulator-margin solver. Geometry probes cost time but consume no official trajectory-score labels; their counts are recorded. Boundary paths are not guaranteed comfort compliant; official tracked execution supplies the labels.",
        "", "The component loss is the mean six-subscore BCE (soft progress target); the additional ranking loss is actual-score-gap-weighted pairwise softplus. This is an adaptation objective, not an exact re-execution of original DrivoR training. Frozen score composition retains NAVSIM v1 DDC weight zero; DDC is reported separately.",
        "", "The sampler reward is a proxy for useful data acquisition, not a directly differentiated final selection-regret objective. Reparameterization is not used to differentiate through the official evaluator or argmax. A single learning cycle does not establish optimal active sampling or novelty.",
        "", "The base checkpoint has only3200 small-corpus updates. Correcting this weak scorer does not establish benefit on a fully trained published planner. Candidate self-attention means representations depend on the candidate context: each evaluation context is kept identical, while training acquired contexts vary as part of the acquisition intervention.",
        "", "Tuning selects one of100/250/500 steps on64 separate-recording scenes; locked192 results are read once after selection. All500 steps are executed for every method. These recordings have been used in previous development work; this is not a pristine final holdout. Bootstrap describes recording variation, not learning-seed uncertainty.",
        "", "## Costs", "", f"Cache GPU occupancy {cache_metadata['gpu_occupied_seconds']:.2f}s, peak allocated {cache_metadata['maximum_allocated_gpu_bytes']/1024**3:.3f}GiB. Sampler and correction training GPU0s. Official candidates physically evaluated: {official_candidates}; shared labels counted once physically and equally in each method's budget.",
        "", "Actual per-method fitting, correction inference, official labeling and sampler costs are in results.json. Cached correction latency excludes the already-frozen encoder and original scorer; total deployment latency must be profiled later.",
        "", "## Next experiment", "", "If the preregistered gate passes, repeat this same controlled comparison with another training seed and a separate evaluation-recording panel. If it fails, inspect native-progress/DAC ranking and acquired-pair coverage before adding modules or scaling training.",
        "", "## Related work (verified on2026-10-11)", "", "- CLOVER: https://arxiv.org/html/2605.15120 (alternating scorer fitting and conservative generator distillation)",
        "- Designing Versatile Samples for Learned Trajectory Scoring: https://arxiv.org/html/2609.01799 (training-only boundary candidates)",
        "", "## Replay", "", "```bash", "cd /rhome/junseong/PlanningAwareFuturePrediction",
        "CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 PYTHONPATH=src runtime/environments/kjs-scorer-selection-pilot/bin/python -u scripts/run_scorer_selection_pilot.py --configuration configs/scorer_selection_learning/pilot_v1.json --phase run", "```",
        "", "Completed phases are not repeated; new scientific settings require a new config/output registration.",
        "", "![Comparison](selection_regret_comparison.png)"])
    (result_directory/"report.md").write_text("\n".join(markdown)+"\n")
    write_json(output/"complete.json",{"complete":True,"replication_gate_passed":promotion,"results_directory":str(result_directory)})
    print(json.dumps({"complete":True,"scores":{name:summaries[name]["native"]["selected_pdms"] for name in summaries},"replication_gate":promotion}),flush=True)


def run(configuration,configuration_path,resume_environment_fix=False):
    output,_=load_registration(configuration)
    assert (output/"cache_complete.json").exists(),"Run cache builder first"
    source_paths=[Path(__file__),PROJECT_ROOT/"src/planning_aware_future_prediction/models/scorer_selection_learning.py",configuration_path]
    source_hashes={str(path.relative_to(PROJECT_ROOT)):file_sha256(path) for path in source_paths}
    source_registration=output/"learning_source_registration.json"
    if source_registration.exists():
        previous_sources=json.loads(source_registration.read_text())["sources"]
        if previous_sources != source_hashes:
            assert resume_environment_fix,"Scientific source changed; require explicit environment-fix resume"
            assert json.loads((output/"queue_state.json").read_text())["status"] == "failed"
            previous_script=output/"source_snapshot_attempt1/run_scorer_selection_pilot.py"
            original_tree=ast.parse(previous_script.read_text())
            revised_tree=ast.parse(Path(__file__).read_text())
            unchanged_functions=("validate_cache","acquire_scout","labels_scout","learn_sampler","control_family_candidates",
                                 "evaluate_pool_job","evaluate_jobs","train_rankers","report")
            def function_ast(tree,name):
                return ast.dump(next(node for node in tree.body if isinstance(node,ast.FunctionDef) and node.name==name))
            assert all(function_ast(original_tree,name)==function_ast(revised_tree,name) for name in unchanged_functions)
            write_json(output/"execution_environment_fix.json",{"reason":"Model environment lacks nuplan; compute training-only boundary geometry in existing official evaluator environment",
                "previous_sources":previous_sources,"revised_sources":source_hashes,"unchanged_completed_and_training_function_AST":list(unchanged_functions),
                "configuration_unchanged":True,"optimizer_scout_state_reused":True,"new_official_labels_for_completed_phases":0})
            write_json(source_registration,{"registered_utc":datetime.now(timezone.utc).isoformat(),"sources":source_hashes,
                "amendment":"execution_environment_fix.json"})
    else:
        write_json(source_registration,{"registered_utc":datetime.now(timezone.utc).isoformat(),"sources":source_hashes})
    official_environment=PROJECT_ROOT/"runtime/environments/drive_jepa_official_evaluation"
    model_python=PROJECT_ROOT/"runtime/environments/kjs-scorer-selection-pilot/bin/python"
    stages=(("validate_cache",model_python,"cache_validation_complete.json"),
        ("acquire_scout",model_python,"scout_candidates_complete.json"),
        ("labels_scout",official_environment/"bin/python","scout_labels_complete.json"),
        ("learn_sampler",model_python,"sampler_complete.json"),
        ("prepare_boundary_geometry",official_environment/"bin/python","boundary_geometry_complete.json"),
        ("acquire_remaining",model_python,"acquisition_complete.json"),
        ("labels_remaining",official_environment/"bin/python","labels_complete.json"),
        ("train_rankers",model_python,"training_complete.json"),
        ("report",model_python,"complete.json"))
    started=time.time()
    for phase,interpreter,marker in stages:
        if (output/marker).exists():
            continue
        assert all(file_sha256(PROJECT_ROOT/path)==digest for path,digest in source_hashes.items())
        environment={**os.environ,"CUDA_VISIBLE_DEVICES":"","OMP_NUM_THREADS":"2","OPENBLAS_NUM_THREADS":"1","MKL_NUM_THREADS":"1","PYTHONPATH":str(PROJECT_ROOT/"src")}
        if phase.startswith("labels") or phase == "prepare_boundary_geometry":
            environment["LD_LIBRARY_PATH"]=str(official_environment/"lib")
            environment["OMP_NUM_THREADS"]="1"
        with (output/(phase+".log")).open("a") as stream:
            process=subprocess.Popen([str(interpreter),"-u",str(Path(__file__)),"--configuration",str(configuration_path),"--phase",phase],
                cwd=PROJECT_ROOT,env=environment,stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT)
            write_json(output/"queue_state.json",{"status":"running","phase":phase,"controller_pid":os.getpid(),"child_pid":process.pid})
            code=process.wait()
        if code or not (output/marker).exists():
            write_json(output/"queue_state.json",{"status":"failed","phase":phase,"return_code":code,"log":phase+".log"})
            raise RuntimeError("Failed phase: "+phase)
    write_json(output/"queue_state.json",{"status":"complete","wall_seconds":time.time()-started})


if __name__ == "__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--configuration",type=Path,required=True)
    parser.add_argument("--phase",choices=("validate_cache","acquire_scout","labels_scout","learn_sampler","prepare_boundary_geometry","acquire_remaining","labels_remaining","train_rankers","report","run"),required=True)
    parser.add_argument("--resume-environment-fix",action="store_true")
    arguments=parser.parse_args()
    configuration=json.loads(arguments.configuration.read_text())
    if arguments.phase == "run":
        run(configuration,arguments.configuration.resolve(),arguments.resume_environment_fix)
    else:
        globals()[arguments.phase](configuration)
