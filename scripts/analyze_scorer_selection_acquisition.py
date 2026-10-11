#!/usr/bin/env python
"""Read-only diagnostics of acquisition coverage and fixed-pool ranking."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
from scipy.stats import spearmanr

PROJECT_ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(PROJECT_ROOT/"scripts"))
from calibrate_frozen_drivor_scorer import recording_bootstrap


def main(configuration_path):
    configuration=json.loads(configuration_path.read_text())
    output=PROJECT_ROOT/configuration["output_directory"]
    destination=PROJECT_ROOT/configuration["results_directory"]
    registration=json.loads((output/"registration.json").read_text())
    results=json.loads((destination/"results.json").read_text())
    candidates=np.load(destination/"fixed_pool_predictions.npz")
    records=[record for record in registration["records"] if record["pilot_split"] == "locked_development"]
    groups=[record["recording_group"] for record in records]
    labels=candidates["official_metrics"][:,:64,6]
    names=("unchanged","A_uniform","B_boundary","C_selection_error")
    selected_scores={}
    diagnostic={"fixed_native_pool":{},"later_acquisition":{}}
    baseline_selected=candidates["unchanged_native"].argmax(1)
    for name in names:
        neural=candidates[name+"_native"]
        correlations=[]
        for row in range(len(labels)):
            if np.ptp(labels[row])>1e-10 and np.ptp(neural[row])>1e-10:
                correlations.append(float(spearmanr(labels[row],neural[row]).statistic))
        chosen=neural.argmax(1)
        selected_scores[name]=labels[np.arange(len(labels)),chosen]*100
        official_gap=labels[:,:,None]-labels[:,None,:]
        neural_gap=neural[:,:,None]-neural[:,None,:]
        non_tied=np.abs(official_gap)>1e-6
        rank_correct=(official_gap*neural_gap)>0
        weights=np.abs(official_gap)
        weighted_correct=(weights*rank_correct*non_tied).sum()/np.maximum((weights*non_tied).sum(),1e-12)
        diagnostic["fixed_native_pool"][name]={"nonconstant_spearman_scenes":len(correlations),
            "mean_within_scene_spearman":float(np.mean(correlations)),
            "actual_score_gap_weighted_pair_accuracy":float(weighted_correct),
            "selection_changed_fraction_from_frozen":float((chosen!=baseline_selected).mean())}
    difference=selected_scores["C_selection_error"]-selected_scores["B_boundary"]
    diagnostic["C_minus_B"]={"selected_pdms_difference":float(difference.mean()),
        "paired_recording_bootstrap_95ci":recording_bootstrap(difference,groups,73,configuration["bootstrap_repetitions"])}
    remainder=np.load(output/"training_remainder_official_metrics.npy",mmap_mode="r")
    for strategy_index,name in enumerate(names[1:]):
        acquired=remainder[:,64+32*strategy_index:64+32*(strategy_index+1)]
        families=np.load(output/"acquisition"/(name+"_family_indices.npy"),mmap_mode="r")[configuration["common_scout_scenes"]:]
        frozen_logits=np.load(output/"acquisition"/(name+"_component_logits.npy"),mmap_mode="r")[configuration["common_scout_scenes"]:]
        weights=results["cache"]["score_weights"]
        probability=1/(1+np.exp(-frozen_logits))
        neural=(weights["noc"]*np.log(probability[...,0])+weights["dac"]*np.log(probability[...,1])
            +weights["ddc"]*np.log(probability[...,5])+np.log(weights["ep"]*probability[...,2]
            +weights["ttc"]*probability[...,3]+weights["comfort"]*probability[...,4]))
        chosen=neural.argmax(1)
        oracle=acquired[:,:,6].max(1)
        selection_regret=oracle-acquired[np.arange(len(acquired)),chosen,6]
        diagnostic["later_acquisition"][name]={"scenes":len(acquired),
            "mean_candidate_actual_pdms":float(acquired[:,:,6].mean()*100),
            "candidate_collision_or_offroad_fraction":float(((acquired[:,:,0]<1)|(acquired[:,:,1]<1)).mean()),
            "helper_comfort_failure_fraction_official":float((acquired[:,:,4]<1).mean()),
            "frozen_acquired_pool_regret_pdms":float(selection_regret.mean()*100),
            "family_frequencies":{str(index):float((families==index).mean()) for index in range(8)} if name != "B_boundary" else None}
    (destination/"acquisition_and_ranking_diagnostics.json").write_text(json.dumps(diagnostic,indent=2,allow_nan=False)+"\n")
    print(json.dumps({"C_minus_B":diagnostic["C_minus_B"],"fixed_native_pool":diagnostic["fixed_native_pool"]}))


if __name__ == "__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--configuration",type=Path,required=True)
    main(parser.parse_args().configuration)
