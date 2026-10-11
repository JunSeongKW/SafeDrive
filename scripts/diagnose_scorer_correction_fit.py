#!/usr/bin/env python
"""Compare fitted-set and held-out choices without any new optimizer steps."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch

PROJECT_ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(PROJECT_ROOT/"src"))
from run_scorer_selection_pilot import load_feature_cache,numpy_log_scores,predict_cached
from planning_aware_future_prediction.models.scorer_selection_learning import CachedScorerCorrection


def main(configuration_path):
    configuration=json.loads(configuration_path.read_text())
    torch.set_num_threads(configuration["cpu_threads"])
    output=PROJECT_ROOT/configuration["output_directory"]
    destination=PROJECT_ROOT/configuration["results_directory"]
    cache=load_feature_cache(output)
    result=json.loads((destination/"results.json").read_text())
    count=configuration["training_scenes"]
    rows=list(range(count))
    scout=np.load(output/"scout_official_metrics.npy")
    remaining=np.load(output/"training_remainder_official_metrics.npy",mmap_mode="r")
    labels=np.concatenate((scout[:,:64],remaining[:,:64]),0)
    weights=result["cache"]["score_weights"]
    baseline=numpy_log_scores(cache["native_component_logits"][:count],weights).argmax(1)
    baseline_score=labels[np.arange(count),baseline,6]*100
    summary={"additional_optimizer_steps":0,"training_scenes":count,"baseline_training_pdms":float(baseline_score.mean()),"strategies":{}}
    for name in ("A_uniform","B_boundary","C_selection_error"):
        chosen_step=result["costs"][name]["training"]["best_tuning_step"]
        checkpoint=torch.load(output/(name+"_step"+str(chosen_step)+".pt"),map_location="cpu",weights_only=False)
        model=CachedScorerCorrection(configuration["hidden_dimension"],configuration["maximum_logit_correction"])
        model.load_state_dict(checkpoint["model"],strict=True)
        model.eval()
        chosen=predict_cached(model,cache,rows,"native",weights).argmax(1)
        value=labels[np.arange(count),chosen,6]*100
        summary["strategies"][name]={"training_native_selected_pdms":float(value.mean()),
            "training_native_difference_from_frozen":float((value-baseline_score).mean()),
            "locked_development_difference_from_frozen":result["summaries"][name]["native"]["difference_from_unchanged"],
            "interpretation":"Descriptive in-sample/out-of-sample comparison; no causal attribution to overfitting alone"}
    (destination/"training_vs_development.json").write_text(json.dumps(summary,indent=2)+"\n")
    print(json.dumps(summary))


if __name__ == "__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--configuration",type=Path,required=True)
    main(parser.parse_args().configuration)
