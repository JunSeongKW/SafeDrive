#!/usr/bin/env python
"""Plot true-score-labelled native candidates and the frozen/corrected choices."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plotting
import numpy as np

PROJECT_ROOT=Path(__file__).resolve().parents[1]


def main(configuration_path):
    configuration=json.loads(configuration_path.read_text())
    output=PROJECT_ROOT/configuration["output_directory"]
    destination=PROJECT_ROOT/configuration["results_directory"]
    registration=json.loads((output/"registration.json").read_text())
    records=[record for record in registration["records"] if record["pilot_split"] == "locked_development"]
    predictions=np.load(destination/"fixed_pool_predictions.npz")
    native=np.load(output/"cache/native_trajectories.npy",mmap_mode="r")
    ground_truth=np.load(output/"cache/ground_truth_trajectory.npy",mmap_mode="r")
    official=predictions["official_metrics"][:,:64,6]*100
    names=("unchanged","A_uniform","B_boundary","C_selection_error")
    selection={name:predictions[name+"_native"].argmax(1) for name in names}
    values={name:official[np.arange(len(records)),selection[name]] for name in names}
    improvements=values["C_selection_error"]-values["unchanged"]
    chosen=[]
    for index in (int(improvements.argmax()),int(improvements.argmin()),int(np.argsort(np.abs(improvements))[len(records)//2])):
        if index not in chosen:
            chosen.append(index)
    figure,axes=plotting.subplots(len(chosen),3,figsize=(15,5*len(chosen)),squeeze=False)
    image_cache=np.load(PROJECT_ROOT/configuration["corpus_directory"]/"images.npy",mmap_mode="r")
    diagnostic_selection=[]
    for panel,index in enumerate(chosen):
        record=records[index]
        paths=native[record["pilot_row"]]
        axes[panel,0].imshow(image_cache[record["cache_row"],-1])
        axes[panel,0].axis("off")
        axes[panel,0].set_title(f"Observed front / command {record['command_index']}\n{record['token']}")
        for column,name in enumerate(("unchanged","C_selection_error"),1):
            axis=axes[panel,column]
            for trajectory in paths:
                axis.plot(trajectory[:,0],trajectory[:,1],color="gray",alpha=.12,lw=.7)
            actual_best=int(official[index].argmax())
            selected=int(selection[name][index])
            for candidate,label,color,style in ((actual_best,"Actual best","#2ca02c","--"),(selected,"Selected","#d62728","-")):
                trajectory=np.concatenate((np.zeros((1,3)),paths[candidate]),0)
                axis.plot(trajectory[:,0],trajectory[:,1],style,color=color,lw=2.5,label=f"{label}: {official[index,candidate]:.2f}")
            gt=np.concatenate((np.zeros((1,3)),ground_truth[record["pilot_row"]]),0)
            axis.plot(gt[:,0],gt[:,1],color="#1f77b4",lw=1.3,label="Logged GT")
            axis.scatter([0],[0],color="black",s=35)
            axis.set_title(f"{name}\nSelected PDMS {values[name][index]:.2f}")
            axis.set_xlabel("Forward x (m)")
            axis.set_ylabel("Left y (m)")
            axis.grid(alpha=.2)
            axis.legend(fontsize=8)
            axis.set_aspect("equal",adjustable="datalim")
        diagnostic_selection.append({"token":record["token"],"C_minus_frozen_pdms":float(improvements[index]),
                                     "selected_pdms":{name:float(values[name][index]) for name in names}})
    figure.suptitle("Fixed native64 candidates: identical paths, changed ranking\nExamples selected by largest gain/loss and median |change|; not representative population samples")
    figure.tight_layout(rect=(0,0,1,.95))
    figure.savefig(destination/"selected_trajectory_examples.png",dpi=150)
    plotting.close(figure)
    (destination/"visualization_selection.json").write_text(json.dumps(diagnostic_selection,indent=2)+"\n")


if __name__ == "__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--configuration",type=Path,required=True)
    main(parser.parse_args().configuration)
