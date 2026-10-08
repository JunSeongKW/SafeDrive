"""Compare native and checkpointed joint loss/gradients on identical real inputs."""
import argparse
import gc
import json
import os
from pathlib import Path
import random
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

import train_small_corpus_common_planner as training
from train_small_corpus_joint_checkpointed import install_joint_checkpointing


def compare_gradients(before, after):
    assert before.keys() == after.keys()
    squared_difference = squared_baseline = 0.
    maximum_absolute = 0.
    for name in before:
        difference = after[name].double() - before[name].double()
        squared_difference += float(difference.square().sum())
        squared_baseline += float(before[name].double().square().sum())
        maximum_absolute = max(maximum_absolute, float(difference.abs().max()))
    return dict(tensors=len(before), relative_l2=(squared_difference / max(squared_baseline, 1e-30)) ** .5,
                maximum_absolute=maximum_absolute)


def main(arguments):
    output = arguments.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    assert not (output / "complete.json").exists()
    assert os.environ["CUDA_VISIBLE_DEVICES"] == "0,1"
    torch.set_num_threads(2)
    torch.cuda.set_device(0)
    torch.cuda.set_per_process_memory_fraction(27_000_000_000 / torch.cuda.get_device_properties(0).total_memory)
    torch.manual_seed(47)
    np.random.seed(47)
    random.seed(47)
    manifest = json.loads(training.CORPUS.read_text())
    records = [row for row in manifest["records"] if row["study_split"] == "train"]
    random.Random(4800).shuffle(records)
    examples = next(iter(DataLoader(training.OfficialSceneDataset(records[::2][:2]), batch_size=2)))
    features, targets = training.device_inputs(examples, 0, 2, "cuda", "navsim_v1")
    clips, _ = training.prepare_records(training.ROOT / "outputs/lpwm_driving_video_512x256_v1/corpus/local_openscene_episodes.jsonl", 32)
    random.Random(4700).shuffle(clips)
    ssl_video = next(iter(DataLoader(training.DrivingClips(clips[::2][:2]), batch_size=2))).cuda().float() / 255
    model = training.CommonPlannerModel("lpwm_joint").cuda().train()
    oracle = training.DrivoROracleClient(training.CORPUS, output, 0, 4)
    objective = training.PlanningObjective(model, oracle, True)
    state_before = training.random_state()
    buffers_before = {name: tensor.detach().clone() for name, tensor in model.named_buffers()}
    gradients, reports = {}, {}
    try:
        for mode in ("native", "checkpointed"):
            model.zero_grad(set_to_none=True)
            with torch.no_grad():
                for name, tensor in model.named_buffers():
                    tensor.copy_(buffers_before[name])
            if mode == "checkpointed":
                install_joint_checkpointing(model, objective.reconstruction)
            training.restore_random_state(state_before)
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
            started = time.time()
            loss = objective(features, targets, examples["token"], ssl_video)
            loss.backward()
            torch.cuda.synchronize()
            gradients[mode] = {name: parameter.grad.detach().cpu().clone() for name, parameter in model.named_parameters()
                               if parameter.grad is not None}
            reports[mode] = dict(objective=float(loss.detach()), losses=objective.last.copy(),
                gradient_groups=training.gradient_groups(model), seconds=time.time() - started,
                peak_allocated_bytes=torch.cuda.max_memory_allocated(), whole_card_bytes=training.card_memory(0))
            del loss
            training.write_json(output / (mode + ".json"), reports[mode])
        differences = compare_gradients(gradients["native"], gradients["checkpointed"])
        loss_relative = abs(reports["native"]["objective"] - reports["checkpointed"]["objective"]) / abs(reports["native"]["objective"])
        checks = dict(loss_preserved=loss_relative < 1e-4,
                      gradients_preserved=differences["relative_l2"] < 1e-3,
                      memory_reduced=reports["checkpointed"]["peak_allocated_bytes"] < reports["native"]["peak_allocated_bytes"] * .8,
                      all_groups_active=all(np.isfinite(value) and value > 0 for report in reports.values() for value in report["gradient_groups"].values()))
        training.write_json(output / "complete.json", dict(passed=all(checks.values()), checks=checks, reports=reports,
            gradient_difference=differences, loss_relative_difference=loss_relative,
            scope="Same real two-scene/two-clip microbatch, initial parameters and RNG; no training weights reused; final PDMS equivalence unproven"))
    finally:
        oracle.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    main(parser.parse_args())
