"""Real-image, real-oracle training audit; its updates never enter main training."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
from planning_aware_future_prediction.object_centric.lpwm_drivor_joint import (
    make_official_loss, oracle_loss_callback)
from planning_aware_future_prediction.object_centric.lpwm_drivor_lora import LPWMDrivoRLoRAModel
from lpwm_drivor_oracle import DrivoROracleClient


def card_memory():
    return int(subprocess.check_output(["nvidia-smi", "--id=" + os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")[0],
        "--query-gpu=memory.used", "--format=csv,noheader,nounits"], text=True).strip()) * 1024**2


def run(arguments):
    torch.set_num_threads(2)
    torch.manual_seed(2)
    torch.cuda.set_device(0)
    output = arguments.output
    output.mkdir(parents=True, exist_ok=True)
    starting_used = card_memory()
    allowance = 48_000_000_000 - starting_used - 768 * 1024**2
    torch.cuda.set_per_process_memory_fraction(allowance / torch.cuda.get_device_properties(0).total_memory)
    records = json.loads(arguments.manifest.read_text())["records"]
    selected = [records[index % len(records)] for index in range(arguments.microbatch)]
    def array(name):
        return np.stack([np.load(Path(record["cache_directory"]) / name, mmap_mode="r")[record["cache_row"]] for record in selected])
    features = {"image": torch.from_numpy(array("images.npy")).cuda().permute(0,1,4,2,3).float()/255,
                "ego_status": torch.from_numpy(array("ego.npy")).cuda()[:, None]}
    targets = {"trajectory": torch.from_numpy(array("trajectory.npy")).cuda(),
               "trajectory_long": torch.from_numpy(array("trajectory_long.npy")).cuda()}
    checkpoint = next((PROJECT_ROOT / "outputs/lpwm_navsim_adaptation_v1/pretrained").glob("*.pth"))
    model = LPWMDrivoRLoRAModel(checkpoint).cuda().train()
    initial_native_digest = model.frozen_native_digest()
    (output / "parameter_inventory.json").write_text(json.dumps(model.parameter_inventory(), indent=2))
    original_planner = torch.load(PROJECT_ROOT / "outputs/lpwm_drivor_joint_v1/navsim_v1/initial_official_planner.pt", map_location="cpu", weights_only=True)
    assert all(torch.equal(tensor.cpu(), original_planner[name]) for name,tensor in model.planner.state_dict().items() if not name.startswith("image_backbone."))
    model.eval()
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        initial_prediction = model(features)["proposals"].float().cpu()
        for adapters in model._adapted_regions.values():
            for adapter in adapters:
                adapter.forward = adapter.pretrained_linear.forward
        native_prediction = model(features)["proposals"].float().cpu()
        for adapters in model._adapted_regions.values():
            for adapter in adapters:
                del adapter.forward
    assert torch.equal(initial_prediction, native_prediction), "Zero LoRA must preserve initial outputs"
    model.train()
    criterion, configuration = make_official_loss()
    optimizer = torch.optim.AdamW(model.optimizer_groups(2e-4, 2e-4))
    snapshots = {name: [parameter.detach().cpu().clone() for parameter in parameters]
                 for name, parameters in model.gradient_groups().items()}
    oracle = DrivoROracleClient(arguments.manifest, output, workers=2)
    history = []
    try:
        for update in range(arguments.updates):
            started = time.time()
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                predicted = model(features)
            assert predicted["proposals"].shape == (arguments.microbatch,64,8,3)
            scores = torch.from_numpy(oracle.score([record["token"] for record in selected],
                predicted["proposals"].detach().float().cpu().numpy())).cuda()
            # Preserve official mixed-precision target conversion. Casting the
            # logits to FP32 first aliases target views that official loss edits.
            with torch.autocast("cuda", dtype=torch.bfloat16):
                losses = criterion(targets, predicted, configuration, scoring_function=oracle_loss_callback(scores))
            if arguments.gradient_routes and update == 0:
                coordinate_grad = torch.autograd.grad(losses["final_score_loss"], predicted["proposals"],
                    retain_graph=True, allow_unused=True)[0]
                assert coordinate_grad is None, "Official scorer must stop at proposal coordinates"
                routes = {}
                for loss_name in ("trajectory_loss", "final_score_loss"):
                    optimizer.zero_grad(set_to_none=True)
                    losses[loss_name].backward(retain_graph=True)
                    routes[loss_name] = {name: float(sum(parameter.grad.detach().float().square().sum().item()
                        for parameter in parameters if parameter.grad is not None)**.5)
                        for name, parameters in model.gradient_groups().items()}
                (output / "gradient_routes.json").write_text(json.dumps(routes, indent=2))
                optimizer.zero_grad(set_to_none=True)
            losses["loss"].backward()
            norms = {name: float(sum(parameter.grad.detach().float().square().sum().item()
                for parameter in parameters if parameter.grad is not None)**.5)
                for name, parameters in model.gradient_groups().items()}
            assert all(np.isfinite(value) and value > 0 for value in norms.values()), norms
            assert all(torch.isfinite(parameter.grad).all() for parameter in model.parameters() if parameter.grad is not None)
            model.assert_native_frozen()
            optimizer.step()
            torch.cuda.synchronize()
            memory = card_memory()
            assert memory <= 48_000_000_000, memory
            row = {"update": update+1, "loss": losses["loss"].item(), "seconds": time.time()-started,
                "card_used_bytes": memory, "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                "peak_reserved_bytes": torch.cuda.max_memory_reserved(), "gradient_norms": norms}
            history.append(row)
            print(json.dumps(row), flush=True)
        deltas = {name: float(sum((parameter.detach().cpu()-before).double().square().sum().item()
            for parameter, before in zip(parameters, snapshots[name]))**.5)
            for name, parameters in model.gradient_groups().items()}
        assert all(np.isfinite(value) and value > 0 for value in deltas.values()), deltas
        assert model.frozen_native_digest() == initial_native_digest, "Native weights or buffers changed"
        (output / "passed.json").write_text(json.dumps({"passed": True, "native_unchanged": True, "initial_zero_lora_outputs_identical": True, "initial_planner_identical": True, "frozen_native_sha256": initial_native_digest, "microbatch": arguments.microbatch,
            "gradient_routes_checked": arguments.gradient_routes, "updates_discarded": True,
            "history": history, "parameter_change_l2": deltas}, indent=2))
    finally:
        oracle.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=PROJECT_ROOT / "outputs/lpwm_drivor_joint_v1/engineering_scene_cache/manifest.json")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--microbatch", type=int, default=8)
    parser.add_argument("--updates", type=int, default=2)
    parser.add_argument("--gradient-routes", action="store_true")
    run(parser.parse_args())
