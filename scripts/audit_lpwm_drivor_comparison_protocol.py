"""Record matched settings and remaining confounds, without training a baseline."""
import json
import math
from pathlib import Path
import sys

import torch
import timm
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"src"))
from planning_aware_future_prediction.object_centric.lpwm_drivor_joint import official_configuration
from navsim.agents.drivoR.layers.image_encoder.dinov2_lora import LoRA_ViT_timm


def main():
    torch.set_num_threads(2)
    reference = ROOT/"reference_repositories/DrivoR"
    configuration, _ = official_configuration()
    default_training = yaml.safe_load((reference/"navsim/planning/script/config/training/default_training.yaml").read_text())
    # Architecture inspection only: no download, no model training or GPU use.
    dino = timm.create_model("vit_small_patch14_reg4_dinov2", pretrained=False, img_size=(672, 1148), num_classes=0)
    depth, width = len(dino.blocks), dino.num_features
    adapted_dino = LoRA_ViT_timm(dino, r=32)
    dino_lora_parameters = sum(parameter.numel() for parameter in adapted_dino.parameters() if parameter.requires_grad)
    assert dino_lora_parameters == depth*4*width*32
    inventory = json.loads((ROOT/"results/lpwm_drivor_lora_v1/parameter_inventory.json").read_text())
    result = {
        "verdict": "Shared official planning backend and benchmark splits; not yet a controlled demonstration of particle-versus-register or intent-conditioned future information benefit.",
        "official_source_commit": "fc6e5aa144bbcb5a046e22c18f1bd5cf3af8634a",
        "matched": {"planner_forward": "Imported DrivoRModel.forward", "loss": "Imported DrivoRLoss",
            "observation": "Four current cameras F0/B0/L0/R0, one frame per camera, current ego11D",
            "planner_memory_shape": [64, 256], "proposals": configuration.proposal_num,
            "generator_layers": configuration.ref_num, "scorer_layers": configuration.scorer_ref_num,
            "trajectory_poses": configuration.num_poses, "nominal_effective_batch": 64,
            "optimizer": "AdamW, base LR2e-4, weight decay0.01, official warmup/cosine convention",
            "lora_parameterization": "Q/V rank32, scale1, pretrained weights frozen",
            "v1_training": "Official train+val membership, 25epochs, longer target+2poses",
            "v2_training": "Independent public init, official train membership, 10epochs, no longer target",
            "oracle_supervision": "Original generated-proposal scoring labels, no direct object auxiliary loss"},
        "differences": {
            "image_resolution_hw": {"drivor": [672, 1148], "lpwm": [128, 128]},
            "pretraining": {"drivor": "DINOv2 LVD-142M", "lpwm": "Public LPWM Sketchy"},
            "encoder_intent": {"drivor": "No direct ego conditioning of DINO/register extraction", "lpwm": "Command4D FiLM at particle CNN conv_in"},
            "image_augmentation": {"drivor": "ImageNet normalization and GridMask probability0.7", "lpwm": "LPWM public normalization, no GridMask"},
            "numeric_precision": {"drivor_default": default_training["trainer"]["params"]["precision"], "lpwm": "BF16 with FP32 particle encoder"},
            "distributed_partition": {"drivor_readme": "4 GPUs, batch16, accumulation1", "lpwm": "2 GPUs, batch16, accumulation2; first update used batch8 accumulation4"},
            "drop_last": {"drivor": True, "lpwm": False},
            "trainable_lora_parameters": {"drivor": dino_lora_parameters, "lpwm": inventory["trainable_lora_parameters"]},
            "compression": {"drivor": "16 learned camera registers per camera", "lpwm": "64 native particles/camera, concatenate current+8predicted states, project and fixed mean groups of4"},
            "future_computation": {"drivor": "No explicit particle future rollout", "lpwm": "8 prior steps; semantic time alignment remains unverified"},
            "backend_initialization": "Current LPWM planner constructed before the replacement backbone; a native DrivoR construction consumes RNG in a different order. Controlled runs must share an explicit backend initial state.",
        },
        "update_count_implications": {},
        "evaluation_version_requirement": "Run both checkpoints through the same sealed NAVSIM commit, scorer config, full token set and v2 human-error filtering; paper number alone is not a paired baseline.",
        "new_training_queued": False,
    }
    for benchmark, count, epochs in (("v1", 103288, 25), ("v2", 85109, 10)):
        official = math.ceil(count/4)//16
        current = math.ceil(math.ceil(count/2)/32)
        result["update_count_implications"][benchmark] = {"official_readme_assuming_standard_distributed_sampler_per_epoch": official,
            "current_per_epoch": current, "official_total": official*epochs, "current_total": current*epochs,
            "status": "Derived from repository drop_last and published4GPU recipe; not measured from an original author training log."}
    output = ROOT/"results/lpwm_drivor_representation_monitor_v1/comparison_protocol_audit.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2)+"\n")
    print(json.dumps({"dino_lora_parameters": dino_lora_parameters, "lpwm_lora_parameters": inventory["trainable_lora_parameters"], "updates": result["update_count_implications"]}), flush=True)


if __name__ == "__main__":
    main()
