"""No-training audit of learned bias scale and ego sensitivity on cached tokens."""
import argparse
import math
from pathlib import Path
import time

import numpy as np
import torch
from torch.nn import functional as functional

from run_soft_token_reweighting import (
    PROJECT_ROOT, EncoderFeatureCache, gpu_setup, model_from_initial,
    prediction_from_minibatch, read_json, setup_official, write_json,
)


def run(arguments):
    configuration = read_json(arguments.config)
    output = PROJECT_ROOT / configuration["output_directory"]
    result_directory = PROJECT_ROOT / configuration["results_directory"]
    setup_official()
    gpu_setup(configuration)
    started = time.perf_counter()
    cache = EncoderFeatureCache(output)
    indices = cache.indices("validation")
    model = model_from_initial(output, "conditioned", "cuda").eval()
    model.load_state_dict(torch.load(output / "conditioned/checkpoint.pt", map_location="cpu")["model"], strict=True)
    permutation = np.random.default_rng(0).permutation(indices)
    content_logit_standard_deviations = []
    importance_standard_deviations = []
    bias_standard_deviations = []
    ego_counterfactual_differences = []
    dynamic_counterfactual_differences = []

    def capture_content_logits(attention, inputs, keyword_arguments):
        queries, keys = inputs[:2]
        dimension = attention.embed_dim
        projected_queries = functional.linear(queries, attention.in_proj_weight[:dimension], attention.in_proj_bias[:dimension])
        projected_keys = functional.linear(keys, attention.in_proj_weight[dimension:2*dimension], attention.in_proj_bias[dimension:2*dimension])
        projected_queries = projected_queries.reshape(queries.shape[0], queries.shape[1], attention.num_heads, -1).transpose(1,2)
        projected_keys = projected_keys.reshape(keys.shape[0], keys.shape[1], attention.num_heads, -1).transpose(1,2)
        logits = projected_queries @ projected_keys.transpose(-1,-2) / math.sqrt(dimension/attention.num_heads)
        content_logit_standard_deviations.extend(logits[..., :128].std(-1, unbiased=False).mean((1,2)).cpu().tolist())

    handle = model.planner["_transformer"].decoder.layers[0].multihead_attn.register_forward_pre_hook(
        capture_content_logits, with_kwargs=True)
    with torch.no_grad():
        for offset in range(0, len(indices), 32):
            observations = cache.minibatch(indices[offset:offset+32])
            prediction_from_minibatch(model, observations)
            original_importance = model.last_importance
            alternate_ego = cache.minibatch(permutation[offset:offset+32])["ego_status"]
            alternate_dynamics = observations["ego_status"].clone()
            alternate_dynamics[:, 4:] = alternate_ego[:, 4:]
            changed_ego = model.importance(observations["visual_tokens"], observations["valid_token_mask"], alternate_ego, model.positions)
            changed_dynamics = model.importance(observations["visual_tokens"], observations["valid_token_mask"], alternate_dynamics, model.positions)
            importance_standard_deviations.extend(original_importance.std(-1, unbiased=False).cpu().tolist())
            bias_standard_deviations.extend((model.importance.beta*original_importance).std(-1, unbiased=False).cpu().tolist())
            ego_counterfactual_differences.extend((changed_ego-original_importance).abs().mean(-1).cpu().tolist())
            dynamic_counterfactual_differences.extend((changed_dynamics-original_importance).abs().mean(-1).cpu().tolist())
    handle.remove()
    entropy_precision_audit = {}
    for condition in ("unconditioned", "conditioned"):
        stored = np.load(output / condition / "token_diagnostics.npz")
        beta = read_json(output / condition / "training_complete.json")["beta"]
        logits = beta * stored["importance"].astype(np.float64)
        logits -= logits.max(-1, keepdims=True)
        probabilities = np.exp(logits); probabilities /= probabilities.sum(-1, keepdims=True)
        entropy = -(probabilities*np.log(probabilities)).sum(-1)/np.log(128)
        entropy_precision_audit[condition] = {
            "normalized_entropy_float64_mean": float(entropy.mean()),
            "mean_probability_coefficient_of_variation": float((probabilities.std(-1)/probabilities.mean(-1)).mean()),
            "maximum_probability_divided_by_uniform": float((probabilities*128).max()),
            "note": "Entropy recomputed in FP64; higher entropy means a more uniform importance-only distribution"}
    native_predictions = np.load(output / "conditioned/predictions_normal.npz")["trajectories"]
    interventions = {}
    for mode in ("beta_zero", "shuffle"):
        alternative = np.load(output / "conditioned" / ("predictions_"+mode+".npz"))["trajectories"]
        distances = np.linalg.norm(native_predictions[..., :2]-alternative[..., :2], axis=-1)
        interventions[mode] = {"mean_waypoint_change_meters": float(distances.mean()),
            "maximum_waypoint_change_meters": float(distances.max()),
            "mean_endpoint_change_meters": float(distances[:, -1].mean())}
    importance_scale = np.maximum(importance_standard_deviations, 1e-12)
    bias_ratio = np.asarray(bias_standard_deviations)/np.maximum(content_logit_standard_deviations,1e-12)
    result = {"no_additional_training": True, "scenes": len(indices),
        "mean_content_qk_logit_std": float(np.mean(content_logit_standard_deviations)),
        "mean_learned_bias_std": float(np.mean(bias_standard_deviations)),
        "mean_bias_to_content_std_ratio": float(bias_ratio.mean()),
        "ego_permutation_mean_absolute_importance_change": float(np.mean(ego_counterfactual_differences)),
        "ego_permutation_mean_change_relative_to_importance_std": float((ego_counterfactual_differences/importance_scale).mean()),
        "dynamic_only_permutation_mean_change_relative_to_importance_std": float((dynamic_counterfactual_differences/importance_scale).mean()),
        "counterfactual_scope": "Only importance inputs are changed; scene tokens and native decoder ego are held fixed. Permuted ego can be out-of-distribution.",
        "entropy_precision_audit": entropy_precision_audit,
        "trajectory_sensitivity": interventions, "seconds": time.perf_counter()-started}
    write_json(result_directory / "bias_scale_and_ego_sensitivity.json", result)
    print(result, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs/soft_token_reweighting/pilot_v1.json")
    run(parser.parse_args())
