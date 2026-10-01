"""Graph contracts + planning-only selection learning on a cheap synthetic task.

No shared data, GPU, NAVSIM, downloaded weights, or external test packages.
Analytic predictor/planner isolate selector learning: not full JEPA training.
"""

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time
import unittest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import torch

from planning_aware_future_prediction.models.selective_entity_future_prediction import (
    ContextConditionedEntityScorer,
    SelectiveEntityFuturePredictionGraph,
    SequentialStraightThroughEntitySelection,
    compute_parameter_gradient_norm,
)

NUM_CANDIDATE_ENTITIES, SELECTED_ENTITY_BUDGET, SYNTHETIC_PREDICTION_TIME_OFFSET = (
    6,
    2,
    2,
)


def generate_synthetic_entity_batch(batch_size, generator):
    """Semantic keys + current x,v; intent requests K distinct categories.

    Entity order is randomly permuted EVERY sample. Stable IDs are only for
    tie resolution, never inputs to the scoring network. Future is analytic.
    """
    semantic_key_features = torch.eye(NUM_CANDIDATE_ENTITIES).expand(batch_size, -1, -1)
    current_position_velocity = torch.randn(
        batch_size, NUM_CANDIDATE_ENTITIES, 2, generator=generator
    )
    requested_semantic_keys = torch.rand(
        batch_size, NUM_CANDIDATE_ENTITIES, generator=generator
    ).argsort(dim=-1)[:, :SELECTED_ENTITY_BUDGET]
    ego_intent = torch.zeros(batch_size, NUM_CANDIDATE_ENTITIES).scatter(
        1, requested_semantic_keys, 1
    )
    future_position_targets = (
        current_position_velocity[..., 0]
        + SYNTHETIC_PREDICTION_TIME_OFFSET * current_position_velocity[..., 1]
    )
    ego_plan_target = (future_position_targets * ego_intent).sum(
        dim=1
    ) / SELECTED_ENTITY_BUDGET**0.5
    entity_permutation = torch.rand(
        batch_size, NUM_CANDIDATE_ENTITIES, generator=generator
    ).argsort(dim=-1)
    entity_features = torch.cat(
        (semantic_key_features, current_position_velocity), dim=-1
    ).gather(
        1, entity_permutation[..., None].expand(-1, -1, NUM_CANDIDATE_ENTITIES + 2)
    )
    relevant_entity_mask = ego_intent.gather(1, entity_permutation).bool()
    return (
        entity_features,
        torch.zeros(batch_size, 1),
        ego_intent,
        entity_permutation,
        relevant_entity_mask,
        ego_plan_target,
    )


def predict_synthetic_ego_plan(entity_features, selection_weights):
    selected_entity_queries = selection_weights @ entity_features
    # Fixed known future dynamics; differentiable through selected selected_entity_queries.
    predicted_future_positions = (
        selected_entity_queries[..., -2]
        + SYNTHETIC_PREDICTION_TIME_OFFSET * selected_entity_queries[..., -1]
    )
    return predicted_future_positions.sum(dim=-1) / SELECTED_ENTITY_BUDGET**0.5


def compute_synthetic_selection_metrics(
    entity_features, selection_weights, relevant_entity_mask, ego_plan_target
):
    predicted_ego_plan = predict_synthetic_ego_plan(entity_features, selection_weights)
    selected_entity_mask = selection_weights.sum(dim=1).bool()
    num_relevant_selected_entities = (selected_entity_mask & relevant_entity_mask).sum(
        dim=1
    )
    return {
        "planning_mse": (predicted_ego_plan - ego_plan_target).square().mean().item(),
        "relevant_recall": (
            num_relevant_selected_entities.float() / SELECTED_ENTITY_BUDGET
        )
        .mean()
        .item(),
        "exact_set_accuracy": (num_relevant_selected_entities == SELECTED_ENTITY_BUDGET)
        .float()
        .mean()
        .item(),
    }


def run_synthetic_selection_learning(
    seed, training_steps, batch_size, holdout_sample_count
):
    start_time_seconds = time.perf_counter()
    torch.manual_seed(seed)
    entity_scorer = ContextConditionedEntityScorer(
        NUM_CANDIDATE_ENTITIES + 2, 1, NUM_CANDIDATE_ENTITIES, hidden_feature_dim=32
    )
    entity_only_scorer = ContextConditionedEntityScorer(
        NUM_CANDIDATE_ENTITIES + 2, 1, NUM_CANDIDATE_ENTITIES, hidden_feature_dim=32
    )
    entity_only_scorer.load_state_dict(entity_scorer.state_dict())
    entity_selection_operator = SequentialStraightThroughEntitySelection(
        SELECTED_ENTITY_BUDGET, temperature=0.7
    )
    context_scorer_optimizer = torch.optim.Adam(entity_scorer.parameters(), lr=0.003)
    entity_only_scorer_optimizer = torch.optim.Adam(
        entity_only_scorer.parameters(), lr=0.003
    )
    initial_batch = generate_synthetic_entity_batch(
        holdout_sample_count, torch.Generator().manual_seed(20_000 + seed)
    )
    (
        initial_entity_features,
        initial_scene_context,
        initial_ego_intent,
        stable_entity_ids,
        initial_relevant_entity_mask,
        initial_ego_plan_target,
    ) = initial_batch
    with torch.no_grad():
        initial_selection_weights = entity_selection_operator(
            entity_scorer(
                initial_entity_features, initial_scene_context, initial_ego_intent
            ),
            torch.ones(holdout_sample_count, NUM_CANDIDATE_ENTITIES, dtype=torch.bool),
            stable_entity_ids,
        ).hard_selection_weights
        initial_metrics = compute_synthetic_selection_metrics(
            initial_entity_features,
            initial_selection_weights,
            initial_relevant_entity_mask,
            initial_ego_plan_target,
        )
    training_random_generator = torch.Generator().manual_seed(10_000 + seed)
    for _ in range(training_steps):
        (
            entity_features,
            scene_context,
            ego_intent,
            stable_entity_ids,
            _,
            ego_plan_target,
        ) = generate_synthetic_entity_batch(batch_size, training_random_generator)
        entity_valid_mask = torch.ones(
            batch_size, NUM_CANDIDATE_ENTITIES, dtype=torch.bool
        )
        for policy_scorer, policy_optimizer, policy_ego_intent in (
            (entity_scorer, context_scorer_optimizer, ego_intent),
            (
                entity_only_scorer,
                entity_only_scorer_optimizer,
                torch.zeros_like(ego_intent),
            ),
        ):
            policy_optimizer.zero_grad(set_to_none=True)
            entity_selection = entity_selection_operator(
                policy_scorer(entity_features, scene_context, policy_ego_intent),
                entity_valid_mask,
                stable_entity_ids,
            )
            training_loss = (
                (
                    predict_synthetic_ego_plan(
                        entity_features, entity_selection.selection_weights
                    )
                    - ego_plan_target
                )
                .square()
                .mean()
            )
            if not torch.isfinite(training_loss):
                raise AssertionError("toy learning produced non-finite loss")
            training_loss.backward()
            policy_optimizer.step()

    holdout_random_generator = torch.Generator().manual_seed(20_000 + seed)
    (
        entity_features,
        scene_context,
        ego_intent,
        stable_entity_ids,
        relevant_entity_mask,
        ego_plan_target,
    ) = generate_synthetic_entity_batch(holdout_sample_count, holdout_random_generator)
    entity_valid_mask = torch.ones(
        holdout_sample_count, NUM_CANDIDATE_ENTITIES, dtype=torch.bool
    )
    with torch.no_grad():
        context_selection_weights = entity_selection_operator(
            entity_scorer(entity_features, scene_context, ego_intent),
            entity_valid_mask,
            stable_entity_ids,
        ).hard_selection_weights
        entity_only_selection_weights = entity_selection_operator(
            entity_only_scorer(
                entity_features, scene_context, torch.zeros_like(ego_intent)
            ),
            entity_valid_mask,
            stable_entity_ids,
        ).hard_selection_weights
        shuffled_intent_selection_weights = entity_selection_operator(
            entity_scorer(entity_features, scene_context, ego_intent.roll(1, 0)),
            entity_valid_mask,
            stable_entity_ids,
        ).hard_selection_weights
        random_selection_weights = entity_selection_operator(
            torch.rand(
                holdout_sample_count,
                NUM_CANDIDATE_ENTITIES,
                generator=holdout_random_generator,
            ),
            entity_valid_mask,
            stable_entity_ids,
        ).hard_selection_weights
        motion_selection_weights = entity_selection_operator(
            entity_features[..., -1].abs(), entity_valid_mask, stable_entity_ids
        ).hard_selection_weights
        fixed_semantic_selection_weights = entity_selection_operator(
            -stable_entity_ids.float(), entity_valid_mask, stable_entity_ids
        ).hard_selection_weights
        oracle_selection_weights = entity_selection_operator(
            relevant_entity_mask.float(), entity_valid_mask, stable_entity_ids
        ).hard_selection_weights
        policy_metrics = {
            name: compute_synthetic_selection_metrics(
                entity_features,
                selection_weights,
                relevant_entity_mask,
                ego_plan_target,
            )
            for name, selection_weights in (
                ("context_conditioned_selection", context_selection_weights),
                ("entity_only_selection_without_intent", entity_only_selection_weights),
                ("selection_with_shuffled_intent", shuffled_intent_selection_weights),
                ("random_entity_selection", random_selection_weights),
                ("motion_based_selection", motion_selection_weights),
                ("fixed_semantic_key_selection", fixed_semantic_selection_weights),
                ("hindsight_relevant_entity_selection", oracle_selection_weights),
            )
        }
        entity_permutation = torch.arange(NUM_CANDIDATE_ENTITIES - 1, -1, -1)
        permuted_selection_weights = entity_selection_operator(
            entity_scorer(
                entity_features[:, entity_permutation], scene_context, ego_intent
            ),
            entity_valid_mask,
            stable_entity_ids[:, entity_permutation],
        ).hard_selection_weights
        torch.testing.assert_close(
            predict_synthetic_ego_plan(entity_features, context_selection_weights),
            predict_synthetic_ego_plan(
                entity_features[:, entity_permutation], permuted_selection_weights
            ),
            rtol=1e-5,
            atol=1e-6,
        )

    # Declared before measuring: engineering diagnostic, not H1/H2 evidence.
    selection_learning_passed = (
        policy_metrics["context_conditioned_selection"]["relevant_recall"] >= 0.8
        and policy_metrics["context_conditioned_selection"]["planning_mse"]
        <= 0.25 * policy_metrics["random_entity_selection"]["planning_mse"]
    )
    return {
        "seed": seed,
        "elapsed_seconds": time.perf_counter() - start_time_seconds,
        "initial_context_selector": initial_metrics,
        "selection_learning_passed": selection_learning_passed,
        "policies": policy_metrics,
    }


def measure_training_signal_gradient_norms():
    torch.manual_seed(19)
    model = SelectiveEntityFuturePredictionGraph().double().eval()
    entity_features, scene_context, ego_intent = (
        torch.randn(*shape, dtype=torch.double) for shape in ((3, 4, 6), (3, 4), (3, 3))
    )
    entity_valid_mask = torch.ones(3, 4, dtype=torch.bool)
    stable_entity_ids = torch.tensor([[30, 10, 20, 40]]).expand(3, -1)
    future_latent_targets = torch.randn(3, 4, 2, 5, dtype=torch.double)
    future_target_valid_mask = torch.ones(3, 4, 2, dtype=torch.bool)
    ego_plan_target = torch.randn(3, 3, dtype=torch.double)
    report = {}
    for training_signal, selection_gradient_mode, detach_future_latents, loss_name in (
        ("planning", "straight_through", False, "planning_loss"),
        ("auxiliary", "straight_through", False, "future_latent_prediction_loss"),
        ("hard_indices", "hard_indices", False, "planning_loss"),
        ("selection_detach", "detached_selection", False, "planning_loss"),
        ("future_detach", "straight_through", True, "planning_loss"),
    ):
        prediction_output = model(
            entity_features,
            scene_context,
            ego_intent,
            entity_valid_mask,
            stable_entity_ids,
            selection_gradient_mode,
            detach_future_latents,
        )
        training_loss = model.compute_training_losses(
            prediction_output,
            entity_features,
            scene_context,
            ego_intent,
            entity_valid_mask,
            future_latent_targets,
            future_target_valid_mask,
            ego_plan_target,
        )[loss_name]
        report[training_signal] = {
            name: compute_parameter_gradient_norm(training_loss, model_component)
            for name, model_component in (
                ("selector", model.entity_scorer),
                ("predictor", model.future_predictor),
                ("planner", model.ego_planner),
            )
        }
    return report


def main():
    argument_parser = argparse.ArgumentParser()
    argument_parser.add_argument("--training-steps", type=int, default=1000)
    argument_parser.add_argument("--batch-size", type=int, default=128)
    argument_parser.add_argument(
        "--holdout-samples", dest="holdout_sample_count", type=int, default=4096
    )
    argument_parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    argument_parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/synthetic_diagnostics/future_prediction_validation.json"),
    )
    arguments = argument_parser.parse_args()
    if (
        min(
            arguments.training_steps,
            arguments.batch_size,
            arguments.holdout_sample_count,
        )
        < 1
    ):
        argument_parser.error("steps and sample counts must be positive")
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    start_time_seconds = time.perf_counter()
    test_suite = unittest.defaultTestLoader.discover(
        str(PROJECT_ROOT / "tests"), pattern="test_future_prediction_graph.py"
    )
    test_result = unittest.TextTestRunner(verbosity=2).run(test_suite)
    selection_learning_runs = []
    if test_result.wasSuccessful():
        for seed in arguments.seeds:
            seed_result = run_synthetic_selection_learning(
                seed,
                arguments.training_steps,
                arguments.batch_size,
                arguments.holdout_sample_count,
            )
            selection_learning_runs.append(seed_result)
            print(
                json.dumps(
                    {
                        "seed": seed,
                        **seed_result["policies"]["context_conditioned_selection"],
                        "passed": seed_result["selection_learning_passed"],
                    }
                ),
                flush=True,
            )
    project_root = PROJECT_ROOT
    reference_commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=project_root, text=True
    ).strip()
    source_paths = (
        "src/planning_aware_future_prediction/models/selective_entity_future_prediction.py",
        "tests/test_future_prediction_graph.py",
        "scripts/validate_future_prediction_graph.py",
    )
    source_sha256 = {
        relative_path: hashlib.sha256(
            (project_root / relative_path).read_bytes()
        ).hexdigest()
        for relative_path in source_paths
    }
    report = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "reference_commit": reference_commit,
        "source_sha256": source_sha256,
        "python": sys.version,
        "python_executable": sys.executable,
        "torch": torch.__version__,
        "torch_source": torch.__file__,
        "device": "cpu",
        "threads": 1,
        "deterministic_algorithms": True,
        "config": {
            **vars(arguments),
            "output": str(arguments.output),
            "num_candidate_entities": NUM_CANDIDATE_ENTITIES,
            "selected_entity_budget": SELECTED_ENTITY_BUDGET,
            "synthetic_prediction_time_offset": SYNTHETIC_PREDICTION_TIME_OFFSET,
            "selection_temperature": 0.7,
            "learning_rate": 0.003,
        },
        "tests_run": test_result.testsRun,
        "tests_passed": test_result.wasSuccessful(),
        "gradient_norms": (
            measure_training_signal_gradient_norms()
            if test_result.wasSuccessful()
            else {}
        ),
        "failures": [
            str(test) + "\n" + trace
            for test, trace in test_result.failures + test_result.errors
        ],
        "selection_learning": selection_learning_runs,
        "elapsed_seconds": time.perf_counter() - start_time_seconds,
        "claim_scope": "Synthetic graph + analytic-dynamics selector learnability ONLY; no NAVSIM/JEPA performance claim.",
    }
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(f"REPORT: {arguments.output}", flush=True)
    return (
        0
        if test_result.wasSuccessful()
        and selection_learning_runs
        and all(
            seed_run["selection_learning_passed"]
            for seed_run in selection_learning_runs
        )
        else 1
    )


if __name__ == "__main__":
    raise SystemExit(main())
