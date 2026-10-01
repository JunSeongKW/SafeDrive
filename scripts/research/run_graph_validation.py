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

import torch

from selective_future_graph import ContextSelector, SelectiveFutureGraph, SequentialSTTopK, gradient_norm


N, K, T = 6, 2, 2


def batch(size, generator):
    """Semantic keys + current x,v; intent requests K distinct categories.

    Entity order is randomly permuted EVERY sample. Stable IDs are only for
    tie resolution, never inputs to the scoring network. Future is analytic.
    """
    keys = torch.eye(N).expand(size, -1, -1)
    state = torch.randn(size, N, 2, generator=generator)
    requested = torch.rand(size, N, generator=generator).argsort(dim=-1)[:, :K]
    intent = torch.zeros(size, N).scatter(1, requested, 1)
    future = state[..., 0] + T * state[..., 1]
    ego_target = (future * intent).sum(dim=1) / K ** 0.5
    order = torch.rand(size, N, generator=generator).argsort(dim=-1)
    entities = torch.cat((keys, state), dim=-1).gather(1, order[..., None].expand(-1, -1, N + 2))
    relevant = intent.gather(1, order).bool()
    return entities, torch.zeros(size, 1), intent, order, relevant, ego_target


def plan_from_selection(entities, weights):
    queries = weights @ entities
    # Fixed known future dynamics; differentiable through selected queries.
    future = queries[..., -2] + T * queries[..., -1]
    return future.sum(dim=-1) / K ** 0.5


def measure(entities, weights, relevant, target):
    prediction = plan_from_selection(entities, weights)
    chosen = weights.sum(dim=1).bool()
    overlap = (chosen & relevant).sum(dim=1)
    return {"planning_mse": (prediction - target).square().mean().item(),
            "relevant_recall": (overlap.float() / K).mean().item(),
            "exact_set_accuracy": (overlap == K).float().mean().item()}


def toy(seed, steps, batch_size, holdout_size):
    started = time.perf_counter()
    torch.manual_seed(seed)
    selector = ContextSelector(N + 2, 1, N, hidden=32)
    global_selector = ContextSelector(N + 2, 1, N, hidden=32)
    global_selector.load_state_dict(selector.state_dict())
    route = SequentialSTTopK(K, temperature=0.7)
    optimizer = torch.optim.Adam(selector.parameters(), lr=0.003)
    global_optimizer = torch.optim.Adam(global_selector.parameters(), lr=0.003)
    initial_batch = batch(holdout_size, torch.Generator().manual_seed(20_000 + seed))
    ih, ic, iu, ids, ir, it = initial_batch
    with torch.no_grad():
        initial = route(selector(ih, ic, iu), torch.ones(holdout_size, N, dtype=torch.bool), ids).hard_weights
        initial_metrics = measure(ih, initial, ir, it)
    train_rng = torch.Generator().manual_seed(10_000 + seed)
    for _ in range(steps):
        entities, context, intent, ids, _, target = batch(batch_size, train_rng)
        valid = torch.ones(batch_size, N, dtype=torch.bool)
        for policy, optim, command in ((selector, optimizer, intent),
                                       (global_selector, global_optimizer, torch.zeros_like(intent))):
            optim.zero_grad(set_to_none=True)
            selection = route(policy(entities, context, command), valid, ids)
            loss = (plan_from_selection(entities, selection.weights) - target).square().mean()
            if not torch.isfinite(loss):
                raise AssertionError("toy learning produced non-finite loss")
            loss.backward()
            optim.step()

    test_rng = torch.Generator().manual_seed(20_000 + seed)
    entities, context, intent, ids, relevant, target = batch(holdout_size, test_rng)
    valid = torch.ones(holdout_size, N, dtype=torch.bool)
    with torch.no_grad():
        learned = route(selector(entities, context, intent), valid, ids).hard_weights
        global_weights = route(global_selector(entities, context, torch.zeros_like(intent)), valid, ids).hard_weights
        shuffled = route(selector(entities, context, intent.roll(1, 0)), valid, ids).hard_weights
        random_weights = route(torch.rand(holdout_size, N, generator=test_rng), valid, ids).hard_weights
        motion = route(entities[..., -1].abs(), valid, ids).hard_weights
        fixed = route(-ids.float(), valid, ids).hard_weights
        oracle = route(relevant.float(), valid, ids).hard_weights
        results = {name: measure(entities, weights, relevant, target) for name, weights in
                   (("learned_context", learned), ("global_learned_no_intent", global_weights),
                    ("shuffled_intent", shuffled), ("random", random_weights),
                    ("motion", motion), ("fixed_semantic", fixed), ("hindsight_oracle", oracle))}
        perm = torch.arange(N - 1, -1, -1)
        reordered = route(selector(entities[:, perm], context, intent), valid, ids[:, perm]).hard_weights
        torch.testing.assert_close(plan_from_selection(entities, learned),
                                   plan_from_selection(entities[:, perm], reordered), rtol=1e-5, atol=1e-6)

    # Declared before measuring: engineering diagnostic, not H1/H2 evidence.
    passed = (results["learned_context"]["relevant_recall"] >= 0.8 and
              results["learned_context"]["planning_mse"] <= 0.25 * results["random"]["planning_mse"])
    return {"seed": seed, "elapsed_seconds": time.perf_counter() - started,
            "initial_context_selector": initial_metrics,
            "selection_learning_passed": passed, "policies": results}


def audit_gradients():
    torch.manual_seed(19)
    model = SelectiveFutureGraph().double().eval()
    h, c, u = (torch.randn(*shape, dtype=torch.double) for shape in ((3, 4, 6), (3, 4), (3, 3)))
    valid = torch.ones(3, 4, dtype=torch.bool)
    ids = torch.tensor([[30, 10, 20, 40]]).expand(3, -1)
    target = torch.randn(3, 4, 2, 5, dtype=torch.double)
    future_valid = torch.ones(3, 4, 2, dtype=torch.bool)
    ego = torch.randn(3, 3, dtype=torch.double)
    report = {}
    for mode, selection_mode, detach_future, key in (("planning", "st", False, "plan"),
            ("auxiliary", "st", False, "jepa"), ("hard_indices", "hard", False, "plan"),
            ("selection_detach", "detach", False, "plan"), ("future_detach", "st", True, "plan")):
        output = model(h, c, u, valid, ids, selection_mode, detach_future)
        loss = model.losses(output, h, c, u, valid, target, future_valid, ego)[key]
        report[mode] = {name: gradient_norm(loss, module) for name, module in
                       (("selector", model.selector), ("predictor", model.predictor), ("planner", model.planner))}
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--holdout-size", type=int, default=4096)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--output", type=Path, default=Path("exp/research/graph_v1/results.json"))
    args = parser.parse_args()
    if min(args.steps, args.batch_size, args.holdout_size) < 1:
        parser.error("steps and sample counts must be positive")
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    started = time.perf_counter()
    suite = unittest.defaultTestLoader.discover(str(Path(__file__).parent), pattern="test_selective_future_graph.py")
    tests = unittest.TextTestRunner(verbosity=2).run(suite)
    runs = []
    if tests.wasSuccessful():
        for seed in args.seeds:
            result = toy(seed, args.steps, args.batch_size, args.holdout_size)
            runs.append(result)
            print(json.dumps({"seed": seed, **result["policies"]["learned_context"],
                              "passed": result["selection_learning_passed"]}), flush=True)
    root = Path(__file__).resolve().parents[2]
    baseline = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    sources = {name: hashlib.sha256((Path(__file__).parent / name).read_bytes()).hexdigest()
               for name in ("selective_future_graph.py", "test_selective_future_graph.py", "run_graph_validation.py")}
    report = {"timestamp_utc": datetime.now(timezone.utc).isoformat(), "baseline_commit": baseline,
              "source_sha256": sources, "python": sys.version, "python_executable": sys.executable,
              "torch": torch.__version__, "torch_source": torch.__file__, "device": "cpu", "threads": 1,
              "deterministic_algorithms": True, "config": {**vars(args), "output": str(args.output),
              "n": N, "k": K, "horizon": T, "temperature": 0.7, "lr": 0.003},
              "tests_run": tests.testsRun, "tests_passed": tests.wasSuccessful(),
              "gradient_norms": audit_gradients() if tests.wasSuccessful() else {},
              "failures": [str(test) + "\n" + trace for test, trace in tests.failures + tests.errors],
              "selection_learning": runs, "elapsed_seconds": time.perf_counter() - started,
              "claim_scope": "Synthetic graph + analytic-dynamics selector learnability ONLY; no NAVSIM/JEPA performance claim."}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(f"REPORT: {args.output}", flush=True)
    return 0 if tests.wasSuccessful() and runs and all(r["selection_learning_passed"] for r in runs) else 1


if __name__ == "__main__":
    raise SystemExit(main())
