"""Resume with a separately recorded physical batch and effective batch16.

The original scientific registration stays immutable. Disposable timing runs
use copies of the saved state and never feed their weights into the study.
"""
import argparse
import ast
import hashlib
import json
import os
from pathlib import Path
import time

from resume_small_corpus_user_memory_limit import replace_card_guard


def compile_execution_main(training, execution_microbatch, stop_after_update=0):
    assert execution_microbatch in (2, 4, 8)
    parsed = ast.parse(Path(training.__file__).read_text())
    definition = next(node for node in parsed.body if isinstance(node, ast.FunctionDef) and node.name == "main")
    changed = dict(micro=0, registration=0, loop=0, ssl_probe=0)
    for node in ast.walk(definition):
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            if node.targets[0].id == "micro":
                assert ast.unparse(node.value) == "arguments.micro_batch"
                node.value = ast.Constant(execution_microbatch)
                changed["micro"] += 1
            if stop_after_update and node.targets[0].id == "ssl_batches" and isinstance(node.value, ast.IfExp):
                # The final disposable update tests the occasional doubled SSL
                # batch. The preceding updates estimate normal throughput.
                node.value = ast.IfExp(
                    test=ast.parse(f"joint and completed + 1 == {stop_after_update}", mode="eval").body,
                    body=ast.Constant(2), orelse=node.value)
                changed["ssl_probe"] += 1
        if isinstance(node, ast.keyword) and node.arg == "microbatch_per_gpu":
            assert ast.unparse(node.value) == "micro"
            node.value = ast.parse("arguments.micro_batch", mode="eval").body
            changed["registration"] += 1
        if isinstance(node, ast.While) and ast.unparse(node.test) == "completed < total_updates":
            if stop_after_update:
                node.test = ast.parse(f"completed < min(total_updates, {stop_after_update})", mode="eval").body
            changed["loop"] += 1
    assert changed == dict(micro=1, registration=1, loop=1, ssl_probe=int(bool(stop_after_update))), changed
    namespace = dict(training.__dict__)
    module = ast.fix_missing_locations(ast.Module(body=[definition], type_ignores=[]))
    exec(compile(module, training.__file__, "exec"), namespace)
    # Use the live module globals: the resume helper replaces its model class
    # and the joint execution helper installs its objective after this compile.
    from types import FunctionType
    return replace_card_guard(FunctionType(namespace["main"].__code__, training.__dict__, "main"))


def main(arguments):
    import torch
    from torch.utils.checkpoint import checkpoint
    from types import MethodType
    import resume_small_corpus_planner_full_state as full_state

    training = full_state.original_training
    output = arguments.output.resolve()
    saved = torch.load(output / "latest.pt", map_location="cpu", weights_only=False, mmap=True)
    registration = json.loads((output / "registration.json").read_text())
    assert saved["registration"] == registration
    assert registration["microbatch_per_gpu"] == 2 and registration["effective_batch"] == 16
    completed = saved["completed_updates"]
    assert 0 < completed < 3200
    assert arguments.stop_after_update == 0 or completed < arguments.stop_after_update <= min(completed + 16, 3200)
    del saved
    assert arguments.micro_batch == 2 and arguments.profile_updates == 0
    assert os.environ.get("PYTORCH_CUDA_ALLOC_CONF") == "expandable_segments:True"

    original_allocator_limit = torch.cuda.set_per_process_memory_fraction

    def limit_allocator(fraction, device=None):
        capacity = torch.cuda.get_device_properties(
            torch.cuda.current_device() if device is None else device).total_memory
        return original_allocator_limit(min(fraction, arguments.allocator_cap_bytes / capacity), device)

    torch.cuda.set_per_process_memory_fraction = limit_allocator

    training.main = compile_execution_main(training, arguments.execution_microbatch, arguments.stop_after_update)
    original_model = training.CommonPlannerModel

    class CheckedModel(original_model):
        def __init__(self, *inputs, **keywords):
            super().__init__(*inputs, **keywords)
            # A changed BatchNorm batch would alter statistics, beyond stochastic
            # floating-point/dropout differences. Reject that unreviewed case.
            batch_norms = [name for name, module in self.named_modules()
                           if isinstance(module, torch.nn.modules.batchnorm._BatchNorm)]
            assert not batch_norms, batch_norms

    training.CommonPlannerModel = CheckedModel
    if arguments.stop_after_update:
        training.save = lambda *_args, **_keywords: None
        training.evaluate = lambda *_args, **_keywords: None

    def run_joint(execution_arguments):
        original_objective = training.PlanningObjective

        class RecomputedObjective(original_objective):
            def __init__(self, model, oracle, joint):
                super().__init__(model, oracle, joint)
                for module in (model.backbone.world_model.decoder_module, self.reconstruction.perceptual_loss):
                    original_forward = module.forward

                    def install_forward(saved_forward):
                        def recomputed(_module, *inputs, **keywords):
                            if torch.is_grad_enabled():
                                return checkpoint(saved_forward, *inputs, use_reentrant=False,
                                                  preserve_rng_state=True, **keywords)
                            return saved_forward(*inputs, **keywords)
                        return recomputed

                    module.forward = MethodType(install_forward(original_forward), module)

        training.PlanningObjective = RecomputedObjective
        training.main(execution_arguments)

    full_state.joint_execution.main = run_joint
    rank = int(os.environ["LOCAL_RANK"])
    previous_proof = output / f"resume_records/from_update{completed}_rank{rank}.json"
    if previous_proof.exists():
        history = output / "batch_execution/previous_resume_proofs"
        history.mkdir(parents=True, exist_ok=True)
        previous_proof.rename(history / f"{time.time_ns()}_{previous_proof.name}")
    execution = dict(start_update=completed, original_registration_microbatch=2,
        physical_microbatch_per_gpu=arguments.execution_microbatch,
        accumulation=8 // arguments.execution_microbatch, world_size=2, effective_batch=16,
        disposable=bool(arguments.stop_after_update), stop_after_update=arguments.stop_after_update,
        optimizer_scheduler_and_rank_rng_restored=True,
        planning_and_ssl_presentations_preserved=not bool(arguments.stop_after_update),
        bitwise_or_final_performance_equivalence_claimed=False,
        note="Microbatch grouping changes stochastic draws and floating-point reduction order",
        wrapper_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        original_trainer_sha256=hashlib.sha256(Path(training.__file__).read_bytes()).hexdigest(),
        card_limit_bytes=48_000_000_000)
    if rank == 0:
        (output / "batch_execution").mkdir(exist_ok=True)
        training.write_json(output / f"batch_execution/from_update{completed}_micro{arguments.execution_microbatch}_{time.time_ns()}.json", execution)
        training.write_json(output / "active_batch_execution.json", execution)
    full_state.main(arguments)
    if arguments.stop_after_update and rank == 0:
        path = output / "complete.json"
        if path.exists():
            result = json.loads(path.read_text())
            result.update(disposable_profile=True, profile_weights_discarded=True,
                          official_pdms_pending=False, source_start_update=completed)
            training.write_json(path, result)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--kind", choices=["lpwm_sequential", "lpwm_joint"], required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--stage1-checkpoint", type=Path)
    parser.add_argument("--micro-batch", type=int, default=2)
    parser.add_argument("--profile-updates", type=int, default=0)
    parser.add_argument("--native-marker", type=Path)
    parser.add_argument("--execution-microbatch", type=int, choices=[2, 4, 8], required=True)
    parser.add_argument("--stop-after-update", type=int, default=0)
    parser.add_argument("--allocator-cap-bytes", type=int, default=44_000_000_000)
    main(parser.parse_args())
