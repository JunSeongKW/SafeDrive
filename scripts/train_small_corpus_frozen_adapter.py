"""Run the matched frozen-native LPWM control without changing preserved code."""
import argparse
import ast
import json
import os
from pathlib import Path
from types import FunctionType, MethodType

import torch

import train_small_corpus_common_planner as original_training
from planning_aware_future_prediction.object_centric.small_corpus_frozen_adapter import (
    SmallCorpusFrozenAdapterPlanner, adapter_gradient_groups,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def matched_execution_main(execution_microbatch, stop_after_update):
    parsed = ast.parse(Path(original_training.__file__).read_text())
    definition = next(node for node in parsed.body if isinstance(node, ast.FunctionDef) and node.name == 'main')
    replacements = {'microbatch': 0, 'registration': 0, 'card_guard': 0, 'boundary': 0}
    for node in ast.walk(definition):
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            if node.targets[0].id == 'micro':
                node.value = ast.Constant(execution_microbatch)
                replacements['microbatch'] += 1
        if isinstance(node, ast.While) and ast.unparse(node.test) == 'completed < total_updates' and stop_after_update:
            node.test = ast.parse(f'completed < min(total_updates, {stop_after_update})', mode='eval').body
            replacements['boundary'] += 1
        if isinstance(node, ast.keyword) and node.arg == 'microbatch_per_gpu':
            node.value = ast.Constant(2)
            replacements['registration'] += 1
        if isinstance(node, ast.Constant) and type(node.value) is int and node.value == 46_500_000_000:
            node.value = 48_000_000_000
            replacements['card_guard'] += 1
    if stop_after_update:
        training_try = next(node for node in ast.walk(definition) if isinstance(node, ast.Try)
                            and any(isinstance(child, ast.While) for child in node.body))
        loop_index = next(index for index, node in enumerate(training_try.body) if isinstance(node, ast.While))
        # Run the existing epoch validation before yielding and persist the
        # non-epoch1937 boundary too. No early-stop flag skips validation.
        training_try.body.insert(loop_index + 1, ast.parse(
            f'if completed >= {stop_after_update}:\n'
            '    save(output, model, optimizer, scheduler, completed, registration, rank)\n').body[0])
    assert replacements == {'microbatch': 1, 'registration': 1, 'card_guard': 1,
                            'boundary': int(bool(stop_after_update))}, replacements
    namespace = dict(original_training.__dict__)
    module = ast.fix_missing_locations(ast.Module(body=[definition], type_ignores=[]))
    exec(compile(module, original_training.__file__, 'exec'), namespace)
    return FunctionType(namespace['main'].__code__, original_training.__dict__, 'main')


def main(arguments):
    specification = json.loads(arguments.config.read_text())
    output = PROJECT_ROOT / specification['output_directory']
    stage1 = PROJECT_ROOT / specification['stage1_checkpoint']
    for filename, expected in specification['preserved_input_sha256'].items():
        assert original_training.digest(PROJECT_ROOT / filename) == expected, filename
    output.mkdir(parents=True, exist_ok=True)
    registration = {'specification': specification, 'sources': {
        str(path.relative_to(PROJECT_ROOT)): original_training.digest(path)
        for path in (Path(__file__), arguments.config,
                     PROJECT_ROOT / 'src/planning_aware_future_prediction/object_centric/small_corpus_frozen_adapter.py',
                     PROJECT_ROOT / 'src/planning_aware_future_prediction/object_centric/lpwm_adapter_full_finetuning.py',
                     PROJECT_ROOT / 'scripts/queue_small_corpus_frozen_adapter.py')}}
    registration_path = output / 'adapter_registration.json'
    if int(os.environ['LOCAL_RANK']) == 0:
        if registration_path.exists():
            assert json.loads(registration_path.read_text()) == registration
        else:
            original_training.write_json(registration_path, registration)
    def create_adapter_model(kind, checkpoint):
        assert kind == 'lpwm_sequential' and Path(checkpoint).resolve() == stage1.resolve()
        model = SmallCorpusFrozenAdapterPlanner(checkpoint, specification['adapter_bottleneck_dimension'])
        expected = torch.load(PROJECT_ROOT / specification['initial_planner_state'],
                              map_location='cpu', weights_only=False)
        downstream = {name: tensor for name, tensor in model.planner.state_dict().items()
                      if not name.startswith('image_backbone.') and name != 'scene_embeds'}
        assert set(downstream) == set(expected)
        assert all(torch.equal(tensor, expected[name]) for name, tensor in downstream.items())
        if (output / 'latest.pt').exists():
            saved = torch.load(output / 'latest.pt', map_location='cpu', mmap=True, weights_only=False)
            saved_random = saved['rank_rng_states'][int(os.environ['LOCAL_RANK'])]
            previous_train = model.train
            def train_with_resume_rng(_model, mode=True):
                result = previous_train(mode)
                if mode and not getattr(_model, '_rng_restored_before_training', False):
                    original_training.restore_random_state(saved_random)
                    _model._rng_restored_before_training = True
                return result
            model.train = MethodType(train_with_resume_rng, model)
        return model

    original_training.CommonPlannerModel = create_adapter_model
    original_training.gradient_groups = adapter_gradient_groups
    original_save = original_training.save

    def save_with_freeze_check(destination, model, *inputs, **keywords):
        assert model.frozen_native_digest() == model.native_initial_digest, 'Native LPWM state changed'
        return original_save(destination, model, *inputs, **keywords)

    original_training.save = save_with_freeze_check
    original_write_json = original_training.write_json
    def write_phase_completion(path, content):
        if path.name == 'complete.json' and content.get('completed_updates', 3200) < 3200:
            path = path.with_name(f'phase_complete_{content["completed_updates"]}.json')
        return original_write_json(path, content)
    original_training.write_json = write_phase_completion
    original_allocator = torch.cuda.set_per_process_memory_fraction
    def bounded_allocator(fraction, device=None):
        capacity = torch.cuda.get_device_properties(torch.cuda.current_device() if device is None else device).total_memory
        return original_allocator(min(fraction, 12_000_000_000 / capacity), device)
    torch.cuda.set_per_process_memory_fraction = bounded_allocator
    # Every original training operation, sample order, optimizer, loss, warmup,
    # microbatch and dev evaluation remains in the preserved implementation.
    execution = matched_execution_main(arguments.execution_microbatch, arguments.stop_after_update)
    execution(argparse.Namespace(kind='lpwm_sequential', output=output,
        stage1_checkpoint=stage1, micro_batch=2, profile_updates=arguments.profile_updates))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--profile-updates', type=int, default=0)
    parser.add_argument('--execution-microbatch', type=int, choices=[2, 4], default=2)
    parser.add_argument('--stop-after-update', type=int, default=0)
    main(parser.parse_args())
