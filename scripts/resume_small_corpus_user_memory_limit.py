"""Resume the same optimizer state with the user-authorized 48GB card guard."""
import argparse
import hashlib
import json
import os
from pathlib import Path
from types import FunctionType

ORIGINAL_CARD_GUARD = 46_500_000_000
USER_CARD_GUARD = 48_000_000_000


def replace_card_guard(function):
    """Change only the unique integer constant used by the trainer stop check."""
    original = function.__code__
    assert sum(type(value) is int and value == ORIGINAL_CARD_GUARD for value in original.co_consts) == 1
    constants = tuple(USER_CARD_GUARD if type(value) is int and value == ORIGINAL_CARD_GUARD else value
                      for value in original.co_consts)
    updated = original.replace(co_consts=constants)
    assert updated.co_code == original.co_code
    replacement = FunctionType(updated, function.__globals__, function.__name__,
                               function.__defaults__, function.__closure__)
    replacement.__kwdefaults__ = function.__kwdefaults__
    return replacement


def main(arguments):
    import resume_small_corpus_planner_full_state as full_state
    training = full_state.original_training
    training.main = replace_card_guard(training.main)
    registration = dict(previous_trainer_guard_bytes=ORIGINAL_CARD_GUARD,
        active_trainer_guard_bytes=USER_CARD_GUARD,
        execution_change="Only the trainer's unique46500000000 stop-comparison constant becomes48000000000",
        original_trainer_sha256=hashlib.sha256(Path(training.__file__).read_bytes()).hexdigest(),
        wrapper_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        training_graph_optimizer_rng_batch_precision_and_data_order_unchanged=True)
    if int(os.environ["LOCAL_RANK"]) == 0:
        path = arguments.output / "execution_user_memory_limit.json"
        if path.exists():
            assert json.loads(path.read_text()) == registration
        else:
            training.write_json(path, registration)
    if arguments.kind == "jepa":
        import resume_small_corpus_jepa_memory_cap as capped_jepa
        capped_jepa.main(arguments)
    else:
        full_state.main(arguments)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--kind", choices=["jepa", "lpwm_sequential", "lpwm_joint"], required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--stage1-checkpoint", type=Path)
    parser.add_argument("--micro-batch", type=int, default=2)
    parser.add_argument("--profile-updates", type=int, default=0)
    parser.add_argument("--native-marker", type=Path)
    main(parser.parse_args())
