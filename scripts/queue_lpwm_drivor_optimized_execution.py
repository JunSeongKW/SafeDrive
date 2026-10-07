"""Retain the evaluation chain and apply the adopted runtime settings to v2."""
import argparse
import hashlib
import json

import queue_lpwm_drivor_planning_path_lora as queue


def main(arguments):
    execution = json.loads((queue.PROJECT_ROOT / arguments.execution).read_text())
    registration = json.loads((queue.PROJECT_ROOT / execution["registration"]).read_text())
    original_check = queue.check_registration
    original_run_command = queue.run_command

    def check_execution():
        original_check()
        for name, expected in registration["sources"].items():
            assert hashlib.sha256((queue.PROJECT_ROOT / name).read_bytes()).hexdigest() == expected, name

    def dispatch(label, command, cpu=False):
        command = ["scripts/train_lpwm_drivor_optimized_execution.py"
                   if item == "scripts/train_lpwm_drivor_planning_path_lora.py" else item for item in command]
        return original_run_command(label, command, cpu)

    queue.check_registration = check_execution
    queue.run_command = dispatch
    queue.EXECUTION_CONFIGURATION = arguments.execution
    try:
        queue.main()
    except Exception as error:
        queue.write_status({"stage": "stopped", "error": repr(error)})
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--execution", required=True)
    main(parser.parse_args())
