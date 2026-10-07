"""Preserve the registered evaluation queue with eight CPU oracle workers."""
import hashlib
import json

import queue_lpwm_drivor_planning_path_lora as queue


def main():
    execution_name = "configs/lpwm_drivor_planning_path_lora/execution_batch16_loader2_oracle8.json"
    execution = json.loads((queue.PROJECT_ROOT / execution_name).read_text())
    registration = json.loads((queue.PROJECT_ROOT / execution["registration"]).read_text())
    original_check = queue.check_registration

    def verify_execution():
        original_check()
        for name, expected in registration["sources"].items():
            assert hashlib.sha256((queue.PROJECT_ROOT / name).read_bytes()).hexdigest() == expected, name

    queue.check_registration = verify_execution
    queue.EXECUTION_CONFIGURATION = execution_name
    try:
        queue.main()
    except Exception as error:
        import time
        queue.write_status({"stage": "stopped", "error": repr(error), "updated_unix": time.time()})
        raise


if __name__ == "__main__":
    main()
