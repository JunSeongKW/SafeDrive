"""Preserve the active LoRA optimizer before the user-authorized batch amendment."""
import json
import os
from pathlib import Path
import shutil
import signal
import time

from evaluate_lpwm_full_planning import digest, write_json
from queue_lpwm_adaptation_methods import process_identity, identity_alive

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def main():
    root = PROJECT_ROOT / "outputs/lpwm_card_budget_planning_v2/queue"
    root.mkdir(parents=True, exist_ok=True)
    destination = root / "execution_handoff.json"
    if destination.exists():
        print(destination.read_text(), flush=True)
        return
    previous_root = PROJECT_ROOT / "outputs/lpwm_48gb_planning_v1/queue"
    previous_state = json.loads((previous_root / "queue_state.json").read_text())
    supervisor = process_identity(previous_state["queue_pid"])
    assert supervisor and supervisor["uid"] == os.getuid()
    assert any(value.endswith("queue_lpwm_48gb_planning.py") for value in supervisor["arguments"])
    assert previous_state["current_stage"] == "attention_lora_training"
    training = process_identity(previous_state["nodes"]["attention_lora_training"]["pid"])
    assert training and training["uid"] == os.getuid()
    assert "torch.distributed.run" in training["arguments"]
    workers = []
    for path in Path("/proc").iterdir():
        if path.name.isdecimal():
            identity = process_identity(int(path.name))
            if identity and identity["parent_pid"] == training["pid"]:
                workers.append(identity)
    assert len(workers) == 2
    assert all(worker["uid"] == os.getuid() and "scripts/run_lpwm_48gb_stage.py" in worker["arguments"] for worker in workers)
    record = {"authorization": "User: batch8 for all queued work if each card remains below48GB",
        "previous_state": previous_state, "supervisor": supervisor, "training": training,
        "workers": workers, "started_unix": time.time()}
    write_json(root / "execution_handoff_started.json", record)
    assert identity_alive(supervisor)
    os.kill(supervisor["pid"], signal.SIGKILL)  # CPU only; avoid torchrun tearing down saving ranks.
    write_json(previous_root / "queue_superseded_card_budget.json", record)
    write_json(previous_root / "queue_state.json", {**previous_state, "status": "superseded",
        "new_queue_directory": str(root), "reason": record["authorization"]})
    for worker in workers:
        assert identity_alive(worker)
        os.kill(worker["pid"], signal.SIGINT)
    deadline = time.monotonic() + 180
    while identity_alive(training) or any(identity_alive(worker) for worker in workers):
        assert time.monotonic() < deadline, "Do not discard an unfinished checkpoint"
        time.sleep(1)
    import torch
    source = PROJECT_ROOT / "outputs/lpwm_48gb_planning_v1/attention_lora/batch4/metric_plus_world/latest.pt"
    saved = torch.load(source, map_location="cpu", weights_only=False)
    configuration = PROJECT_ROOT / "configs/lpwm_planning/execution_48gb_v1/attention_lora_batch4.json"
    assert saved["configuration_sha256"] == digest(configuration)
    assert saved["reason"] == "signal" and 0 < saved["completed_updates"] < 4707
    assert {int(value["step"]) for value in saved["optimizer"]["state"].values()} == {saved["completed_updates"]}
    record.update(completed_updates=saved["completed_updates"], optimizer_states=len(saved["optimizer"]["state"]),
        checkpoint_sha256=digest(source), source_configuration=str(configuration))
    del saved
    preserved = root / "lora_resume_source.pt"
    shutil.copy2(source, preserved)
    assert digest(preserved) == record["checkpoint_sha256"]
    record.update(checkpoint=str(preserved), finished_unix=time.time())
    write_json(destination, record)
    write_json(PROJECT_ROOT / "results/lpwm_card_budget_planning_v2/queue/execution_handoff.json", record)
    print(json.dumps({key: record[key] for key in ("completed_updates", "optimizer_states", "checkpoint_sha256", "checkpoint")}), flush=True)


if __name__ == "__main__":
    main()
