"""Seal the explicitly authorized three-path LoRA experiment after GPU audits."""
import hashlib
import json
from pathlib import Path
import time

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "outputs/lpwm_drivor_planning_path_lora_v1"
RESULTS = ROOT / "results/lpwm_drivor_planning_path_lora_v1"
MONITOR = ROOT / "outputs/lpwm_drivor_planning_path_representation_monitor_v1"


def digest(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024**2), b""):
            result.update(chunk)
    return result.hexdigest()


def write_json(path, values):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(values, indent=2, allow_nan=False) + "\n")


def main():
    assert not (OUTPUT / "registration.json").exists(), "Never silently reseal an experiment"
    audit = json.loads((OUTPUT / "engineering_audit/passed.json").read_text())
    gradients = json.loads((OUTPUT / "engineering_audit/adapter_gradients.json").read_text())
    assert audit["passed"] and audit["native_weights_and_buffers_unchanged"]
    assert len(gradients) == 273 and all(value["connected"] and value["output_gradient_l2"] > 0 for value in gradients.values())
    execution_name = "configs/lpwm_drivor_planning_path_lora/execution_batch16_loader2_oracle4.json"
    probe_root = OUTPUT / "navsim_v1/ddp_engineering" / Path(execution_name).stem
    probe = json.loads((probe_root / "passed.json").read_text())
    assert probe["passed"] and probe["updates"] == 2
    rank_rows = [json.loads(line) for rank in range(2)
                 for line in (probe_root / f"rank{rank}_training.jsonl").read_text().splitlines()]
    assert len(rank_rows) == 4 and max(row["card_used_bytes"] for row in rank_rows) < 48_000_000_000
    inherited = json.loads((ROOT / "outputs/lpwm_drivor_geometry_lora_v1/registration.json").read_text())
    sources = inherited["sources"]
    for filename, expected in sources.items():
        assert digest(ROOT / filename) == expected, filename
    additions = [ROOT / "src/planning_aware_future_prediction/object_centric/lpwm_drivor_planning_path_lora.py",
                 ROOT / "scripts/visualize_lpwm_planning_path_geometry.py",
                 ROOT / "scripts/monitor_lpwm_drivor_planning_path_representations.py"]
    additions += list((ROOT / "scripts").glob("*lpwm_drivor_planning_path_lora.py"))
    additions += list((ROOT / "configs/lpwm_drivor_planning_path_lora").glob("*.json"))
    for path in additions:
        sources[str(path.relative_to(ROOT))] = digest(path)
    configurations = {str(path.relative_to(ROOT)): digest(path)
                      for path in (ROOT / "configs/lpwm_drivor_planning_path_lora").glob("navsim_v*.json")}
    configuration = json.loads((ROOT / "configs/lpwm_drivor_planning_path_lora/navsim_v1.json").read_text())
    registration = {
        "created_unix": time.time(), "sources": sources, "configurations": configurations,
        "authorization": "User selected geometry+attribute/prior CNN, appearance+interaction, context/dynamics/future heads and requested training.",
        "initialization": "Fresh public LPWM and identical seed2 planner; prior epoch and engineering weights excluded",
        "immutable_inputs": {configuration[key]: digest(ROOT / configuration[key]) for key in ("manifest", "public_checkpoint")},
        "adaptation": audit["parameter_inventory"], "all_273_adapter_output_gradients_positive": True,
        "engineering_audit_passed": True, "ddp_batch16_effective64_passed": True,
        "old_conditions_and_pause_files_preserved": True,
        "final_checkpoint_policy": "25 epochs; DrivoR comparison after completion; training-panel monitor cannot tune on navtest"}
    write_json(OUTPUT / "registration.json", registration)
    for filename in ("parallelism_registration.json", "queue_registration.json"):
        write_json(OUTPUT / filename, {"sources": sources, "created_unix": time.time()})
    write_json(OUTPUT / "parallelism_selection.json", {"execution_configuration": execution_name,
        "effective_batch": 64, "ddp_probe": probe, "maximum_card_used_bytes_by_rank":
        {str(rank): max(row["card_used_bytes"] for row in rank_rows[2*rank:2*rank+2]) for rank in range(2)}})
    original_monitor = ROOT / "outputs/lpwm_drivor_representation_monitor_v1"
    for filename in ("panel.json", "images.npy", "road_proxy_masks.npy", "oracle_manifest.json"):
        destination = MONITOR / filename
        if not destination.exists():
            destination.symlink_to(original_monitor / filename)
    monitor_registration = json.loads((ROOT / "outputs/lpwm_drivor_geometry_representation_monitor_v1/registration.json").read_text())
    monitor_registration.update(sources=sources, panel_sha256=digest(MONITOR / "panel.json"),
        condition="geometry_appearance_future_planning_path_lora", created_unix=time.time(),
        exact_epoch_checkpoints=True, geometry_size_presence_plots=True,
        diagnostic_admission_total_card_bytes=42_000_000_000)
    write_json(MONITOR / "registration.json", monitor_registration)
    for filename, content in (("registration.json", registration), ("engineering_audit.json", audit),
                              ("adapter_gradients.json", gradients), ("ddp_engineering.json", {"probe": probe, "rank_rows": rank_rows})):
        write_json(RESULTS / filename, content)
    print(json.dumps({"sealed": True, "registered_sources": len(sources), "lora_parameters": audit["parameter_inventory"]["trainable_lora_parameters"],
        "maximum_total_card_bytes": max(row["card_used_bytes"] for row in rank_rows)}))


if __name__ == "__main__":
    main()
