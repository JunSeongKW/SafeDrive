"""Package measured diagnostics and read-only repository evidence; no training."""
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))


def digest_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8*1024*1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    destination = PROJECT_ROOT / "results/residual_control_sampler_feasibility_20261011"
    local_metadata_path = PROJECT_ROOT / "outputs/residual_control_sampler_feasibility_20261011/actual_scorer_cache_metadata.json"
    (destination/"actual_scorer_cache_metadata.json").write_text(local_metadata_path.read_text())
    metadata = json.loads((PROJECT_ROOT / "outputs/soft_token_reweighting_v1/cache/metadata.json").read_text())
    manifest = json.loads((PROJECT_ROOT / "outputs/soft_token_reweighting_v1/subset_manifest.json").read_text())
    historical_analysis = json.loads((PROJECT_ROOT / "outputs/small_corpus_planning_diagnosis_20261010/drivor/candidate_analysis.json").read_text())
    registry_path = PROJECT_ROOT / "outputs/lpwm_front_history_stage1_lora_v1/registration.json"
    registry = json.loads(registry_path.read_text())
    evidence = {"packaged_utc": datetime.now(timezone.utc).isoformat(),
        "git_head_before_review_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, text=True).strip(),
        "visual_cache_metadata": metadata,
        "visual_cache_manifest_sha256": digest_file(PROJECT_ROOT / "outputs/soft_token_reweighting_v1/subset_manifest.json"),
        "visual_cache_samples": len(manifest["records"]),
        "historical_scorer_analysis": {key: historical_analysis[key] for key in (
            "complete", "count", "selected_pdms", "oracle_best_pdms", "selection_regret", "score_definition", "by_command")},
        "historical_analysis_path": "outputs/small_corpus_planning_diagnosis_20261010/drivor/candidate_analysis.json",
        "historical_analysis_is_new_sampler_result": False,
        "historical_analysis_checkpoint": "outputs/four_model_small_corpus_v1/drivor/pass3.pt",
        "historical_analysis_checkpoint_updates": 1920,
        "historical_analysis_uses_same_checkpoint_as_actual_gradient_diagnostic": False,
        "current_LPWM_registration_sha256": digest_file(registry_path),
        "source_files": {}, "unavailable_prescribed_TOAD_checkpoint": None}
    sources = ["src/planning_aware_future_prediction/object_centric/small_corpus_models.py",
        "src/planning_aware_future_prediction/models/residual_control_sampler.py",
        "scripts/diagnose_residual_control_sampler.py",
        "scripts/package_residual_sampler_feasibility.py",
        "configs/residual_control_sampler/feasibility_20261011.json",
        "reference_repositories/DrivoR/navsim/agents/drivoR/drivor_model.py",
        "reference_repositories/DrivoR/navsim/agents/drivoR/transformer_decoder.py",
        "reference_repositories/DrivoR/navsim/agents/drivoR/score_module/scorer.py",
        "reference_repositories/DriveJEPAOfficialEvaluation/navsim_v1/navsim/agents/drive_jepa_perception_based/score_module/scorer.py",
        "reference_repositories/TOAD/drivor_cem_portable/navsim/agents/utils/cem_utils.py",
        "reference_repositories/TOAD/drivor_cem_portable/navsim/agents/utils/drivor_mpc.py",
        "scripts/score_lpwm_drivor_navtest.py"]
    for source in sources:
        evidence["source_files"][source] = digest_file(PROJECT_ROOT/source)
    missing_checkpoint = "reference_repositories/TOAD/drivor_cem_portable/checkpoints/epoch_29_54.6.ckpt"
    evidence["unavailable_prescribed_TOAD_checkpoint"] = {
        "path": missing_checkpoint, "exists": (PROJECT_ROOT/missing_checkpoint).exists(),
        "scope": "checked this prescribed local path only; no claim that the authors have not released weights"}
    # The registry stores paths in historical formats; inspect its schema rather
    # than replacing any existing hashes. A separate inspection is recorded below.
    evidence["current_LPWM_registration_keys"] = list(registry)
    protected_paths = {**registry["sources"], **registry["configurations"]}
    changed_paths = [relative for relative, expected_hash in protected_paths.items()
                     if not (PROJECT_ROOT/relative).exists() or digest_file(PROJECT_ROOT/relative) != expected_hash]
    evidence["existing_training_source_preservation"] = {"checked_source_and_configuration_files": len(protected_paths),
                                                        "mismatches": changed_paths}
    assert not changed_paths
    evidence["reference_commits"] = {name: subprocess.check_output(
        ["git", "-C", str(PROJECT_ROOT / "reference_repositories" / name), "rev-parse", "HEAD"], text=True).strip()
        for name in ("TOAD", "DrivoR", "DriveJEPAOfficialEvaluation")}
    evidence["full_Drive_JEPA_planning_checkpoint_exists"] = Path(metadata["full_planning_checkpoint"]["path"]).exists()
    progress_path = PROJECT_ROOT / "outputs/lpwm_front_history_stage1_lora_v1/stage2/progress.json"
    if progress_path.exists():
        progress = json.loads(progress_path.read_text())
        evidence["existing_training_read_only_snapshot"] = {
            key: value for key, value in progress.items() if isinstance(value, (str, int, float, bool))}
    from planning_aware_future_prediction.models.residual_control_sampler import ResidualControlSampler
    evidence["pooled_Drivor_sampler_parameters"] = sum(parameter.numel() for parameter in ResidualControlSampler(visual_dimension=256).parameters())
    evidence["pooled_Drive_JEPA_sampler_parameters"] = sum(parameter.numel() for parameter in ResidualControlSampler(visual_dimension=1024).parameters())
    (destination/"repository_evidence.json").write_text(json.dumps(evidence, ensure_ascii=False, indent=2)+"\n")
    literature_lines = (destination/"literature_comparison.md").read_text().splitlines()
    table_rows = [line for line in literature_lines if line.startswith("|")]
    with (destination/"literature_comparison.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        for index, line in enumerate(table_rows):
            if index == 1:
                continue
            writer.writerow([cell.strip() for cell in line.strip("|").split("|")])
    print(json.dumps({"review_folder": str(destination), "literature_methods": len(table_rows)-2,
                      "pooled_Drivor_sampler_parameters": evidence["pooled_Drivor_sampler_parameters"]}), flush=True)


if __name__ == "__main__":
    main()
