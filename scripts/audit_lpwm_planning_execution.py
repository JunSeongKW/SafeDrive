"""Read-only audit: actual saved weights, unique samples, and inference causality."""
import contextlib
import hashlib
import io
import json
import sys
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
from planning_aware_future_prediction.object_centric.lpwm_bridge import load_official_lpwm, checkpoint_digest
from planning_aware_future_prediction.object_centric.lpwm_planner import IntentConditionedParticlePlanner

OUTPUT_ROOT = PROJECT_ROOT / "outputs/lpwm_planning_v1"
RESULT_ROOT = PROJECT_ROOT / "results/lpwm_planning_v1"


def main():
    torch.set_num_threads(2)
    specification = json.loads((PROJECT_ROOT / "configs/lpwm_planning/controlled_v1.json").read_text())
    checkpoint = PROJECT_ROOT / specification["initial_checkpoint"]
    with contextlib.redirect_stdout(io.StringIO()):
        official, _ = load_official_lpwm("cpu", checkpoint)
    parameter_accounting = {"official_full_unique_parameters": sum(parameter.numel() for parameter in official.parameters()),
                            "official_image_encoder_unique_parameters_excluding_context": sum(parameter.numel() for parameter in official.encoder_module.parameters()) - sum(parameter.numel() for parameter in official.encoder_module.ctx_enc.parameters())}
    del official
    initial_model = IntentConditionedParticlePlanner(checkpoint, "planning_joint").eval()
    initial_parameters = {name: parameter.detach().clone() for name, parameter in initial_model.named_parameters() if name.startswith("particle_encoder.")}
    reports = {}
    for condition in specification["conditions"]:
        for seed in specification["seeds"]:
            name = f"{condition}_seed{seed}"
            report = json.loads((OUTPUT_ROOT / "runs" / name / "results.json").read_text())
            trained = torch.load(OUTPUT_ROOT / "runs" / name / "model.pt", map_location="cpu", weights_only=True)
            changed_parameters, changed_scalars, maximum_change = 0, 0, 0.
            for parameter_name, before in initial_parameters.items():
                difference = (trained[parameter_name] - before).abs()
                changed_parameters += int(bool(torch.any(difference)))
                changed_scalars += int((difference > 0).sum())
                maximum_change = max(maximum_change, float(difference.max()))
            assert (changed_parameters == 0) == (condition in ("frozen_particles", "ego_only"))
            reports[name] = {"completed_updates": report["completed_updates"], "changed_unique_encoder_parameter_tensors": changed_parameters,
                             "changed_unique_encoder_parameter_scalars": changed_scalars, "encoder_max_absolute_change": maximum_change,
                             "checkpoint_sha256": checkpoint_digest(OUTPUT_ROOT / "runs" / name / "model.pt")}
            print("WEIGHT_AUDIT", name, changed_parameters, changed_scalars, flush=True)
    del initial_model, initial_parameters
    manifest = json.loads((OUTPUT_ROOT / "manifest.json").read_text())
    records = manifest["records"]
    cache = torch.load(OUTPUT_ROOT / "supervised_cache.pt", map_location="cpu", weights_only=True)
    train_indices = np.array([index for index, record in enumerate(records) if record["split"] == "train"])
    sampled = {str(seed): int(len(np.unique(np.random.default_rng(seed).choice(train_indices, (specification["updates"], specification["batch_size"])))) for seed in specification["seeds"]}
    sample_accounting = {"unique_training_windows": len(train_indices), "unique_training_image_pairs": len({hashlib.sha256(cache["observed_images"][index].numpy().tobytes()).hexdigest() for index in train_indices}),
                         "unique_training_windows_actually_sampled_by_seed": sampled, "sampled_clips_per_model": specification["updates"] * specification["batch_size"]}
    model = IntentConditionedParticlePlanner(checkpoint, "object_future_risk").eval()
    model.load_state_dict(torch.load(OUTPUT_ROOT / "runs/object_future_risk_seed47/model.pt", map_location="cpu", weights_only=True), strict=True)
    dev_indices = [index for index, record in enumerate(records) if record["split"] == "development"][:8]
    def causal_forward():
        images = cache["observed_images"][dev_indices].permute(0, 1, 4, 2, 3).float() / 255
        return model(images, cache["ego_status"][dev_indices])["trajectory"]
    with torch.no_grad():
        before = causal_forward()
        for key, value in cache.items():
            if key not in ("observed_images", "ego_status"):
                if value.dtype == torch.bool:
                    value[dev_indices] = ~value[dev_indices]
                else:
                    value[dev_indices] = 12345
        after = causal_forward()
        assert torch.equal(before, after)
        blank_images = torch.zeros(8, 2, 3, 128, 128)
        blank_trajectory = model(blank_images, cache["ego_status"][dev_indices])["trajectory"]
    stored_rows = json.loads((OUTPUT_ROOT / "runs/object_future_risk_seed47/results.json").read_text())["evaluations"]["development"]["windows"][:8]
    stored_trajectories = torch.tensor([row["trajectory"] for row in stored_rows])
    torch.testing.assert_close(before, stored_trajectories, atol=2e-4, rtol=2e-4)
    causality = {"tested_development_windows": len(dev_indices), "privileged_target_intervention_max_trajectory_difference": float((before - after).abs().max()),
                 "cpu_checkpoint_replay_max_trajectory_difference": float((before - stored_trajectories).abs().max()),
                 "blank_image_intervention_mean_xy_trajectory_change_m": float((blank_trajectory[..., :2] - before[..., :2]).norm(dim=-1).mean())}
    registration = json.loads((OUTPUT_ROOT / "registration.json").read_text())
    source_hashes_preserved = {path: checkpoint_digest(PROJECT_ROOT / path) == digest for path, digest in registration["source_files_sha256"].items()}
    assert all(source_hashes_preserved.values())
    result = {"parameter_accounting": parameter_accounting, "sample_accounting": sample_accounting, "weight_audits": reports,
              "causality_and_replay": causality, "registered_training_sources_unchanged": source_hashes_preserved,
              "original_checkpoint_preserved": all(json.loads((OUTPUT_ROOT / f"worker{worker}_complete.json").read_text())["initial_checkpoint_preserved"] for worker in (0, 1)),
              "scope": "Read-only CPU verification; training parameter names are deduplicated; unchanged buffers and shared state_dict aliases excluded"}
    (RESULT_ROOT / "execution_audit.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(causality, indent=2), flush=True)


if __name__ == "__main__":
    main()
