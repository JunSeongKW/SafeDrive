"""Verify an explicit, evidence-linked scientific amendment without editing history."""
import json
from pathlib import Path

from evaluate_lpwm_full_planning import digest


def require_stage2_admission(specification, project_root, stage1_root, checkpoint_path):
    original_path = stage1_root / "adaptation_gate.json"
    original = json.loads(original_path.read_text())
    checkpoint_sha256 = digest(checkpoint_path)
    assert checkpoint_sha256 == original["training"]["checkpoint_sha256"]
    if "stage1_admission_amendment" not in specification:
        assert original["adaptation_gate_passed"], "Stage1 adaptation gate failed"
        return
    amendment_path = project_root / specification["stage1_admission_amendment"]
    amendment = json.loads(amendment_path.read_text())
    assert amendment["authorize_stage2_experiment"] and amendment["checkpoint_sha256"] == checkpoint_sha256
    assert amendment["original_adaptation_gate_sha256"] == digest(original_path)
    assert amendment["reclassified_diagnostic_checks"] == ["object_box_recall_noninferiority"]
    remaining = {name: passed for name, passed in original["gate_checks"].items()
        if name not in amendment["reclassified_diagnostic_checks"]}
    assert remaining and all(remaining.values()), "A retained stage1 criterion failed"
    for relative, checksum in amendment["evidence_sha256"].items():
        assert digest(project_root / relative) == checksum, f"Admission evidence changed: {relative}"
    readiness = json.loads((project_root / amendment["historical_transition_gate"]).read_text())
    assert readiness["checkpoint_sha256"] == checkpoint_sha256
    assert all(passed for name, passed in readiness["checks"].items() if name != "registered_stage1_adaptation_gate")
    probe = json.loads((project_root / amendment["object_readout_summary"]).read_text())
    assert probe["status"] == "completed_current_state_readout_diagnostic"
