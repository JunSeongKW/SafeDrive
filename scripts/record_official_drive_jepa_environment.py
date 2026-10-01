"""Record the isolated Conda environment without modifying any other environment."""

import json
import os
import platform
import subprocess
import sys
from pathlib import Path


def main():
    workspace = Path(__file__).resolve().parents[1]
    expected_prefix = workspace / "runtime/environments/drive_jepa_official_evaluation"
    if Path(sys.prefix).resolve() != expected_prefix.resolve():
        raise RuntimeError("Run with the project-specific Conda Python, not base or a pilot interpreter")
    import numpy
    import torch

    output_root = workspace / "results/official_drive_jepa_reproduction"
    output_root.mkdir(parents=True, exist_ok=True)
    freeze = subprocess.run([sys.executable, "-m", "pip", "freeze"], check=True, capture_output=True, text=True).stdout
    (output_root / "environment_pip_freeze.txt").write_text(freeze)
    conda_packages = [json.loads(record.read_text()) for record in sorted((expected_prefix / "conda-meta").glob("*.json"))]
    report = {
        "environment_kind": "independent_conda_prefix_not_venv_overlay",
        "python_executable": sys.executable,
        "conda_prefix": sys.prefix,
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "torch_version": torch.__version__,
        "torch_cuda_runtime_version": torch.version.cuda,
        "numpy_version": numpy.__version__,
        "python_no_user_site": os.environ.get("PYTHONNOUSERSITE"),
        "conda_packages": [{key: record.get(key) for key in ("name", "version", "build", "channel")} for record in conda_packages],
        "pip_freeze": "results/official_drive_jepa_reproduction/environment_pip_freeze.txt",
        "existing_base_pilot_other_researcher_environments_modified": False,
        "operating_system_or_cuda_driver_modified": False,
        "dependency_note": "Official PF inference/scorer subset; versions not pinned upstream locked here; no PB-only mmcv/mmdet",
    }
    (output_root / "environment.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: value for key, value in report.items() if key != "conda_packages"}, indent=2))


if __name__ == "__main__":
    main()
