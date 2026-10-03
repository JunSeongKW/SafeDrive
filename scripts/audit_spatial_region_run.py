"""Read-only checkpoint/paired-schedule audit; does not load the official model."""

import argparse
import json
from pathlib import Path

import torch


def audit_run(run_directory):
    specification = json.loads((run_directory / "specification.json").read_text())
    reports = {}
    for condition, options in specification["conditions"].items():
        for seed in specification["seeds"]:
            directory = run_directory / f"{condition}_seed{seed}"
            if not (directory / "results.json").is_file():
                continue
            initial = torch.load(
                directory / "initial.pt", map_location="cpu", weights_only=False
            )
            final = torch.load(
                directory / "complete.pt", map_location="cpu", weights_only=False
            )
            report = json.loads((directory / "results.json").read_text())
            assert final["completed_update"] == specification["joint_updates"]
            assert initial["completed_update"] == 0
            assert initial["batch_schedule"] == final["batch_schedule"]
            assert final["specification"] == specification
            assert len(final["curve"]) == specification["joint_updates"]
            assert [point["update"] for point in final["curve"]] == list(
                range(1, specification["joint_updates"] + 1)
            )
            changes = {}
            for prefix in ("patch_selector.", "future_predictor.", "future_bridge."):
                names = sorted(
                    name
                    for name in initial["extension_parameters"]
                    if name.startswith(prefix)
                )
                before = torch.cat(
                    [initial["extension_parameters"][name].flatten() for name in names]
                )
                after = torch.cat(
                    [final["extension_parameters"][name].flatten() for name in names]
                )
                changes[prefix[:-1]] = float(
                    (after - before).norm() / before.norm().clamp_min(1e-12)
                )
            if options["selection_mode"] == "random":
                assert changes["patch_selector"] == 0
            else:
                assert changes["patch_selector"] > 0
            assert changes["future_predictor"] > 0 and changes["future_bridge"] > 0
            expected_native_area = (
                options["region_budget"] * options["region_side"] ** 2
            )
            assert report["covered_native_cells"] == expected_native_area
            for split in ("train", "development"):
                windows = final["evaluations"]["800"][split]["windows"]
                assert len(windows) == specification["expected_cache_counts"][split]
                assert len({row["token"] for row in windows}) == len(windows)
                for row in windows:
                    region_ids = row["selected_region_ids"]
                    assert (
                        len(region_ids)
                        == len(set(region_ids))
                        == options["region_budget"]
                    )
                    assert (
                        min(region_ids) >= 0
                        and max(region_ids) < 512 // options["region_side"] ** 2
                    )
            reports[f"{condition}_seed{seed}"] = {
                "relative_parameter_change": changes,
                "completed_update": final["completed_update"],
                "batch_schedule_sha256": final["batch_schedule_sha256"],
                "gradient_contract": report["gradient_contract"],
                "counts_and_unique_ids_verified": True,
            }
    for seed in specification["seeds"]:
        matching = [
            report for name, report in reports.items() if name.endswith(f"_seed{seed}")
        ]
        assert len({report["batch_schedule_sha256"] for report in matching}) <= 1
    return {
        "complete_run_count": len(reports),
        "expected_run_count": len(specification["conditions"])
        * len(specification["seeds"]),
        "runs": reports,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-directory", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise RuntimeError("New audit output required")
    torch.set_num_threads(1)
    report = audit_run(args.run_directory)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({key: value for key, value in report.items() if key != "runs"}))


if __name__ == "__main__":
    main()
