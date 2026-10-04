"""Checks for fractional ROI association, missing support and probe split/statistics."""

import numpy as np

from planning_aware_future_prediction.object_centric.object_readout_diagnostics import (
    fit_ridge_readout, fractional_box_masks, paired_recording_interval,
    pool_particle_attributes, predict_ridge_readout, split_probe_recordings,
)


def test_fractional_box_uses_xy_axes_and_preserves_area():
    masks = fractional_box_masks([[1.25, .5, 2.75, 1.5], [-1, -1, 0, 0]], 3, 4)
    np.testing.assert_allclose(masks[0].sum(), 1.5)
    np.testing.assert_allclose(masks[0, :2, 1:3], .375)
    np.testing.assert_array_equal(masks[1], 0)


def test_multiple_particles_and_unsupported_objects_are_not_discarded():
    alpha = np.zeros((2, 2, 3))
    alpha[0, :, 0] = .4
    alpha[1, :, 1] = .8
    pooled, coverage, effective = pool_particle_attributes(alpha, [[3., 0], [0, 6.]],
        [[0, 0, 2, 2], [2, 0, 3, 2]])
    np.testing.assert_allclose(pooled[0], [1., 4.])
    np.testing.assert_allclose(coverage, [.6, 0])
    np.testing.assert_allclose(effective, [1.8, 0])
    np.testing.assert_array_equal(pooled[1], [0, 0])


def test_linear_probe_uses_training_normalization_when_evaluation_distribution_changes():
    training_features = np.arange(30.).reshape(-1, 1)
    readout = fit_ridge_readout(training_features, 3 * training_features + 7, 1e-8)
    evaluation_features = np.array([[100.], [200.]])
    np.testing.assert_allclose(predict_ridge_readout(readout, evaluation_features),
                               3 * evaluation_features + 7, rtol=1e-6)
    np.testing.assert_allclose(readout["feature_mean"], 14.5)


def test_recording_split_is_order_invariant_and_cluster_difference_is_paired():
    names = [f"recording_{index}" for index in range(10)]
    fitting, validation = split_probe_recordings(names, .2, 47)
    assert not fitting & validation
    assert fitting | validation == set(names)
    assert (fitting, validation) == split_probe_recordings(names[::-1], .2, 47)
    interval = paired_recording_interval([1, 2, 3, 4], [3, 4, 5, 6], ["a", "a", "b", "b"], 100, 47)
    np.testing.assert_allclose(interval["confidence_interval_95"], [2, 2])
    assert interval["difference"] == 2


def test_complete_readout_report_handles_missing_state_annotations(tmp_path):
    """Exercise CPU report serialization before the full extraction finishes."""
    import importlib.util
    import json
    from pathlib import Path

    project_root = Path(__file__).resolve().parents[1]
    specification = importlib.util.spec_from_file_location("object_readout_runner",
        project_root / "scripts/validate_lpwm_object_readouts.py")
    runner = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(runner)
    configuration = json.loads((project_root / "configs/lpwm_navsim_adaptation/object_readout_validation_v1.json").read_text())
    generator = np.random.default_rng(47)
    object_count = 36
    objects = []
    for object_index in range(object_count):
        partition = "probe_fit" if object_index < 18 else ("probe_validation" if object_index < 24 else "development")
        objects.append({"probe_partition": partition, "split": "development" if partition == "development" else "train",
            "category": object_index % 3, "observed_valid": [True] * 4,
            "observed_boxes": [[2., 3., 4., 5.]] * 4, "scenario": "straight",
            "small": object_index % 3 == 0, "far": False,
            "state_target": [float(object_index), 1., None if object_index % 3 == 0 else 2., 0., 20.]})
    (tmp_path / "observed_object_manifest.json").write_text(json.dumps({"objects": objects,
        "counts": {"synthetic_fixture": object_count}, "fitting_recordings": ["fit"],
        "validation_recordings": ["validation"], "development_recordings": ["development"],
        "state_target_order": ["ego_x_m", "ego_y_m", "ego_vx_mps", "ego_vy_mps", "camera_depth_m"]}))
    (tmp_path / "published").mkdir()
    np.savez_compressed(tmp_path / "published/features.npz",
        particle_geometry=generator.normal(size=(object_count, 4, 6)),
        particle_appearance=generator.normal(size=(object_count, 4, 4)),
        background=generator.normal(size=(object_count, 4, 4)),
        alpha_coverage=np.zeros((object_count, 4)))
    report = runner.compute_probe_report(configuration, tmp_path, "published")
    assert set(report["readouts"]) == set(configuration["readouts"])
    assert report["current_no_alpha_support_fraction"] == 1
    assert report["readouts"]["particle_combined"]["strata"]["small"]["state_mae"][2] is None
    assert json.loads((tmp_path / "published/probe_report.json").read_text())["model"] == "published"
