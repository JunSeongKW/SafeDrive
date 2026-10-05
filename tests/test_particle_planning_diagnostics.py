import numpy as np

from planning_aware_future_prediction.object_centric.particle_planning_diagnostics import (
    project_box_full_image, particle_geometry, object_particle_weights, region_distribution,
    matched_control_particles)


def test_full_image_projection_has_no_historical_crop():
    camera = {"sensor2lidar_rotation": np.eye(3), "sensor2lidar_translation": np.zeros(3),
              "cam_intrinsic": np.array([[100, 0, 100], [0, 100, 50], [0, 0, 1]])}
    rectangle = project_box_full_image(np.array([0, 0, 10, 2, 2, 2, 0]), camera, (200, 100))
    np.testing.assert_allclose(rectangle, [64-64/9, 64-128/9, 64+64/9, 64+128/9])
    assert project_box_full_image(np.array([0, 0, -10, 2, 2, 2, 0]), camera, (200, 100)) is None


def test_geometry_axis_order_and_no_support_not_dropped():
    attributes = np.zeros((2, 14)); attributes[:, 4] = 1
    attributes[:, :2] = [[-.5, .5], [.5, -.5]]
    attributes[:, 2:4] = [.2, .4]
    centers, sizes, _, _ = particle_geometry(attributes)
    np.testing.assert_allclose(centers, [[96, 32], [32, 96]])
    np.testing.assert_allclose(sizes, [[51.2, 25.6]]*2)
    weights, support = object_particle_weights(attributes, [[90, 25, 100, 35], [0, 0, 1, 1]])
    np.testing.assert_allclose(weights[0], [1, 0])
    np.testing.assert_equal(weights[1], [0, 0])
    assert support[1] == 0


def test_region_area_baseline_and_outside_particles():
    attributes = np.zeros((4, 14)); attributes[:, 4] = 1
    attributes[:, :2] = [[-.5, -.5], [-.5, .5], [.5, -.5], [.5, .5]]
    mask = np.zeros((128, 128), bool); mask[:64] = True
    values = region_distribution(attributes, {"region": mask})
    assert values["region"]["area_normalized_enrichment"] == 1
    attributes[0, :2] = [3, 3]
    assert region_distribution(attributes, {"region": mask})["outside_image_fraction"] == .25


def test_controls_match_camera_count_and_exclude_intervened_ids():
    attributes = np.random.default_rng(71).random((4, 9, 64, 14))
    selected = np.array([1, 2, 65, 130, 200])
    controls = matched_control_particles(attributes, selected)
    assert not set(selected) & set(controls)
    assert len(set(controls)) == len(selected)
    np.testing.assert_equal(controls//64, selected//64)
