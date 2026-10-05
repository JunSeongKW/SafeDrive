"""Evaluation-only geometry diagnostics for full-image LPWM inputs.

Projected annotation boxes and planar map masks are diagnostic proxies, not
semantic predictions, instance masks, or supervision for the training model.
"""
import numpy as np


def project_box_full_image(box, camera, image_size, output_size=128):
    box = np.asarray(box, dtype=float)
    signs = np.array([[horizontal, lateral, vertical] for horizontal in (-.5, .5)
                      for lateral in (-.5, .5) for vertical in (-.5, .5)])
    cosine, sine = np.cos(box[6]), np.sin(box[6])
    rotation = np.array([[cosine, -sine, 0], [sine, cosine, 0], [0, 0, 1]])
    lidar_corners = (signs * box[3:6]) @ rotation.T + box[:3]
    corners = (lidar_corners - camera["sensor2lidar_translation"]) @ np.linalg.inv(camera["sensor2lidar_rotation"]).T
    points = list(corners[corners[:, 2] >= .1])
    for first in range(8):
        for second in range(first+1, 8):
            if np.count_nonzero(signs[first] != signs[second]) != 1:
                continue
            origin, destination = corners[[first, second]]
            if (origin[2] < .1) != (destination[2] < .1):
                points.append(origin + (.1-origin[2])/(destination[2]-origin[2])*(destination-origin))
    if not points:
        return None
    projected = np.asarray(points) @ np.asarray(camera["cam_intrinsic"]).T
    pixels = projected[:, :2] / projected[:, 2:3]
    pixels *= output_size / np.asarray(image_size)
    rectangle = np.clip(np.r_[pixels.min(0), pixels.max(0)], 0, output_size)
    return rectangle if np.isfinite(rectangle).all() and np.all(rectangle[2:]-rectangle[:2] >= 1) else None


def particle_geometry(attributes, image_size=128):
    """LPWM native position/scale order is y,x; output is pixel x,y / width,height."""
    attributes = np.asarray(attributes)
    centers = (attributes[..., :2][..., ::-1] + 1) * image_size / 2
    sizes = attributes[..., 2:4][..., ::-1] * image_size
    boxes = np.concatenate((centers-sizes/2, centers+sizes/2), -1)
    return centers, sizes, boxes, np.clip(attributes[..., 4], 0, 1)


def object_particle_weights(current_attributes, object_boxes):
    _, _, particle_boxes, presence = particle_geometry(current_attributes)
    objects = np.asarray(object_boxes).reshape(-1, 4)
    lower = np.maximum(objects[:, None, :2], particle_boxes[None, :, :2])
    upper = np.minimum(objects[:, None, 2:], particle_boxes[None, :, 2:])
    intersection = np.maximum(upper-lower, 0).prod(-1)
    mass = intersection * presence[None]
    totals = mass.sum(-1)
    weights = mass / np.maximum(totals[:, None], 1e-12)
    return weights, totals


def region_distribution(current_attributes, masks):
    centers, _, _, presence = particle_geometry(current_attributes)
    inside = np.all((centers >= 0) & (centers < 128), -1)
    indices = np.clip(np.floor(centers).astype(int), 0, 127)
    result = {"outside_image_fraction": float((~inside).mean()),
              "presence_mean": float(presence.mean()),
              "upper_third_fraction": float((inside & (centers[:, 1] < 128/3)).mean())}
    for name, mask in masks.items():
        occupied = inside & mask[indices[:, 1], indices[:, 0]]
        fraction = float(occupied.mean())
        area = float(mask.mean())
        weighted = float((presence * occupied).sum() / max(presence.sum(), 1e-12))
        result[name] = {"center_fraction": fraction, "presence_weighted_fraction": weighted,
                        "image_area_fraction": area, "area_normalized_enrichment": weighted/area if area > 0 else None}
    return result


def matched_control_particles(attributes, selected_indices):
    """Same per-camera count; greedily match presence and log glimpse area."""
    flattened = np.asarray(attributes)[:, 0].reshape(-1, 14)
    _, sizes, _, presence = particle_geometry(flattened)
    descriptors = np.column_stack((presence, np.log(np.maximum(sizes.prod(-1), 1))))
    descriptors /= np.maximum(descriptors.std(0), 1e-3)
    excluded = set(map(int, selected_indices))
    controls = []
    for selected in selected_indices:
        camera = int(selected)//64
        eligible = [index for index in range(camera*64, (camera+1)*64) if index not in excluded]
        chosen = min(eligible, key=lambda index: (float(np.square(descriptors[index]-descriptors[selected]).sum()), index))
        controls.append(chosen)
        excluded.add(chosen)
    return np.array(controls, dtype=int)
