"""Evaluation-only GT-localized pooling and train-only linear representation readouts."""

import hashlib

import numpy as np


def split_probe_recordings(recording_names, validation_fraction, seed):
    ordered = sorted(set(recording_names), key=lambda name: hashlib.sha256(
        f"{seed}/{name}".encode()).hexdigest())
    validation_count = max(1, round(len(ordered) * validation_fraction))
    if validation_count >= len(ordered):
        raise ValueError("Need distinct fitting and validation recordings")
    return set(ordered[validation_count:]), set(ordered[:validation_count])


def fractional_box_masks(boxes, image_height, image_width):
    """Exact overlap of continuous xyxy boxes with unit pixel cells (not rounded crops)."""
    boxes = np.asarray(boxes, dtype=np.float64).reshape(-1, 4)
    pixel_columns = np.arange(image_width)[None]
    pixel_rows = np.arange(image_height)[None]
    horizontal = np.maximum(0, np.minimum(boxes[:, 2:3], pixel_columns + 1)
                            - np.maximum(boxes[:, :1], pixel_columns))
    vertical = np.maximum(0, np.minimum(boxes[:, 3:4], pixel_rows + 1)
                          - np.maximum(boxes[:, 1:2], pixel_rows))
    return vertical[:, :, None] * horizontal[:, None, :]


def pool_particle_attributes(alpha_contributions, particle_attributes, object_boxes):
    """Many-particle pooling; return zero features for empty support, never drop objects."""
    alpha_contributions = np.asarray(alpha_contributions, dtype=np.float64)
    particle_attributes = np.asarray(particle_attributes, dtype=np.float64)
    if alpha_contributions.ndim != 3 or len(alpha_contributions) != len(particle_attributes):
        raise ValueError("Expected particle x height x width alpha and particle x attributes")
    object_masks = fractional_box_masks(object_boxes, *alpha_contributions.shape[-2:])
    object_particle_mass = object_masks.reshape(len(object_masks), -1) @ alpha_contributions.reshape(
        len(alpha_contributions), -1).T
    support_mass = object_particle_mass.sum(-1)
    weights = object_particle_mass / np.maximum(support_mass[:, None], 1e-12)
    pooled = weights @ particle_attributes
    mean_alpha_coverage = support_mass / np.maximum(object_masks.sum((1, 2)), 1e-12)
    effective_particles = 1 / np.maximum((weights ** 2).sum(-1), 1e-12)
    effective_particles[support_mass <= 1e-12] = 0
    return pooled.astype(np.float32), mean_alpha_coverage.astype(np.float32), effective_particles.astype(np.float32)


def fit_ridge_readout(features, targets, regularization, sample_weights=None):
    features = np.asarray(features, dtype=np.float64)
    targets = np.asarray(targets, dtype=np.float64)
    if targets.ndim == 1:
        targets = targets[:, None]
    if len(features) == 0 or not np.isfinite(features).all() or not np.isfinite(targets).all():
        raise ValueError("Readout fitting requires finite nonempty data")
    feature_mean = features.mean(0)
    feature_scale = features.std(0)
    feature_scale[feature_scale < 1e-6] = 1
    normalized = np.column_stack([(features - feature_mean) / feature_scale, np.ones(len(features))])
    weights = np.ones(len(features)) if sample_weights is None else np.asarray(sample_weights)
    weights = weights / weights.sum()
    weighted_transpose = normalized.T * weights
    penalty = np.eye(normalized.shape[1]) * regularization
    penalty[-1, -1] = 0
    coefficients = np.linalg.solve(weighted_transpose @ normalized + penalty, weighted_transpose @ targets)
    return {"feature_mean": feature_mean, "feature_scale": feature_scale, "coefficients": coefficients}


def predict_ridge_readout(readout, features):
    normalized = (np.asarray(features, dtype=np.float64) - readout["feature_mean"]) / readout["feature_scale"]
    return np.column_stack([normalized, np.ones(len(normalized))]) @ readout["coefficients"]


def classification_metrics(true_categories, predicted_categories, number_of_categories):
    confusion = np.zeros((number_of_categories, number_of_categories), dtype=np.int64)
    np.add.at(confusion, (true_categories, predicted_categories), 1)
    true_positive = np.diag(confusion).astype(float)
    actual_counts = confusion.sum(1)
    predicted_counts = confusion.sum(0)
    recall = np.divide(true_positive, actual_counts, out=np.zeros_like(true_positive), where=actual_counts > 0)
    f1 = np.divide(2 * true_positive, actual_counts + predicted_counts,
                   out=np.zeros_like(true_positive), where=(actual_counts + predicted_counts) > 0)
    present = actual_counts > 0
    return {"macro_f1_present_classes": float(f1[present].mean()) if present.any() else None,
            "balanced_accuracy_present_classes": float(recall[present].mean()) if present.any() else None,
            "per_class_recall": recall.tolist(), "per_class_f1": f1.tolist(),
            "class_counts": actual_counts.tolist(), "confusion": confusion.tolist()}


def paired_recording_interval(values_before, values_after, recording_names, replicates, seed):
    """Object-weighted paired mean difference with whole-recording resampling."""
    differences = np.asarray(values_after) - np.asarray(values_before)
    recordings = np.asarray(recording_names)
    valid = np.isfinite(differences)
    groups = sorted(set(recordings[valid]))
    if len(groups) < 2:
        return {"difference": float(differences[valid].mean()) if valid.any() else None,
                "confidence_interval_95": None, "recordings": len(groups)}
    counts = np.array([np.sum(valid & (recordings == group)) for group in groups])
    sums = np.array([differences[valid & (recordings == group)].sum() for group in groups])
    draws = np.random.default_rng(seed).integers(0, len(groups), (replicates, len(groups)))
    bootstrap_means = sums[draws].sum(1) / counts[draws].sum(1)
    return {"difference": float(differences[valid].mean()),
            "confidence_interval_95": np.quantile(bootstrap_means, [.025, .975]).tolist(),
            "recordings": len(groups), "objects": int(valid.sum())}
