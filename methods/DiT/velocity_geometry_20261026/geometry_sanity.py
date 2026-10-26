"""Geometric witnesses for the research note; no DiT or image experiments.

Run from any directory. Writes geometry_sanity_results.json beside this file.
Only NumPy is needed. Passing checks establish algebraic properties, not erasure.
"""

import json
from pathlib import Path

import numpy as np


def dve_score(user, target, anchor):
    direction = anchor - target
    return float(user @ direction / np.linalg.norm(direction))


def pair_coordinate(user, target, anchor):
    direction = target - anchor
    return float((user - anchor) @ direction / (direction @ direction))


def kernel_transport(user, targets, anchors, sigma=0.5, eta=0.05):
    """Use valid non-degenerate pairs, FP64 here to check the identities."""
    differences = anchors - targets
    lengths_sq = np.sum(differences**2, axis=1)
    valid = lengths_sq > 1e-12
    targets, anchors = targets[valid], anchors[valid]
    differences, lengths_sq = differences[valid], lengths_sq[valid]
    if not len(targets):
        return np.zeros_like(user), np.zeros(0)
    target_kernel = np.exp(-np.sum((user - targets) ** 2, axis=1) / (sigma**2 * lengths_sq))
    anchor_kernel = np.exp(-np.sum((user - anchors) ** 2, axis=1) / (sigma**2 * lengths_sq))
    denominator = eta * len(targets) + np.sum(target_kernel + anchor_kernel)
    if denominator == 0:
        raise ValueError("The eta=0 diagnostic underflowed; choose a nearer witness.")
    weights = np.maximum(target_kernel - anchor_kernel, 0) / denominator
    return weights @ differences, weights


def main():
    target, anchor = np.array([2.0, 0.0]), np.zeros(2)
    shift = np.array([-100.0, 50.0])
    scores = [dve_score(target, target, anchor), dve_score(target + shift, target + shift, anchor + shift)]
    coordinates = [pair_coordinate(target, target, anchor), pair_coordinate(target + shift, target + shift, anchor + shift)]
    assert scores[0] < 0 < scores[1]
    assert np.allclose(coordinates, [1.0, 1.0])

    targets, anchors = target[None, :], anchor[None, :]
    correction, weights = kernel_transport(target, targets, anchors)
    shifted, _ = kernel_transport(target + shift, targets + shift, anchors + shift)
    safe_correction, _ = kernel_transport(anchor, targets, anchors)
    assert np.allclose(correction, shifted)
    assert np.allclose(safe_correction, 0)

    # An orthogonal displacement cancels out of normalized kernels if eta=0.
    far_user = target + np.array([0.0, 4.0])
    far_correction, _ = kernel_transport(far_user, targets, anchors)
    zero_eta_near, _ = kernel_transport(target, targets, anchors, eta=0)
    zero_eta_far, _ = kernel_transport(far_user, targets, anchors, eta=0)
    assert np.allclose(zero_eta_near, zero_eta_far)
    assert np.linalg.norm(far_correction) < np.linalg.norm(correction) / 1000

    # Two paired modes cancel under averaging, but query-dependent selection
    # preserves opposite edits for opposite inputs. This concerns mean caching,
    # not a claim that online DVE must average these pairs.
    mode_targets = np.array([[1.0, 0.0], [-1.0, 0.0]])
    mode_anchors = np.array([[0.0, 1.0], [0.0, -1.0]])
    mean_difference = np.mean(mode_anchors - mode_targets, axis=0)
    mode_positive, _ = kernel_transport(mode_targets[0], mode_targets, mode_anchors)
    mode_negative, _ = kernel_transport(mode_targets[1], mode_targets, mode_anchors)
    assert np.allclose(mean_difference, 0)
    assert np.linalg.norm(mode_positive) > 0.1
    assert np.allclose(mode_positive, -mode_negative)

    # Independent edits also need not be parallel to the mean difference.
    independent_targets = np.eye(4)[[0, 2]]
    independent_anchors = np.eye(4)[[1, 3]]
    independent_correction, _ = kernel_transport(independent_targets[0], independent_targets, independent_anchors)
    mean_direction = np.mean(independent_anchors - independent_targets, axis=0)
    cosine = float(independent_correction @ mean_direction / (np.linalg.norm(independent_correction) * np.linalg.norm(mean_direction)))
    assert abs(cosine) < 0.9

    # eta*K makes duplicating every reference pair leave the correction intact.
    duplicated, _ = kernel_transport(target, np.repeat(targets, 2, axis=0), np.repeat(anchors, 2, axis=0))
    assert np.allclose(correction, duplicated)
    assert np.sum(weights) <= 1

    # For the linear noise-time convention x0_hat = x_t - t*v, normalized
    # kernels and paired edits are equivalent in the predicted clean space.
    time, latent = 0.37, np.array([0.3, -0.2])
    clean_user = latent - time * target
    clean_targets, clean_anchors = latent - time * targets, latent - time * anchors
    clean_correction, _ = kernel_transport(clean_user, clean_targets, clean_anchors)
    equivalence_error = float(np.max(np.abs(clean_correction + time * correction)))
    assert equivalence_error < 1e-12

    results = {
        "scope": "Synthetic geometry only; no model generations or erasure metrics.",
        "dve_common_shift_scores": scores,
        "pair_centered_coordinates": coordinates,
        "kernel_common_shift_max_abs_error": float(np.max(np.abs(correction - shifted))),
        "single_pair_safe_anchor_correction_norm": float(np.linalg.norm(safe_correction)),
        "kernel_near_correction_norm": float(np.linalg.norm(correction)),
        "kernel_far_correction_norm_with_positive_eta": float(np.linalg.norm(far_correction)),
        "zero_eta_near_far_max_abs_difference": float(np.max(np.abs(zero_eta_near - zero_eta_far))),
        "opposite_modes_mean_difference": mean_difference.tolist(),
        "opposite_modes_selected_corrections": [mode_positive.tolist(), mode_negative.tolist()],
        "independent_mode_correction_cosine_with_mean": cosine,
        "reference_duplication_max_abs_error": float(np.max(np.abs(correction - duplicated))),
        "velocity_clean_prediction_equivalence_max_abs_error": equivalence_error,
        "all_checks_passed": True,
    }
    destination = Path(__file__).with_name("geometry_sanity_results.json")
    destination.write_text(json.dumps(results, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
