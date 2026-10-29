"""Closed-form local temporal fit and covariance, including shared background."""

import numpy as np


def fit_weights(times, reference_time=0.0):
    offsets = np.asarray(times, dtype=float) - reference_time
    design = np.column_stack((np.ones(len(offsets)), offsets))
    if np.linalg.matrix_rank(design) < 2:
        raise ValueError('At least two distinct acquisition times are needed.')
    return np.linalg.pinv(design)


def baseline_weights(times):
    """Reference frame and the two adjacent frames for a centered difference."""
    times = np.asarray(times)
    mid = len(times) // 2
    if len(times) < 3 or not np.isclose(times[mid], 0):
        raise ValueError('Use an odd frame count with the middle time at zero.')
    weights = np.zeros((2, len(times)))
    weights[0, mid] = 1
    gap = times[mid+1] - times[mid-1]
    weights[1, mid-1], weights[1, mid+1] = -1/gap, 1/gap
    return weights


def fit_transients(counts, background_estimate, weights):
    return weights @ (counts - background_estimate)


def coefficient_variance(expected_total_counts, weights, background_variance=0.0):
    """Diagonal covariance of each fitted coefficient in measurement space.

    Frame counts are independent Poisson; a single independent background
    estimate is subtracted from every frame. Its error is therefore correlated
    across frames and cancels for any derivative weights with sum zero.
    """
    return weights**2 @ expected_total_counts + weights.sum(axis=1)[:, None]**2 * background_variance
