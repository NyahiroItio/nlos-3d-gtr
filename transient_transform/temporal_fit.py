"""Fixed linear intercept/slope estimation in slow acquisition time."""

import numpy as np


def fit_weights(times, reference_time=0.):
    times = np.asarray(times, dtype=np.float64)
    if times.ndim != 1 or len(times) < 2 or not np.all(np.isfinite(times)):
        raise ValueError('Use at least two finite acquisition times.')
    if not np.isfinite(reference_time):
        raise ValueError('Reference time must be finite.')
    design = np.column_stack((np.ones(len(times)), times-reference_time))
    if np.linalg.matrix_rank(design) < 2:
        raise ValueError('Distinct acquisition times are required.')
    return np.linalg.pinv(design)


def centered_difference_weights(times):
    times = np.asarray(times, dtype=np.float64)
    if times.ndim != 1 or len(times) < 3 or len(times) % 2 == 0 or np.any(np.diff(times) <= 0):
        raise ValueError('Use an odd, increasing sequence with at least three times.')
    mid = len(times)//2
    result = np.zeros((2, len(times)))
    result[0, mid] = 1
    result[1, mid-1] = -1/(times[mid+1]-times[mid-1])
    result[1, mid+1] = -result[1, mid-1]
    return result


class AffineTransientEstimator:
    """Stream exposure-normalized frames through fixed OLS coefficient weights."""

    def __init__(self, times, reference_time=0.):
        self.weights = fit_weights(times, reference_time)
        self._coefficients = None
        self.frame_count = 0

    def add(self, counts, exposure=1.):
        values = np.asarray(counts, dtype=np.float64)
        if self.frame_count == self.weights.shape[1]:
            raise ValueError('All scheduled frames have already been added.')
        if not np.isfinite(exposure) or exposure <= 0:
            raise ValueError('Exposure must be finite and positive.')
        if values.size == 0 or np.any(~np.isfinite(values)) or np.any(values < 0):
            raise ValueError('Counts must be finite, nonempty and nonnegative.')
        if self._coefficients is not None and values.shape != self._coefficients.shape[1:]:
            raise ValueError('All frames must have the same shape.')
        if self._coefficients is None:
            self._coefficients = np.zeros((2,)+values.shape)
        for coefficient in range(2):
            self._coefficients[coefficient] += self.weights[coefficient, self.frame_count]*values/exposure
        self.frame_count += 1

    def estimate(self):
        if self.frame_count != self.weights.shape[1]:
            raise ValueError('The complete scheduled sequence is required.')
        return self._coefficients.copy()
