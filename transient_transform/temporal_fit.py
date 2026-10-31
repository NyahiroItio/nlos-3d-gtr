"""Fixed first- or second-order fits in slow acquisition time."""

import numpy as np


def fit_weights(times, reference_time=0., degree=1):
    times = np.asarray(times, dtype=np.float64)
    if degree not in (1, 2):
        raise ValueError('Only first- and second-order fits are supported.')
    if times.ndim != 1 or len(times) < degree+1 or not np.all(np.isfinite(times)):
        raise ValueError('Use enough finite acquisition times for the fit degree.')
    if not np.isfinite(reference_time):
        raise ValueError('Reference time must be finite.')
    shifted = times-reference_time
    scale = np.max(np.abs(shifted))
    if scale == 0:
        raise ValueError('Distinct acquisition times are required.')
    design = np.column_stack([(shifted/scale)**power for power in range(degree+1)])
    if np.linalg.matrix_rank(design) < degree+1:
        raise ValueError('Enough distinct acquisition times are required.')
    return np.linalg.pinv(design)/scale**np.arange(degree+1)[:, None]


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


class PolynomialTransientEstimator:
    """Stream exposure-normalized frames through fixed OLS coefficient weights."""

    def __init__(self, times, reference_time=0., degree=2):
        self.weights = fit_weights(times, reference_time, degree)
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
            self._coefficients = np.zeros((self.weights.shape[0],)+values.shape)
        for coefficient in range(self.weights.shape[0]):
            self._coefficients[coefficient] += self.weights[coefficient, self.frame_count]*values/exposure
        self.frame_count += 1

    def estimate(self):
        if self.frame_count != self.weights.shape[1]:
            raise ValueError('The complete scheduled sequence is required.')
        return self._coefficients.copy()


class AffineTransientEstimator(PolynomialTransientEstimator):
    """Compatibility entry point for the original first-order fit."""

    def __init__(self, times, reference_time=0.):
        super().__init__(times, reference_time, degree=1)
