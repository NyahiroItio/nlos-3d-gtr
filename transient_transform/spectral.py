"""One factorization and a fixed linear map; no iterative reconstruction.

This is a small-scale spectral reference inverse, not an FFT-based LCT solver.
"""

import numpy as np
from scipy.linalg import eigh


class SpectralInverse:
    def __init__(self, operator, relative_ridge=1e-5):
        if relative_ridge < 0:
            raise ValueError('Regularization must be nonnegative.')
        gram = (operator.T @ operator).toarray()
        eigenvalues, eigenvectors = eigh(gram, check_finite=False)
        eigenvalues = np.maximum(eigenvalues, 0)
        maximum = float(eigenvalues[-1])
        if maximum <= 0:
            raise ValueError('The measurement operator is zero.')
        self.ridge = relative_ridge * maximum
        if self.ridge:
            inverse_eigenvalues = 1 / (eigenvalues + self.ridge)
        else:
            inverse_eigenvalues = np.zeros_like(eigenvalues)
            usable = eigenvalues > maximum * 1e-12
            inverse_eigenvalues[usable] = 1 / eigenvalues[usable]
        normal_inverse = (eigenvectors * inverse_eigenvalues) @ eigenvectors.T
        self.matrix = np.asarray(operator @ normal_inverse).T
        self.resolution = normal_inverse @ gram
        self.noise_trace_weights = np.sum(self.matrix**2, axis=0)
        self.singular_values = np.sqrt(eigenvalues[::-1])
        self.effective_dof = float(np.sum(eigenvalues * inverse_eigenvalues))

    def __call__(self, measurements):
        return np.asarray(measurements) @ self.matrix.T

    def noise_energy(self, input_variance):
        return np.asarray(input_variance) @ self.noise_trace_weights
