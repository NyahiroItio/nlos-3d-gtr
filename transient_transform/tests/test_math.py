import unittest

import numpy as np
from scipy import sparse

from transient_transform.operator import C, Grid, build_operator, compact_blob
from transient_transform.spectral import SpectralInverse
from transient_transform.temporal import coefficient_variance, fit_transients, fit_weights


class MathematicalChecks(unittest.TestCase):
    def test_projector_has_correct_path_time_and_bin_integral(self):
        grid = Grid(np.array([0., .01]), np.array([0., .01]), np.array([.6, .61]))
        edges = np.linspace(0, 8e-9, 801)
        operator = build_operator(grid, np.array([[0., 0., 0.]]), edges, 30e-12)
        response = operator[:, 0].toarray().ravel()
        expected_total = grid.voxel_volume / .6**4
        self.assertAlmostEqual(response.sum() / expected_total, 1., places=10)
        center = (edges[1:] + edges[:-1]) / 2
        self.assertLess(abs(response @ center / response.sum() - 2*.6/C), 1e-14)
        self.assertTrue(np.all(operator.data >= 0))

    def test_signed_derivative_and_absolute_intercept(self):
        times = np.array([-.3, -.1, .1, .3])
        reference = np.array([2., 3.])
        slope = np.array([-1., 2.])
        frames = reference + times[:, None]*slope
        estimate = fit_transients(frames + .7, np.array([.7, .7]), fit_weights(times))
        np.testing.assert_allclose(estimate, np.stack((reference, slope)), atol=1e-13)

    def test_exact_spectral_inverse_and_temporal_commutation(self):
        operator = sparse.csr_matrix([[1., .2], [.3, 1.], [.4, .5]])
        inverse = SpectralInverse(operator, 0)
        frames = np.array([[1., 2.], [2., 1.], [3., 0.]])
        measurements = np.asarray(operator @ frames.T).T
        weights = fit_weights([-1, 0, 1])
        np.testing.assert_allclose(inverse(measurements), frames, atol=1e-12)
        np.testing.assert_allclose(inverse(weights @ measurements), weights @ inverse(measurements), atol=1e-12)
        self.assertLess(inverse(weights @ measurements)[1, 1], 0)

    def test_shared_background_covariance_and_derivative_cancellation(self):
        rng = np.random.default_rng(13)
        times = np.array([-1., 0., 1.])
        weights = fit_weights(times)
        total = np.array([[12., 14.], [13., 13.], [14., 12.]])
        background, repeats, trials = 5., 4, 40000
        draw = rng.poisson(total, size=(trials, 3, 2))
        bg = rng.poisson(repeats*background, size=(trials, 2)) / repeats
        coefficients = np.einsum('cf,bfm->bcm', weights, draw-bg[:, None])
        predicted = coefficient_variance(total, weights, background/repeats)
        np.testing.assert_allclose(coefficients.var(axis=0), predicted, rtol=.035)
        no_calibration_noise = coefficient_variance(total, weights, 0)
        np.testing.assert_allclose(predicted[1], no_calibration_noise[1], atol=1e-13)
        self.assertTrue(np.all(predicted[0] > no_calibration_noise[0]))

    def test_constant_velocity_blob_has_nonzero_density_acceleration(self):
        points = np.array([[.02, .01, .6], [-.03, -.01, .58]])
        center = np.array([0., 0., .6])
        radii = (.1, .12, .1)
        velocity = np.array([.08, 0., -.03])
        h = 1e-4
        value, gradient = compact_blob(points, center, radii)
        plus = compact_blob(points, center+h*velocity, radii)[0]
        minus = compact_blob(points, center-h*velocity, radii)[0]
        np.testing.assert_allclose((plus-minus)/(2*h), -gradient @ velocity, rtol=1e-6)
        self.assertGreater(np.linalg.norm((plus-2*value+minus)/h**2), .1)

    def test_window_bias_for_curved_density_is_explicit(self):
        times = np.linspace(-.3, .3, 7)
        values = 2 + times + 4*times**2
        coefficients = fit_weights(times) @ values
        self.assertAlmostEqual(coefficients[1], 1.)
        self.assertAlmostEqual(coefficients[0] - 2, 4*np.mean(times**2))


if __name__ == '__main__':
    unittest.main()
