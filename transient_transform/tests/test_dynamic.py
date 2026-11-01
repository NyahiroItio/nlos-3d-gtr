import unittest

import numpy as np

from transient_transform.motion_simulation import C, confocal_response
from transient_transform.temporal_fit import (AffineTransientEstimator,
    PolynomialTransientEstimator, fit_weights, centered_difference_weights)


class TemporalChecks(unittest.TestCase):
    def test_reference_and_signed_slope_with_irregular_times_and_exposures(self):
        times = np.array([-.3, -.1, .2, .5])
        exposures = [.5, 1., 2., .8]
        center = np.array([10., 8.])
        slope = np.array([-2., 3.])
        estimator = AffineTransientEstimator(times)
        for time, exposure in zip(times, exposures):
            estimator.add(exposure*(center+time*slope), exposure)
        np.testing.assert_allclose(estimator.estimate(), np.stack([center, slope]), atol=1e-12)

    def test_poisson_coefficient_covariance_matches_time_weights(self):
        times = np.array([-.3, -.1, .2, .5])
        exposures = np.array([.5, 1., 2., .8])
        center = np.array([10., 8.])
        slope = np.array([-2., 3.])
        means = center+times[:, None]*slope
        weights = fit_weights(times)
        estimator = AffineTransientEstimator(times)
        rng = np.random.default_rng(91)
        for mean, exposure in zip(means, exposures):
            estimator.add(rng.poisson(exposure*mean, size=(30000, 2)), exposure)
        fitted = estimator.estimate()
        predicted = np.einsum('in,jn,nk->ijk', weights, weights, means/exposures[:, None])
        for point in range(2):
            covariance = np.cov(fitted[:, :, point])
            np.testing.assert_allclose(np.diag(covariance), np.diag(predicted[:, :, point]), rtol=.035)
            self.assertLess(abs(covariance[0, 1]-predicted[0, 1, point]),
                            .04*np.sqrt(predicted[0, 0, point]*predicted[1, 1, point]))

    def test_centered_window_curvature_bias_and_derivative_noise_factor(self):
        times = np.linspace(-.3, .3, 7)
        values = 2+times+4*times**2
        weights = fit_weights(times)
        coefficients = weights@values
        self.assertAlmostEqual(coefficients[0]-2, 4*np.mean(times**2))
        self.assertAlmostEqual(coefficients[1], 1.)
        adjacent = centered_difference_weights(times)
        self.assertAlmostEqual(np.sum(weights[1]**2)/np.sum(adjacent[1]**2), 1/14)

    def test_quadratic_fit_removes_curvature_and_retains_centered_slope(self):
        times = np.linspace(-.3, .3, 7)
        coefficients = np.array([[8., 5.], [2., -1.], [4., .5]])
        exposures = np.array([.5, 1., 2., 1., 3., 1., .8])
        estimator = PolynomialTransientEstimator(times)
        for time, exposure in zip(times, exposures):
            estimator.add(exposure*(coefficients[0]+time*coefficients[1]
                                    +time**2*coefficients[2]), exposure)
        np.testing.assert_allclose(estimator.estimate(), coefficients, atol=1e-12)
        quadratic = fit_weights(times, degree=2)
        linear = fit_weights(times)
        np.testing.assert_allclose(quadratic[1], linear[1], atol=1e-14)
        self.assertAlmostEqual(np.sum(quadratic[0]**2)/np.sum(linear[0]**2), 7/3)
        arbitrary_frames = np.random.default_rng(11).normal(size=(len(times), 4))
        design = np.column_stack((np.ones(len(times)), times, times**2))
        np.testing.assert_allclose(linear@arbitrary_frames,
                                   (linear@design)@(quadratic@arbitrary_frames), atol=1e-13)

    def test_quadratic_fit_requires_three_distinct_times(self):
        for times in ([0, 0, 1], [0, 1], [0, 1, np.nan]):
            with self.assertRaises(ValueError):
                fit_weights(times, degree=2)
        with self.assertRaises(ValueError):
            fit_weights([0, 1, 2], degree=3)

    def test_incomplete_degenerate_and_invalid_inputs_are_rejected(self):
        for times in ([0, 0], [0, np.nan], [0]):
            with self.assertRaises(ValueError):
                fit_weights(times)
        estimator = AffineTransientEstimator([-1, 0, 1])
        with self.assertRaises(ValueError):
            estimator.estimate()
        estimator.add([1., 2.])
        for counts, exposure in [([-1., 2.], 1), ([1., 2.], 0), ([[1., 2.]], 1)]:
            with self.assertRaises(ValueError):
                estimator.add(counts, exposure)
        self.assertEqual(estimator.frame_count, 1)
        estimator.add([2., 2.])
        estimator.add([3., 2.])
        np.testing.assert_allclose(estimator.estimate(), [[2., 2.], [1., 0.]], atol=1e-12)
        with self.assertRaises(ValueError):
            estimator.add([4., 2.])


class MotionPhysicsChecks(unittest.TestCase):
    def test_integrated_photons_and_flight_time_of_single_scatterer(self):
        points = np.array([[0., 0., .6]])
        wall = np.array([[0., 0., 0.]])
        edges = np.arange(1001)*8e-12
        response, derivative = confocal_response(points, np.array([.01]), wall, edges)
        self.assertAlmostEqual(response.sum()/(.01/.6**4), 1., places=7)
        centers = (edges[1:]+edges[:-1])/2
        self.assertLess(abs(response[0]@centers/response.sum()-2*.6/C), 1e-14)
        np.testing.assert_array_equal(derivative, 0)

    def test_analytic_motion_derivative_matches_independent_finite_difference(self):
        points = np.array([[-.03, .02, .6], [.04, -.01, .66]])
        weights = np.array([.01, .02])
        wall = np.array([[0., 0., 0.], [.1, .1, 0.]])
        edges = np.arange(1001)*8e-12
        velocity = np.array([.08, .01, -.02])
        response, derivative = confocal_response(points, weights, wall, edges, velocity)
        step = 1e-5
        plus, _ = confocal_response(points+step*velocity, weights, wall, edges)
        minus, _ = confocal_response(points-step*velocity, weights, wall, edges)
        finite_difference = (plus-minus)/(2*step)
        self.assertLess(np.linalg.norm(finite_difference-derivative)/np.linalg.norm(derivative), 1e-5)
        self.assertTrue(np.any(derivative > 0) and np.any(derivative < 0))

    def test_point_dependent_expansion_derivative_matches_finite_difference(self):
        points = np.array([[-.04, .02, .60], [.05, -.01, .64]])
        weights = np.array([.01, .02])
        wall = np.array([[0., 0., 0.], [.08, -.03, 0.]])
        edges = np.arange(1001)*8e-12
        center = np.array([0., 0., .62])
        velocities = .5*(points-center)
        _, derivative = confocal_response(points, weights, wall, edges, velocities)
        step = 1e-5
        plus, _ = confocal_response(points+step*velocities, weights, wall, edges)
        minus, _ = confocal_response(points-step*velocities, weights, wall, edges)
        finite_difference = (plus-minus)/(2*step)
        self.assertLess(np.linalg.norm(finite_difference-derivative)/np.linalg.norm(derivative), 1e-5)
        with self.assertRaises(ValueError):
            confocal_response(points, weights, wall, edges, np.zeros((3, 3)))
