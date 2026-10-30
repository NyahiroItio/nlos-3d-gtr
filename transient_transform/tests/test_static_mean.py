import unittest

import numpy as np

from transient_transform.static_mean import StaticMeanEstimator


class StaticMeanChecks(unittest.TestCase):
    def test_unequal_exposures_use_pooled_photon_rate(self):
        estimator = StaticMeanEstimator()
        estimator.add([4, 0], exposure=2)
        estimator.add([0, 12], exposure=6)
        np.testing.assert_allclose(estimator.estimate(), [.5, 1.5])
        self.assertEqual(estimator.frame_count, 2)
        self.assertEqual(estimator.exposure_sum, 8)

    def test_independent_poisson_repeats_have_correct_mean_and_variance(self):
        rng = np.random.default_rng(317)
        expected = np.array([2., 8., 20.])
        exposures = [.5, 1., 2.]
        estimator = StaticMeanEstimator()
        for exposure in exposures:
            estimator.add(rng.poisson(exposure*expected, size=(20000, 3)), exposure)
        estimates = estimator.estimate()
        np.testing.assert_allclose(estimates.mean(axis=0), expected, rtol=.015)
        np.testing.assert_allclose(estimates.var(axis=0), expected/sum(exposures), rtol=.04)

    def test_inputs_are_not_aliased_or_modified(self):
        original = np.array([2., 4.])
        estimator = StaticMeanEstimator()
        estimator.add(original)
        original[:] = 0
        estimator.add([4., 2.])
        np.testing.assert_allclose(estimator.estimate(), [3., 3.])
        result = estimator.estimate()
        result[:] = 0
        np.testing.assert_allclose(estimator.estimate(), [3., 3.])

    def test_invalid_frames_do_not_change_accumulated_measurement(self):
        estimator = StaticMeanEstimator()
        with self.assertRaises(ValueError):
            estimator.estimate()
        estimator.add([1., 2.])
        for counts, exposure in [([1., 2.], 0), ([1., 2.], np.nan),
                                 ([-1., 2.], 1), ([np.nan, 1.], 1),
                                 ([[1., 2.]], 1), ([], 1)]:
            with self.subTest(counts=counts, exposure=exposure):
                with self.assertRaises(ValueError):
                    estimator.add(counts, exposure)
        self.assertEqual(estimator.frame_count, 1)
        np.testing.assert_allclose(estimator.estimate(), [1., 2.])
