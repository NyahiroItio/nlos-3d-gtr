import unittest

import numpy as np

from transient_transform.temporal_fit import fit_weights
from transient_transform.window_scan import (fit_two_coefficients, sample_two_coefficients,
                                             window_indices)


class WindowScanChecks(unittest.TestCase):
    def test_nested_scan_windows(self):
        windows = window_indices(7)
        self.assertEqual([len(item) for item in windows], [3, 5, 7])
        self.assertTrue(all(3 in item for item in windows))
        with self.assertRaises(ValueError):
            window_indices(4)

    def test_center_and_slope_with_nonunit_exposure_and_selected_frames(self):
        times = np.linspace(-.3, .3, 7)
        indices = window_indices(7)[1]
        weights = fit_weights(times[indices], degree=2)
        expected = np.array([8., -2., 4.])
        exposure = 7/len(indices)
        frames = {int(index): np.array([exposure*(expected[0]+times[index]*expected[1]
                                                 +times[index]**2*expected[2])])
                  for index in indices}
        np.testing.assert_allclose(fit_two_coefficients(frames, indices, weights, exposure)[:, 0],
                                   expected[:2], atol=1e-12)

    def test_seeded_nested_windows_use_the_same_overlapping_counts(self):
        means = np.full((7, 2, 2, 2), 20.)
        short = window_indices(7)[0]
        long = window_indices(7)[-1]
        short_weights = fit_weights(np.linspace(-.1, .1, 3), degree=2)
        actual = sample_two_coefficients(means, short, short_weights, 1., 9, 2, False)
        repeated = sample_two_coefficients(means, short, short_weights, 1., 9, 2, False)
        np.testing.assert_array_equal(actual, repeated)
        reconstructed = []
        for index in long:
            stream = np.random.default_rng(np.random.SeedSequence([9, 2, int(index), 0, 0]))
            reconstructed.append(stream.poisson(means[index]))
        expected = fit_two_coefficients(reconstructed, short, short_weights, 1.)
        np.testing.assert_array_equal(actual, expected)
