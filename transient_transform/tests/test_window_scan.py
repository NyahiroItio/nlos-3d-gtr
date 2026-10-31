import unittest

import numpy as np

from transient_transform.temporal_fit import fit_weights
from transient_transform.window_scan import fit_two_coefficients, window_indices


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
