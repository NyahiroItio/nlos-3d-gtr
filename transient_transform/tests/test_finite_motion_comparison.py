"""Check finite-window motion selection with an independent analytic pulse."""

import unittest

import numpy as np

from transient_transform.finite_motion_comparison import (
    fit_fixed_gain, fit_free_gain)


class FiniteMotionSelectionTests(unittest.TestCase):
    def test_full_and_slope_select_shift_with_static_background(self):
        times = np.linspace(-.2, .2, 5)
        fast = np.linspace(-4., 4., 2001)
        speeds = np.array([.1, .4, .7])
        candidates = np.exp(-(
            fast[None, None, :]-speeds[:, None, None]*times[None, :, None]*4
        )**2/2)
        gain = 3.
        observation = gain*candidates[1]+7*np.cos(fast)[None]
        centered = observation-observation.mean(axis=0, keepdims=True)
        centered_bank = candidates-candidates.mean(axis=1, keepdims=True)
        weight = np.ones_like(fast)
        self.assertEqual(fit_fixed_gain(centered, centered_bank, weight,
                                        gain)[0], 1)
        winner, fitted_gain = fit_free_gain(centered, centered_bank, weight)
        self.assertEqual(winner, 1)
        self.assertAlmostEqual(fitted_gain, gain, places=10)

        slope_weights = times/np.dot(times, times)
        slope_bank = np.tensordot(candidates, slope_weights, axes=(1, 0))
        slope_data = np.tensordot(observation, slope_weights, axes=(0, 0))
        self.assertEqual(fit_fixed_gain(slope_data, slope_bank, weight,
                                        gain)[0], 1)
        winner, fitted_gain = fit_free_gain(slope_data, slope_bank, weight)
        self.assertEqual(winner, 1)
        self.assertAlmostEqual(fitted_gain, gain, places=10)


if __name__ == '__main__':
    unittest.main()
