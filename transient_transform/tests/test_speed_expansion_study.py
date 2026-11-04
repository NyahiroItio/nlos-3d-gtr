"""Check the analytic shift error against direct pulse integration."""

import unittest

import numpy as np

from transient_transform.speed_expansion_study import gaussian_shift_relative_error


class GaussianShiftErrorTests(unittest.TestCase):
    def test_matches_continuous_pulse_quadrature(self):
        times = np.linspace(-1., 1., 5)
        fast = np.linspace(-16., 16., 32001)
        for eta in (.2, 1., 2., 4.):
            samples = np.exp(-(fast[None, :] - eta*times[:, None])**2/2)
            estimated = np.tensordot(times/np.dot(times, times), samples,
                                     axes=(0, 0))
            exact = eta*fast*np.exp(-fast**2/2)
            measured = np.linalg.norm(estimated-exact)/np.linalg.norm(exact)
            self.assertAlmostEqual(measured,
                                   gaussian_shift_relative_error(eta, times),
                                   places=6)

    def test_quadratic_small_shift_limit(self):
        times = np.linspace(-1., 1., 5)
        coefficient = np.sqrt(15)/12*np.sum(times**4)/np.sum(times**2)
        self.assertAlmostEqual(
            gaussian_shift_relative_error(.05, times)/.05**2,
            coefficient, delta=.0002)


if __name__ == '__main__':
    unittest.main()
