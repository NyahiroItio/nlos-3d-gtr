import unittest

import numpy as np
from scipy import fft
from threadpoolctl import threadpool_limits

from transient_transform.lct import LCTInverse, interpolation_matrix
from transient_transform.operator import Grid, build_operator, make_geometry
from transient_transform.temporal import fit_weights


class LCTChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = threadpool_limits(2)
        cls.grid, cls.wall, cls.edges = make_geometry(4, 4, 5, wall_res=6, fast_bins=64)
        cls.inverse = LCTInverse(cls.grid, cls.wall, cls.edges, gain=3., transformed_bins=64)

    @classmethod
    def tearDownClass(cls):
        cls.threads.restore_original_limits()

    def test_signed_linearity_and_time_fit_commutation(self):
        rng = np.random.default_rng(31)
        data = rng.normal(size=(5, self.inverse.input_size))
        weights = fit_weights(np.linspace(-.2, .2, 5))
        np.testing.assert_allclose(self.inverse(-data), -self.inverse(data), atol=1e-12)
        np.testing.assert_allclose(self.inverse(weights @ data), weights @ self.inverse(data), atol=1e-12)

    def test_exact_adjoint_with_nonmatching_output_and_wall_grids(self):
        rng = np.random.default_rng(52)
        measurement = rng.normal(size=(2, self.inverse.input_size))
        volume = rng.normal(size=(2, self.inverse.output_size))
        left = np.sum(self.inverse(measurement)*volume)
        right = np.sum(measurement*self.inverse.adjoint(volume))
        self.assertAlmostEqual(left, right, places=11)

    def test_reordered_wall_points_preserve_result(self):
        rng = np.random.default_rng(9)
        order = rng.permutation(len(self.wall))
        other = LCTInverse(self.grid, self.wall[order], self.edges, gain=3., transformed_bins=64)
        data = rng.normal(size=(len(self.wall), len(self.edges)-1))
        np.testing.assert_allclose(other(data[order].ravel()), self.inverse(data.ravel()), atol=1e-12)

    def test_paraboloid_delay_and_lateral_quadrature(self):
        inverse = self.inverse
        kernel = fft.irfftn(inverse.forward_spectrum, s=inverse.fft_shape)
        dx = np.unique(self.wall[:, 0])[1]-np.unique(self.wall[:, 0])[0]
        dy = np.unique(self.wall[:, 1])[1]-np.unique(self.wall[:, 1])[0]
        response = kernel[1, 0, :inverse.nu]
        self.assertAlmostEqual(response.sum(), dx*dy, places=12)
        self.assertAlmostEqual(np.dot(response, inverse.u)/response.sum(), dx**2, places=12)
        self.assertAlmostEqual(kernel[0, 0, 0], dx*dy, places=12)

    def test_full_padded_convolution_roundtrip_has_correct_phase_and_scale(self):
        inverse = LCTInverse(self.grid, self.wall, self.edges, relative_ridge=1e-10,
                             transformed_bins=64)
        q = np.zeros(inverse.fft_shape)
        q[2, 3, 15] = 1.0
        data = fft.irfftn(fft.rfftn(q)*inverse.forward_spectrum, s=inverse.fft_shape)
        recovered = fft.irfftn(fft.rfftn(data)*inverse.inverse_filter, s=inverse.fft_shape)
        self.assertEqual(np.unravel_index(recovered.argmax(), recovered.shape), (2, 3, 15))
        # This discrete kernel has exact spectral zeros. Only its observable
        # subspace can be recovered, even for full padded noiseless data.
        kernel = fft.irfftn(inverse.forward_spectrum, s=inverse.fft_shape)
        spectrum = fft.fftn(kernel)
        observable = np.abs(spectrum) > np.abs(spectrum).max()*1e-8
        projected = fft.ifftn(fft.fftn(q)*observable).real
        self.assertLess(np.linalg.norm(recovered-projected), 1e-4)

    def test_absolute_exposure_gain_is_removed(self):
        rng = np.random.default_rng(18)
        data = rng.normal(size=self.inverse.input_size)
        other = LCTInverse(self.grid, self.wall, self.edges, gain=15., transformed_bins=64)
        np.testing.assert_allclose(other(5*data), self.inverse(data), atol=1e-12)

    def test_covariance_trace_against_explicit_small_linear_map(self):
        # The reconstruction itself uses FFTs; a tiny explicit map is assembled
        # only here as an independent check of squared-adjoint covariance.
        grid = Grid(np.array([-.05, .05]), np.array([-.05, .05]), np.array([.5, .7]))
        wall = np.array([[x, y, 0.] for x in (-.2, .2) for y in (-.2, .2)])
        inverse = LCTInverse(grid, wall, np.linspace(0, 8e-9, 17), transformed_bins=16)
        explicit = inverse(np.eye(inverse.input_size)).T
        variance = np.linspace(.2, 2., inverse.input_size)
        prediction = np.sum(explicit**2*variance)
        self.assertAlmostEqual(inverse.noise_energy(variance), prediction, places=9)

    def test_rejects_missing_or_nonplanar_measurements(self):
        with self.assertRaises(ValueError):
            LCTInverse(self.grid, self.wall[:-1], self.edges)
        shifted = self.wall.copy()
        shifted[:, 2] = .1
        with self.assertRaises(ValueError):
            LCTInverse(self.grid, shifted, self.edges)

    def test_independent_physical_projector_localizes_three_points(self):
        grid, wall, edges = make_geometry()
        physical = build_operator(grid, wall, edges)
        inverse = LCTInverse(grid, wall, edges)
        indices = [np.ravel_multi_index(point, grid.shape)
                   for point in ((2, 3, 3), (5, 4, 6), (4, 2, 5))]
        reconstructed = inverse(physical[:, indices].toarray().T)
        for index, volume in zip(indices, reconstructed):
            self.assertEqual(int(volume.argmax()), index)


if __name__ == '__main__':
    unittest.main()
