"""Signed linear light-cone transform, implemented from its integral identity.

For a time-density transient with delta(s-2r/c), let v=(cs/2)^2,
u=z^2, q=rho/(2z), and g(v)=v^(3/2)*tau(2*sqrt(v)/c)/(gain*c).
Then g=h*q, h=delta(v-u-dx^2-dy^2). Here H is applied/inverted
with zero-padded 3-D FFTs, and R_s / R_z are fixed linear interpolations.

The implementation is independent of the authors' MATLAB code. It has no
positivity clamp, magnitude extraction, input normalization, or fitted gain.
Finite aperture, time resampling, IRF, and quadrature remain sources of bias.
"""

import numpy as np
from scipy import fft

from .operator import C


def interpolation_matrix(source, target):
    """Piecewise-linear interpolation, zero outside the source sample centers."""
    source, target = np.asarray(source), np.asarray(target)
    if len(source) < 2 or np.any(np.diff(source) <= 0):
        raise ValueError('Interpolation coordinates must increase.')
    matrix = np.zeros((len(target), len(source)))
    right = np.searchsorted(source, target, side='right').clip(1, len(source)-1)
    fraction = (target-source[right-1]) / (source[right]-source[right-1])
    valid = (target >= source[0]) & (target <= source[-1])
    rows = np.flatnonzero(valid)
    matrix[rows, right[valid]-1] = 1-fraction[valid]
    matrix[rows, right[valid]] = fraction[valid]
    return matrix


class LCTInverse:
    def __init__(self, grid, wall, fast_edges_s, gain=1., relative_ridge=1e-4,
                 transformed_bins=256):
        if not np.isfinite(gain) or gain <= 0 or not np.isfinite(relative_ridge) or relative_ridge <= 0:
            raise ValueError('Gain and LCT regularization must be finite and positive.')
        if transformed_bins < 8:
            raise ValueError('Use at least eight squared-range bins.')
        wall = np.asarray(wall, dtype=float)
        if wall.ndim != 2 or wall.shape[1] != 3 or not np.allclose(wall[:, 2], 0):
            raise ValueError('LCT requires confocal wall points on z=0.')
        wx, wy = np.unique(wall[:, 0]), np.unique(wall[:, 1])
        for axis in (wx, wy):
            if len(axis) < 2 or not np.allclose(np.diff(axis), axis[1]-axis[0]):
                raise ValueError('LCT requires a complete uniform rectangular wall grid.')
        self.nx, self.ny = len(wx), len(wy)
        ix, iy = np.searchsorted(wx, wall[:, 0]), np.searchsorted(wy, wall[:, 1])
        self.wall_indices = ix*self.ny+iy
        if len(wall) != self.nx*self.ny or len(np.unique(self.wall_indices)) != len(wall):
            raise ValueError('LCT requires every wall-grid position exactly once.')
        edges = np.asarray(fast_edges_s)
        if edges.ndim != 1 or edges[0] < 0 or np.any(np.diff(edges) <= 0):
            raise ValueError('Fast-time edges must be increasing and nonnegative.')
        self.nfast = len(edges)-1
        self.nu = int(transformed_bins)
        self.input_size = len(wall)*self.nfast
        self.output_size = int(np.prod(grid.shape))
        self.output_shape = grid.shape
        self.u = np.linspace(0, (C*edges[-1]/2)**2, self.nu)
        self.du = self.u[1]-self.u[0]
        dx, dy = wx[1]-wx[0], wy[1]-wy[0]
        self.fft_shape = (fft.next_fast_len(2*self.nx), fft.next_fast_len(2*self.ny),
                          fft.next_fast_len(2*self.nu))

        fast_centers = (edges[1:]+edges[:-1])/2
        squared_range_times = 2*np.sqrt(self.u)/C
        self.time_map = interpolation_matrix(fast_centers, squared_range_times)
        # The simulator supplies bin counts, not a density per second.
        self.time_map *= (self.u**1.5 / (gain*C))[:, None] / np.diff(edges)[None, :]
        self.x_map = interpolation_matrix(wx, grid.x)
        self.y_map = interpolation_matrix(wy, grid.y)
        self.z_map = interpolation_matrix(self.u, grid.z**2)
        self.depth_scale = 2*grid.z
        if any(np.any(matrix.sum(axis=1) == 0) for matrix in (self.x_map, self.y_map, self.z_map)):
            raise ValueError('Requested output grid must lie inside the native LCT volume.')

        # h(dx,dy,v) is deposited on v=(dx^2+dy^2), with lateral quadrature
        # area. The delta's 1/du cancels the du in discrete convolution.
        kernel = np.zeros(self.fft_shape)
        for ox in range(1-self.nx, self.nx):
            for oy in range(1-self.ny, self.ny):
                delay = ((ox*dx)**2+(oy*dy)**2) / self.du
                lower = int(np.floor(delay))
                fraction = delay-lower
                for depth, weight in ((lower, 1-fraction), (lower+1, fraction)):
                    if depth < self.nu:
                        kernel[ox % self.fft_shape[0], oy % self.fft_shape[1], depth] += dx*dy*weight
        spectrum = fft.rfftn(kernel)
        self.forward_spectrum = spectrum
        power = np.abs(spectrum)**2
        self.ridge = float(relative_ridge*power.max())
        self.relative_ridge = relative_ridge
        self.inverse_filter = spectrum.conj() / (power+self.ridge)
        self._noise_trace_weights = None

    def _convolve(self, native, adjoint=False):
        padded = np.zeros((len(native),)+self.fft_shape)
        padded[:, :self.nx, :self.ny, :self.nu] = native
        multiplier = self.inverse_filter.conj() if adjoint else self.inverse_filter
        result = fft.irfftn(fft.rfftn(padded, axes=(-3, -2, -1))*multiplier,
                           s=self.fft_shape, axes=(-3, -2, -1))
        return result[:, :self.nx, :self.ny, :self.nu]

    def __call__(self, measurements):
        measurements = np.asarray(measurements, dtype=float)
        if measurements.shape[-1] != self.input_size:
            raise ValueError('Transient size does not match the LCT wall/time grid.')
        batch_shape = measurements.shape[:-1]
        raw = measurements.reshape(-1, self.nx*self.ny, self.nfast) @ self.time_map.T
        ordered = np.empty_like(raw)
        ordered[:, self.wall_indices, :] = raw
        q = self._convolve(ordered.reshape(-1, self.nx, self.ny, self.nu))
        q = np.einsum('ix,bxyz->biyz', self.x_map, q, optimize=True)
        q = np.einsum('jy,biyz->bijz', self.y_map, q, optimize=True)
        rho = (q @ self.z_map.T)*self.depth_scale
        return rho.reshape(batch_shape+(self.output_size,))

    def adjoint(self, volumes):
        """Exact adjoint of this discrete linear implementation, for covariance."""
        volumes = np.asarray(volumes, dtype=float)
        if volumes.shape[-1] != self.output_size:
            raise ValueError('Volume size does not match the LCT output grid.')
        batch_shape = volumes.shape[:-1]
        rho = volumes.reshape((-1,)+self.output_shape)*self.depth_scale
        native = rho @ self.z_map
        native = np.einsum('jy,bijz->biyz', self.y_map, native, optimize=True)
        native = np.einsum('ix,biyz->bxyz', self.x_map, native, optimize=True)
        transformed = self._convolve(native, adjoint=True)
        ordered = transformed.reshape(-1, self.nx*self.ny, self.nu) @ self.time_map
        raw = ordered[:, self.wall_indices, :]
        return raw.reshape(batch_shape+(self.input_size,))

    def noise_energy(self, input_variance):
        """Exact tr(L diag(v) L^T); diagnostic precompute, not reconstruction.

        Accumulate squared adjoint responses in blocks without storing a dense
        L or constructing A. This cost scales with output size and is reported
        separately from the FFT reconstruction cost.
        """
        if self._noise_trace_weights is None:
            self._noise_trace_weights = np.zeros(self.input_size)
            for first in range(0, self.output_size, 8):
                indices = np.arange(first, min(first+8, self.output_size))
                basis = np.zeros((len(indices), self.output_size))
                basis[np.arange(len(indices)), indices] = 1
                response = self.adjoint(basis)
                self._noise_trace_weights += np.sum(response**2, axis=0)
        return np.asarray(input_variance) @ self._noise_trace_weights
