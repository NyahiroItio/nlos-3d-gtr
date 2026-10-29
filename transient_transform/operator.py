"""Confocal volumetric-albedo operator with physical ToF and integrated bins.

Slow acquisition time is t; fast photon flight time is s, in seconds.
The simplified LCT-style model is linear and has no scene-dependent occlusion.
"""

from dataclasses import dataclass

import numpy as np
from scipy import sparse
from scipy.special import ndtr

C = 299_792_458.0


@dataclass(frozen=True)
class Grid:
    x: np.ndarray
    y: np.ndarray
    z: np.ndarray

    @property
    def shape(self):
        return (len(self.x), len(self.y), len(self.z))

    @property
    def points(self):
        return np.stack(np.meshgrid(self.x, self.y, self.z, indexing='ij'), -1).reshape(-1, 3)

    @property
    def voxel_volume(self):
        return float(np.prod([axis[1] - axis[0] for axis in (self.x, self.y, self.z)]))


def make_geometry(nx=8, ny=8, nz=10, wall_res=12, fast_bins=128):
    grid = Grid(np.linspace(-0.18, 0.18, nx), np.linspace(-0.18, 0.18, ny),
                np.linspace(0.45, 0.85, nz))
    wall_axis = np.linspace(-0.45, 0.45, wall_res)
    wx, wy = np.meshgrid(wall_axis, wall_axis, indexing='ij')
    wall = np.column_stack((wx.ravel(), wy.ravel(), np.zeros(wx.size)))
    edges = np.linspace(1e-9, 8e-9, fast_bins + 1)
    return grid, wall, edges


def build_operator(grid, wall, fast_edges_s, irf_sigma_s=60e-12):
    """A[(wall,fast_bin),voxel] = dV/r^4 * integrated Gaussian IRF.

    Spatial integration uses voxel-center quadrature on a uniform XYZ grid.
    IRF is an instrument response, not a spatial Gaussian primitive. The
    calibrated external laser-to-wall/detector paths are omitted.
    """
    edges = np.asarray(fast_edges_s, dtype=float)
    if irf_sigma_s <= 0 or np.any(np.diff(edges) <= 0):
        raise ValueError('IRF sigma and time-bin widths must be positive.')
    if min(grid.shape) < 2 or np.any(grid.z <= 0):
        raise ValueError('Use at least two samples per axis and z > 0.')
    for axis in (grid.x, grid.y, grid.z):
        if np.any(np.diff(axis) <= 0) or not np.allclose(np.diff(axis), axis[1] - axis[0]):
            raise ValueError('This projector requires an increasing uniform XYZ grid.')
    points = grid.points
    blocks = []
    for p in wall:
        r = np.linalg.norm(points - p, axis=1)
        arrival = 2 * r / C
        probability = np.diff(ndtr((edges[:, None] - arrival) / irf_sigma_s), axis=0)
        probability[probability < 1e-12] = 0
        blocks.append(sparse.csr_matrix(probability * (grid.voxel_volume / r**4)))
    return sparse.vstack(blocks, format='csr')


def compact_blob(points, center, radii):
    """Compact polynomial object and its XYZ gradient; no Gaussian scene model."""
    delta = points - np.asarray(center)
    radii = np.asarray(radii)
    interior = np.maximum(1 - np.sum((delta / radii)**2, axis=1), 0)
    value = interior**3
    gradient = -6 * interior[:, None]**2 * delta / radii**2
    return value, gradient


def make_scene(grid, times, case):
    """Return exact frames, reference density, and instantaneous reference slope.

    linear_density is a local density model, not constant material velocity.
    translation uses constant material velocity and generally rho_tt != 0.
    """
    points = grid.points
    center_a = np.array([-0.065, -0.025, 0.59])
    center_b = np.array([0.065, 0.035, 0.72])
    radii_a = (0.115, 0.13, 0.105)
    radii_b = (0.105, 0.12, 0.10)
    a, ga = compact_blob(points, center_a, radii_a)
    b, gb = compact_blob(points, center_b, radii_b)
    reference = 0.65 * (a + b)
    if case == 'linear_density':
        derivative = 0.9 * (b - a)
        frames = reference + np.asarray(times)[:, None] * derivative
    elif case == 'translation':
        velocity = np.array([0.08, 0.0, -0.03])  # m/s
        derivative = -0.65 * ((ga + gb) @ velocity)
        frames = np.stack([
            0.65 * (compact_blob(points, center_a + velocity*t, radii_a)[0]
                    + compact_blob(points, center_b + velocity*t, radii_b)[0])
            for t in times
        ])
    else:
        raise ValueError(f'Unknown case: {case}')
    if frames.min() < -1e-12:
        raise ValueError('The requested window produces a negative true density.')
    return frames, reference, derivative
