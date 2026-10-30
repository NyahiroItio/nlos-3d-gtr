"""Numerical reimplementation of the pinned 2018 author LCT engineering path.

Reference: O'Toole, Lindell, Wetzstein, Nature 2018, cnlos_code.zip (author
project download). Numerical conventions are intentionally kept: c=3e8,
half-wall width, pairwise count summation, normalized nearest-layer PSF,
fine-grid weighted resampling and its transpose, and the absolute 1/SNR
Wiener penalty. This file does not copy or redistribute the upstream source.

Unlike lct.py (the earlier continuous-form prototype), the output is the
author's native [depth, y, x] reconstruction in author-code units. It is not
calibrated physical density and is not interpolated onto the old 640 voxels.
"""

from dataclasses import dataclass, asdict

import numpy as np
from scipy import fft, sparse
from scipy.io import loadmat


SCENES = {
    1: ('resolution_chart_40cm', 350), 2: ('resolution_chart_65cm', 700),
    3: ('dot_chart_40cm', 350), 4: ('dot_chart_65cm', 700),
    5: ('mannequin', 300), 6: ('exit_sign', 600), 7: ('s_u', 800),
    8: ('outdoor_s', 700), 9: ('diffuse_s', 100), 10: ('bunny', 0),
}


@dataclass(frozen=True)
class SceneSettings:
    scene: int
    name: str
    width_m: float
    n: int
    m: int
    bin_resolution_s: float
    range_m: float
    diffuse: bool
    snr: float
    z_trim: int
    z_offset: int
    downsample_steps: int = 2

    def metadata(self):
        return asdict(self)


def load_scene(source_dir, scene, downsample_steps=2):
    name, offset = SCENES[scene]
    original = loadmat(source_dir/f'data_{name}.mat')
    counts = np.asarray(original['rect_data'], dtype=np.float64)
    if counts.shape[0] != counts.shape[1] or counts.ndim != 3:
        raise ValueError('Original data must be [N,N,fast_time].')
    width = float(original['width'].item())
    dt = 8e-12 if scene == 10 else 4e-12
    range_m = counts.shape[2]*3e8*dt
    trim = 0 if scene == 10 else 600
    for _ in range(downsample_steps):
        if counts.shape[-1] % 2:
            raise ValueError('Pairwise temporal downsampling requires an even number of bins.')
        counts = counts[..., ::2]+counts[..., 1::2]
        dt *= 2
        trim, offset = int(np.floor(trim/2+.5)), int(np.floor(offset/2+.5))
    counts[..., :trim] = 0
    diffuse = scene in (9, 10)
    settings = SceneSettings(scene, name, width, counts.shape[0], counts.shape[2], dt,
                             range_m, diffuse, .08 if diffuse else .8, trim, offset,
                             downsample_steps)
    # MATLAB permute(rect_data,[3 2 1]), not [3 1 2].
    return np.transpose(counts, (2, 1, 0)).copy(), settings


def resampling_matrices(m):
    if m < 2 or m & (m-1):
        raise ValueError('Author resampling requires a power-of-two temporal dimension.')
    # Analytic aggregation of the author's M^2 fine samples, followed by log2(M)
    # pairwise averages. Each coarse row contains M consecutive fine samples.
    fine = np.arange(1, m*m+1, dtype=np.float64)
    row = np.arange(m*m)//m
    column = np.ceil(np.sqrt(fine)).astype(np.int64)-1
    values = 1/(m*np.sqrt(fine))
    forward = sparse.coo_matrix((values, (row, column)), shape=(m, m)).tocsr()
    return forward, forward.T.tocsr()


def define_psf(n, m, slope):
    lateral = np.linspace(-1., 1., 2*n)
    depth = np.linspace(0., 2., 2*m)
    paraboloid = (4*slope)**2*(lateral[:, None]**2+lateral[None, :]**2)
    distance = np.abs(depth[:, None, None]-paraboloid)
    psf = (distance == np.min(distance, axis=0)[None]).astype(np.float64)
    # MATLAB index U corresponds to Python U-1.
    psf /= psf[:, n-1, n-1].sum()
    psf /= np.linalg.norm(psf.ravel())
    return np.roll(psf, (n, n), axis=(1, 2))


class OriginalLCT:
    def __init__(self, n, m, width_m, range_m, diffuse=False, snr=.8,
                 backprojection=False):
        if n < 2 or width_m <= 0 or range_m <= 0 or snr <= 0:
            raise ValueError('Invalid original LCT geometry or regularization.')
        self.n, self.m = n, m
        self.width_m, self.range_m = width_m, range_m
        self.diffuse, self.snr = diffuse, snr
        self.mtx, self.mtxi = resampling_matrices(m)
        self.psf = define_psf(n, m, width_m/range_m)
        self.spectrum = fft.rfftn(self.psf)
        self.inverse_filter = (self.spectrum.conj() if backprojection else
                               self.spectrum.conj()/(np.abs(self.spectrum)**2+1/snr))
        self.weight = np.linspace(0., 1., m)[:, None, None]**(4 if diffuse else 2)
        self.fft_shape = (2*m, 2*n, 2*n)

    def __call__(self, data, return_stages=False):
        data = np.asarray(data, dtype=np.float64)
        if data.shape != (self.m, self.n, self.n):
            raise ValueError('Use the native [M,N,N] depth/y/x array, without extra ROI interpolation.')
        weighted = data*self.weight
        resampled = np.asarray(self.mtx @ weighted.reshape(self.m, -1)).reshape(data.shape)
        padded = np.zeros(self.fft_shape)
        padded[:self.m, :self.n, :self.n] = resampled
        filtered = fft.irfftn(fft.rfftn(padded)*self.inverse_filter, s=self.fft_shape)
        filtered = filtered[:self.m, :self.n, :self.n]
        signed = np.asarray(self.mtxi @ filtered.reshape(self.m, -1)).reshape(data.shape)
        if return_stages:
            return signed, dict(weighted_data=weighted, resampled_data=resampled,
                                filtered_data=filtered, signed_volume=signed)
        return signed

    def display_volume(self, signed_volume, settings, clip=True):
        """Exact author positivity, last-axis flip, depth crop and coordinates."""
        volume = np.maximum(signed_volume, 0) if clip else signed_volume
        count = int(np.floor(self.m*2*self.width_m/(self.range_m/2)+.5))
        end = min(count+settings.z_offset, self.m)
        volume = volume[settings.z_offset:end, :, ::-1]
        lateral = np.linspace(-self.width_m, self.width_m, self.n)
        depth = np.linspace(0., self.range_m/2, self.m)[settings.z_offset:end]
        return volume, lateral, depth
