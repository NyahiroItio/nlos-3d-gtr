"""Independent confocal point-quadrature simulator for a translating thin glyph.

Single scattering, isotropic r^-p response and bin-integrated Gaussian IRF.
The positive surface quadrature moves with constant material velocity. Its
Eulerian density is generally not affine in slow time. No LCT kernel is used.
"""

import numpy as np
from scipy.special import ndtr
from PIL import Image, ImageDraw, ImageFont
from matplotlib import font_manager


C = 3e8


def glyph_points(resolution=48, width=.30, depth=.65):
    font_path = font_manager.findfont(font_manager.FontProperties(family='DejaVu Sans', weight='bold'))
    image = Image.new('L', (resolution, resolution), 0)
    draw = ImageDraw.Draw(image)
    font = ImageFont.truetype(font_path, int(resolution*.95))
    bounds = draw.textbbox((0, 0), 'S', font=font)
    position = ((resolution-(bounds[2]-bounds[0]))/2-bounds[0],
                (resolution-(bounds[3]-bounds[1]))/2-bounds[1])
    draw.text(position, 'S', fill=255, font=font)
    strength = np.asarray(image, dtype=float)/255
    row, col = np.nonzero(strength > 0)
    points = np.column_stack(((col+.5-resolution/2)*width/resolution,
                              (row+.5-resolution/2)*width/resolution,
                              np.full(len(row), depth)))
    area_weights = strength[row, col]*(width/resolution)**2
    return points, area_weights


def confocal_response(points, weights, wall, fast_edges, velocity=None,
                      irf_fwhm_s=60e-12, power=4, wall_batch=128):
    """Return expected bin counts and analytic slow-time derivative before gain."""
    points, weights, wall = map(np.asarray, (points, weights, wall))
    edges = np.asarray(fast_edges)
    sigma = irf_fwhm_s/(2*np.sqrt(2*np.log(2)))
    if sigma <= 0 or np.any(np.diff(edges) <= 0) or not np.allclose(np.diff(edges), edges[1]-edges[0], rtol=1e-10, atol=0):
        raise ValueError('Use positive IRF width and uniform increasing fast-time edges.')
    if np.any(weights < 0) or len(points) != len(weights):
        raise ValueError('Quadrature weights must be nonnegative and match the points.')
    velocity = np.zeros(3) if velocity is None else np.asarray(velocity)
    bins = len(edges)-1
    dt = edges[1]-edges[0]
    radius = int(np.ceil(6*sigma/dt))+1
    data = np.zeros((len(wall), bins))
    derivative = np.zeros_like(data)
    normalizer = 1/np.sqrt(2*np.pi)
    need_rate = bool(np.any(velocity != 0))
    for start in range(0, len(wall), wall_batch):
        end = min(start+wall_batch, len(wall))
        delta = points[None]-wall[start:end, None]
        distance = np.linalg.norm(delta, axis=-1)
        if np.any(distance <= 0):
            raise ValueError('Scatterers must be separated from the measurement wall.')
        radial_speed = np.einsum('wpi,i->wp', delta, velocity)/distance if need_rate else 0.
        arrival = 2*distance/C
        arrival_rate = 2*radial_speed/C
        amplitude = weights[None]/distance**power
        center = np.floor((arrival-edges[0])/dt).astype(np.int64)
        offsets = np.arange(end-start)[:, None]*bins
        for offset in range(-radius, radius+1):
            index = center+offset
            valid = (index >= 0)&(index < bins)
            low = (edges[0]+index*dt-arrival)/sigma
            high = low+dt/sigma
            probability = np.maximum(ndtr(high)-ndtr(low), 0.)
            values = amplitude*probability
            if need_rate:
                rate = amplitude*((-power*radial_speed/distance)*probability+
                                  arrival_rate/sigma*normalizer*(np.exp(-low**2/2)-np.exp(-high**2/2)))
            flat = (offsets+index)[valid]
            data[start:end] += np.bincount(flat, weights=values[valid],
                                          minlength=(end-start)*bins).reshape(end-start, bins)
            if need_rate:
                derivative[start:end] += np.bincount(flat, weights=rate[valid],
                                                    minlength=(end-start)*bins).reshape(end-start, bins)
    return data, derivative



def translation_sequence(times, settings, cache_path):
    """Generate/cache a physical moving scene; one fixed exposure gain for all times."""
    import hashlib
    import json
    from pathlib import Path

    cache_path = Path(cache_path)
    velocity = np.array([.08, 0., 0.])
    points, weights = glyph_points()
    description = {'geometry': settings.metadata(), 'times': np.asarray(times).tolist(),
                   'velocity_m_per_s': velocity.tolist(), 'irf_fwhm_s': 60e-12,
                   'radiometric_power': 4, 'reference_expected_photons': 4500000.,
                   'quadrature_points': len(points), 'source': 'Thin S surface / positive moving quadrature',
                   'wall_x_descending': True,
                   'generator_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    key = json.dumps(description, sort_keys=True)
    if cache_path.exists():
        with np.load(cache_path, allow_pickle=False) as cached:
            if str(cached['key']) == key:
                print('Physical forward sequence: using verified parameter/source cache.', flush=True)
                description['exposure_gain'] = float(cached['exposure_gain'])
                return cached['means'], cached['instantaneous_derivative'], points, weights, description
    axis = np.linspace(-settings.width_m, settings.width_m, settings.n)
    wx, wy = np.meshgrid(axis[::-1], axis, indexing='xy')
    wall = np.column_stack((wx.ravel(), wy.ravel(), np.zeros(wx.size)))
    edges = np.arange(settings.m+1)*settings.bin_resolution_s
    center, derivative = confocal_response(points, weights, wall, edges, velocity)
    gain = description['reference_expected_photons']/center.sum()
    shape = (settings.n, settings.n, settings.m)
    derivative = (gain*derivative).reshape(shape).transpose(2, 0, 1)
    means = []
    for index, time in enumerate(times):
        if time == 0:
            response = center
        else:
            response, _ = confocal_response(points+time*velocity, weights, wall, edges)
        means.append((gain*response).reshape(shape).transpose(2, 0, 1))
        print(f'  Physical forward frame {index+1}/{len(times)}: t={time:+.3f} s', flush=True)
    means = np.stack(means)
    description['exposure_gain'] = float(gain)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    # The key deliberately precedes derived gain; the generator and all inputs
    # determine it. Cached arrays retain the same count scale across frames.
    np.savez_compressed(cache_path, key=key, means=means, instantaneous_derivative=derivative,
                        exposure_gain=gain)
    return means, derivative, points, weights, description
