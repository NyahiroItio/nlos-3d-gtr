"""Quantify Gaussian shift-fit bias and scan faster physical planar expansions."""

import argparse
import json
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

from .motion_simulation import expansion_sequence, layered_glyph_points
from .original_lct import OriginalLCT, SceneSettings


RESULTS = Path(__file__).resolve().parent/'results'
DEFAULT_OUTPUT = RESULTS/'planar_speed_study'
DEPTH_ROOT = RESULTS/'depth_expansion_fixed_h02'/'depth_expansion'


def gaussian_shift_relative_error(eta, normalized_times):
    """Exact continuous-fast-time relative L2 slope bias for one shifted Gaussian.

    The pulse has constant amplitude and center q(t)=q(0)+u*t. The slow-time
    slope is ordinary least squares with an intercept at symmetric times.
    eta=abs(u)*H/sigma, where H is the window half-width.
    """
    t = np.asarray(normalized_times, dtype=float)
    if eta <= 0 or t.ndim != 1 or len(t) < 3 or not np.allclose(t, -t[::-1]):
        raise ValueError('Use positive eta and symmetric slow times.')
    coefficients = t/np.dot(t, t)
    norm_ratio_squared = 2/eta**2*np.sum(
        coefficients[:, None]*coefficients[None, :]*
        np.exp(-eta**2*(t[:, None]-t[None, :])**2/4))
    correlation = np.sum(coefficients*t*np.exp(-eta**2*t**2/4))
    return float(np.sqrt(max(0., 1+norm_ratio_squared-2*correlation)))


def relative_error(estimate, reference):
    return float(np.linalg.norm((estimate-reference).ravel())/
                 np.linalg.norm(reference.ravel()))


def endpoint_shift_summary(points, weights, velocities, half_window, settings):
    """Diagnostic on a fixed 16x16 subset of the 64x64 relay scan."""
    axis = np.linspace(-settings.width_m, settings.width_m, settings.n)[::4]
    wx, wy = np.meshgrid(axis[::-1], axis, indexing='xy')
    wall = np.column_stack((wx.ravel(), wy.ravel(), np.zeros(wx.size)))
    distance_0 = np.linalg.norm(points[None]-wall[:, None], axis=2)
    radial_speed = np.abs(np.einsum(
        'wpi,pi->wp', points[None]-wall[:, None], velocities)/distance_0)
    distance_h = np.linalg.norm((points+half_window*velocities)[None]-wall[:, None],
                                axis=2)
    shift_ps = np.abs(distance_h-distance_0)*2e12/3e8
    photon_weight = np.broadcast_to(weights[None]/distance_0**4, shift_ps.shape)
    active = np.linalg.norm(velocities, axis=1) > 1e-12
    active_weight = photon_weight[:, active]
    return {'weighted_mean_absolute_shift_ps':
                float(np.sum(shift_ps*photon_weight)/photon_weight.sum()),
            'active_weighted_mean_absolute_shift_ps':
                float(np.sum(shift_ps[:, active]*active_weight)/active_weight.sum()),
            'weighted_mean_absolute_radial_speed_m_per_s':
                float(np.sum(radial_speed*photon_weight)/photon_weight.sum()),
            'weighted_fraction_shift_over_60ps':
                float(photon_weight[shift_ps > 60].sum()/photon_weight.sum()),
            'active_weighted_fraction_shift_over_60ps':
                float(active_weight[shift_ps[:, active] > 60].sum()/active_weight.sum()),
            'sampled_wall_points': len(wall)}


def plot_results(rows, output, axial_reference):
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib import pyplot as plt

    speeds = np.array([row['maximum_point_speed_m_per_s'] for row in rows])
    shifts = np.array([row['endpoint_shift']['weighted_mean_absolute_shift_ps']
                       for row in rows])
    raw = np.array([row['raw_transient_slope_bias'] for row in rows])
    lct = np.array([row['lct_slope_bias'] for row in rows])
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(speeds, shifts, 'o-')
    axes[0].axhline(60, color='gray', linestyle=':', label='IRF FWHM = 60 ps')
    axes[0].set(xlabel='Maximum point speed (m/s)',
                ylabel='Photon-weighted mean endpoint shift (ps)',
                title='Planar expansion / 0.2 s half-window')
    axes[0].legend(fontsize=8)
    axes[1].plot(shifts, 100*raw, 'o-', label='Raw transient derivative')
    axes[1].plot(shifts, 100*lct, 's-', label='After author LCT')
    axes[1].axhline(100*axial_reference['raw_transient_slope_bias'],
                    color='C0', linestyle=':', label='Axial raw bias')
    axes[1].axhline(100*axial_reference['lct_slope_bias'],
                    color='C1', linestyle=':', label='Axial LCT bias')
    axes[1].set(xlabel='Mean endpoint shift (ps)',
                ylabel='Noiseless relative slope bias (%)',
                title='Bias rises as echoes move farther')
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output, dpi=150)
    fig.savefig(output.with_suffix('.svg'))
    plt.close(fig)


def run(output=DEFAULT_OUTPUT, speeds=(.08, .16, .24, .32, .40, .48),
        trials=12, seed=20261104):
    output = Path(output)
    if trials < 1 or any(speed <= 0 for speed in speeds):
        raise ValueError('Use positive speeds and at least one noisy trial.')
    output.mkdir(parents=True, exist_ok=True)
    times = np.linspace(-.2, .2, 5)
    half_window = float(times[-1])
    slope_weights = times/np.dot(times, times)
    settings = SceneSettings(0, 'physical_s_expansion', .35, 64, 512, 16e-12,
                             2.4576, True, .08, 150, 25)
    inverse = OriginalLCT(settings.n, settings.m, settings.width_m,
                          settings.range_m, settings.diffuse, settings.snr)
    depth = json.loads((DEPTH_ROOT/'report.json').read_text(encoding='utf-8'))
    with np.load(DEPTH_ROOT/'forward_cache.npz', allow_pickle=False) as axial:
        axial_slope = np.tensordot(slope_weights, axial['means'], axes=(0, 0))
        axial_raw_reference = relative_error(
            axial_slope, axial['instantaneous_derivative'])
    axial_lct_reference = depth['metrics']['ltai']['relative_temporal_bias'][1]
    axial_points, axial_weights = layered_glyph_points()
    axial_center = np.asarray(depth['data_model']['expansion_center_m'])
    axial_velocity = depth['data_model']['expansion_rate_per_s']*(axial_points-axial_center)
    axial_velocity[:, :2] = 0
    axial_shift = endpoint_shift_summary(axial_points, axial_weights,
                                         axial_velocity, half_window, settings)
    rows = []
    with threadpool_limits(limits=2):
        for speed in speeds:
            cache = (Path(__file__).resolve().parent/'results'/'expansion_fixed_h02'/
                     'expansion'/'forward_cache.npz' if np.isclose(speed, .08) else
                     output/f'speed_{int(round(speed*1000)):03d}'/'forward_cache.npz')
            means, derivative, points, weights, metadata = expansion_sequence(
                times, settings, cache, max_point_speed=speed, mode='planar')
            slope = np.tensordot(slope_weights, means, axes=(0, 0))
            reference = inverse(derivative)
            fitted = inverse(slope)
            radial = points-np.asarray(metadata['expansion_center_m'])
            radial[:, 2] = 0
            velocities = metadata['expansion_rate_per_s']*radial
            displacement = endpoint_shift_summary(points, weights, velocities,
                                                   half_window, settings)
            row = {'maximum_point_speed_m_per_s': float(speed),
                   'scale_range': [metadata['minimum_scale'], metadata['maximum_scale']],
                   'expected_sequence_photons': float(means.sum()),
                   'endpoint_shift': displacement,
                   'raw_transient_slope_bias': relative_error(slope, derivative),
                   'lct_slope_bias': relative_error(fitted, reference),
                   'reference_norm': float(np.linalg.norm(reference.ravel())),
                   'cache': str(cache)}
            rows.append(row)
            print(f'  speed={speed:.2f} m/s, shift={displacement["weighted_mean_absolute_shift_ps"]:.1f} ps, '
                  f'raw bias={row["raw_transient_slope_bias"]:.3f}, '
                  f'LCT bias={row["lct_slope_bias"]:.3f}', flush=True)
            del means, derivative, slope, reference, fitted

        selected = min(rows, key=lambda row: abs(row['raw_transient_slope_bias']-
                                                 axial_raw_reference))
        selected_speed = selected['maximum_point_speed_m_per_s']
        means, derivative, _, _, _ = expansion_sequence(
            times, settings, Path(selected['cache']),
            max_point_speed=selected_speed, mode='planar')
        ideal_slope = np.tensordot(slope_weights, means, axes=(0, 0))
        reference = inverse(derivative)
        ideal_reconstruction = inverse(ideal_slope)
        rng = np.random.default_rng(seed)
        errors, noise_errors = [], []
        for trial in range(trials):
            sampled_slope = np.zeros_like(derivative)
            for time_weight, mean in zip(slope_weights, means):
                sampled_slope += time_weight*rng.poisson(mean)
            reconstructed = inverse(sampled_slope)
            errors.append(relative_error(reconstructed, reference)**2)
            noise_errors.append(relative_error(reconstructed, ideal_reconstruction)**2)
            del sampled_slope, reconstructed
            if (trial+1) % 4 == 0 or trial+1 == trials:
                print(f'  matched-speed noisy trials: {trial+1}/{trials}', flush=True)
        selected['noisy_slope_total_rms'] = float(np.sqrt(np.mean(errors)))
        selected['noisy_slope_noise_rms'] = float(np.sqrt(np.mean(noise_errors)))
        del means, derivative, ideal_slope, reference, ideal_reconstruction

    alpha = times/half_window
    sigma_ps = 60/(2*np.sqrt(2*np.log(2)))
    report = {'times_s': times.tolist(), 'half_window_s': half_window,
              'irf_fwhm_ps': 60., 'irf_sigma_ps': sigma_ps,
              'settings': settings.metadata(),
              'theory': {'small_eta_coefficient':
                             float(np.sqrt(15)/12*np.sum(alpha**4)/np.sum(alpha**2)),
                         'eta': [.25, .5, 1., 2., 3., 4.],
                         'single_gaussian_relative_error':
                             [gaussian_shift_relative_error(eta, alpha)
                              for eta in (.25, .5, 1., 2., 3., 4.)]},
              'axial_reference': {'raw_transient_slope_bias': axial_raw_reference,
                                  'lct_slope_bias': axial_lct_reference,
                                  'endpoint_shift': axial_shift,
                                  'lct_noisy_total_rms':
                                      depth['metrics']['ltai']['relative_total_rms_error'][1]},
              'planar_speeds': rows,
              'selected_speed_for_noisy_validation_m_per_s': selected_speed,
              'selected_speed_rule': 'Closest raw transient slope bias to the saved axial experiment among tested planar speeds.',
              'relative_error_reference': 'Noiseless instantaneous analytic derivative, before or after the same signed author LCT. No physical density calibration.',
              'limitations': 'Planar S sheet and axial three-layer slab are different geometries. Increasing planar expansion speed also changes scene scale over the fixed window; this scan tests the combined physical motion, not a pure direction-only ablation.'}
    (output/'report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    plot_results(rows, output/'planar_speed_bias.png', report['axial_reference'])
    print(json.dumps({'axial_reference': report['axial_reference'],
                      'selected_planar': selected}, indent=2))
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument('--speeds', type=float, nargs='+',
                        default=[.08, .16, .24, .32, .40, .48])
    parser.add_argument('--trials', type=int, default=12)
    parser.add_argument('--seed', type=int, default=20261104)
    args = parser.parse_args()
    run(args.output, args.speeds, args.trials, args.seed)
