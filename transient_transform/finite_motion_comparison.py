"""Compare slope compression with finite-motion fitting on one known S geometry.

This is an oracle mechanism test: planar point positions, reflectivity weights,
motion center, IRF, and exposure gain are supplied. Only the global maximum
point speed is selected from a grid. It is not a general dynamic inversion.
"""

import argparse
import json
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

from .motion_simulation import confocal_response, expansion_sequence, glyph_points
from .original_lct import OriginalLCT, SceneSettings
from .speed_expansion_study import relative_error


RESULTS = Path(__file__).resolve().parent/'results'
DEFAULT_OUTPUT = RESULTS/'finite_motion_comparison'
SPEEDS = (.08, .37, .40)


def wall_subset(settings, stride=4):
    """Spread a 16x16 subset across the full 64x64 relay aperture."""
    indices = np.arange(0, settings.n, stride)
    rows, cols = np.meshgrid(indices, indices, indexing='ij')
    wall_ids = (rows*settings.n+cols).ravel()
    axis = np.linspace(-settings.width_m, settings.width_m, settings.n)
    wx, wy = np.meshgrid(axis[::-1], axis, indexing='xy')
    full_wall = np.column_stack((wx.ravel(), wy.ravel(), np.zeros(settings.n**2)))
    return full_wall[wall_ids], wall_ids


def region(frames, wall_ids, start, stop):
    """Extract [slow time, wall point, fast bin] from native layout."""
    array = np.asarray(frames)
    if array.ndim == 3:
        array = array[None]
    nt, _, n, _ = array.shape
    selected = array[:, start:stop].transpose(0, 2, 3, 1)
    return selected.reshape(nt, n*n, stop-start)[:, wall_ids]


def fit_fixed_gain(observation, templates, weight, gain):
    """Find the candidate with least weighted residual at a supplied gain."""
    difference = templates*gain-observation[None]
    losses = np.einsum('c...,...->c', difference**2, weight, optimize=True)
    return int(np.argmin(losses)), losses


def fit_free_gain(observation, templates, weight):
    """Profile one nonnegative amplitude for each finite-motion candidate."""
    dot = np.einsum('c...,...,...->c', templates, observation, weight,
                    optimize=True)
    norm = np.einsum('c...,...->c', templates**2, weight,
                     optimize=True)
    gains = np.maximum(dot/norm, 0.)
    improvement = 2*gains*dot-gains**2*norm
    winner = int(np.argmax(improvement))
    return winner, float(gains[winner])


def parameterized_result(index, estimated_gain, gain, true_speed,
                         candidate_speeds):
    estimated_speed = float(candidate_speeds[index])
    error = float(abs(estimated_gain/gain*estimated_speed/true_speed-1))
    return {'estimated_speed_m_per_s': estimated_speed,
            'estimated_gain_ratio': float(estimated_gain/gain),
            'raw_derivative_error': error,
            'lct_derivative_error': error}


def candidate_bank(speeds, times, settings, wall, start, stop):
    points, weights = glyph_points()
    center = np.average(points, axis=0, weights=weights)
    radial = points-center
    radial[:, 2] = 0.
    max_radius = np.max(np.linalg.norm(radial, axis=1))
    edges = np.arange(start, stop+1)*settings.bin_resolution_s
    zero, unit_derivative = confocal_response(
        points, weights, wall, edges, radial/max_radius)
    frames = []
    for index, speed in enumerate(speeds):
        sequence = []
        for time in times:
            if time == 0:
                response = zero
            else:
                response, _ = confocal_response(
                    points+time*speed*radial/max_radius, weights, wall, edges)
            sequence.append(response)
        frames.append(sequence)
        print(f'  finite-motion template {index+1}/{len(speeds)}: '
              f'{speed:.3f} m/s', flush=True)
    return np.asarray(frames), unit_derivative


def measure(observation, full_bank, slope_bank, full_weight, slope_weight,
            gain, candidate_speeds, slope_weights, true_speed,
            baseline_raw_error, baseline_lct_error):
    centered = observation-observation.mean(axis=0, keepdims=True)
    slope = np.tensordot(slope_weights, observation, axes=(0, 0))
    full_index, full_losses = fit_fixed_gain(
        centered, full_bank, full_weight, gain)
    slope_index, slope_losses = fit_fixed_gain(
        slope, slope_bank, slope_weight, gain)
    full_free_index, full_free_gain = fit_free_gain(
        centered, full_bank, full_weight)
    slope_free_index, slope_free_gain = fit_free_gain(
        slope, slope_bank, slope_weight)
    row = {
        'ols': {
            'raw_derivative_error': baseline_raw_error,
            'lct_derivative_error': baseline_lct_error},
        'slope_finite_template': {
            'estimated_speed_m_per_s': float(candidate_speeds[slope_index]),
            'raw_derivative_error':
                float(abs(candidate_speeds[slope_index]/true_speed-1)),
            'lct_derivative_error':
                float(abs(candidate_speeds[slope_index]/true_speed-1)),
            'relative_score_gap':
                float((np.partition(slope_losses, 1)[1]-slope_losses[slope_index])/
                      max(np.linalg.norm(slope)**2, 1e-30))},
        'full_finite_template': {
            'estimated_speed_m_per_s': float(candidate_speeds[full_index]),
            'raw_derivative_error':
                float(abs(candidate_speeds[full_index]/true_speed-1)),
            'lct_derivative_error':
                float(abs(candidate_speeds[full_index]/true_speed-1)),
            'relative_score_gap':
                float((np.partition(full_losses, 1)[1]-full_losses[full_index])/
                      max(np.linalg.norm(centered)**2, 1e-30))},
        'slope_finite_template_free_gain': parameterized_result(
            slope_free_index, slope_free_gain, gain, true_speed,
            candidate_speeds),
        'full_finite_template_free_gain': parameterized_result(
            full_free_index, full_free_gain, gain, true_speed,
            candidate_speeds)}
    return row


def summarize_trials(trials):
    summary = {}
    for method in ('ols', 'slope_finite_template', 'full_finite_template',
                   'slope_finite_template_free_gain',
                   'full_finite_template_free_gain'):
        rows = [trial[method] for trial in trials]
        summary[method] = {
            'raw_derivative_error_rms': float(np.sqrt(np.mean(
                [row['raw_derivative_error']**2 for row in rows]))),
            'lct_derivative_error_rms': float(np.sqrt(np.mean(
                [row['lct_derivative_error']**2 for row in rows])))}
        if method != 'ols':
            speed = np.array([row['estimated_speed_m_per_s'] for row in rows])
            summary[method]['estimated_speed_m_per_s'] = speed.tolist()
            summary[method]['speed_mean_m_per_s'] = float(speed.mean())
            summary[method]['speed_std_m_per_s'] = float(speed.std())
        if method.endswith('free_gain'):
            summary[method]['estimated_gain_ratio'] = [
                row['estimated_gain_ratio'] for row in rows]
    return summary


def plot_results(report, output):
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib import pyplot as plt

    speeds = np.array([float(key) for key in report['scenes']])
    positions = np.arange(len(speeds))
    methods = [('ols', 'Quadratic slope', 'C0'),
               ('slope_finite_template_free_gain', 'Finite slope template', 'C1'),
               ('full_finite_template_free_gain', 'Full five-frame template', 'C2')]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    for method, label, color in methods:
        clean = [100*report['scenes'][f'{speed:.2f}']['clean'][method][
            'lct_derivative_error'] for speed in speeds]
        noisy = [100*report['scenes'][f'{speed:.2f}']['poisson_trials'][method][
            'lct_derivative_error_rms'] for speed in speeds]
        axes[0].plot(positions, clean, 'o-', color=color, label=label)
        axes[1].plot(positions, noisy, 'o-', color=color, label=label)
        if method != 'ols':
            axes[2].plot(positions, noisy, 'o-', color=color, label=label)
    for ax in axes:
        ax.axvline(1, color='gray', linestyle=':', alpha=.6)
        ax.set(xlabel='Maximum point speed (m/s)',
               ylabel='Relative LCT derivative error (%)')
        ax.set_xticks(positions, [f'{speed:.2f}' for speed in speeds])
        ax.grid(alpha=.2)
    axes[0].set_title('Noiseless / known S geometry')
    axes[1].set_title(f'{report["trials"]} Poisson trials / RMS')
    axes[2].set_title('Finite-template errors, enlarged')
    axes[2].set_ylim(0, 1.)
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output, dpi=150)
    fig.savefig(output.with_suffix('.svg'))
    plt.close(fig)


def run(output=DEFAULT_OUTPUT, trials=12, seed=20261105):
    if trials < 1:
        raise ValueError('At least one Poisson trial is required.')
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    settings = SceneSettings(0, 'physical_s_expansion', .35, 64, 512,
                             16e-12, 2.4576, True, .08, 150, 25)
    inverse = OriginalLCT(settings.n, settings.m, settings.width_m,
                          settings.range_m, settings.diffuse, settings.snr)
    times = np.linspace(-.2, .2, 5)
    slope_weights = times/np.dot(times, times)
    start, stop = 190, 350
    wall, wall_ids = wall_subset(settings)
    candidates = np.arange(.02, .501, .02)
    assert any(np.isclose(candidates, .08)) and any(np.isclose(candidates, .40))
    with threadpool_limits(limits=2):
        bank, unit_derivative = candidate_bank(
            candidates, times, settings, wall, start, stop)
        full_bank = bank-bank.mean(axis=1, keepdims=True)
        slope_bank = np.tensordot(bank, slope_weights, axes=(1, 0))
        scenes = {}
        rng = np.random.default_rng(seed)
        for true_speed in SPEEDS:
            if true_speed == .08:
                source = RESULTS/'expansion_fixed_h02'/'expansion'/'forward_cache.npz'
            elif true_speed == .40:
                source = RESULTS/'planar_speed_study'/'speed_400'/'forward_cache.npz'
            else:
                source = output/'speed_370'/'forward_cache.npz'
            means, derivative, _, _, source_metadata = expansion_sequence(
                times, settings, source, max_point_speed=true_speed,
                mode='planar')
            gain = source_metadata['exposure_gain']
            true_local = region(derivative, wall_ids, start, stop)[0]
            expected_local = gain*true_speed*unit_derivative
            if relative_error(expected_local, true_local) > 1e-10:
                raise AssertionError('Candidate and cached analytic derivatives differ.')
            reference_reconstruction = inverse(derivative)
            clean_local = region(means, wall_ids, start, stop)
            if np.any(np.isclose(candidates, true_speed)):
                template = bank[np.flatnonzero(
                    np.isclose(candidates, true_speed))[0]]
                if relative_error(gain*template, clean_local) > 1e-10:
                    raise AssertionError('Candidate forward model differs from cached frames.')
            weight = 1/np.maximum(clean_local.mean(axis=0), 1.)
            clean_slope = np.tensordot(slope_weights, means, axes=(0, 0))
            clean = measure(clean_local, full_bank, slope_bank,
                            weight, weight, gain, candidates, slope_weights,
                            true_speed, relative_error(clean_slope, derivative),
                            relative_error(inverse(clean_slope),
                                           reference_reconstruction))
            noisy = []
            for trial in range(trials):
                slope = np.zeros_like(derivative)
                sample_local = np.empty_like(clean_local)
                for i, mean in enumerate(means):
                    sample = rng.poisson(mean)
                    slope += slope_weights[i]*sample
                    sample_local[i] = region(sample, wall_ids, start, stop)[0]
                centered = sample_local-sample_local.mean(axis=0, keepdims=True)
                local_slope = np.tensordot(slope_weights, sample_local,
                                           axes=(0, 0))
                full_index, _ = fit_fixed_gain(centered, full_bank, weight, gain)
                slope_index, _ = fit_fixed_gain(local_slope, slope_bank,
                                                weight, gain)
                full_free_index, full_free_gain = fit_free_gain(
                    centered, full_bank, weight)
                slope_free_index, slope_free_gain = fit_free_gain(
                    local_slope, slope_bank, weight)
                raw_error = relative_error(slope, derivative)
                lct_error = relative_error(inverse(slope),
                                           reference_reconstruction)
                noisy.append({
                    'ols': {'raw_derivative_error': raw_error,
                            'lct_derivative_error': lct_error},
                    'slope_finite_template': {
                        'estimated_speed_m_per_s': float(candidates[slope_index]),
                        'raw_derivative_error':
                            float(abs(candidates[slope_index]/true_speed-1)),
                        'lct_derivative_error':
                            float(abs(candidates[slope_index]/true_speed-1))},
                    'full_finite_template': {
                        'estimated_speed_m_per_s': float(candidates[full_index]),
                        'raw_derivative_error':
                            float(abs(candidates[full_index]/true_speed-1)),
                        'lct_derivative_error':
                            float(abs(candidates[full_index]/true_speed-1))},
                    'slope_finite_template_free_gain': parameterized_result(
                        slope_free_index, slope_free_gain, gain, true_speed,
                        candidates),
                    'full_finite_template_free_gain': parameterized_result(
                        full_free_index, full_free_gain, gain, true_speed,
                        candidates)})
            scenes[f'{true_speed:.2f}'] = {
                'source_cache': str(source), 'exposure_gain': gain,
                'clean': clean, 'poisson_trials': summarize_trials(noisy)}
            print(f'  scene={true_speed:.2f} m/s, clean LCT errors '
                  f'{clean["ols"]["lct_derivative_error"]:.3f} / '
                  f'{clean["slope_finite_template"]["lct_derivative_error"]:.3f} / '
                  f'{clean["full_finite_template"]["lct_derivative_error"]:.3f}',
                  flush=True)
    report = {
        'design': 'Known 613-point planar S, conserved weights, center and IRF; select global expansion speed. Main arms use true gain; additional arms profile one nonnegative unknown gain. Same native five-frame Poisson data for all arms.',
        'slow_times_s': times.tolist(),
        'wall_subset': '16x16 spread across original 64x64 aperture',
        'fast_bin_range': [start, stop],
        'candidate_speeds_m_per_s': candidates.tolist(),
        'gain_convention': 'Both true-gain and unknown-gain fits are reported. Unknown gain is separately profiled for each candidate.',
        'weight_convention': 'Fixed inverse of max(noiseless temporal mean count, 1) for each wall/fast bin; shared by all candidate scores.',
        'trials': trials, 'seed': seed, 'scenes': scenes,
        'interpretation_limit': 'The exact scene geometry and forward model are supplied. The 0.08 and 0.40 m/s speeds lie on the candidate grid, while 0.37 m/s is off-grid. Zero clean error on-grid checks model consistency, not blind reconstruction. Poisson trials do not include calibration or model mismatch.'}
    (output/'report.json').write_text(json.dumps(report, indent=2),
                                      encoding='utf-8')
    plot_results(report, output/'comparison.png')
    print(json.dumps({key: {
        'clean_lct_error': {method: value['clean'][method]['lct_derivative_error']
                            for method in ('ols', 'slope_finite_template_free_gain',
                                           'full_finite_template_free_gain')},
        'poisson_lct_rms': {method: value['poisson_trials'][method][
            'lct_derivative_error_rms']
            for method in ('ols', 'slope_finite_template_free_gain',
                           'full_finite_template_free_gain')},
        'estimated_clean_speed': {
            method: value['clean'][method]['estimated_speed_m_per_s']
            for method in ('slope_finite_template_free_gain',
                           'full_finite_template_free_gain')}}
        for key, value in scenes.items()}, indent=2))
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument('--trials', type=int, default=12)
    parser.add_argument('--seed', type=int, default=20261105)
    args = parser.parse_args()
    run(args.output, args.trials, args.seed)
