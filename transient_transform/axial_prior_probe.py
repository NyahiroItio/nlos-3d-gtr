"""Probe a direct axial-expansion prior on the saved three-layer experiment.

The layer positions and expansion rate are estimated from LCT reconstructions.
The noiseless center and true rate are used only for diagnostic controls.
"""

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.signal import find_peaks

from .motion_simulation import confocal_response, glyph_points
from .original_lct import OriginalLCT


DEFAULT_INPUT = Path(__file__).resolve().parent / 'results/depth_expansion_fixed_h02/depth_expansion'


def relative_error(estimate, target):
    return float(np.linalg.norm((estimate-target).ravel())/np.linalg.norm(target.ravel()))


def layer_gap(volume, depth, center, prominence_fraction=.03):
    """Measure the two outer-layer peaks of a three-layer depth profile."""
    # Keep the visible layer peak from each lateral column instead of
    # accumulating low-level LCT sidelobes over the full y axis.
    profile = np.maximum(volume, 0).max(axis=1).sum(axis=1)
    peaks, _ = find_peaks(profile, distance=6,
                          prominence=prominence_fraction*profile.max())
    left = [i for i in peaks if center-.09 < depth[i] < center-.012]
    right = [i for i in peaks if center+.012 < depth[i] < center+.09]
    if not left or not right:
        return None
    low = max(left, key=lambda i: depth[i])
    high = min(right, key=lambda i: depth[i])
    return float((depth[high]-depth[low])/2)


def fit_expansion(times, volumes, depth, center):
    gaps = [layer_gap(volume, depth, center) for volume in volumes]
    usable = [(time, gap) for time, gap in zip(times, gaps) if gap is not None]
    if len(usable) < 3:
        raise ValueError('Fewer than three frames have resolvable outer layers.')
    t, g = np.asarray(usable).T
    intercept, slope = np.linalg.lstsq(np.column_stack((np.ones(len(t)), t)),
                                       g, rcond=None)[0]
    if intercept <= 0:
        raise ValueError('Estimated central half-gap must be positive.')
    return float(slope/intercept), gaps


def depth_sample(volume, requested, depth):
    """Linear interpolation along depth, with zero outside the measured grid."""
    index = (requested-depth[0])/(depth[1]-depth[0])
    left = np.floor(index).astype(int)
    valid = (left >= 0) & (left < len(depth)-1)
    safe = np.clip(left, 0, len(depth)-2)
    fraction = (index-safe)[:, None, None]
    sampled = (1-fraction)*volume[safe]+fraction*volume[safe+1]
    sampled[~valid] = 0
    return sampled


def align_to_center(volume, time, rate, depth, center):
    scale = 1+rate*time
    if scale <= 0:
        raise ValueError('Nonpositive expansion scale.')
    return scale*depth_sample(volume, center+scale*(depth-center), depth)


def transport_slope(center_volume, rate, depth, center):
    """Eulerian derivative of an axially dilated proxy volume at t=0."""
    return -rate*np.gradient((depth-center)[:, None, None]*center_volume,
                             depth[1]-depth[0], axis=0)


def point_derivative_match(means, counts, analytic_derivative, times, geometry):
    """Scan depth with a single-point d(A delta)/dz template at one known xy.

    This is a detection diagnostic with oracle lateral coordinate, not a
    reconstruction of the unknown moving scene.
    """
    planar, weights = glyph_points()
    center_xy = np.average(planar[:, :2], axis=0, weights=weights)
    eligible = np.flatnonzero(weights > .7*weights.max())
    chosen = eligible[np.argmin(np.linalg.norm(planar[eligible, :2]-center_xy,
                                             axis=1))]
    xy = planar[chosen, :2]
    axis = np.linspace(-geometry['width_m'], geometry['width_m'], geometry['n'])
    wx, wy = np.meshgrid(axis[::-1], axis, indexing='xy')
    wall = np.column_stack((wx.ravel(), wy.ravel(), np.zeros(wx.size)))
    nearest = np.argmin(np.linalg.norm(wall[:, :2]-xy, axis=1))
    row, col = divmod(nearest, geometry['n'])
    row_ids = np.arange(max(0, row-3), min(geometry['n'], row+4))
    col_ids = np.arange(max(0, col-3), min(geometry['n'], col+4))
    wall_ids = (row_ids[:, None]*geometry['n']+col_ids[None, :]).ravel()
    local_wall = wall[wall_ids]
    m, n = geometry['m'], geometry['n']

    def select(transient):
        return transient.transpose(1, 2, 0).reshape(n*n, m)[wall_ids]

    # Independent Poisson frames: variance of the centered difference.
    variance = (select(means[3])+select(means[1]))/(times[3]-times[1])**2
    inverse_variance = 1/np.maximum(variance, 1.)
    signals = {
        'analytic_instantaneous': select(analytic_derivative),
        'noiseless_adjacent': select((means[3]-means[1])/(times[3]-times[1])),
        'first_noisy_adjacent': select((counts[3]-counts[1])/(times[3]-times[1])),
    }
    candidate_depths = np.linspace(.57, .73, 81)
    scores = {name: [] for name in signals}
    edges = np.arange(m+1)*geometry['bin_resolution_s']
    for z in candidate_depths:
        point = np.array([[xy[0], xy[1], z]])
        _, kernel = confocal_response(point, np.ones(1), local_wall, edges,
                                      velocity=np.array([0., 0., 1.]),
                                      power=4, wall_batch=len(local_wall))
        denominator = np.sqrt(np.sum(kernel**2*inverse_variance))
        for name, signal in signals.items():
            scores[name].append(float(np.sum(kernel*signal*inverse_variance)/denominator))
    peak_locations = {}
    for name, values in scores.items():
        values = np.asarray(values)
        below = np.flatnonzero(candidate_depths < .65)
        above = np.flatnonzero(candidate_depths > .65)
        peak_locations[name] = {
            'negative_below_center_m': float(candidate_depths[below[np.argmin(values[below])]]),
            'positive_above_center_m': float(candidate_depths[above[np.argmax(values[above])]]),
        }
    return {'oracle_lateral_coordinate_m': xy.tolist(),
            'wall_patch_size': len(local_wall), 'candidate_depths_m': candidate_depths.tolist(),
            'normalization': 'Poisson inverse-variance weighted derivative-template correlation',
            'scores': scores, 'signed_peak_locations': peak_locations,
            'true_outer_layer_depths_m': [.61, .69],
            'note': 'Signed peak localization can have sidelobes and cross-talk; scores are not density or velocity.'}


def save_match_figure(match, output):
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib import pyplot as plt

    z = np.asarray(match['candidate_depths_m'])
    fig, ax = plt.subplots(figsize=(8, 4.5))
    names = {'analytic_instantaneous': 'Instantaneous analytic derivative',
             'noiseless_adjacent': 'Noiseless adjacent difference',
             'first_noisy_adjacent': 'First noisy adjacent difference'}
    for key, label in names.items():
        values = np.asarray(match['scores'][key])
        ax.plot(z, values/np.max(np.abs(values)), label=label)
    for i, true_z in enumerate(match['true_outer_layer_depths_m']):
        ax.axvline(true_z, color='black', linestyle=':', alpha=.7,
                   label='True outer layer' if i == 0 else None)
    ax.axhline(0, color='gray', linewidth=.7)
    ax.set(xlabel='Candidate depth (m)', ylabel='Signed score / maximum absolute score',
           title='One-point axial derivative template, known lateral coordinate')
    ax.legend(loc='best', fontsize=8)
    fig.tight_layout()
    fig.savefig(output, dpi=150)
    plt.close(fig)


def run(source=DEFAULT_INPUT, output=None, trials=1):
    source = Path(source)
    output = Path(output) if output is not None else source/'axial_prior_probe.json'
    report = json.loads((source/'report.json').read_text(encoding='utf-8'))
    geometry = report['settings']
    center = report['data_model']['expansion_center_m'][2]
    true_rate = report['data_model']['expansion_rate_per_s']
    model = OriginalLCT(geometry['n'], geometry['m'], geometry['width_m'],
                        geometry['range_m'], geometry['diffuse'], geometry['snr'])
    depth = np.linspace(0, geometry['range_m']/2, geometry['m'])
    with np.load(source/'observations.npz') as data:
        times = data['slow_times_s']
        means = data['expected_transients']
        counts = data['counts']/float(data['exposure_per_frame'])
        analytic_derivative = data['instantaneous_transient_derivative']
    with np.load(source/'coefficients.npz') as data:
        target_center, target_slope = data['reference_signed']
        baseline_noiseless = relative_error(data['expected_window_signed'][1], target_slope)
        baseline_first_noisy = relative_error(data['ltai_signed'][1], target_slope)

    result = {'source': str(source), 'center_depth_m': center,
              'true_rate_per_s_diagnostic_only': true_rate,
              'reference': 'Instantaneous signed noise-free original-LCT derivative.',
              'conditions': {}}
    for name, transients in [('noiseless', means), ('first_noisy_sequence', counts)]:
        volumes = [model(frame) for frame in transients]
        rate, gaps = fit_expansion(times, volumes, depth, center)
        aligned = np.mean([align_to_center(volume, time, rate, depth, center)
                           for volume, time in zip(volumes, times)], axis=0)
        estimated_slope = transport_slope(aligned, rate, depth, center)
        control_slope = transport_slope(target_center, true_rate, depth, center)
        result['conditions'][name] = {
            'outer_layer_half_gaps_m': gaps,
            'estimated_rate_per_s': rate,
            'rate_relative_error': abs(rate-true_rate)/abs(true_rate),
            'aligned_center_relative_error': relative_error(aligned, target_center),
            'transport_slope_relative_error': relative_error(estimated_slope, target_slope),
            'oracle_center_and_rate_transport_slope_relative_error':
                relative_error(control_slope, target_slope),
            'same_sequence_baseline_window_slope_relative_error':
                baseline_noiseless if name == 'noiseless' else baseline_first_noisy,
        }
        del volumes, aligned, estimated_slope, control_slope
    result['point_derivative_match'] = point_derivative_match(
        means, counts, analytic_derivative, times, geometry)
    if trials > 1:
        rng = np.random.default_rng(20261102)
        rates = [result['conditions']['first_noisy_sequence']['estimated_rate_per_s']]
        aligned_errors = [result['conditions']['first_noisy_sequence']['aligned_center_relative_error']]
        slope_errors = [result['conditions']['first_noisy_sequence']['transport_slope_relative_error']]
        baseline_errors = [result['conditions']['first_noisy_sequence'][
            'same_sequence_baseline_window_slope_relative_error']]
        slope_weights = times/np.dot(times, times)
        for _ in range(trials-1):
            volumes = [model(rng.poisson(mean)) for mean in means]
            rate, _ = fit_expansion(times, volumes, depth, center)
            aligned = np.mean([align_to_center(volume, time, rate, depth, center)
                               for volume, time in zip(volumes, times)], axis=0)
            slope = transport_slope(aligned, rate, depth, center)
            baseline = np.einsum('i,ijkl->jkl', slope_weights, volumes)
            rates.append(rate)
            aligned_errors.append(relative_error(aligned, target_center))
            slope_errors.append(relative_error(slope, target_slope))
            baseline_errors.append(relative_error(baseline, target_slope))
            del volumes, aligned, slope, baseline
        result['monte_carlo'] = {
            'trials': trials, 'first_sequence_is_saved_original': True,
            'estimated_rate_per_s_mean': float(np.mean(rates)),
            'estimated_rate_per_s_std': float(np.std(rates, ddof=1)),
            'estimated_rate_per_s_min_max': [float(np.min(rates)), float(np.max(rates))],
            'aligned_center_relative_rms': float(np.sqrt(np.mean(np.square(aligned_errors)))),
            'transport_slope_relative_rms': float(np.sqrt(np.mean(np.square(slope_errors)))),
            'same_sequence_window_slope_relative_rms':
                float(np.sqrt(np.mean(np.square(baseline_errors)))),
        }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding='utf-8')
    save_match_figure(result['point_derivative_match'], output.with_name('axial_derivative_match.png'))
    print(json.dumps({'conditions': result['conditions'],
                      'monte_carlo': result.get('monte_carlo'),
                      'point_derivative_match': {
                          key: value for key, value in result['point_derivative_match'].items()
                          if key != 'scores' and key != 'candidate_depths_m'}}, indent=2))
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=DEFAULT_INPUT)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--trials', type=int, default=1)
    args = parser.parse_args()
    run(args.source, args.output, args.trials)
