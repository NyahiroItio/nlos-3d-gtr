"""Match complete finite axial point shifts to five raw transient frames.

This is a small phase-space detection experiment with the lateral coordinate
supplied from the simulated scene. It does not recover a general velocity field.
"""

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.special import ndtr

from .motion_simulation import C, glyph_points


DEFAULT_INPUT = Path(__file__).resolve().parent / 'results/depth_expansion_fixed_h02/depth_expansion'


def point_frames(xy, z, velocity, wall, times, fast_edges, fwhm=60e-12, power=4):
    """Bin-integrated response of one unit-weight point at all slow times."""
    xy, wall, times, fast_edges = map(np.asarray, (xy, wall, times, fast_edges))
    sigma = fwhm/(2*np.sqrt(2*np.log(2)))
    depths = z+velocity*times
    if np.any(depths <= 0):
        raise ValueError('Moving point must remain beyond the wall.')
    lateral_squared = np.sum((wall[:, :2]-xy)**2, axis=1)
    radius = np.sqrt(depths[:, None]**2+lateral_squared[None, :])
    arrivals = 2*radius/C
    arguments = (fast_edges[None, None, :]-arrivals[:, :, None])/sigma
    probabilities = np.diff(ndtr(arguments), axis=-1)
    return probabilities/radius[:, :, None]**power


def centered_score(template, observations, inverse_variance):
    """Positive-amplitude matched score after removing each bin's time mean."""
    kernel = template-template.mean(axis=0, keepdims=True)
    signal = observations-observations.mean(axis=0, keepdims=True)
    norm = np.sqrt(np.sum(kernel**2*inverse_variance[None]))
    return float(np.sum(kernel*signal*inverse_variance[None])/norm)


def oracle_lateral_patch(geometry):
    planar, weights = glyph_points()
    centroid = np.average(planar[:, :2], axis=0, weights=weights)
    candidates = np.flatnonzero(weights > .7*weights.max())
    chosen = candidates[np.argmin(np.linalg.norm(planar[candidates, :2]-centroid, axis=1))]
    xy = planar[chosen, :2]
    n = geometry['n']
    axis = np.linspace(-geometry['width_m'], geometry['width_m'], n)
    wx, wy = np.meshgrid(axis[::-1], axis, indexing='xy')
    wall = np.column_stack((wx.ravel(), wy.ravel(), np.zeros(wx.size)))
    nearest = np.argmin(np.linalg.norm(wall[:, :2]-xy, axis=1))
    row, col = divmod(nearest, n)
    rows = np.arange(max(0, row-3), min(n, row+4))
    cols = np.arange(max(0, col-3), min(n, col+4))
    ids = (rows[:, None]*n+cols[None, :]).ravel()
    return xy, wall[ids], ids


def select_wall_and_bins(frames, wall_ids, start, stop):
    frames = np.asarray(frames)
    nt, m, n, _ = frames.shape
    return frames.transpose(0, 2, 3, 1).reshape(nt, n*n, m)[:, wall_ids, start:stop]


def peak(surface, depths, velocities, depth_selector=None):
    values = surface if depth_selector is None else surface[depth_selector]
    index = np.unravel_index(np.argmax(values), values.shape)
    row = index[0] if depth_selector is None else np.flatnonzero(depth_selector)[index[0]]
    col = index[1]
    return {'depth_m': float(depths[row]), 'velocity_m_per_s': float(velocities[col]),
            'score': float(surface[row, col])}


def distinct_depth_peaks(surface, depths, velocities, count=2, separation_m=.02):
    """Choose prominent responses without supplying a depth split or motion sign."""
    available = np.ones(len(depths), dtype=bool)
    selected = []
    for _ in range(count):
        if not np.any(available):
            break
        selected.append(peak(surface, depths, velocities, available))
        available &= np.abs(depths-selected[-1]['depth_m']) > separation_m
    return selected


def save_figure(depths, velocities, surfaces, isolated_truth, output):
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib import pyplot as plt

    keys = [('isolated_noiseless', 'Isolated point, noiseless'),
            ('isolated_noisy', 'Isolated point, Poisson'),
            ('three_layer_noiseless', 'Three-layer scene, noiseless'),
            ('three_layer_noisy', 'Three-layer scene, Poisson')]
    fig, axes = plt.subplots(2, 2, figsize=(10, 8), sharex=True, sharey=True)
    for ax, (key, title) in zip(axes.ravel(), keys):
        score = surfaces[key]
        display = score/np.max(np.abs(score))
        image = ax.imshow(display.T, origin='lower', aspect='auto', cmap='coolwarm',
                          vmin=-1, vmax=1,
                          extent=[depths[0], depths[-1], velocities[0], velocities[-1]])
        truth = ([isolated_truth] if key.startswith('isolated') else
                 [(.61, -.08), (.69, .08)])
        ax.plot([item[0] for item in truth], [item[1] for item in truth],
                'kx', markersize=7, markeredgewidth=1.5)
        ax.set_title(title)
        ax.set_xlabel('Candidate center depth (m)')
        ax.set_ylabel('Candidate axial velocity (m/s)')
    fig.colorbar(image, ax=axes, shrink=.75, label='Signed score / maximum absolute score')
    fig.suptitle('Finite-shift point-response matching; lateral position supplied')
    fig.savefig(output, dpi=140, bbox_inches='tight')
    plt.close(fig)


def run(source=DEFAULT_INPUT, output=None, isolated_photons=1000., seed=20261102,
        trials=12):
    if trials < 1 or isolated_photons <= 0:
        raise ValueError('Use at least one trial and positive isolated photon budget.')
    source = Path(source)
    output = Path(output) if output is not None else source/'finite_shift_probe.json'
    prior = json.loads((source/'report.json').read_text(encoding='utf-8'))
    geometry = prior['settings']
    with np.load(source/'observations.npz') as data:
        times = data['slow_times_s']
        means = data['expected_transients']
        counts = data['counts']
    if float(prior['data_model']['minimum_scale']) <= 0:
        raise ValueError('Invalid source expansion scale.')
    xy, wall, wall_ids = oracle_lateral_patch(geometry)
    start, stop = 190, 350
    edges = np.arange(start, stop+1)*geometry['bin_resolution_s']
    scene_clean = select_wall_and_bins(means, wall_ids, start, stop)
    scene_noisy = select_wall_and_bins(counts, wall_ids, start, stop)

    isolated_truth = (.613, -.073)  # Deliberately between dictionary grid points.
    isolated = point_frames(xy, *isolated_truth, wall, times, edges)
    gain = isolated_photons/isolated[len(times)//2].sum()
    isolated_clean = gain*isolated
    isolated_noisy = np.random.default_rng(seed).poisson(isolated_clean)
    observations = {'isolated_noiseless': isolated_clean,
                    'isolated_noisy': isolated_noisy,
                    'three_layer_noiseless': scene_clean,
                    'three_layer_noisy': scene_noisy}
    weights = {key: 1/np.maximum(value.mean(axis=0), 1.)
               for key, value in observations.items()}

    rng = np.random.default_rng(seed+1)
    isolated_trials = np.stack([isolated_noisy]+[
        rng.poisson(isolated_clean) for _ in range(trials-1)]).astype(float)
    scene_trials = np.stack([scene_noisy]+[
        rng.poisson(scene_clean) for _ in range(trials-1)]).astype(float)
    isolated_trial_weights = 1/np.maximum(isolated_trials.mean(axis=1), 1.)
    scene_trial_weights = 1/np.maximum(scene_trials.mean(axis=1), 1.)
    isolated_trials -= isolated_trials.mean(axis=1, keepdims=True)
    scene_trials -= scene_trials.mean(axis=1, keepdims=True)
    isolated_trial_scores = np.empty((trials, 71, 12))
    scene_trial_scores = np.empty((trials, 71, 12))

    depths = np.linspace(.58, .72, 71)
    velocities = np.r_[np.arange(-.12, 0, .02), np.arange(.02, .121, .02)]
    surfaces = {key: np.empty((len(depths), len(velocities))) for key in observations}
    for i, z in enumerate(depths):
        for j, velocity in enumerate(velocities):
            template = point_frames(xy, z, velocity, wall, times, edges)
            for key, observed in observations.items():
                surfaces[key][i, j] = centered_score(template, observed, weights[key])
            centered = template-template.mean(axis=0, keepdims=True)
            for trial_data, trial_weight, destination in [
                    (isolated_trials, isolated_trial_weights, isolated_trial_scores),
                    (scene_trials, scene_trial_weights, scene_trial_scores)]:
                denominator = np.sqrt(np.einsum('twk,nwk->n',
                                                 centered**2, trial_weight))
                destination[:, i, j] = np.einsum('twk,ntwk,nwk->n',
                    centered, trial_data, trial_weight)/denominator

    left, right = depths < .65, depths > .65
    summary = {'source': str(source), 'known_lateral_coordinate_m': xy.tolist(),
               'wall_patch_size': len(wall), 'fast_bin_range': [start, stop],
               'isolated_center_photons_in_patch': isolated_photons,
               'isolated_true_motion_off_grid': {'depth_m': isolated_truth[0],
                                                  'velocity_m_per_s': isolated_truth[1]},
               'true_outer_motion': [{'depth_m': .61, 'velocity_m_per_s': -.08},
                                     {'depth_m': .69, 'velocity_m_per_s': .08}],
               'criterion': 'Time-centered, approximately Poisson-weighted, positive-amplitude matched score; weights from each observed temporal mean.',
               'depth_grid_m': depths.tolist(), 'velocity_grid_m_per_s': velocities.tolist(),
               'scores': {key: value.tolist() for key, value in surfaces.items()},
               'peaks': {key: {'global': peak(value, depths, velocities),
                               'two_distinct_depths': distinct_depth_peaks(value, depths, velocities),
                               'below_center': peak(value, depths, velocities, left),
                               'above_center': peak(value, depths, velocities, right)}
                         for key, value in surfaces.items()},
               'monte_carlo': {
                   'trials': trials, 'first_three_layer_sequence_is_saved_original': True,
                   'isolated_global_peaks': [peak(score, depths, velocities)
                                             for score in isolated_trial_scores],
                   'three_layer_below_peaks': [peak(score, depths, velocities, left)
                                               for score in scene_trial_scores],
                   'three_layer_above_peaks': [peak(score, depths, velocities, right)
                                               for score in scene_trial_scores],
                   'three_layer_two_distinct_depth_peaks': [
                       distinct_depth_peaks(score, depths, velocities)
                       for score in scene_trial_scores]},
               'limitations': 'Single supplied lateral coordinate, local 7x7 wall patch, no sparsity inversion; time-centering induces correlated noise, so weighting is approximate.'}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2), encoding='utf-8')
    save_figure(depths, velocities, surfaces, isolated_truth,
                output.with_name('finite_shift_match.png'))
    print(json.dumps({'isolated_true_motion_off_grid': summary['isolated_true_motion_off_grid'],
                      'isolated_first_noisy_peak': summary['peaks']['isolated_noisy']['global'],
                      'three_layer_first_noisy_peaks':
                          summary['peaks']['three_layer_noisy']['two_distinct_depths'],
                      'monte_carlo_trial_count': trials}, indent=2))
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=DEFAULT_INPUT)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--isolated-photons', type=float, default=1000.)
    parser.add_argument('--seed', type=int, default=20261102)
    parser.add_argument('--trials', type=int, default=12)
    args = parser.parse_args()
    run(args.source, args.output, args.isolated_photons, args.seed, args.trials)
