"""Shape-blind finite-motion dictionary probe with held-out wall measurements.

Only calibrated confocal geometry, IRF, search bounds and constant point
velocities are supplied to the detector. Scene truth is used for scoring only.
"""

import argparse
import itertools
import json
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares, nnls
from scipy.special import ndtr
from threadpoolctl import threadpool_limits

from .motion_simulation import confocal_response, glyph_points


OUTPUT = Path(__file__).resolve().parent/'results'/'blind_motion_probe'
PEAK_COUNTS = (1, 2, 10, 50)


def plot_report(report, output):
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib import pyplot as plt
    fig, axes = plt.subplots(1, 4, figsize=(14, 3.7))
    for ax, (name, scene) in zip(axes, report['summary'].items()):
        for mode, label in [('full', 'Full five frames'), ('slope', 'Finite slope')]:
            ax.plot(PEAK_COUNTS, [100*scene[mode]['noisy_derivative_rms_by_K'][str(k)]
                                   for k in PEAK_COUNTS], 'o-', label=label)
        ax.set(title=name.replace('_', ' '), xlabel='Number of selected peaks',
               ylabel='Held-out derivative RMS error (%)')
        ax.grid(alpha=.2)
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output, dpi=150)
    fig.savefig(output.with_suffix('.svg'))
    plt.close(fig)


def frames(parameters, wall, times, edges):
    """Vectorized unit-point templates, independent of the source simulator."""
    parameters = np.atleast_2d(parameters)
    positions = parameters[:, None, :3]+times[None, :, None]*parameters[:, None, 3:]
    radius = np.linalg.norm(positions[:, :, None]-wall[None, None], axis=-1)
    sigma = 60e-12/(2*np.sqrt(2*np.log(2)))
    probabilities = np.diff(ndtr(
        (edges[None, None, None]-2*radius[:, :, :, None]/3e8)/sigma), axis=-1)
    return probabilities/radius[:, :, :, None]**4


def generate(points, mass, velocities, wall, times, edges, photons):
    response, derivative = confocal_response(points, mass, wall, edges, velocities)
    gain = photons/response.sum()
    sequence = [gain*confocal_response(points+t*velocities, mass, wall, edges)[0]
                for t in times]
    return np.array(sequence), gain*derivative


def representation(array, mode, weights):
    if mode == 'full':
        return array-array.mean(axis=-3, keepdims=True)
    return np.einsum('t,...twb->...wb', weights, array)


def spatial_peaks(scores, parameters, count=50, separation=.015):
    selected = []
    for index in np.argsort(scores)[::-1]:
        if scores[index] <= 0:
            break
        if all(np.linalg.norm(parameters[index, :3]-parameters[j, :3]) >= separation
               for j in selected):
            selected.append(int(index))
        if len(selected) == count:
            break
    return selected


def relative(estimate, truth):
    return float(np.linalg.norm(estimate-truth)/np.linalg.norm(truth))


def refine_single(start, observation, mode, wall, train, times, edges,
                  weights, truth):
    """Continuous fit from the detected peak, solely to diagnose grid error."""
    signal = representation(observation[:, train], mode, weights)
    scale = np.linalg.norm(signal)

    def residual(parameter):
        model = representation(frames(parameter, wall[train], times, edges)[0],
                               mode, weights)
        gain = max(0., np.sum(model*signal)/np.sum(model**2))
        return ((gain*model-signal)/scale).ravel()

    fit = least_squares(residual, start,
        bounds=([-.12, -.12, .59, -.4, -.4, -.08],
                [.12, .12, .71, .4, .4, .08]),
        x_scale=[.04, .04, .02, .2, .2, .08],
        max_nfev=100, ftol=1e-10, xtol=1e-10, gtol=1e-10)
    model = frames(fit.x, wall, times, edges)[0]
    kernel = representation(model[:, train], mode, weights)
    gain = max(0., np.sum(kernel*signal)/np.sum(kernel**2))
    derivative = gain*confocal_response(fit.x[None, :3], np.ones(1), wall,
                                        edges, fit.x[None, 3:])[1]
    return {'parameters': fit.x.tolist(), 'function_evaluations': fit.nfev,
            'success': bool(fit.success),
            'heldout_derivative_error': relative(derivative[~train],
                                                 truth['derivative'][~train])}


def run(output=OUTPUT, trials=4, seed=20261104):
    if trials < 1:
        raise ValueError('At least one Poisson trial is required.')
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    times = np.linspace(-.2, .2, 5)
    weights = times/np.dot(times, times)
    axis = np.linspace(-.35, .35, 5)
    wx, wy = np.meshgrid(axis, axis, indexing='xy')
    wall = np.column_stack((wx.ravel(), wy.ravel(), np.zeros(wx.size)))
    train = (np.indices((5, 5)).sum(axis=0).ravel() % 2 == 0)
    edges = np.arange(90, 201)*32e-12
    positions = np.array(list(itertools.product(np.linspace(-.12, .12, 7),
                                                np.linspace(-.12, .12, 7),
                                                np.linspace(.59, .71, 7))))
    velocities = np.array([v for v in itertools.product([-.4, -.2, 0, .2, .4],
                                                        [-.4, -.2, 0, .2, .4],
                                                        [-.08, 0, .08]) if np.any(v)])
    parameters = np.array([np.r_[p, v] for p in positions for v in velocities])
    p1, v1 = np.array([-.04, .04, .63]), np.array([.2, 0, .08])
    p2, v2 = np.array([.04, -.04, .67]), np.array([-.2, .2, -.08])
    glyph, mass = glyph_points()
    center = np.average(glyph, axis=0, weights=mass)
    radial = glyph-center
    radial[:, 2] = 0
    glyph_velocity = .4*radial/np.max(np.linalg.norm(radial, axis=1))
    sources = {
        'single_on_grid': (p1[None], np.ones(1), v1[None]),
        'single_off_grid': (np.array([[-.037, .043, .633]]), np.ones(1),
                            np.array([[.18, .02, .07]])),
        'two_points': (np.array([p1, p2]), np.array([1., .8]), np.array([v1, v2])),
        'unknown_S': (glyph, mass, glyph_velocity)}
    rng = np.random.default_rng(seed)
    observations, truths, labels = [], {}, []
    for name, (points, masses, velocity) in sources.items():
        mean, derivative = generate(points, masses, velocity, wall, times, edges, 50000.)
        truths[name] = {'mean': mean, 'derivative': derivative,
                        'points': points, 'velocities': velocity}
        for trial in range(trials+1):
            observations.append(mean if trial == 0 else rng.poisson(mean).astype(float))
            labels.append((name, trial))
    observations = np.array(observations)
    scores = {mode: np.empty((len(observations), len(parameters)))
              for mode in ('full', 'slope')}
    with threadpool_limits(limits=2):
        data = {mode: representation(observations[:, :, train], mode, weights).reshape(
                    len(observations), -1)
                for mode in scores}
        for start in range(0, len(parameters), 32):
            bank = frames(parameters[start:start+32], wall[train], times, edges)
            for mode in scores:
                kernel = representation(bank, mode, weights).reshape(len(bank), -1)
                norm = np.linalg.norm(kernel, axis=1)
                scores[mode][:, start:start+len(bank)] = data[mode]@kernel.T/norm[None]
            if start % 3200 == 0:
                print(f'  blind dictionary {start}/{len(parameters)}', flush=True)
        rows = []
        for observation_index, (name, trial) in enumerate(labels):
            for mode in scores:
                peaks = spatial_peaks(scores[mode][observation_index], parameters)
                bank = frames(parameters[peaks], wall, times, edges)
                template = representation(bank, mode, weights)
                signal = representation(observations[observation_index], mode, weights)
                analytic = []
                for parameter in parameters[peaks]:
                    analytic.append(confocal_response(parameter[None, :3], np.ones(1),
                                    wall, edges, parameter[None, 3:])[1])
                evaluations = {}
                for count in PEAK_COUNTS:
                    use = min(count, len(peaks))
                    design = template[:use, ..., train, :].reshape(use, -1).T
                    coefficient = nnls(design, signal[..., train, :].ravel())[0]
                    predicted = np.tensordot(coefficient, bank[:use], axes=(0, 0))
                    derivative = np.tensordot(coefficient, analytic[:use], axes=(0, 0))
                    expected = truths[name]
                    evaluations[str(count)] = {
                        'heldout_dynamic_error': relative(
                            representation(predicted[:, ~train], 'full', weights),
                            representation(expected['mean'][:, ~train], 'full', weights)),
                        'heldout_instantaneous_derivative_error': relative(
                            derivative[~train], expected['derivative'][~train])}
                row = {'scene': name, 'trial': trial, 'mode': mode,
                             'top_peaks': parameters[peaks].tolist(),
                             'evaluation_by_number_of_peaks': evaluations}
                if name == 'single_off_grid':
                    row['continuous_refinement_diagnostic'] = refine_single(
                        parameters[peaks[0]], observations[observation_index], mode,
                        wall, train, times, edges, weights, truths[name])
                rows.append(row)
        print('  blind scoring complete', flush=True)
    report = {'design': 'No source shape, center, lateral position, point count or common expansion field supplied to matcher. Generic constant-velocity point dictionary, followed by small nonnegative amplitude fit to fixed top-K spatial peaks.',
              'remaining_assumptions': 'Known wall and IRF, single scattering, no occlusion, constant point velocities, finite spatial/velocity bounds; zero velocities omitted because temporal centering removes static response.',
              'wall_points': len(wall), 'training_walls': int(train.sum()),
              'heldout_walls': int((~train).sum()), 'fast_bin_width_ps': 32,
              'times_s': times.tolist(), 'candidate_count': len(parameters),
              'velocity_candidates': velocities.tolist(),
              'spatial_grid': {'xy_step_m': .04, 'z_step_m': .02,
                               'bounds_m': [[-.12, .12], [-.12, .12], [.59, .71]]},
              'center_expected_photons_all_walls': 50000, 'poisson_trials': trials,
              'truth_for_evaluation_only': {name: {'points_m': p.tolist(),
                'velocities_m_per_s': v.tolist()} for name, (p, _, v) in sources.items()},
              'results': rows,
              'limitations': 'Coarse dictionary, one noise level and four scenes. Top-K counts are diagnostic and not selected using truth. One velocity is retained per spatial grid cell (15 mm suppression is below the 20 mm minimum spatial spacing). The small NNLS adds a joint amplitude solve after matching. A separate continuous nonlinear fit of the off-grid point diagnoses discretization error; it is iterative. Failure does not establish an impossibility result for a denser dictionary or a full joint inverse.'}
    (output/'report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    summary = {}
    for name in sources:
        summary[name] = {}
        for mode in scores:
            selected = [row for row in rows if row['scene'] == name and row['mode'] == mode]
            summary[name][mode] = {'clean_top_peak': selected[0]['top_peaks'][0],
                'clean_errors_by_K': selected[0]['evaluation_by_number_of_peaks'],
                'noisy_derivative_rms_by_K': {str(k): float(np.sqrt(np.mean([
                    row['evaluation_by_number_of_peaks'][str(k)][
                        'heldout_instantaneous_derivative_error']**2
                    for row in selected[1:]]))) for k in PEAK_COUNTS}}
    report['summary'] = summary
    report['off_grid_refinement'] = {mode: {
        'clean': [row['continuous_refinement_diagnostic'] for row in rows
                  if row['scene'] == 'single_off_grid' and row['mode'] == mode
                  and row['trial'] == 0][0],
        'noisy_derivative_rms': float(np.sqrt(np.mean([
            row['continuous_refinement_diagnostic']['heldout_derivative_error']**2
            for row in rows if row['scene'] == 'single_off_grid'
            and row['mode'] == mode and row['trial'] > 0])))}
        for mode in scores}
    (output/'report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    plot_report(report, output/'blind_errors.png')
    print(json.dumps({'noisy_RMS_K50': {name: {mode:
        value['noisy_derivative_rms_by_K']['50'] for mode, value in scene.items()}
        for name, scene in summary.items()},
        'off_grid_refinement': report['off_grid_refinement']}, indent=2), flush=True)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=OUTPUT)
    parser.add_argument('--trials', type=int, default=4)
    args = parser.parse_args()
    run(args.output, args.trials)
