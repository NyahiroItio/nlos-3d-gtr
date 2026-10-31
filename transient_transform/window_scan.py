"""Window-length scan for the quadratic temporal fit on the physical S motion."""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from threadpoolctl import threadpool_limits

from .motion_simulation import translation_sequence
from .original_lct import OriginalLCT, SceneSettings
from .temporal_fit import fit_weights


DEFAULT_OUTPUT = Path(__file__).resolve().parent/'results'/'window_scan'
FORWARD_CACHE = Path(__file__).resolve().parent/'results'/'dynamic'/'translation'/'forward_cache.npz'


def window_indices(frame_count=7):
    """Nested odd windows about the central scan at fixed cadence."""
    if frame_count < 3 or frame_count % 2 == 0:
        raise ValueError('Use an odd sequence of at least three frames.')
    middle = frame_count//2
    return [np.arange(middle-radius, middle+radius+1)
            for radius in range(1, middle+1)]


def fit_two_coefficients(frames, indices, weights, exposure):
    """Keep center and slope; quadratic curvature remains a nuisance coefficient."""
    result = np.zeros((2,)+frames[int(indices[0])].shape, dtype=np.float64)
    for column, index in enumerate(indices):
        frame = frames[index]
        result += weights[:2, column].reshape((2,)+(1,)*frame.ndim)*frame/exposure
    return result


def plot_report(rows, output):
    fig, axes = plt.subplots(2, 2, figsize=(10, 7), constrained_layout=True)
    names = ('center state', 'signed derivative')
    budgets = ('fixed_exposure_per_frame', 'fixed_total_exposure')
    for column, budget in enumerate(budgets):
        subset = [row[budget] for row in rows]
        x = [row['half_window_s'] for row in rows]
        for coefficient, name in enumerate(names):
            ax = axes[coefficient, column]
            for key, label, style in (('relative_temporal_bias', 'time bias', 'o--'),
                                      ('relative_noise_rms', 'shot noise', 's--'),
                                      ('relative_total_rms_error', 'total RMS', 'o-')):
                ax.plot(x, [100*item[key][coefficient] for item in subset],
                        style, label=label)
            ax.set_title(f'{name} / {budget.replace("_", " ")}')
            ax.set_xlabel('Half-window H (s); 3, 5, 7 scans')
            ax.set_ylabel('Relative error (%)')
            ax.set_xticks(x)
            ax.grid(alpha=.25)
            ax.legend(fontsize=8)
    fig.suptitle('Quadratic fit / moving S / fixed 0.1 s synthetic scan cadence')
    fig.savefig(output/'window_scan.png', dpi=180)
    fig.savefig(output/'window_scan.svg')
    plt.close(fig)


def run(output=DEFAULT_OUTPUT, forward_cache=FORWARD_CACHE, trials=12, seed=20261031):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    times = np.linspace(-.3, .3, 7)
    settings = SceneSettings(0, 'physical_s_translation', .35, 64, 512, 16e-12,
                             2.4576, True, .08, 150, 25)
    means, derivative, _, _, metadata = translation_sequence(times, settings, forward_cache)
    model = OriginalLCT(settings.n, settings.m, settings.width_m, settings.range_m,
                        settings.diffuse, settings.snr)
    reference = np.stack((model(means[len(times)//2]), model(derivative)))
    norms = np.sum(reference**2, axis=(1, 2, 3))
    cases = []
    for indices in window_indices(len(times)):
        window_times = times[indices]
        weights = fit_weights(window_times, degree=2)
        expected_input = fit_two_coefficients(means, indices, weights, 1.)
        expected = np.stack([model(value) for value in expected_input])
        bias_energy = np.sum((expected-reference)**2, axis=(1, 2, 3))
        cases.append({'indices': indices, 'weights': weights, 'expected': expected,
                      'bias_energy': bias_energy,
                      'noise_energy': {'fixed_exposure_per_frame': [], 'fixed_total_exposure': []},
                      'total_energy': {'fixed_exposure_per_frame': [], 'fixed_total_exposure': []}})
        print(f'Noiseless H={window_times[-1]:.1f} s, N={len(indices)}: '
              f'bias={np.sqrt(bias_energy/norms)}', flush=True)
    rng = np.random.default_rng(seed)
    for trial in range(trials):
        shared_counts = [rng.poisson(mean) for mean in means]
        for case in cases:
            indices, weights = case['indices'], case['weights']
            count = len(indices)
            for budget in ('fixed_exposure_per_frame', 'fixed_total_exposure'):
                if budget == 'fixed_exposure_per_frame':
                    frames, exposure = shared_counts, 1.
                else:
                    exposure = len(times)/count
                    frames = {int(index): rng.poisson(exposure*means[index]) for index in indices}
                fitted_input = fit_two_coefficients(frames, indices, weights, exposure)
                fitted = np.stack([model(value) for value in fitted_input])
                case['noise_energy'][budget].append(np.sum((fitted-case['expected'])**2,
                                                           axis=(1, 2, 3)))
                case['total_energy'][budget].append(np.sum((fitted-reference)**2,
                                                           axis=(1, 2, 3)))
        if (trial+1) % 3 == 0 or trial+1 == trials:
            print(f'  {trial+1}/{trials} shared noisy sequences', flush=True)
    rows = []
    for case in cases:
        indices = case['indices']
        row = {'half_window_s': float(times[indices[-1]]), 'frames': len(indices),
               'time_indices': indices.tolist(),
               'relative_temporal_bias': np.sqrt(case['bias_energy']/norms).tolist()}
        for budget in ('fixed_exposure_per_frame', 'fixed_total_exposure'):
            exposure = 1. if budget == 'fixed_exposure_per_frame' else len(times)/len(indices)
            row[budget] = {
                'exposure_per_frame': exposure,
                'expected_total_photons': float(exposure*means[indices].sum()),
                'relative_temporal_bias': row['relative_temporal_bias'],
                'relative_noise_rms': np.sqrt(np.mean(case['noise_energy'][budget], axis=0)/norms).tolist(),
                'relative_total_rms_error': np.sqrt(np.mean(case['total_energy'][budget], axis=0)/norms).tolist()}
        rows.append(row)
    report = {'scene': 'Independent physical moving S; v=(0.08, 0, 0) m/s',
              'settings': settings.metadata(), 'forward_model': metadata,
              'times_s': times.tolist(), 'trials': trials, 'seed': seed,
              'reference': 'Instantaneous noiseless author LCT center and signed derivative, not true density.',
              'budget_note': 'Fixed per-frame exposure uses nested observations from each same noisy sequence; '
                             'its total photons increase with the number of scans. Fixed total exposure '
                             'uses seven exposure units per window, sampled independently. The 0.1 s '
                             'cadence is synthetic and does not establish hardware feasibility.',
              'rows': rows}
    (output/'report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    plot_report(rows, output)
    print(f'Results: {output.resolve()}', flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument('--forward-cache', type=Path, default=FORWARD_CACHE)
    parser.add_argument('--trials', type=int, default=12)
    parser.add_argument('--seed', type=int, default=20261031)
    parser.add_argument('--threads', type=int, default=2)
    args = parser.parse_args()
    if args.trials < 2 or args.threads < 1:
        parser.error('Use at least two trials and a positive thread count.')
    with threadpool_limits(args.threads):
        run(args.output, args.forward_cache, args.trials, args.seed)


if __name__ == '__main__':
    main()
