"""Window-length scan for the quadratic temporal fit on the physical S motion."""

import argparse
import json
from pathlib import Path
import shutil
import subprocess

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from threadpoolctl import threadpool_limits

from .motion_simulation import confocal_response, translation_sequence
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


def sample_two_coefficients(means, indices, weights, exposure, seed, trial, fixed_total):
    """Replay frame seeds so nested windows share counts without retaining frames."""
    result = np.zeros((2,)+means.shape[1:], dtype=np.float64)
    for column, index in enumerate(indices):
        stream = np.random.default_rng(np.random.SeedSequence(
            [seed, trial, int(index), int(fixed_total), len(indices) if fixed_total else 0]))
        counts = stream.poisson(exposure*means[index])
        result += weights[:2, column, None, None, None]*counts/exposure
        del counts
    return result


def speed_sequence(times, settings, base_cache, speed, output):
    """Use the verified base sequence and simulate only newly needed offsets."""
    base_means, base_derivative, points, weights, base_metadata = translation_sequence(
        times, settings, base_cache)
    if speed == .08:
        return base_means, base_derivative, base_metadata
    metadata = dict(base_metadata)
    metadata['velocity_m_per_s'] = [speed, 0., 0.]
    metadata['base_velocity_m_per_s'] = [.08, 0., 0.]
    metadata['speed_forward_version'] = 1
    cache_key = {'settings': settings.metadata(), 'times': times.tolist(),
                 'speed_m_per_s': speed, 'base_generator_sha256': base_metadata['generator_sha256']}
    key = json.dumps(dict(cache_key, speed_forward_version=1), sort_keys=True)
    cache = output/'forward_cache.npz'
    if cache.exists():
        with np.load(cache, allow_pickle=False) as saved:
            saved_key = json.loads(str(saved['key']))
            # Accept the first cache version, whose only additional key was the
            # runner source hash; the forward physics and inputs are unchanged.
            if all(saved_key.get(name) == value for name, value in cache_key.items()) and (
                    saved_key.get('speed_forward_version', 1) == 1):
                print(f'Physical speed {speed:.3f} m/s: using verified cache.', flush=True)
                return saved['means'], saved['derivative'], metadata
    axis = np.linspace(-settings.width_m, settings.width_m, settings.n)
    wx, wy = np.meshgrid(axis[::-1], axis, indexing='xy')
    wall = np.column_stack((wx.ravel(), wy.ravel(), np.zeros(wx.size)))
    edges = np.arange(settings.m+1)*settings.bin_resolution_s
    gain = base_metadata['exposure_gain']
    shape = (settings.n, settings.n, settings.m)
    velocity = np.array([speed, 0., 0.])
    means = []
    for index, time in enumerate(times):
        base_time = time*speed/.08
        matches = np.flatnonzero(np.isclose(times, base_time, rtol=0, atol=1e-12))
        if len(matches):
            mean = base_means[matches[0]]
        else:
            response, _ = confocal_response(points+time*velocity, weights, wall, edges)
            mean = (gain*response).reshape(shape).transpose(2, 0, 1)
        means.append(mean)
        print(f'  speed={speed:.3f}: physical frame {index+1}/{len(times)}', flush=True)
    means = np.stack(means)
    derivative = base_derivative*(speed/.08)
    np.savez_compressed(cache, key=key, means=means, derivative=derivative)
    return means, derivative, metadata


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


def plot_speed_comparison(root=DEFAULT_OUTPUT):
    """Compare completed slow, original and fast physical-speed scans."""
    root = Path(root)
    paths = {0.04: root/'speed_004'/'report.json',
             0.08: root/'report.json',
             0.16: root/'speed_016'/'report.json'}
    reports = {speed: json.loads(path.read_text(encoding='utf-8'))
               for speed, path in paths.items()}
    fig, axes = plt.subplots(2, 2, figsize=(8, 5.6), constrained_layout=True)
    budgets = ('fixed_exposure_per_frame', 'fixed_total_exposure')
    for column, budget in enumerate(budgets):
        for coefficient, name in enumerate(('center state', 'signed derivative')):
            ax = axes[coefficient, column]
            for speed, report in reports.items():
                rows = report['rows']
                ax.plot([row['half_window_s'] for row in rows],
                        [100*row[budget]['relative_total_rms_error'][coefficient] for row in rows],
                        'o-', label=f'{speed:.2f} m/s')
            ax.set_title(f'{name} / {budget.replace("_", " ")}')
            ax.set_xlabel('Half-window H (s); 3, 5, 7 scans')
            ax.set_ylabel('Relative total RMS (%)')
            ax.set_xticks([.1, .2, .3])
            ax.grid(alpha=.25)
            ax.legend()
    fig.suptitle('Physical S translation / quadratic temporal fit / speed sensitivity')
    svg = root/'speed_comparison.svg'
    fig.savefig(svg)
    plt.close(fig)
    converter = shutil.which('rsvg-convert')
    if converter is not None:
        subprocess.run([converter, '-o', str(root/'speed_comparison.png'), str(svg)], check=True)
    print(f'Speed comparison: {root.resolve()}', flush=True)


def run(output=DEFAULT_OUTPUT, forward_cache=FORWARD_CACHE, trials=12, seed=20261031,
        speed=.08, max_new_trials=None):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    times = np.linspace(-.3, .3, 7)
    settings = SceneSettings(0, 'physical_s_translation', .35, 64, 512, 16e-12,
                             2.4576, True, .08, 150, 25)
    means, derivative, metadata = speed_sequence(times, settings, forward_cache, speed, output)
    model = OriginalLCT(settings.n, settings.m, settings.width_m, settings.range_m,
                        settings.diffuse, settings.snr)
    reference = np.stack((model(means[len(times)//2]), model(derivative)))
    del derivative
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
        del expected_input, expected
    progress_path = output/'progress.json'
    completed = 0
    if progress_path.exists():
        progress = json.loads(progress_path.read_text(encoding='utf-8'))
        if (progress['seed'], progress['speed'], progress['trials']) != (seed, speed, trials):
            raise ValueError('Saved progress uses different speed, seed or trial count.')
        completed = progress['completed']
        if len(progress['cases']) != len(cases) or not 0 <= completed <= trials:
            raise ValueError('Saved progress does not match the window design.')
        for case, saved in zip(cases, progress['cases']):
            for budget in ('fixed_exposure_per_frame', 'fixed_total_exposure'):
                if (len(saved['noise_energy'][budget]) != completed or
                        len(saved['total_energy'][budget]) != completed):
                    raise ValueError('Saved progress has an incomplete trial.')
                case['noise_energy'][budget] = [np.asarray(value) for value in saved['noise_energy'][budget]]
                case['total_energy'][budget] = [np.asarray(value) for value in saved['total_energy'][budget]]
        print(f'Resuming {completed}/{trials} completed noisy sequences.', flush=True)
    stop = trials if max_new_trials is None else min(trials, completed+max_new_trials)
    for trial in range(completed, stop):
        for case in cases:
            indices, weights = case['indices'], case['weights']
            count = len(indices)
            for budget in ('fixed_exposure_per_frame', 'fixed_total_exposure'):
                fixed_total = budget == 'fixed_total_exposure'
                exposure = len(times)/count if fixed_total else 1.
                fitted_input = sample_two_coefficients(means, indices, weights, exposure,
                                                       seed, trial, fixed_total)
                noise_energy = np.empty(2)
                total_energy = np.empty(2)
                for coefficient in range(2):
                    fitted = model(fitted_input[coefficient])
                    noise_energy[coefficient] = np.sum((fitted-case['expected'][coefficient])**2)
                    total_energy[coefficient] = np.sum((fitted-reference[coefficient])**2)
                    del fitted
                case['noise_energy'][budget].append(noise_energy)
                case['total_energy'][budget].append(total_energy)
                del fitted_input
        progress = {'seed': seed, 'speed': speed, 'trials': trials, 'completed': trial+1,
                    'cases': [{'noise_energy': {budget: [v.tolist() for v in case['noise_energy'][budget]]
                                                for budget in ('fixed_exposure_per_frame', 'fixed_total_exposure')},
                               'total_energy': {budget: [v.tolist() for v in case['total_energy'][budget]]
                                                for budget in ('fixed_exposure_per_frame', 'fixed_total_exposure')}}
                              for case in cases]}
        temporary = output/'progress.tmp'
        temporary.write_text(json.dumps(progress), encoding='utf-8')
        temporary.replace(progress_path)
        if (trial+1) % 3 == 0 or trial+1 == trials:
            print(f'  {trial+1}/{trials} shared noisy sequences', flush=True)
    if stop < trials:
        print(f'Batch finished: {stop}/{trials}; resume with the same command.', flush=True)
        return None
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
    report = {'scene': f'Independent physical moving S; v=({speed}, 0, 0) m/s',
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
    progress_path.unlink(missing_ok=True)
    print(f'Results: {output.resolve()}', flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument('--forward-cache', type=Path, default=FORWARD_CACHE)
    parser.add_argument('--trials', type=int, default=12)
    parser.add_argument('--seed', type=int, default=20261031)
    parser.add_argument('--threads', type=int, default=2)
    parser.add_argument('--speed', type=float, default=.08, help='Lateral material speed in m/s.')
    parser.add_argument('--max-new-trials', type=int, default=None,
                        help='Stop after this many new trials, saving resumable progress.')
    parser.add_argument('--compare-speeds', action='store_true',
                        help='Render the three completed speed reports without rerunning LCT.')
    args = parser.parse_args()
    if args.compare_speeds:
        plot_speed_comparison(args.output)
        return
    if (args.trials < 2 or args.threads < 1 or not np.isfinite(args.speed) or args.speed <= 0
            or (args.max_new_trials is not None and args.max_new_trials < 1)):
        parser.error('Use at least two trials, a positive thread count and a positive finite speed.')
    with threadpool_limits(args.threads):
        run(args.output, args.forward_cache, args.trials, args.seed, args.speed,
            args.max_new_trials)


if __name__ == '__main__':
    main()
