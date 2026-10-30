"""Run an independent, non-iterative fixed-voxel transient experiment."""

import argparse
import json
from pathlib import Path
from time import perf_counter

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy import sparse, fft
from threadpoolctl import threadpool_limits

from .operator import Grid, build_operator, make_geometry, make_scene
from .lct import LCTInverse
from .spectral import SpectralInverse
from .temporal import baseline_weights, coefficient_variance, fit_transients, fit_weights


def ratio_db(signal_energy, error_energy):
    if signal_energy == 0:
        return None
    return float(10 * np.log10(signal_energy / max(error_energy, 1e-300)))


def summarize(expected, truth, variance, noise_samples, reconstruction):
    bias_energy = float(np.sum((expected - truth)**2))
    empirical_noise = float(np.mean(np.sum((noise_samples - expected)**2, axis=1)))
    signal_energy = float(np.sum(truth**2))
    return {
        'theoretical_noise_rms_l2': float(np.sqrt(variance)),
        'empirical_noise_rms_l2': float(np.sqrt(empirical_noise)),
        'empirical_to_theoretical_noise_energy': empirical_noise / variance,
        'deterministic_bias_l2': float(np.sqrt(bias_energy)),
        'relative_deterministic_bias': float(np.sqrt(bias_energy / signal_energy)),
        'expected_shape_cosine': float(np.dot(expected, truth) / max(np.linalg.norm(expected)*np.linalg.norm(truth), 1e-30)),
        'expected_amplitude_projection': float(np.dot(expected, truth) / signal_energy),
        'output_signal_snr_db': ratio_db(float(np.sum(expected**2)), variance),
        'expected_reconstruction_snr_db_including_bias': ratio_db(signal_energy, variance + bias_energy),
        'single_draw_relative_error': float(np.linalg.norm(reconstruction - truth) / np.linalg.norm(truth)),
    }


def simulate_case(case, grid, operator, inverse, times, args, rng, directory):
    directory.mkdir(parents=True, exist_ok=True)
    frames, reference, derivative = make_scene(grid, times, case)
    signal = np.asarray(operator @ frames.T).T
    total = signal + args.background
    bg_variance = args.background / args.background_repeats
    background_estimate = rng.poisson(args.background_repeats * args.background,
                                     size=signal.shape[1]) / args.background_repeats
    counts = rng.poisson(total)
    weights = {'baseline': baseline_weights(times), 'window': fit_weights(times)}
    estimates, expected, variances, fitted = {}, {}, {}, {}
    start = perf_counter()
    for name, w in weights.items():
        fitted[name] = fit_transients(counts, background_estimate, w)
        estimates[name] = inverse(fitted[name])
        expected[name] = inverse(w @ signal)
        variances[name] = inverse.noise_energy(coefficient_variance(total, w, bg_variance))
    online_seconds = perf_counter() - start

    # Fit-then-invert must agree with invert-then-fit for the same fixed operator.
    reconstructed_frames = inverse(counts - background_estimate)
    commute_error = np.linalg.norm(estimates['window'] - weights['window'] @ reconstructed_frames)
    commute_error /= max(np.linalg.norm(estimates['window']), 1e-30)
    true_local_coefficients = weights['window'] @ frames
    temporal_bias = [float(np.linalg.norm(true_local_coefficients[i] - target)
                           / np.linalg.norm(target))
                     for i, target in enumerate((reference, derivative))]

    # Independent repeated count realizations, including an independently
    # remeasured shared background in every trial. No refitting of regularization.
    samples = {name: [] for name in weights}
    mc_start = perf_counter()
    for first in range(0, args.trials, 8):
        batch = min(8, args.trials - first)
        draw = rng.poisson(total, size=(batch,) + total.shape).astype(float)
        bg = rng.poisson(args.background_repeats * args.background,
                         size=(batch, total.shape[1])) / args.background_repeats
        corrected = draw - bg[:, None, :]
        for name, w in weights.items():
            coefficients = np.einsum('cf,bfm->bcm', w, corrected)
            samples[name].append(inverse(coefficients))
    samples = {name: np.concatenate(items) for name, items in samples.items()}
    metrics = {}
    for name in weights:
        metrics[name] = {
            field: summarize(expected[name][i], target, variances[name][i],
                             samples[name][:, i], estimates[name][i])
            for i, (field, target) in enumerate((('reference', reference), ('derivative', derivative)))
        }
    roi = np.flatnonzero(np.abs((weights['window'] @ signal)[1]) > 0)
    diagnostic_bin = int(roi[np.argmax(np.abs((weights['window'] @ signal)[1, roi]))])
    result = {
        'case': case,
        'scope': ('Exactly affine Eulerian density; not constant material velocity.'
                  if case == 'linear_density' else
                  'Constant XYZ material translation; affine temporal fit is an approximation.'),
        'expected_signal_counts_per_frame': signal.sum(axis=1).tolist(),
        'expected_total_signal_counts': float(signal.sum()),
        'expected_background_counts_per_frame': float(args.background * signal.shape[1]),
        'commutation_relative_error': float(commute_error),
        'pure_temporal_model_relative_bias_reference_derivative': temporal_bias,
        'metrics': metrics,
        'online_seconds_for_two_methods': online_seconds,
        'monte_carlo_seconds': perf_counter() - mc_start,
        'window_to_baseline_theoretical_noise_rms': {
            field: float(np.sqrt(variances['window'][i] / variances['baseline'][i]))
            for i, field in enumerate(('reference', 'derivative'))
        },
    }
    np.savez_compressed(directory / 'experiment.npz',
                        slow_times_t=times, rho_frames=frames, rho_reference=reference,
                        rho_derivative=derivative, expected_signal_counts=signal,
                        raw_counts=counts, expected_background=args.background,
                        background_estimate=background_estimate,
                        fit_weights=weights['window'], baseline_weights=weights['baseline'],
                        window_fitted_transients=fitted['window'],
                        baseline_reconstruction=estimates['baseline'],
                        window_reconstruction=estimates['window'],
                        expected_window_reconstruction=expected['window'],
                        diagonal_input_variance_window=coefficient_variance(total, weights['window'], bg_variance))
    visual = dict(truth=(reference, derivative), estimates=estimates,
                  diagnostic_signal=signal[:, diagnostic_bin],
                  diagnostic_counts=counts[:, diagnostic_bin] - background_estimate[diagnostic_bin],
                  diagnostic_fit=fitted['window'][:, diagnostic_bin])
    return result, visual


def window_tradeoff(grid, operator, inverse, args):
    """Expectation/covariance only: vary the local window of a translating object."""
    records = []
    for multiplier in (.25, .5, 1., 1.5):
        half = args.half_window * multiplier
        times = np.linspace(-half, half, args.frames)
        frames, reference, derivative = make_scene(grid, times, 'translation')
        weights = fit_weights(times)
        signal = np.asarray(operator @ frames.T).T
        expected = inverse(weights @ signal)
        noise_energy = inverse.noise_energy(coefficient_variance(
            signal + args.background, weights, args.background/args.background_repeats))
        temporal = weights @ frames
        record = {'half_window_s': half, 'full_scans': args.frames}
        for i, (name, truth) in enumerate((('reference', reference), ('derivative', derivative))):
            norm = np.linalg.norm(truth)
            bias = np.linalg.norm(expected[i] - truth) / norm
            noise = np.sqrt(noise_energy[i]) / norm
            record[name] = {'pure_temporal_relative_bias': float(np.linalg.norm(temporal[i]-truth)/norm),
                            'total_relative_bias': float(bias), 'relative_noise_rms': float(noise),
                            'expected_relative_rmse': float(np.hypot(bias, noise))}
        records.append(record)
    return records


def aperture_diagnostic(inverse, wall):
    """Matched transformed-domain experiment isolating measurement truncation.

    This diagnostic is not used to calibrate the inverse or rescale its output.
    Full padded measurements are a mathematical control unavailable to the
    actual finite-aperture scan. Both outputs are compared on the same ROI.
    """
    native_grid = Grid(np.unique(wall[:, 0]), np.unique(wall[:, 1]), np.sqrt(inverse.u))
    rho = make_scene(native_grid, [0.], 'linear_density')[1].reshape(inverse.nx, inverse.ny, inverse.nu)
    depth = 2*np.sqrt(inverse.u)[None, None, :]
    q = np.divide(rho, depth, out=np.zeros_like(rho), where=depth > 0)
    padded = np.zeros(inverse.fft_shape)
    padded[:inverse.nx, :inverse.ny, :inverse.nu] = q
    full_data = fft.irfftn(fft.rfftn(padded)*inverse.forward_spectrum, s=inverse.fft_shape)
    control = fft.irfftn(fft.rfftn(full_data)*inverse.inverse_filter, s=inverse.fft_shape)
    control = control[:inverse.nx, :inverse.ny, :inverse.nu]
    cropped = inverse._convolve(full_data[None, :inverse.nx, :inverse.ny, :inverse.nu])[0]
    return {
        'full_padded_measurements_relative_error': float(np.linalg.norm(control-q)/np.linalg.norm(q)),
        'cropped_and_zero_padded_measurements_relative_error': float(np.linalg.norm(cropped-q)/np.linalg.norm(q)),
        'note': 'Matched H, same source, same filter, same ROI. Only the observed data domain differs. '
                'This isolates truncation bias in transformed space, not every physical-model error.'}


def plot_results(grid, times, results, visuals, output):
    fig, axes = plt.subplots(4, 3, figsize=(12, 12), constrained_layout=True)
    extent = [grid.x[0], grid.x[-1], grid.z[0], grid.z[-1]]
    for case_idx, (case, visual) in enumerate(visuals.items()):
        for component in range(2):
            row = 2*case_idx + component
            fields = [visual['truth'][component], visual['estimates']['baseline'][component],
                      visual['estimates']['window'][component]]
            images = [field.reshape(grid.shape).sum(axis=1).T for field in fields]
            vmax = max(float(np.max(np.abs(img))) for img in images)
            for column, image in enumerate(images):
                ax = axes[row, column]
                im = ax.imshow(image, origin='lower', extent=extent, aspect='auto',
                               cmap='RdBu_r' if component else 'viridis',
                               vmin=-vmax if component else min(float(min(img.min() for img in images)), 0),
                               vmax=vmax)
                ax.set_title(f'{case} / {"rho(t=0)" if component == 0 else "d rho / dt"}\n'
                             + ['Truth', 'Single frame / adjacent difference', 'Whole-window linear fit'][column],
                             fontsize=9)
                ax.set_xlabel('x (m)')
                ax.set_ylabel('z (m)')
                fig.colorbar(im, ax=ax, shrink=0.75)
    fig.savefig(output / 'reconstruction.png', dpi=160)
    plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(11, 7), constrained_layout=True)
    for row, (case, visual) in enumerate(visuals.items()):
        ax = axes[row, 0]
        ax.plot(times, visual['diagnostic_signal'], 'k-', label='True transient bin')
        ax.plot(times, visual['diagnostic_counts'], 'o', label='Measured minus background')
        ax.plot(times, visual['diagnostic_fit'][0] + times*visual['diagnostic_fit'][1], '--',
                label='Direct temporal fit')
        ax.set_title(case + ': diagnostic bin with largest true slope')
        ax.set_xlabel('Slow time t (s)')
        ax.set_ylabel('Counts / bin')
        ax.legend(fontsize=8)
        ax = axes[row, 1]
        labels, theoretical, empirical = [], [], []
        for method in ('baseline', 'window'):
            for field in ('reference', 'derivative'):
                metric = results[case]['metrics'][method][field]
                normalization = results[case]['metrics']['baseline'][field]['theoretical_noise_rms_l2']
                labels.append(method + '\n' + field)
                theoretical.append(metric['theoretical_noise_rms_l2'] / normalization)
                empirical.append(metric['empirical_noise_rms_l2'] / normalization)
        positions = np.arange(4)
        ax.bar(positions - .18, theoretical, .36, label='Covariance prediction')
        ax.bar(positions + .18, empirical, .36, label='Monte Carlo')
        ax.set_xticks(positions, labels)
        ax.set_title(case + ': noise propagation validation')
        ax.set_ylabel('Noise RMS / baseline theory\n(normalized separately per component)', fontsize=9)
        ax.legend(fontsize=8)
    fig.savefig(output / 'temporal_and_noise.png', dpi=160)
    plt.close(fig)


def plot_tradeoff(records, output):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    widths = [2*r['half_window_s'] for r in records]
    for ax, component in zip(axes, ('reference', 'derivative')):
        for field, label in (('total_relative_bias', 'Bias (temporal + inverse)'),
                             ('relative_noise_rms', 'Noise RMS'),
                             ('expected_relative_rmse', 'Expected RMSE')):
            ax.plot(widths, [r[component][field] for r in records], 'o-', label=label)
        ax.set_title('Translation: ' + component)
        ax.set_xlabel('Full slow-time window (s); 7 scans by default')
        ax.set_ylabel('Relative L2 error')
        ax.legend(fontsize=8)
        ax.grid(alpha=.2)
    fig.savefig(output / 'window_tradeoff.png', dpi=160)
    plt.close(fig)


def run(args):
    if args.frames < 3 or args.frames % 2 == 0 or args.trials < 2:
        raise ValueError('Use an odd frame count >= 3 and at least two Monte Carlo trials.')
    if args.half_window <= 0 or args.peak_counts <= 0 or args.background < 0 or args.background_repeats < 1:
        raise ValueError('Check window, photon counts, and background calibration settings.')
    output = Path(args.output or f'transient_transform/results/{args.inverse}_demo').resolve()
    output.mkdir(parents=True, exist_ok=True)
    times = np.linspace(-args.half_window, args.half_window, args.frames)
    grid, wall, fast_edges = make_geometry()
    start = perf_counter()
    physical = build_operator(grid, wall, fast_edges)
    reference = make_scene(grid, times, 'linear_density')[1]
    # Synthetic exposure calibration: choose one global gain, then hold it fixed
    # for every frame, both cases, all inverse methods and Monte Carlo trials.
    gain = args.peak_counts / np.max(physical @ reference)
    operator = (physical * gain).tocsr()
    build_seconds = perf_counter() - start
    print(f'Operator: {operator.shape}, {operator.nnz:,} nonzeros', flush=True)
    start = perf_counter()
    if args.inverse == 'lct':
        inverse = LCTInverse(grid, wall, fast_edges, gain, args.lct_ridge, args.lct_bins)
        inverse_info = {'backend': 'lct', 'relative_ridge': args.lct_ridge,
                        'absolute_ridge': inverse.ridge, 'native_shape': (inverse.nx, inverse.ny, inverse.nu),
                        'fft_shape': inverse.fft_shape, 'dense_inverse_stored': False,
                        'filter_memory_mib': inverse.inverse_filter.nbytes / 2**20,
                        'positivity_clamp': False, 'irf_deconvolution': False}
    else:
        inverse = SpectralInverse(operator, args.ridge)
        inverse_info = {'backend': 'spectral', 'relative_ridge': args.ridge,
                        'absolute_ridge': inverse.ridge,
                        'effective_degrees_of_freedom': inverse.effective_dof,
                        'singular_values': inverse.singular_values.tolist(),
                        'kernel_memory_mib': inverse.matrix.nbytes / 2**20}
    factor_seconds = perf_counter() - start
    print(f'{args.inverse} inverse prepared in {factor_seconds:.2f} s', flush=True)
    # Exact covariance is a diagnostic, not part of online reconstruction.
    start = perf_counter()
    inverse.noise_energy(np.zeros(operator.shape[0]))
    covariance_seconds = perf_counter() - start
    rng = np.random.default_rng(args.seed)
    report = {
        'seed': args.seed,
        'method': f'Fixed XYZ voxel density, physical confocal bin-integrated data, direct {args.inverse} inverse.',
        'not_implemented': ['opaque-surface self-occlusion',
                            'material velocity estimation', 'SPAD first-event pile-up or dead time',
                            'real hardware calibration'],
        'inverse_model_note': ('LCT uses the ideal continuous confocal identity and fixed interpolation; '
                               'it does not invert the exact sparse simulator A or its IRF. '
                               'Finite-aperture zero padding and voxel-center quadrature can produce '
                               'large amplitude/shape bias. Absolute gain is physical, not fitted to truth.'
                               if args.inverse == 'lct' else
                               'Spectral inversion uses the same discrete A as simulation, including IRF.'),
        'geometry': {'grid_shape': grid.shape, 'wall_points': len(wall),
                     'fast_bins': len(fast_edges)-1, 'fast_bin_width_s': float(np.diff(fast_edges)[0]),
                     'irf_sigma_s': 60e-12, 'operator_shape': operator.shape,
                     'operator_nnz': operator.nnz, 'synthetic_global_gain': float(gain)},
        'settings': vars(args),
        'inverse': inverse_info,
        'timing': {'operator_seconds': build_seconds, 'inverse_preparation_seconds': factor_seconds,
                   'exact_covariance_preparation_seconds': covariance_seconds},
        'budget_note': 'Same acquired sequence and gain for all methods. Window reference uses all frames; '
                       'baseline reference uses one, derivative uses two adjacent frames. '
                       'Noise reduction consumes more observations and is not a same-two-frame photon advantage.',
        'noise_model': 'Independent Poisson(signal+background) frames; a shared independent background '
                       'estimate from background_repeats exposures. Shared-background covariance is included.',
        'cases': {},
    }
    visuals = {}
    for case in ('linear_density', 'translation'):
        print(f'Simulating {case} and {args.trials} noise repetitions...', flush=True)
        result, visual = simulate_case(case, grid, operator, inverse, times, args, rng, output / case)
        report['cases'][case], visuals[case] = result, visual
        print(f'  window/baseline noise RMS: {result["window_to_baseline_theoretical_noise_rms"]}', flush=True)
        print('  window noiseless relative bias (rho, derivative): '
              + str([round(result['metrics']['window'][field]['relative_deterministic_bias'], 4)
                     for field in ('reference', 'derivative')]), flush=True)
    report['checks'] = {
        'fit_and_reconstruction_commute': all(r['commutation_relative_error'] < 1e-10
                                             for r in report['cases'].values()),
        'affine_temporal_model_exact': max(report['cases']['linear_density']
                                          ['pure_temporal_model_relative_bias_reference_derivative']) < 1e-10,
        'monte_carlo_matches_covariance': all(
            0.7 < metric['empirical_to_theoretical_noise_energy'] < 1.3
            for result in report['cases'].values() for method in result['metrics'].values()
            for metric in method.values()),
        'signed_derivative_preserved': all(np.min(v['estimates']['window'][1]) < 0 <
                                          np.max(v['estimates']['window'][1]) for v in visuals.values()),
    }
    report['translation_window_tradeoff'] = window_tradeoff(grid, operator, inverse, args)
    if args.inverse == 'lct':
        report['aperture_diagnostic'] = aperture_diagnostic(inverse, wall)
    report['validation_scope'] = (
        'Checks validate signs, linearity and noise propagation, not accurate physical reconstruction. '
        'Inspect deterministic bias and amplitude projection before judging reconstruction quality.')
    report['window_tradeoff_note'] = (
        'Separate hypothetical windows with the same number of full scans '
        'and the same per-scan gain. Exposure durations and scan switching '
        'overheads are not simulated; this is not a hardware frame-rate claim.')
    plot_results(grid, times, report['cases'], visuals, output)
    plot_tradeoff(report['translation_window_tradeoff'], output)
    np.savez_compressed(output / 'geometry.npz', x=grid.x, y=grid.y, z=grid.z,
                        wall_points=wall, fast_edges_s=fast_edges)
    sparse.save_npz(output / 'operator.npz', operator)
    (output / 'report.json').write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding='utf-8')
    print(json.dumps(report['checks']), flush=True)
    print(f'Results: {output}', flush=True)
    if not all(report['checks'].values()):
        raise RuntimeError('A demo validation check failed; inspect report.json.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', default=None, help='Default: results/<inverse>_demo inside this package.')
    parser.add_argument('--inverse', choices=('lct', 'spectral'), default='lct')
    parser.add_argument('--seed', type=int, default=20261029)
    parser.add_argument('--frames', type=int, default=7)
    parser.add_argument('--half-window', type=float, default=0.3, help='Slow-time half-window in seconds.')
    parser.add_argument('--peak-counts', type=float, default=150.0)
    parser.add_argument('--background', type=float, default=1.5, help='Mean counts per fast bin per scan.')
    parser.add_argument('--background-repeats', type=int, default=32)
    parser.add_argument('--ridge', type=float, default=1e-5)
    parser.add_argument('--lct-ridge', type=float, default=1e-4, help='Relative to the maximum FFT kernel power.')
    parser.add_argument('--lct-bins', type=int, default=256, help='Uniform squared-range grid samples.')
    parser.add_argument('--trials', type=int, default=80)
    parser.add_argument('--threads', type=int, default=2)
    args = parser.parse_args()
    if args.threads < 1:
        parser.error('--threads must be positive.')
    with threadpool_limits(limits=args.threads):
        run(args)


if __name__ == '__main__':
    main()
