"""Validate the LTAI static mean using the verified original LCT backend.

The author histogram is a fixed empirical Poisson mean, not a noise-free
physical scene. All reconstruction errors are conditional errors relative
to LCT of that histogram. Repeated exposures do not remove its original noise.
"""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from threadpoolctl import threadpool_limits

from .original_demo import check_reference, relative_error
from .original_lct import OriginalLCT, load_scene
from .reference import DEFAULT_OUTPUT as DEFAULT_REFERENCE
from .upstream import DEFAULT_ROOT, SOURCE_SHA256, sha256, verify_source


DEFAULT_OUTPUT = Path(__file__).resolve().parent/'results'/'static_mean'


class StaticMeanEstimator:
    """Streaming Poisson rate estimate: total counts / total relative exposure."""

    def __init__(self):
        self._count_sum = None
        self.frame_count = 0
        self.exposure_sum = 0.

    def add(self, counts, exposure=1.):
        exposure = float(exposure)
        counts = np.asarray(counts, dtype=np.float64)
        if not np.isfinite(exposure) or exposure <= 0:
            raise ValueError('Exposure must be finite and positive.')
        if counts.size == 0 or not np.all(np.isfinite(counts)) or np.any(counts < 0):
            raise ValueError('Counts must be nonempty, finite and nonnegative.')
        if self._count_sum is not None and counts.shape != self._count_sum.shape:
            raise ValueError('All frames must have the same shape.')
        if self._count_sum is None:
            self._count_sum = counts.copy()
        else:
            self._count_sum += counts
        self.frame_count += 1
        self.exposure_sum += exposure

    def estimate(self):
        if self._count_sum is None:
            raise ValueError('At least one frame is required.')
        return self._count_sum/self.exposure_sum


def energy_summary(single, averaged, frames):
    """Paired Monte Carlo errors; CI uses a delta-method ratio standard error."""
    single, averaged = np.asarray(single), np.asarray(averaged)
    ratio = float(averaged.mean()/single.mean())
    influence = (averaged-ratio*single)/single.mean()
    standard_error = float(influence.std(ddof=1)/np.sqrt(len(single)))
    prediction = 1/frames
    return {
        'single_mean_squared_l2_error': float(single.mean()),
        'average_mean_squared_l2_error': float(averaged.mean()),
        'average_to_single_noise_energy': ratio,
        'predicted_average_to_single_noise_energy': prediction,
        'ratio_standard_error': standard_error,
        'approximate_95_percent_ratio_interval': [max(0., ratio-1.96*standard_error),
                                                 ratio+1.96*standard_error],
        'measured_snr_gain_db': float(-10*np.log10(ratio)),
        'predicted_snr_gain_db': float(10*np.log10(frames)),
        'consistent_with_prediction': bool(abs(ratio-prediction) <=
                                           4*standard_error+.02*prediction),
    }


def plot_comparison(model, settings, volumes, frames, output):
    labels = ('Reference mean transient', 'Original LCT / one exposure',
              f'LTAI static mean / {frames} exposures')
    displayed = [model.display_volume(volume, settings)[0] for volume in volumes]
    _, lateral, depth = model.display_volume(volumes[0], settings)
    maximum = float(displayed[0].max())
    fig, axes = plt.subplots(3, 3, figsize=(12, 10.4), constrained_layout=True)
    limits = ((lateral, lateral), (lateral, depth), (depth, lateral))
    xylabels = (('x (m)', 'y (m)'), ('x (m)', 'z (m)'), ('z (m)', 'y (m)'))
    for row, (volume, label) in enumerate(zip(displayed, labels)):
        projections = (volume.max(axis=0), volume.max(axis=1), volume.max(axis=2).T)
        for col, (projection, (horizontal, vertical), (xlabel, ylabel), view) in enumerate(zip(
                projections, limits, xylabels, ('Front', 'Top', 'Side'))):
            ax = axes[row, col]
            image = ax.imshow(projection, cmap='gray', origin='upper', vmin=0, vmax=maximum,
                              extent=(horizontal[0], horizontal[-1], vertical[-1], vertical[0]))
            ax.set_title(f'{label}\n{view}', fontsize=10)
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)
            ax.set_box_aspect(1)
    fig.colorbar(image, ax=axes, shrink=.6, label='Author-code units / common intensity scale')
    fig.suptitle(f'{settings.name}: static mean validation\n'
                 'Same fixed LCT filter; positivity is applied only for display', fontsize=13)
    fig.savefig(output, dpi=200)
    fig.savefig(output.with_suffix('.svg'))
    plt.close(fig)


def run_scene(scene, source_dir, reference_dir, output, frames, exposure, trials, seed):
    mean, settings = load_scene(source_dir, scene)
    if np.any(mean < 0) or not np.all(np.isfinite(mean)):
        raise ValueError('The empirical Poisson mean must be finite and nonnegative.')
    model = OriginalLCT(settings.n, settings.m, settings.width_m, settings.range_m,
                        settings.diffuse, settings.snr)
    reference, stages = model(mean, return_stages=True)
    displayed, _, _ = model.display_volume(reference, settings)
    parity = check_reference(reference_dir, scene, settings, reference, displayed, stages, model)
    del stages, displayed
    reference_energy = float(np.sum(reference**2))

    noiseless = StaticMeanEstimator()
    for _ in range(frames):
        noiseless.add(exposure*mean, exposure)
    noiseless_input_error = relative_error(noiseless.estimate(), mean)
    noiseless_output_error = relative_error(model(noiseless.estimate()), reference)
    del noiseless

    rng = np.random.default_rng(np.random.SeedSequence([seed, scene]))
    input_single, input_mean, output_single, output_mean = [], [], [], []
    shown = None
    for trial in range(trials):
        estimator = StaticMeanEstimator()
        pooled_counts = np.zeros_like(mean)
        first_counts = None
        for index in range(frames):
            counts = rng.poisson(exposure*mean)
            estimator.add(counts, exposure)
            pooled_counts += counts
            if index == 0:
                first_counts = counts
        single_input = first_counts/exposure
        mean_input = estimator.estimate()
        single_volume, mean_volume = model(single_input), model(mean_input)
        input_single.append(float(np.sum((single_input-mean)**2)))
        input_mean.append(float(np.sum((mean_input-mean)**2)))
        output_single.append(float(np.sum((single_volume-reference)**2)))
        output_mean.append(float(np.sum((mean_volume-reference)**2)))
        if trial == 0:
            equal_budget_input = pooled_counts/(frames*exposure)
            equal_budget_volume = model(equal_budget_input)
            shown = (single_volume, mean_volume)
            equality = {
                'input_relative_difference': relative_error(mean_input, equal_budget_input),
                'signed_volume_relative_difference': relative_error(mean_volume, equal_budget_volume),
                'same_recorded_photons': True,
                'note': 'Direct original LCT of pooled counts / total exposure. '
                        'For a static mean this is algebraically the same estimator, not another denoiser.'}
            target = output/f'scene_{scene}_{settings.name}'
            target.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(target/'observations.npz', empirical_mean=mean,
                                first_counts=first_counts, pooled_counts=pooled_counts,
                                estimated_mean=mean_input, frame_count=frames,
                                exposure_per_frame=exposure)
            np.savez_compressed(target/'reconstruction.npz', reference_signed=reference,
                                original_single_signed=single_volume, ltai_mean_signed=mean_volume)
            del equal_budget_input, equal_budget_volume
        if (trial+1) % 4 == 0 or trial+1 == trials:
            print(f'  scene {scene}: {trial+1}/{trials} independent sequences', flush=True)
        del estimator, pooled_counts, first_counts, single_input, mean_input, single_volume, mean_volume

    input_stats = energy_summary(input_single, input_mean, frames)
    output_stats = energy_summary(output_single, output_mean, frames)
    output_stats['single_relative_rms_error'] = float(np.sqrt(np.mean(output_single)/reference_energy))
    output_stats['average_relative_rms_error'] = float(np.sqrt(np.mean(output_mean)/reference_energy))
    output_stats['single_conditional_snr_db'] = float(10*np.log10(reference_energy/np.mean(output_single)))
    output_stats['average_conditional_snr_db'] = float(10*np.log10(reference_energy/np.mean(output_mean)))
    input_stats['predicted_single_mean_squared_l2_error'] = float(mean.sum()/exposure)
    input_stats['predicted_average_mean_squared_l2_error'] = float(mean.sum()/(frames*exposure))
    input_stats['single_empirical_to_predicted_energy'] = float(np.mean(input_single)/(mean.sum()/exposure))
    input_stats['average_empirical_to_predicted_energy'] = float(np.mean(input_mean)/(mean.sum()/(frames*exposure)))
    plot_comparison(model, settings, (reference, *shown), frames, target/'comparison.png')
    selected = {'original_single_relative_error': relative_error(shown[0], reference),
                'ltai_mean_relative_error': relative_error(shown[1], reference),
                'selection': 'First random sequence; no selection by reconstruction quality.'}
    checks = {'noiseless_mean': noiseless_input_error < 1e-12,
              'noiseless_reconstruction': noiseless_output_error < 1e-10,
              'equal_photon_budget_equivalence': equality['signed_volume_relative_difference'] < 1e-10,
              'input_noise_scaling': input_stats['consistent_with_prediction'],
              'output_noise_scaling': output_stats['consistent_with_prediction']}
    if parity['available']:
        checks['matlab_anchor'] = parity['passed']
    result = {'settings': settings.metadata(),
              'source_data_sha256': sha256(source_dir/f'data_{settings.name}.mat'),
              'reference_mean_photons': float(mean.sum()),
              'expected_photons_per_frame': float(exposure*mean.sum()),
              'expected_photons_per_sequence': float(frames*exposure*mean.sum()),
              'matlab_reference': parity,
              'noiseless_mean_relative_error': noiseless_input_error,
              'noiseless_reconstruction_relative_error': noiseless_output_error,
              'equal_total_exposure_original': equality,
              'input_noise': input_stats, 'output_noise': output_stats,
              'displayed_sequence': selected, 'checks': checks,
              'monte_carlo_squared_errors': {'input_single': input_single, 'input_mean': input_mean,
                                            'output_single': output_single, 'output_mean': output_mean},
              'passed': bool(all(checks.values()))}
    print(f'  scene {scene}: output noise energy ratio '
          f'{output_stats["average_to_single_noise_energy"]:.4f}, predicted {1/frames:.4f}; '
          f'passed={result["passed"]}', flush=True)
    return result


def plot_noise(report, output):
    fig, axes = plt.subplots(1, len(report['scenes']), figsize=(9, 4), squeeze=False,
                             constrained_layout=True)
    for ax, result in zip(axes[0], report['scenes'].values()):
        stats = result['output_noise']
        ratio = stats['average_to_single_noise_energy']
        ax.bar([0, 1], [1, ratio], color=['#637b91', '#ca7848'], width=.6)
        ax.errorbar(1, ratio, yerr=1.96*stats['ratio_standard_error'], color='black', capsize=5)
        ax.axhline(stats['predicted_average_to_single_noise_energy'], color='black', ls='--',
                   label=f'Prediction: 1/{report["frames"]}')
        ax.set_xticks([0, 1], ['Original / one exposure', 'LTAI static mean'])
        ax.set_ylim(0, 1.08)
        ax.set_ylabel('Output squared error / single-exposure mean')
        ax.set_title(result['settings']['name'])
        ax.legend()
    fig.suptitle('Conditional added-noise reduction; total exposure increases with frame count')
    fig.savefig(output, dpi=200)
    fig.savefig(output.with_suffix('.svg'))
    plt.close(fig)


def run(source_dir=DEFAULT_ROOT/'cnlos2018', output=DEFAULT_OUTPUT, reference_dir=DEFAULT_REFERENCE,
        scenes=(7, 9), frames=7, exposure=1., trials=16, seed=20261030):
    source_dir, output = Path(source_dir), Path(output)
    verify_source(source_dir)
    output.mkdir(parents=True, exist_ok=True)
    report = {'scope': 'LTAI static mean only; no slow-time derivative or physical forward simulation.',
              'source_sha256': SOURCE_SHA256, 'frames': frames, 'exposure_per_frame': exposure,
              'monte_carlo_sequences': trials, 'seed': seed,
              'data_model': 'Y_n | empirical_mean ~ independent Poisson(exposure * empirical_mean). '
                            'The author measurement is held fixed; its original noise and reconstruction bias remain.',
              'error_reference': 'Signed original LCT of the empirical author histogram, not true physical density.',
              'budget': 'N exposures versus one has N times the expected photons. At equal recorded photons '
                        'the new static mean and direct original LCT of normalized pooled counts coincide.',
              'display': 'Author positivity/crop/flip only for figures; all errors use signed native volumes '
                         'with no per-panel gain fitting.', 'scenes': {}}
    for scene in scenes:
        print(f'Static mean scene {scene}: preparing verified original LCT...', flush=True)
        report['scenes'][str(scene)] = run_scene(scene, source_dir, reference_dir, output,
                                                frames, exposure, trials, seed)
    report['matlab_anchor_complete'] = all(item['matlab_reference']['available'] and
                                          item['matlab_reference']['passed']
                                          for item in report['scenes'].values())
    report['passed'] = all(item['passed'] for item in report['scenes'].values())
    plot_noise(report, output/'noise_scaling.png')
    (output/'report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(f'Results: {output.resolve()}', flush=True)
    if not report['passed']:
        raise RuntimeError('Static mean validation failed; inspect report.json.')
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-dir', type=Path, default=DEFAULT_ROOT/'cnlos2018')
    parser.add_argument('--output', type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument('--reference-dir', type=Path, default=DEFAULT_REFERENCE)
    parser.add_argument('--scenes', nargs='+', type=int, default=[7, 9], choices=(7, 9))
    parser.add_argument('--frames', type=int, default=7)
    parser.add_argument('--exposure', type=float, default=1.)
    parser.add_argument('--trials', type=int, default=16)
    parser.add_argument('--seed', type=int, default=20261030)
    parser.add_argument('--threads', type=int, default=2)
    args = parser.parse_args()
    if args.frames < 2 or args.trials < 4 or args.threads < 1:
        parser.error('Use at least 2 frames, 4 trials and 1 thread.')
    if not np.isfinite(args.exposure) or args.exposure <= 0:
        parser.error('--exposure must be finite and positive.')
    with threadpool_limits(args.threads):
        run(args.source_dir, args.output, args.reference_dir, args.scenes, args.frames,
            args.exposure, args.trials, args.seed)


if __name__ == '__main__':
    main()
