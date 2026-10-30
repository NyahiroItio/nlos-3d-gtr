"""Run the author-aligned static LCT on the original 2018 datasets."""

import argparse
import json
from pathlib import Path
from time import perf_counter

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.io import loadmat
from threadpoolctl import threadpool_limits

from .original_lct import OriginalLCT, load_scene
from .reference import DEFAULT_OUTPUT as DEFAULT_REFERENCE
from .upstream import DEFAULT_ROOT, SOURCE_SHA256, verify_source, sha256


DEFAULT_OUTPUT = Path(__file__).resolve().parent/'results'/'original_alignment'


def relative_error(actual, expected):
    return float(np.linalg.norm(actual-expected)/max(np.linalg.norm(expected), 1e-300))


def check_reference(reference_dir, scene, settings, signed, displayed, stages, model):
    filename = Path(reference_dir)/f'scene_{scene}.mat'
    if not filename.exists():
        return {'available': False, 'passed': False, 'reason': 'MATLAB reference has not been generated.'}
    reference = loadmat(filename, squeeze_me=True)
    if str(reference['source_sha256']) != SOURCE_SHA256:
        raise ValueError('Reference was not produced from the pinned original source.')
    comparisons = {name: relative_error(stages[name], reference[name])
                   for name in ('weighted_data', 'resampled_data', 'filtered_data', 'signed_volume')}
    comparisons['clipped_volume'] = relative_error(np.maximum(signed, 0), reference['clipped_volume'])
    comparisons['display_volume'] = relative_error(displayed, reference['display_volume'])
    comparisons['mtx'] = relative_error(model.mtx.toarray(), reference['mtx'])
    comparisons['mtxi'] = relative_error(model.mtxi.toarray(), reference['mtxi'])
    checks = {'signed_shape': list(signed.shape) == list(reference['signed_volume'].shape),
              'snr': np.isclose(settings.snr, reference['snr']),
              'diffuse': settings.diffuse == bool(reference['isdiffuse']),
              'native_bin_width': np.isclose(settings.bin_resolution_s, reference['bin_resolution'], rtol=1e-12, atol=0),
              'all_numeric_stages': max(comparisons.values()) < 1e-9}
    return {'available': True, 'passed': bool(all(checks.values())),
            'relative_errors': comparisons, 'checks': {k: bool(v) for k, v in checks.items()}}


def plot_volume(volume, lateral, depth, title, filename):
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.8), constrained_layout=True)
    images = (volume.max(axis=0), volume.max(axis=1), volume.max(axis=2).T)
    limits = ((lateral, lateral), (lateral, depth), (depth, lateral))
    labels = (('x (m)', 'y (m)'), ('x (m)', 'z (m)'), ('z (m)', 'y (m)'))
    maximum = float(volume.max())
    for ax, image, (horizontal, vertical), (xlabel, ylabel), view in zip(
            axes, images, limits, labels, ('Front', 'Top', 'Side')):
        ax.imshow(image, cmap='gray', origin='upper', vmin=0, vmax=maximum,
                  extent=(horizontal[0], horizontal[-1], vertical[-1], vertical[0]))
        ax.set_title(view)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_box_aspect(1)
    fig.suptitle(title)
    fig.savefig(filename, dpi=240)
    fig.savefig(filename.with_suffix('.svg'))
    plt.close(fig)


def plot_input(data, settings, output):
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.8), constrained_layout=True)
    width = settings.width_m
    integrated = data.sum(axis=0)
    axes[0].imshow(integrated, cmap='gray', extent=(-width, width, width, -width))
    axes[0].set_title('Prepared measured counts / sum over fast time')
    axes[0].set_xlabel('Wall x (m)')
    axes[0].set_ylabel('Wall y (m)')
    axes[1].imshow(np.log1p(data[:, settings.n//2]), cmap='magma', aspect='auto',
                   extent=(-width, width, settings.m*settings.bin_resolution_s*1e9, 0))
    axes[1].set_title('Measured streak / log(1 + counts)')
    axes[1].set_xlabel('Wall x (m)')
    axes[1].set_ylabel('Fast time s (ns)')
    fig.savefig(output, dpi=220)
    plt.close(fig)


def run(source_dir=DEFAULT_ROOT/'cnlos2018', output=DEFAULT_OUTPUT,
        scenes=(7, 9), reference_dir=DEFAULT_REFERENCE):
    source_dir, output = Path(source_dir), Path(output)
    verify_source(source_dir)
    output.mkdir(parents=True, exist_ok=True)
    report = {'source_sha256': SOURCE_SHA256,
              'scope': 'Static author-code alignment on original data. No new geometry simulation or temporal deformation.',
              'units': 'Author engineering units, not calibrated physical density.', 'scenes': {}}
    for scene in scenes:
        print(f'Original dataset scene {scene}: loading...', flush=True)
        data, settings = load_scene(source_dir, scene)
        start = perf_counter()
        model = OriginalLCT(settings.n, settings.m, settings.width_m, settings.range_m,
                            settings.diffuse, settings.snr)
        preparation = perf_counter()-start
        start = perf_counter()
        signed, stages = model(data, return_stages=True)
        solve_time = perf_counter()-start
        displayed, lateral, depth = model.display_volume(signed, settings)
        target = output/f'scene_{scene}_{settings.name}'
        target.mkdir(exist_ok=True)
        plot_volume(displayed, lateral, depth,
                    f'Original LCT / {settings.name} / {settings.n} x {settings.n} x {settings.m}',
                    target/'reconstruction.png')
        plot_input(data, settings, target/'measured_transient.png')
        parity = check_reference(reference_dir, scene, settings, signed, displayed, stages, model)
        # A second full-size FFT checks odd linearity without applying the
        # author positivity postprocess to a signed input.
        sign_error = relative_error(model(-data), -signed)
        np.savez_compressed(target/'reconstruction.npz', signed_volume=signed,
                            display_volume=displayed, lateral_m=lateral, depth_m=depth)
        result = {'settings': settings.metadata(), 'signed_shape': list(signed.shape),
                  'data_sha256': sha256(source_dir/f'data_{settings.name}.mat'),
                  'display_shape': list(displayed.shape), 'preparation_seconds': preparation,
                  'reconstruction_seconds': solve_time, 'matlab_reference': parity,
                  'signed_linearity_relative_error': sign_error}
        report['scenes'][str(scene)] = result
        print(f'  native {signed.shape}, MATLAB reference: {parity.get("passed")}, '
              f'signed linearity: {sign_error:.3g}', flush=True)
        del data, model, signed, stages, displayed
    available = all(item['matlab_reference']['available'] for item in report['scenes'].values())
    report['matlab_reference_complete'] = available
    report['passed'] = available and all(item['matlab_reference']['passed'] and
                                         item['signed_linearity_relative_error'] < 1e-10
                                         for item in report['scenes'].values())
    (output/'report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(f'Results: {output.resolve()}', flush=True)
    if available and not report['passed']:
        raise RuntimeError('Original MATLAB/Python numerical alignment failed.')
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-dir', type=Path, default=DEFAULT_ROOT/'cnlos2018')
    parser.add_argument('--output', type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument('--scenes', nargs='+', type=int, default=[7, 9], choices=range(1, 11))
    parser.add_argument('--reference-dir', type=Path, default=DEFAULT_REFERENCE)
    parser.add_argument('--threads', type=int, default=2)
    args = parser.parse_args()
    with threadpool_limits(args.threads):
        run(args.source_dir, args.output, args.scenes, args.reference_dir)


if __name__ == '__main__':
    main()
