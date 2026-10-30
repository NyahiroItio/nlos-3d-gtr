"""Compare two saved inverse backends on identical physical Poisson data."""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy import sparse


def compare(lct_dir, spectral_dir, output):
    lct_dir, spectral_dir, output = map(Path, (lct_dir, spectral_dir, output))
    reports = {name: json.loads((folder/'report.json').read_text(encoding='utf-8'))
               for name, folder in (('lct', lct_dir), ('spectral', spectral_dir))}
    operators = [sparse.load_npz(folder/'operator.npz') for folder in (lct_dir, spectral_dir)]
    if operators[0].shape != operators[1].shape or (operators[0]-operators[1]).nnz:
        raise ValueError('Comparison requires identical forward operators and exposure gains.')
    output.mkdir(parents=True, exist_ok=True)
    with np.load(lct_dir/'geometry.npz', allow_pickle=False) as geometry:
        x, y, z = (geometry[axis].copy() for axis in ('x', 'y', 'z'))
    shape = (len(x), len(y), len(z))
    fig, axes = plt.subplots(4, 3, figsize=(12, 12), constrained_layout=True)
    summary = {'identical_forward_operator': True, 'identical_raw_counts': True,
               'note': 'Same data and absolute units. Spectral includes the discrete projector and IRF; '
                       'LCT uses a continuous ideal-model FFT inverse. No truth-fitted rescaling.',
               'cases': {}}
    for case_index, case in enumerate(('linear_density', 'translation')):
        data = {}
        for name, folder in (('lct', lct_dir), ('spectral', spectral_dir)):
            with np.load(folder/case/'experiment.npz', allow_pickle=False) as archive:
                data[name] = {key: archive[key].copy() for key in archive.files}
        for key in ('raw_counts', 'background_estimate', 'rho_frames'):
            if not np.array_equal(data['lct'][key], data['spectral'][key]):
                raise ValueError(f'Comparison data mismatch: {case}/{key}. Use identical seeds and settings.')
        summary['cases'][case] = {}
        for component, field in enumerate(('reference', 'derivative')):
            summary['cases'][case][field] = {
                name: reports[name]['cases'][case]['metrics']['window'][field]
                for name in ('spectral', 'lct')}
            truth = data['lct']['rho_reference' if component == 0 else 'rho_derivative']
            fields = [truth, data['spectral']['window_reconstruction'][component],
                      data['lct']['window_reconstruction'][component]]
            images = [a.reshape(shape).sum(axis=1).T for a in fields]
            maximum = max(np.max(np.abs(image)) for image in images)
            minimum = -maximum if component else min(np.min(image) for image in images)
            for column, image in enumerate(images):
                ax = axes[2*case_index+component, column]
                im = ax.imshow(image, origin='lower', extent=(x[0], x[-1], z[0], z[-1]),
                               aspect='auto', cmap='RdBu_r' if component else 'viridis',
                               vmin=minimum, vmax=maximum)
                ax.set_title(f'{case}: {field}\n'+['Truth', 'Spectral / window', 'LCT / window'][column], fontsize=9)
                ax.set_xlabel('x (m)')
                ax.set_ylabel('z (m)')
                fig.colorbar(im, ax=ax, shrink=.75)
    fig.savefig(output/'backends.png', dpi=160)
    plt.close(fig)
    (output/'comparison.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print(f'Identical-data comparison: {output.resolve()}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--lct', default='transient_transform/results/lct_demo')
    parser.add_argument('--spectral', default='transient_transform/results/spectral_demo')
    parser.add_argument('--output', default='transient_transform/results/backend_comparison')
    args = parser.parse_args()
    compare(args.lct, args.spectral, args.output)


if __name__ == '__main__':
    main()
