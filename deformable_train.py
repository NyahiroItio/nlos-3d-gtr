"""3D-GTR 整体三轴伸缩扩展：demo / fit / render。见 deformable_gtr/README.md。"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from deformable_gtr import AxisScaling, DeformableGTR, Measurements
from deformable_gtr.demo import run_demo
from deformable_gtr.training import export_prediction, fit, load_canonical, load_dynamic, save_dynamic, write_report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--threads', type=int, default=2, help='Torch CPU threads; keep small for the demo.')
    sub = parser.add_subparsers(dest='command', required=True)
    demo = sub.add_parser('demo', help='Generate and invert uniform and anisotropic synthetic sequences.')
    demo.add_argument('--output', type=Path, default=Path('deformable_gtr/results/demo'))
    demo.add_argument('--iterations', type=int, default=200)
    demo.add_argument('--device', default='cpu')
    demo.add_argument('--seed', type=int, default=20261029)
    demo.add_argument('--noise', choices=['none', 'poisson'], default='poisson')
    train = sub.add_parser('fit', help='Fit a timed NPZ sequence from a canonical NPZ or original 3D-GTR checkpoint.')
    train.add_argument('--data', type=Path, required=True)
    train.add_argument('--init', type=Path, required=True)
    train.add_argument('--output', type=Path, required=True)
    train.add_argument('--mode', choices=['static', 'isotropic', 'anisotropic'], default='anisotropic')
    train.add_argument('--pivot', type=float, nargs=3, required=True, metavar=('X', 'Y', 'Z'))
    train.add_argument('--knots', type=float, nargs='+', help='Must include reference time and cover every exposure.')
    train.add_argument('--translation', action='store_true')
    train.add_argument('--fit-canonical', choices=['none', 'centers_albedo', 'all'], default='centers_albedo')
    train.add_argument('--iterations', type=int, default=200)
    train.add_argument('--device', default='cpu')
    train.add_argument('--backend', choices=['auto', 'torch', 'triton'], default='auto')
    train.add_argument('--loss', choices=['mse', 'poisson'], default='mse')
    train.add_argument('--magnification', type=float, help='Override data render_magnification (default 1 if absent).')
    train.add_argument('--quadrature', type=int, default=3)
    train.add_argument('--validation', type=float, default=0.2)
    train.add_argument('--batch-size', type=int, default=0)
    train.add_argument('--seed', type=int, default=20261029)
    render = sub.add_parser('render', help='Reload a dynamic checkpoint and predict a timed sequence.')
    render.add_argument('--checkpoint', type=Path, required=True)
    render.add_argument('--data', type=Path, required=True)
    render.add_argument('--output', type=Path, required=True)
    render.add_argument('--device', default='cpu')
    render.add_argument('--backend', choices=['auto', 'torch', 'triton'], default='auto')
    args = parser.parse_args()
    if args.threads < 1:
        parser.error('--threads must be positive')
    torch.set_num_threads(args.threads)
    if args.command == 'demo':
        report = run_demo(args.output, iterations=args.iterations, device=args.device, seed=args.seed, noise=args.noise)
        print(f'Demo passed. Report: {args.output / "report.json"}')
    elif args.command == 'fit':
        canonical = load_canonical(args.init, args.device)
        dtype = next(canonical.parameters()).dtype
        data = Measurements.load(args.data, args.device, dtype)
        knots = args.knots if args.knots else np.unique(np.r_[data.exposure_start.min(), data.times, data.exposure_end.max()])
        motion = AxisScaling(knots, args.pivot, args.mode, args.translation).to(device=args.device, dtype=dtype)
        model = DeformableGTR(canonical, motion, data.dt, data.t_range,
                              data.magnification if args.magnification is None else args.magnification,
                              args.quadrature, args.backend)
        args.output.mkdir(parents=True, exist_ok=True)
        report = fit(model, data, iterations=args.iterations, fit_canonical=args.fit_canonical,
                     validation=args.validation, seed=args.seed, loss=args.loss, batch_size=args.batch_size)
        write_report(args.output/'report.json', report)
        save_dynamic(model, args.output/'model.pth')
        export_prediction(model, data, args.output/'prediction.npz')
    else:
        model = load_dynamic(args.checkpoint, args.device, args.backend)
        data = Measurements.load(args.data, args.device, next(model.parameters()).dtype)
        if data.dt != model.dt or data.t_range != model.t_range:
            raise ValueError('Prediction data must use checkpoint dt/t_range.')
        args.output.parent.mkdir(parents=True, exist_ok=True)
        export_prediction(model, data, args.output)


if __name__ == '__main__':
    main()
