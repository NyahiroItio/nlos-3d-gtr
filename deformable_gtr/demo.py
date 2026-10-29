"""带噪、空间留出的整体膨胀验证；不是完整气球或真实数据实验。"""

from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import torch

from .data import Measurements
from .model import AxisScaling, DeformableGTR, canonical_from_arrays
from .training import export_prediction, fit, save_dynamic, write_report


def make_case(case, output, device='cpu', seed=20261029, noise='poisson'):
    rng = np.random.default_rng(seed)
    n = 24
    j = np.arange(n)
    z = 1-2*(j+0.5)/n
    azimuth = np.pi*(3-np.sqrt(5))*j
    directions = np.column_stack([np.sqrt(1-z*z)*np.cos(azimuth), np.sqrt(1-z*z)*np.sin(azimuth), z])
    pivot = np.array([0.025, -0.015, 0.72])
    centers = pivot+directions*np.array([0.12, 0.10, 0.14])
    scales = np.tile([0.012, 0.009, 0.007], (n, 1))
    quats = rng.normal(size=(n, 4))
    quats /= np.linalg.norm(quats, axis=1, keepdims=True)
    albedos = 0.35+0.10*np.cos(j*1.7)
    truth = canonical_from_arrays(centers, scales, quats, albedos, device)
    frame_times = np.linspace(0, 1, 5)
    knots = np.array([0.0, 0.5, 1.0])
    xx, yy = np.meshgrid(np.linspace(-0.5, 0.5, 4), np.linspace(-0.5, 0.5, 4))
    laser = np.column_stack([xx.ravel(), yy.ravel(), np.zeros(xx.size)])
    p = len(laser)
    times = np.repeat(frame_times, p)
    # 固定接收墙点，one-hot 激光扫描；不假设共焦硬件。
    arrays = dict(laser_points=laser, camera_points=np.zeros((len(times), 3)),
                  patterns=np.tile(np.eye(p), (len(frame_times), 1)),
                  transients=np.zeros((len(times), 280)), times=times,
                  dt=np.array(24e-12), t_range=np.array([80, 360]), dose=np.ones(len(times)),
                  background=np.full((len(times), 280), 2.0))
    width = np.where((times == 0) | (times == 1), 0, 0.08)
    arrays.update(exposure_start=times-width/2, exposure_end=times+width/2)
    data = Measurements(arrays, device)
    final_scales = np.array([1.25]*3) if case == 'uniform' else np.array([1.15, 1.25, 1.35])
    motion = AxisScaling(knots, pivot, 'anisotropic').to(device)
    with torch.no_grad():
        motion.log_scales.copy_(torch.tensor(knots[1:, None]*np.log(final_scales), device=device, dtype=torch.float32))
    generator = DeformableGTR(truth, motion, data.dt, data.t_range, quadrature=7, backend='torch')
    with torch.no_grad():
        # 固定合成系统增益，使信号峰值约 2000 counts/bin；不是真实效率标定。
        # 标定时先关闭背景，避免 tiny signal + background 再相减的 float32 消减。
        known_background = data.background.clone()
        data.background.zero_()
        signal = generator(data)
        generator.magnification = 2000/float(signal.max())
        data.background.copy_(known_background)
        mean = generator(data).cpu().numpy()
    arrays['render_magnification'] = np.array(generator.magnification)
    arrays['transients'] = rng.poisson(mean).astype(np.float32) if noise == 'poisson' else mean
    output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output/'measurements.npz', **arrays)
    init_centers = centers+rng.normal(0, 0.0015, centers.shape)
    np.savez_compressed(output/'canonical_init.npz', centers=init_centers, scales=scales,
                        quats=quats, albedos=albedos*0.9)
    np.savez_compressed(output/'truth.npz', centers=centers, scales=scales, quats=quats,
                        albedos=albedos, times=frame_times,
                        motion_scales=np.exp(frame_times[:, None]*np.log(final_scales)))
    initial = canonical_from_arrays(init_centers, scales, quats, albedos*0.9, device)
    data = Measurements(arrays, device)
    return initial, data, knots, pivot, generator.magnification, final_scales


def run_demo(output, *, iterations=200, device='cpu', seed=20261029, noise='poisson'):
    output = Path(output)
    report = {'seed': seed, 'noise': noise, 'device': str(device), 'cases': {},
              'scope': 'Warm-start synthetic Gaussian shell; fixed canonical scales/orientations; no real balloon data.'}
    curves = {}
    for case_index, case in enumerate(['uniform', 'anisotropic']):
        directory = output/case
        initial, data, knots, pivot, magnification, truth_final = make_case(case, directory, device, seed+case_index, noise)
        case_report = {'true_final_scales': truth_final.tolist(), 'magnification': magnification, 'models': {}}
        times = np.unique(data.times)
        truth_scales = np.exp(times[:, None]*np.log(truth_final))
        curves[case] = {'times': times, 'truth': truth_scales}
        for mode in ['static', 'isotropic', 'anisotropic']:
            torch.manual_seed(seed)
            model = DeformableGTR(copy.deepcopy(initial), AxisScaling(knots, pivot, mode).to(device),
                                  data.dt, data.t_range, magnification=magnification, quadrature=3, backend='torch')
            print(f'\nCase={case}, fit={mode}', flush=True)
            result = fit(model, data, iterations=iterations, seed=seed, log_every=max(1, iterations//4))
            with torch.no_grad():
                estimated, _ = model.motion.values(times)
                estimated = estimated.cpu().numpy()
            result.update(final_scales=estimated[-1].tolist(),
                          scale_rmse=float(np.sqrt(np.mean((estimated-truth_scales)**2))),
                          max_scale_error=float(np.max(np.abs(estimated-truth_scales))))
            case_report['models'][mode] = result
            curves[case][mode] = estimated
            save_dynamic(model, directory/f'{mode}.pth')
            export_prediction(model, data, directory/f'{mode}_prediction.npz')
        models = case_report['models']
        case_report['dynamic_beats_static'] = bool(
            models['anisotropic']['final_validation_nmse'] < 0.5*models['static']['final_validation_nmse'])
        case_report['scale_recovered_within_4_percent'] = bool(models['anisotropic']['max_scale_error'] < 0.04)
        report['cases'][case] = case_report
    report['passed'] = all(row['dynamic_beats_static'] and row['scale_recovered_within_4_percent']
                           for row in report['cases'].values())
    write_report(output/'report.json', report)
    plot_demo(report, curves, output/'comparison.png')
    if not report['passed']:
        raise RuntimeError(f'Demo acceptance criteria failed; inspect {output / "report.json"}')
    return report


def plot_demo(report, curves, path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    for column, case in enumerate(['uniform', 'anisotropic']):
        row = curves[case]
        ax = axes[0, column]
        for axis in range(3):
            ax.plot(row['times'], row['truth'][:, axis], '--', color=f'C{axis}', alpha=0.65)
            ax.plot(row['times'], row['anisotropic'][:, axis], 'o-', color=f'C{axis}', label=f'Axis {"xyz"[axis]}')
        ax.set(title=f'{case}: anisotropic fit (dashed = truth)', xlabel='Acquisition time (s)', ylabel='Stretch ratio')
        ax.legend()
        ax = axes[1, column]
        models = report['cases'][case]['models']
        ax.bar(list(models), [x['final_validation_nmse'] for x in models.values()], color=['C7', 'C1', 'C2'])
        ax.set(yscale='log', ylabel='Held-out measurement NMSE', title=f'{case}: same data and spatial split')
    fig.suptitle('Shared canonical 3D-GTR + time-dependent axis scaling')
    fig.savefig(path, dpi=160)
    plt.close(fig)
