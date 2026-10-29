"""比较已知球心、错误固定球心和未知共享球心；仅拟合瞬态测量。"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import torch

from .opaque_ellipsoid import OpaqueEllipsoid
from .training import normalized_mse, write_report


def fit_ellipsoid(model, data, iterations=300, log_every=100):
    groups = [
        {'params': [model.log_radii], 'lr': 0.008},
        {'params': [model.motion.log_scales], 'lr': 0.015},
        {'params': [model.log_gain], 'lr': 0.015},
    ]
    if model.motion.learn_pivot:
        groups.append({'params': [model.motion.pivot], 'lr': 0.0015})
    optimizer = torch.optim.Adam(groups)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=[int(iterations*0.55), int(iterations*0.8)], gamma=0.35)
    train, valid = data['train'], data['valid']
    history = []
    started = time.perf_counter()
    for iteration in range(iterations):
        optimizer.zero_grad(set_to_none=True)
        predicted = model(data['laser'][train], data['camera'][train], data['times'][train], data['edges'])
        objective = normalized_mse(predicted, data['observed'][train])
        if not torch.isfinite(objective):
            raise FloatingPointError('Nonfinite opaque ellipsoid objective.')
        objective.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 100, error_if_nonfinite=True)
        optimizer.step()
        scheduler.step()
        if iteration == 0 or (iteration+1) % log_every == 0 or iteration+1 == iterations:
            with torch.no_grad():
                val_pred = model(data['laser'][valid], data['camera'][valid], data['times'][valid], data['edges'])
                val = float(normalized_mse(val_pred, data['observed'][valid]))
            history.append({'iteration': iteration+1, 'train_nmse': float(objective.detach()), 'validation_nmse': val})
            print(f'  {iteration+1}/{iterations}: train={float(objective.detach()):.6g}, validation={val:.6g}, '
                  f'center={model.motion.pivot.detach().cpu().numpy()}', flush=True)
    with torch.no_grad():
        predicted = model(data['laser'], data['camera'], data['times'], data['edges'])
        scales, _ = model.motion.values(torch.linspace(0, 1, 5, device=data['laser'].device))
    return {
        'center_m': model.motion.pivot.detach().cpu().tolist(),
        'reference_radii_m': model.log_radii.detach().exp().cpu().tolist(),
        'final_scales': scales[-1].cpu().tolist(),
        'log_gain': float(model.log_gain.detach()),
        'validation_nmse': float(normalized_mse(predicted[valid], data['observed'][valid])),
        'history': history, 'elapsed_seconds': time.perf_counter()-started,
    }, predicted.cpu().numpy()


def run_pivot_demo(output, iterations=200, seed=20261029, device='cpu', noise='poisson'):
    if iterations < 1:
        raise ValueError('iterations must be positive.')
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    dtype = torch.float32
    center = np.array([0.025, -0.015, 0.72])
    wrong_center = center+np.array([0.025, -0.020, 0.030])
    radii = np.array([0.12, 0.10, 0.14])
    initial_radii = np.array([0.11, 0.11, 0.12])
    grid_x, grid_y = np.meshgrid(np.linspace(-0.5, 0.5, 3), np.linspace(-0.5, 0.5, 3))
    wall = np.column_stack([grid_x.ravel(), grid_y.ravel(), np.zeros(grid_x.size)])
    times_np = np.repeat(np.linspace(0, 1, 5), len(wall))
    laser = torch.tensor(np.tile(wall, (5, 1)), device=device, dtype=dtype)
    camera = torch.zeros_like(laser)
    times = torch.tensor(times_np, device=device, dtype=dtype)
    edges = torch.arange(80, 257, device=device, dtype=dtype)*32
    wall_ids = np.tile(np.arange(len(wall)), 5)
    valid = torch.tensor(np.flatnonzero(np.isin(wall_ids, [0, 8])), device=device)
    train = torch.tensor(np.flatnonzero(~np.isin(wall_ids, [0, 8])), device=device)
    report = {
        'scope': 'Analytic opaque convex ellipsoid, uniform unknown surface gain; geometry family known, center and reference radii estimated.',
        'seed': seed, 'noise': noise, 'iterations': iterations, 'cases': {},
        'true_center_m': center.tolist(), 'wrong_initial_center_m': wrong_center.tolist(),
        'true_reference_radii_m': radii.tolist(), 'initial_reference_radii_m': initial_radii.tolist(),
        'fixed_receiver': [0, 0, 0], 'truth_quadrature': [32, 64], 'fit_quadrature': [16, 32],
        'time_zero_known': True, 'irf_sigma_ps': 60.0,
        'surface_normals_and_area_updated': True, 'self_occlusion': 'exact one-sided visibility for convex ellipsoid',
        'external_occlusion_or_transmission': False, 'free_gaussian_shape_reconstruction': False,
        'registration_loss': 'measurement-domain NMSE; no point correspondences or ICP',
    }
    for name, mode, final_scales in [
        ('uniform', 'isotropic', np.array([1.25]*3)),
        ('anisotropic', 'anisotropic', np.array([1.15, 1.25, 1.35])),
    ]:
        print(f'\nOpaque case={name}', flush=True)
        directory = output/name
        directory.mkdir(exist_ok=True)
        truth = OpaqueEllipsoid(center, radii, mode, False, polar=32, azimuth=64, background=0.2).to(device)
        with torch.no_grad():
            beta = np.log(final_scales[:1] if mode == 'isotropic' else final_scales)
            truth.motion.log_scales.copy_(torch.tensor(beta[None], device=device, dtype=dtype))
            truth.background = 0.0
            base = truth(laser, camera, times, edges)
            truth.magnification = 6000/float(base.sum(dim=1).mean())
            truth.background = 0.2
            mean = truth(laser, camera, times, edges)
            # 明确检查背向相机的表面权重为零，不使用固定 z 半球掩码。
            weight, _ = truth.transport(laser, camera, times)
            _, normals, _ = truth.surface(times)
            backward = normals[..., 2] > 0
            back_fraction = float(weight[backward].sum()/weight.sum())
            assert back_fraction == 0.0
            visible_fraction = float((weight > 0).float().mean())
        observed_np = rng.poisson(mean.cpu().numpy()).astype(np.float32) if noise == 'poisson' else mean.cpu().numpy()
        observed = torch.tensor(observed_np, device=device)
        data = {'laser': laser, 'camera': camera, 'times': times, 'edges': edges,
                'observed': observed, 'train': train, 'valid': valid}
        np.savez_compressed(directory/'measurements.npz', laser_points=laser.cpu().numpy(),
                            camera_points=camera.cpu().numpy(), times=times_np, edges_ps=edges.cpu().numpy(),
                            transients=observed_np, train_rows=train.cpu().numpy(), validation_rows=valid.cpu().numpy())
        rows = {}
        for label, initial_center, learn in [
            ('known_center', center, False), ('wrong_fixed_center', wrong_center, False),
            ('estimated_center', wrong_center, True),
        ]:
            print(f'Fit={label}', flush=True)
            model = OpaqueEllipsoid(initial_center, initial_radii, mode, learn,
                                    magnification=truth.magnification, background=0.2).to(device)
            result, prediction = fit_ellipsoid(model, data, iterations, max(1, iterations//3))
            result['center_error_mm'] = float(np.linalg.norm(np.array(result['center_m'])-center)*1000)
            result['final_scale_max_error'] = float(np.max(np.abs(np.array(result['final_scales'])-final_scales)))
            result['reference_radius_max_error_mm'] = float(np.max(np.abs(np.array(result['reference_radii_m'])-radii))*1000)
            rows[label] = result
            np.savez_compressed(directory/f'{label}_prediction.npz', predicted_transients=prediction,
                                center_m=result['center_m'], radii_m=result['reference_radii_m'], final_scales=result['final_scales'])
            torch.save({'state_dict': model.state_dict(), 'mode': mode, 'learn_center': learn,
                        'magnification': model.magnification, 'sigma_ps': model.sigma_ps,
                        'background': model.background, 'quadrature': [16, 32]}, directory/f'{label}.pth')
        report['cases'][name] = {
            'true_final_scales': final_scales.tolist(), 'magnification': truth.magnification,
            'back_facing_signal_fraction': back_fraction, 'mean_visible_quadrature_fraction': visible_fraction,
            'models': rows,
            'estimated_center_improves_wrong_fixed': rows['estimated_center']['validation_nmse'] < rows['wrong_fixed_center']['validation_nmse'],
        }
        write_report(output/'report.json', report)
    report['quadrature_check'] = check_quadrature(report, output, device)
    write_report(output/'report.json', report)
    plot_report(report, output/'comparison.png')
    return report


def check_quadrature(report, output, device='cpu'):
    """在真实参数处单独评估积分网格误差，和噪声/优化误差分开。"""
    results = {}
    for case, values in report['cases'].items():
        with np.load(Path(output)/case/'measurements.npz', allow_pickle=False) as arrays:
            laser = torch.tensor(arrays['laser_points'], device=device)
            camera = torch.tensor(arrays['camera_points'], device=device)
            times = torch.tensor(arrays['times'], device=device, dtype=laser.dtype)
            edges = torch.tensor(arrays['edges_ps'], device=device)
        mode = 'isotropic' if case == 'uniform' else 'anisotropic'
        predictions = []
        for polar, azimuth in [(16, 32), (32, 64)]:
            model = OpaqueEllipsoid(report['true_center_m'], report['true_reference_radii_m'],
                                    mode, False, polar=polar, azimuth=azimuth,
                                    magnification=values['magnification']).to(device=device, dtype=laser.dtype)
            beta = np.log(values['true_final_scales'][:1] if mode == 'isotropic' else values['true_final_scales'])
            with torch.no_grad():
                model.motion.log_scales.copy_(torch.tensor(beta[None], device=device, dtype=laser.dtype))
                predictions.append(model(laser, camera, times, edges))
        results[case] = {'true_parameter_coarse_vs_fine_nmse': float(normalized_mse(*predictions))}
    return results


def plot_report(report, path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    for axis, field, label in [(axes[0], 'center_error_mm', 'Center error (mm)'),
                                (axes[1], 'validation_nmse', 'Held-out transient NMSE')]:
        names = ['known_center', 'wrong_fixed_center', 'estimated_center']
        for index, (case, values) in enumerate(report['cases'].items()):
            axis.bar(np.arange(3)+(index-0.5)*0.35,
                     [values['models'][key][field] for key in names], width=0.35, label=case)
        axis.set(xticks=np.arange(3), xticklabels=['Known center', 'Wrong, fixed', 'Estimated'], ylabel=label)
        if field == 'validation_nmse':
            axis.set_yscale('log')
        axis.legend()
    fig.suptitle('Opaque ellipsoid: fixed unknown center estimated from transients')
    fig.savefig(path, dpi=160)
    plt.close(fig)
