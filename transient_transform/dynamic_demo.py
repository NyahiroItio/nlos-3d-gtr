"""Dynamic quadratic LTAI validation: affine blend, translation and expansion."""

import argparse
from dataclasses import replace
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
import numpy as np
from scipy.ndimage import gaussian_filter
from threadpoolctl import threadpool_limits

from .motion_simulation import expansion_sequence, translation_sequence
from .original_demo import relative_error
from .original_lct import OriginalLCT, SceneSettings, load_scene
from .temporal_fit import PolynomialTransientEstimator, fit_weights, centered_difference_weights
from .upstream import DEFAULT_ROOT, SOURCE_SHA256, verify_source, sha256


DEFAULT_OUTPUT = Path(__file__).resolve().parent/'results'/'dynamic'


def paired_ratio(baseline, fitted, prediction=None):
    baseline, fitted = np.asarray(baseline), np.asarray(fitted)
    ratio = float(fitted.mean()/baseline.mean())
    stderr = float(np.std((fitted-ratio*baseline)/baseline.mean(), ddof=1)/np.sqrt(len(fitted)))
    result = {'noise_energy_ratio': ratio, 'ratio_standard_error': stderr,
              'approximate_95_percent_interval': [max(0., ratio-1.96*stderr), ratio+1.96*stderr],
              'noise_snr_gain_db': float(-10*np.log10(ratio))}
    if prediction is not None:
        result.update(predicted_noise_energy_ratio=float(prediction),
                      consistent_with_prediction=bool(abs(ratio-prediction) <= 4*stderr+.02*prediction))
    return result


def prepare_case(name, times, source_dir, target):
    if name == 'affine':
        first, settings = load_scene(source_dir, 7)
        second = first[:, :, ::-1].copy()
        mean = (first+second)/2
        derivative = (second-first)/(times[-1]-times[0])
        means = mean+times[:, None, None, None]*derivative
        settings = replace(settings, name='affine_density_blend')
        metadata = {'kind': 'Exactly affine empirical transient / density blend, not material motion.',
                    'model': 'tau(t) = tau_0 + t*d; endpoints are author S/U and its lateral reflection.',
                    'source_data_sha256': sha256(source_dir/'data_s_u.mat'),
                    'error_reference': 'Noise-free LCT of the empirical transient and its exact slope. '
                                       'Original measurement noise and inverse bias remain.'}
        return means, derivative, settings, metadata, None
    if name not in ('translation', 'expansion', 'depth_expansion'):
        raise ValueError(f'Unknown dynamic case: {name}')
    settings = SceneSettings(0, f'physical_s_{name}', .35, 64, 512, 16e-12,
                             2.4576, True, .08, 150, 25)
    if name == 'translation':
        means, derivative, points, weights, metadata = translation_sequence(
            times, settings, target/'forward_cache.npz')
        point_velocities = np.broadcast_to(metadata['velocity_m_per_s'], points.shape)
        kind = 'Physical constant-velocity lateral translation; generally non-affine density.'
    else:
        mode = 'depth' if name == 'depth_expansion' else 'planar'
        means, derivative, points, weights, metadata = expansion_sequence(
            times, settings, target/'forward_cache.npz', mode=mode)
        radial = points-np.asarray(metadata['expansion_center_m'])
        if mode == 'depth':
            radial[:, :2] = 0.
        else:
            radial[:, 2] = 0.
        point_velocities = metadata['expansion_rate_per_s']*radial
        kind = ('Physical axial expansion of a three-layer S slab.' if mode == 'depth' else
                'Physical planar radial expansion with fixed positive point weights.')
    metadata.update(kind=kind,
                    error_reference='Noise-free instantaneous LCT of independently simulated data, '
                                    'including analytic transient derivative. Not calibrated physical density.')
    np.savez_compressed(target/'source_geometry.npz', points=points, weights=weights,
                        point_velocities_m_per_s=point_velocities, slow_times_s=times)
    edges = np.linspace(-settings.width_m, settings.width_m, 129)
    source_fronts = []
    for time in times:
        moved = points+time*point_velocities
        raster, _, _ = np.histogram2d(moved[:, 1], moved[:, 0], bins=(edges, edges), weights=weights)
        source_fronts.append(gaussian_filter(raster, .7))
    return means, derivative, settings, metadata, np.stack(source_fronts)


def front(model, settings, volume, signed=False):
    displayed = model.display_volume(volume, settings, clip=not signed)[0]
    return displayed.sum(axis=0) if signed else displayed.max(axis=0)


def save_visuals(fronts, source_fronts, times, width, derivative_maps, title, target,
                 derivative_slices, slice_depth):
    names = ['Noise-free LCT reference', 'Original / framewise LCT',
             'LTAI / linear fit', 'LTAI / quadratic fit']
    panels = list(fronts)
    if source_fronts is not None:
        panels.insert(0, source_fronts)
        names.insert(0, 'True source projection')
    maximum = float(np.max(fronts[0]))
    limits = (-width, width, width, -width)
    fig, axes = plt.subplots(1, len(panels), figsize=(3.25*len(panels), 3.7),
                             constrained_layout=True)
    images = []
    for index, (ax, frames, name) in enumerate(zip(axes, panels, names)):
        vmax = float(frames.max()) if source_fronts is not None and index == 0 else maximum
        images.append(ax.imshow(frames[0], cmap='gray', vmin=0, vmax=vmax, extent=limits))
        ax.set_title(name, fontsize=10)
        ax.set_xlabel('x (m)')
        ax.set_ylabel('y (m)')
        ax.set_box_aspect(1)
    heading = fig.suptitle('')

    def update(index):
        for image, frames in zip(images, panels):
            image.set_data(frames[index])
        heading.set_text(f'{title} / t={times[index]:+.2f} s')
        return images+[heading]

    animation = FuncAnimation(fig, update, frames=len(times), interval=500, blit=False)
    animation.save(target/'dynamic.gif', writer=PillowWriter(fps=2), dpi=120)
    plt.close(fig)

    chosen = [0, len(times)//2, len(times)-1]
    fig, axes = plt.subplots(len(panels), 3, figsize=(10, 3.15*len(panels)),
                             constrained_layout=True, squeeze=False)
    for row, (frames, name) in enumerate(zip(panels, names)):
        vmax = float(frames.max()) if source_fronts is not None and row == 0 else maximum
        for col, index in enumerate(chosen):
            ax = axes[row, col]
            ax.imshow(frames[index], cmap='gray', vmin=0, vmax=vmax, extent=limits)
            ax.set_title(f'{name}\nt={times[index]:+.2f} s', fontsize=10)
            ax.set_xlabel('x (m)')
            ax.set_ylabel('y (m)')
            ax.set_box_aspect(1)
    fig.suptitle(title)
    fig.savefig(target/'frames.png', dpi=180)
    fig.savefig(target/'frames.svg')
    plt.close(fig)

    save_derivative_maps(derivative_maps, width,
                         f'{title}: signed depth sum', target/'derivative_depth_sum.png',
                         'Signed depth sum / author units per second')
    save_derivative_maps(derivative_slices, width,
                         f'{title}: signed derivative at z={slice_depth:.3f} m',
                         target/'derivative.png', 'Signed derivative / author units per second')


def save_derivative_maps(maps, width, title, output, units):
    labels = ('Instantaneous noise-free slope', 'Original / adjacent difference',
              'Original / endpoint difference', 'LTAI / whole-window slope')
    vmax = float(np.max(np.abs(maps[0])))
    fig, axes = plt.subplots(1, 4, figsize=(13.2, 3.8), constrained_layout=True)
    for ax, values, label in zip(axes, maps, labels):
        image = ax.imshow(values, cmap='RdBu_r', vmin=-vmax, vmax=vmax,
                          extent=(-width, width, width, -width))
        ax.set_title(label, fontsize=10)
        ax.set_xlabel('x (m)')
        ax.set_ylabel('y (m)')
        ax.set_box_aspect(1)
    fig.colorbar(image, ax=axes, shrink=.7, label=units)
    fig.suptitle(title)
    fig.savefig(output, dpi=180)
    fig.savefig(output.with_suffix('.svg'))
    plt.close(fig)


def depth_views(target, times, settings, reference_frames, framewise, predicted,
                reference, adjacent, endpoint, fitted):
    """Store axial source and reconstruction views before releasing full volumes."""
    depth = np.linspace(0, settings.range_m/2, settings.m)
    mask = (depth >= .52)&(depth <= .76)
    selected_depth = depth[mask]
    with np.load(target/'source_geometry.npz') as data:
        points = data['points']
        weights = data['weights']
        velocities = data['point_velocities_m_per_s']
    x_edges = np.linspace(-settings.width_m, settings.width_m, settings.n+1)
    z_edges = np.linspace(selected_depth[0], selected_depth[-1], 97)
    source = []
    for time in times:
        moved = points+time*velocities
        raster, _, _ = np.histogram2d(moved[:, 2], moved[:, 0],
                                      bins=(z_edges, x_edges), weights=weights)
        source.append(gaussian_filter(raster, (.7, .7)))
    def side(volume, signed=False):
        selected = volume[mask]
        values = selected.sum(axis=1) if signed else np.maximum(selected, 0).max(axis=1)
        return values[:, ::-1]
    np.savez_compressed(target/'depth_projections.npz',
                        source=np.stack(source),
                        reference=np.stack([side(v) for v in reference_frames]),
                        original_framewise=np.stack([side(v) for v in framewise]),
                        ltai=np.stack([side(v) for v in predicted]),
                        derivative=np.stack([side(v, signed=True) for v in
                                             (reference[1], adjacent[1], endpoint, fitted[1])]),
                        z_min=float(selected_depth[0]), z_max=float(selected_depth[-1]),
                        width_m=settings.width_m, slow_times_s=times)


def save_depth_visuals(target):
    """Render axial motion and signed depth derivative from saved small views."""
    with np.load(target/'depth_projections.npz') as data:
        times = data['slow_times_s']
        panels = [data[key] for key in ('source', 'reference', 'original_framewise', 'ltai')]
        derivatives = data['derivative']
        z_min, z_max, width = float(data['z_min']), float(data['z_max']), float(data['width_m'])
    names = ('True source / x-z', 'Noise-free LCT / x-z',
             'Original framewise / x-z', 'Quadratic LTAI / x-z')
    extent = (-width, width, z_max, z_min)
    reference_max = float(panels[1].max())
    source_max = float(panels[0].max())
    chosen = [0, len(times)//2, len(times)-1]
    fig, axes = plt.subplots(4, 3, figsize=(10, 9), constrained_layout=True)
    for row, (frames, name) in enumerate(zip(panels, names)):
        for column, index in enumerate(chosen):
            ax = axes[row, column]
            ax.imshow(frames[index], cmap='gray', vmin=0,
                      vmax=source_max if row == 0 else reference_max,
                      extent=extent, aspect='auto')
            ax.set_title(f'{name} / t={times[index]:+.2f} s', fontsize=9)
            ax.set_xlabel('x (m)')
            ax.set_ylabel('depth z (m)')
    fig.suptitle('Axial S expansion / fixed 0.2 s half-window')
    fig.savefig(target/'depth_motion.png', dpi=150)
    fig.savefig(target/'depth_motion.svg')
    plt.close(fig)

    fig, axes = plt.subplots(1, 4, figsize=(13, 3.8), constrained_layout=True)
    labels = ('Instantaneous noise-free slope', 'Adjacent difference',
              'Endpoint difference', 'Quadratic LTAI slope')
    vmax = float(np.max(np.abs(derivatives[0])))
    for ax, values, name in zip(axes, derivatives, labels):
        image = ax.imshow(values, cmap='RdBu_r', vmin=-vmax, vmax=vmax,
                          extent=extent, aspect='auto')
        ax.set_title(name, fontsize=9)
        ax.set_xlabel('x (m)')
        ax.set_ylabel('depth z (m)')
    fig.colorbar(image, ax=axes, shrink=.7, label='Signed depth derivative / author units per second')
    fig.savefig(target/'depth_derivative.png', dpi=150)
    fig.savefig(target/'depth_derivative.svg')
    plt.close(fig)

    fig, axes = plt.subplots(1, 4, figsize=(13, 3.3), constrained_layout=True)
    images = []
    for row, (ax, frames, name) in enumerate(zip(axes, panels, names)):
        images.append(ax.imshow(frames[0], cmap='gray', vmin=0,
                                vmax=source_max if row == 0 else reference_max,
                                extent=extent, aspect='auto'))
        ax.set_title(name, fontsize=9)
        ax.set_xlabel('x (m)')
        ax.set_ylabel('depth z (m)')
    heading = fig.suptitle('')
    def update(index):
        for image, frames in zip(images, panels):
            image.set_data(frames[index])
        heading.set_text(f'Axial S expansion / t={times[index]:+.2f} s')
        return images+[heading]
    animation = FuncAnimation(fig, update, frames=len(times), interval=500, blit=False)
    animation.save(target/'depth_motion.gif', writer=PillowWriter(fps=2), dpi=100)
    plt.close(fig)


def render_saved(output):
    """Refresh figures without rerunning forward simulation or Monte Carlo."""
    output = Path(output)
    report = json.loads((output/'report.json').read_text(encoding='utf-8'))
    times = np.asarray(report['slow_times_s'])
    for name, result in report['cases'].items():
        target = output/name
        with np.load(target/'projections.npz') as data:
            fronts = tuple(data[key] for key in ('reference', 'original_framewise',
                                                 'linear', 'ltai'))
            derivative_maps = data['derivative_maps']
        with np.load(target/'coefficients.npz') as data:
            reference = data['reference_signed']
            index = int(np.argmax(np.maximum(reference[0], 0).max(axis=(1, 2))))
            derivative_slices = [reference[1, index, :, ::-1],
                                 data['adjacent_signed'][1, index, :, ::-1],
                                 data['endpoint_slope_signed'][index, :, ::-1],
                                 data['ltai_signed'][1, index, :, ::-1]]
        settings = result['settings']
        depth = np.linspace(0, settings['range_m']/2, settings['m'])[index]
        source_fronts = None
        if name in ('translation', 'expansion', 'depth_expansion'):
            with np.load(target/'source_geometry.npz') as data:
                points, weights = data['points'], data['weights']
                velocity = (data['point_velocities_m_per_s'] if 'point_velocities_m_per_s' in data
                            else data['velocity_m_per_s'])
            edges = np.linspace(-settings['width_m'], settings['width_m'], 129)
            source_fronts = []
            for time in times:
                moved = points+time*velocity
                raster, _, _ = np.histogram2d(moved[:, 1], moved[:, 0], bins=(edges, edges), weights=weights)
                source_fronts.append(gaussian_filter(raster, .7))
            source_fronts = np.stack(source_fronts)
        save_visuals(fronts, source_fronts, times, settings['width_m'], derivative_maps,
                     'Affine density blend (not motion)' if name == 'affine' else
                     ('Physical S translation / v=0.08 m/s' if name == 'translation' else
                      'Physical S radial expansion / fixed point weights' if name == 'expansion' else
                      'Physical S axial expansion / fixed point weights'),
                     target, derivative_slices, depth)
        if name == 'depth_expansion':
            save_depth_visuals(target)
        result['derivative_display_depth_m'] = float(depth)
        (target/'report.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
        print(f'Rendered {name}: fixed derivative plane z={depth:.3f} m', flush=True)
    report['display'] = 'Common fixed LCT grayscale; signed fixed-depth slices and signed depth sums. No positivity on slope or metric volumes.'
    (output/'report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')


def run_case(name, times, source_dir, output, exposure, trials, seed):
    target = output/name
    target.mkdir(parents=True, exist_ok=True)
    means, instantaneous_derivative, settings, metadata, source_fronts = prepare_case(
        name, times, source_dir, target)
    if np.min(means) < -1e-10:
        raise ValueError('Simulated photon rates must be nonnegative.')
    means = np.maximum(means, 0)
    model = OriginalLCT(settings.n, settings.m, settings.width_m, settings.range_m,
                        settings.diffuse, settings.snr)
    weights = fit_weights(times, degree=2)
    linear_weights = fit_weights(times)
    linear_from_quadratic = linear_weights@np.column_stack((np.ones(len(times)), times, times**2))
    baseline_weights = centered_difference_weights(times)
    endpoint_weights = np.zeros(len(times))
    endpoint_weights[[0, -1]] = [-1/(times[-1]-times[0]), 1/(times[-1]-times[0])]
    middle = len(times)//2
    reference = np.stack((model(means[middle]), model(instantaneous_derivative)))
    expected_coefficients = np.tensordot(weights, means, axes=(1, 0))
    baseline_expected = np.tensordot(baseline_weights, means, axes=(1, 0))
    endpoint_expected = np.tensordot(endpoint_weights, means, axes=(0, 0))
    ideal = np.stack([model(value) for value in expected_coefficients])
    linear_ideal = np.einsum('ij,jklm->iklm', linear_from_quadratic, ideal)
    baseline_ideal = np.stack([model(value) for value in baseline_expected])
    endpoint_ideal = model(endpoint_expected)
    reference_frames = []
    for index, mean in enumerate(means):
        reference_frames.append(model(mean))
        print(f'  {name}: noise-free frame {index+1}/{len(times)} reconstructed', flush=True)
    reference_frames = np.stack(reference_frames)
    fit_after_inverse = np.tensordot(weights, reference_frames, axes=(1, 0))
    noiseless_commutation = relative_error(ideal, fit_after_inverse)
    del fit_after_inverse, expected_coefficients, baseline_expected, endpoint_expected
    norms = np.sum(reference**2, axis=(1, 2, 3))
    if np.any(norms == 0):
        raise ValueError('Both reference coefficients must have nonzero signal energy.')
    biases = {'ltai': np.sum((ideal[:2]-reference)**2, axis=(1, 2, 3)),
              'linear': np.sum((linear_ideal-reference)**2, axis=(1, 2, 3)),
              'adjacent': np.sum((baseline_ideal-reference)**2, axis=(1, 2, 3)),
              'endpoint_slope': np.sum((endpoint_ideal-reference[1])**2)}
    noise = {key: [] for key in ('ltai', 'linear', 'adjacent', 'endpoint')}
    mse = {key: [] for key in noise}
    rng = np.random.default_rng(np.random.SeedSequence([seed, 0 if name == 'affine' else 1]))
    selected = None
    selected_frame_errors = None
    noisy_commutation = None
    derivative_maps = None
    derivative_slices = None
    slice_index = int(np.argmax(np.maximum(reference[0], 0).max(axis=(1, 2))))
    slice_depth = np.linspace(0, settings.range_m/2, settings.m)[slice_index]
    for trial in range(trials):
        estimator = PolynomialTransientEstimator(times, degree=2)
        adjacent_input = np.zeros_like(reference)
        endpoint_input = np.zeros_like(reference[0])
        saved_frames = []
        for index, mean in enumerate(means):
            counts = rng.poisson(exposure*mean)
            estimator.add(counts, exposure)
            for coefficient in range(2):
                adjacent_input[coefficient] += baseline_weights[coefficient, index]*counts/exposure
            endpoint_input += endpoint_weights[index]*counts/exposure
            if trial == 0:
                saved_frames.append(counts)
        fitted_inputs = estimator.estimate()
        fitted = np.stack([model(value) for value in fitted_inputs])
        linear_fitted = np.einsum('ij,jklm->iklm', linear_from_quadratic, fitted)
        adjacent = np.stack([model(value) for value in adjacent_input])
        endpoint = model(endpoint_input)
        noise['ltai'].append(np.sum((fitted[:2]-ideal[:2])**2, axis=(1, 2, 3)))
        noise['linear'].append(np.sum((linear_fitted-linear_ideal)**2, axis=(1, 2, 3)))
        noise['adjacent'].append(np.sum((adjacent-baseline_ideal)**2, axis=(1, 2, 3)))
        noise['endpoint'].append(float(np.sum((endpoint-endpoint_ideal)**2)))
        mse['ltai'].append(np.sum((fitted[:2]-reference)**2, axis=(1, 2, 3)))
        mse['linear'].append(np.sum((linear_fitted-reference)**2, axis=(1, 2, 3)))
        mse['adjacent'].append(np.sum((adjacent-reference)**2, axis=(1, 2, 3)))
        mse['endpoint'].append(float(np.sum((endpoint-reference[1])**2)))
        if trial == 0:
            framewise = []
            for counts in saved_frames:
                framewise.append(model(counts/exposure))
            framewise = np.stack(framewise)
            fit_after_inverse = np.tensordot(weights, framewise, axes=(1, 0))
            noisy_commutation = relative_error(fitted, fit_after_inverse)
            predicted = (fitted[0]+times[:, None, None, None]*fitted[1]
                         +times[:, None, None, None]**2*fitted[2])
            linear_predicted = linear_fitted[0]+times[:, None, None, None]*linear_fitted[1]
            selected_frame_errors = {
                'framewise_relative_error_by_time': [relative_error(a, b) for a, b in zip(framewise, reference_frames)],
                'linear_relative_error_by_time': [relative_error(a, b) for a, b in zip(linear_predicted, reference_frames)],
                'ltai_relative_error_by_time': [relative_error(a, b) for a, b in zip(predicted, reference_frames)]}
            selected = (np.stack([front(model, settings, v) for v in reference_frames]),
                        np.stack([front(model, settings, v) for v in framewise]),
                        np.stack([front(model, settings, v) for v in linear_predicted]),
                        np.stack([front(model, settings, v) for v in predicted]))
            derivative_maps = [front(model, settings, v, signed=True)
                               for v in (reference[1], adjacent[1], endpoint, fitted[1])]
            derivative_slices = [v[slice_index, :, ::-1].copy()
                                 for v in (reference[1], adjacent[1], endpoint, fitted[1])]
            np.savez_compressed(target/'observations.npz', slow_times_s=times,
                                expected_transients=means, counts=np.stack(saved_frames),
                                exposure_per_frame=exposure, fitted_transients=fitted_inputs,
                                instantaneous_transient_derivative=instantaneous_derivative)
            np.savez_compressed(target/'coefficients.npz', reference_signed=reference,
                                expected_window_signed=ideal, ltai_signed=fitted,
                                linear_expected_signed=linear_ideal, linear_signed=linear_fitted,
                                adjacent_signed=adjacent, endpoint_slope_signed=endpoint,
                                lateral_m=np.linspace(-settings.width_m, settings.width_m, settings.n),
                                depth_m=np.linspace(0, settings.range_m/2, settings.m))
            np.savez_compressed(target/'projections.npz', reference=selected[0],
                                original_framewise=selected[1], linear=selected[2], ltai=selected[3],
                                derivative_maps=np.stack(derivative_maps), slow_times_s=times)
            if name == 'depth_expansion':
                depth_views(target, times, settings, reference_frames, framewise, predicted,
                            reference, adjacent, endpoint, fitted)
            del framewise, predicted, linear_predicted, fit_after_inverse, saved_frames
        if (trial+1) % 3 == 0 or trial+1 == trials:
            print(f'  {name}: {trial+1}/{trials} independent noisy sequences', flush=True)
        del estimator, fitted_inputs, adjacent_input, endpoint_input, fitted, linear_fitted, adjacent, endpoint

    noise = {key: np.asarray(value) for key, value in noise.items()}
    mse = {key: np.asarray(value) for key, value in mse.items()}
    exact_affine = name == 'affine'
    ref_prediction = float(np.sum(weights[0]**2)/np.sum(baseline_weights[0]**2))
    slope_prediction = float(np.sum(weights[1]**2)/np.sum(baseline_weights[1]**2))
    endpoint_prediction = float(np.sum(weights[1]**2)/np.sum(endpoint_weights**2))
    ratios = {'reference_vs_single': paired_ratio(noise['adjacent'][:, 0], noise['ltai'][:, 0],
                                                  ref_prediction if exact_affine else None),
              'quadratic_reference_vs_linear': paired_ratio(noise['linear'][:, 0], noise['ltai'][:, 0],
                                                              float(np.sum(weights[0]**2)/np.sum(linear_weights[0]**2))
                                                              if exact_affine else None),
              'slope_vs_adjacent': paired_ratio(noise['adjacent'][:, 1], noise['ltai'][:, 1],
                                                slope_prediction if exact_affine else None),
              'slope_vs_endpoints': paired_ratio(noise['endpoint'], noise['ltai'][:, 1],
                                                 endpoint_prediction if exact_affine else None)}
    metrics = {}
    for key in ('ltai', 'linear', 'adjacent'):
        metrics[key] = {'relative_temporal_bias': np.sqrt(biases[key]/norms).tolist(),
                        'relative_noise_rms': np.sqrt(noise[key].mean(axis=0)/norms).tolist(),
                        'relative_total_rms_error': np.sqrt(mse[key].mean(axis=0)/norms).tolist()}
    metrics['endpoint_slope'] = {
        'relative_temporal_bias': float(np.sqrt(biases['endpoint_slope']/norms[1])),
        'relative_noise_rms': float(np.sqrt(noise['endpoint'].mean()/norms[1])),
        'relative_total_rms_error': float(np.sqrt(mse['endpoint'].mean()/norms[1]))}
    checks = {'noiseless_fit_inverse_commutation': noiseless_commutation < 1e-10,
              'noisy_fit_inverse_commutation': noisy_commutation < 1e-10,
              'finite_results': all(np.all(np.isfinite(values)) for values in mse.values())}
    if exact_affine:
        checks['exact_affine_noiseless_reference_and_slope'] = bool(
            np.all(np.sqrt(biases['ltai']/norms) < 1e-10) and
            np.all(np.sqrt(biases['linear']/norms) < 1e-10))
        for key, result in ratios.items():
            checks[f'noise_scaling_{key}'] = result['consistent_with_prediction']
    else:
        checks['physical_reference_depth'] = bool(abs(np.linspace(0, settings.range_m/2, settings.m)[
            np.argmax(np.maximum(reference[0], 0).max(axis=(1, 2)))]-.65) < .02)
    save_visuals(selected, source_fronts, times, settings.width_m, derivative_maps,
                 ('Affine density blend (not motion)' if exact_affine else
                  'Physical S translation / v=0.08 m/s' if name == 'translation' else
                  'Physical S radial expansion / fixed point weights' if name == 'expansion' else
                  'Physical S axial expansion / fixed point weights'),
                 target, derivative_slices, slice_depth)
    if name == 'depth_expansion':
        save_depth_visuals(target)
    result = {'settings': settings.metadata(), 'data_model': metadata, 'metrics': metrics,
              'noise_ratios': ratios, 'noiseless_commutation_relative_error': noiseless_commutation,
              'noisy_commutation_relative_error': noisy_commutation,
              'derivative_display_depth_m': float(slice_depth),
              'selected_sequence_frame_errors': selected_frame_errors,
              'expected_total_sequence_photons': float(exposure*means.sum()),
              'inverse_calls': {'quadratic_coefficient_fit': 3, 'center_and_slope_only': 2,
                                'framewise_then_same_fit': len(times)},
              'implementation_checks': checks, 'passed': bool(all(checks.values())),
              'note': 'Passed means numerical/statistical checks passed, not that physical density or velocity '
                      'was recovered exactly. Temporal bias is measured against the instantaneous noise-free LCT target.'}
    (target/'report.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(f'{name}: total relative RMS (reference, slope): {metrics["ltai"]["relative_total_rms_error"]}; '
          f'passed={result["passed"]}', flush=True)
    return result


def run(source_dir=DEFAULT_ROOT/'cnlos2018', output=DEFAULT_OUTPUT, cases=('affine', 'translation'),
        frames=7, half_window=.3, exposure=1., trials=12, seed=20261030):
    source_dir, output = Path(source_dir), Path(output)
    verify_source(source_dir)
    output.mkdir(parents=True, exist_ok=True)
    times = np.linspace(-half_window, half_window, frames)
    report = {'source_sha256': SOURCE_SHA256, 'frames': frames, 'slow_times_s': times.tolist(),
              'exposure_per_frame': exposure, 'trials': trials, 'seed': seed,
              'scope': 'Quadratic slow-time fit followed by fixed author LCT. No velocity-field inversion.',
              'budget': 'All methods share one recorded sequence. Single-frame, adjacent and endpoint baselines '
                        'use subsets of it; linear and quadratic fits use the same complete sequence. '
                        'Original LCT of all frames followed by identical quadratic OLS is numerically '
                        'equivalent to quadratic LTAI. Its equivalence is checked.',
              'display': 'Common fixed LCT grayscale for each sequence; signed fixed-depth slices and depth sums, '
                         'no positivity on slope or metric volumes.', 'cases': {}}
    for case in cases:
        print(f'Dynamic case {case}: preparing...', flush=True)
        report['cases'][case] = run_case(case, times, source_dir, output, exposure, trials, seed)
    report['passed'] = all(case['passed'] for case in report['cases'].values())
    (output/'report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    if not report['passed']:
        raise RuntimeError('Dynamic implementation/statistical validation failed; inspect report.json.')
    print(f'Results: {output.resolve()}', flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-dir', type=Path, default=DEFAULT_ROOT/'cnlos2018')
    parser.add_argument('--output', type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument('--cases', nargs='+', choices=('affine', 'translation', 'expansion', 'depth_expansion'),
                        default=['affine', 'translation'])
    parser.add_argument('--frames', type=int, default=7)
    parser.add_argument('--half-window', type=float, default=.3)
    parser.add_argument('--exposure', type=float, default=1.)
    parser.add_argument('--trials', type=int, default=12)
    parser.add_argument('--seed', type=int, default=20261030)
    parser.add_argument('--threads', type=int, default=2)
    parser.add_argument('--render-only', action='store_true', help='Refresh saved figures without simulation.')
    args = parser.parse_args()
    if args.frames < 3 or args.frames % 2 == 0 or args.trials < 4 or args.threads < 1:
        parser.error('Use an odd frame count >=3, at least 4 trials and a positive thread count.')
    if not np.isfinite(args.half_window) or args.half_window <= 0 or not np.isfinite(args.exposure) or args.exposure <= 0:
        parser.error('Window and exposure must be finite and positive.')
    with threadpool_limits(args.threads):
        if args.render_only:
            render_saved(args.output)
        else:
            run(args.source_dir, args.output, args.cases, args.frames, args.half_window,
                args.exposure, args.trials, args.seed)


if __name__ == '__main__':
    main()
