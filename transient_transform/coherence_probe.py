"""Measure finite-shift template coherence on the ideal planar relay wall."""

import argparse
import json
from pathlib import Path

import numpy as np

from .finite_shift_probe import (DEFAULT_INPUT, oracle_lateral_patch,
                                 point_frames, select_wall_and_bins)


def weighted_feature(response, precision):
    """Remove an arbitrary static transient at each wall point and fast bin."""
    if response.shape != precision.shape or np.any(precision <= 0):
        raise ValueError('Response and positive precision must share shape.')
    static = np.sum(precision*response, axis=0)/np.sum(precision, axis=0)
    return ((response-static[None])*np.sqrt(precision)).ravel()


def normalized_feature(xy, z, velocity, wall, times, edges, precision):
    vector = weighted_feature(point_frames(xy, z, velocity, wall, times, edges),
                              precision)
    norm = float(np.linalg.norm(vector))
    if norm <= 0:
        raise ValueError('A static point has zero time-centered template.')
    return vector/norm, norm


def summarize_map(values, first_axis, second_axis, thresholds=(.95, .99)):
    result = {}
    for threshold in thresholds:
        indices = np.argwhere(values >= threshold)
        result[str(threshold)] = {
            'count': int(len(indices)),
            'first_range': [float(first_axis[indices[:, 0]].min()),
                            float(first_axis[indices[:, 0]].max())] if len(indices) else None,
            'second_range': [float(second_axis[indices[:, 1]].min()),
                             float(second_axis[indices[:, 1]].max())] if len(indices) else None,
        }
    return result


def save_figure(report, output):
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib import pyplot as plt

    depths = np.asarray(report['depths_m'])
    velocities = np.asarray(report['velocities_m_per_s'])
    offsets = np.asarray(report['lateral_offsets_m'])
    maps = report['maps']
    fig, axes = plt.subplots(2, 2, figsize=(10, 8), constrained_layout=True)
    panels = [
        (axes[0, 0], np.asarray(maps['depth_velocity']).T,
         [depths[0], depths[-1], velocities[0], velocities[-1]],
         'Depth / axial velocity, fixed x,y', 'Center depth (m)', 'Velocity (m/s)'),
        (axes[0, 1], np.asarray(maps['lateral_xy']).T,
         [offsets[0], offsets[-1], offsets[0], offsets[-1]],
         'Lateral x/y offsets, fixed depth and velocity', 'x offset (m)', 'y offset (m)'),
        (axes[1, 0], np.asarray(maps['x_depth_best_velocity']).T,
         [offsets[0], offsets[-1], depths[0], depths[-1]],
         'x/depth after choosing the best velocity', 'x offset (m)', 'Center depth (m)'),
    ]
    for ax, values, extent, title, xlabel, ylabel in panels:
        image = ax.imshow(values, origin='lower', aspect='auto', extent=extent,
                          cmap='viridis', vmin=0, vmax=1)
        ax.set(title=title, xlabel=xlabel, ylabel=ylabel)
    axes[0, 0].plot([report['anchor']['depth_m']],
                    [report['anchor']['velocity_m_per_s']], 'rx')
    axes[0, 1].plot([0], [0], 'rx')
    axes[1, 0].plot([0], [report['anchor']['depth_m']], 'rx')
    speed = report['velocity_profiles']
    for name, values in speed.items():
        axes[1, 1].plot(values['candidate_velocities_m_per_s'],
                        values['coherence'], marker='o', label=name)
    axes[1, 1].axhline(.95, color='gray', linestyle=':', linewidth=1)
    axes[1, 1].set(xlabel='Candidate velocity (m/s)', ylabel='Signed coherence',
                   title='Velocity ambiguity at a fixed point')
    axes[1, 1].legend(fontsize=8)
    fig.colorbar(image, ax=axes[:, :], shrink=.75,
                 label='Signed normalized template inner product')
    fig.savefig(output, dpi=150)
    fig.savefig(output.with_suffix('.svg'))
    plt.close(fig)


def run(source=DEFAULT_INPUT, output=None):
    source = Path(source)
    output = Path(output) if output is not None else source/'coherence_probe.json'
    metadata = json.loads((source/'report.json').read_text(encoding='utf-8'))
    geometry = metadata['settings']
    with np.load(source/'observations.npz') as data:
        times = data['slow_times_s']
        expected = data['expected_transients']
    xy, local_wall, local_ids = oracle_lateral_patch(geometry)
    n = geometry['n']
    axis = np.linspace(-geometry['width_m'], geometry['width_m'], n)
    wx, wy = np.meshgrid(axis[::-1], axis, indexing='xy')
    all_wall = np.column_stack((wx.ravel(), wy.ravel(), np.zeros(wx.size)))
    spread = np.array([0, 10, 21, 31, 42, 53, 63])
    spread_ids = (spread[:, None]*n+spread[None, :]).ravel()
    spread_wall = all_wall[spread_ids]
    start, stop = 175, 400
    edges = np.arange(start, stop+1)*geometry['bin_resolution_s']
    local_means = select_wall_and_bins(expected, local_ids, start, stop)
    spread_means = select_wall_and_bins(expected, spread_ids, start, stop)
    # Unit precision exposes geometry of the response dictionary itself.
    # The scene-derived Poisson approximation is retained as a sensitivity
    # comparison, since it can strongly emphasize low-count time bins.
    local_precision = np.ones_like(local_means)
    spread_precision = np.ones_like(spread_means)
    local_poisson = 1/np.maximum(local_means, 1.)
    spread_poisson = 1/np.maximum(spread_means, 1.)
    anchor_z, anchor_v = .61, -.08
    local_anchor, anchor_norm = normalized_feature(
        xy, anchor_z, anchor_v, local_wall, times, edges, local_precision)
    spread_anchor, spread_norm = normalized_feature(
        xy, anchor_z, anchor_v, spread_wall, times, edges, spread_precision)
    local_poisson_anchor, local_poisson_norm = normalized_feature(
        xy, anchor_z, anchor_v, local_wall, times, edges, local_poisson)
    spread_poisson_anchor, spread_poisson_norm = normalized_feature(
        xy, anchor_z, anchor_v, spread_wall, times, edges, spread_poisson)

    def local(z=anchor_z, v=anchor_v, offset=(0., 0.)):
        candidate_xy = xy+np.asarray(offset)
        feature, norm = normalized_feature(candidate_xy, z, v, local_wall,
                                           times, edges, local_precision)
        return float(local_anchor@feature), float(norm/anchor_norm)

    def wide(z=anchor_z, v=anchor_v, offset=(0., 0.)):
        candidate_xy = xy+np.asarray(offset)
        feature, norm = normalized_feature(candidate_xy, z, v, spread_wall,
                                           times, edges, spread_precision)
        return float(spread_anchor@feature), float(norm/spread_norm)

    def poisson(z, v, offset, wall, precision, anchor, anchor_size):
        feature, norm = normalized_feature(xy+np.asarray(offset), z, v,
                                           wall, times, edges, precision)
        return float(anchor@feature), float(norm/anchor_size)

    depths = np.linspace(.59, .63, 21)
    velocities = np.linspace(-.12, -.04, 9)
    offsets = np.linspace(-.03, .03, 13)
    depth_velocity = np.empty((len(depths), len(velocities)))
    for i, z in enumerate(depths):
        for j, v in enumerate(velocities):
            depth_velocity[i, j] = local(z=z, v=v)[0]
    lateral_xy = np.empty((len(offsets), len(offsets)))
    for i, dx in enumerate(offsets):
        for j, dy in enumerate(offsets):
            lateral_xy[i, j] = local(offset=(dx, dy))[0]
    x_depth_best_velocity = np.empty((len(offsets), len(depths)))
    x_depth_best_velocity_argmax = np.empty_like(x_depth_best_velocity)
    for i, dx in enumerate(offsets):
        for j, z in enumerate(depths):
            values = np.array([local(z=z, v=v, offset=(dx, 0.))[0]
                               for v in velocities])
            index = int(np.argmax(values))
            x_depth_best_velocity[i, j] = values[index]
            x_depth_best_velocity_argmax[i, j] = velocities[index]

    # A coarse four-parameter neighborhood checks for compensation between
    # lateral position, depth, and velocity beyond the plotted 2-D slices.
    joint_offsets = np.linspace(-.015, .015, 7)
    joint_depths = np.linspace(.602, .618, 9)
    joint_velocities = np.linspace(-.10, -.06, 5)
    joint = np.empty((len(joint_offsets), len(joint_offsets),
                      len(joint_depths), len(joint_velocities)))
    for i, dx in enumerate(joint_offsets):
        for j, dy in enumerate(joint_offsets):
            for k, z in enumerate(joint_depths):
                for l, v in enumerate(joint_velocities):
                    joint[i, j, k, l] = local(z=z, v=v, offset=(dx, dy))[0]
    joint_summary = {}
    for threshold in (.95, .99):
        found = np.argwhere(joint >= threshold)
        joint_summary[str(threshold)] = {
            'count': int(len(found)),
            'xy_offset_range_m': [float(joint_offsets[found[:, :2]].min()),
                                  float(joint_offsets[found[:, :2]].max())]
                                 if len(found) else None,
            'depth_range_m': [float(joint_depths[found[:, 2]].min()),
                              float(joint_depths[found[:, 2]].max())]
                             if len(found) else None,
            'velocity_range_m_per_s': [float(joint_velocities[found[:, 3]].min()),
                                       float(joint_velocities[found[:, 3]].max())]
                                      if len(found) else None}

    candidate_speeds = np.array([-.12, -.10, -.08, -.06, -.04, -.02,
                                 -.01, -.005, -.002, -.001])
    velocity_profiles = {}
    for name, reference_v in [('fast -0.08 m/s', -.08),
                              ('slow -0.005 m/s', -.005)]:
        reference, reference_norm = normalized_feature(
            xy, anchor_z, reference_v, local_wall, times, edges, local_precision)
        correlations, norm_ratios = [], []
        for candidate_v in candidate_speeds:
            feature, norm = normalized_feature(
                xy, anchor_z, candidate_v, local_wall, times, edges,
                local_precision)
            correlations.append(float(reference@feature))
            norm_ratios.append(float(norm/reference_norm))
        velocity_profiles[name] = {
            'candidate_velocities_m_per_s': candidate_speeds.tolist(),
            'coherence': correlations, 'norm_ratio_to_reference': norm_ratios}

    pairs = {
        '2mm_depth_offset': (anchor_z+.002, anchor_v, (0., 0.)),
        '20mm_per_s_speed_offset': (anchor_z, anchor_v-.02, (0., 0.)),
        '5mm_x_offset': (anchor_z, anchor_v, (.005, 0.)),
        '5mm_y_offset': (anchor_z, anchor_v, (0., .005)),
        '2mm_depth_plus_5mm_x': (anchor_z+.002, anchor_v, (.005, 0.)),
        'opposite_outer_layer': (.69, .08, (0., 0.)),
    }
    comparison = {}
    for name, (z, v, offset) in pairs.items():
        comparison[name] = {
            'candidate': {'depth_m': z, 'velocity_m_per_s': v,
                          'xy_offset_m': list(offset)}}
        for label, result in [
                ('local_7x7_uniform', local(z, v, offset)),
                ('spread_7x7_uniform', wide(z, v, offset)),
                ('local_7x7_poisson', poisson(z, v, offset, local_wall,
                    local_poisson, local_poisson_anchor, local_poisson_norm)),
                ('spread_7x7_poisson', poisson(z, v, offset, spread_wall,
                    spread_poisson, spread_poisson_anchor, spread_poisson_norm))]:
            comparison[name][label] = {'coherence': result[0], 'norm_ratio': result[1]}
    # The normalized two-atom Gram eigenvalues are 1 +/- |mu|.
    for case in comparison.values():
        for sampling in ('local_7x7_uniform', 'spread_7x7_uniform',
                         'local_7x7_poisson', 'spread_7x7_poisson'):
            mu = abs(case[sampling]['coherence'])
            case[sampling]['two_atom_gram_condition'] = float((1+mu)/(1-mu))

    upper_anchor = {}
    for label, wall, precision in [('local_7x7_uniform', local_wall, local_precision),
                                   ('spread_7x7_uniform', spread_wall, spread_precision)]:
        reference, _ = normalized_feature(xy, .69, .08, wall, times, edges,
                                          precision)
        upper_anchor[label] = {}
        for candidate_name, candidate in {
                '5mm_x_offset': (xy+np.array([.005, 0.]), .69, .08),
                '2mm_depth_offset': (xy, .692, .08),
                '20mm_per_s_speed_offset': (xy, .69, .10)}.items():
            feature, _ = normalized_feature(candidate[0], candidate[1],
                                            candidate[2], wall, times, edges,
                                            precision)
            upper_anchor[label][candidate_name] = float(reference@feature)

    report = {
        'source': str(source), 'anchor': {'xy_m': xy.tolist(), 'depth_m': anchor_z,
                                         'velocity_m_per_s': anchor_v},
        'measurement': {'times_s': times.tolist(), 'irf_fwhm_s': 60e-12,
                        'fast_bin_range': [start, stop], 'local_wall_points': 49,
                        'spread_wall_points': 49,
                        'primary_precision': 'Uniform per frame/wall/bin',
                        'sensitivity_precision': '1/max(ideal expected Poisson counts,1), per frame/wall/bin',
                        'static_nuisance': 'Weighted least-squares projection of a constant slow-time waveform per wall/bin'},
        'definition': 'Signed inner product of unit, noise-whitened finite-shift point templates.',
        'depths_m': depths.tolist(), 'velocities_m_per_s': velocities.tolist(),
        'lateral_offsets_m': offsets.tolist(),
        'maps': {'depth_velocity': depth_velocity.tolist(),
                 'lateral_xy': lateral_xy.tolist(),
                 'x_depth_best_velocity': x_depth_best_velocity.tolist(),
                 'x_depth_best_velocity_argmax_m_per_s':
                     x_depth_best_velocity_argmax.tolist()},
        'regions': {'depth_velocity': summarize_map(depth_velocity, depths, velocities),
                    'lateral_xy': summarize_map(lateral_xy, offsets, offsets),
                    'x_depth_best_velocity': summarize_map(
                        x_depth_best_velocity, offsets, depths),
                    'joint_xy_depth_velocity': joint_summary},
        'joint_grid': {'xy_offsets_m': joint_offsets.tolist(),
                       'depths_m': joint_depths.tolist(),
                       'velocities_m_per_s': joint_velocities.tolist(),
                       'coherence': joint.tolist()},
        'velocity_profiles': velocity_profiles,
        'aperture_comparison': comparison,
        'upper_layer_anchor_comparison': upper_anchor,
        'limitations': 'Single point templates and one anchor; 49-point aperture comparisons keep count of wall samples fixed, but not exposure or photon budget. Geometry remains ideal planar confocal.'
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding='utf-8')
    save_figure(report, output.with_name('coherence_probe.png'))
    print(json.dumps({'regions': report['regions'],
                      'selected_uniform_comparisons': {
                          name: {key: value['coherence'] for key, value in case.items()
                                 if key.endswith('_uniform')}
                          for name, case in comparison.items()},
                      'upper_layer_anchor_comparison': upper_anchor}, indent=2))
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=DEFAULT_INPUT)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    run(args.source, args.output)
