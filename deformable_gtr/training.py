"""动态联合反演、明确的空间留出验证，以及可重载的扩展 checkpoint。"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch

from .model import AxisScaling, DeformableGTR, canonical_from_arrays, canonical_from_config


def load_canonical(path, device='cpu'):
    path = Path(path)
    if path.suffix.lower() == '.npz':
        with np.load(path, allow_pickle=False) as data:
            return canonical_from_arrays(data['centers'], data['scales'], data['quats'], data['albedos'], device)
    # 复用原始 checkpoint 格式，不依赖训练入口的 EasyDict/YAML/CUDA 环境。
    from save_load_model import clean_state_dict
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    state = clean_state_dict(checkpoint['state_dict'])
    source = checkpoint.get('config')
    if not isinstance(source, dict):
        raise ValueError('Original checkpoint must include a dictionary config.')
    names = ['x_min', 'x_max', 'y_min', 'y_max', 'z_min', 'z_max',
             'scale_min', 'scale_max', 'scale_init', 'albedo_init',
             'activation_mode', 'isplanar', 'isconfocal', 'isretroreflective']
    missing = [name for name in names if name not in source]
    if missing:
        raise ValueError(f'Canonical checkpoint config is missing: {missing}')
    config = {name: source[name] for name in names}
    config.update(num_gaussians=int(state['xy_centers_raw'].shape[0]),
                  scale_activation_mode=source.get('scale_activation_mode', 'relu'),
                  albedo_activation_mode=source.get('albedo_activation_mode', 'relu'),
                  anisotropy_threshold=source.get('anisotropy_threshold', 0.0))
    model = canonical_from_config(config, device, state['xy_centers_raw'].dtype)
    model.load_state_dict(state)
    return model


def save_dynamic(model, path):
    torch.save({'schema_version': 1, 'specification': model.specification(),
                'dtype': str(next(model.parameters()).dtype).split('.')[-1],
                'state_dict': model.state_dict()}, path)


def load_dynamic(path, device='cpu', backend=None):
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    if checkpoint.get('schema_version') != 1:
        raise ValueError('Not a supported deformable 3D-GTR checkpoint.')
    spec = checkpoint['specification']
    dtype = getattr(torch, checkpoint['dtype'])
    canonical = canonical_from_config(spec['canonical'], device, dtype)
    motion = AxisScaling(**spec['motion']).to(device=device, dtype=dtype)
    render = dict(spec['render'])
    if backend is not None:
        render['backend'] = backend
    model = DeformableGTR(canonical, motion, **render)
    model.load_state_dict(checkpoint['state_dict'])
    return model


def split_spatial_rows(data, fraction=0.2, seed=0):
    """相同照明图案+相机墙点的所有时刻分到同一侧，避免时序重复泄漏。

    验证的是未参与训练的采集配置；它不等于未见物体或未见时间的泛化。
    """
    if not 0 <= fraction < 1:
        raise ValueError('validation fraction must be in [0,1).')
    groups = {}
    cameras = data.camera_points.detach().cpu().numpy()
    patterns = data.patterns.detach().cpu().numpy()
    for row, (camera, pattern) in enumerate(zip(cameras, patterns)):
        key = (tuple(np.round(camera, 7)), tuple(np.round(pattern, 7)))
        groups.setdefault(key, []).append(row)
    values = list(groups.values())
    if fraction == 0:
        return torch.arange(len(data), device=data.device), torch.empty(0, dtype=torch.long, device=data.device)
    if len(values) < 2:
        raise ValueError('Spatial validation needs >=2 measurement configurations; otherwise use --validation 0.')
    order = np.random.default_rng(seed).permutation(len(values))
    count = min(max(1, round(len(values)*fraction)), len(values)-1)
    valid = [row for group in order[:count] for row in values[group]]
    train = [row for group in order[count:] for row in values[group]]
    return (torch.tensor(sorted(train), device=data.device), torch.tensor(sorted(valid), device=data.device))


def normalized_mse(predicted, observed):
    denominator = observed.square().mean().clamp_min(torch.finfo(observed.dtype).tiny)
    return (predicted-observed).square().mean()/denominator


def fit(model, data, *, iterations=400, fit_canonical='centers_albedo', validation=0.2,
        seed=0, loss='mse', canonical_lr=0.008, motion_lr=0.02, smoothness=1e-5,
        anchor_weight=1e-5, batch_size=0, log_every=100, pivot_lr=0.001):
    """联合优化共享参考和运动参数；原始静态训练入口保持不变。

    loss=mse 兼容基线强度单位。loss=poisson 要求原始整数计数和已校准均值；
    它适用于完整/低通量计数，不能拿强 pile-up 数据直接代入。
    """
    if iterations < 1 or batch_size < 0:
        raise ValueError('Invalid iteration or batch count.')
    if fit_canonical not in ('none', 'centers_albedo', 'all'):
        raise ValueError(fit_canonical)
    if loss not in ('mse', 'poisson'):
        raise ValueError(loss)
    if loss == 'poisson' and ((data.transients < 0).any() or not torch.allclose(data.transients, data.transients.round())):
        raise ValueError('Poisson mode requires nonnegative raw integer counts, not scaled/denoised transients.')
    if model.dt != data.dt or tuple(model.t_range) != data.t_range:
        raise ValueError('Data/model temporal coordinates do not match.')
    if data.exposure_start.min() < float(model.motion.knot_times[0])-1e-7 or data.exposure_end.max() > float(model.motion.knot_times[-1])+1e-7:
        raise ValueError('Motion knots must cover every exposure interval.')
    canonical = model.canonical
    active_names = {'xy_centers_raw', 'z_centers_raw', 'albedo'} if fit_canonical == 'centers_albedo' else set()
    for name, parameter in canonical.named_parameters():
        parameter.requires_grad_(fit_canonical == 'all' or name in active_names)
    groups = []
    canonical_parameters = [p for p in canonical.parameters() if p.requires_grad]
    # pivot 的单位是米，log_scale 无量纲；不能共享同一数值学习率。
    motion_parameters = [p for name, p in model.motion.named_parameters() if p.requires_grad and name != 'pivot']
    if canonical_parameters:
        groups.append({'params': canonical_parameters, 'lr': canonical_lr})
    if motion_parameters:
        groups.append({'params': motion_parameters, 'lr': motion_lr})
    if model.motion.learn_pivot:
        groups.append({'params': [model.motion.pivot], 'lr': pivot_lr})
    if not groups:
        raise ValueError('No trainable parameters: use a dynamic mode or enable canonical fitting.')
    optimizer = torch.optim.Adam(groups)
    train_rows, valid_rows = split_spatial_rows(data, validation, seed)
    generator = torch.Generator(device=data.device).manual_seed(seed)
    anchor = canonical.get_gs_centers().detach().clone()
    initial_pivot = model.motion.pivot.detach().cpu().tolist()
    history = []
    started = time.perf_counter()
    for step in range(iterations):
        model.train()
        rows = train_rows
        if batch_size and len(rows) > batch_size:
            rows = rows[torch.randperm(len(rows), generator=generator, device=data.device)[:batch_size]]
        optimizer.zero_grad(set_to_none=True)
        predicted = model(data, rows)
        observed = data.transients[rows]
        if loss == 'mse':
            objective = normalized_mse(predicted, observed)
        else:
            # 数值 floor 只用于训练 log；不用于计算 FI 或宣称物理信息。
            objective = torch.nn.functional.poisson_nll_loss(
                predicted.clamp_min(1e-8), observed, log_input=False,
            )/observed.mean().clamp_min(1.0)
        penalty = smoothness*model.motion.smoothness()
        if canonical_parameters:
            penalty = penalty+anchor_weight*((canonical.get_gs_centers()-anchor)/0.01).square().mean()
        total = objective+penalty
        if not torch.isfinite(total):
            raise FloatingPointError('Nonfinite training loss.')
        total.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0, error_if_nonfinite=True)
        optimizer.step()
        # 第一版限制尺度 [0.5,2]，保证数值稳定；触边会在报告中显式记录。
        with torch.no_grad():
            model.motion.log_scales.clamp_(-np.log(2), np.log(2))
        if step == 0 or (step+1) % max(1, log_every) == 0 or step+1 == iterations:
            with torch.no_grad():
                train_nmse = float(normalized_mse(model(data, train_rows), data.transients[train_rows]))
                valid_nmse = float(normalized_mse(model(data, valid_rows), data.transients[valid_rows])) if len(valid_rows) else None
            history.append({'step': step+1, 'objective': float(total.detach()),
                            'train_nmse': train_nmse, 'validation_nmse': valid_nmse})
            print(f'{model.motion.mode} {step+1}/{iterations}: train NMSE={train_nmse:.6g}, validation={valid_nmse}', flush=True)
    elapsed = time.perf_counter()-started
    return {'history': history, 'train_rows': train_rows.cpu().tolist(), 'validation_rows': valid_rows.cpu().tolist(),
            'elapsed_seconds': elapsed, 'iterations': iterations, 'fit_canonical': fit_canonical, 'loss': loss,
            'final_train_nmse': history[-1]['train_nmse'], 'final_validation_nmse': history[-1]['validation_nmse'],
            'learn_pivot': model.motion.learn_pivot, 'initial_pivot_m': initial_pivot,
            'estimated_pivot_m': model.motion.pivot.detach().cpu().tolist(),
            'scale_bound_hits': int((model.motion.log_scales.detach().abs() >= np.log(2)-1e-5).sum())}


def export_prediction(model, data, path):
    model.eval()
    with torch.no_grad():
        predictions = model(data).cpu().numpy()
        times = np.unique(data.times)
        scales, offsets = model.motion.values(times)
        state = model.canonical.gaussian_state()
    np.savez_compressed(path, predicted_transients=predictions, times=times,
                        scales=scales.cpu().numpy(), translation_m=offsets.cpu().numpy(),
                        pivot_m=model.motion.pivot.detach().cpu().numpy(),
                        canonical_centers_m=state['centers'].cpu().numpy(),
                        canonical_covariance_m2=state['covariance'].cpu().numpy())


def write_report(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
