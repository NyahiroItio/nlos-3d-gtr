"""共享参考高斯 + 三轴伸缩 + 原始 3D-GTR 瞬态渲染。"""

from __future__ import annotations

import torch
from torch import nn

from nlos_3dgs import NLOS_3DGS


def canonical_config(model):
    names = ['x_min', 'x_max', 'y_min', 'y_max', 'z_min', 'z_max',
             'scale_min', 'scale_max', 'activation_mode', 'scale_activation_mode',
             'albedo_activation_mode', 'anisotropy_threshold', 'isplanar',
             'isconfocal', 'isretroreflective']
    result = {name: getattr(model, name) for name in names}
    result.update(num_gaussians=int(model.xy_centers_raw.shape[0]),
                  scale_init=(model.scale_min+model.scale_max)/2, albedo_init=0.3)
    return result


def canonical_from_config(config, device='cpu', dtype=torch.float32):
    model = NLOS_3DGS(device=device)
    model.initialize(**config)
    return model.to(device=device, dtype=dtype)


def canonical_from_arrays(centers, scales, quats, albedos, device='cpu', dtype=torch.float32):
    """从有物理单位的初始化构造真正的 NLOS_3DGS；也可使用原始 checkpoint。"""
    centers = torch.as_tensor(centers, device=device, dtype=dtype)
    scales = torch.as_tensor(scales, device=device, dtype=dtype)
    quats = torch.as_tensor(quats, device=device, dtype=dtype)
    albedos = torch.as_tensor(albedos, device=device, dtype=dtype).reshape(-1, 1)
    n = centers.shape[0]
    if centers.shape != (n, 3) or scales.shape != (n, 3) or quats.shape != (n, 4) or albedos.shape != (n, 1):
        raise ValueError('Expected centers/scales=(N,3), quats=(N,4), albedos=(N,1).')
    if not all(torch.isfinite(x).all() for x in (centers, scales, quats, albedos)):
        raise ValueError('Canonical arrays must be finite.')
    if (scales <= 0).any() or (albedos <= 0).any() or (albedos >= 1).any() or (quats.norm(dim=1) == 0).any():
        raise ValueError('Require scales>0, 0<albedo<1 and nonzero quaternions.')
    lo, hi = centers.amin(dim=0)-0.2, centers.amax(dim=0)+0.2
    config = dict(num_gaussians=n, x_min=float(lo[0]), x_max=float(hi[0]),
                  y_min=float(lo[1]), y_max=float(hi[1]), z_min=float(lo[2]), z_max=float(hi[2]),
                  scale_min=float(scales.min()/4), scale_max=float(scales.max()*4),
                  scale_init=float(scales.mean()), albedo_init=0.3,
                  isplanar=True, isconfocal=False, isretroreflective=False,
                  activation_mode='tanh', scale_activation_mode='tanh',
                  albedo_activation_mode='tanh', anisotropy_threshold=0.0)
    model = canonical_from_config(config, device, dtype)
    with torch.no_grad():
        model.xy_centers_raw.copy_(model.xy_to_raw(centers[:, :2]))
        model.z_centers_raw.copy_(model.z_to_raw(centers[:, 2:]))
        model.raw_scaling.copy_(model.scale_to_raw(scales))
        model.quats.copy_(quats)
        model.albedo.copy_(model.albedo_to_raw(albedos))
    return model


class AxisScaling(nn.Module):
    """A(s)=diag(exp(lx),exp(ly),exp(lz))，第一时间结点严格固定为单位变换。

    isotropic: 每个时间结点共享一个 log_scale，严格包含 aI。
    anisotropic: 每个结点有三个独立 log_scale。
    static: 所有时刻 A=I。允许固定/可学习的附加平移，默认关闭。
    时间结点间线性插值 log_scale；不在未知区间盲目外推。
    """

    def __init__(self, knot_times, pivot, mode='anisotropic', translation=False):
        super().__init__()
        times = torch.as_tensor(knot_times, dtype=torch.get_default_dtype())
        pivot = torch.as_tensor(pivot, dtype=times.dtype)
        if mode not in ('static', 'isotropic', 'anisotropic'):
            raise ValueError(mode)
        if times.ndim != 1 or len(times) < 2 or not torch.all(times[1:] > times[:-1]):
            raise ValueError('At least two strictly increasing knot times are required.')
        if pivot.shape != (3,) or not torch.isfinite(times).all() or not torch.isfinite(pivot).all():
            raise ValueError('Invalid pivot or times.')
        self.register_buffer('knot_times', times)
        self.register_buffer('pivot', pivot)
        self.mode, self.translation = mode, bool(translation)
        channels = 1 if mode == 'isotropic' else 3
        self.log_scales = nn.Parameter(torch.zeros(len(times)-1, channels), requires_grad=mode != 'static')
        self.offsets = nn.Parameter(torch.zeros(len(times)-1, 3), requires_grad=translation and mode != 'static')

    def values(self, time):
        time = torch.as_tensor(time, device=self.knot_times.device, dtype=self.knot_times.dtype)
        if not torch.isfinite(time).all() or (time < self.knot_times[0]-1e-7).any() or (time > self.knot_times[-1]+1e-7).any():
            raise ValueError('Measurement/exposure time is outside the motion knot range.')
        if self.mode == 'static':
            return time.new_ones((*time.shape, 3)), time.new_zeros((*time.shape, 3))
        index = torch.searchsorted(self.knot_times, time.contiguous(), right=True)-1
        index = index.clamp(0, len(self.knot_times)-2)
        alpha = ((time-self.knot_times[index])/(self.knot_times[index+1]-self.knot_times[index])).unsqueeze(-1)
        logs = torch.cat([self.log_scales.new_zeros((1, self.log_scales.shape[1])), self.log_scales])
        log_value = (1-alpha)*logs[index]+alpha*logs[index+1]
        scales = log_value.exp().expand(*time.shape, 3)
        offsets = torch.cat([self.offsets.new_zeros((1, 3)), self.offsets])
        offset = (1-alpha)*offsets[index]+alpha*offsets[index+1] if self.translation else torch.zeros_like(scales)
        return scales, offset

    def push_forward(self, state, time):
        """精确仿射推送，不只移动中心：

        x=c+A(x0-c)+b；Sigma=A Sigma0 A^T；P=A^-T P0 A^-1；
        sqrt(det Sigma)=det(A)*sqrt(det Sigma0)。
        P 与体积因子继承基线的数值约定，单位变换严格回到基线。
        反射率暂按同一 material Gaussian 的属性保持；不额外假定质量守恒。
        """
        scales, offset = self.values(time)
        if scales.ndim != 1:
            raise ValueError('push_forward expects a scalar state time.')
        outer = scales[:, None]*scales[None, :]
        return {
            'centers': self.pivot+scales*(state['centers']-self.pivot)+offset,
            'albedos': state['albedos'],
            'covariance': state['covariance']*outer,
            'precision': state['precision']/outer,
            'sqrt_det': state['sqrt_det']*scales.prod(),
        }

    def smoothness(self):
        # 非等间隔时间上的 log_scale 斜率变化；参考结点也参加约束。
        logs = torch.cat([self.log_scales.new_zeros((1, self.log_scales.shape[1])), self.log_scales])
        velocity = (logs[1:]-logs[:-1])/(self.knot_times[1:]-self.knot_times[:-1])[:, None]
        return (velocity[1:]-velocity[:-1]).square().mean() if len(velocity) > 1 else logs.sum()*0


class DeformableGTR(nn.Module):
    """按真实采集区间积分的运动补偿前向算子，复用原始 3D-GTR 渲染。

    每行测量有一个相机墙点和非负照明权重，支持 one-hot 扫描及固定强度编码。
    对曝光内运动使用归一化 Gauss–Legendre 积分；dose 是整次曝光的总剂量。
    此处不做相干场叠加、不做事件滤波、不自动学习 SLM 相位。
    """

    def __init__(self, canonical, motion, dt, t_range, magnification=1.0, quadrature=3, backend='auto'):
        super().__init__()
        self.canonical, self.motion = canonical, motion
        self.dt, self.t_range = float(dt), tuple(map(int, t_range))
        self.magnification, self.quadrature, self.backend = float(magnification), int(quadrature), backend
        if self.dt <= 0 or self.t_range[1] <= self.t_range[0] or self.quadrature < 1 or self.magnification <= 0:
            raise ValueError('Invalid rendering parameters.')

    def forward(self, data, rows=None):
        import numpy as np
        if data.dt != self.dt or tuple(data.t_range) != self.t_range:
            raise ValueError('Measurement and renderer dt/t_range must match.')
        if rows is None:
            rows = torch.arange(len(data), device=data.device)
        output = data.camera_points.new_zeros((len(rows), data.bins))
        state0 = self.canonical.gaussian_state()
        if state0['centers'].dtype != output.dtype or state0['centers'].device != output.device:
            raise ValueError('Measurements and canonical model must use the same dtype and device.')
        self.canonical.ensure_timebin_buffer(self.t_range, data.device, state0['centers'].dtype)
        nodes, weights = np.polynomial.legendre.leggauss(self.quadrature)
        # 按相同曝光区间分组，避免每条测量重复变形整个物体。
        selected = rows.detach().cpu().tolist()
        lookup = {int(row): position for position, row in enumerate(selected)}
        for start, end, group_rows in data.groups:
            keep = [row for row in group_rows if row in lookup]
            if not keep:
                continue
            group = torch.tensor(keep, device=data.device, dtype=torch.long)
            target_rows = torch.tensor([lookup[row] for row in keep], device=data.device, dtype=torch.long)
            patterns = data.patterns[group]
            pair = (patterns > 0).nonzero(as_tuple=False)
            local_row, laser_index = pair[:, 0], pair[:, 1]
            camera = data.camera_points[group[local_row]]
            laser = data.laser_points[laser_index]
            if self.canonical.isconfocal and not torch.allclose(camera, laser, atol=1e-7, rtol=1e-7):
                raise ValueError('Confocal checkpoints require coincident camera/laser points for every active pair.')
            accumulated = output.new_zeros((len(keep), data.bins))
            sample_nodes = [(0.0, 2.0)] if start == end else zip(nodes, weights)
            for node, weight in sample_nodes:
                time = (start+end)/2+(end-start)*float(node)/2
                state = self.motion.push_forward(state0, time)
                transient = self.canonical(
                    camera, self.dt, self.t_range, laser,
                    laser_normals=data.laser_normals[laser_index],
                    camera_normals=data.camera_normals[group[local_row]],
                    gaussian_state=state, temporal_backend=self.backend,
                )
                weighted = transient*patterns[local_row, laser_index, None]
                integrated = accumulated.new_zeros(accumulated.shape).index_add(0, local_row, weighted)
                accumulated = accumulated+float(weight)/2*integrated
            output = output.index_add(0, target_rows, accumulated)
        # 总剂量已包含脉冲数/曝光长度；不能再乘一次 duration。
        return self.magnification*data.dose[rows, None]*output+data.background[rows]

    def specification(self):
        return {
            'canonical': canonical_config(self.canonical),
            'motion': {'knot_times': self.motion.knot_times.detach().cpu().tolist(),
                       'pivot': self.motion.pivot.detach().cpu().tolist(), 'mode': self.motion.mode,
                       'translation': self.motion.translation},
            'render': {'dt': self.dt, 't_range': self.t_range, 'magnification': self.magnification,
                       'quadrature': self.quadrature, 'backend': self.backend},
        }
