"""时间化测量 NPZ 格式。旧 MAT 不会被猜测成有时间信息的动态数据。"""

from __future__ import annotations

import numpy as np
import torch


class Measurements:
    def __init__(self, arrays, device='cpu', dtype=torch.float32):
        self.device = torch.device(device)
        required = ['laser_points', 'camera_points', 'patterns', 'transients', 'times', 'dt', 't_range']
        missing = [key for key in required if key not in arrays]
        if missing:
            raise ValueError(f'Missing dynamic data fields: {missing}')
        convert = lambda value: torch.as_tensor(np.asarray(value).copy(), device=device, dtype=dtype)
        self.laser_points = convert(arrays['laser_points'])
        self.camera_points = convert(arrays['camera_points'])
        self.patterns = convert(arrays['patterns'])
        self.transients = convert(arrays['transients'])
        self.times = np.asarray(arrays['times'], dtype=float).reshape(-1)
        m, p = len(self.times), len(self.laser_points)
        self.dt = float(np.asarray(arrays['dt']).item())
        self.magnification = float(np.asarray(arrays.get('render_magnification', 1.0)).item())
        if not np.isfinite(self.magnification) or self.magnification <= 0:
            raise ValueError('render_magnification must be finite and positive.')
        self.t_range = tuple(map(int, np.asarray(arrays['t_range']).reshape(-1)))
        if len(self.t_range) != 2 or self.dt <= 0 or self.t_range[1] <= self.t_range[0]:
            raise ValueError('Invalid dt/t_range.')
        self.bins = self.t_range[1]-self.t_range[0]
        if (self.laser_points.shape != (p, 3) or self.camera_points.shape != (m, 3)
                or self.patterns.shape != (m, p) or self.transients.shape != (m, self.bins)):
            raise ValueError('Expected laser=(P,3), camera=(M,3), W=(M,P), transients=(M,K), times=(M,).')
        if (self.patterns < 0).any() or not torch.allclose(self.patterns.sum(dim=1), torch.ones(m, device=device, dtype=dtype), atol=1e-6):
            raise ValueError('Patterns must be nonnegative with each row summing to 1; put total dose in dose.')
        self.dose = convert(np.broadcast_to(arrays.get('dose', 1.0), (m,)))
        self.background = convert(np.broadcast_to(arrays.get('background', 0.0), (m, self.bins)))
        normal = np.array([0.0, 0.0, 1.0])
        self.laser_normals = convert(np.broadcast_to(arrays.get('laser_normals', normal), (p, 3)))
        self.camera_normals = convert(np.broadcast_to(arrays.get('camera_normals', normal), (m, 3)))
        if (self.dose <= 0).any() or (self.background < 0).any():
            raise ValueError('Require positive total dose and nonnegative background.')
        for value in [self.laser_points, self.camera_points, self.patterns, self.transients,
                      self.dose, self.background, self.laser_normals, self.camera_normals]:
            if not torch.isfinite(value).all():
                raise ValueError('Dynamic data contains nonfinite values.')
        if (self.laser_normals.norm(dim=-1) == 0).any() or (self.camera_normals.norm(dim=-1) == 0).any():
            raise ValueError('Relay normals must be nonzero.')
        if ('exposure_start' in arrays) != ('exposure_end' in arrays):
            raise ValueError('Provide both exposure_start and exposure_end.')
        starts = np.asarray(arrays.get('exposure_start', self.times), dtype=float).reshape(-1)
        ends = np.asarray(arrays.get('exposure_end', self.times), dtype=float).reshape(-1)
        if starts.shape != (m,) or ends.shape != (m,) or not np.isfinite(starts).all() or not np.isfinite(ends).all() or not np.isfinite(self.times).all():
            raise ValueError('Invalid measurement times.')
        if np.any(ends < starts) or np.any(self.times < starts) or np.any(self.times > ends):
            raise ValueError('Require exposure_start <= times <= exposure_end.')
        groups = {}
        for row, key in enumerate(zip(starts.tolist(), ends.tolist())):
            groups.setdefault(key, []).append(row)
        self.groups = [(a, b, rows) for (a, b), rows in groups.items()]
        self.exposure_start, self.exposure_end = starts, ends

    def __len__(self):
        return len(self.times)

    @classmethod
    def load(cls, path, device='cpu', dtype=torch.float32):
        with np.load(path, allow_pickle=False) as data:
            return cls(dict(data), device, dtype)
