"""不透光凸椭球表面模型：用于未知固定球心的条件可行性验证。

这不是对任意 Gaussian 集合添加半球 mask。表面由解析椭球定义，法向、
面积和单侧 Lambert 方向响应明确；凸性保证朝向外部墙点的表面没有自身遮挡。
不包括外部遮挡、透射、复杂 BRDF、首光子删失或光学相干。
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn

from .model import AxisScaling


def sphere_quadrature(polar=16, azimuth=32):
    """dOmega=dz*dphi：Gauss–Legendre z 积分和周期角度梯形积分。

    不是把经纬度网格直接等权求和；solid_angle 的和应为 4*pi。
    """
    if polar < 2 or azimuth < 4:
        raise ValueError('Need polar>=2 and azimuth>=4.')
    z, wz = np.polynomial.legendre.leggauss(polar)
    phi = np.arange(azimuth)*2*np.pi/azimuth
    zz, pp = np.meshgrid(z, phi, indexing='ij')
    r = np.sqrt(1-zz**2)
    directions = np.stack([r*np.cos(pp), r*np.sin(pp), zz], axis=-1).reshape(-1, 3)
    solid_angle = np.repeat(wz, azimuth)*(2*np.pi/azimuth)
    return directions, solid_angle


class OpaqueEllipsoid(nn.Module):
    """未知参考半轴、增益及一个跨时刻共享的固定几何球心。

    x_j(s)=c+D(s)n_j，D=diag(r0*a(s))。c 同时定义参考表面和膨胀中心，
    所以 dx/dc=I；这比任意固定参考点云上的未知运动中心有更强的形状先验。
    a(s) 复用 AxisScaling 的正伸缩与时间锚定，不启用独立平移。
    """

    def __init__(self, center, radii, mode='anisotropic', learn_center=True,
                 polar=16, azimuth=32, sigma_ps=60.0, magnification=1.0, background=0.2):
        super().__init__()
        self.motion = AxisScaling([0, 1], center, mode, learn_pivot=learn_center)
        radii = torch.as_tensor(radii, dtype=torch.get_default_dtype())
        if radii.shape != (3,) or not torch.isfinite(radii).all() or (radii <= 0).any():
            raise ValueError('Reference radii must be three positive finite lengths.')
        self.log_radii = nn.Parameter(radii.log())
        self.log_gain = nn.Parameter(torch.tensor(0.0))
        directions, solid_angle = sphere_quadrature(polar, azimuth)
        self.register_buffer('directions', torch.as_tensor(directions, dtype=radii.dtype))
        self.register_buffer('solid_angle', torch.as_tensor(solid_angle, dtype=radii.dtype))
        self.sigma_ps = float(sigma_ps)
        self.magnification, self.background = float(magnification), float(background)
        if self.sigma_ps <= 0 or self.magnification <= 0 or self.background < 0:
            raise ValueError('Invalid instrument parameters.')

    def surface(self, times):
        """返回位置、外法向与每个积分点的面积权重。

        n_raw=D^-T n；n_world=n_raw/|n_raw|；
        dS=det(D)*|D^-T n|*dOmega。
        这是椭球对单位球的仿射映射所诱导的精确表面 Jacobian。
        """
        scales, _ = self.motion.values(times)
        axes = self.log_radii.exp()[None, :]*scales
        points = self.motion.pivot[None, None, :]+axes[:, None, :]*self.directions[None, :, :]
        raw_normal = self.directions[None, :, :]/axes[:, None, :]
        norm = raw_normal.norm(dim=-1)
        normals = raw_normal/norm[..., None]
        areas = axes.prod(dim=-1)[:, None]*norm*self.solid_angle[None, :]
        return points, normals, areas

    def transport(self, laser, camera, times):
        """每个墙点对、每个表面积分点的强度权重和双程光程。

        ell_in=(p-x)/d1，ell_out=(d-x)/d2，都从目标指向墙面。
        [n.ell_in]+[n.ell_out]+ 对不透光凸表面实施精确单侧可见性：
        任一侧背向，该表面点的三次反射贡献严格为零。
        墙面为 z=0、法向 +z，另含两个墙面余弦与 1/(d1^2*d2^2)。
        未知 log_gain 吸收常数反射率和系统比例，不用亮度真值直接锁定球心。
        """
        points, normals, areas = self.surface(times)
        v1 = laser[:, None, :]-points
        v2 = camera[:, None, :]-points
        d1 = v1.norm(dim=-1).clamp_min(1e-10)
        d2 = v2.norm(dim=-1).clamp_min(1e-10)
        ell1, ell2 = v1/d1[..., None], v2/d2[..., None]
        target_in = (normals*ell1).sum(dim=-1).clamp_min(0)
        target_out = (normals*ell2).sum(dim=-1).clamp_min(0)
        wall_in = (-ell1[..., 2]).clamp_min(0)
        wall_out = (-ell2[..., 2]).clamp_min(0)
        weight = areas*target_in*target_out*wall_in*wall_out/(d1.square()*d2.square())
        return weight, d1+d2

    def forward(self, laser, camera, times, edges_ps, chunk=256):
        """物理表面积分 -> Gaussian IRF 的 bin 积分 -> 期望计数。

        h_rk=sum_j w_rj * integral_bin Normal(t;L_rj/c,sigma_IRF)dt。
        没有输入第二组已重建点云，也不使用最近邻、Chamfer 或 ICP 损失。
        """
        weight, length = self.transport(laser, camera, times)
        mean_ps = length/3e-4
        result = weight.new_zeros((len(laser), len(edges_ps)-1))
        for start in range(0, weight.shape[1], chunk):
            stop = start+chunk
            z = (edges_ps[None, None, :]-mean_ps[:, start:stop, None])/self.sigma_ps
            # 稳定尾部差，保留每个时间箱的概率质量而非 bin 中心采样。
            lo, hi = z[..., :-1], z[..., 1:]
            mass = torch.where(lo > 0,
                               0.5*(torch.erfc(lo/np.sqrt(2))-torch.erfc(hi/np.sqrt(2))),
                               0.5*(torch.erf(hi/np.sqrt(2))-torch.erf(lo/np.sqrt(2))))
            result = result+(weight[:, start:stop, None]*mass).sum(dim=1)
        return self.background+self.magnification*self.log_gain.exp()*result
