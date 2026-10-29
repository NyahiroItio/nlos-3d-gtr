"""解析表面面积、可见性和未知球心梯度的独立检查。"""

import unittest

import numpy as np
import torch

from deformable_gtr.opaque_ellipsoid import OpaqueEllipsoid, sphere_quadrature


class OpaqueTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_sphere_area_and_surface_normals(self):
        directions, omega = sphere_quadrature(8, 16)
        self.assertAlmostEqual(float(omega.sum()), 4*np.pi, places=12)
        np.testing.assert_allclose(np.linalg.norm(directions, axis=1), 1, rtol=1e-14)
        model = OpaqueEllipsoid([0, 0, 0.72], [0.1]*3, polar=8, azimuth=16).double()
        points, normals, areas = model.surface(torch.tensor([0.0], dtype=torch.float64))
        # 构造时默认 float32，随后 double 不会恢复初始参数/权重被舍入的位数。
        self.assertTrue(np.isclose(float(areas.sum()), 4*np.pi*0.1**2, rtol=2e-7, atol=0))
        torch.testing.assert_close(normals[0], model.directions, rtol=1e-7, atol=1e-7)
        self.assertTrue((areas > 0).all())

    def test_backside_zero_and_pair_dependent_visibility(self):
        model = OpaqueEllipsoid([0, 0, 0.72], [0.12, 0.10, 0.14], polar=12, azimuth=24).double()
        with torch.no_grad():
            model.motion.log_scales.fill_(0.2)
        laser = torch.tensor([[-0.7, 0, 0], [0.7, 0, 0]], dtype=torch.float64)
        camera = torch.zeros_like(laser)
        times = torch.ones(2, dtype=torch.float64)
        weights, _ = model.transport(laser, camera, times)
        points, normals, _ = model.surface(times)
        self.assertEqual(float(weights[normals[..., 2] > 0].sum()), 0.0)
        incoming = ((laser[:, None]-points)*normals).sum(dim=-1)
        outgoing = ((camera[:, None]-points)*normals).sum(dim=-1)
        self.assertTrue(torch.equal(weights > 0, (incoming > 0) & (outgoing > 0)))
        self.assertTrue(torch.any((weights[0] > 0) != (weights[1] > 0)))

    def test_center_derivative_in_measurement_domain(self):
        model = OpaqueEllipsoid([0.015, -0.01, 0.72], [0.12, 0.10, 0.14],
                                polar=6, azimuth=12, sigma_ps=80, magnification=1000).double()
        laser = torch.tensor([[-0.3, 0.1, 0], [0.4, -0.2, 0]], dtype=torch.float64)
        camera = torch.zeros_like(laser)
        times = torch.tensor([0.0, 0.7], dtype=torch.float64)
        edges = torch.arange(80, 257, dtype=torch.float64)*32
        weights = torch.linspace(0.2, 1.0, len(edges)-1, dtype=torch.float64)
        value = (model(laser, camera, times, edges)*weights).sum()
        gradient = torch.autograd.grad(value, model.motion.pivot)[0]
        finite = torch.zeros_like(gradient)
        delta = 1e-6
        for axis in range(3):
            with torch.no_grad():
                model.motion.pivot[axis] += delta
                plus = (model(laser, camera, times, edges)*weights).sum()
                model.motion.pivot[axis] -= 2*delta
                minus = (model(laser, camera, times, edges)*weights).sum()
                model.motion.pivot[axis] += delta
                finite[axis] = (plus-minus)/(2*delta)
        torch.testing.assert_close(gradient, finite, rtol=2e-5, atol=1e-7)


if __name__ == '__main__':
    unittest.main()
