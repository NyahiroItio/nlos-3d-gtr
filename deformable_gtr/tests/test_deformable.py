"""小规模物理/数值回归；运行 python -m unittest discover -s deformable_gtr/tests -v。"""

import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from deformable_gtr import AxisScaling, DeformableGTR, Measurements, canonical_from_arrays
from deformable_gtr.model import canonical_config
from deformable_gtr.training import load_canonical, load_dynamic, save_dynamic


def scene():
    rng = np.random.default_rng(12)
    return canonical_from_arrays(
        [[0.02, -0.03, 0.72], [-0.08, 0.04, 0.67], [0.09, 0.02, 0.78]],
        [[0.02, 0.01, 0.006]]*3, rng.normal(size=(3, 4)), [0.3, 0.4, 0.35], dtype=torch.float64,
    )


def arrays(time=0.0):
    return dict(laser_points=np.array([[-0.25, 0, 0], [0.25, 0.15, 0]]),
                camera_points=np.zeros((2, 3)), patterns=np.eye(2),
                transients=np.zeros((2, 180)), times=np.full(2, time),
                dt=32e-12, t_range=[80, 260])


def motion(mode='anisotropic'):
    return AxisScaling([0, 1], [0, 0, 0.72], mode).double()


class DeformableTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_identity_matches_released_static_equations(self):
        # 非平凡 quaternion、各向异性 Gaussian、非共焦墙点，不能只测球形特例。
        model = scene()
        data = Measurements(arrays(), dtype=torch.float64)
        model.ensure_timebin_buffer(data.t_range, 'cpu', torch.float64)
        direct = model(data.camera_points, data.dt, data.t_range, data.laser_points, temporal_backend='torch')
        state = motion().push_forward(model.gaussian_state(), 0.0)
        dynamic = model(data.camera_points, data.dt, data.t_range, data.laser_points,
                        gaussian_state=state, temporal_backend='torch')
        torch.testing.assert_close(dynamic, direct, rtol=2e-12, atol=1e-20)

    def test_uniform_is_exact_anisotropic_special_case(self):
        canonical = scene()
        iso, aniso = motion('isotropic'), motion('anisotropic')
        with torch.no_grad():
            iso.log_scales.fill_(np.log(1.25))
            aniso.log_scales.fill_(np.log(1.25))
        for time in [0, 0.2, 0.7, 1]:
            left, right = iso.push_forward(canonical.gaussian_state(), time), aniso.push_forward(canonical.gaussian_state(), time)
            for name in left:
                torch.testing.assert_close(left[name], right[name], rtol=1e-14, atol=1e-18)

    def test_covariance_and_precision_affine_pushforward(self):
        canonical = scene().gaussian_state()
        transform = motion()
        stretches = torch.tensor([1.1, 1.2, 1.3], dtype=torch.float64)
        with torch.no_grad():
            transform.log_scales.copy_(stretches.log()[None])
        changed = transform.push_forward(canonical, 1.0)
        matrix = torch.diag(stretches)
        torch.testing.assert_close(changed['covariance'], matrix@canonical['covariance']@matrix.T)
        inverse = matrix.inverse()
        torch.testing.assert_close(changed['precision'], inverse.T@canonical['precision']@inverse)
        self.assertTrue((torch.linalg.eigvalsh(changed['covariance']) > 0).all())
        torch.testing.assert_close(changed['sqrt_det'], stretches.prod()*canonical['sqrt_det'])
        translated = AxisScaling([0, 1], [0, 0, 0.72], translation=True).double()
        with torch.no_grad():
            translated.offsets.copy_(torch.tensor([[0.01, -0.02, 0.03]], dtype=torch.float64))
        torch.testing.assert_close(translated.push_forward(canonical, 1)['centers'],
                                   canonical['centers']+translated.offsets[0])
        torch.testing.assert_close(translated.push_forward(canonical, 0)['centers'], canonical['centers'])

    def test_motion_gradient_through_actual_renderer(self):
        data = Measurements(arrays(1.0), dtype=torch.float64)
        model = DeformableGTR(scene(), motion(), data.dt, data.t_range, magnification=1e6, backend='torch')
        with torch.no_grad():
            model.motion.log_scales.copy_(torch.tensor([[0.05, 0.10, 0.15]], dtype=torch.float64))
        weights = torch.linspace(0.2, 1.0, data.bins, dtype=torch.float64)
        value = (model(data)*weights).sum()
        analytical = torch.autograd.grad(value, model.motion.log_scales)[0].clone()
        finite = torch.zeros_like(analytical)
        delta = 1e-5
        for axis in range(3):
            with torch.no_grad():
                model.motion.log_scales[0, axis] += delta
                plus = (model(data)*weights).sum()
                model.motion.log_scales[0, axis] -= 2*delta
                minus = (model(data)*weights).sum()
                model.motion.log_scales[0, axis] += delta
                finite[0, axis] = (plus-minus)/(2*delta)
        torch.testing.assert_close(analytical, finite, rtol=3e-5, atol=1e-10)

    def test_exposure_integral_and_static_limit(self):
        values = arrays(0.5)
        values.update(exposure_start=[0.45]*2, exposure_end=[0.55]*2)
        data = Measurements(values, dtype=torch.float64)
        model = DeformableGTR(scene(), motion(), data.dt, data.t_range, backend='torch')
        with torch.no_grad():
            model.motion.log_scales.copy_(torch.tensor([[0.1, 0.2, 0.3]], dtype=torch.float64))
        model.quadrature = 3
        integral = model(data)
        model.quadrature = 15
        reference = model(data)
        self.assertLess(float((integral-reference).norm()/reference.norm()), 1e-5)
        model.quadrature = 1
        midpoint = model(data)
        self.assertGreater(float((midpoint-reference).norm()/reference.norm()), 1e-5)
        static = DeformableGTR(scene(), motion('static'), data.dt, data.t_range, backend='torch')
        static.quadrature = 1
        first = static(data)
        static.quadrature = 7
        torch.testing.assert_close(static(data), first, rtol=1e-12, atol=1e-20)

    def test_fixed_intensity_pattern_operator(self):
        model = DeformableGTR(scene(), motion(), 32e-12, [80, 260], backend='torch')
        individual = model(Measurements(arrays(), dtype=torch.float64))
        values = arrays()
        values.update(patterns=np.array([[0.3, 0.7], [0.6, 0.4]]))
        mixed = model(Measurements(values, dtype=torch.float64))
        torch.testing.assert_close(mixed, torch.tensor(values['patterns'])@individual)

    def test_checkpoint_roundtrip(self):
        data = Measurements(arrays(0.5), dtype=torch.float64)
        model = DeformableGTR(scene(), motion(), data.dt, data.t_range, backend='torch')
        with torch.no_grad():
            model.motion.log_scales.fill_(0.1)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'dynamic.pth'
            save_dynamic(model, path)
            recovered = load_dynamic(path, backend='torch')
            torch.testing.assert_close(recovered(data), model(data), rtol=1e-12, atol=1e-20)

    def test_original_checkpoint_initialization(self):
        original = scene()
        config = canonical_config(original)
        config['num_gaussians_init'] = config['num_gaussians']
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'original.pth'
            torch.save({'state_dict': original.float().state_dict(), 'config': config,
                        'num_gaussians': 3}, path)
            recovered = load_canonical(path)
            torch.testing.assert_close(recovered.get_gs_centers(), original.get_gs_centers())

    def test_invalid_data_and_time_rejected(self):
        values = arrays()
        values['patterns'][0, 0] = -1
        with self.assertRaises(ValueError):
            Measurements(values)
        with self.assertRaises(ValueError):
            motion().values(2.0)
        values = arrays()
        del values['times']
        with self.assertRaises(ValueError):
            Measurements(values)


if __name__ == '__main__':
    unittest.main()
