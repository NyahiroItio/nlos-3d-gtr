from pathlib import Path
import unittest

import numpy as np
from threadpoolctl import threadpool_limits

from transient_transform.original_lct import OriginalLCT, SceneSettings
from transient_transform.temporal import fit_weights
from transient_transform.upstream import SOURCE_SHA256


FIXTURE = Path(__file__).parent/'fixtures'/'author_matlab_reference.npz'


class AuthorEngineeringParity(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = threadpool_limits(2)
        cls.reference = np.load(FIXTURE, allow_pickle=False)

    @classmethod
    def tearDownClass(cls):
        cls.reference.close()
        cls.threads.restore_original_limits()

    def field(self, label, name):
        return self.reference[f'{label}_{name}']

    def model(self, label):
        return OriginalLCT(int(self.field(label, 'N')), int(self.field(label, 'M')),
                           float(self.field(label, 'width')), float(self.field(label, 'range')),
                           bool(self.field(label, 'isdiffuse')), float(self.field(label, 'snr')))

    def test_fixture_identifies_original_source(self):
        self.assertEqual(str(self.reference['source_sha256']), SOURCE_SHA256)

    def test_original_psf_normalization_grid_and_phase(self):
        for label in ('retro', 'diffuse'):
            model = self.model(label)
            np.testing.assert_allclose(model.psf, self.field(label, 'psf'), rtol=1e-13, atol=1e-15)
            np.testing.assert_allclose(model.inverse_filter,
                                       self.field(label, 'inverse_filter')[..., :model.n+1],
                                       rtol=1e-12, atol=1e-14)

    def test_original_resampling_and_transpose_weights(self):
        model = self.model('retro')
        np.testing.assert_allclose(model.mtx.toarray(), self.field('retro', 'mtx'), atol=1e-15)
        np.testing.assert_allclose(model.mtxi.toarray(), self.field('retro', 'mtxi'), atol=1e-15)

    def test_every_linear_stage_matches_matlab_for_both_radiometric_modes(self):
        for label in ('retro', 'diffuse'):
            model = self.model(label)
            _, stages = model(self.field(label, 'prepared_data'), return_stages=True)
            for name, actual in stages.items():
                with self.subTest(mode=label, stage=name):
                    np.testing.assert_allclose(actual, self.field(label, name), rtol=1e-10, atol=1e-12)

    def test_author_positivity_crop_flip_and_coordinates(self):
        for label in ('retro', 'diffuse'):
            model = self.model(label)
            settings = SceneSettings(0, 'fixture', model.width_m, model.n, model.m,
                                     float(self.field(label, 'bin_resolution')), model.range_m,
                                     model.diffuse, model.snr, int(self.field(label, 'z_trim')),
                                     int(self.field(label, 'z_offset')))
            signed = model(self.field(label, 'prepared_data'))
            np.testing.assert_allclose(np.maximum(signed, 0), self.field(label, 'clipped_volume'), atol=1e-12)
            displayed, lateral, depth = model.display_volume(signed, settings)
            np.testing.assert_allclose(displayed, self.field(label, 'display_volume'), atol=1e-12)
            np.testing.assert_allclose(lateral, self.field(label, 'tic_x'), atol=1e-15)
            np.testing.assert_allclose(depth, self.field(label, 'tic_z'), atol=1e-15)

    def test_signed_core_and_slow_time_fit_are_linear(self):
        model = self.model('retro')
        center = self.field('retro', 'prepared_data')
        slope = np.roll(center, 1, axis=0)-center
        times = np.array([-.1, 0., .1])
        frames = center+times[:, None, None, None]*slope
        weights = fit_weights(times)
        fit_first = (weights @ frames.reshape(3, -1)).reshape(2, *center.shape)
        actual = np.stack([model(frame) for frame in fit_first])
        inverse_first = np.stack([model(frame) for frame in frames])
        expected = (weights @ inverse_first.reshape(3, -1)).reshape(actual.shape)
        np.testing.assert_allclose(actual, expected, rtol=1e-10, atol=1e-12)
        np.testing.assert_allclose(model(-center), -model(center), atol=1e-12)
