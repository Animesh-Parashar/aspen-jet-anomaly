import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.data.corrected_preprocessing import represent, partition, leading_lhco_constituents


def vectors(pt, eta, phi):
    pt, eta, phi = map(np.asarray, (pt, eta, phi))
    return np.column_stack((pt*np.cos(phi), pt*np.sin(phi), pt*np.sinh(eta), pt*np.cosh(eta)))


class PhysicsTests(unittest.TestCase):
    def test_shared_features_and_pretruncation_normalization(self):
        c = vectors([1., 9., 2.], [0., 0., 0.], [0., 0., 0.])
        r = represent(c, 2)
        np.testing.assert_allclose(np.exp(r['constituents'][:, 3]), [9, 2], rtol=1e-6)
        np.testing.assert_allclose(np.exp(r['constituents'][:, 2]), np.array([9, 2])/12, rtol=1e-6)
        self.assertAlmostEqual(float(r['retained_pt_fraction']), 11/12, places=6)
        aspen = np.zeros((3, 11)); aspen[:, :4] = c
        np.testing.assert_array_equal(represent(aspen, 2, True)['constituents'], r['constituents'])

    def test_phi_wrap_and_padding(self):
        c = vectors([2., 1.], [0., 0.], [np.pi-.01, -np.pi+.01])
        r = represent(c, 5)
        self.assertLess(abs(r['constituents'][:2, 1]).max(), .03)
        np.testing.assert_array_equal(r['mask'], [False, False, True, True, True])
        self.assertTrue((r['constituents'][r['mask']] == 0).all())

    def test_detector_columns_and_real_negative_impact(self):
        c = np.zeros((2, 11)); c[:, :4] = vectors([2., 1.], [0., 0.], [0., 0.])
        c[0, 4:9] = [-1., .04, -2., .08, 1.]
        c[1, 4:9] = [-1., -1., -1., -1., 0.]
        r = represent(c, 3, True)
        np.testing.assert_allclose(r['detector_features'][0], [-1., -2., 1.])
        np.testing.assert_array_equal(r['detector_observed'][0], [True, True, True])
        np.testing.assert_array_equal(r['detector_observed'][1], [False, False, True])
        self.assertTrue((r['detector_features'][1:] == 0).all())

    def test_leading_jet_is_not_hardest_particle_cone(self):
        p = np.array([[100., 0., 0.], [80., 0., 3.], [80., 0., 3.05]])
        c = leading_lhco_constituents(p)
        self.assertEqual(len(c), 2)
        self.assertGreater(represent(c)['jet_kinematics'][0], 150)

    def test_invalid_inputs_are_not_fake_jets(self):
        with self.assertRaises(ValueError): represent(np.zeros((2, 4)))
        with self.assertRaises(ValueError): represent(np.full((2, 4), np.nan))
        with self.assertRaises(ValueError): leading_lhco_constituents(np.zeros((2, 3)))

    def test_split_identity_signal_exclusion_and_audit_override(self):
        ids = [(1, 2, n) for n in range(500)]
        forward = {i: partition('CMS2016', i) for i in ids}
        reverse = {i: partition('CMS2016', i) for i in ids[::-1]}
        self.assertEqual(forward, reverse)
        for i in ids:
            self.assertIn(partition('LHCO_RD_v2', i, 1), ['validation', 'test'])
            self.assertEqual(partition('CMS2016', i, development=True), 'validation')


class PipelineTests(unittest.TestCase):
    def test_aspen_event_splits_provenance_and_no_overwrite(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp); source = tmp/'raw.h5'; audit = tmp/'audit.json'; out = tmp/'out'
            with h5py.File(source, 'w') as f:
                c = np.zeros((20, 3, 11), dtype=np.float32)
                c[:, :, :4] = vectors([3., 2., 1.], [0., 0., 0.], [0., .01, .02])
                f['PFCands'] = c
                f['event_info'] = np.array([(1, 2, i//2) for i in range(20)])
                f['jet_kinematics'] = np.zeros((20, 4), dtype=np.float32)
            Path(str(source)+'.download.json').write_text(json.dumps(dict(state='verified',
                expected_bytes=source.stat().st_size, md5='synthetic-fixture')))
            audit.write_text(json.dumps({'aspen': {'sampled_indices': [0]}}))
            command = [sys.executable, '-B', str(ROOT/'scripts/preprocess/preprocess_v2.py'),
                '--domain','aspen','--input',str(source),'--out',str(out),'--audit',str(audit),'--chunk-size','3']
            subprocess.run(command, check=True, capture_output=True)
            manifest = json.loads((out/'COMPLETE.json').read_text())
            self.assertEqual(sum(manifest['output_counts'].values()),20)
            seen = {}
            for path in out.glob('*.h5'):
                with h5py.File(path) as f:
                    self.assertEqual(f['constituents'].shape[-1], 4)
                    for event in map(tuple, f['event_id'][:]):
                        self.assertEqual(seen.setdefault(event,path.stem),path.stem)
                    self.assertTrue((f['label'][:] == -1).all())
            self.assertEqual(seen[(1,2,0)],'validation')
            self.assertFalse((out/'RUNNING.json').exists())
            self.assertNotEqual(subprocess.run(command,capture_output=True).returncode,0)


if __name__ == '__main__':
    unittest.main()
