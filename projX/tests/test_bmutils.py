__author__ = 'gph82'

import unittest
import pyemma
import os
import tempfile
import numpy as np
import shutil
from projX import bmutils
class TestBmutils(unittest.TestCase):

    def setUp(self):
        self.MD_trajectory = os.path.join(pyemma.__path__[0],'coordinates/tests/data/bpti_mini.xtc')
        self.topology = os.path.join(pyemma.__path__[0],'coordinates/tests/data/bpti_ca.pdb')
        self.tempdir = tempfile.mkdtemp('test_projX')
        self.projected_file = os.path.join(self.tempdir,'Y.npy')
        feat = pyemma.coordinates.featurizer(self.topology)
        feat.add_all()
        source = pyemma.coordinates.source(self.MD_trajectory, features=feat)
        tica = pyemma.coordinates.tica(source,lag=1, dim=2)
        self.Y = tica.get_output()[0]
        self.F = source.get_output()
        print(self.tempdir)
        np.save(self.projected_file,self.Y)
        np.savetxt(self.projected_file.replace('.npy','.dat'),self.Y)

    def tearDown(self):
        shutil.rmtree(self.tempdir)

    def test_data_from_input_npy(self):
        # Just one string
        assert np.allclose(self.Y, bmutils.data_from_input(self.projected_file)[0])
        # List of one string
        assert np.allclose(self.Y, bmutils.data_from_input([self.projected_file])[0])
        # List of two strings
        Ys = bmutils.data_from_input([self.projected_file,
                                      self.projected_file])
        assert np.all([np.allclose(self.Y, iY) for iY in Ys])

    def test_data_from_input_ascii(self):
        # Just one string
        assert np.allclose(self.Y, bmutils.data_from_input(self.projected_file.replace('.npy','.dat'))[0])
        # List of one string
        assert np.allclose(self.Y, bmutils.data_from_input([self.projected_file.replace('.npy','.dat')])[0])
        # List of two strings
        Ys = bmutils.data_from_input([self.projected_file.replace('.npy','.dat'),
                                      self.projected_file.replace('.npy','.dat')])
        assert np.all([np.allclose(self.Y, iY) for iY in Ys])

    def test_data_from_input_ndarray(self):
        # Just one ndarray
        assert np.allclose(self.Y, bmutils.data_from_input(self.Y)[0])
        # List of one ndarray
        assert np.allclose(self.Y, bmutils.data_from_input([self.Y])[0])
        # List of two ndarray
        Ys = bmutils.data_from_input([self.Y,
                                      self.Y])
        assert np.all([np.allclose(self.Y, iY) for iY in Ys])

    # Not implemented yet
    def _test_data_from_input_ndarray_ascii_npy(self):
        # List of everything
        Ys = bmutils.data_from_input([self.projected_file,
                                      self.projected_file.replace('.npy','.dat'),
                                      self.Y])
        assert np.all([np.allclose(self.Y, iY) for iY in Ys])

    def test_cluster_to_target(self):
        cl = bmutils.regspace_cluster_to_target(self.F, 100, n_try_max=10, verbose=True)
        print(cl.n_clusters)
if __name__ == '__main__':
    unittest.main()
