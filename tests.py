__author__ = 'gph82'

import unittest
import pyemma
import os
import tempfile
import numpy as np
import shutil
import projX
class MyTestCase(unittest.TestCase):

    def setUp(self):
        self.MD_trajectory = os.path.join(pyemma.__path__[0],'coordinates/tests/data/bpti_mini.xtc')
        self.topology = os.path.join(pyemma.__path__[0],'coordinates/tests/data/bpti_ca.pdb')
        self.tempdir = tempfile.mkdtemp('test_projX')
        self.projected_file = os.path.join(self.tempdir,'Y.npy')
        feat = pyemma.coordinates.featurizer(self.topology)
        feat.add_all()
        source = pyemma.coordinates.source(self.MD_trajectory, features=feat)
        tica = pyemma.coordinates.tica(source,lag=1, dim=2)
        Y =    tica.get_output()[0]
        print(self.tempdir)
        np.save(self.projected_file,Y)

    def tearDown(self):
        shutil.rmtree(self.tempdir)

    def test_generate_paths(self):
        projX.generate_paths(self.MD_trajectory, self.topology, self.projected_file)

if __name__ == '__main__':
    unittest.main()
