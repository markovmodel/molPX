__author__ = 'gph82'

import unittest
import pyemma
import os
import tempfile
import numpy as np
import shutil
import molpx
from matplotlib import pyplot as plt
class MyTestCase(unittest.TestCase):

    def setUp(self):
        self.MD_trajectory = os.path.join(pyemma.__path__[0],'coordinates/tests/data/bpti_mini.xtc')
        self.topology = os.path.join(pyemma.__path__[0],'coordinates/tests/data/bpti_ca.pdb')
        self.tempdir = tempfile.mkdtemp('test_molpx')
        self.projected_file = os.path.join(self.tempdir,'Y.npy')
        feat = pyemma.coordinates.featurizer(self.topology)
        feat.add_all()
        source = pyemma.coordinates.source(self.MD_trajectory, features=feat)
        tica = pyemma.coordinates.tica(source,lag=1, dim=2)
        Y = tica.get_output()[0]
        print(self.tempdir)
        np.save(self.projected_file,Y)

    def tearDown(self):
        shutil.rmtree(self.tempdir)

    def test_generate_paths(self):
        molpx.generate.projection_paths(self.MD_trajectory, self.topology, self.projected_file)

    def test_generate_sample(self):
        molpx.generate.sample(self.MD_trajectory, self.topology, self.projected_file)

    # Cannot get the widget to run outside the notebook because it needs an interact bar
    def _test_visualize_qsample(self):
        pos, geom_smpl = molpx.generate.sample(self.MD_trajectory, self.topology, self.projected_file)
        plt.figure()
        iwd = molpx.visualize.sample(pos, geom_smpl, plt.gca())

if __name__ == '__main__':
    unittest.main()
