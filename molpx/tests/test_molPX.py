__author__ = 'gph82'

import unittest
import pyemma
import os
import tempfile
import numpy as np
import shutil
import molpx
from matplotlib import pyplot as plt
plt.switch_backend('Agg') # allow tests

class MyVersion(unittest.TestCase):
    import molpx
    molpx.__version__

class MyTestCase(unittest.TestCase):

    def setUp(self):
        self.MD_trajectory = os.path.join(pyemma.__path__[0],'coordinates/tests/data/bpti_mini.xtc')
        self.topology = os.path.join(pyemma.__path__[0],'coordinates/tests/data/bpti_ca.pdb')
        self.tempdir = tempfile.mkdtemp('test_molpx')
        self.projected_file = os.path.join(self.tempdir,'Y.npy')
        feat = pyemma.coordinates.featurizer(self.topology)
        feat.add_all()
        source = pyemma.coordinates.source(self.MD_trajectory, features=feat)
        self.tica = pyemma.coordinates.tica(source,lag=1, dim=2)
        Y = self.tica.get_output()[0]
        print(self.tempdir)
        np.save(self.projected_file,Y)

    def tearDown(self):
        shutil.rmtree(self.tempdir)

    def test_generate_paths(self):
        molpx.generate.projection_paths(self.MD_trajectory, self.topology, self.projected_file)

    def test_generate_sample(self):
        molpx.generate.sample(self.MD_trajectory, self.topology, self.projected_file)

    def test_generate_sample_atom_selections(self):
        molpx.generate.sample(self.MD_trajectory, self.topology, self.projected_file, atom_selection='symbol != H')

        __, geom_smpl = molpx.generate.sample(self.MD_trajectory, self.topology, self.projected_file, atom_selection=np.array([2,4,6,8]))
        assert geom_smpl[0].n_atoms == 4

    # Cannot get the widget to run outside the notebook because it needs an interact bar
    def test_visualize_qsample(self):
        pos, geom_smpl = molpx.generate.sample(self.MD_trajectory, self.topology, self.projected_file)
        plt.figure()
        __ = molpx.visualize.sample(pos, geom_smpl, plt.gca())

    def test_visualize_path_w_tica(self):
        paths_dict, idata = molpx.generate.projection_paths(self.MD_trajectory, self.topology, self.projected_file)
        plt.figure()
        path_type = 'min_disp'
        igeom = paths_dict[0][path_type]["geom"]
        ipath = paths_dict[0][path_type]["proj"]
        __ = molpx.visualize.sample(ipath, igeom, plt.gca(), projection=self.tica)


if __name__ == '__main__':
    unittest.main()
