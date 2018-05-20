__author__ = 'gph82'

import unittest
import pyemma
import numpy as np
import molpx
from matplotlib import pyplot as plt
plt.switch_backend('Agg') # allow tests
try:
    from .test_bmutils import TestWithBPTIData
except:
    from test_bmutils import TestWithBPTIData
class MyVersion(unittest.TestCase):
    import molpx
    molpx.__version__

class TestSample(TestWithBPTIData):

    @classmethod
    def setUpClass(self):
        TestWithBPTIData.setUpClass()

    @classmethod
    def tearDownClass(self):
        TestWithBPTIData.tearDownClass()

    def test_just_runs_input_file_one(self):
        molpx.generate.sample(self.MD_trajectory_files[0], self.MD_topology_file, self.projected_files_npy[0])

    def test_just_runs_input_file_many(self):
        molpx.generate.sample(self.MD_trajectory_files, self.MD_topology_file, self.projected_files_npy)

    def test_just_runs_input_objects(self):
        molpx.generate.sample(self.MD_trajectories, self.MD_topology, self.Ys)

    def test_gen_and_keep_n_samples(self):
        molpx.generate.sample(self.MD_trajectories, self.MD_topology, self.Ys, n_geom_samples=5, keep_all_samples=True)

    def test_atom_selections(self):
        __, geom_smpl = molpx.generate.sample(self.MD_trajectories, self.MD_topology, self.Ys, atom_selection=np.array([2,4,6,8]))
        assert geom_smpl.n_atoms == 4

    def test_use_cl_as_input(self):
        cl = pyemma.coordinates.cluster_kmeans(self.Ys, 10)
        molpx.generate.sample(self.MD_trajectories, self.MD_topology, cl)

    def test_right_data_are_returned(self):
        # The only way to test this easily is inputting the clusterobject oneself
        cl = pyemma.coordinates.cluster_kmeans(self.Ys, 10)
        pos, geom_sampl = molpx.generate.sample(self.MD_trajectories, self.MD_topology, cl)
        output_assignments = cl.assign(self.tica.transform(self.feat.transform(geom_sampl))[:,:2])
        for ii, oa in enumerate(output_assignments):
            # Each frame of the output geometries must've been assigned to each clustercenter
            assert ii == oa
            # The output sample corresponds to that of the input (in projected space)
            assert np.allclose(cl.clustercenters[ii], pos[oa])

    def test_return_data(self):
        molpx.generate.sample(self.MD_trajectories, self.MD_topology, self.Ys, return_data=True)

class TestProjectionPath(TestWithBPTIData):

    @classmethod
    def setUpClass(self):
        TestWithBPTIData.setUpClass()

    @classmethod
    def tearDownClass(self):
        TestWithBPTIData.tearDownClass()

    def test_just_runs(self):
        molpx.generate.projection_paths(self.MD_trajectories, self.MD_topology, self.Ys)

    def test_just_runs_one_proj_idx(self):
        molpx.generate.projection_paths(self.MD_trajectories, self.MD_topology, self.Ys, proj_idxs=1)

    def _test_right_geoms_are_returned(self):
        #Each individual method of generate.projection_path has already been tested.
        # TODO

if __name__ == '__main__':
    unittest.main()
