__author__ = 'gph82'

import unittest
import pyemma
import os
import tempfile
import numpy as np
import shutil
from molpx import bmutils
import mdtraj as md

class TestBmutils(unittest.TestCase):

    def setUp(self):
        self.MD_trajectory = os.path.join(pyemma.__path__[0],'coordinates/tests/data/bpti_mini.xtc')
        self.topology = os.path.join(pyemma.__path__[0],'coordinates/tests/data/bpti_ca.pdb')
        self.tempdir = tempfile.mkdtemp('test_molpx')
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

        self.data_for_cluster = [np.array([[1, 3],
                                           [2, 3],
                                           [3, 3]]),
                                 np.array([[17,1]]),
                                 np.array([[11, 2],
                                           [12, 2],
                                           [13, 3]]),
                                 np.array([[1,  3],
                                           [11, 2],
                                           [2,  3],
                                           [12, 2],
                                           [3,  3],
                                           [13, 2]])
                                 ]

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
        n_target = 15
        data = [np.random.randn(100, 1), np.random.randn(100,1)+10]
        cl = bmutils.regspace_cluster_to_target(data, n_target, n_try_max=10, delta=0)
        assert cl.n_clusters >= n_target-1 and cl.n_clusters <= n_target+1

    def test_catalogues(self):
        cl = bmutils.regspace_cluster_to_target(self.data_for_cluster, 3, n_try_max=10, delta=0)
        #print(cl.clustercenters)
        cat_idxs, cat_cont = bmutils.catalogues(cl)

        # This test is extra, since this is a pure pyemma functions
        assert np.allclose(cat_idxs[0], [[0,0], [0,1], [0,2],
                                         [3,0], [3,2], [3,4]
                                         ])
        assert np.allclose(cat_idxs[1], [[1,0]])
        assert np.allclose(cat_idxs[2], [[2,0], [2,1], [2,2],
                                         [3,1], [3,3], [3,5],
                                         ])

        # Continous catalogue
        assert np.allclose(cat_cont[0], [[1, 3],
                                         [2, 3],
                                         [3, 3],
                                         [1, 3],
                                         [2, 3],
                                         [3, 3],])
        assert np.allclose(cat_cont[1], [[17,1]])
        assert np.allclose(cat_cont[2], [[11, 2],
                                         [12, 2],
                                         [13, 3],
                                         [11, 2],
                                         [12, 2],
                                         [13, 2]])

    def test_catalogues_sort_by_zero(self):
        cl = bmutils.regspace_cluster_to_target(self.data_for_cluster, 3, n_try_max=10, delta=0)
        cat_idxs, cat_cont = bmutils.catalogues(cl, sort_by=0)

        # This test is extra, since this is a pure pyemma functions
        assert np.allclose(cat_idxs[0], [[0, 0], [0, 1], [0, 2],
                                         [3, 0], [3, 2], [3, 4]
                                         ])
        assert np.allclose(cat_idxs[2], [[1, 0]])                     # notice indices inverted
        assert np.allclose(cat_idxs[1], [[2, 0], [2, 1], [2, 2],      # in these two lines
                                         [3, 1], [3, 3], [3, 5],
                                         ])

        # Notice that indices 2,1 have inverted wrt to previous test case
        # Continous catalogue
        assert np.allclose(cat_cont[0], [[1, 3],
                                         [2, 3],
                                         [3, 3],
                                         [1, 3],
                                         [2, 3],
                                         [3, 3],])
        assert np.allclose(cat_cont[2], [[17,1]])                   # notice indices inverted
        assert np.allclose(cat_cont[1], [[11, 2],                   # in these two lines
                                         [12, 2],
                                         [13, 3],
                                         [11, 2],
                                         [12, 2],
                                         [13, 2]])

    def test_catalogues_sort_by_other_than_zero(self):
        cl = bmutils.regspace_cluster_to_target(self.data_for_cluster, 3, n_try_max=10, delta=0)
        cat_idxs, cat_cont = bmutils.catalogues(cl, sort_by=1)
        # This test is extra, since this is a pure pyemma functions
        assert np.allclose(cat_idxs[0], [[1,0]])
        assert np.allclose(cat_idxs[1], [[2, 0],
                                         [2, 1],
                                         [2, 2],
                                         [3, 1],
                                         [3, 3],
                                         [3, 5]
                                         ])
        assert np.allclose(cat_idxs[2], [[0,0],
                                         [0,1],
                                         [0,2],
                                         [3,0],
                                         [3,2],
                                         [3,4]])
        # Continous catalogue (all indices inverted)
        assert np.allclose(cat_cont[2], [[1, 3],
                                         [2, 3],
                                         [3, 3],
                                         [1, 3],
                                         [2, 3],
                                         [3, 3], ])
        assert np.allclose(cat_cont[0], [[17, 1]])
        assert np.allclose(cat_cont[1], [[11, 2],
                                         [12, 2],
                                         [13, 3],
                                         [11, 2],
                                         [12, 2],
                                         [13, 2]])

    def test_find_centers_GMM(self):
        # Two 1D gaussians, one centered at 5 the other at 10

        data = [np.random.randn(100, 1)+5, np.random.randn(100,1)+10]
        data = np.vstack(data)
        grid = np.arange(0,20)
        # This is the super easy test (array is ordered, index and value are the same)
        idxs, igmm = bmutils.find_centers_gmm(data, grid)
        assert np.allclose(np.sort(idxs), [5,10])

        # Works also with arrays that do not necessarily contain the centers or are unsorted
        grid = [-25, 10, 1, 4, -100, 10]
        idxs, igmm = bmutils.find_centers_gmm(data, grid)
        assert np.allclose(np.sort(idxs), [1, 3])

class TestGetGoodStartingPoint(unittest.TestCase):

    def setUp(self):
        # The setup creates the typical, "geometries-sampled along cluster-scenario"
        n_geom_samples = 20
        traj = md.load(os.path.join(pyemma.__path__[0],'coordinates/tests/data/bpti_ca.pdb'))
        traj = traj.atom_slice([0,1,3,4]) # create a trajectory with four atoms
        # Create a fake bi-modal trajectory with a compact and an open structure
        ixyz = np.array([[0., 0., 0.],
                         [1., 0., 0.],
                         [0., 1., 0.],
                         [0., 0., 1.]]) # The position of this atom will be varied. It's a proxy for rgyr

        coords = np.zeros((2000, 4, 3))
        for ii in range(2000):
            coords[ii] = ixyz
        # Here's were we create a bimodal distro were the most populated is the open structure
        z = np.hstack((np.random.rand(200)*2 + 1, np.random.randn(1800) * 3 + 15))
        z = np.random.permutation(z)
        coords[:, -1, -1] = z
        self.traj = md.Trajectory(coords, traj.top)
        self.cl = bmutils.regspace_cluster_to_target(self.traj.xyz[:,-1, -1], 50, n_try_max=10)
        self.cat_smpl = self.cl.sample_indexes_by_cluster(np.arange(self.cl.n_clusters), n_geom_samples)
        self.geom_smpl = self.traj[np.vstack(self.cat_smpl)[:,1]]
        self.geom_smpl = bmutils.re_warp(self.geom_smpl, [n_geom_samples]*self.cl.n_clusters)

    def test_smallest_rgyr(self):
        start_idx = bmutils.get_good_starting_point(self.cl, self.geom_smpl)
        # Should be the clustercenter with the smallest possible rgyr
        assert self.cl.clustercenters[start_idx] ==  self.cl.clustercenters.squeeze().min()

    def test_most_pop(self):
        start_idx = bmutils.get_good_starting_point(self.cl, self.geom_smpl, strategy="most_pop")
        #print(start_idx, self.cl.clustercenters[start_idx], np.sort(self.cl.clustercenters.squeeze()))
        assert 12 <= self.cl.clustercenters[start_idx] <= 18, "The coordinate distribution was created with a max pop " \
                                                              "around the value 15. The found starting point should be" \
                                                              "in this interval (see the setUp)"

    def _test_most_pop_x_rgyr(self):
        start_idx = bmutils.get_good_starting_point(self.cl, self.geom_smpl, strategy="most_pop_x_smallest_Rgyr")
        #print(start_idx, self.cl.clustercenters[start_idx], np.sort(self.cl.clustercenters.squeeze()))
        # TODO: figure out a good way of testing this, at the moment it just chekcs that it runs

    def test_bimodal_compact(self):
        start_idx = bmutils.get_good_starting_point(self.cl, self.geom_smpl, strategy="bimodal_compact")
        #print(self.cl.clustercenters[start_idx])
        #print(np.sort(self.cl.clustercenters.squeeze()))
        assert -1 <= self.cl.clustercenters[start_idx] <= 3, "The coordinate distribution was created with pop "\
                                                            "maxima the values 1 (compact) and 15 (open)." \
                                                            " The found COMPACT starting "\
                                                            "point should be in the interval [0,3] approx (see setUp)"

    def test_bimodal_open(self):
        start_idx = bmutils.get_good_starting_point(self.cl, self.geom_smpl, strategy="bimodal_open")
        # print(self.cl.clustercenters[start_idx])
        #print(np.sort(self.cl.clustercenters.squeeze()))
        assert 13 <= self.cl.clustercenters[start_idx] <= 17, "The coordinate distribution was created with pop " \
                                                            "maxima the values 1 (compact) and 15 (open)." \
                                                            " The found OPEN starting " \
                                                            "point should be in the interval [13,16] approx (see setUp)"

    def test_most_pop_ordering(self):
        order = np.random.permutation(np.arange(self.cl.n_clusters))
        geom_smpl = [self.geom_smpl[ii] for ii in order]
        start_idx = bmutils.get_good_starting_point(self.cl, geom_smpl,
                                                    cl_order=order, strategy='most_pop')

        # Test might fail in some cases because it's not deterministi
        # For the check to work via the clustercenter value, we have to re-order the clusters themselves
        assert 12 <= self.cl.clustercenters[order][start_idx] <= 18, "The coordinate distribution was created with " \
                                                                     "a max pop around the value 15. The found starting " \
                                                                     "point should be in this interval (see the setUp)"
        # However, for the geom_smpl object, start_idx can be used directly:
        assert 12 <= geom_smpl[start_idx].xyz[:,-1,-1].mean() <= 18, "The coordinate distribution was created with " \
                                                                     "a max pop around the value 15. The found starting " \
                                                                     "point should be in this interval (see the setUp)"

    # The rest of strategies do not need a test, since ordering does not play a role in them

class TestPaths(unittest.TestCase):

    def setUp(self):
        self.start = np.array([0, 0, 0])
        self.path_of_candidates = [ # Two candidates two choose for each point along the path
                                    np.array([[1,  0,    1],    #this one is closest overall to 0 0 0
                                              [1,  20,  20],    # filler
                                              [1,  1.5,  0]]),  #this one is closest if the second coord is ignored

                                   np.array([[2,  1.5,  0],     # closest if 2nd coord excluded
                                             [2,  0,    1],     # closest overall
                                             [2,  20,  20]]),   # filler

                                   np.array([[3,  20,  20],     #filler
                                             [3,  1.5,  0],     # closest if 2nd coord excluded
                                             [3,  0,    1]]),    # closest overall

                                   np.array([[4,  0,    0],     # This one is only closer if we choose to have excluded some coords
                                             [4,  0., .75],    # This one is closest to the last frame no matter what
                                             [2,  0,  .75]])     # The history aware option should pick this one out
                                   ]

        # the "all-coords-included-path" should be
        # [0], [1], [2]

        # the "some-coords-excluded should be
        # [2], [0], [1]

        # variable "now" with history aware option is
        # array([ 1.5 ,  0.  ,  0.75])

    def test_closest_all_coords_no_history(self):
        path = bmutils.min_disp_path(self.start, self.path_of_candidates)
        assert np.allclose(path, [0, 1, 2, 1]), path

    def test_closest_exclude_some_no_history(self):
        path = bmutils.min_disp_path(self.start, self.path_of_candidates, exclude_coords=1)
        assert np.allclose(path, [2, 0, 1, 0]), path

    def test_closest_all_coords_history(self):
        path = bmutils.min_disp_path(self.start, self.path_of_candidates, history_aware=True)
        assert np.allclose(path, [0, 1, 2, 2]), path

if __name__ == '__main__':
    unittest.main()
