__author__ = 'gph82'

import unittest
import pyemma
import os
import tempfile
import numpy as np
import shutil
from molpx import _bmutils
import mdtraj as md
from glob import glob
import molpx

class TestWithBPTIData(unittest.TestCase):
    r"""
    A class that contains all the TestCase with the MD info
    """
    @classmethod
    def setUpClass(self):
        self.MD_trajectory_files = glob(molpx._molpxdir(join='notebooks/data/c-alpha_centered.stride.1000*xtc'))
        self.MD_topology_file = molpx._molpxdir(join='notebooks/data/bpti-c-alpha_centered.pdb')
        self.MD_topology = md.load(self.MD_topology_file).top
        self.MD_trajectories = [md.load(ff, top=self.MD_topology_file) for ff in self.MD_trajectory_files]

        self.xyz_flat = [igeom.xyz.squeeze().reshape(igeom.n_frames, -1) for igeom in self.MD_trajectories]
        self.feat = pyemma.coordinates.featurizer(self.MD_topology_file)
        self.feat.add_all()
        self.source = pyemma.coordinates.source(self.MD_trajectory_files, features=self.feat)
        self.Fs = self.source.get_output()

        self.tica = pyemma.coordinates.tica(self.source, lag=1, dim=2)
        self.pca = pyemma.coordinates.tica(self.source, dim=2)
        self.Ys = self.tica.get_output()
        self.tempdir = tempfile.mkdtemp('test_molpx')
        self.projected_files = [os.path.join(self.tempdir, 'Y.%u.npy'%ii) for ii in range(len(self.MD_trajectories))]
        [np.save(ifile, iY) for ifile, iY in zip(self.projected_files, self.Ys)]
        [np.savetxt(ifile.replace('.npy', '.dat'), iY) for ifile, iY in zip(self.projected_files, self.Ys)]

    @classmethod
    def tearDownClass(self):
        shutil.rmtree(self.tempdir)

class TestReadingInput(unittest.TestCase):

    def setUp(self):
        self.MD_trajectory = os.path.join(pyemma.__path__[0],'coordinates/tests/data/bpti_mini.xtc')
        self.MD_topology = os.path.join(pyemma.__path__[0], 'coordinates/tests/data/bpti_ca.pdb')
        self.tempdir = tempfile.mkdtemp('test_molpx')
        self.projected_file = os.path.join(self.tempdir,'Y.npy')
        self.feat = pyemma.coordinates.featurizer(self.MD_topology)
        self.feat.add_all()
        source = pyemma.coordinates.source(self.MD_trajectory, features=self.feat)
        self.tica = pyemma.coordinates.tica(source,lag=1, dim=2)
        self.Y = self.tica.get_output()[0]
        self.F = source.get_output()
        np.save(self.projected_file,self.Y)
        np.savetxt(self.projected_file.replace('.npy','.dat'),self.Y)

    def tearDown(self):
        shutil.rmtree(self.tempdir)

    def test_data_from_input_npy(self):
        # Just one string
        assert np.allclose(self.Y, _bmutils.data_from_input(self.projected_file)[0])
        # List of one string
        assert np.allclose(self.Y, _bmutils.data_from_input([self.projected_file])[0])
        # List of two strings
        Ys = _bmutils.data_from_input([self.projected_file,
                                       self.projected_file])
        assert np.all([np.allclose(self.Y, iY) for iY in Ys])

    def test_data_from_input_throws_exception(self):
        try:
            _bmutils.data_from_input(np.random.randn(1000))
        except ValueError:
            pass

    def test_data_from_input_ascii(self):
        # Just one string
        assert np.allclose(self.Y, _bmutils.data_from_input(self.projected_file.replace('.npy', '.dat'))[0])
        # List of one string
        assert np.allclose(self.Y, _bmutils.data_from_input([self.projected_file.replace('.npy', '.dat')])[0])
        # List of two strings
        Ys = _bmutils.data_from_input([self.projected_file.replace('.npy', '.dat'),
                                       self.projected_file.replace('.npy','.dat')])
        assert np.all([np.allclose(self.Y, iY) for iY in Ys])

    def test_data_from_input_ndarray(self):
        # Just one ndarray
        assert np.allclose(self.Y, _bmutils.data_from_input(self.Y)[0])
        # List of one ndarray
        assert np.allclose(self.Y, _bmutils.data_from_input([self.Y])[0])
        # List of two ndarray
        Ys = _bmutils.data_from_input([self.Y,
                                       self.Y])
        assert np.all([np.allclose(self.Y, iY) for iY in Ys])

    # Not implemented yet
    def _test_data_from_input_ndarray_ascii_npy(self):
        # List of everything
        Ys = _bmutils.data_from_input([self.projected_file,
                                       self.projected_file.replace('.npy','.dat'),
                                       self.Y])
        assert np.all([np.allclose(self.Y, iY) for iY in Ys])

    def test_moldata_from_input(self):
        # Traj and top strings
        moldata = _bmutils.moldata_from_input(self.MD_trajectory, MD_top=self.MD_topology)
        assert isinstance(moldata, _bmutils._FeatureReader)

        # Source object directly
        moldata = _bmutils.moldata_from_input(moldata)
        assert isinstance(moldata, _bmutils._FeatureReader)

        # Typerror:
        try:
            _bmutils.moldata_from_input(11)
        except TypeError:
            pass

        # List of trajectories
        geom = md.load(self.MD_trajectory, top=self.MD_topology)
        moldata = _bmutils.moldata_from_input(geom)
        assert isinstance(moldata[0], md.Trajectory), moldata

    def test_assert_moldata_belong_data(self):
        # Traj vs data
        geom = md.load(self.MD_trajectory, top=self.MD_topology)
        _bmutils.assert_moldata_belong_data([geom], [self.Y])

        # src vs data
        moldata = _bmutils.moldata_from_input(self.MD_trajectory, MD_top=self.MD_topology)
        _bmutils.assert_moldata_belong_data(moldata, [self.Y])

        # With stride
        _bmutils.assert_moldata_belong_data([geom], [iY[::5] for iY in [self.Y]], data_stride=5)

    def test_most_corr_info_works(self):
        most_corr = _bmutils.most_corr(self.tica)

        ref_idxs = [np.abs(self.tica.feature_TIC_correlation[:, ii]).argmax() for ii in range(self.tica.dim)]

        # Since the results are always lists, that's how the comparsion will go
        # Idxs are okay
        assert np.all([ri == mci for ri, mci in zip(ref_idxs, most_corr['idxs'])])

        ref_corrs = [self.tica.feature_TIC_correlation[jj, ii] for jj, ii in zip(ref_idxs,range(self.tica.dim) )]
        # Values are ok
        assert np.all([rv == mcv for rv, mcv in zip(ref_corrs, most_corr['vals'])])

        # Labels are strings and are the right number
        assert len(most_corr['labels']) == len(ref_idxs)
        assert [isinstance(istr, str) for istr in most_corr['labels']]

        # No geom was parsed, show the last onew should be empty
        assert most_corr['feats'] == []

    def test_most_corr_info_works_with_options(self):
        geoms = md.load(self.MD_trajectory, top=self.MD_topology)
        most_corr = _bmutils.most_corr(self.tica, geoms=geoms)

        # Idxs are okay
        ref_idxs = [np.abs(self.tica.feature_TIC_correlation[:, ii]).argmax() for ii in range(self.tica.dim)]
        assert np.all([ri == mci for ri, mci in zip(ref_idxs, most_corr['idxs'])])

        # Corr values
        ref_corrs = [self.tica.feature_TIC_correlation[jj, ii] for jj, ii in zip(ref_idxs, range(self.tica.dim))]
        assert np.all([rv == mcv for rv, mcv in zip(ref_corrs, most_corr['vals'])])

        # Check that we got the right most correlated feature trajectory
        ref_feats = self.feat.transform(geoms)
        ref_feats = [ref_feats[:, ii] for ii in ref_idxs]
        assert np.all(np.allclose(rv, mcv) for rv, mcv in zip(ref_feats, np.squeeze(most_corr['feats'])))

    def test_most_corr_info_works_with_options_and_proj_idxs(self):
        geoms = md.load(self.MD_trajectory, top=self.MD_topology)

        proj_idxs = [1, 0] # the order shouldn't matter
        corr_dict = _bmutils.most_corr(self.tica, geoms=geoms, proj_idxs=proj_idxs)

        # Idxs are okay
        ref_idxs = [np.abs(self.tica.feature_TIC_correlation[:, ii]).argmax() for ii in proj_idxs]
        assert np.all([ri == mci for ri, mci in zip(ref_idxs, corr_dict['idxs'])]), (ref_idxs, corr_dict["idxs"])


        # Corr values
        ref_corrs = [self.tica.feature_TIC_correlation[jj, ii] for jj, ii in zip(ref_idxs, proj_idxs)]
        assert np.all([rv == mcv for rv, mcv in zip(ref_corrs, corr_dict['vals'])])

        # Labels are strings and are the right number
        assert len(corr_dict["labels"]) == len(ref_idxs)
        assert [isinstance(istr, str) for istr in corr_dict["labels"]]

        # Feature values are ok
        ref_feats = self.feat.transform(geoms)
        ref_feats = [ref_feats[:, ii] for ii in ref_idxs]
        assert np.all(np.allclose(rv, mcv) for rv, mcv in zip(ref_feats, np.squeeze(corr_dict['feats'])))

    def test_most_corr_info_wrong_proj_idxs(self):

        proj_idxs = [1, 0, 10] # we don't have 10 TICs
        try:
            _bmutils.most_corr(self.tica, proj_idxs=proj_idxs)
        except(ValueError):
            pass #this should given this type of error


class TestClusteringAndCatalogues(unittest.TestCase):

    def setUp(self):
        self.data_for_cluster = [np.array([[1, 3],
                                           [2, 3],
                                           [3, 3]]),
                                 np.array([[17, 1]]),
                                 np.array([[11, 2],
                                           [12, 2],
                                           [13, 3]]),
                                 np.array([[1, 3],
                                           [11, 2],
                                           [2, 3],
                                           [12, 2],
                                           [3, 3],
                                           [13, 2]])
                                 ]

    def test_cluster_to_target(self):
        n_target = 15
        data = [np.random.randn(100, 1), np.random.randn(100,1)+10]
        cl = _bmutils.regspace_cluster_to_target(data, n_target, n_try_max=10, delta=0, verbose=True)
        assert n_target - 1 <= cl.n_clusters <= n_target + 1

    def test_catalogues(self):
        cl = _bmutils.regspace_cluster_to_target(self.data_for_cluster, 3, n_try_max=10, delta=0)
        #print(cl.clustercenters)
        cat_idxs, cat_cont = _bmutils.catalogues(cl)

        # Shape testing
        assert len(cat_idxs) == len(cat_cont)
        for iter1, iter2 in zip(cat_idxs, cat_cont):
            assert len(iter1)==len(iter2)

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

    def test_catalogues_with_data(self):
        cl = _bmutils.regspace_cluster_to_target(self.data_for_cluster, 3, n_try_max=10, delta=0)
        #print(cl.clustercenters)
        cat_idxs, cat_cont = _bmutils.catalogues(cl, data=self.data_for_cluster)

        # Shape testing
        assert len(cat_idxs) == len(cat_cont)
        for iter1, iter2 in zip(cat_idxs, cat_cont):
            assert len(iter1)==len(iter2)

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
        cl = _bmutils.regspace_cluster_to_target(self.data_for_cluster, 3, n_try_max=10, delta=0)
        cat_idxs, cat_cont = _bmutils.catalogues(cl, sort_by=0)

        # This test is extra, since this is a pure pyemma function
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
        cl = _bmutils.regspace_cluster_to_target(self.data_for_cluster, 3, n_try_max=10, delta=0)
        cat_idxs, cat_cont = _bmutils.catalogues(cl, sort_by=1)
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

class TestReWarp(unittest.TestCase):

    def setUp(self):
        self.ref = [0,1,2,3,4,5,6,7,8,9,10]

    def test_warp_to_int(self):
        warped = _bmutils.re_warp(self.ref, 4)

        assert np.allclose(warped[0], [0,1,2,3])
        assert np.allclose(warped[1], [4, 5, 6, 7])
        assert np.allclose(warped[2], [8, 9, 10])

    def test_warp_to_longer_seq(self):
        warped = _bmutils.re_warp(self.ref, [2,3,5,4,4])
        assert np.allclose(warped[0],[0,1])
        assert np.allclose(warped[1],[2,3, 4])
        assert np.allclose(warped[2],[5,6,7,8,9])
        assert np.allclose(warped[3], [10])
        assert np.allclose(warped[4], [])

    def test_warp_to_shorter_seq(self):
        warped = _bmutils.re_warp(self.ref, [1])
        assert np.allclose(warped[0], self.ref[0])
        assert len(warped)==1

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
        self.cl = _bmutils.regspace_cluster_to_target(self.traj.xyz[:, -1, -1], 50, n_try_max=10)
        self.cat_smpl = self.cl.sample_indexes_by_cluster(np.arange(self.cl.n_clusters), n_geom_samples)
        self.geom_smpl = self.traj[np.vstack(self.cat_smpl)[:,1]]
        self.geom_smpl = _bmutils.re_warp(self.geom_smpl, [n_geom_samples] * self.cl.n_clusters)


    def test_throws_exception(self):
        try:
            start_idx = _bmutils.get_good_starting_point(self.cl, self.geom_smpl, strategy="what?")
        except NotImplementedError:
            pass

    # This test doesn't exactly belong here but this is the best class for now
    def test_find_centers_GMM(self):
        # Two 1D gaussians, one centered at 5 the other at 10

        data = [np.random.randn(100, 1) + 5, np.random.randn(100, 1) + 10]
        data = np.vstack(data)
        grid = np.arange(0, 20)
        # This is the super easy test (array is ordered, index and value are the same)
        idxs, igmm = _bmutils.find_centers_gmm(data, grid)
        assert np.allclose(np.sort(idxs), [5, 10])

        # Works also with arrays that do not necessarily contain the centers or are unsorted
        grid = [-25, 10, 1, 4, -100, 10]
        idxs, igmm = _bmutils.find_centers_gmm(data, grid)
        assert np.allclose(np.sort(idxs), [1, 3])

    def test_smallest_rgyr(self):
        start_idx = _bmutils.get_good_starting_point(self.cl, self.geom_smpl)
        # Should be the clustercenter with the smallest possible rgyr
        assert self.cl.clustercenters[start_idx] ==  self.cl.clustercenters.squeeze().min()

    def test_most_pop(self):
        start_idx = _bmutils.get_good_starting_point(self.cl, self.geom_smpl, strategy="most_pop")
        #print(start_idx, self.cl.clustercenters[start_idx], np.sort(self.cl.clustercenters.squeeze()))
        assert 12 <= self.cl.clustercenters[start_idx] <= 18, "The coordinate distribution was created with a max pop " \
                                                              "around the value 15. The found starting point should be" \
                                                              "in this interval (see the setUp)"

    def _test_most_pop_x_rgyr(self):
        start_idx = _bmutils.get_good_starting_point(self.cl, self.geom_smpl, strategy="most_pop_x_smallest_Rgyr")
        #print(start_idx, self.cl.clustercenters[start_idx], np.sort(self.cl.clustercenters.squeeze()))
        # TODO: figure out a good way of testing this, at the moment it just chekcs that it runs

    def test_bimodal_compact(self):
        start_idx = _bmutils.get_good_starting_point(self.cl, self.geom_smpl, strategy="bimodal_compact")
        #print(self.cl.clustercenters[start_idx])
        #print(np.sort(self.cl.clustercenters.squeeze()))
        assert -1 <= self.cl.clustercenters[start_idx] <= 3, "The coordinate distribution was created with pop "\
                                                            "maxima the values 1 (compact) and 15 (open)." \
                                                            " The found COMPACT starting "\
                                                            "point should be in the interval [0,3] approx (see setUp)"

    def test_bimodal_open(self):
        start_idx = _bmutils.get_good_starting_point(self.cl, self.geom_smpl, strategy="bimodal_open")
        # print(self.cl.clustercenters[start_idx])
        #print(np.sort(self.cl.clustercenters.squeeze()))
        assert 12 <= self.cl.clustercenters[start_idx] <= 18, "The coordinate distribution was created with pop " \
                                                            "maxima the values 1 (compact) and 15 (open)." \
                                                            " The found OPEN starting " \
                                                            "point should be in the interval [12,18] approx (see setUp)"

    def test_most_pop_ordering(self):
        order = np.random.permutation(np.arange(self.cl.n_clusters))
        geom_smpl = [self.geom_smpl[ii] for ii in order]
        start_idx = _bmutils.get_good_starting_point(self.cl, geom_smpl,
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

class TestVisualPath(TestWithBPTIData):

    @classmethod
    def setUpClass(self):
        TestWithBPTIData.setUpClass()
        n_sample = 20
        self.cl_cont = _bmutils.regspace_cluster_to_target([ixyz[:, :2] for ixyz in self.xyz_flat], n_sample, verbose=True, n_try_max=10,
                                                           #delta=0
                                                           )
        self.cat_idxs, self.cat_data = _bmutils.catalogues(self.cl_cont)
        # Create the MD catalogue with pyemma
        self.cat_data_md = [pyemma.coordinates.save_traj(self.MD_trajectory_files, icat, None, top=self.MD_topology)
                            for icat in self.cat_idxs]

    def test_just_works_data(self):
        _bmutils.visual_path(self.cat_idxs, self.cat_data)

    def test_just_works_data_md(self):
        _bmutils.visual_path(self.cat_idxs, self.cat_data_md,path_type="min_rmsd")

    def test_input_parsing(self):
        _bmutils.visual_path(self.cat_idxs, self.cat_data, start_pos="maxpop")
        _bmutils.visual_path(self.cat_idxs, self.cat_data, start_pos=1)

        # Not implemented Errors
        try:
            _bmutils.visual_path(self.cat_idxs, self.cat_data, start_pos="other")
        except NotImplementedError:
            pass
        try:
            _bmutils.visual_path(self.cat_idxs, self.cat_data, path_type="xxxx")
        except NotImplementedError:
            pass

class TestMinDispPaths(unittest.TestCase):

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
        path = _bmutils.min_disp_path(self.start, self.path_of_candidates)
        assert np.allclose(path, [0, 1, 2, 1]), path

    def test_closest_exclude_some_no_history(self):
        path = _bmutils.min_disp_path(self.start, self.path_of_candidates, exclude_coords=1)
        assert np.allclose(path, [2, 0, 1, 0]), path

    def test_closest_all_coords_history(self):
        path = _bmutils.min_disp_path(self.start, self.path_of_candidates, history_aware=True)
        assert np.allclose(path, [0, 1, 2, 2]), path

class TestSliceListOfGeoms(unittest.TestCase):

    def setUp(self):
        self.topology = os.path.join(pyemma.__path__[0],'coordinates/tests/data/bpti_ca.pdb')
        self.ref_frame = 4
        self.MD_trajectory = md.load(os.path.join(pyemma.__path__[0],
                                                  'coordinates/tests/data/bpti_mini.xtc'),
                                     top=self.topology)
    def test_slice(self):
        geom_list = [self.MD_trajectory,
                     self.MD_trajectory[::-1]]

        equal_to_ref_frames = _bmutils.slice_list_of_geoms_to_closest_to_ref(geom_list, self.MD_trajectory[self.ref_frame])
        rmsd_to_ref_should_be_zero = md.rmsd(equal_to_ref_frames, self.MD_trajectory[self.ref_frame])
        assert np.allclose(rmsd_to_ref_should_be_zero, [0, 0], atol=1e-3), rmsd_to_ref_should_be_zero

class TestGetAscendingCoordIdx(unittest.TestCase):

    def setUp(self):
        self.data = np.zeros((100, 4))
        self.data[:,0] = np.arange(100)
        self.data[:,1] = np.arange(100)[::-1]
        self.data[:,2] = np.random.randn(100)
        self.data[:,3] = np.linspace(0, 50, num=100)
    def test_it_works(self):
        assert np.allclose([0], _bmutils.get_ascending_coord_idx(self.data[:,:-1]))
    def test_empty_no_fail(self):

        result = _bmutils.get_ascending_coord_idx(self.data[:,[1,2]], fail_if_empty=False)
        assert len(result)==0
    def test_empty_fail(self):
        try:
            _bmutils.get_ascending_coord_idx(self.data[:, 1:], fail_if_empty=True)
        except ValueError:
            pass
    def test_more_than_one_fails(self):
        try:
            _bmutils.get_ascending_coord_idx(self.data[:, :], fail_if_more_than_one=True)
        except:
            pass

class TestMinRmsdPaths(unittest.TestCase):

    def setUp(self):
        self.topology = os.path.join(pyemma.__path__[0],'coordinates/tests/data/bpti_ca.pdb')
        self.reftraj = md.load(self.topology)

    def test_find_buried_best_candidate(self):
        n_cands = 20
        path_length = 50
        # Create a path of candidates that's just perturbed versions of the same geometry
        xyz = [np.reshape([self.reftraj.xyz[0]+np.random.randn(self.reftraj.n_atoms,3) for ii in range(n_cands)],
                          (n_cands, self.reftraj.n_atoms,3)) for jj in range(path_length)]

        # Now "bury" the true geometry somewhere in these candidates
        frames_where_actual_geometry_is =  np.random.randint(0, high=n_cands, size=path_length)
        for pp, ff in enumerate(frames_where_actual_geometry_is):
            xyz[pp][ff,:,:] = self.reftraj.xyz
        # Create the path of candidates as mdtraj objects
        path_of_candidates = [md.Trajectory(ixyz, topology=self.reftraj.top) for ixyz in xyz]

        # Let mirmsd_path find these frames for you
        inferred_frames = _bmutils.min_rmsd_path(self.reftraj, path_of_candidates)

        assert np.allclose(frames_where_actual_geometry_is, inferred_frames)

    def test_find_buried_best_candidate_with_selection(self):
        r"""

        This test is tricky to understand:
            in the list with candidates, all geoms are random except two:
                - A copy of the reference with a small selection of perturbed atoms
                    => because the randomization is small, this frame will always be found if no selection is given
                    to minrmsd_path
                - A random geometry with a small selection if untouched atoms
                    ==> when this selection is given, minrsmd_path will find this frame although the rest of the
                    frame is noise

        """
        n_cands = 20
        path_length = 50
        selection = np.array([0, 1, 2, 3])

        # Create a path of candidates that's just noise
        xyz = [np.reshape([np.random.randn(self.reftraj.n_atoms, 3) for ii in range(n_cands)],
                          (n_cands, self.reftraj.n_atoms, 3)) for jj in range(path_length)]

        # Create selection-perturbed versions of the reference
        ref_w_sel_perturbed = []
        for jj in range(path_length):
            ixyz = np.copy(self.reftraj.xyz[0]) # copy, otherwise we're overwriting the original coords
            ixyz[selection, :] += np.random.randn(len(selection), 3)
            ref_w_sel_perturbed.append(ixyz)

        # Bury ref_w_sel_perturbed somewhere in the random candidates
        frames_ref_w_sel_perturbed =  np.random.randint(0, high=n_cands, size=path_length)
        for pp, ff in enumerate(frames_ref_w_sel_perturbed):
            xyz[pp][ff,:,:] = ref_w_sel_perturbed[pp]

        # Create the path of candidates as mdtraj objects
        path_of_candidates = [md.Trajectory(ixyz, topology=self.reftraj.top) for ixyz in xyz]

        # PRE-TEST
        # as long as the perturbation is small,
        # minrmsd_path will find these ref_w_sel_perturbed frames among the noise no problem
        inferred_frames = _bmutils.min_rmsd_path(self.reftraj, path_of_candidates)
        assert np.allclose(frames_ref_w_sel_perturbed, inferred_frames)

        # create a second batch of frames where everything is random EXCEPT THE SELECTION, thus a search limited
        #  to the selection should ignore it and still find it
        frames_random_w_sel_untouched = np.random.randint(0, high=n_cands, size=path_length)

        # Likely some frames overlap, since the chances to find at least one overlap is path_len/n_cands > 1)
        for ii in np.argwhere([fi == fj for fi, fj in zip(frames_ref_w_sel_perturbed,
                                                          frames_random_w_sel_untouched)]):
            frames_random_w_sel_untouched[ii] -= 1
            if frames_random_w_sel_untouched[ii] == -1:
                frames_random_w_sel_untouched[ii] = n_cands-1

        assert not np.any([fi == fj for fi, fj in zip(frames_ref_w_sel_perturbed,
                                                      frames_random_w_sel_untouched)])

        # In the random frames, insert the intouched selection
        for pp, ff in enumerate(frames_random_w_sel_untouched):
            xyz[pp][ff,selection,:] = np.copy(self.reftraj.xyz[0,selection, :])
        path_of_candidates = [md.Trajectory(ixyz, topology=self.reftraj.top) for ixyz in xyz]

        # This should still be OK, because
        # even if the selected atoms have been perturbed, the comparsion [ref+sel_per] vs [random] is robust
        inferred_frames = _bmutils.min_rmsd_path(self.reftraj, path_of_candidates)
        assert np.allclose(frames_ref_w_sel_perturbed, inferred_frames), ("min_rmsd_path wasn't able to distinguish slightly perturbed geometries "
                                                                          "from totally random geometries")
        # This should fail
        assert not np.any([fi == fj for fi, fj in zip(frames_random_w_sel_untouched, inferred_frames)])
        # This should be OK because we're only looking at the selection
        inferred_frames = _bmutils.min_rmsd_path(self.reftraj, path_of_candidates, selection=selection)
        assert np.allclose(frames_random_w_sel_untouched, inferred_frames), ("Even when limiting the selection for minRMSD computation to "
                                                                             "the selected atoms, min_rmsd_path wasn't able to find the"
                                                                             "random frames with the selected atoms as reference buried"
                                                                             "in totally random coordinates:\n %s \nvs\n%s"%(frames_random_w_sel_untouched, inferred_frames))

    def test_history_aware_just_works(self):
        n_cands = 20
        path_length = 50
        # Create a path of candidates that's just perturbed versions of the same geometry
        xyz = [np.reshape([self.reftraj.xyz[0] + np.random.randn(self.reftraj.n_atoms, 3) for ii in range(n_cands)],
                          (n_cands, self.reftraj.n_atoms, 3)) for jj in range(path_length)]

        # Now "bury" the true geometry somewhere in these candidates
        frames_where_actual_geometry_is = np.random.randint(0, high=n_cands, size=path_length)
        for pp, ff in enumerate(frames_where_actual_geometry_is):
            xyz[pp][ff, :, :] = self.reftraj.xyz
        # Create the path of candidates as mdtraj objects
        path_of_candidates = [md.Trajectory(ixyz, topology=self.reftraj.top) for ixyz in xyz]

        # Let mirmsd_path find these frames for you
        inferred_frames = _bmutils.min_rmsd_path(self.reftraj, path_of_candidates, history_aware=True)

        assert np.allclose(frames_where_actual_geometry_is, inferred_frames)

class TestSmoothingFunctions(unittest.TestCase):

    def setUp(self):
        # The setup creates the typical, "geometries-sampled along cluster-scenario"
        traj = md.load(os.path.join(pyemma.__path__[0], 'coordinates/tests/data/bpti_ca.pdb'))
        traj = traj.atom_slice([0, 1])  # create a trajectory with two atoms
        # Create a fake bi-modal trajectory with a compact and an open structure
        ixyz = np.array([[10.,  20.,  30.],
                         [100., 200., 300.],
                         ])
        xyz =[ixyz+np.ones((2,3))*ii for ii in range(10)] # Easy way to generate an easy to average geometry
        self.traj = md.Trajectory(xyz, traj.top)

    def tearDown(self):
        pass

    def test_running_avg_idxs_none(self):
        idxs, windows = _bmutils.running_avg_idxs(10, 0)
        # If the running average is with radius zero, it's just a normal average
        assert np.allclose([0,1,2,3,4,5,6,7,8,9], idxs)
        assert np.all(np.allclose(ii,jj) for ii, jj in zip(([0,1,2,3,4,5,6,7,8,9], windows)))

    def test_running_avg_idxs_one(self):
        idxs, windows = _bmutils.running_avg_idxs(10, 1)
        assert np.allclose([1,2,3,4,5,6,7,8], idxs) # we've lost first and last frame
        # The '#' marks where the average is centered on:      #
        assert np.all(np.allclose(ii,jj) for ii, jj in zip([[0,1,2],
                                                            [1,2,3],
                                                            [2,3,4],
                                                            [3,4,5],
                                                            [5,6,7],
                                                            [6,7,8],
                                                            [7,8,9]],
                                                           windows))

    def test_running_avg_idxs_two(self):
        idxs, windows = _bmutils.running_avg_idxs(10, 2)
        assert np.allclose([2, 3, 4, 5, 6, 7], idxs)  # we've lost 2 first and last frames
        # The '#' marks where the average is centered on:           #
        assert np.all(np.allclose(ii, jj) for ii, jj in zip([[0, 1, 2, 3, 4],
                                                             [1, 2, 3, 4, 5],
                                                             [2, 3, 4, 5, 6],
                                                             [3, 5, 6, 7, 8],
                                                             [4, 6, 7, 8, 9]],
                                                            windows))

    def test_running_avg_idxs_too_large_window(self):
        try:
            idxs, windows = _bmutils.running_avg_idxs(10, 5)
        except AssertionError:
            pass


    def test_smooth_geom_it_just_runs_and_gives_correct_output_type(self):

        # No data
        assert isinstance(_bmutils.smooth_geom(self.traj, 0), md.Trajectory)
        assert isinstance(_bmutils.smooth_geom(self.traj, 0, superpose=False), md.Trajectory)

        # One frame, no data
        geom = _bmutils.smooth_geom(self.traj, 1, superpose=False)
        assert isinstance(geom, md.Trajectory)

        # One frame, data (fake data, but data)
        geom, xyz = _bmutils.smooth_geom(self.traj, 1, geom_data=self.traj.xyz.mean(-1))
        assert isinstance(geom, md.Trajectory)
        assert isinstance(xyz, np.ndarray)

    # The running average is equal the center of the window bc the window is linear around its center
    def test_smooth_geom_right_output_linear(self):

        # IDK how to compare superimposed geoms consistently (easily)
        geom = _bmutils.smooth_geom(self.traj, 0, superpose=False)
        assert np.allclose(geom.xyz, self.traj.xyz), (geom.xyz, self.traj.xyz)

        # Weak but easy to write tests
        geom = _bmutils.smooth_geom(self.traj, 1, superpose=False)
        assert np.allclose(geom.xyz, self.traj[1:-1].xyz), (geom.xyz, self.traj[1:-1].xyz)

        # Weak but easy to write tests
        geom = _bmutils.smooth_geom(self.traj, 2, superpose=False)
        assert np.allclose(geom.xyz, self.traj[2:-2].xyz), (geom.xyz, self.traj[1:-1].xyz)

        # Weak but easy to write tests with data
        geom, xyz = _bmutils.smooth_geom(self.traj, 2, superpose=False, geom_data=self.traj.xyz.mean(-1))
        assert np.allclose(geom.xyz, self.traj[2:-2].xyz), (geom.xyz, self.traj[1:-1].xyz)
        assert np.allclose(self.traj.xyz.mean(-1)[2:-2], xyz)

    def test_smooth_geom_right_output_square(self):
        # TODO FINISH THIS TESTS: make it so that the averaged value is not linear in the running variable
        # This geometry is harder to do
        xyz = [self.traj.xyz[0] + np.ones((2, 3)) * ii**2 for ii in range(10)]
        self.traj = md.Trajectory(xyz, self.traj.top)
        #print(self.traj.xyz)

class TestListTransposeGeomList(unittest.TestCase):

    def test_it(self):
        # Create a dummy topology
        traj = md.load(os.path.join(pyemma.__path__[0], 'coordinates/tests/data/bpti_ca.pdb'))
        top = traj.atom_slice([0]).top  # Single atom topology

        ixyz_row_0 =  [[0, 0, 0]],    [[0, 1, 0]],    [[0, 2, 0]] # 3 frames of 1 atom
        ixyz_row_1 =  [[1, 0, 0]],    [[1, 1, 0]],    [[2, 2, 0]] # 3 frames of 1 atom

        # A list of 2 elements each of 3 frames
        geom_list = [md.Trajectory(ixyz_row_0, topology=top),
                     md.Trajectory(ixyz_row_1, topology=top)]



        # A list of 3 elements each of 2 frames
        transposed_geom_list = _bmutils.transpose_geom_list(geom_list)

        new_ixyz_column_0 = [igeom[0].xyz for igeom in transposed_geom_list]
        new_ixyz_column_1 = [igeom[1].xyz for igeom in transposed_geom_list]

        assert np.allclose(np.squeeze(ixyz_row_0), np.squeeze(new_ixyz_column_0)), (ixyz_row_0, new_ixyz_column_0)
        assert np.allclose(np.squeeze(ixyz_row_1), np.squeeze(new_ixyz_column_1)), (ixyz_row_1, new_ixyz_column_1)

class geom_list_2_geom(unittest.TestCase):

    def test_it(self):
        # Create a dummy topology
        MD_trajectory = os.path.join(pyemma.__path__[0], 'coordinates/tests/data/bpti_mini.xtc')
        MD_topology = os.path.join(pyemma.__path__[0], 'coordinates/tests/data/bpti_ca.pdb')
        traj = md.load(MD_trajectory, top=MD_topology)

        traj_list = [itraj for itraj in traj]
        new_geom = _bmutils.geom_list_2_geom(traj_list)

        assert np.allclose(np.hstack([igeom.xyz for igeom in new_geom]).squeeze(), np.vstack(traj.xyz))

class TestIndexFromFeatures(unittest.TestCase):

    def setUp(self):
        self.MD_topology = os.path.join(pyemma.__path__[0], 'coordinates/tests/data/bpti_ca.pdb')
        self.feat= pyemma.coordinates.featurizer(self.MD_topology)

        self.ang_idxs = [[ii + jj for jj in range(3)] for ii in range(self.feat.topology.n_atoms - 3)]
        self.dih_idxs = [[ii + jj for jj in range(4)] for ii in range(self.feat.topology.n_atoms - 4)]

        self.feat.add_all()
        self.feat.add_distances_ca(excluded_neighbors=0)
        self.feat.add_angles(self.ang_idxs)
        self.feat.add_angles(self.ang_idxs, cossin=True)
        self.feat.add_dihedrals(self.dih_idxs)
        self.feat.add_dihedrals(self.dih_idxs, cossin=True)

    def tearDown(self):
        pass

    def test_atom_idxs_from_feature_xyz(self):
        ai = _bmutils.atom_idxs_from_feature(self.feat.active_features[0])
        assert np.allclose(np.repeat(np.arange(self.feat.topology.n_atoms),3), ai)

    def test_atom_idxs_from_feature_D_CA(self):
        ai = _bmutils.atom_idxs_from_feature(self.feat.active_features[1])
        ref = np.vstack(np.triu_indices(self.feat.topology.n_atoms, k=1)).T
        assert np.allclose(ref, ai),ai

    def test_atom_idxs_from_feature_ang(self):
        ai = _bmutils.atom_idxs_from_feature(self.feat.active_features[2])
        assert np.allclose(self.ang_idxs, ai)

    def test_atom_idxs_from_feature_ang_cossin(self):
        ai = _bmutils.atom_idxs_from_feature(self.feat.active_features[3])
        ref = np.tile(self.ang_idxs, 2).reshape(-1,3)
        assert np.allclose(ai, ref)

    def test_atom_idxs_from_feature_dih(self):
        ai = _bmutils.atom_idxs_from_feature(self.feat.active_features[4])
        assert np.allclose(self.dih_idxs, ai)
    def test_atom_idxs_from_feature_dih_cossin(self):
        ai = _bmutils.atom_idxs_from_feature(self.feat.active_features[5])
        ref = np.tile(self.dih_idxs, 2).reshape(-1,4)
        assert np.allclose(ref, ai)

class TestInputManipulationShaping(unittest.TestCase):

    def test_listify(self):
        input = [1,2,3,4]
        output = _bmutils.listify_if_not_list(input)
        # Doesn't do anything to a list
        assert isinstance(output, list)
        assert np.all([ii==jj for ii, jj in zip(input, output)])
        # Does do to a tuple
        input = (1,2,3,4)
        output = _bmutils.listify_if_not_list(input)
        assert isinstance(output, list)
        assert np.all([ii == jj for ii, jj in zip(input, output[0])])
        # Not if we exclude it, though
        assert not isinstance(_bmutils.listify_if_not_list(input, except_for_these_types=tuple), list)
        # And the exclusion is a list
        assert not isinstance(_bmutils.listify_if_not_list(input, except_for_these_types=[tuple, 'str']), list)
        input = 1
        output = _bmutils.listify_if_not_list(input)
        assert isinstance(output, list)
        assert output[0] == input

    def test_labelize(self):
        # Labels get constructed
        labels = _bmutils.labelize('test',[0,1])
        assert labels[0] == '$\mathregular{test_{0}}$', labels[0]
        assert labels[1] == '$\mathregular{test_{1}}$', labels[1]

        # Labels get accepted as they are and nothing happpens
        labels = _bmutils.labelize(['mylabel0', 'mylabel1'], [0, 1])
        assert labels[0] == "mylabel0", labels[0]
        assert labels[1] == "mylabel1", labels[1]

        # features objects can function
        feat = pyemma.coordinates.featurizer(os.path.join(pyemma.__path__[0], 'coordinates/tests/data/bpti_ca.pdb'))
        feat.add_all()
        labels = _bmutils.labelize(feat, [0,1])
        assert labels[0] == feat.describe()[0]
        assert labels[1] == feat.describe()[1]

    def test_superpose_list_of_geoms(self):
        geom = md.load(os.path.join(pyemma.__path__[0], 'coordinates/tests/data/bpti_ca.pdb'))

        # Nothing happens
        _bmutils.superpose_to_most_compact_in_list(False, [geom])
        # Something happens
        _bmutils.superpose_to_most_compact_in_list(True, [geom])
        # Something more specific happens
        _bmutils.superpose_to_most_compact_in_list('residue 10 or residue 11', [geom])
        _bmutils.superpose_to_most_compact_in_list(np.arange(10), [geom])

if __name__ == '__main__':
    unittest.main()
