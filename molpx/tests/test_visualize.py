__author__ = 'gph82'
# Since the complicated stuff and the potential for undetectable errors is in bmutils, heavy testing should be done there. This is an API test for checking
# input parsing and decision making (if then) inside the visualize API
import unittest
import numpy as np
import molpx
from glob import glob
from molpx import visualize, _bmutils
import mdtraj as md
import matplotlib.pyplot as plt
import nglview
from numpy.testing import assert_raises
import pyemma

from .test_bmutils import TestWithBPTIData

class TestTrajInputs(TestWithBPTIData):

    @classmethod
    def setUpClass(self):
        TestWithBPTIData.setUpClass()

        self.ala2_topology_file = molpx._molpxdir(join='notebooks/data/ala2.pdb')
        self.metad_trajectory_files = glob(molpx._molpxdir(join='notebooks/data/ala2.meta*.xtc'))
        self.metad_colvar_files = glob(molpx._molpxdir(join='notebooks/data/ala2.meta.CV.*txt'))

    def test_simplest_inputs_memory(self):
        visualize.traj(self.MD_trajectories, self.MD_topology, self.Ys)

    def test_simplest_inputs_memory_stride(self):
            visualize.traj(self.MD_trajectories, self.MD_topology, self.Ys, stride=2)

    def test_simplest_inputs_memory_small_max_frames(self):
            visualize.traj(self.MD_trajectories, self.MD_topology, self.Ys, max_frames=3)

    def test_simplest_inputs_disk(self):
        visualize.traj(self.MD_trajectory_files, self.MD_topology_file, self.projected_files_npy)
        visualize.traj(self.MD_trajectory_files, self.MD_topology_file, [ifile.replace('.npy', '.dat') for ifile in self.projected_files_npy])

    def test_simplest_inputs_memory_and_proj(self):
        visualize.traj(self.MD_trajectories, self.MD_topology, self.Ys, projection=self.tica)

    def test_simplest_inputs_memory_and_proj_just_matrix(self):
        visualize.traj(self.MD_trajectories, self.MD_topology, self.Ys,
                       projection=self.tica.feature_TIC_correlation)

    def test_simplest_inputs_memory_just_one_row(self):
        visualize.traj(self.MD_trajectories, self.MD_topology, self.Ys,
                       traj_selection=0,
                       proj_idxs=0)

    def test_simplest_inputs_memory_FES(self):
        visualize.traj(self.MD_trajectories, self.MD_topology, self.Ys,
                       plot_FES=True
                      )

    def test_fail_without_top(self):
        # This should fail
        try:
            visualize.traj(self.MD_trajectory_files, "unexisting_file", self.Ys)
        except (OSError, IOError):
            pass
        # This should pass
        visualize.traj(self.MD_trajectories, "unexisting_file", self.Ys)

    def test_listify_inputs(self):
        visualize.traj(self.MD_trajectories[0], self.MD_topology, self.Ys[0])

    def test_listify_params(self):
        visualize.traj(self.MD_trajectories, self.MD_topology, self.Ys, proj_idxs=1)

    def test_plotting_params(self):
        visualize.traj(self.MD_trajectories, self.MD_topology, self.Ys, sharey_traj=True)

    def test_fail_on_FES(self):
        try:
            fake_4D_Ys = [np.hstack((iY, iY)) for iY in self.Ys]
            visualize.traj(self.MD_trajectories, self.MD_topology,
                           fake_4D_Ys, sharey_traj=True,
                           plot_FES=True, proj_idxs=[0,1,2])
        except Exception as e:
            assert isinstance(e,NotImplementedError)

    def test_weights_on_biased_FES(self):
        weights = [np.exp(np.loadtxt(iw)[:,7]) for iw in self.metad_colvar_files]
        visualize.FES(self.metad_trajectory_files, self.ala2_topology_file, self.metad_colvar_files,
                      proj_idxs=[1,2], weights=weights)

class TestFES(TestWithBPTIData):

    @classmethod
    def setUpClass(self):
        TestWithBPTIData.setUpClass()

    def test_just_works_min_input_disk(self):
        molpx.visualize.FES(self.MD_trajectory_files,
                            self.MD_topology_file,
                            self.projected_files_npy)

    def test_just_works_min_input_memory(self):
        molpx.visualize.FES(self.MD_trajectories,
                            self.MD_topology,
                            self.Ys)

    def test_overlays(self):
        molpx.visualize.FES(self.MD_trajectories,
                            self.MD_topology,
                            self.Ys,
                            n_overlays=5)

    def test_1D(self):
        molpx.visualize.FES(self.MD_trajectories,
                            self.MD_topology,
                            self.Ys,
                            proj_idxs=[0])

    def test_with_weigths(self):
        weights = [np.ones(len(iY)) for iY in self.Ys]
        molpx.visualize.FES(self.MD_trajectories,
                            self.MD_topology,
                            self.Ys, weights=weights)

class TestNGLWidgetWrapper(unittest.TestCase):

    def setUp(self):
        self.MD_file = molpx._molpxdir(join='notebooks/data/bpti-c-alpha_centered.pdb')
        self.MD_geom = md.load(self.MD_file)

    def test_widget_wrapper_w_None(self):
        iwd = molpx.visualize._nglwidget_wrapper(None)

    def test_widget_wrapper_w_file(self):
        iwd = molpx.visualize._nglwidget_wrapper(self.MD_file)

    def test_widget_wrapper_w_instantiated_wdg(self):
        iwd = molpx.visualize._nglwidget_wrapper(self.MD_file)
        molpx.visualize._nglwidget_wrapper(self.MD_geom, ngl_wdg=iwd)

def test_colors():
    _bmutils.matplotlib_colors_no_blue()
class TestSample(TestWithBPTIData):

    @classmethod
    def setUpClass(self):
        TestWithBPTIData.setUpClass()
        self.n_sample = 4
        self.pos = np.zeros((self.n_sample, 2))
        self.pos[:,0] = np.linspace(0,1,num=self.n_sample)
        self.pos[:,1] = np.random.rand(self.n_sample)
        self.geom = self.MD_trajectories[0][:self.n_sample]

    @classmethod
    def tearDownClass(self):
        TestWithBPTIData.tearDownClass()

    def test_sample_not_sticky_just_works(self):
        plt.figure()
        __ = molpx.visualize.sample(self.pos, self.geom, plt.gca())

    def test_sample_not_sticky_just_works_with_path(self):
        plt.figure()
        __ = molpx.visualize.sample(self.pos, self.geom, plt.gca(), plot_path=True)

    def test_sample_not_sticky_just_works_with_projection(self):
        plt.figure()
        __ = molpx.visualize.sample(self.pos, self.geom, plt.gca(), projection=self.tica)

    def test_sample_not_sticky_smooth(self):
        plt.figure()
        __ = molpx.visualize.sample(self.pos, self.geom, plt.gca(),
                                    n_smooth=1)

    def test_sample_sticky_just_works_one_geom(self):
        plt.figure()
        __ = molpx.visualize.sample(self.pos, self.geom, plt.gca(), sticky=True)

    def test_sample_sticky_just_works_list_geom(self):
        plt.figure()
        __ = molpx.visualize.sample(self.pos, [self.geom, self.geom[::-1]], plt.gca(), sticky=True)

    def test_sample_sticky_just_works_list_geom_dont_orient(self):
        plt.figure()
        __ = molpx.visualize.sample(self.pos, [self.geom, self.geom[::-1]], plt.gca(),
                                    superpose=False,
                                    sticky=True)

    def test_sample_sticky_just_works_list_geom_colors(self):
        plt.figure()
        __ = molpx.visualize.sample(self.pos, [self.geom, self.geom[::-1]], plt.gca(),
                                    color_list=['r', 'b', 'g', 'magenta'],
                                    sticky=True)

        __ = molpx.visualize.sample(self.pos, [self.geom, self.geom[::-1]], plt.gca(),
                                    color_list='rand',
                                    sticky=True)

        try:
            __ = molpx.visualize.sample(self.pos, [self.geom, self.geom[::-1]], plt.gca(),
                                        color_list=1,
                                        sticky=True)
        except TypeError:
            pass

    def test_sample_sticky_just_works_list_geom_small_molecule(self):
        __ = molpx.visualize.sample(self.pos, [self.geom.atom_slice(np.arange(5)),
                                               self.geom[::-1].atom_slice(np.arange(5))],
                                    plt.gca(),
                                    color_list=['r', 'b', 'g', 'magenta'],
                                    sticky=True)

class TestCorrelations(TestWithBPTIData):

    @classmethod
    def setUpClass(self):
        TestWithBPTIData.setUpClass()

    def test_correlations_input_tica(self):
        visualize.correlations(self.tica)

    def test_correlations_input_pca(self):
        visualize.correlations(self.pca)

    def test_correlations_input_feat(self):
        visualize.correlations(self.feat)

    def test_correlations_input_tica_and_geoms(self):
        visualize.correlations(self.tica, geoms=self.MD_trajectories[0])

    def test_correlations_input_warn(self):
        visualize.correlations(self.tica.feature_TIC_correlation, geoms=self.MD_trajectories[0])

    def test_correlations_inputs_verbose(self):
        visualize.correlations(self.tica, verbose=True)

    def test_correlations_inputs_verbose_and_widget(self):
        visualize.correlations(self.tica,
                               geoms=self.MD_trajectories[0],
                               verbose=True)

    def test_correlations_inputs_color_list_parsing(self):
        visualize.correlations(self.tica, proj_color_list=['green'])

    def test_correlations_inputs_FAIL_color_list_parsing(self):
        try:
            visualize.correlations(self.tica, proj_color_list='green')
        except TypeError:
            pass

class TestMSM(TestWithBPTIData):
    @classmethod
    def setUpClass(self):
        TestWithBPTIData.setUpClass()
        self.cl = pyemma.coordinates.cluster_kmeans([iY[:, :2] for iY in self.Ys], 5)
        self.iMSM = pyemma.msm.estimate_markov_model(self.cl.dtrajs, 1)
        self.iHMM = self.iMSM.coarse_grain(3)

    @classmethod
    def tearDownClass(self):
        TestWithBPTIData.tearDownClass()

    def test_just_runs_MSM(self):
        visualize.MSM(self.iMSM, self.MD_trajectories)
        # More overlays
        visualize.MSM(self.iMSM, self.MD_trajectories, n_overlays=10)

    def test_just_runs_HMM(self):
        visualize.MSM(self.iHMM, self.MD_trajectories)
        visualize.MSM(self.iHMM, self.MD_trajectories, n_overlays=10)
        visualize.MSM(self.iHMM, self.MD_trajectories, sharpen=True)

    def test_just_runs_position_input(self):
        visualize.MSM(self.iMSM, self.MD_trajectories, pos=self.cl.clustercenters)
        visualize.MSM(self.iHMM, self.MD_trajectories, pos=self.cl.clustercenters)

        visualize.MSM(self.iHMM, self.MD_trajectories, pos=self.cl.clustercenters, sharpen=True)
        visualize.MSM(self.iHMM, self.MD_trajectories, pos=self.cl.clustercenters[:self.iHMM.nstates])
        # test assert
        assert_raises(TypeError, visualize.MSM, self.iMSM, self.MD_trajectories, pos=['a'])
        assert_raises(AssertionError, visualize.MSM, self.iMSM, self.MD_trajectories, pos=self.cl.clustercenters[:-1])

    def test_source_inputs(self):
        visualize.MSM(self.iMSM, self.MD_trajectory_files, top=self.MD_topology_file)
        visualize.MSM(self.iMSM, self.MD_trajectory_files, top=self.MD_topology)
        visualize.MSM(self.iMSM, self.source)


    def test_returns_the_right_things_MSM(self):
        # Returning the right things should be guaranteed by all the
        # lower-level methods, which are also unit-tested. Still, here we go
        mpxbox = visualize.MSM(self.iMSM, self.MD_trajectories, n_overlays=3)
        # We re-featurize, re-tic-transform, and re-cl-assign the output geoms
        for igeoms in mpxbox.linked_mdgeoms:
            out_assign = self.cl.assign(self.tica.transform(self.feat.transform(igeoms))[:,:2])
            assert np.allclose(out_assign, np.arange(self.cl.n_clusters))

        # Now with sticky
        mpxbox = visualize.MSM(self.iMSM, self.MD_trajectories, n_overlays=3, sticky=True)
        # We re-featurize, re-tic-transform, and re-cl-assign the output geoms
        for igeoms in mpxbox.linked_mdgeoms:
            out_assign = self.cl.assign(self.tica.transform(self.feat.transform(igeoms))[:, :2])
            assert np.allclose(out_assign, np.arange(self.cl.n_clusters))

    def test_returns_the_right_things_HMSM_sharpen(self):
        # Returning the right things should be guaranteed by all the
        # lower-level methods, which are also unit-tested. Still, here we go
        mpxbox = visualize.MSM(self.iHMM, self.MD_trajectories, n_overlays=3, sharpen=True)
        # We re-featurize, re-tic-transform, and re-cl-assign the output geoms
        out_set = np.argmax(self.iHMM.observation_probabilities, axis=1)
        for igeoms in mpxbox.linked_mdgeoms:
            out_assign = self.cl.assign(self.tica.transform(self.feat.transform(igeoms))[:,:2])
            assert np.allclose(out_assign, out_set)


class TestFeature(TestWithBPTIData):
    @classmethod
    def setUpClass(self):
        TestWithBPTIData.setUpClass()

    @classmethod
    def tearDownClass(self):
        TestWithBPTIData.tearDownClass()

    def test_feature(self):
        plt.figure()
        iwd = nglview.show_mdtraj(self.MD_trajectories[0])
        visualize.feature(self.feat, iwd)

    def test_feature_color_list(self):
        plt.figure()
        iwd = nglview.show_mdtraj(self.MD_trajectories[0])
        visualize.feature(self.feat, iwd,
                          idxs=[0,1],
                          color_list=['blue'])
        try:
            visualize.feature(self.feat, iwd,
                              idxs=[0,1],
                              color_list='blue')
        except TypeError:
            pass

class Contacts(TestWithBPTIData):
    @classmethod
    def setUpClass(self):
        TestWithBPTIData.setUpClass()
        self.geom = self.MD_trajectories[0]
        ctcs, res_idxs = md.compute_contacts(self.geom)
        self.ctcs = md.geometry.squareform(ctcs, res_idxs)

    def test_just_runs(self):
        visualize.contacts(self.ctcs, self.geom)

    def test_one_ctcframe(self):
        # this should fail
        try:
            visualize.contacts(self.ctcs.mean(0), self.geom)
        except AssertionError:
            pass
        # This should pass
        visualize.contacts(self.ctcs.mean(0), self.geom, average=True)

    def test_raises(self):
        assert_raises(NotImplementedError, visualize.contacts, self.ctcs.mean(0), self.geom, average=True, residue_indices=[1,2,3])

class TestBoxMe(unittest.TestCase):

    def test_just_runs_and_exits_gracefully(self):
        from matplotlib.widgets import AxesWidget
        from ipywidgets.widgets import HBox
        ifig = plt.figure()
        iax = plt.gca()
        nglwdg = nglview.demo()
        mplw = AxesWidget(iax)

        box_out = visualize._box_me((ifig, iax, nglwdg,
                                     mplw
                                     ))

        assert isinstance(box_out, HBox)


        assert None is visualize._box_me((ifig, iax, nglwdg,
                                     mplw, 1
                                     ))


if __name__ == '__main__':
    unittest.main()
