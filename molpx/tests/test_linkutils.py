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
plt.switch_backend('Agg') # allow tests

from .test_bmutils import TestWithBPTIData
import molpx._linkutils
import nglview


class TestLinkAxWPos2NGLWidget(unittest.TestCase):

    @classmethod
    def setUpClass(self):
        self.MD_trajectory_files = glob(molpx._molpxdir(join='notebooks/data/c-alpha_centered.stride.1000*xtc'))
        self.MD_topology_file = molpx._molpxdir(join='notebooks/data/bpti-c-alpha_centered.pdb')
        self.MD_topology = md.load(self.MD_topology_file).top
        self.MD_trajectories = [md.load(ff, top=self.MD_topology_file) for ff in self.MD_trajectory_files]

        self.n_sample = 20
        self.pos = np.zeros((self.n_sample,2))
        self.pos[:,0] = np.linspace(0,1,self.n_sample)
        self.pos[:,1] = np.random.randn(self.n_sample)
        self.geom = self.MD_trajectories[0][:20]
        self.ngl_wdg = nglview.show_mdtraj(self.geom)

    def test_just_works(self):
        plt.figure()
        __ = molpx._linkutils.link_ax_w_pos_2_nglwidget(plt.gca(), self.pos, self.ngl_wdg)

    def test_just_works_bandwidth(self):
        plt.figure()
        __ = molpx._linkutils.link_ax_w_pos_2_nglwidget(plt.gca(), self.pos, self.ngl_wdg,
                                                        band_width=[0.1, .1])
    def test_just_works_exclude_coord(self):
        plt.figure()
        __ = molpx._linkutils.link_ax_w_pos_2_nglwidget(plt.gca(), self.pos, self.ngl_wdg,
                                                        exclude_coord=1)

    def test_force_exceptions(self):
        plt.figure()
        try:
            __ = molpx._linkutils.link_ax_w_pos_2_nglwidget(plt.gca(), self.pos, self.ngl_wdg,
                                                            dot_color=2)
        except TypeError:
            pass

        plt.figure()
        try:
            pos_rnd = np.random.randn(self.n_sample, 2)
            __ = molpx._linkutils.link_ax_w_pos_2_nglwidget(plt.gca(), pos_rnd, self.ngl_wdg,
                                                            band_width=[.1, .1])
        except ValueError:
            pass

    def test_just_works_radius(self):
        plt.figure()
        __ = molpx._linkutils.link_ax_w_pos_2_nglwidget(plt.gca(), self.pos, self.ngl_wdg,
                                                        band_width=[.1, .1],
                                                        radius=True)
class TestGeometryInNGLWidget(unittest.TestCase):

    # TODO abstact this to a test class
    @classmethod
    def setUpClass(self):
        self.MD_trajectory_files = glob(molpx._molpxdir(join='notebooks/data/c-alpha_centered.stride.1000*xtc'))
        self.MD_topology_file = molpx._molpxdir(join='notebooks/data/bpti-c-alpha_centered.pdb')
        self.MD_topology = md.load(self.MD_topology_file).top
        self.MD_trajectories = [md.load(ff, top=self.MD_topology_file) for ff in self.MD_trajectory_files]

        self.n_sample = 20
        self.pos = np.zeros((self.n_sample,2))
        self.pos[:,0] = np.linspace(0,1,self.n_sample)
        self.pos[:,1] = np.random.randn(self.n_sample)
        self.geom = self.MD_trajectories[0][:20]
        self.ngl_wdg = nglview.show_mdtraj(self.geom)

    def test_just_works(self):
        __ = molpx._linkutils.GeometryInNGLWidget(self.geom, self.ngl_wdg)
        __ = molpx._linkutils.GeometryInNGLWidget(self.geom, self.ngl_wdg, n_small=500)


    def test_quickhands(self):
        giw = molpx._linkutils.GeometryInNGLWidget(self.geom, self.ngl_wdg)
        assert giw.is_empty()
        assert giw.have_repr == []
        assert ~giw.all_reps_are_on()
        assert giw.all_reps_are_off()
        assert ~giw.any_rep_is_on()
        assert ~giw.is_visible()


    def test_show_just_runs_and_changes_quickhands(self):
        giw = molpx._linkutils.GeometryInNGLWidget(self.geom, self.ngl_wdg)
        giw.show()
        assert ~giw.is_empty()
        assert len(giw.have_repr) == 1
        assert giw.all_reps_are_on()
        assert ~giw.all_reps_are_off()
        assert giw.any_rep_is_on()
        assert giw.is_visible()

    def test_show_and_hide_just_runs_and_changes_quickhands(self):
        # We initialize with two frames, it's easy to test the ends
        giw = molpx._linkutils.GeometryInNGLWidget(self.geom[:2], self.ngl_wdg)
        giw.show()
        giw.hide()
        assert ~giw.is_empty()
        assert len(giw.have_repr) == 1 #should still be one
        assert ~giw.all_reps_are_on() # but it´s off
        assert giw.all_reps_are_off()
        assert ~giw.any_rep_is_on()
        assert ~giw.is_visible()
        assert giw.any_rep_is_off()
        # Now we show again
        giw.show() # Turn on the one we had already, does not change the have_repr
        assert len(giw.have_repr)==1
        giw.show()
        assert len(giw.have_repr)==2
        # Turn on a new one, which isn't there nothing should happen
        [giw.show() for ii in range(10)]
        # Now we hide many times, should arrive at the end without anything happening
        [giw.hide() for ii in range(10)]

class TestUpdate2DLines(unittest.TestCase):

    def test_all(self):
        plt.plot(0,0)
        line = plt.gca().lines[0]
        for ii, attr in enumerate(['lineh', 'linev', 'dot']):
            setattr(line,"whatisthis",attr)
            molpx.visualize._linkutils.update2Dlines(line, ii, ii)

    def test_force_exceptions(self):
        plt.plot(0, 0)
        line = plt.gca().lines[0]
        try:
            molpx.visualize._linkutils.update2Dlines(line,0,0)
        except AttributeError:
            pass

        setattr(line, "whatisthis", "non_existing_line")
        try:
            molpx.visualize._linkutils.update2Dlines(line,0,0)
        except TypeError:
            pass



if __name__ == '__main__':
    unittest.main()
