__author__ = 'gph82'
# Since the complicated stuff and the potential for undetectable errors is in bmutils, heavy testing should be done there. This is an API test for checking
# input parsing and decision making (if then) inside the visualize API
import unittest
import numpy as np
import molpx
from glob import glob
from matplotlib.backend_bases import MouseEvent
import mdtraj as md
import matplotlib.pyplot as plt
plt.switch_backend('Agg') # allow tests


import nglview
from scipy.spatial import cKDTree as _cKDTree



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
            molpx._linkutils.update2Dlines(line, ii, ii)

    def test_force_exceptions(self):
        plt.plot(0, 0)
        line = plt.gca().lines[0]
        try:
            molpx._linkutils.update2Dlines(line,0,0)
        except AttributeError:
            pass

        setattr(line, "whatisthis", "non_existing_line")
        try:
            molpx._linkutils.update2Dlines(line,0,0)
        except TypeError:
            pass

class TestClickOnAxisListener(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        self.MD_trajectory_files = glob(molpx._molpxdir(join='notebooks/data/c-alpha_centered.stride.1000*xtc'))
        self.MD_topology_file = molpx._molpxdir(join='notebooks/data/bpti-c-alpha_centered.pdb')
        self.MD_topology = md.load(self.MD_topology_file).top
        self.MD_trajectories = [md.load(ff, top=self.MD_topology_file) for ff in self.MD_trajectory_files]
        self.MD_trajectory = self.MD_trajectories[0]

        self.pos = np.random.rand(self.MD_trajectory.n_frames, 2)
        self.kdtree = _cKDTree(self.pos)

    def just_runs(self, ngl_wdg, button=None):
        # Create the linked objects
        plt.plot(self.pos[:,0], self.pos[:,1])
        iax = plt.gca()

        # Prepare a mouse event in the middle of the plot
        x, y = np.array(iax.get_window_extent()).mean(0)

        # Prepare event
        lineh = iax.axhline(iax.get_ybound()[0])
        setattr(lineh, 'whatisthis', 'lineh')
        dot = iax.plot(self.pos[0, 0], self.pos[0, 1])[0]
        setattr(dot, 'whatisthis', 'dot')

        # Instantiate the ClickOnAxisListener and call it with the event
        return molpx._linkutils.ClickOnAxisListener(ngl_wdg, True,
                                             [lineh],
                                             iax, self.pos,
                                             [dot]
                                             )(MouseEvent(" ", plt.gcf().canvas, x,y,
                                                          button=button, key=None, step=0,
                                                          dblclick=False,
                                                          guiEvent=None))

    def test_just_runs(self):
        self.just_runs(nglview.show_mdtraj(self.MD_trajectory))

    def test_just_runs_sticky(self):
        # Create the linked objects (for sticky case, better use _sample
        ngl_wdg, __ = molpx.visualize.sample(self.pos, self.MD_trajectory, plt.gca(), sticky=True)
        self.just_runs(ngl_wdg, button=1)
        [self.just_runs(ngl_wdg, button=2) for ii in range(5)]

    def test_just_runs_recomputes_kdtree(self):
        plt.plot(self.pos[:, 0], self.pos[:, 1])
        iax = plt.gca()

        # Prepare a mouse event in the middle of the plot
        x, y = np.array(iax.get_window_extent()).mean(0)

        # Prepare event
        lineh = iax.axhline(iax.get_ybound()[0])
        setattr(lineh, 'whatisthis', 'lineh')
        dot = iax.plot(self.pos[0, 0], self.pos[0, 1])[0]
        setattr(dot, 'whatisthis', 'dot')
        CLAL = molpx._linkutils.ClickOnAxisListener(nglview.show_mdtraj(self.MD_trajectory), True,
                                             [lineh],
                                             iax, self.pos,
                                             [dot]
                                             )
        # Resize the figure
        CLAL.ax.figure.set_size_inches(1,1)
        # Send an event
        CLAL(MouseEvent(" ",  CLAL.ax.figure.canvas, x, y,
                    button=1, key=None, step=0,
                    dblclick=False,
                    guiEvent=None))

class TestChangeInNGLWidgetListener(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        self.MD_trajectory_files = glob(molpx._molpxdir(join='notebooks/data/c-alpha_centered.stride.1000*xtc'))
        self.MD_topology_file = molpx._molpxdir(join='notebooks/data/bpti-c-alpha_centered.pdb')
        self.MD_topology = md.load(self.MD_topology_file).top
        self.MD_trajectories = [md.load(ff, top=self.MD_topology_file) for ff in self.MD_trajectory_files]
        self.MD_trajectory = self.MD_trajectories[0]

        self.pos = np.random.rand(self.MD_trajectory.n_frames, 2)
        plt.plot(self.pos[:,0], self.pos[:,1])
        iax = plt.gca()
        # Prepare event
        self.lineh = iax.axhline(iax.get_ybound()[0])
        setattr(self.lineh, 'whatisthis', 'lineh')
        self.dot = iax.plot(self.pos[0, 0], self.pos[0, 1])[0]
        setattr(self.dot, 'whatisthis', 'dot')
        self.ngl_wdg = nglview.show_mdtraj(self.MD_trajectory)
    def test_just_runs(self):
        molpx._linkutils.ChangeInNGLWidgetListener(self.ngl_wdg, [self.lineh, self.dot], self.pos)({"new":0})

    def test_just_runs_past_last_frame(self):
        molpx._linkutils.ChangeInNGLWidgetListener(self.ngl_wdg, [self.lineh, self.dot], self.pos)({"new":self.pos.shape[0]+1,
                                                                                                    "old":1})

if __name__ == '__main__':
    unittest.main()
