__author__ = 'gph82'
# Since the complicated stuff and the potential for undetectable errors is in bmutils, heavy testing should be done there. This is an API test for checking
# input parsing and decision making (if then) inside the visualize API
import unittest
import numpy as np
import molpx
from glob import glob
from matplotlib.backend_bases import MouseEvent
import mdtraj as md
#plt.switch_backend('Agg') # allow tests


import nglview
from numpy.testing import assert_raises
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
        import matplotlib.pyplot as plt
        self.plt = plt

    def test_just_works(self):
        self.plt.figure()
        __ = molpx._linkutils.link_ax_w_pos_2_nglwidget(self.plt.gca(), self.pos, self.ngl_wdg)

    def test_just_works_bandwidth(self):
        self.plt.figure()
        __ = molpx._linkutils.link_ax_w_pos_2_nglwidget(self.plt.gca(), self.pos, self.ngl_wdg,
                                                        band_width=[0.1, .1])
    def test_just_works_exclude_coord(self):
        self.plt.figure()
        __ = molpx._linkutils.link_ax_w_pos_2_nglwidget(self.plt.gca(), self.pos, self.ngl_wdg,
                                                        exclude_coord=1)

    def test_force_exceptions(self):
        self.plt.figure()
        assert_raises(TypeError, molpx._linkutils.link_ax_w_pos_2_nglwidget, self.plt.gca(), self.pos, self.ngl_wdg,
                                                            dot_color=2)

        self.plt.figure()
        pos_rnd = np.random.randn(self.n_sample, 2)
        assert_raises(ValueError, molpx._linkutils.link_ax_w_pos_2_nglwidget, self.plt.gca(), pos_rnd, self.ngl_wdg,
                                                            band_width=[.1, .1])

    def test_just_works_radius(self):
        self.plt.figure()
        __ = molpx._linkutils.link_ax_w_pos_2_nglwidget(self.plt.gca(), self.pos, self.ngl_wdg,
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

class TestContactInNGLWidget(unittest.TestCase):

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

    def test_just_works(self):
        self.ngl_wdg = nglview.show_mdtraj(self.geom)
        molpx._linkutils.ContactInNGLWidget(self.ngl_wdg, [0, 1], 0)

    def _test_right_atoms_are_represented(self):
        self.ngl_wdg = nglview.show_mdtraj(self.geom)
        ctcNwid = molpx._linkutils.ContactInNGLWidget(self.ngl_wdg, [0, 1], 0)

        # TODO cannot be tested since the representation is yet to be shown!
        # we have to find this out!
        #irepr = [value for value in self.ngl_wdg._ngl_repr_dict["0"].items() if value["type"] == "distance"]
        #assert len(irepr)==1
        #assert np.allclose(np.sort(irepr["params"]["atomPair"]), [0,1])

    def test_method_show_and_hide(self):
        self.ngl_wdg = nglview.show_mdtraj(self.geom)
        self.ngl_wdg.display()
        ctcInwid = molpx._linkutils.ContactInNGLWidget(self.ngl_wdg, [0, 1], 0, verbose=True)
        ctcInwid.show()
        ctcInwid.hide() # Hide isn't really doing anything, since nothing is really shown
        # TODO find out how to force the presentation on nglview from terminal

    def _test_quickhands(self):
        giw = molpx._linkutils.GeometryInNGLWidget(self.geom, self.ngl_wdg)
        assert giw.is_empty()
        assert giw.have_repr == []
        assert ~giw.all_reps_are_on()
        assert giw.all_reps_are_off()
        assert ~giw.any_rep_is_on()
        assert ~giw.is_visible()


    def _test_show_just_runs_and_changes_quickhands(self):
        giw = molpx._linkutils.GeometryInNGLWidget(self.geom, self.ngl_wdg)
        giw.show()
        assert ~giw.is_empty()
        assert len(giw.have_repr) == 1
        assert giw.all_reps_are_on()
        assert ~giw.all_reps_are_off()
        assert giw.any_rep_is_on()
        assert giw.is_visible()

    def _test_show_and_hide_just_runs_and_changes_quickhands(self):
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
        import matplotlib.pyplot as plt
        plt.plot(0,0)
        line = plt.gca().lines[0]
        for ii, attr in enumerate(['lineh', 'linev', 'dot']):
            setattr(line,"whatisthis",attr)
            molpx._linkutils.update2Dlines(line, ii, ii)

    def test_force_exceptions(self):
        import matplotlib.pyplot as plt
        plt.plot(0, 0)
        line = plt.gca().lines[0]
        # This passes interactively in the console...?
        try:
            assert_raises(AttributeError, molpx._linkutils.update2Dlines, line,0,0)
        except AssertionError:
            pass

        setattr(line, "whatisthis", "non_existing_line")
        assert_raises(TypeError, molpx._linkutils.update2Dlines, line,0,0)

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
        import matplotlib.pyplot as plt
        self.plt = plt

    def just_runs(self, ngl_wdg, button=None):
        # Create the linked objects
        self.plt.plot(self.pos[:,0], self.pos[:,1])
        iax = self.plt.gca()

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
                                             )(MouseEvent(" ", self.plt.gcf().canvas, x,y,
                                                          button=button, key=None, step=0,
                                                          dblclick=False,
                                                          guiEvent=None))

    def test_just_runs(self):
        self.just_runs(nglview.show_mdtraj(self.MD_trajectory))

    def test_just_runs_sticky(self):
        # Create the linked objects (for sticky case, better use _sample
        ngl_wdg, __ = molpx.visualize.sample(self.pos, self.MD_trajectory, self.plt.gca(), sticky=True)
        self.just_runs(ngl_wdg, button=1)
        [self.just_runs(ngl_wdg, button=2) for ii in range(5)]

    def test_just_runs_recomputes_kdtree(self):
        self.plt.plot(self.pos[:, 0], self.pos[:, 1])
        iax = self.plt.gca()

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

        # Change axis lims to trigger recomputation
        old_xlim = iax.get_xlim()
        new_xmin = old_xlim[0]+abs(np.diff(iax.get_xlim())*.10)
        iax.set_xlim(new_xmin, old_xlim[1])
        # Recompute the position of the new click
        x, y = np.array(iax.get_window_extent()).mean(0)
        # Send an event
        CLAL(MouseEvent(" ", CLAL.ax.figure.canvas, x, y,
                        button=1, key=None, step=0,
                        dblclick=False,
                        guiEvent=None))

    def test_click_w_contacts(self):
        # Plot contact map
        ctcs, idxs = md.compute_contacts(self.MD_trajectory)
        ctcs = md.geometry.squareform(ctcs, idxs)
        self.plt.imshow(ctcs[0])
        iax = self.plt.gca()

        # Create matching "positions" array
        nres = ctcs.shape[-1]
        positions = np.vstack(np.unravel_index(range(nres ** 2), (nres, nres))).T

        # Create widget and monkey-patch its _CtcsInWid attribute
        ngl_wdg = nglview.show_mdtraj(self.MD_trajectory)
        ngl_wdg._CtcsInWid = [molpx._linkutils.ContactInNGLWidget(ngl_wdg, [0, 1], 0)]

        # Create the CLA object linking the wid and the axis via "positions"
        CLAL = molpx._linkutils.ClickOnAxisListener(ngl_wdg, True,
                                             [],
                                             iax, positions,
                                             []
                                             )

        # Get the left, uppermost pixel of the image (matplotlib voodoo,
        # see https://matplotlib.org/users/transforms_tutorial.html
        x, y = iax.get_window_extent().x0, iax.get_window_extent().y1

        # Before the click, this should just pass
        CLAL.remove_last_contacts()

        # Now we instantiate a mouseclick on the first contact (0,1)
        ME = MouseEvent(" ", CLAL.ax.figure.canvas, x, y,
                        button=1, key=None, step=0,
                        dblclick=False,
                        guiEvent=None)

        # Make sure the simulated click was actually on the canvas
        assert CLAL.ax.get_window_extent().contains(ME.x, ME.y)

        # Send the mouseclick to the CLAL
        CLAL(ME)

        # Assert that a rectangle was created
        assert CLAL.list_of_rects[0] is not None, CLAL.list_of_rects

        # This should remove that rectangle and the contacts
        CLAL.remove_last_contacts()
        # Check that "remove" worked:
        assert ~ngl_wdg._CtcsInWid[0].shown

        # Now click again one on left and one on right click:
        ME = MouseEvent(" ", CLAL.ax.figure.canvas, x, y,
                        button=1, key=None, step=0,
                        dblclick=False,
                        guiEvent=None)
        ME_right = MouseEvent(" ", CLAL.ax.figure.canvas, x, y,
                        button=2, key=None, step=0,
                        dblclick=False,
                        guiEvent=None)
        CLAL(ME)
        CLAL(ME_right)



class TestChangeInNGLWidgetListener(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        self.MD_trajectory_files = glob(molpx._molpxdir(join='notebooks/data/c-alpha_centered.stride.1000*xtc'))
        self.MD_topology_file = molpx._molpxdir(join='notebooks/data/bpti-c-alpha_centered.pdb')
        self.MD_topology = md.load(self.MD_topology_file).top
        self.MD_trajectories = [md.load(ff, top=self.MD_topology_file) for ff in self.MD_trajectory_files]
        self.MD_trajectory = self.MD_trajectories[0]

        self.pos = np.random.rand(self.MD_trajectory.n_frames, 2)
        from matplotlib import pyplot as plt
        self.plt = plt
        self.plt.plot(self.pos[:,0], self.pos[:,1])
        iax = self.plt.gca()
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

    def test_update_contact_map(self):
        from matplotlib import pyplot as _plt
        contact_map = md.geometry.squareform(*md.compute_contacts(self.MD_trajectories[0]))
        _plt.figure()
        iax = _plt.gca()
        self.ngl_wdg._MatshowData = {"image" : iax.matshow(contact_map[0]),
                                     "data"  : contact_map}
        CINL = molpx._linkutils.ChangeInNGLWidgetListener(self.ngl_wdg, [], None)
        # Run trough all available contact maps
        for ii in range(len(contact_map)):
            CINL({"new":ii})


if __name__ == '__main__':
    unittest.main()
