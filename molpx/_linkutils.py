from __future__ import print_function
import numpy as _np

from matplotlib.widgets import AxesWidget as _AxesWidget
from matplotlib.colors import is_color_like as _is_color_like
from matplotlib.axes import Axes as _mplAxes
from matplotlib.figure import Figure as _mplFigure
from IPython.display import display as _ipydisplay

from pyemma.util.types import is_int as _is_int
from scipy.spatial import cKDTree as _cKDTree

from ._bmutils import get_ascending_coord_idx

from mdtraj import Trajectory as _mdTrajectory
from nglview import NGLWidget as _NGLwdg

from ipywidgets import HBox as _HBox, VBox as _VBox

def pts_per_axis_unit(mplax, pt_per_inch=72):
    r"""
    Return how many pt per axis unit of a given maptplotlib axis a figure has

    Parameters
    ----------

    mplax : :obj:`matplotlib.axes._subplots.AxesSubplot`

    pt_per_inch : how many points are in an inch (this number should not change)

    Returns
    --------

    pt_per_xunit, pt_per_yunit

    """

    # matplotlib voodoo
    # Get bounding box
    bbox = mplax.get_window_extent().transformed(mplax.get_figure().dpi_scale_trans.inverted())

    span_inch = _np.array([bbox.width, bbox.height], ndmin=2).T

    span_units = [mplax.get_xlim(), mplax.get_ylim()]
    span_units = _np.diff(span_units, axis=1)

    inch_per_unit = span_inch / span_units
    return inch_per_unit * pt_per_inch



def update2Dlines(iline, x, y):
    """
    provide a common interface to update objects on the plot to a new position (x,y) depending
    on whether they are hlines, vlines, dots etc

    Parameters
    ----------

    iline: :obj:`matplotlib.lines.Line2D` object

    x : float with new position

    y : float with new position

    tested:False
    """
    # TODO FIND OUT A CLEANER WAY TO DO THIS (dict or class)

    if not hasattr(iline,'whatisthis'):
        raise AttributeError("This method will only work if iline has the attribute 'whatsthis'")
    else:
        # TODO find cleaner way of distinguishing these 2Dlines
        if iline.whatisthis in ['dot']:
            iline.set_xdata((x))
            iline.set_ydata((y))
        elif iline.whatisthis in ['lineh']:
            iline.set_ydata((y,y))
        elif iline.whatisthis in ['linev']:
            iline.set_xdata((x,x))
        else:
            # TODO: FIND OUT WNY EXCEPTIONS ARE NOT BEING RAISED
            raise TypeError("what is this type of 2Dline?")


class ClickOnAxisListener(object):
    def __init__(self, ngl_wdg, kdtree, crosshairs, showclick_objs, ax, pos,
                 list_mpl_objects_to_update):
        self.ngl_wdg = ngl_wdg
        self.kdtree = kdtree
        self.crosshairs = crosshairs
        self.showclick_objs = showclick_objs
        self.ax = ax
        self.pos = pos
        self.list_mpl_objects_to_update = list_mpl_objects_to_update
        self.list_of_dots = [None]*self.pos.shape[0]

    def __call__(self, event):

        data = [event.xdata, event.ydata]
        # Was the click inside the bounding box?
        if self.ax.get_window_extent().contains(event.x, event.y):
            if self.crosshairs:
                for iline in self.showclick_objs:
                    update2Dlines(iline, event.xdata, event.ydata)

            _, index = self.kdtree.query(x=data, k=1)
            for idot in self.list_mpl_objects_to_update:
                update2Dlines(idot, self.pos[index, 0], self.pos[index, 1])

            self.ngl_wdg.isClick = True
            if hasattr(self.ngl_wdg, '_GeomsInWid'):
                # We're in a sticky situation
                if event.button == 1:
                    # Pressed left
                    self.ngl_wdg._GeomsInWid[index].show()
                    if self.list_of_dots[index] is None:
                        # Plot and store the dot in case there wasn't
                        self.list_of_dots[index] = self.ax.plot(self.pos[index, 0], self.pos[index, 1], 'o',
                                                                c=self.ngl_wdg._GeomsInWid[index].color_dot, ms=7)[0]
                elif event.button in [2, 3]:
                    #  Pressed right or middle
                    self.ngl_wdg._GeomsInWid[index].hide()
                    # Delete dot if the geom is not visible anymore
                    if not self.ngl_wdg._GeomsInWid[index].is_visible() and self.list_of_dots[index] is not None:
                        self.list_of_dots[index].remove()
                        self.list_of_dots[index] = None
            else:
                # We're not sticky, just go to the frame
                self.ngl_wdg.frame = index

class MolPXBox(object):
    r"""
    Class created to be the parent class of MolPXHBox and MolPXVBox, which inherit from
    MolPXBox and the ipywidget classes HBox and VBox (*args and **kwargs are for these)

    The sole purpose of this class is to  avoid monkey-patching elsewhere in the code,
     this class creates them as empty lists on instantiation.

    It also implements two methods:

    * self.display (=IPython.display(self)

    * append_if_existing
    """
    def __init__(self, *args, **kwargs):
        self.linked_axes = []
        self.linked_mdgeoms = []
        self.linked_ngl_wdgs = []
        self.linked_data_arrays = []
        self.linked_ax_wdgs = []
        self.linked_figs = []

    def display(self):
        _ipydisplay(self)

    def append_if_existing(self, args0, startswith_arg="linked_"):
        r"""
        args0  is the tuple containing all widgets to be included in the MolPXBox
        this tuple can contain itself other MolPXWidget
        so we iterate through them and appending linked stuff
        """
        for iarg in args0:
            for attrname in dir(iarg):
                if attrname.startswith(startswith_arg) and len(iarg.__dict__[attrname]) != 0:
                    self.__dict__[attrname] += iarg.__dict__[attrname]

def auto_append_these_mpx_attrs(iobj, *attrs):
    r""" The attribute s name is automatically derived
    from the attribute s type via a type:name dictionary

    *attrs : any number of unnamed objects of the types in type2attrname.
             If the object type is a list, it will be flattened prior to attempting
    """

    attrs_flat_list = []
    for sublist in attrs:
        if isinstance(sublist, list):
            for item in sublist:
                attrs_flat_list.append(item)
        else:
            attrs_flat_list.append(sublist)

    # Go through the arguments and assign them an attrname according to their types
    for iattr in attrs_flat_list:
        for attrname, itype in type2attrname.items():
            if isinstance(iattr, itype):
                iobj.__dict__[attrname].append(iattr)
                break

class MolPXHBox(_HBox, MolPXBox):
    def __init__(self, *args, **kwargs):
        super(MolPXHBox, self).__init__(*args, **kwargs)
        self.append_if_existing(args[0])

class MolPXVBox(_VBox, MolPXBox):
    def __init__(self, *args, **kwargs):
        super(MolPXVBox, self).__init__(*args, **kwargs)
        self.append_if_existing(args[0])

type2attrname = {"linked_axes": _mplAxes,
                 "linked_mdgeoms": _mdTrajectory,
                 "linked_ngl_wdgs": _NGLwdg,
                 "linked_data_arrays": _np.ndarray,
                 "linked_ax_wdgs": _AxesWidget,
                 "linked_figs": _mplFigure,
                 }

class ChangeInNGLWidgetListener(object):

    def __init__(self, ngl_wdg, list_mpl_objects_to_update, pos):
        self.ngl_wdg = ngl_wdg
        self.list_mpl_objects_to_update = list_mpl_objects_to_update
        self.pos = pos
    def __call__(self, change):
        self.ngl_wdg.isClick = False
        _idx = change["new"]
        try:
            for idot in self.list_mpl_objects_to_update:
                update2Dlines(idot, self.pos[_idx, 0], self.pos[_idx, 1])
            #print("caught index error with index %s (new=%s, old=%s)" % (_idx, change["new"], change["old"]))
        except IndexError as e:
            for idot in self.list_mpl_objects_to_update:
                update2Dlines(idot, self.pos[0, 0], self.pos[0, 1])
            print("caught index error with index %s (new=%s, old=%s)" % (_idx, change["new"], change["old"]))
        #print("set xy = (%s, %s)" % (x[_idx], y[_idx]))

class GeometryInNGLWidget(object):
    r"""
    returns an object that is aware of where its geometries are located in the NGLWidget their representation status


    The object exposes two methods, show and hide, to automagically know what to do
    """

    def __init__(self, geom, ngl_wdg, list_of_repr_dicts=None,
                 color_molecule_hex='Element', n_small=10):
        self.lives_at_components = []
        self.geom = geom
        self.ngl_wdg = ngl_wdg
        self.have_repr = []

        sticky_rep = 'cartoon'
        if self.geom[0].top.n_residues < n_small:
            sticky_rep = 'ball+stick'
        if list_of_repr_dicts is None:
            list_of_repr_dicts = [{'repr_type': sticky_rep, 'selection': 'all'}]

        self.list_of_repr_dicts = list_of_repr_dicts
        self.color_molecule_hex = color_molecule_hex
        self.color_dot = color_molecule_hex
        if isinstance(self.color_molecule_hex, str) and color_molecule_hex == 'Element':
            self.color_dot = 'red'

    def show(self):
        # Show can mean either
        #    - add a whole new component (case 1)
        #    - add the representation again to a representation-less component (case 2)

        # CASE 1
        if self.is_empty() or self.all_reps_are_on():
            if len(self.have_repr) == self.geom.n_frames:
                print("arrived at the end")
                component = None
            else:
                idx = len(self.have_repr)

                self.ngl_wdg.add_trajectory(self.geom[idx])
                self.lives_at_components.append(len(self.ngl_wdg._ngl_component_ids) - 1)
                self.ngl_wdg.clear_representations(component=self.lives_at_components[-1])
                self.have_repr.append(True)
                component = self.lives_at_components[-1]

        # CASE 2
        elif self.any_rep_is_off():  # Some are living in the widget already but have no rep
            idx = _np.argwhere(~_np.array(self.have_repr))[0].squeeze()
            component = self.lives_at_components[idx]
            self.have_repr[idx] = True
        else:
            raise Exception("This situation should not arise. This is a bug")

        if component is not None:
            for irepr in self.list_of_repr_dicts:
                self.ngl_wdg.add_representation(irepr['repr_type'],
                                                selection=irepr['selection'],
                                                component=component,
                                                color=self.color_molecule_hex)

    def hide(self):
        if self.is_empty() or self.all_reps_are_off():
            print("nothing to hide")
            pass
        elif self.any_rep_is_on():  # There's represented components already in the widget
            idx = _np.argwhere(self.have_repr)[-1].squeeze()
            self.ngl_wdg.clear_representations(component=self.lives_at_components[idx])
            self.have_repr[idx] = False
        else:
            raise Exception("This situation should not arise. This is a bug")

    # Quickhand methods for knowing what's up
    def is_empty(self):
        if len(self.have_repr) == 0:
            return True
        else:
            return False

    def all_reps_are_off(self):
        if len(self.have_repr) == 0:
            return True
        else:
            return _np.all(~_np.array(self.have_repr))

    def all_reps_are_on(self):
        if len(self.have_repr) == 0:
            return False
        else:
            return _np.all(self.have_repr)

    def any_rep_is_off(self):
        return _np.any(~_np.array(self.have_repr))

    def any_rep_is_on(self):
        return _np.any(self.have_repr)

    def is_visible(self):
        if self.is_empty() or self.all_reps_are_off():
            return False
        else:
            return True

def link_ax_w_pos_2_nglwidget(ax, pos, ngl_wdg,
                              crosshairs=True,
                              dot_color='red',
                              band_width=None,
                              radius=False,
                              directionality=None,
                              exclude_coord=None,
                              ):
    r"""
    Initial idea for this function comes from @arose, the rest is @gph82

    Parameters
    ----------
    ax : matplotlib axis object to be linked

    pos : ndarray of shape (N,2) with the positions of the geoms in the ngl_wdg

    crosshairs : Boolean or str
        If True, a crosshair will show where the mouse-click ocurred. If 'h' or 'v', only the horizontal or
        vertical line of the crosshair will be shown, respectively. If False, no crosshair will appear

    dot_color : Anything that yields matplotlib.colors.is_color_like(dot_color)==True
        Default is 'red'. dot_color='None' yields no dot

    band_width : None or iterable of len = 2
        If band_width is not None, the method tries to figure out on its own if
        there is an ascending coordinate and will include a moving band on :obj:ax
        of this width (in units of the axis along which the band is plotted)

        If the method cannot find an ascending coordinate, an exception is thrown

    directionality : str or None, default is None
        If not None, directionality can be either 'a2w' or 'w2a', meaning that connectivity
         between axis and widget will be only established as
         * 'a2w' : action in axis   triggers action in widget, but not the other way around
         * 'w2a' : action in widget triggers action in axis, but not the other way around

    exclude_coord : None or int , default is None
        The excluded coordinate will not be considered when computing the nearest-point-to-click.
        Typical use case is for visualize.traj to only compute distances horizontally along the time axis

    Returns
    -------

    axes_widget : :obj:`matplotlib.Axes.Axeswidget` that has been linked to the NGLWidget
    """

    assert directionality in [None, 'a2w', 'w2a'], "The directionality parameter has to be in [None, 'a2w', 'w2a'] " \
                                                   "not %s"%directionality

    assert crosshairs in [True, False, 'h', 'v'], "The crosshairs parameter has to be in [True, False, 'h','v'], " \
                                                   "not %s" % crosshairs
    ipos = _np.copy(pos)
    if _is_int(exclude_coord):
        ipos[:,exclude_coord] = 0
    kdtree = _cKDTree(ipos)

    # Are we in a sticky situation?
    if hasattr(ngl_wdg, '_GeomsInWid'):
        sticky = True
    else:
        assert ngl_wdg.trajectory_0.n_frames == pos.shape[0], \
            ("Mismatching frame numbers %u vs %u" % (ngl_wdg.trajectory_0.n_frames, pos.shape[0]))
        sticky = False

    # Basic interactive objects
    showclick_objs = []
    if crosshairs in [True, 'h']:
        lineh = ax.axhline(ax.get_ybound()[0], c="black", ls='--')
        setattr(lineh, 'whatisthis', 'lineh')
        showclick_objs.append(lineh)
    if crosshairs in [True, 'v']:
        linev = ax.axvline(ax.get_xbound()[0], c="black", ls='--')
        setattr(linev, 'whatisthis', 'linev')
        showclick_objs.append(linev)

    if _is_color_like(dot_color):
        pass
    else:
        raise TypeError('dot_color should be a matplotlib color')

    dot = ax.plot(pos[0,0],pos[0,1], 'o', c=dot_color, ms=7, zorder=100)[0]
    setattr(dot,'whatisthis','dot')
    list_mpl_objects_to_update = [dot]

    # Other objects, related to smoothing options
    if band_width is not None:
        if radius:
            band_width_in_pts = int(_np.round(pts_per_axis_unit(ax).mean() * _np.mean(band_width)))
            rad = ax.plot(pos[0, 0], pos[0, 1], 'o',
                          ms=_np.round(band_width_in_pts),
                          c='green', alpha=.25, markeredgecolor='None')[0]
            setattr(rad, 'whatisthis', 'dot')
            if not sticky:
                list_mpl_objects_to_update.append(rad)
        else:
            # print("Band_width(x,y) is %s" % (band_width))
            coord_idx = get_ascending_coord_idx(pos)
            if _np.ndim(coord_idx)>0 and len(coord_idx)==0:
                raise ValueError("Must have an ascending coordinate for band_width usage")
            band_width_in_pts = int(_np.round(pts_per_axis_unit(ax)[coord_idx] * band_width[coord_idx]))
            # print("Band_width in %s is %s pts"%('xy'[coord_idx], band_width_in_pts))

            band_call = [ax.axvline, ax.axhline][coord_idx]
            band_init = [ax.get_xbound, ax.get_ybound][coord_idx]
            band_type = ['linev',  'lineh'][coord_idx]
            band = band_call(band_init()[0],
                             lw=band_width_in_pts,
                             c="green", ls='-',
                             alpha=.25)
            setattr(band, 'whatisthis', band_type)
            list_mpl_objects_to_update.append(band)

    ngl_wdg.isClick = False

    CLA_listener = ClickOnAxisListener(ngl_wdg, kdtree, crosshairs, showclick_objs, ax, pos,
                                       list_mpl_objects_to_update)

    NGL_listener = ChangeInNGLWidgetListener(ngl_wdg, list_mpl_objects_to_update, pos)
    # Connect axes to widget
    axes_widget = _AxesWidget(ax)
    if directionality in [None, 'a2w']:
        axes_widget.connect_event('button_release_event', CLA_listener)

    # Connect widget to axes
    if directionality in [None, 'w2a']:
        ngl_wdg.observe(NGL_listener, "frame", "change")

    ngl_wdg.center()
    return axes_widget
