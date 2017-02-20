from __future__ import print_function

__author__ = 'gph82'


from pyemma.plots import plot_free_energy
import numpy as _np
from .bmutils import link_ax_w_pos_2_nglwidget as _link_ax_w_pos_2_nglwidget, \
    data_from_input as _data_from_input, \
    extract_visual_fnamez as _extract_visual_fnamez
from . import generate

from matplotlib import pylab as _plt
import nglview as _nglview
import mdtraj as _md
from os.path import basename as _basename


def FES(MD_trajfile, MD_top, projected_trajectory,
        nbins=100, n_sample = 100,
        xlabel='proj_0', ylabel='proj_1'):
    r"""
    Return a molecular visualization widget connected with a free energy plot.

    Parameters
    ----------

    MD_trajfile : str, name of the file the the molecular dynamics (MD) trajectory.
            :obj:`mdtraj.Trajectory` object. Any extension that :py:obj:`mdtraj` can read is accepted

    MD_top : str to topology filename directly :obj:`mdtraj.Topology` object

    projected_trajectory : str to a filename or numpy ndarray of shape (n_frames, n_dims)
        Time-series with the projection(s) that want to be explored. If these have been computed externally,
        you can provide .npy-filenames or readable asciis (.dat, .txt etc).
        NOTE: projX assumes that there is no time column.

    nbins : int, default 100
        The number of bins per axis to used in the histogram (FES)

    n_sample : int, default is 100
        The number of geometries that will be used to represent the FES. The higher the number, the higher the spatial
        resolution of the "click"-action.

    xlabel : str, default is 'proj_0'
        xlabel of the FES plot

    ylabel : str, default is 'proj_1'
        ylabel of the FES plot

    Returns
    --------

    ax :
        :obj:`pylab.Axis` object
    iwd :
        :obj:`nglview.NGLWidget`
    data_sample:
        numpy ndarray of shape (n, n_sample) with the position of the dots in the plot
    geoms:
        :obj:`mdtraj.Trajectory` object with the geometries n_sample geometries shown by the nglwidget

    """
    data_sample, geoms, data = generate.sample(MD_trajfile, MD_top,projected_trajectory,
                                               n_points=n_sample,
                                        return_data=True
                                         )
    data = _np.vstack(data)
    #h, (x, y) = _np.histogramdd(data, bins=nbins)

    #irange = _np.hstack((x[[0,-1]], y[[0,-1]]))
    _plt.figure()
    plot_free_energy(data[:,0], data[:,1], nbins=nbins)
    #_plt.contourf(-_np.log(h).T, extent=irange)
    ax = _plt.gca()
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    iwd = sample(data_sample, geoms.superpose(geoms[0]), ax)

    return _plt.gca(), _plt.gcf(), iwd, data_sample, geoms

def traj(trajectory,
         MD_top, projected_trajectory, max_frames=1000,
         stride=1, proj_idxs=[0,1], plot_FES=False):
    r"""Link a projected trajectory, X(t) with the molecular structures behind it. Optionally plot also the resulting
    FES.

    Parameters
    -----------

    trajectory : str,  or :obj:`mdtraj.Trajectory` object.
        Filename (any extension that :py:obj:`mdtraj` can read is accepted) or directly the :obj:`mdtraj.Trajectory`
        object containing the MD trajectory

    MD_top : str to topology filename directly :obj:`mdtraj.Topology` object

    projected_trajectory : str to a filename or numpy ndarray of shape (n_frames, n_dims)
        Time-series with the projection(s) that want to be explored. If these have been computed externally,
        you can provide .npy-filenames or readable asciis (.dat, .txt etc).
        NOTE: projX assumes that there is no time column.

    max_frames : int, default is 1000
        If the trajectoy is longer than this, stride to this length (in frames)

    stride : int, default is 1
        Stride value in case of large datasets. In case of having :obj:`trajectory` and :obj:`projected_trajectory`
        in memmory (and not on disk) the stride can take place before calling :obj:`traj`. This parameter only
        has effect when reading things from disk. NOTE:

    proj_idxs : iterable of ints, default is [0,1]
        Indices of the projected coordinates to use in the various representations

    plot_FES : bool, default is False
        Plot (and interactively link) the FES as well

    Returns
    ---------

     ax, iwd, data_sample, geoms
         return _plt.gca(), _plt.gcf(), widget, geoms

    ax :
        :obj:`pylab.Axis` object
    fig :
        :obj:`pylab.Figure` object
    iwd :
        :obj:`nglview.NGLWidget`
    geoms:
        :obj:`mdtraj.Trajectory` object with the geometries n_sample geometries shown by the nglwidget

    """
    assert len(proj_idxs) == 2

    # Parse input
    data = _data_from_input(projected_trajectory)[0][:,proj_idxs]
    if isinstance(trajectory, _md.Trajectory):
        geoms = trajectory
    else: # let mdtraj fail
        geoms = _md.load(trajectory, top=MD_top)

    # Does the projected trajectory and the data match?
    assert geoms.n_frames == len(data), (geoms.n_frames, len(data))

    # Stride even more if necessary
    time = _np.arange(geoms.n_frames)
    if len(time[::stride]) > max_frames:
        stride = int(_np.floor(geoms.n_frames/max_frames))
    # Stride
    geoms = geoms[::stride]
    data = data[::stride]
    time = time[::stride]

    myfig, myax = _plt.subplots(2,1, sharex=True)

    widget = None
    for idata, iax in zip(data.T, myax):
        data_sample =_np.vstack((time, idata)).T
        iax.plot(time, idata)
        widget = sample(data_sample, geoms.superpose(geoms[0]), iax, clear_lines=False, widget=widget)

    if plot_FES:
        h, (x, y) = _np.histogramdd(data, bins=50)
        irange = _np.hstack((x[[0,-1]], y[[0,-1]]))
        _plt.figure()
        _plt.contourf(-_np.log(h).T, extent=irange)
        ax = _plt.gca()
        widget = sample(data, geoms.superpose(geoms[0]), ax, widget=widget)

    return _plt.gca(), _plt.gcf(), widget, geoms


def sample(positions, geom,  ax,
           plot_path=False,
           clear_lines=True,
           widget=None,
           **link_ax2wdg_kwargs
                   ):

    r"""
    Visualize the geometries in :obj:`geom` according to the data in :obj:`positions` on an existing matplotlib axes :obj:`ax`

    Use this method when the array of positions, the geometries, the axes (and the widget, optionally) have already been
    generated elsewhere.

    Parameters
    ----------
    positions : numpy nd.array of shape (n_frames, 2)
        Contains the position associated with each frame in :obj:`geom` in that order

    geom : :obj:`mdtraj.Trajectory` object
        Contains n_frames, each frame

    ax : matplotlib.pyplot.Axes object
        The axes to be linked with the nglviewer widget

    plot_path : bool, default is False
        whether to draw a line connecting the positions in :obj:`positions`

    clear_lines : bool, default is True
        whether to clear all the lines that were previously drawn in :obj:`ax`

    widget : None or existing nglview widget
        you can provide an already instantiated nglviewer widget here

    link_ax2wdg_kwargs: dictionary of named arguments, optional
        named arguments for the function :any:`_link_ax_w_pos_2_nglwidget`, which is the one that internally
        provides the interactivity. Non-expert users can safely ignore this option.

    Returns
    --------

    iwd : :obj:`nglview.NGLWidget`

    """

    # Create ngl_viewer widget
    if widget is None:
        iwd = _nglview.show_mdtraj(geom)
    else:
        iwd = widget

    if clear_lines == True:
        [ax.lines.pop() for ii in range(len(ax.lines))]
    # Plot the path on top of it
    if plot_path:
        ax.plot(positions[:,0], positions[:,1], '-g', lw=3)

    # Link the axes widget with the ngl widget
    _link_ax_w_pos_2_nglwidget(ax,
                               positions,
                               iwd,
                               **link_ax2wdg_kwargs
                               )
    return iwd

def fnamez(fname,
           path_type='min_rmsd',
           proj_type='TIC',
           only_selection=True,
           selection_label='within 1 sigma',
           figsize=(10,10)
                   ):

    proj_name = _basename(fname)[:_basename(fname).find(proj_type)].strip('.')

    data, selection, x, y, h, v_crd_1, v_crd_2 = _extract_visual_fnamez(fname, path_type)
    data = data[:,[v_crd_1,v_crd_2]]

    # Load geometry
    geom = _md.load(fname.replace('.npz','.%s.pdb'%path_type))

    # Create contourplot
    _plt.figure(figsize=figsize)
    _plt.contourf(x[:-1],y[:-1], h.T, alpha=.50)
    #_plt.contourf(project_dict["h"].T, alpha=.50)
    # This can be take care of in "visualize sample"
    _plt.plot(data[:, v_crd_1],
              data[:, v_crd_2],
              alpha=.25, label=path_type)

    if only_selection:
        geom = geom[selection]
        data = data[selection]
        _plt.plot(data[:,0],data[:,1],'o', label=path_type+' '+selection_label)

    _plt.legend(loc='best')
    _plt.xlabel('%s %u'%(proj_type, v_crd_1))
    _plt.ylabel('%s %u'%(proj_type, v_crd_2))
    _plt.xlim(x[[0,-1]])
    _plt.ylim(y[[0,-1]])
    _plt.title('%s\n-np.log(counts)'%proj_name)

    iwd = sample(data, geom, _plt.gca(), plot_path=False, clear_lines=False)

    project_dict = {}
    for key, value in _np.load(fname).items():
        project_dict[key] = value

    return iwd, project_dict

def project_dict(project_dict,
                 path_type='min_rmsd',
                 proj_type='TIC',
                 only_compact_path=True,
                 selection_label='within 1 sigma',
                 figsize=(10,10),
                 project_name = None
                   ):



    # From dict to variables
    if path_type == 'min_rmsd':
        path = project_dict["Y_path_smpl"]
        compact_path = project_dict["compact_path_sample"]
    elif path_type == 'min_disp':
        path = project_dict["Y_path"]
        compact_path = project_dict["compact_path"]
    v_crd_1 = project_dict["v_crd_1"]
    v_crd_2 = project_dict["v_crd_2"]
    path = path[:, [v_crd_1, v_crd_2]]
    geom = project_dict["geom_"+path_type]
    x = project_dict["x"]
    y = project_dict["y"]
    h = project_dict["h"]

    # Create contourplot
    _plt.figure(figsize=figsize)
    _plt.contourf(x[:-1],y[:-1], h.T, alpha=.50) #ATTN h is already log(PDF)

    # This can be taken care of in "visualize sample", but i'm doing it here
    _plt.plot(path[:, 0],
              path[:, 1],
              alpha=.25, label=path_type)

    if only_compact_path:
        geom = geom[compact_path]
        path = path[compact_path]
        _plt.plot(path[:,0],path[:,1],'o', label=path_type+' '+selection_label)

    _plt.legend(loc='best')
    _plt.xlabel('%s %u'%(proj_type, v_crd_1))
    _plt.ylabel('%s %u'%(proj_type, v_crd_2))
    _plt.xlim(x[[0,-1]])
    _plt.ylim(y[[0,-1]])
    if project_name is not None:
        _plt.title('-np.log(counts)\n%s'%project_name)

    iwd = sample(path, geom, _plt.gca(), plot_path=False, clear_lines=False)

    return iwd