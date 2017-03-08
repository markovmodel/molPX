from __future__ import print_function

__author__ = 'gph82'


from pyemma.plots import plot_free_energy
import numpy as _np
from .bmutils import link_ax_w_pos_2_nglwidget as _link_ax_w_pos_2_nglwidget, \
    data_from_input as _data_from_input, \
    smooth_geom as _smooth_geom
    #extract_visual_fnamez as _extract_visual_fnamez
from . import generate

from matplotlib import pylab as _plt
import nglview as _nglview
import mdtraj as _md
from os.path import basename as _basename


def FES(MD_trajectories, MD_top, projected_trajectory,
        proj_idxs = [0,1],
        nbins=100, n_sample = 100,
        axlabel='proj'):
    r"""
    Return a molecular visualization widget connected with a free energy plot.

    Parameters
    ----------

    MD_trajectories : str, or list of strings with the filename(s) the the molecular dynamics (MD) trajectories.
        Any file extension that :py:obj:`mdtraj` (.xtc, .dcd etc) can read is accepted.

        Alternatively, a single :obj:`mdtraj.Trajectory` object or a list of them can be given as input.

    MD_top : str to topology filename or directly an :obj:`mdtraj.Topology` object

    projected_trajectory : str to a filename or numpy ndarray of shape (n_frames, n_dims)
        Time-series with the projection(s) that want to be explored. If these have been computed externally,
        you can provide .npy-filenames or readable asciis (.dat, .txt etc).
        NOTE: molpx assumes that there is no time column.

    proj_idxs: list or ndarray of length 2
        Selection of projection idxs (zero-idxd) to visualize.

    nbins : int, default 100
        The number of bins per axis to used in the histogram (FES)

    n_sample : int, default is 100
        The number of geometries that will be used to represent the FES. The higher the number, the higher the spatial
        resolution of the "click"-action.

    axlabel : str, default is 'proj'
        Format of the labels in the FES plot

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
    data_sample, geoms, data = generate.sample(MD_trajectories, MD_top, projected_trajectory, proj_idxs=proj_idxs,
                                               n_points=n_sample,
                                        return_data=True
                                         )
    data = _np.vstack(data)

    _plt.figure()
    # Use PyEMMA's plotting routing
    plot_free_energy(data[:,proj_idxs[0]], data[:,proj_idxs[1]], nbins=nbins)

    #h, (x, y) = _np.histogramdd(data, bins=nbins)
    #irange = _np.hstack((x[[0,-1]], y[[0,-1]]))
    #_plt.contourf(-_np.log(h).T, extent=irange)

    ax = _plt.gca()
    ax.set_xlabel('$\mathregular{%s_{%u}}$'%(axlabel, proj_idxs[0]))
    ax.set_ylabel('$\mathregular{%s_{%u}}$'%(axlabel, proj_idxs[1]))

    iwd = sample(data_sample, geoms.superpose(geoms[0]), ax)

    return _plt.gca(), _plt.gcf(), iwd, data_sample, geoms

def traj(MD_trajectories,
         MD_top, projected_trajectories,
         active_traj=0,
         max_frames=2000,
         stride=1,
         proj_stride=1,
         proj_idxs=[0,1],
         plot_FES=False,
         panel_height = 1,
         sharey_traj=True,
         dt = 1.0,
         tunits = 'ns',
         traj_selection = None,
         ):
    r"""Link one or many :obj:`projected trajectories`, [Y_0(t), Y_1(t)...], with the :obj:`MD_trajectories` that
    originated them.

    Optionally plot also the resulting FES.

    Parameters
    -----------

    MD_trajectories : str, or list of strings with the filename(s) the the molecular dynamics (MD) trajectories.
        Any file extension that :py:obj:`mdtraj` (.xtc, .dcd etc) can read is accepted.

        Alternatively, a single :obj:`mdtraj.Trajectory` object or a list of them can be given as input.

    MD_top : str to topology filename or directly :obj:`mdtraj.Topology` object

    projected_trajectories : str to a filename or numpy ndarray of shape (n_frames, n_dims)
        Time-series with the projection(s) that want to be explored. If these have been computed externally,
        you can provide .npy-filenames or readable asciis (.dat, .txt etc).
        NOTE: molpx assumes that there is no time column.

    active_traj : int, default 0
        Index of the trajectory that will be responsive. (zero-indexing)

    max_frames : int, default is 1000
        If the trajectoy is longer than this, stride to this length (in frames)

    stride : int, default is 1
        Stride value in case of large datasets. In case of having :obj:`MD_trajectories` and :obj:`projected_trajectories`
        in memory (and not on disk) the stride can take place also before calling this method.

    proj_stride : int, default is 1
        Stride value that was used in the :obj:`projected_trajectories` relative to the :obj:`MD_trajectories`
        If the original :obj:`MD_trajectories` were stored every 5 ps but the projected trajectories were stored
        every 50 ps, :obj:`proj_stride` = 10 has to be provided, otherwise an exception will be thrown informing
        the user that the :obj:`MD_trajectories` and the :obj:`projected_trajectories` have different number of frames.

    proj_idxs : iterable of ints, default is [0,1]
        Indices of the projected coordinates to use in the various representations

    plot_FES : bool, default is False
        Plot (and interactively link) the FES as well

    panel_height : int, default  1
        Height, in inches, of each panel of each trajectory subplots

    sharey_traj : bool, default is True
        Force the panels of each projection to have the same yaxes across trajectories (Note: Not across coordinates)

    dt : float, default is 1.0
        Physical time-unit equivalent to one frame of the :obj:`projected_trajectories`

    tunits : str, default is 'ns'
        Name of the physical time unit provided in :obj:`dt`

    traj_selection : None, int, iterable of ints, default is None
        Don't plot all trajectories but only few of them. The default None implies that all trajs will be plotted.
        Note: the data used for the FES will always include all trajectories, regardless of this value

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
    data = [iY[:,proj_idxs] for iY in _data_from_input(projected_trajectories)]
    if not isinstance(MD_trajectories, list):
        MD_trajectories = [MD_trajectories]
    assert len(data) == len(MD_trajectories), "Mismatch between number of MD-trajectories " \
                                           "and projected trajectores %u vs %u"%(len(MD_trajectories), len(data))
    assert len(MD_trajectories) > active_traj, "parameter active_traj selected for traj nr. %u to be active " \
                                            " but your input has only %u trajs. Note: the parameter active_traj " \
                                            "is zero-indexed"%(active_traj, len(MD_trajectories))
    if isinstance(traj_selection, int):
        traj_selection = [traj_selection]
    elif traj_selection is None:
        traj_selection = _np.arange(len(data))
    assert _np.max(traj_selection) < len(data), "Selected up to traj. nr. %u via the parameter traj_selection, " \
                                                "but only provided %u trajs"%(_np.max(traj_selection), len(data))
    assert active_traj in traj_selection, "Selected traj. nr. %u to be the active one, " \
                                          "but it is not contained in traj_selection: %s"%(active_traj, traj_selection)

    if isinstance(MD_trajectories[active_traj], _md.Trajectory):
        geoms = MD_trajectories[active_traj][::proj_stride]
    else: # let mdtraj fail
        geoms = _md.load(MD_trajectories[active_traj], stride=proj_stride, top=MD_top)

    # Do the projected trajectory and the data match?
    assert geoms.n_frames == len(data[active_traj]), (geoms.n_frames, len(data[active_traj]))

    # Stride to avoid representing huge vectors
    times = []
    for ii in range(len(data)):
        time = _np.arange(data[ii].shape[0])*dt*proj_stride
        if len(time[::stride]) > max_frames:
            stride = int(_np.floor(data[ii].shape[0]/max_frames))

        times.append(time[::stride])
        data[ii] = data[ii][::stride]
        if ii == active_traj:
            geoms = geoms[::stride]

    # For later axes-cosmetics
    tmax, tmin = _np.max([time[-1] for time in times]), _np.min([time[0] for time in times])
    ylims = _np.zeros((2,2))
    for ii in range(2):
        ylims[0, ii] = _np.min([idata[:,ii].min() for idata in data])
        ylims[1, ii] = _np.max([idata[:,ii].max() for idata in data])
    ylabels = ['$\mathregular{proj_%u}$'%ii for ii in proj_idxs]


    myfig, myax = _plt.subplots(len(traj_selection)*2,1, sharex=True, figsize=(7, len(data)*2*panel_height))
    myax = myax.reshape(len(traj_selection), -1)
    widget = None
    for jj, time, jdata, jax in zip(traj_selection,
                                    [times[jj] for jj in traj_selection],
                                    [data[jj] for jj in traj_selection],
                                    myax):

        for ii, (idata, iax) in enumerate(zip(jdata.T, jax)):
            data_sample =_np.vstack((time, idata)).T
            iax.plot(time, idata)
            if jj == active_traj:
                widget = sample(data_sample, geoms.superpose(geoms[0]), iax, clear_lines=False, widget=widget)

            # Axis-Cosmetics
            if ii == 0:
                iax.set_title('traj %u'%jj)
            iax.set_ylabel(ylabels[ii])
            iax.set_xlim([tmin, tmax])
            if sharey_traj:
                iax.set_ylim(ylims[:,ii]+[-1,1]*_np.diff(ylims[:,ii])*.1)
    iax.set_xlabel(tunits)

    if plot_FES:
        h, (x, y) = _np.histogramdd(_np.vstack(data), bins=50)
        irange = _np.hstack((x[[0,-1]], y[[0,-1]]))
        _plt.figure()
        _plt.contourf(-_np.log(h).T, extent=irange)
        _plt.xlabel(ylabels[0])
        _plt.ylabel(ylabels[1])
        ax = _plt.gca()
        widget = sample(data[active_traj], geoms.superpose(geoms[0]), ax, widget=widget)

    return _plt.gca(), _plt.gcf(), widget, geoms


def sample(positions, geom,  ax,
           plot_path=False,
           clear_lines=True,
           smooth = 0,
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

    smooth : int, default is 0,
        if smooth > 0, the shown geometries and paths will be smoothed out by 2*n frames.
        See :any:`bmutils.smooth_geom` for more information

    widget : None or existing nglview widget
        you can provide an already instantiated nglviewer widget here (avanced use)

    link_ax2wdg_kwargs: dictionary of named arguments, optional
        named arguments for the function :obj:`_link_ax_w_pos_2_nglwidget`, which is the one that internally
        provides the interactivity. Non-expert users can safely ignore this option.

    Returns
    --------

    iwd : :obj:`nglview.NGLWidget`

    """

    if smooth > 0:
        geom, positions = _smooth_geom(geom, smooth, geom_data=positions)

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
    ax_wdg = _link_ax_w_pos_2_nglwidget(ax,
                                        positions,
                                        iwd,
                                        radius=smooth,
                                        **link_ax2wdg_kwargs
                               )
    # somehow returning the ax_wdg messes the displaying of both widgets

    return iwd

# Do I really need this? Don't think so
def _fnamez(fname,
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

# Do I really need this? Don't think so
def _project_dict(project_dict,
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
