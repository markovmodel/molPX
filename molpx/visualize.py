from __future__ import print_function

__author__ = 'gph82'


from pyemma.plots import plot_free_energy
import numpy as _np
from .bmutils import link_ax_w_pos_2_nglwidget as _link_ax_w_pos_2_nglwidget, \
    data_from_input as _data_from_input, \
    smooth_geom as _smooth_geom,\
    most_corr as _most_corr_info, \
    re_warp as _re_warp, \
    add_atom_idxs_widget as _add_atom_idxs_widget, \
    matplotlib_colors_no_blue as _bmcolors

from . import generate

from matplotlib import pylab as _plt, rcParams as _rcParams
import nglview as _nglview
import mdtraj as _md

# All calls to nglview call actually this function
def _initialize_nglwidget_if_safe(geom, mock=True):
    try:
        return _nglview.show_mdtraj(geom)
    except:
        if mock:
            print("molPX has to be used inside a notebook, not from terminal. A mock nglwidget is being returned."
                  "Ignore this message if testing, "
                  "otherwise refer to molPX documentation")
            return _mock_nglwidget(geom)
        else:
            raise Exception("molPX has to be used inside a notebook, not from terminal")

class _mock_nglwidget(object):
    r"""
    mock widget, which isn't even a widget, to allow for testing inside of the terminal
    """
    def __init__(self, geom):
        self.trajectory_0 = geom
    def observe(self,*args, **kwargs):
        pass

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
         tunits = 'frames',
         traj_selection = None,
         projection = None,
         n_feats = 1,
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

    tunits : str, default is 'frames'
        Name of the physical time unit provided in :obj:`dt`

    traj_selection : None, int, iterable of ints, default is None
        Don't plot all trajectories but only few of them. The default None implies that all trajs will be plotted.
        Note: the data used for the FES will always include all trajectories, regardless of this value

    projection : object that generated the projection, default is None
        The projected coordinates may come from a variety of sources. When working with :ref:`pyemma` a number of objects
        might have generated this projection, like a
        * :obj:`pyemma.coordinates.transform.TICA` or a
        * :obj:`pyemma.coordinates.transform.PCA` or a

        Pass this object along and observe and the features that are most correlated with the projections
        will be plotted for the active trajectory, allowing the user to establish a visual connection between the
        projected coordinate and the original features (distances, angles, contacts etc)

    n_feats : int, default is 1
        If a :obj:`projection` is passed along, the first n_feats features that most correlate the
        the projected trajectories will be represented, both in form of trajectories feat vs t as well as in
        the nglwidget

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


    if isinstance(proj_idxs, int):
        proj_idxs = [proj_idxs]

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

    # For axes-cosmetics later on
    tmax, tmin = _np.max([time[-1] for time in times]), _np.min([time[0] for time in times])
    ylims = _np.zeros((2, len(proj_idxs)))
    for ii, __ in enumerate(proj_idxs):
        ylims[0, ii] = _np.min([idata[:,ii].min() for idata in data])
        ylims[1, ii] = _np.max([idata[:,ii].max() for idata in data])
    ylabels = ['$\mathregular{proj_%u}$'%ii for ii in proj_idxs]

    # Do we have usable projection information?
    corr_dict = _most_corr_info(projection, geoms=geoms, proj_idxs=proj_idxs, n_args=n_feats)
    if corr_dict["feats"] != []:
        # Then extend the trajectory selection to include the active trajectory twice
        traj_selection = _np.insert(traj_selection,
                                    _np.argwhere([active_traj==ii for ii in traj_selection]).squeeze(),
                                    [active_traj] * n_feats)
    else:
        # squash whatever input we had if the projection-info input wasn't actually usable
        n_feats = 0

    myfig, myax = _plt.subplots(len(traj_selection)*len(proj_idxs),1, sharex=True, figsize=(7, len(data)*len(proj_idxs)*panel_height), squeeze=False)
    myax = myax.reshape(len(traj_selection), -1)

    # Initialize some things
    widget = None
    projections_plotted = 0
    for kk, (jj, time, jdata, jax) in enumerate(zip(traj_selection,
                                                    [times[jj] for jj in traj_selection],
                                                    [data[jj] for jj in traj_selection],
                                                    myax)):
        for ii, (idata, iax) in enumerate(zip(jdata.T, jax)):
            data_sample =_np.vstack((time, idata)).T

            # Inactive trajectories, act normal
            if jj != active_traj:
                iax.plot(time, idata)
            # Active trajectory, distinguish between projection and feature
            else:
                if projections_plotted < len(proj_idxs): #projection
                    iax.plot(time, idata)
                    widget = sample(data_sample, geoms.superpose(geoms[0]), iax,
                                    clear_lines=False, widget=widget,
                                    crosshairs='v')
                    projections_plotted += 1
                    time_feat = time
                    # TODO find out why this is needed

            # Axis-Cosmetics
            iax.set_ylabel(ylabels[ii])
            iax.set_xlim([tmin, tmax])
            iax2 = iax.twinx()
            iax2.set_yticklabels('')
            iax2.set_ylabel('traj %u'%jj, rotation =-90, va='bottom', ha='center')
            if sharey_traj:
                iax.set_ylim(ylims[:,ii]+[-1,1]*_np.diff(ylims[:,ii])*.1)

    # Last of axis cosmetics
    iax.set_xlabel(tunits)

    # Now let's go to the feature axes
    # Some bookkeping about axis and features
    first_empty_axis = _np.argwhere(traj_selection==active_traj)[0]*len(proj_idxs)+len(proj_idxs)
    last_empty_axis = first_empty_axis+ len(proj_idxs) * n_feats
    rows, cols = _np.unravel_index(_np.arange(first_empty_axis,last_empty_axis), myax.shape)
    colors = _bmcolors()
    smallfontsize = int(_rcParams['font.size']/2)
    for kk, (ir, ic)  in enumerate(zip(rows, cols)):
        # Determine axis
        iax = myax[ir, ic]
        # Grab the right properties
        iproj, ifeat = _np.unravel_index(kk, (len(proj_idxs), n_feats))
        ifeat_val = corr_dict["feats"][iproj][:, ifeat]
        ilabel = corr_dict["labels"][iproj][ifeat]
        icol = colors[iproj]
        # Plot
        lines = iax.plot(time_feat, ifeat_val, color=icol)[0]
        #Cosmetics
        iax.set_ylabel('\n'.join(_re_warp(ilabel, 16)), fontsize=smallfontsize)
        iax.set_ylim([ifeat_val.min(),
                      ifeat_val.max(),
                      ] + [-1, 1] * _np.diff(ylims[:, ii]) * .05)

        # Link widget
        fdata_sample = _np.vstack((time_feat, ifeat_val)).T
        widget = sample(fdata_sample, geoms.superpose(geoms[0]), iax,
                        clear_lines=False, widget=widget,
                        crosshairs=False, directionality='w2a')

        # Add the correlation value
        iax.legend([lines],['Corr(feat|%s)=%2.1f' % (ylabels[iproj], corr_dict["vals"][iproj][ifeat])],
                   fontsize=smallfontsize, loc='best', frameon=False)

        # Add visualization (let the method decide if it's possible or not)
        widget = _add_atom_idxs_widget([corr_dict["atom_idxs"][iproj][ifeat]], widget, color_list=[icol])

    if plot_FES:
        if len(proj_idxs)!=2:
            raise NotImplementedError('Can only plot 2D FES if more than one projection idxs has been '
                                      'specificed, but only got %s. \n In the future a 1D histogramm will '
                                      'be shown.'%proj_idxs)
        h, (x, y) = _np.histogramdd(_np.vstack(data), bins=50)
        irange = _np.hstack((x[[0,-1]], y[[0,-1]]))
        _plt.figure()
        _plt.contourf(-_np.log(h).T, extent=irange)
        _plt.xlabel(ylabels[0])
        _plt.ylabel(ylabels[1])
        ax = _plt.gca()
        widget = sample(data[active_traj], geoms.superpose(geoms[0]), ax, widget=widget)

    return _plt.gca(), _plt.gcf(), widget, geoms

def correlations(correlation_input,
                 geoms=None,
                 proj_idxs=None,
                 feat_name=None,
                 widget=None,
                 proj_color_list=None,
                 n_feats=1,
                 verbose=False):
    r"""
    Provide a visual and textual representation of the linear correlations between projected coordinates (PCA, TICA)
     and original features.

    Parameters
    ---------

    correlation_input : anything
        Something that could, in principle, be a :obj:`pyemma.coordinates.transformer,
        like a TICA or PCA object
        (this method will be extended to interpret other inputs, so for now this parameter is pretty flexible)

    geoms : None or obj:`md.Trajectory`, default is None
        The values of the most correlated features will be returned for the geometires in this object. If widget is
        left to its default, None, :obj:`correlations` will create a new widget and try to show the most correlated
          features on top of the widget

    widget : None or nglview widget
        Provide an already existing widget to visualize the correlations on top of. This is only for expert use,
        because no checks are done to see if :obj:`correlation_input` and the geometry contained in the
        widget **actually match**. Use with caution.

        Note
        ----
            When objects :obj:`geoms` and :obj:`widget` are provided simultaneously, three things happen:
             * no new widget will be instantiated
             * the display of features will be on top of whatever geometry :obj:`widget` contains
             * the value of the features is computed for the geometry of :obj:`geom`

            Use with caution and clean bookkeeping!

    proj_color_list: list, default is None
        projection specific list of colors to provide the representations with. The default None yields blue.
        In principle, the list can contain one color for each projection (= as many colors as len(proj_idxs)
        but if your list is short it will just default to the last color. This way, proj_color_list=['black'] will paint
        all black regardless len(proj_idxs)


    proj_idxs: None, or int, or iterable of integers, default is None
        The indices of the projections for which the most correlated feture will be returned
        If none it will default to the dimension of the correlation_input object

    feat_name : None or str, default is None
        The prefix with which to prepend the labels of the most correlated features. If left to None, the feature
        description found in :obj:`correlation_input` will be used (if available)

    n_feats : int, default is 1
        Number of argmax correlation to return for each feature.

    verbose : Bool, default is True
        print to standard output

    :return:
    most_corr_idxs, most_corr_vals, most_corr_labels, most_corr_feats, most_corr_atom_idxs, lines, widget, lines
    """
    # todo document
    # todo test

    corr_dict = _most_corr_info(correlation_input, geoms=geoms, proj_idxs=proj_idxs, feat_name=feat_name, n_args=n_feats)

    # Create ngl_viewer widget
    if geoms is not None and widget is None:
        widget = _initialize_nglwidget_if_safe(geoms.superpose(geoms))

    if proj_color_list is None:
        proj_color_list = ['blue'] * len(corr_dict["idxs"])
    elif isinstance(proj_color_list, list) and len(proj_color_list)<len(corr_dict["idxs"]):
        proj_color_list += [proj_color_list[-1]] * (len(corr_dict["idxs"]) - len(proj_color_list))
    elif not isinstance(proj_color_list, list):
        raise TypeError("parameter proj_color_list should be either None or a list, not %s of type %s"%(proj_color_list, type(proj_color_list)))

    # Add the represenation
    if widget is not None:
        for idxs, icol in zip(corr_dict["atom_idxs"], proj_color_list):
            _add_atom_idxs_widget(idxs, widget, color_list=[icol])

    if verbose:
        for ii, line in enumerate(corr_dict["info"]):
            print('%s is most correlated with '%(line["name"] ))
            for line in line["lines"]:
                if widget is not None:
                    line += ' (in %s in the widget)'%(proj_color_list[ii])
                print(line)

    return corr_dict, widget


def sample(positions, geom, ax,
           plot_path=False,
           clear_lines=True,
           n_smooth = 0,
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

    n_smooth : int, default is 0,
        if n_smooth > 0, the shown geometries and paths will be smoothed out by 2*n frames.
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

    if n_smooth > 0:
        geom, positions = _smooth_geom(geom, n_smooth, geom_data=positions)
        mean_smooth_radius = _np.diff(positions, axis=0).mean(0) * n_smooth
        band_width = 2 * mean_smooth_radius
    else:
        band_width = None

    # Create ngl_viewer widget
    if widget is None:
        iwd = _initialize_nglwidget_if_safe(geom)
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
                                        band_width=band_width,
                                        **link_ax2wdg_kwargs
                                        )
    # somehow returning the ax_wdg messes the displaying of both widgets

    return iwd

