from __future__ import print_function

__author__ = 'gph82'


from pyemma.plots import plot_free_energy as _plot_free_energy
import numpy as _np
from .bmutils import link_ax_w_pos_2_nglwidget as _link_ax_w_pos_2_nglwidget, \
    data_from_input as _data_from_input, \
    smooth_geom as _smooth_geom,\
    most_corr as _most_corr_info, \
    re_warp as _re_warp, \
    add_atom_idxs_widget as _add_atom_idxs_widget, \
    matplotlib_colors_no_blue as _bmcolors, \
    get_ascending_coord_idx as _get_ascending_coord_idx, \
    listify_if_int as _listify_if_int, listfiy_if_not_list as _listfiy_if_not_list

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
    mock widget, which isn't even a widget, to allow for testing inside of the terminal.
    """
    # TODO nglvwidget inside terminal one should follow this comment
    # https://github.com/markovmodel/PyEMMA/issues/1062#issuecomment-288494497

    def __init__(self, geom):
        self.trajectory_0 = geom
    def observe(self,*args, **kwargs):
        print("The method 'observe' of a mock nglwidget is called. "
              "Ignore this message if testing, otherwise refer to molPX documentation.")

    def add_spacefill(self, *args, **kwargs):
        print("The method 'add_spacefill' of a mock nglwidget is called. "
              "Ignore this message if testing, otherwise refer to molPX documentation.")
        pass

def FES(MD_trajectories, MD_top, projected_trajectory,
        proj_idxs = [0,1],
        nbins=100, n_sample = 100,
        axlabel='proj',
        n_overlays=1):
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

    proj_idxs: int, list or ndarray
        Selection of projection idxs (zero-idxd) to visualize.

    nbins : int, default 100
        The number of bins per axis to used in the histogram (FES)

    n_sample : int, default is 100
        The number of geometries that will be used to represent the FES. The higher the number, the higher the spatial
        resolution of the "click"-action.

    axlabel : str, default is 'proj'
        Format of the labels in the FES plot

    n_overlays : int, default is 1
        The number of structures that will be simultaneously displayed as overlays for every sampled point of the FES.
        This parameter can seriously slow down the method, it is currently limited to a maximum value of 50

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

    # Prepare the overlay option
    n_overlays = _np.min([n_overlays,50])
    if n_overlays>1:
        keep_all_samples = True
    else:
        keep_all_samples = False

    # Prepare for 1D case
    proj_idxs = _listify_if_int(proj_idxs)

    data_sample, geoms, data = generate.sample(MD_trajectories, MD_top, projected_trajectory, proj_idxs=proj_idxs,
                                               n_points=n_sample,
                                               return_data=True,
                                               n_geom_samples=n_overlays,
                                               keep_all_samples=keep_all_samples
                                         )

    data = _np.vstack(data)

    ax, FES_data, edges = _plot_ND_FES(data[:,proj_idxs],
                                  ['$\mathregular{%s_{%u}}$' % (axlabel, ii) for ii in proj_idxs],
                                  bins=nbins)
    if edges[0] is not None:
        # We have the luxury of sorting!
        sorts_data = data_sample[:,0].argsort()
        data_sample[:,0] = data_sample[sorts_data,0]
        if isinstance(geoms, _md.Trajectory):
            geoms = [geoms]
            geoms = [_md.Trajectory([igeom[ii].xyz.squeeze() for ii in sorts_data], igeom.topology) for igeom in geoms]

        # TODO: look closely at this x[:-2]  (bins, edges, and off-by-one errors
        FES_sample = FES_data[_np.digitize(data_sample, edges[0][:-2])]
        data_sample = _np.hstack((data_sample, FES_sample))

    iwd = sample(data_sample, geoms, ax, clear_lines=False)

    return _plt.gca(), _plt.gcf(), iwd, data_sample, geoms

def _plot_ND_FES(data, ax_labels, bins=50):
    r""" A wrapper for pyemmas FESs plotting function that can also plot 1D

    Parameters
    ----------

    data : list of numpy nd.arrays

    ax_labels : list

    Returns
    -------

    ax : :obj:`pylab.Axis` object

    FES_data : numpy nd.array containing the FES (only for 1D data)

    edges : tuple containimg the axes along which FES is to be plotted (only in the 1D case so far, else it's None)

    """
    _plt.figure()
    ax = _plt.gca()
    idata = _np.vstack(data)
    ax.set_xlabel(ax_labels[0])
    if idata.shape[1] == 1:
        h, edges = _np.histogramdd(idata, bins=bins, normed=True)
        FES_data = -_np.log(h)
        FES_data -= FES_data.min()
        ax.plot(edges[0][:-1], FES_data)
        ax.set_ylabel('$\Delta G / \kappa T $')

    elif idata.shape[1] == 2:
        _plot_free_energy(idata[:,0], idata[:,1], nbins=bins, ax=ax)
        ax.set_ylabel(ax_labels[1])
        edges, FES_data = [None], None
        # TODO: retrieve the actual edges from pyemma's "plot_free_energy"'s axes
    else:
        raise NotImplementedError('Can only plot 1D or 2D FESs, but data has %s columns' % _np.shape(idata)[0])

    return ax, FES_data, edges,

def traj(MD_trajectories,
         MD_top, projected_trajectories,
         active_traj=0,
         max_frames=2000,
         stride=1,
         proj_stride=1,
         proj_idxs=[0,1],
         proj_labels='proj',
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

    proj_labels : either string or list of strings
	The projection plots will get this paramter for labeling their yaxis. If a str is 
        provided, that will be the base name proj_labels='%s_%u'%(proj_labels,ii) for each 
        projection. If a list, the list will be used. If not enough labels are there
        the module will complain

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
        the nglwidget. If :obj:`projection` is None, :obj:`nfeats`  will be ignored.

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

    proj_idxs = _listify_if_int(proj_idxs)

    # Parse input
    data = [iY[:,proj_idxs] for iY in _data_from_input(projected_trajectories)]

    MD_trajectories = _listfiy_if_not_list(MD_trajectories)

    assert len(data) == len(MD_trajectories), "Mismatch between number of MD-trajectories " \
                                           "and projected trajectores %u vs %u"%(len(MD_trajectories), len(data))
    assert len(MD_trajectories) > active_traj, "parameter active_traj selected for traj nr. %u to be active " \
                                            " but your input has only %u trajs. Note: the parameter active_traj " \
                                            "is zero-indexed"%(active_traj, len(MD_trajectories))

    traj_selection = _listify_if_int(traj_selection)

    if traj_selection is None:
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
    if isinstance(proj_labels, str):
       ylabels = ['$\mathregular{%s_{%u}}$'%(proj_labels, ii) for ii in proj_idxs]
    elif isinstance(proj_labels, list):
       ylabels = proj_labels
    else:
       raise TypeError("Parameter proj_labels has to be of type str or list, not %s"%type(proj_labels))

    # Do we have usable projection information?
    if projection is not None:
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
                                    crosshairs='v',
                                    exclude_coord=1)
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
    first_empty_axis = _np.argwhere([active_traj==ii for ii in traj_selection])[0]*len(proj_idxs)+len(proj_idxs)
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
                      ] + _np.array([-1, 1]) * (ifeat_val.max()-ifeat_val.min()) * .05)

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
        ax, FES_data, edges = _plot_ND_FES(data, ylabels)
        if edges[0] is not None:
            print(edges)
            FES_data = [FES_data[_np.digitize(idata, edges[0][:-2])] for idata in data]
            data = [_np.hstack((idata, iFES_data)) for idata, iFES_data in zip(data, FES_data)]

        widget = sample(data[active_traj], geoms.superpose(geoms[0]), ax, widget=widget, clear_lines=False)

    return _plt.gca(), _plt.gcf(), widget, geoms

def correlations(correlation_input,
                 geoms=None,
                 proj_idxs=None,
                 feat_name=None,
                 widget=None,
                 proj_color_list=None,
                 n_feats=1,
                 verbose=False,
                 featurizer=None):
    r"""
    Provide a visual and textual representation of the linear correlations between projected coordinates (PCA, TICA)
     and original features.

    Parameters
    ---------

    correlation_input : anything
        Something that could, in principle, be a :obj:`pyemma.coordinates.transformer,
        like a TICA, PCA or featurizer object or directly a correlation matrix, with a row for each feature and a column
        for each projection, very much like the :obj:`feature_TIC_correlation` of the TICA object of pyemma.


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

    featurizer : optional featurizer, default is None
        If :obj:`correlation_input` is not an :obj:`_MDFeautrizer` itself or doesn't have a
        data_producer.featurizer attribute, the user can input one here. If both an _MDfeaturizer *and* an :obj:`featurizer`
         are provided, the latter will be ignored.

    verbose : Bool, default is True
        print to standard output

    :return:
    most_corr_idxs, most_corr_vals, most_corr_labels, most_corr_feats, most_corr_atom_idxs, lines, widget, lines
    """
    # todo document
    # todo test
    # todo consider kwargs for most_corr_info

    corr_dict = _most_corr_info(correlation_input,
                                geoms=geoms, proj_idxs=proj_idxs, feat_name=feat_name, n_args=n_feats, featurizer=featurizer
                                )

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
           superpose=True,
           projection = None,
           n_feats = 1,
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

    geom : :obj:`mdtraj.Trajectory` objects or a list thereof.
        The geometries associated with the the :obj:`positions`. Hence, all have to have the same number of n_frames

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

    superpose : boolean, default is True
        The geometries in :obj:`geom` may or may not be oriented, depending on where they were generated.
        Since this method is mostly for visualization purposes, the default behaviour is to orient them all to
        maximally overlap with the first frame (of the first :obj:`mdtraj.Trajectory` object, in case :obj:`geom`
        is a list)
    projection : object that generated the projection, default is None
        The projected coordinates may come from a variety of sources. When working with :ref:`pyemma` a number of objects
        might have generated this projection, like a
        * :obj:`pyemma.coordinates.transform.TICA` or a
        * :obj:`pyemma.coordinates.transform.PCA` or a

        Expert use. Pass this object along ONLY if the :obj:`positions` have been generetaed using :any:`projection_paths`,
        so that looking at linear correlations makes sense. Observe the features that are most correlated with the projections
        will be plotted for the sample, allowing the user to establish a visual connection between the
        projected coordinate and the original features (distances, angles, contacts etc)

    n_feats : int, default is 1
        If a :obj:`projection` is passed along, the first n_feats features that most correlate the
        the projected trajectories will be represented, both in form of trajectories feat vs t as well as in
        the nglwidget. If :obj:`projection` is None, :obj:`nfeats`  will be ignored.

    link_ax2wdg_kwargs: dictionary of named arguments, optional
        named arguments for the function :obj:`_link_ax_w_pos_2_nglwidget`, which is the one that internally
        provides the interactivity. Non-expert users can safely ignore this option.

    Returns
    --------

    iwd : :obj:`nglview.NGLWidget`

    """

    assert isinstance(geom, (list, _md.Trajectory))

    # Dow I need to smooth things out?
    if n_smooth > 0:
        if isinstance(geom, _md.Trajectory): # smoothing only makes sense for paths, and paths cannot be lists at the moment
            geom, positions = _smooth_geom(geom, n_smooth, geom_data=positions)
            mean_smooth_radius = _np.diff(positions, axis=0).mean(0) * n_smooth
            band_width = 2 * mean_smooth_radius
    else:
        band_width = None

    # Create ngl_viewer widget
    if widget is None:
        if isinstance(geom, _md.Trajectory):
            iwd = _initialize_nglwidget_if_safe(geom.superpose(geom[0]))
        else:
            iwd = _initialize_nglwidget_if_safe(geom[0].superpose(geom[0]))
            for igeom in geom[1:]:
                iwd.add_trajectory(igeom.superpose(geom[0]))


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
    # Do we have usable projection information?
    if projection is not None:
        corr_dict = _most_corr_info(projection, geoms = geom, n_args=n_feats)
        if corr_dict["labels"] != []:
            iproj = _get_ascending_coord_idx(positions)
            for ifeat in range(n_feats):
                ilabel = corr_dict["labels"][iproj][ifeat]
                print(ilabel)
                iwd = _add_atom_idxs_widget([corr_dict["atom_idxs"][iproj][ifeat]], iwd,
                                            color_list=['green']
                                            )

    # somehow returning the ax_wdg messes the displaying of both widgets

    return iwd

