from __future__ import print_function

__author__ = 'gph82'

from pyemma.plots import plot_free_energy as _plot_free_energy
import numpy as _np

from matplotlib.cm import get_cmap as _get_cmap
from matplotlib.colors import rgb2hex as _rgb2hex, to_hex as _to_hex

from . import generate
from . import _bmutils
from . import _linkutils

from matplotlib import pylab as _plt, rcParams as _rcParams
import nglview as _nglview
import mdtraj as _md
from ipywidgets import VBox as _VBox, Layout as _Layout, Button as _Button

import warnings as _warnings

# All calls to nglview call actually this function
def _nglwidget_wrapper(geom, ngl_wdg=None, n_small=10):
    r""" Wrapper to nlgivew.show_geom's method that allows for some other automatic choice of
    representation

    Parameters
    ----------

    geom : :obj:`mdtraj.Trajectory` object or str with a filename to anything that :obj:`mdtraj` can read

    ngl_wdg : an already instantiated widget to add the geom, default is None

    n_small : if the geometry has less than n_small residues, force a "ball and stick" represenation


    :return: :nglview.widget object
    """

    if isinstance(geom, str):
        geom = _md.load(geom)

    if ngl_wdg is None:
        if geom is None:
            ngl_wdg = _nglview.NGLWidget()
        else:
            ngl_wdg = _nglview.show_mdtraj(geom)
    else:
        ngl_wdg.add_trajectory(geom)

    # Let' some customization take place
    ## Do we need a ball+stick representation?
    if geom is not None:
        if geom.top.n_residues < n_small:
            for ic in range(len(ngl_wdg._ngl_component_ids)):
                # TODO FIND OUT WHY THIS FAILS FOR THE LAST REPRESENTATION
                #print("removing reps for component",ic)
                ngl_wdg.remove_cartoon(component=ic)
                ngl_wdg.clear_representations(component=ic)
                ngl_wdg.add_ball_and_stick(component=ic)

    return ngl_wdg

def _add_y2_label(iax, label):
    iax2 = iax.twinx()
    iax2.set_yticklabels('')
    iax2.set_ylabel(label, rotation = -90, va = 'bottom', ha = 'center')

def FES(MD_trajectories, MD_top, projected_trajectories,
        proj_idxs = [0,1],
        nbins=100,
        n_sample = 100,
        proj_stride=1,
        weights=None,
        proj_labels='proj',
        n_overlays=1,
        atom_selection=None,
        **sample_kwargs):
    r"""
    Return a molecular visualization widget connected with a free energy plot.

    Parameters
    ----------

    MD_trajectories : str, or list of strings with the filename(s) the the molecular dynamics (MD) trajectories.
        Any file extension that :py:obj:`mdtraj` (.xtc, .dcd etc) can read is accepted.

        Alternatively, a single :obj:`mdtraj.Trajectory` object or a list of them can be given as input.

    MD_top : str to topology filename or directly an :obj:`mdtraj.Topology` object

    projected_trajectories : numpy ndarray (or list thereof) of shape (n_frames, n_dims) with the time-series
    of the projection(s) that want to be explored. Alternatively, strings or list of string with .npy or ascii filenames
     filenames (.dat, .txt etc)
    NOTE: molpx assumes that there is no time column.

    proj_idxs: int, list or ndarray
        Selection of projection idxs (zero-idxd) to visualize.

    nbins : int, default 100
        The number of bins per axis to used in the histogram (FES)

    n_sample : int, default is 100
        The number of geometries that will be used to represent the FES. The higher the number, the higher the spatial
        resolution of the "click"-action.

    proj_stride : int, default is 1
        Stride value that was used in the :obj:`projected_trajectories` relative to the :obj:`MD_trajectories`
        If the original :obj:`MD_trajectories` were stored every 5 ps but the projected trajectories were stored
        every 50 ps, :obj:`proj_stride` = 10 has to be provided, otherwise an exception will be thrown informing
        the user that the :obj:`MD_trajectories` and the :obj:`projected_trajectories` have different number of frames.

    weights : iterable of floats (or list thereof) each of shape (n_frames, 1) or (n_frames)
        The sample weights, typically coming from a metadynamics run. Has to have the same length
        as the :py:obj:`projected_trajectories` argument.

    proj_labels : either string or list of strings or (experimental PyEMMA featurizer)
        The projection plots will get this paramter for labeling their yaxis. If a str is
        provided, that will be the base name proj_labels='%s_%u'%(proj_labels,ii) for each
        projection. If a list, the list will be used. If not enough labels are there
        the module will complain

    n_overlays : int, default is 1
        The number of structures that will be simultaneously displayed as overlays for every sampled point of the FES.
        This parameter can seriously slow down the method, it is currently limited to a maximum value of 50

    atom_selection : string or iterable of integers, default is None
        The geometries of the original trajectory files will be filtered down to these atoms. It can be any DSL string
        that   :obj:`mdtraj.Topology.select` could understand or directly the iterable of integers.
        If :py:obj`MD_trajectories` is already a (list of) md.Trajectory objects, the atom-slicing can take place
        before calling this method.

    sample_kwargs : dictionary of named arguments, optional
        named arguments for the function :obj:`visualize.sample`. Non-expert users can safely ignore this option. Examples
        are superpose or proj_

    Returns
    --------

    widgetbox:
        :obj:`ipywidgets.HBox` containing both the NGLWidget (ngl_wdg) and the interactive figure. It also
        contains the extra attributes
        # TODO reshape this docstring
         ax :
        :obj:`pylab.Axis` object
    fig :
        :obj:`pylab.Figure` object
    ngl_wdg :
        :obj:`nglview.NGLWidget`
    data_sample:
        numpy ndarray of shape (n, n_sample) with the position of the dots in the plot
    geoms:
        :obj:`mdtraj.Trajectory` object with the geometries n_sample geometries shown by the ngl_wdg

    """

    # Prepare the overlay option
    n_overlays = _np.min([n_overlays,50])
    if n_overlays>1:
        keep_all_samples = True
    else:
        keep_all_samples = False

    # Prepare for 1D case
    proj_idxs = _bmutils.listify_if_int(proj_idxs)

    data_sample, geoms, data = generate.sample(MD_trajectories, MD_top, projected_trajectories,
                                               atom_selection=atom_selection,
                                               proj_idxs=proj_idxs,
                                               n_points=n_sample,
                                               return_data=True,
                                               n_geom_samples=n_overlays,
                                               keep_all_samples=keep_all_samples,
                                               proj_stride=proj_stride
                                               )

    data = _np.vstack(data)
    if weights is not None:
        weights = _bmutils.listify_if_not_list(weights)
        if weights[0].ndim == 1:
            weights = [_np.array(iw, ndmin=2).T for iw in weights]
        weights = _np.vstack(weights).squeeze()

    _plt.ioff()
    ax, FES_data, edges = _plot_ND_FES(data[:,proj_idxs],
                                       _bmutils.labelize(proj_labels, proj_idxs),
                                       weights=weights, bins=nbins)
    _plt.ion()

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

    ngl_wdg, axes_wdg = sample(data_sample, geoms, ax, clear_lines=False, **sample_kwargs)
    ngl_wdg._set_size(*['%fin' % inches for inches in ax.get_figure().get_size_inches()])
    ax.figure.tight_layout()
    axes_wdg.canvas.set_window_title("FES")
    outbox = _linkutils.MolPXHBox([ngl_wdg, axes_wdg.canvas])
    _linkutils.auto_append_these_mpx_attrs(outbox, geoms, ax, _plt.gcf(), ngl_wdg, axes_wdg, data_sample)

    return outbox

def _plot_ND_FES(data, ax_labels, weights=None, bins=50, figsize=(4,4)):
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

    _plt.figure(figsize=figsize)
    ax = _plt.gca()
    idata = _np.vstack(data)
    ax.set_xlabel(ax_labels[0])
    if idata.shape[1] == 1:
        h, edges = _np.histogramdd(idata, weights=weights, bins=bins, normed=True)
        FES_data = -_np.log(h)
        FES_data -= FES_data.min()
        ax.plot(edges[0][:-1], FES_data)
        ax.set_ylabel('$\Delta G / \kappa T $')

    elif idata.shape[1] == 2:
        _plot_free_energy(idata[:,0], idata[:,1], weights=weights, nbins=bins, ax=ax,
                          cmap='nipy_spectral'
                           )
        ax.set_ylabel(ax_labels[1])
        edges, FES_data = [None], None
        # TODO: retrieve the actual edges from pyemma's "plot_free_energy"'s axes
    else:
        raise NotImplementedError('Can only plot 1D or 2D FESs, but data has %s columns' % _np.shape(idata)[0])

    return ax, FES_data, edges

def traj(MD_trajectories,
         MD_top, projected_trajectories,
         max_frames=1e4,
         stride=1,
         proj_stride=1,
         proj_idxs=[0,1],
         proj_labels='proj',
         plot_FES=False,
         weights=None,
         panel_height = 1,
         sharey_traj=True,
         dt = 1.0,
         tunits = 'frames',
         traj_selection = None,
         projection = None,
         n_feats = 1,
         ):
    r"""Link one or many :obj:`projected trajectories`, [Y_0(t), Y_1(t)...], with the :obj:`MD_trajectories` that
    originated them. Optionally plot also the resulting FES.

    Parameters
    -----------

    MD_trajectories : str, or list of strings with the filename(s) the the molecular dynamics (MD) trajectories.
        Any file extension that :py:obj:`mdtraj` (.xtc, .dcd etc) can read is accepted.

        Alternatively, a single :obj:`mdtraj.Trajectory` object or a list of them can be given as input.

    MD_top : str to topology filename or directly :obj:`mdtraj.Topology` object

    projected_trajectories : numpy ndarray (or list thereof) of shape (n_frames, n_dims) with the time-series
    of the projection(s) that want to be explored. Alternatively, strings or list of string with .npy or ascii filenames
    (.dat, .txt etc)
    NOTE: molpx assumes that there is no time column.

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

    weights : ndarray(n_frames), default = None
        sample weights. By default all samples have the same weight (used for FES calculation only)

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
        The projected coordinates may come from a variety of sources. When working with :obj:`pyemma` a number of objects
        might have generated this projection, like a
            :obj:`pyemma.coordinates.transform.TICA` or a
            :obj:`pyemma.coordinates.transform.PCA`
        Pass this object along and observe and the features that are most correlated with the projections
        will be plotted for the active trajectory, allowing the user to establish a visual connection between the
        projected coordinate and the original features (distances, angles, contacts etc)

    n_feats : int, default is 1
        If a :obj:`projection` is passed along, the first n_feats features that most correlate the
        the projected trajectories will be represented, both in form of trajectories feat vs t as well as in
        the ngl_wdg. If :obj:`projection` is None, :obj:`nfeats`  will be ignored.

    Returns
    ---------

    ax, ngl_wdg, data_sample, geoms
        return _plt.gca(), _plt.gcf(), widget, geoms

    ax :
        :obj:`pylab.Axis` object
    fig :
        :obj:`pylab.Figure` object
    ngl_wdg :
        :obj:`nglview.NGLWidget`
    geoms:
        :obj:`mdtraj.Trajectory` object with the geometries n_sample geometries shown by the ngl_wdg


    """
    smallfontsize = int(_rcParams['font.size'] / 1.5)
    proj_idxs = _bmutils.listify_if_int(proj_idxs)

    # Parse input
    Y = _bmutils.data_from_input(projected_trajectories)
    data = [iY[:,proj_idxs] for iY in Y]

    MD_trajectories = _bmutils.listify_if_not_list(MD_trajectories)

    assert len(data) == len(MD_trajectories), "Mismatch between number of MD-trajectories " \
                                           "and projected trajectores %u vs %u"%(len(MD_trajectories), len(data))

    traj_selection = _bmutils.listify_if_int(traj_selection)
    if traj_selection is None:
        traj_selection = _np.arange(len(data))
    assert _np.max(traj_selection) < len(data), "Selected up to traj. nr. %u via the parameter traj_selection, " \
                                                "but only provided %u trajs"%(_np.max(traj_selection), len(data))

    n_trajs = len(traj_selection)
    n_projs = len(proj_idxs)
    # Get the geometries as usable mdtraj.Trajectory
    geoms = []
    for igeom, idata in zip(MD_trajectories, data):
        if isinstance(igeom, _md.Trajectory):
            geoms.append(igeom[::proj_stride])
        else: # let mdtraj fail
            geoms.append(_md.load(igeom, stride=proj_stride, top=MD_top))

        # Do the projected trajectory and the data match?
        assert geoms[-1].n_frames == len(idata), (geoms[-1].n_frames, len(idata))

    # Stride to avoid representing huge vectors
    times = []
    for proj_counter in range(len(data)):
        time = _np.arange(data[proj_counter].shape[0])*dt*proj_stride
        if len(time[::stride]) > max_frames:
            stride = int(_np.floor(data[proj_counter].shape[0]/max_frames))

        times.append(time[::stride])
        data[proj_counter] = data[proj_counter][::stride]
        geoms[proj_counter] = geoms[proj_counter][::stride]


    # For axes-cosmetics later on
    tmax, tmin = _np.max([time[-1] for time in times]), _np.min([time[0] for time in times])
    ylims = _np.zeros((2, n_projs))
    for proj_counter, __ in enumerate(proj_idxs):
        ylims[0, proj_counter] = _np.min([idata[:,proj_counter].min() for idata in data])
        ylims[1, proj_counter] = _np.max([idata[:,proj_counter].max() for idata in data])
    
    ylabels = _bmutils.labelize(proj_labels, proj_idxs)

    # Do we have usable projection information?
    corr_dicts = [[]]*n_trajs
    if projection is not None:
        corr_dicts = [_bmutils.most_corr(projection, geoms=igeom, proj_idxs=proj_idxs, n_args=n_feats)
                      for igeom in geoms]
        if corr_dicts[0]["feats"] != []:
            colors = _bmutils.matplotlib_colors_no_blue()
            colors = [colors[ii] for ii in proj_idxs]
        else:
            n_feats=0
    else:
        # squash whatever input we had if the projection-info input wasn't actually usable
        n_feats = 0

    _plt.ioff()
    n_rows = n_trajs*n_projs+n_trajs*n_projs*n_feats
    if n_rows == 1:
        panel_height = _np.max((panel_height, 2) )

    myfig, myax = _plt.subplots(n_rows,1, sharex=True, figsize=(5, n_rows*panel_height),
                                squeeze=True
                                )
    if n_rows == 1:
        myax = _np.array(myax, ndmin=1)

    # Initialize some things
    ngl_wdg_list = []
    axes_iterator = iter(myax)
    linked_data_arrays = []
    linked_axes_wdgs = []
    vboxes_left = [_Button(description='NGL widgets', layout=_Layout(width='100%'))]
    for traj_idx, time, jdata, jgeom, jcorr_dict in zip(traj_selection,
                                                        [times[jj] for jj in traj_selection],
                                                        [data[jj] for jj in traj_selection],
                                                        [geoms[jj] for jj in traj_selection],
                                                        [corr_dicts[jj] for jj in traj_selection]
                                                        ):
        ngl_wdg = _nglwidget_wrapper(jgeom)
        ngl_wdg_list.append(ngl_wdg)
        vboxes_left.append(_VBox([ngl_wdg], layout=_Layout(border='solid')))
        ngl_wdg._set_size(*["%fin"%ll for ll in myfig.get_size_inches()])

        for proj_counter, idata in enumerate(jdata.T):
            iax = next(axes_iterator)
            data_sample = _np.vstack((time, idata)).T
            linked_data_arrays.append(data_sample)
            iax.plot(time, idata)
            ngl_wdg, axes_traj_wdg = sample(data_sample, jgeom.superpose(geoms[0]), iax,
                                               clear_lines=False,
                                               ngl_wdg=ngl_wdg,
                                               crosshairs='v',
                                               exclude_coord=1
                                               )
            linked_axes_wdgs.append(axes_traj_wdg)

            # Axis-Cosmetics
            iax.set_ylabel(ylabels[proj_counter])
            iax.set_xlim([tmin, tmax])
            _add_y2_label(iax, 'traj %u' % traj_idx)
            if sharey_traj:
                iax.set_ylim(ylims[:,proj_counter]+[-1,1]*_np.diff(ylims[:,proj_counter])*.1)

            # Now go over the correlated features
            for ifeat in range(n_feats):
                iax = next(axes_iterator)
                ifeat_val = jcorr_dict["feats"][proj_counter][:, ifeat]
                ilabel = jcorr_dict["labels"][proj_counter][ifeat]
                icol = colors[proj_counter]

                # Plot
                lines = iax.plot(time, ifeat_val,
                                 color=icol
                                 )[0]
                iax.set_ylabel('\n'.join(_bmutils.re_warp(ilabel, 16)), fontsize=smallfontsize)
                _add_y2_label(iax, 'traj %u' % traj_idx)

                # Add the correlation value
                iax.legend([lines], ['Corr(feat|%s)=%2.1f' % (ylabels[proj_counter],
                                                              jcorr_dict["vals"][proj_counter][ifeat])],
                           fontsize=smallfontsize, loc='best', frameon=False)

                # Link widget
                fdata_sample = _np.vstack((time, ifeat_val)).T
                ngl_wdg, axes_traj_corr_wdg = sample(fdata_sample, jgeom.superpose(geoms[0]), iax,
                                           clear_lines=False, ngl_wdg=ngl_wdg,
                                           crosshairs='v',
                                           exclude_coord=1,
                                           )
                linked_data_arrays.append(fdata_sample)
                linked_axes_wdgs.append(axes_traj_corr_wdg)
                # Add visualization (let the method decide if it's possible or not)
                ngl_wdg = _bmutils.add_atom_idxs_widget([jcorr_dict["atom_idxs"][proj_counter][ifeat]], ngl_wdg,
                                                        color_list=[icol])

    # Last of axis cosmetics
    iax.set_xlabel('t / %s'%tunits)
    myfig.tight_layout()

    # Widget cosmetics
    fig_w, fig_h = myfig.get_size_inches()
    widget_h = fig_h / len(traj_selection)
    [iwd._set_size('4in', '%fin' % widget_h) for iwd in ngl_wdg_list]
    [iwd.center() for iwd in ngl_wdg_list]


    mpx_wdg_box = _linkutils.MolPXHBox([_VBox(vboxes_left,
                                              layout=_Layout(width='%sin'%fig_w, height='%sin'%fig_h,
                                                      #align_items='baseline'
                                                             )),
                                        axes_traj_wdg.canvas]
                                       )

    # Append the linked objects
    _linkutils.auto_append_these_mpx_attrs(mpx_wdg_box, _plt.gcf(),
                                            list(myax.flat),
                                            linked_data_arrays,
                                            ngl_wdg_list,
                                            linked_axes_wdgs,
                                            [geoms[jj] for jj in traj_selection]
                                            )

    if plot_FES:
        FES_HBox = FES([MD_trajectories[jj] for jj in traj_selection],
                               MD_top,
                               [Y[jj] for jj in traj_selection],
                               proj_idxs=proj_idxs,
                               proj_labels=ylabels,
                               proj_stride=proj_stride,
                               weights=weights
                               )

        FES_HBox.linked_ngl_wdgs[0].center()
        FES_HBox.linked_figs[0].set_size_inches(fig_w, h=fig_w)
        FES_HBox.linked_ngl_wdgs[0]._set_size("%sin"%fig_w, "%sin"%fig_w)
        FES_HBox.linked_figs[0].tight_layout()

        mpx_wdg_box = _linkutils.MolPXVBox([mpx_wdg_box, FES_HBox])

    _plt.ion()

    return mpx_wdg_box

def correlations(correlation_input,
                 geoms=None,
                 proj_idxs=None,
                 feat_name=None,
                 widget=None,
                 proj_color_list=None,
                 n_feats=1,
                 verbose=False,
                 featurizer=None):
    r""" Provide a visual and textual representation of the linear correlations between projected coordinates (PCA, TICA) and original features.

    Parameters
    ---------

    correlation_input : anything
        Something that could, in principle, be a :obj:`pyemma.coordinates.transformer`,
        like a TICA, PCA object or directly a correlation matrix, with a row for each feature and a column
        for each projection, very much like the :obj:`feature_TIC_correlation` of the TICA object of pyemma.

    geoms : None or :obj:`mdtraj.Trajectory`, default is None
        The values of the most correlated features will be returned for the geometries in this object. If widget is
        left to its default, None, :obj:`correlations` will create a new widget and try to show the most correlated
        features on top of the widget.

    widget : None or :obj:`nglview NGLWidget`
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

    proj_color_list : list, default is None
        projection specific list of colors to provide the representations with. The default None yields blue.
        In principle, the list can contain one color for each projection (= as many colors as len(proj_idxs)
        but if your list is short it will just default to the last color. This way, proj_color_list=['black'] will paint
        all black regardless len(proj_idxs)

    proj_idxs : None, or int, or iterable of integers, default is None
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

    Returns
    -------
    corr_dict and ngl_wdg

    corr_dict:
        A dictionary with items:

        idxs :
            List of length len(proj_idxs) with lists of length n_feat with the idxs of the most correlated features

        vals :
            List of length len(proj_idxs) with lists of length n_feat with the corelation values of the
            most correlated features

        labels :
            List of length len(proj_idxs) with lists of length n_feat with the labels of the
            most correlated features

        feats :
            If an :obj:`mdtraj.Trajectory` is passed as an :obj:`geom` argument, the most correlated features will
            be evaluated for that geom and returned as list of length len(proj_idxs) with arrays with shape

        atom_idxs :
            List of length len(proj_idxs) each with an nd.array of shape (nfeat, m), where m is the number of atoms needed
            to describe each feature (1 of cartesian, 2 for distances, 3 for angles, 4 for dihedrals)

        info :
            List of length len(proj_idxs) with lists of length n_feat with strings describing the correlations

    widget :
        obj:`nglview.NGLwidget` with the correlations visualized on top of it

    """
    # todo consider kwargs for most_corr
    corr_dict = _bmutils.most_corr(correlation_input,
                                   geoms=geoms, proj_idxs=proj_idxs, feat_name=feat_name, n_args=n_feats,
                                   featurizer=featurizer
                                )

    # Create ngl_viewer widget
    if geoms is not None and widget is None:
        widget = _nglwidget_wrapper(_bmutils.superpose_to_most_compact_in_list(True, [geoms])[0])

    if proj_color_list is None:
        proj_color_list = ['blue'] * len(corr_dict["idxs"])
    elif isinstance(proj_color_list, list) and len(proj_color_list)<len(corr_dict["idxs"]):
        proj_color_list += [proj_color_list[-1]] * (len(corr_dict["idxs"]) - len(proj_color_list))
    elif not isinstance(proj_color_list, list):
        raise TypeError("parameter proj_color_list should be either None or a list, not %s of type %s"%(proj_color_list, type(proj_color_list)))

    if len(corr_dict["atom_idxs"]) == 0:
        _warnings.warn("Not enough information to display atoms on widget. Turning verbose on.")
        verbose = True

    # Add the represenation
    if widget is not None:
        for idxs, icol in zip(corr_dict["atom_idxs"], proj_color_list):
            _bmutils.add_atom_idxs_widget(idxs, widget, color_list=[icol])

    if verbose:
        for ii, line in enumerate(corr_dict["info"]):
            print('%s is most correlated with '%(line["name"] ))
            for line in line["lines"]:
                # TODO: this is for when tica is there but no featurizer is there
                if widget is not None and len(corr_dict["atom_idxs"]) != 0:
                    line += ' (in %s in the widget)'%(proj_color_list[ii])
                print(line)

    return corr_dict, widget

def feature(feat,
            widget,
            idxs=0,
            color_list=None,
            **kwargs
               ):
    r"""
    Provide a visual representation of a PyEMMA feature. PyEMMA's features are found as a list of the MDFeaturizers's
    active_features attribute

    Parameters
    ----------

    featurizer : py:obj:`_MDFeautrizer`
        A PyEMMA MDFeaturizer object (either a feature or a featurizer, works with both)

    widget : None or nglview widget
        Provide an already existing widget to visualize the correlations on top of. This is only for expert use,
        because no checks are done to see if :obj:`correlation_input` and the geometry contained in the
        widget **actually match**. Use with caution.

    idxs: int or iterable of integers, default is 0
        Features can have many contributions, e.g. a distance feature can include many distances. Use this parameter
        to control which one of them gets represented.

    color_list: list, default is None
        list of colors to represent each feature in feat_idxs. The default None yields blue for everything.
        In principle, the list can contain one color for each projection (= as many colors as len(feat_idxs)
        but if your list is short it will just default to the last color. This way, color_list=['black'] will paint
        all black regardless len(proj_idxs)

    **kwargs : optional keyword arguments for _bmutils.add_atom_idxs_widget
        currently, only "radius" is left for the user to determine

    Returns :
    --------

    widget :
        the input widget with the features in :py:obj:`idxs` represented as either distances (for distance features)
        or "spacefill" spheres (for angular features)

    """

    idxs = _bmutils.listify_if_int(idxs)
    atom_idxs = _bmutils.atom_idxs_from_feature(feat)[idxs]

    if color_list is None:
        color_list = ['blue'] * len(idxs)

    elif isinstance(color_list, list) and len(color_list)<len(idxs):
        color_list += [color_list[-1]] * (len(idxs) - len(color_list))
    elif not isinstance(color_list, list):
        raise TypeError("parameter color_list should be either None "
                        "or a list, not %s of type %s"%(color_list, type(color_list)))

    # Add the represenation
    _bmutils.add_atom_idxs_widget(atom_idxs, widget, color_list=color_list, **kwargs)

    return widget

def sample(positions, geom, ax,
           plot_path=False,
           clear_lines=True,
           n_smooth = 0,
           ngl_wdg=None,
           superpose=True,
           projection = None,
           n_feats = 1,
           sticky=False,
           list_of_repr_dicts=None,
           color_list=None,
           **link_ax2wdg_kwargs
           ):

    r"""
    Visualize the geometries in :obj:`geom` according to the data in :obj:`positions` on an existing matplotlib axes :obj:`ax`

    Use this method when the array of positions, the geometries, the axes (and the ngl_wdg, optionally) have already been
    generated elsewhere.

    Parameters
    ----------
    positions : numpy nd.array of shape (n_frames, 2)
        Contains the position associated with each frame in :obj:`geom` in that order

    geom : :obj:`mdtraj.Trajectory` objects or a list thereof.
        The geometries associated with the the :obj:`positions`. Hence, all have to have the same number of n_frames

    ax : matplotlib.pyplot.Axes object
        The axes to be linked with the nglviewer ngl_wdg

    plot_path : bool, default is False
        whether to draw a line connecting the positions in :obj:`positions`

    clear_lines : bool, default is True
        whether to clear all the lines that were previously drawn in :obj:`ax`

    n_smooth : int, default is 0,
        if n_smooth > 0, the shown geometries and paths will be smoothed out by 2*n frames.
        See :obj:`molpx._bmutils.smooth_geom` for more information

    ngl_wdg : None or existing nglview ngl_wdg
        you can provide an already instantiated nglviewer ngl_wdg here (avanced use)

    superpose : boolean, default is True
        The geometries in :obj:`geom` may or may not be oriented, depending on where they were generated.
        Since this method is mostly for visualization purposes, the default behaviour is to orient them all to
        maximally overlap with the most compact frame available

    projection : object that generated the projection, default is None
        The projected coordinates may come from a variety of sources. When working with :obj:`pyemma` a number of objects
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
        the ngl_wdg. If :obj:`projection` is None, :obj:`nfeats`  will be ignored.

    sticky : boolean, default is False,
        If set to True, the ngl_wdg the generated visualizations will be sticky in that they do not disappear with
        the next click event. Particularly useful for represeting more minima simultaneously.

    color_list : None or list of len(pos)
        The colors with which the sticky frames will be plotted.
        Can by anything that yields matplotlib.colors.is_color_like == True

    list_of_repr_dicts : None or list of dictionaries having at least keys 'repr_type' and 'selection' keys.
        Other **kwargs are currently ignored but will be implemented in the future (see nglview.add_representation
        for more info). Only active for sticky widgets

    link_ax2wdg_kwargs: dictionary of named arguments, optional
        named arguments for the function :obj:`_link_ax_w_pos_2_nglwidget`, which is the one that internally
        provides the interactivity. Non-expert users can safely ignore this option.

    Returns
    --------

    ngl_wdg : :obj:`nglview.NGLWidget`

    axes_wdg: obj:`matplotlib.Axes.AxesWidget`

    """

    if not sticky:
        return _sample(positions, geom, ax,
                       plot_path = plot_path,
                       clear_lines = clear_lines,
                       n_smooth = n_smooth,
                       ngl_wdg= ngl_wdg,
                       superpose = superpose,
                       projection = projection,
                       n_feats = n_feats,
                       **link_ax2wdg_kwargs)
    else:

        if isinstance(geom, _md.Trajectory):
            geom=[geom]

        # The method takes care of whatever superpose
        geom = _bmutils.superpose_to_most_compact_in_list(superpose, geom)

        if color_list is None:
            sticky_colors_hex = ['Element' for ii in range(len(positions))]
        elif isinstance(color_list, list) and len(color_list) == len(positions):
            sticky_colors_hex = [_to_hex(cc) for cc in color_list]
        elif isinstance(color_list, str) and color_list.lower().startswith('rand'):
            # TODO: create a path through the colors that maximizes distance between averages (otherwise some colors
            # are too close
            cmap = _get_cmap('rainbow')
            cmap_table = _np.linspace(0, 1, len(positions))
            sticky_colors_hex = [_rgb2hex(cmap(ii)) for ii in _np.random.permutation(cmap_table)]
        else:
            raise TypeError('argument color_list should be either None, "random", or a list of len(pos)=%u, '
                            'instead of type %s and len %u' % (len(positions), type(color_list), len(color_list)))
        sticky_rep = 'cartoon'
        if geom[0].top.n_residues < 10:
            sticky_rep = 'ball+stick'
        if list_of_repr_dicts is None:
            list_of_repr_dicts = [{'repr_type': sticky_rep, 'selection': 'all'}]

        # Now instantiate the ngl_wdg
        ngl_wdg = _nglwidget_wrapper(None)
        # Prepare Geometry_in_widget_list
        ngl_wdg._GeomsInWid = [_linkutils.GeometryInNGLWidget(igeom, ngl_wdg,
                                                          color_molecule_hex= cc,
                                                          list_of_repr_dicts=list_of_repr_dicts) for igeom, cc in zip(_bmutils.transpose_geom_list(geom), sticky_colors_hex)]

        axes_wdg = _linkutils.link_ax_w_pos_2_nglwidget(ax,
                                   positions,
                                   ngl_wdg,
                                   directionality='a2w',
                                   dot_color = 'None',
                                   **link_ax2wdg_kwargs
                                   )

        return ngl_wdg, axes_wdg

def _sample(positions, geoms, ax,
            plot_path=False,
            clear_lines=True,
            n_smooth = 0,
            ngl_wdg=None,
            superpose=True,
            projection = None,
            n_feats = 1,
            **link_ax2wdg_kwargs
            ):

    r"""
    Visualize the geometries in :obj:`geoms` according to the data in :obj:`positions` on an existing matplotlib axes :obj:`ax`

    Use this method when the array of positions, the geometries, the axes (and the ngl_wdg, optionally) have already been
    generated elsewhere.

    Parameters
    ----------
    positions : numpy nd.array of shape (n_frames, 2)
        Contains the position associated with each frame in :obj:`geoms` in that order

    geoms : :obj:`mdtraj.Trajectory` objects or a list thereof.
        The geometries associated with the the :obj:`positions`. Hence, all have to have the same number of n_frames

    ax : matplotlib.pyplot.Axes object
        The axes to be linked with the nglviewer ngl_wdg

    plot_path : bool, default is False
        whether to draw a line connecting the positions in :obj:`positions`

    clear_lines : bool, default is True
        whether to clear all the lines that were previously drawn in :obj:`ax`

    n_smooth : int, default is 0,
        if n_smooth > 0, the shown geometries and paths will be smoothed out by 2*n frames.
        See :any:`bmutils.smooth_geom` for more information

    ngl_wdg : None or existing nglview ngl_wdg
        you can provide an already instantiated nglviewer ngl_wdg here (avanced use)

    superpose : boolean, default is True
        The geometries in :obj:`geoms` may or may not be oriented, depending on where they were generated.
        Since this method is mostly for visualization purposes, the default behaviour is to orient them all to
        maximally overlap with the frame that is most compact (=a heuristic to identify folded frames)
    projection : object that generated the projection, default is None
        The projected coordinates may come from a variety of sources. When working with :obj:`pyemma` a number of objects
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
        the ngl_wdg. If :obj:`projection` is None, :obj:`nfeats`  will be ignored.

    link_ax2wdg_kwargs: dictionary of named arguments, optional
        named arguments for the function :obj:`_link_ax_w_pos_2_nglwidget`, which is the one that internally
        provides the interactivity. Non-expert users can safely ignore this option.

    Returns
    --------

    ngl_wdg : :obj:`nglview.NGLWidget`

    axes_wdg :obj:`matplotlib.Axes.AxesWidget`

    """

    assert isinstance(geoms, (list, _md.Trajectory))

    # Dow I need to smooth things out?
    if n_smooth > 0:
        if isinstance(geoms, _md.Trajectory): # smoothing only makes sense for paths, and paths cannot be lists at the moment
            geoms, positions = _bmutils.smooth_geom(geoms, n_smooth, geom_data=positions)
            mean_smooth_radius = _np.abs(_np.diff(positions, axis=0).mean(0) * n_smooth)
            band_width = 2 * mean_smooth_radius
    else:
        band_width = None

    # Now we can listify the geoms object
    if isinstance(geoms, _md.Trajectory):
        geoms = [geoms]

    geoms = _bmutils.superpose_to_most_compact_in_list(superpose, geoms)

    # Create ngl_viewer ngl_wdg
    if ngl_wdg is None:
        ngl_wdg = _nglwidget_wrapper(geoms[0])
        for igeom in geoms[1:]:
            ngl_wdg = _nglwidget_wrapper(igeom, ngl_wdg=ngl_wdg)
    else:
        ngl_wdg = ngl_wdg

    if clear_lines == True:
        [ax.lines.pop() for ii in range(len(ax.lines))]
    # Plot the path on top of it
    if plot_path:
        ax.plot(positions[:,0], positions[:,1], '-g', lw=3)

    # Link the axes ngl_wdg with the ngl ngl_wdg
    axes_wdg = _linkutils.link_ax_w_pos_2_nglwidget(ax,
                                         positions,
                                         ngl_wdg,
                                        band_width=band_width,
                                        **link_ax2wdg_kwargs
                                        )

    # Do we have usable projection information?
    if projection is not None:
        corr_dict = _bmutils.most_corr(projection, n_args=n_feats)
        if corr_dict["labels"] != []:
            iproj = _bmutils.get_ascending_coord_idx(positions)
            for ifeat in range(n_feats):
                ilabel = corr_dict["labels"][iproj][ifeat]
                print(ilabel)
                ngl_wdg = _bmutils.add_atom_idxs_widget([corr_dict["atom_idxs"][iproj][ifeat]], ngl_wdg,
                                            color_list=['green']
                                            )

    return ngl_wdg, axes_wdg

