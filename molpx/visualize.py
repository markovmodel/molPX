
__author__ = 'gph82'

from pyemma.plots import plot_free_energy as _plot_free_energy
import numpy as _np
from matplotlib.widgets import AxesWidget as _AxesWidget
from matplotlib.figure import Figure as _mplFigure
from matplotlib.axes import Axes as _mplAxes

from matplotlib.cm import get_cmap as _get_cmap
from matplotlib.colors import rgb2hex as _rgb2hex, to_hex as _to_hex

from . import generate as _generate
from . import _bmutils
from . import _linkutils

from matplotlib import rcParams as _rcParams
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
        verbose=False,
        **sample_kwargs):
    r"""
    Return a molecular visualization widget connected with a free energy plot.

    Parameters
    ----------

    MD_trajectories : str, or list of strings with the filename(s) the the molecular dynamics (MD) trajectories.
        Any file extension that :obj:`mdtraj` (.xtc, .dcd, etc) can read is accepted.
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
        resolution of the "click"-action. (And the longer it will take to generate the plot)

    proj_stride : int, default is 1
        Stride value that was used in the :obj:`projected_trajectories` relative to the :obj:`MD_trajectories`
        If the original :obj:`MD_trajectories` were stored every 5 ps but the projected trajectories were stored
        every 50 ps, :obj:`proj_stride` = 10 has to be provided, otherwise an exception will be thrown informing
        the user that the :obj:`MD_trajectories` and the :obj:`projected_trajectories` have different number of frames.

    weights : iterable of floats (or list thereof) each of shape (n_frames, 1) or (n_frames)
        The sample weights can come from a metadynamics run or an MSM-object, e.g.
        via the method :obj:`pyemma.msm.BayesianMSM.trajectory_weights`.
        Has to have the same length as the :py:obj:`projected_trajectories`

    proj_labels : either string or list of strings or (experimental) PyEMMA featurizer
        The projection plots will get this parameter for labeling their yaxis. If a str is
        provided, that will be the base name ``proj_labels='%s_%u'%(proj_labels,ii)`` for each
        projection. If :obj:`proj_labels` is a list, the list will be used as is. If there are not enough labels,
        the module will complain

    n_overlays : int, default is 1
        The number of structures that will be simultaneously displayed as overlays for every sampled point of the FES.
        This parameter can seriously slow down the method, it is currently limited to a maximum value of 50

    atom_selection : string or iterable of integers, default is None
        The geometries of the original trajectory files will be filtered down to these atoms. It can be any DSL string
        that :obj:`mdtraj.Topology.select` could understand or directly the iterable of integers.
        If :obj:`MD_trajectories` is already a (list of) :obj:`mdtraj.Trajectory` objects, the atom-slicing can be
        done by the user place before calling this method.

    verbose : bool, default is False
        Be verbose while computing the FES

    sample_kwargs : dictionary of named arguments, optional
        named arguments for the function :obj:`molpx.visualize.sample`. Non-expert users can safely ignore this option.
        Examples are :obj:`superpose` or :obj:`proj_idxs`

    Returns
    -------

    widgetbox :
        A :obj:`molpx._linkutils.MolPXHBox`-object.

        It contains the :obj:`~nglview.NGLWidget` and the :obj:`~matplotlib.widgets.AxesWidget` (which is
        responsible for the interactive figure). It is child-class of the :obj:`ipywidgets.HBox`-class and
        has been monkey-patched to have the following extra attributes so that the user has access to all the
        information being displayed.

        linked_axes :
            list with all the :obj:`pyplot.Axis`-objects contained in the :obj:`widgetbox`

        linked_ax_wdgs :
            list with all the :obj:`matplotlib.widgets.AxesWidget`objects contained in the :obj:`widgetbox`

        linked_figs :
            list with all the :obj:`pyplot.Figure`-objects contained in the :obj:`widgetbox`

        linked_ngl_wdgs :
            list with all the :obj:`nglview.NGLWidget`-objects contained in the :obj:`widgetbox`

        linked_data_arrays :
            list with all the numpy ndarrays contained in the :obj:`widgetbox`

        linked_mdgeoms:
            list with all the :obj:`mdtraj.Trajectory`-objects contained in the :obj:`widgetbox`

    """
    from matplotlib import pyplot as _plt
    # Prepare the overlay option
    n_overlays = _np.min([n_overlays,50])
    if n_overlays>1:
        keep_all_samples = True
    else:
        keep_all_samples = False

    # Prepare for 1D case
    proj_idxs = _bmutils.listify_if_int(proj_idxs)

    data_sample, geoms, data = _generate.sample(MD_trajectories, MD_top, projected_trajectories,
                                               atom_selection=atom_selection,
                                               proj_idxs=proj_idxs,
                                               n_points=n_sample,
                                               return_data=True,
                                               n_geom_samples=n_overlays,
                                               keep_all_samples=keep_all_samples,
                                               proj_stride=proj_stride,
                                               verbose=verbose
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
def _box_me(tuple_in, auto_resize=True):
    r"""
    A wrapper that tries to put in an HBox whatever it s in
    the tuple_in as long as it of  the following types:
     * nglwidget
     * matplotlib axes widget
     * matplotlib figure
     * matplotlib axes

    If it does not succeed, it will let you know without throwing an exception

    auto_resize : bool, default is True
        Resize everything to the average size (wxh) of the input objects

    :return: obj:IPywdigets.HBox:, if possible
    """
    # TODO THIS IS UNUSED
    # TODO EITHER USE IT OR REMOVE IT BEFORE RELEASING
    widgets_and_canvas = []
    size_inches = []
    for obj in tuple_in:
        if isinstance(obj, _nglview.NGLWidget):
            toappend = obj
        elif isinstance(obj, (_AxesWidget, _mplFigure)):
            toappend = obj.canvas
        elif isinstance(obj, _mplAxes):
            toappend = obj.figure.canvas
        else:
            _warnings.warn("\nSorry, object %s of type %s is unboxable at the moment"%(obj, type(obj)))
            return
        widgets_and_canvas.append(toappend)

    # We ve collected everything, now unique and get sizes:
    tuple_out = []
    for obj in widgets_and_canvas:
        if obj not in tuple_out:
            tuple_out.append(obj)
            try:
                size_inches.append(obj.figure.get_size_inches())
            except AttributeError:
                pass

    size_inches = _np.array(_np.vstack(size_inches), ndmin=2).T.mean(1)

    if auto_resize:
        for obj in tuple_out:
            if isinstance(obj, _nglview.NGLWidget):
                obj._set_size("%fin"%size_inches[0],
                              "%fin"%size_inches[1])
            elif isinstance(obj, _mplFigure):
                obj.set_size_inches(*size_inches)

    return _linkutils._HBox(tuple_out)

def _plot_ND_FES(data, ax_labels, weights=None, bins=50, figsize=(4,4)):
    r""" A wrapper for pyemmas FESs plotting function that can also plot 1D

    Parameters
    ----------

    data : list of numpy nd.arrays

    ax_labels : list

    Returns
    -------

    ax : :obj:`pyplot.Axis` object

    FES_data : numpy nd.array containing the FES (only for 1D data)

    edges : tuple containimg the axes along which FES is to be plotted (only in the 1D case so far, else it's None)

    """
    from matplotlib import pyplot as _plt
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
         input_feature_traj = None,
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
        Note: the data used for the FES will only include these trajectories

    projection : object that generated the projection, default is None
        The projected coordinates may come from a variety of sources. When working with :obj:`pyemma` a number of objects
        might have generated this projection, like a
            :obj:`pyemma.coordinates.transform.TICA` or a
            :obj:`pyemma.coordinates.transform.PCA`
        Pass this object along and observe the features that are most correlated with the projections
        will be plotted for the active trajectory, allowing the user to establish a visual connection between the
        projected coordinate and the original features (distances, angles, contacts etc)
        These trajectories will be re-computed by applyiing
        :obj:`projection.transform(MD_trajectories)', unless :obj:`input_feature_traj` is parsed

    input_feature_traj : TODO

    n_feats : int, default is 1
        If a :obj:`projection` is passed along, the first n_feats features that most correlate the
        the projected trajectories will be represented, both in form of trajectories feat vs t as well as in
        the ngl_wdg. If :obj:`projection` is None, :obj:`nfeats`  will be ignored.

    Returns
    ---------

    ax, ngl_wdg, data_sample, geoms
        return _plt.gca(), _plt.gcf(), widget, geoms

    ax :
        :obj:`pyplot.Axis` object
    fig :
        :obj:`pyplot.Figure` object
    ngl_wdg :
        :obj:`nglview.NGLWidget`
    geoms:
        :obj:`mdtraj.Trajectory` object with the geometries n_sample geometries shown by the ngl_wdg


    """
    from matplotlib import pyplot as _plt
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
    corr_dicts = [[]]*len(data)
    if projection is not None:
        corr_dicts = [_bmutils.most_corr(projection, geoms=igeom, proj_idxs=proj_idxs, n_args=n_feats)
                      for igeom in geoms]
        if corr_dicts[0]["feats"] != []:
            colors = _bmutils.matplotlib_colors_no_blue(ncycles=int(_np.ceil((_np.max(proj_idxs)+1)/6.))) # Hack
            colors = [colors[ii] for ii in proj_idxs]
            #colors = ['red']*10 # for the paper, to be deleted later
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

    correlation_input : numpy ndarray or some PyEMMA objects

        if array : 
            (m,m) correlation matrix, with a row for each feature and a column for each projection

        if PyEMMA-object :
            :obj:`~pyemma.coordinates.transform.TICA`, :obj:`~pyemma.coordinates.transform.PCA` or
            :obj:`~pyemma.coordinates.data.featurization.featurizer.MDFeaturizer`.

    geoms : None or :obj:`mdtraj.Trajectory`, default is None
        The values of the most correlated features will be returned for the geometries in this object. If widget is
        left to its default, None, :obj:`correlations` will create a new widget and try to show the most correlated
        features on top of the widget.

    widget : None or :obj:`nglview.NGLWidget`, default is None
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
        all black regardless len(proj_idxs). Input anything that yields :obj:`matplotlib.colors.is_color_like` == True

    proj_idxs : None, or int, or iterable of integers, default is None
        The indices of the projections for which the most correlated feture will be returned.
        If none it will default to the dimension of the correlation_input object

    feat_name : None or str, default is None
        The prefix with which to prepend the labels of the most correlated features. If left to None, the feature
        description found in :obj:`correlation_input` will be used, if available

    n_feats : int, default is 1
        Number of most correlated features to return for each projection

    featurizer : None or :obj:`~pyemma.coordinates.data.featurization.featurizer.MDFeaturizer`, default is None
        If :obj:`correlation_input` is not an :obj:`~pyemma.coordinates.data.featurization.featurizer.MDFeaturizer`
        itself, or doesn't have a :obj:`~pyemma.coordinates.transform.TICA.data_producer` attribute, the user can input one here.
        If :obj:`correlation_input` and :obj:`featurizer`
        **are both** :obj:`~pyemma.coordinates.data.featurization.featurizer.MDFeaturizer`-objects,
        :obj:`featurizer`  will be ignored.

    verbose : Bool, default is True
        print to standard output

    Returns
    -------
    corr_dict and ngl_wdg

    corr_dict :
        Dictionary containing correlation information. For an overview, just issue `print(corr_dict)`. The
        values are stored under the following keys.

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
            be evaluated and returned as list of length len(:obj:`proj_idxs`). Each element in th elist
            is an arrays with shape (:obj:`geom.n_frames`, :obj:`n_feats`)

        atom_idxs :
            List of length len(proj_idxs) each with an nd.array of shape (nfeat, m), where m is the number of atoms needed
            to describe each feature (1 of cartesian, 2 for distances, 3 for angles, 4 for dihedrals)

        info :
            List of length len(:obj:`proj_idxs`) with lists of length :obj:`n_feat` with strings describing the correlations

    ngl_wdg :
        :obj:`nglview.NGLWidget` with the most correlated features (distances, angles, dihedrals, positions)
        visualized on top of it.

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
        A PyEMMA MDfeaturizer object with any number of .active_features()

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
    atom_idxs = _bmutils.atom_idxs_from_general_input(feat)[slice(*idxs)]

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
        Contains the position associated the geometries in :obj:`geom`. See below for more details

    geom : :obj:`mdtraj.Trajectory` object or a list thereof.
        The geometries associated with the the :obj:`positions`.
        # TODO: WRITE HOW THE LISTS-LENGTHS CORRESPONDS FOR THE STICKY OPTION

    ax : :obj:`matplotlib.pyplot.Axes`  object
        The axes to be linked with the :obj:`~nglview.NGLWidget`

    plot_path : bool, default is False
        whether to draw a line connecting the positions in :obj:`positions`

    clear_lines : bool, default is True
        whether to clear all the 2D objects that were previously drawn in :obj:`ax`

    n_smooth : int, default is 0,
        if :obj:`n_smooth` > 0, the shown geometries and paths will be smoothed out by 2* :obj:`n_smooth` +1.
        See :obj:`molpx._bmutils.smooth_geom` for more information

    ngl_wdg : None or an existing :obj:`~nglview.NGLWidget`, default is None
        you can provide an already instantiated  :obj:`~nglview.NGLWidget` here (avanced use)

    superpose : boolean, default is True
        The geometries in :obj:`geom` may or may not be oriented, depending on how they were generated.
        Since this method is mostly for visualization purposes, the default behaviour is to orient them all to
        maximally overlap with the most compact frame available

    projection : object that generated the :obj:`positions`, default is None
        The projected coordinates may come from a variety of sources. When working with PyEMMA, a number of objects
        might have generated this projection, like a

        * :obj:`~pyemma.coordinates.transform.TICA`- or a
        * :obj:`~pyemma.coordinates.transform.PCA`- or an
        * :obj:`~pyemma.coordinates.data.featurization.featurizer.MDFeaturizer`-object.

        Makes most sense when :obj:`positions` where generated with :obj:`molpx.generate.projection_paths`,
        otherwise might produce rubbish. See :obj:`n_feats` for more info.
        # TODO: delete from here below?
        The features most correlated with the :obj:`positions` will be shown
        in the widget
        # TODO CHECK THIS
        geometries in :obj:`geom`, allowing the user to establish a visual connection between the
        projected coordinate and the original features (distances, angles, contacts etc).

    n_feats : int, default is 1
        If a :obj:`projection` is passed along, the first :obj:`n_feats` features that most correlate the
        the projected trajectories will be represented, both in form of figures of feat(t) as well as in
        the on top of the :obj:`ngl_wdg`. If :obj:`projection == None`, :obj:`nfeats` will be ignored.

    sticky : boolean, default is False
        If set to True, the :obj:`ngl_wdg` will be *sticky* in that every click adds a new molecular
        structure without deleting the previous one. Left-clicks adds a structure, right-clicks deletes
        a structure. Particularly useful for representing more minima simultaneously.

    color_list : None, list of len(:obj:`positions`), or "random"
        The colors with which the sticky frames will be plotted.
        A color is anything that yields :obj:`matplotlib.colors.is_color_like == True`
        "None" defaults to coloring by element.
        "random" randomizes the color choice

    list_of_repr_dicts : None or list of dictionaries, default is None
        Has an effect only for :obj:`sticky == True`, s.t. these reps instead of the default
        ones are used. The dictionaries must have at least the keys 'repr_type' and 'selection'.
        Other key-value pairs are currently ignored but, will be implemented in the future.
        See :obj:`nglview.NGLWidget.add_representation` for more info).

    link_ax2wdg_kwargs: dictionary of named arguments, optional
        named arguments for the function :obj:`molpx._linkutils.link_ax_w_pos_2_nglwidget`, which is the one that internally
        provides the interactivity. Non-expert users can safely ignore this option.

    Returns
    --------

    ngl_wdg : :obj:`~nglview.NGLWidget`

    axes_wdg: :obj:`~matplotlib.Axes.AxesWidget`

    """
    # Make a copy of the geometry, otherwise the input gets destroyed
    if isinstance(geom, list):
        copy_geom = [gg[:] for gg in geom]
    elif isinstance(geom, _md.Trajectory):
        copy_geom = geom[:]

    if not sticky:
        return _sample(positions, copy_geom, ax,
                       plot_path = plot_path,
                       clear_lines = clear_lines,
                       n_smooth = n_smooth,
                       ngl_wdg= ngl_wdg,
                       superpose = superpose,
                       projection = projection,
                       n_feats = n_feats,
                       **link_ax2wdg_kwargs)
    else:

        if isinstance(copy_geom, _md.Trajectory):
            copy_geom=[copy_geom]

        # The method takes care of whatever superpose
        copy_geom = _bmutils.superpose_to_most_compact_in_list(superpose, copy_geom)

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
        if copy_geom[0].top.n_residues < 10:
            sticky_rep = 'ball+stick'
        if list_of_repr_dicts is None:
            list_of_repr_dicts = [{'repr_type': sticky_rep, 'selection': 'all'}]

        # Now instantiate the ngl_wdg
        ngl_wdg = _nglwidget_wrapper(None)
        # Prepare Geometry_in_widget_list
        ngl_wdg._GeomsInWid = [_linkutils.GeometryInNGLWidget(igeom, ngl_wdg,
                                                          color_molecule_hex= cc,
                                                          list_of_repr_dicts=list_of_repr_dicts) for igeom, cc in zip(_bmutils.transpose_geom_list(copy_geom), sticky_colors_hex)]

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

    assert isinstance(geoms, (list, _md.Trajectory)), type(geoms)

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
        ax.plot(positions[:,0], positions[:,1], '-k', lw=3)

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

def contacts(contact_map, input, residue_indices=None, average=False, panelsize=4):
    r"""
    Return a plot of the contact map and a linked :obj:`nglview.NGLWidget`. Clicking on
    a pixel of interest on the contact map will a) highlight that pixel and b)
    add lines in the widget ,connecting the corresponding atoms. Also, any updates in
    the widget's frame, via the sliding bar, will update the shown contact map (in
     case more than one contact map was provided)

    Parameters
    ----------
    contact_map : square nd.array or iterable thereof.
        These square arrays contain the contact map(s)

    input : :obj:`mdtraj.Trajectory` object or a list thereof.
        An :obj:`nglview.NGLWidget` will be instantiated with this input

    residue_indices : boolean or iterable of integers
        Residue indices corresponding to the :obj:`contact_map`. If None, an array
        (0,1,...n_residues) will be created.
        # TODO if not None, a NotImplementedError will be raised, because the relabeling of
        zoomable plots is not yet implemented by molpx)

    average : boolean, default is False
        Plot only the average of the contact maps provided in :obj:`contact_map`. If only one
        such map is given, this keyword has no effect. If average is false but the
        number of frames in :obj:`input` and :obj:`contact_map` don match, an exception is thrown.

    panelsize : int, default is 4
        The size of the figure and widget that will be outputted inside the molpxbox

    Returns
    --------

    mpxbox : An :obj:`nglview.NGLWidget`
    """

    from matplotlib import pyplot as _plt
    # Add one axis to the input if necessary
    if _np.ndim(contact_map)==2:
        contact_map = _np.array(contact_map, ndmin=3)

    # Check that the number of frames match if no average is requested
    if _np.ndim(contact_map)==3 and not average:
        assert len(contact_map) == input.n_frames, "If average is False, the number of contact maps (%u) must " \
                                                   "match the number of frames in input (%u)" % (
                                                   len(contact_map), input.n_frames)
    # Assert squaredness
    assert all([ict.shape[0] == ict.shape[1] for ict in contact_map]), "The input has to be a square matrix"

    # Needed arrays
    nres = contact_map[0].shape[0]
    positions = _np.vstack(_np.unravel_index(range(nres**2), (nres,nres))).T
    if residue_indices is None:
        residue_pairs = positions
    else:
        raise NotImplementedError("This feature is not implemented yet!")

    # Create a color list
    cmap = _get_cmap('rainbow')
    cmap_table = _np.linspace(0, 1, len(positions))
    sticky_colors_hex = [_rgb2hex(cmap(ii)) for ii in _np.random.permutation(cmap_table)]

    # Instantiate widget
    iwd = _nglwidget_wrapper(input)

    # Do the plot
    _plt.ioff()
    _plt.figure(figsize=(panelsize, panelsize))
    iax = _plt.gca()
    # _plt.plot(positions[:,0], positions[:,1], ' ok')
    # Make the average if wanted
    if average:
        iax.matshow(_np.average(contact_map, axis=0))
    else:
        # Monkey-Patch the matshow_data into the object
        iwd._MatshowData = {"image" : iax.matshow(contact_map[0]),
                            "data"  : contact_map}
    _plt.ion()


    # TODO: if residues is not None,
    # TODO make sure that zooming works even if a sub-set of res_idxs is given
    """
    # Relabel the plot
    for axtype in ['x', 'y']:
        tic_idxs = [int(tl) for tl in getattr(iax, 'get_%sticks'%axtype)()[1:-1]]
        tic_labels = ['']+['%u'%residue_idxs[ii] for ii in tic_idxs]+['']
        getattr(iax,'set_%sticklabels'%axtype)(tic_labels)
    """

    # Monkey-Patch the ContactInNGLWidgets into the widget
    iwd._CtcsInWid = [_linkutils.ContactInNGLWidget
        (iwd, [_bmutils.get_repr_atom_for_residue(input.top.residue(aa)).index for aa in [ii,jj]], rp_idx,
         #verbose=True,
         color= sticky_colors_hex[rp_idx]
         )
                      for rp_idx, (ii,jj) in enumerate(residue_pairs)]

    # Turn axes into a widget
    axes_wdg = _linkutils.link_ax_w_pos_2_nglwidget(iax,
                                                    positions,
                                                    iwd,
                                                    crosshairs=False,
                                                    dot_color='None',
                                                    #**link_ax2wdg_kwargs
                                                    )

    iwd._set_size(*['%fin' % inches for inches in iax.get_figure().get_size_inches()])
    #iax.figure.tight_layout()
    axes_wdg.canvas.set_window_title("Contact Map")
    outbox = _linkutils.MolPXHBox([iwd, axes_wdg.canvas])
    _linkutils.auto_append_these_mpx_attrs(outbox, input, iax, _plt.gcf(), iwd, axes_wdg, positions)

    return outbox


def MSM(msm_obj, traj_inp,
        pos=None,
        sharpen=False,
        n_overlays=1,
        top=None,
        sticky=False,
        panelsize=6,
        **networkplot_kwargs):

    r"""
    Visualize an MSM or an HMM as a network of nodes and egdes, together with an :obj:~`nglview.NGLWidget`
    containing representative structures of node/state. Clicking on the node will update the widget.

    Parameters
    ----------
    msm_obj: input MSM-object
        One of PyEMMA's MSM-objects, either a "normal" MSM (:obj:`~pyemma.msm.MaximumLikelihoodMSM`) or a hidden MSM (:obj:`~pyemma.msm.MaximumLikelihoodHMSM`)

    traj_inp : trajectory input
        Where to get the geometries from. It must be the same input with which the :obj:`msm_obj` was built.
        No checks are done by the method as to whether this is true, i.e. *rubbish-in->rubbish-out*.
        It can be of three different types (and lists thereof):

           * filenames (for which a :obj:`top` is needed, see below)
           * :obj:`mdtraj.Trajectory` objects
           * a PyEMMA's :obj:`~pyemma.coordinates.data.feature_reader.FeatureReader`

    pos : node positions, either None or a numpy ndarray of ndim=2
        By default, node positions are optimized to represent connectivity
        (see PyEMMA's :obj:`~pyemma.plots.plot_markov_model`). However, the user can override with
        custom node-positions by passing an array as :obj:`pos`. In many cases, it is useful for that array to be
        the clustercenter-positions with which the MSM/HMM was constructed: :obj:`pos=cl.clustercenters`.
        The input in :obj:`pos` has to be compatible with the provided :obj:`msm_obj`, i.e. have the necessary
        number entries. The error messages will inform about what's wrong.

    sharpen : boolean, default is False,
        This keyword only has effect for an HMM as an :obj:`input_msm`.
        By default, the method samples from the distribution of microstate inside each macrostate, using the object's
        :obj:`~pyemma.msm.MaximumLikelihoodHMSM.sample_by_observation_probabilities`- method. This can lead
        to fuzzy samples where the overlay of molecular structures is not very informative.

        If :obj:`sharpen` is True, only the microstate that maximizes each macrostate's probabilites
        (i.e., its argmax) will be sampled. Produces a more *sharpened* sample,
        which is less representative of the whole set but very representative of the most probable
        microstate within that set.

    n_overlays : int, default is 1
        Number of structures to represent simultaneously for each node of the network

    top : str or :obj:`mdtraj.Topology`, default is None
        If the filenames in :obj:`traj_inp` need a topology, here's where you pass it along

    sticky : bool, default is False
        Behaviour of the mouseclick when clicking a node in the network.
        If True, left click adds-structures, right-click deletes them

    panelsize : int, default is 6
        Size of the network figure, in inches. If  :obj:`pos` is provided, the panelsize will be adapted
        slightly to match the proportions of :obj:`pos`

    networkplot_kwargs : named keyword arguments for :obj:`~pyemma.plots.plot_markov_model`

    Returns :
    ---------

    mpxbox : A :obj:`~molpx.linkutils.MolPXHBox`-object. It contains the :obj:`nglview.NGLWidget` and the network
    plot. Check the :obj:`mpxbox_linked*` attributes to see what the object contains

    """
    from pyemma.msm.estimators.maximum_likelihood_hmsm import MaximumLikelihoodHMSM as _MLHMSM
    from pyemma.msm.estimators.maximum_likelihood_msm  import MaximumLikelihoodMSM as _MLMSM
    from pyemma.plots import plot_markov_model as _pyemma_plt_msm
    from pyemma.util.discrete_trajectories import  sample_indexes_by_state as _sample_indexes_by_state
    from matplotlib import pyplot as _plt

    assert isinstance(msm_obj, (_MLHMSM, _MLMSM)), "Allowed input types are %s, not %s" % ((_MLHMSM, _MLMSM), type(MSM))

    # Input parsing of the position object (#TODO reduce code?)
    if pos is None:
       pass
    elif isinstance(pos, _np.ndarray):
            if isinstance(msm_obj, _MLMSM):
                assert len(pos) == msm_obj.nstates, \
                    ("Number of input positions (%u) "
                     "does not match number %u active states of the MSM. Try slicing the input positions with "
                     "MSM.active_set" % (len(pos), msm_obj.nstates))
            elif isinstance(msm_obj, _MLHMSM):
                assert len(pos) == msm_obj.nstates or len(pos) == msm_obj.nstates_obs, \
                    ("With the input HMSM, the input positions have to have either "
                               "%u entries (number of coarse states in the HMSM) or %u entries (number of observed states of the HMSM). "
                               "You have provided neither: %u" % (msm_obj.nstates, msm_obj.nstates_obs, len(pos)))
    else:
        raise TypeError("The object_for_positions has the wrong type %s" % type(pos))

    # MSM without coarse-graining
    if isinstance(msm_obj, _MLMSM):
        sample_frames = _np.vstack(msm_obj.sample_by_state(n_overlays))

    # HMM
    else:
        if sharpen:
            active_state_indexes = msm_obj.observable_state_indexes
            subset = _np.argmax(msm_obj.observation_probabilities, axis=1)
            sample_frames = _np.vstack(_sample_indexes_by_state(active_state_indexes, n_overlays,
                                                                subset=subset, replace=True))
            # The user gave one position entry per metastable set
            if pos is not None and len(pos)==msm_obj.nstates:
                pass # im leaving this case for clarity, in theory it could be removed
            # The user gave a full array of positions that matches the total number of microstates
            # and wants the method to choose automagically the positions that match the argmax(PDF)
            elif pos is not None and len(pos)==msm_obj.nstates_obs:
                pos = pos[subset]
            # Any other case has ben caught before by the above ValueErrors
        else:
            sample_frames = _np.vstack(msm_obj.sample_by_observation_probabilities(n_overlays))
            # The user gave one position entry per metastable set
            if pos is not None and len(pos)==msm_obj.nstates:
                pass  # im leaving this case for clarity, in theory it could be removed
            # The user gave a full array of positions that matches the total number of microstates
            # and wants the method to wheight them using the observation probabilities
            elif pos is not None and len(pos)==msm_obj.nstates_obs:
                isample_pos = []
                for idist in msm_obj.observation_probabilities:
                    isample_pos.append(_np.average(pos, weights=idist, axis=0))
                pos = _np.vstack(isample_pos)

    sample_geoms = _bmutils.save_traj_wrapper(traj_inp, sample_frames, None, top=top)
    sample_geoms = _bmutils.re_warp(sample_geoms, n_overlays)
    sample_geoms = _bmutils.transpose_geom_list(sample_geoms)

    _plt.ioff()
    ifig, pos = _pyemma_plt_msm(msm_obj.P, pos=pos,
                                **networkplot_kwargs,
                                )
    # Conserve the proportion of the circles in the MSM plot
    figw, figh = ifig.get_size_inches()
    ifig.set_size_inches((panelsize, panelsize * figh / figw))
    iax = ifig.gca()

    ngl_wdg, axes_wdg = sample(pos, sample_geoms, iax,
                               sticky=sticky,
                               crosshairs=False,
                               )#, clear_lines=False, **sample_kwargs)
    ngl_wdg._set_size(*['%fin' % inches for inches in ifig.get_size_inches()])
    ifig.tight_layout()
    outbox = _linkutils.MolPXHBox([ngl_wdg, ifig.canvas])
    _linkutils.auto_append_these_mpx_attrs(outbox, sample_geoms, _plt.gca(), ifig, ngl_wdg, axes_wdg, pos)
    _plt.ion()

    return outbox