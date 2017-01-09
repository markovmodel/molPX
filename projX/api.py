from __future__ import print_function

__author__ = 'gph82'


__all__ = [
    'generate_paths',
    'visualize_traj',
    'generate_sample',
    'visualize_sample',
    'visualize_FES'
]

from pyemma.coordinates import source as _source, save_traj as _save_traj
import numpy as _np
from .bmutils import cluster_to_target as _cluster_to_target, \
    catalogues as _catalogues, \
    re_warp as _re_warp, \
    get_good_starting_point as _get_good_starting_point, \
    visual_path as _visual_path, \
    link_ax_w_pos_2_nglwidget as _link_ax_w_pos_2_nglwidget, \
    data_from_input as _data_from_input, \
    minimize_rmsd2ref_in_sample as _minimize_rmsd2ref_in_sample

from matplotlib import pylab as _plt
from collections import defaultdict as _defdict
import nglview as _nglview
import mdtraj as _md

def generate_paths(MDtrajectory_files, topology, projected_data,
                   n_projs=1, proj_dim=2, proj_idxs=None,
                   n_points=100, n_geom_samples=100, proj_stride=1,
                   history_aware=True, verbose=False):
    r"""
    projected_data : data to be connected with geometries
        Contains the data either in form of:
             np.ndarray or filename_str  (one single trajectory)
        list of np.ndarrays or filename_str (lists of trajectories)
        NOT: list of lists are not yet implemented (equivalent to pyemma fragmented trajectory reader)

    proj_dim : int, dimensionality of the space in which distances will be computed
    n_projs : int, number of projections to visualize
    proj_idxs: int, indices of projections to visualize, defaultis None
        Selection of projection idxs (zero-idxd) to visualize. The default
        behaviour is that proj_idxs = range(n_projs).
        However, if proj_idxs != None, then n_projs is ignored and proj_dim is set automatically
    """

    src = _source(MDtrajectory_files, top=topology)
    idata = _data_from_input(projected_data)
    #TODO: assert total_n_frames (strided) coincies with the n_frames in data
    # What's the hightest dimensionlatiy that the input data allows?
    input_dim = idata[0].shape[1]
    if proj_idxs is None:
       proj_idxs = _np.arange(n_projs)
    else:
        if isinstance(proj_idxs, int):
            proj_idxs = [proj_idxs]

    proj_dim = _np.max((proj_dim, _np.max(proj_idxs)+1))
    proj_dim = _np.min((proj_dim,input_dim))
    # Load data  up to :proj_dim column
    idata = [dd[:,:proj_dim] for dd in idata]

    # Iterate over wanted coords
    out_dict = {}
    for coord in proj_idxs:
        out_dict[coord] = {"min_rmsd": _defdict(dict),
                           "min_disp": _defdict(dict)
                           }
        # Cluster in regspace along the dimension you want to advance, to approximately n_points
        cl = _cluster_to_target([jdata[:,coord] for jdata in idata],
                                n_points, n_try_max=3,
                                verbose=verbose,
                                )

        # Create full catalogues (discrete and continuous) in ascending order of the coordinate of interest
        cat_idxs, cat_cont = _catalogues(cl,
                                        data=idata,
                                        sort_by=0, #here we always have to sort by the 1st coord
                                            )

        # Create sampled catalogues in ascending order of the cordinate of interest
        sorts_coord = _np.argsort(cl.clustercenters[:,0]) # again, here we're ALWAYS USING "1" because cl. is 1D
        cat_smpl = cl.sample_indexes_by_cluster(sorts_coord, n_geom_samples)
        geom_smpl = _save_traj(src, _np.vstack(cat_smpl), None, topology, stride=proj_stride)
        geom_smpl = _re_warp(geom_smpl, [n_geom_samples]*cl.n_clusters)

        # Initialze stuff
        path_sample = {}
        path_rmsd = {}
        start_idx = {}
        start_frame = {}
        for strategy in [
            'smallest_Rgyr',
            'most_pop',
            'most_pop_x_smallest_Rgyr',
            'bimodal_compact'
            ]:
            # Choose the starting point for the fwd and bwd paths, see the options of the method for more info
            istart_idx = _get_good_starting_point(cl, geom_smpl, cl_order=sorts_coord,
                                                  strategy=strategy
                                                  )

            # Of all the sampled geometries that share this starting point of "coord"
            # pick the one that's closest to zero
            istart_Y = _np.vstack([idata[ii][jj] for ii,jj in cat_smpl[istart_idx]])
            istart_Y[:,coord] = 0
            istart_frame = _np.sum(istart_Y**2,1).argmin()

            selection = geom_smpl[0].top.select('backbone')
            # Create a path minimising minRMSD between frames
            path_smpl, __ = _visual_path(cat_smpl, geom_smpl,
                                        start_frame=istart_frame,
                                        path_type='min_rmsd',
                                        start_pos=istart_idx,
                                        history_aware=history_aware,
                                        selection=selection)

            y = _np.vstack([idata[ii][jj] for ii,jj in path_smpl])
            y[:,coord] = 0
            path_diffusion = _np.sqrt(_np.sum(_np.diff(y.T).T**2,1).mean())
            #print('Strategy %s starting at %g diffuses %g in %uD proj-space'%(strategy,
            #                                                                  cl.clustercenters[sorts_coord][istart_idx],
            #                                                                  path_diffusion,
            #                                                                  args.proj_dim),
            #      flush=True)

            path_sample[strategy] = path_smpl
            path_rmsd[strategy] = path_diffusion
            start_idx[strategy] = istart_idx
            start_frame[strategy] = istart_frame

        # Stick the rmsd-path that diffuses the least
        min_key = sorted(path_rmsd.keys())[_np.argmin([path_rmsd[key] for key in sorted(path_rmsd.keys())])]
        #print("Sticking with %s"%min_key)
        path_smpl = path_sample[min_key]
        istart_idx = start_idx[min_key]

        out_dict[coord]["min_rmsd"]["proj"] = _np.vstack([idata[ii][jj] for ii,jj in path_smpl])
        out_dict[coord]["min_rmsd"]["geom"] = _save_traj(src.filenames, path_smpl, None,
                                                         stride=proj_stride, top=topology
                                                        )

        # With the starting point the creates the minimally diffusive path,
        # create a path minimising displacemnt in the projected space (minimally diffusing path)
        istart_Y = cat_cont[istart_idx]
        istart_Y[:,coord] = 0
        istart_frame = _np.sum(istart_Y**2,1).argmin()
        path, __ = _visual_path(cat_idxs, cat_cont,
                               start_pos=istart_idx,
                               start_frame=istart_frame,
                               history_aware=history_aware,
                               exclude_coords=[coord])
        out_dict[coord]["min_disp"]["geom"]  = _save_traj(src.filenames, path, None, stride=proj_stride, top=topology)
        out_dict[coord]["min_disp"]["proj"] = _np.vstack([idata[ii][jj] for ii,jj in path])

        #TODO : consider storing the data in each dict. It's redundant but makes each dict kinda standalone
    return out_dict, idata

def visualize_FES(MD_trajfile, MD_top, projected_trajectory):
    r"""
    TODO: document everything, parse options to generate_sample. Right now everything is
    taking its default values (which work well)

    returns: ax, iwd, sample, geoms
        ax : matplotlib axis object
        iwd : nglwidget
        sample: the position of the red dots in the plot
        geoms: the geometries incorporated into the nglwidget
    """
    sample, geoms, data= generate_sample(MD_trajfile, MD_top,projected_trajectory,
                                        return_data=True
                                         )
    h, (x, y) = _np.histogramdd(_np.vstack(data), bins=50)

    irange = _np.hstack((x[[0,-1]], y[[0,-1]]))
    _plt.figure()
    _plt.contourf(-_np.log(h).T, extent=irange)
    ax = _plt.gca()

    iwd = visualize_sample(sample, geoms.superpose(geoms[0]), ax)

    return _plt.gca(), _plt.gcf(), iwd, sample, geoms

def visualize_traj(trajectory,
                   MD_top, projected_trajectory, max_frames=1000,
                   stride=1, proj_idxs=[0,1], dt=1, plot_FES=False):
    r"""
    TODO: document everything, parse options to generate_sample. Right now everything is
    taking its default values (which work well)

    returns: ax, iwd, sample, geoms
        ax : matplotlib axis object
        iwd : nglwidget
        sample: the position of the red dots in the plot
        geoms: the geometries incorporated into the nglwidget
    """
    assert len(proj_idxs) == 2

    # Parse input
    data = _data_from_input(projected_trajectory)[0][:,proj_idxs]
    if isinstance(trajectory, _md.Trajectory):
        geoms = trajectory
    else:
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
        sample =_np.vstack((time, idata)).T
        iax.plot(time, idata)
        widget = visualize_sample(sample, geoms.superpose(geoms[0]), iax, clear_lines=False, widget=widget)

    if plot_FES:
        h, (x, y) = _np.histogramdd(data, bins=50)
        irange = _np.hstack((x[[0,-1]], y[[0,-1]]))
        _plt.figure()
        _plt.contourf(-_np.log(h).T, extent=irange)
        ax = _plt.gca()
        widget = visualize_sample(data, geoms.superpose(geoms[0]), ax, widget=widget)

    return _plt.gca(), _plt.gcf(), widget, geoms


def visualize_sample(sample, geom,
                    ax,
                    plot_path=False,
                    clear_lines=True,
                     widget=None,
                     **link_ax2wdg_kwargs
                   ):

    r"""
    TODO
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
        ax.plot(sample[:,0], sample[:,1], '-g', lw=3)

    # Link the axes widget with the ngl widget
    _link_ax_w_pos_2_nglwidget(ax,
                               sample,
                               iwd,
                               **link_ax2wdg_kwargs
                               )
    return iwd

def generate_sample(MDtrajectory_files, topology, projected_data,
                 idxs=[0,1], n_points=100, n_geom_samples=1,
                 keep_all_samples = False,
                 proj_stride=1,
                 verbose=False,
                    return_data=False
                 ):
    r"""
    n_geoms_samples : int, default is 1
        The number of sampled geometries per clustercenter. If you increase this number to n, generate_sample
         will look for 1) the most populated clustercenter
                       2) the most compact geometry, among the "n" available in that center. That's the reference
                       3) for all other clustercenters, each with "n" candidates, the geometry that's closest
                       in rmsd to the refrence
        This is a trade-off parameter between how smooth the transitons between geometries can be and how long it takes
         to generate the sample

    keep_all_samples = boolean, default is False
        In principle, once the closest-to-ref geometry has been kept, the other geometries are discarded, and the
        output sample contains only n_point geometries. HOWEVER, there are special cases where the user might
        want to keep all sampled geometries. Typical use-case is when the n_points is low and many representatives
        per clustercenters will be much more informative than the other way around
        (i know, this is confusing TODO: write this better)


    projected data: nd.array or list of nd.arrays OR pyemma.clustering object
        Although 2D is the most usual case, the dimensionality of the clustering and the one of the visualization (2D)
        do not necessarily have to be the same
    """


    src = _source(MDtrajectory_files, top=topology)

    # Find out if we already have a clustering object
    try:
        projected_data.dtrajs
        cl = projected_data
    except:
        idata = _data_from_input(projected_data)
        cl = _cluster_to_target([dd[:,idxs] for dd in idata], n_points, n_try_max=10, verbose=verbose)

    pos = cl.clustercenters
    cat_smpl = cl.sample_indexes_by_cluster(_np.arange(cl.n_clusters), n_geom_samples)
    geom_smpl = _save_traj(src, _np.vstack(cat_smpl), None, stride=proj_stride)
    if n_geom_samples>1:
        if not keep_all_samples:
            geom_smpl = _re_warp(geom_smpl, [n_geom_samples] * cl.n_clusters)
            # Of the most populated geom, get the most compact
            most_pop = _np.bincount(_np.hstack(cl.dtrajs)).argmax()
            geom_most_pop = geom_smpl[most_pop][_md.compute_rg(geom_smpl[most_pop]).argmin()]
            geom_smpl = _minimize_rmsd2ref_in_sample(geom_smpl, geom_most_pop)
        else:
            # Need to repeat the pos-vector
            pos = _np.tile(pos,n_geom_samples).reshape(-1,2)

    if not return_data:
        return pos, geom_smpl
    else:
        return pos, geom_smpl, idata

    pass
