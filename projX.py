__author__ = 'gph82'


__all__ = ['projX',
           ]

from pyemma.coordinates import source as _source, save_traj as _save_traj
import numpy as _np
from bmutils import cluster_to_target as _cluster_to_target, \
    catalogues as _catalogues, \
    re_warp as _re_warp, \
    get_good_starting_point as _get_good_starting_point, \
    visual_path as _visual_path, \
    link_ax_w_pos_2_nglwidget as _link_ax_w_pos_2_nglwidget
from collections import defaultdict as _defdict
import nglview as _nglview
from matplotlib import pylab as _plt

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
    src = _source(MDtrajectory_files, top=topology)#, projected_file)

    # Load data  up to :proj_dim column
    if isinstance(projected_data, str) or isinstance(projected_data, _np.ndarray):
        projected_data = [projected_data]
    elif not isinstance(projected_data, list):
        raise ValueError("Data type not understood %"%type(projected_data))
    if isinstance(projected_data[0],str):
        idata = [_np.load(f) for f in projected_data]
    else:
        idata = projected_data


    # What's the hightest dimensionlatiy that the input data allows?
    input_dim = idata[0].shape[1]
    if proj_idxs is None:
       proj_idxs = _np.arange(n_projs)
    else:
        if isinstance(proj_idxs, int):
            proj_idxs = [proj_idxs]

    proj_dim = _np.max((proj_dim, _np.max(proj_idxs)+1))
    proj_dim = _np.min((proj_dim,input_dim))
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

def visualize_paths(path, geom,
                    ax,
                    plot_path=False,
                    clear_lines=True
                   ):

    r"""
    TODO
    """
    # Create ngl_viewer widget
    iwd = _nglview.show_mdtraj(geom)

    if plot_path:
        if clear_lines == True:
            [ax.lines.pop() for ii in range(len(ax.lines))]


        # Plot the path on top of it
        ax.plot(path[:,0], path[:,1], '-g', lw=3)

    # Link the axes widget with the ngl widget
    _link_ax_w_pos_2_nglwidget(ax,
                               path,
                               iwd
                               )
    return iwd

