from __future__ import print_function

__author__ = 'gph82'



from pyemma.coordinates import source as _source

import numpy as _np
from . import _bmutils
from collections import defaultdict as _defdict
import mdtraj as _md

def projection_paths(MD_trajectories, MD_top, projected_trajectories,
                   n_projs=1, proj_dim=2, proj_idxs=None,
                   n_points=100, n_geom_samples=100, proj_stride=1,
                   history_aware=True, verbose=False, minRMSD_selection='backbone'):
    r"""
    Return a path along a given projection. More info on what this means exactly will follow soon.

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

    n_projs : int, default is 1
        Number of projection paths to generate. If the input :obj:`projected_trajectories` are n-dimensional,
        in principle up to n-paths can be generated

    proj_dim : int, default is 2
        Dimensionality of the space in which distances will be computed

    proj_idxs: int, defaultis None
        Selection of projection idxs (zero-idxd) to visualize. The default behaviour is that proj_idxs = range(n_projs).
        However, if proj_idxs != None, then n_projs is ignored and proj_dim is set automatically

    n_points : int, default is 100
        Number of points along the projection path. The higher this number, the higher the projected coordinate is
        resolved, at the cost of more computational effort. It's a trade-off parameter

    n_geom_samples : int, default is 100
        For each of the :obj:`n_points` along the projection path, :obj:`n_geom_samples` will be retrieved from
        the trajectory files. The higher this number, the *smoother* the minRMSD projection path. Also, the longer
        it takes for the path to be computed

    proj_stride : int, default is 1
        The stride of the :obj:`projected_trajectories` relative to the :obj:`MD_trajectories`.
        This will play a role particularly if :obj:`projected_trajectories` is already strided (because the user is
        holding it in memory) but the MD-data on disk has not been strided.

    history_aware : bool, default is True
        The path-searching algorigthm the can minimize distances between adjacent points along the path or minimize
        the distance between each point and the mean value of all the other up to that point. Use this parameter
        to avoid a situation in which the path gets "derailed" because an outlier is chosen at a given point.

    verbose : bool, default is False
        The verbosity level

    minRMSD_selection : str, default is 'backbone'
        When computing minRMSDs between a given point and adjacent candidates, use this string to select the
        atoms that will be considered. Check mdtraj's selection language here http://mdtraj.org/latest/atom_selection.html

    Returns
    --------

    paths_dict :
        dictionary of dictionaries containing the projection paths.

        * :obj:`paths_dict[idxs][type_of_path]`

            * idxs represent the index of the projected coordinate ([0], [1]...)
            * types of paths "min_rmsd" or "min_disp"

        * What the dictionary actually contains

            * :obj:`paths_dict[idxs][type_of_path]["proj"]` : ndarray of shape (n_points, proj_dim) with the coordinates of the projection along the path

            * :obj:`paths_dict[idxs][type_of_path]["geom"]` : :obj:`mdtraj.Trajectory` geometries along the path


    idata :
        list of ndarrays with the the data in  :obj:`projected_trajectories`
    """

    # Some input parsing
    MD_trajectories = _bmutils.moldata_from_input(MD_trajectories, MD_top=MD_top)
    idata = _bmutils.data_from_input(projected_trajectories)
    _bmutils.assert_moldata_belong_data(MD_trajectories, idata, proj_stride)


    # What's the hightest dimensionlatiy that the input data allows?
    input_dim = idata[0].shape[1]
    if proj_idxs is None:
       proj_idxs = _np.arange(n_projs)
    else:
        proj_idxs = _bmutils.listify_if_int(proj_idxs)

    proj_dim = _np.max((proj_dim, _np.max(proj_idxs)+1))
    proj_dim = _np.min((proj_dim,input_dim))

    # Load data  up to :proj_dim column
    idata = [dd[:,:proj_dim] for dd in idata]

    # Iterate over wanted coords
    paths_dict = {}
    for coord in proj_idxs:
        paths_dict[coord] = {"min_rmsd": _defdict(dict),
                             "min_disp": _defdict(dict)
                           }
        # Cluster in regspace along the dimension you want to advance, to approximately n_points
        cl = _bmutils.regspace_cluster_to_target([jdata[:,[coord]] for jdata in idata],
                                n_points, n_try_max=3,
                                verbose=verbose,
                                )

        # Create full catalogues (discrete and continuous) in ascending order of the coordinate of interest
        cat_idxs, cat_cont = _bmutils.catalogues(cl,
                                        data=idata,
                                        sort_by=0, #here we always have to sort by the 1st coord
                                            )

        # Create sampled catalogues in ascending order of the cordinate of interest
        sorts_coord = _np.argsort(cl.clustercenters[:,0]) # again, here we're ALWAYS USING "1" because cl. is 1D
        cat_smpl = cl.sample_indexes_by_cluster(sorts_coord, n_geom_samples)
        geom_smpl = _bmutils.save_traj_wrapper(MD_trajectories, _np.vstack(cat_smpl), None, stride=proj_stride)
        geom_smpl = _bmutils.re_warp(geom_smpl, [n_geom_samples]*cl.n_clusters)

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
            # Choose the starting coordinate-value for the fwd and bwd projection_paths,
            # see the options of the method for more info
            istart_idx = _bmutils.get_good_starting_point(cl, geom_smpl, cl_order=sorts_coord,
                                                  strategy=strategy
                                                  )

            # Of all the sampled geometries that share this starting point of "coord"
            # pick the one that's closest to zero in the other coordinates
            istart_Y = _np.vstack([idata[ii][jj] for ii,jj in cat_smpl[istart_idx]])
            istart_Y = _np.delete(istart_Y, coord, axis=1)
            istart_frame = _np.sum(istart_Y**2,1).argmin()
            # TODO consider starting of the most populated value and see which one diffuses the least

            # Starting from the sampled geometries,
            # create a path minimising minRMSD between frames.
            path_smpl, __ = _bmutils.visual_path(cat_smpl, geom_smpl,
                                         start_pos=istart_idx,
                                         start_frame=istart_frame,
                                         path_type='min_rmsd',
                                         history_aware=history_aware,
                                         selection=geom_smpl[0].top.select(minRMSD_selection))

            # Compute the diffusion along the path
            y = _np.vstack([idata[ii][jj] for ii,jj in path_smpl])
            y[:,coord] = 0
            path_diffusion = _np.sqrt(_np.sum(_np.diff(y.T).T**2,1).mean())
            #print('Strategy %s starting at %g diffuses %g in %uD proj-space'%(strategy,
            #                                                                  cl.clustercenters[sorts_coord][istart_idx],
            #                                                                  path_diffusion,
            #                                                                  proj_dim),
            #      flush=True)

            # Store the results of the minRMSD sampling with this strategy in a dictionary
            path_sample[strategy] = path_smpl
            path_rmsd[strategy] = path_diffusion
            start_idx[strategy] = istart_idx
            start_frame[strategy] = istart_frame

        # Stick the rmsd-path that diffuses the least
        min_key = sorted(path_rmsd.keys())[_np.argmin([path_rmsd[key] for key in sorted(path_rmsd.keys())])]
        #print("Sticking with %s"%min_key)
        path_smpl = path_sample[min_key]
        istart_idx = start_idx[min_key]

        paths_dict[coord]["min_rmsd"]["proj"] = _np.vstack([idata[ii][jj] for ii,jj in path_smpl])
        paths_dict[coord]["min_rmsd"]["geom"] = _bmutils.save_traj_wrapper(MD_trajectories, path_smpl, None,
                                                        stride=proj_stride, top=MD_top)

        # With the starting point the creates the minimally diffusive path,
        # create a path minimising displacement in the projected space (minimally diffusing path)
        istart_Y = cat_cont[istart_idx]
        istart_Y[:,coord] = 0
        istart_frame = _np.sum(istart_Y**2,1).argmin()
        # TODO IMPLEMENT ANOTHER STRATEGY (max pop?) FOR THE STARTING FRAME
        path, __ = _bmutils.visual_path(cat_idxs, cat_cont,
                               start_pos=istart_idx,
                               start_frame=istart_frame,
                               history_aware=history_aware,
                               exclude_coords=[coord])
        paths_dict[coord]["min_disp"]["geom"]  = _bmutils.save_traj_wrapper(MD_trajectories, path, None, stride=proj_stride, top=MD_top)
        paths_dict[coord]["min_disp"]["proj"] = _np.vstack([idata[ii][jj] for ii,jj in path])

        #TODO : consider storing the data in each dict. It's redundant but makes each dict kinda standalone

    return paths_dict, idata


def sample(MD_trajectories, MD_top, projected_trajectories,
           atom_selection = None,
           proj_idxs=[0,1], n_points=100, n_geom_samples=1,
           keep_all_samples = False,
           proj_stride=1,
           verbose=False,
           return_data=False
                 ):
    r"""
    Returns a sample of molecular geometries  and their positions in the projected space

    Parameters
    ----------

    MD_trajectories : list of strings
        Filenames (any extension that :py:obj:`mdtraj` can read is accepted) containing the trajectory data.
        There is an untested input mode where the user parses directly :obj:`mdtraj.Trajectory` objects

    MD_top : str to topology filename or directly :obj:`mdtraj.Topology` object

    projected_trajectories : (list of) strings or (list of) numpy ndarrays of shape (n_frames, n_dims)
        Time-series with the projection(s) that want to be explored. You can provide .npy-filenames or readable asciis
        (.dat, .txt etc). Alternatively, you can feed in your own PyEMMA-clustering object
        NOTE: molpx assumes that there is no time column.

    atom_selection : string or iterable of integers, default is None
        The geometries of the original trajectory files will be filtered down to these atoms. It can be any DSL string
        that   mdtraj.Topology.select could understand or directly the iterable of integers.
        If :py:obj`MD_trajectories` is already a (list of) md.Trajectory objects, the atom-slicing can take place
        before calling this method.

    proj_idxs: int, default is None
        Selection of projection idxs (zero-idxd) to visualize. The default behaviour is that proj_idxs = range(n_projs).
        However, if proj_idxs != None, then n_projs is ignored and proj_dim is set automatically

    n_points : int, default is 100
        Number of points along the projection path. The higher this number, the higher the projected coordinate is
        resolved, at the cost of more computational effort. It's a trade-off parameter

    n_geom_samples : int, default is 1
        For each of the :obj:`n_points` along the projection path, :obj:`n_geom_samples` will be retrieved from
        the trajectory files. The higher this number, the *smoother* the minRMSD projection path. Also, the longer
        it takes for the path to be computed. This is a trade-off parameter between how smooth the transitons between
        geometries can be and how long it takes to generate the sample

    keep_all_samples : boolean, default is False
        In principle, once the closest-to-ref geometry has been kept, the other geometries are discarded, and the
        output sample contains only n_point geometries. There are, still, special cases where the user might
        want to keep all sampled geometries. Typical use-case is when the n_points is low and many representatives
        per clustercenters will be much more informative than the other way around.
        This is an advanced feature that other methods of molPX use internally for generating overlays, be awere that
        it changes the return type of :obj:`geom_smpl` from the default (an :obj:`mdtraj.Trajectory` with :obj:`n_points`-frames)
        to a list list of length :obj:`n_geom_samples`, each element is an :obj:`mdtraj.Trajectory` object of :obj:`n_points`-frames

    proj_stride : int, default is 1
        Stride value that was used in the :obj:`projected_trajectories` relative to the :obj:`MD_trajectories`
        If the original :obj:`MD_trajectories` were stored every 5 ps but the projected trajectories were stored
        every 50 ps, :obj:`proj_stride` = 10 has to be provided, otherwise an exception will be thrown informing
        the user that the :obj:`MD_trajectories` and the :obj:`projected_trajectories` have different number of frames.

    Returns
    --------

    pos :
        ndarray with the positions of the sample
    geom_smpl :
        sampled geometries. Can be of two types:

        * default: :obj:`mdtraj.Trajectory` with :obj:`n_points`-frames
        * if keep_all_samples = True: list of length :obj:`n_geom_samples`. Each element is an :obj:`mdtraj.Trajectory` object of :obj:`n_points`-frames.

    """

    MD_trajectories = _bmutils.listify_if_not_list(MD_trajectories)
    if isinstance(MD_trajectories[0], _md.Trajectory):
        src = MD_trajectories
    else:
        src = _source(MD_trajectories, top=MD_top)


    # Find out if we already have a clustering object
    try:
        projected_trajectories.dtrajs
        cl = projected_trajectories
    except:
        idata = _bmutils.data_from_input(projected_trajectories)
        cl = _bmutils.regspace_cluster_to_target([dd[:,proj_idxs] for dd in idata], n_points, n_try_max=10, verbose=verbose)

    pos = cl.clustercenters
    cat_smpl = cl.sample_indexes_by_cluster(_np.arange(cl.n_clusters), n_geom_samples)

    geom_smpl = _bmutils.save_traj_wrapper(src, _np.vstack(cat_smpl), None, top=MD_top, stride=proj_stride)

    atom_slice = _bmutils.parse_atom_sel(atom_selection, geom_smpl.top)
    if atom_slice is not None:
        geom_smpl = geom_smpl.atom_slice(atom_slice)

    if n_geom_samples>1:
        geom_smpl = _bmutils.re_warp(geom_smpl, [n_geom_samples] * cl.n_clusters)
        if not keep_all_samples:
            # Of the most populated geom, get the most compact
            most_pop = _np.bincount(_np.hstack(cl.dtrajs)).argmax()
            geom_most_pop = geom_smpl[most_pop][_md.compute_rg(geom_smpl[most_pop]).argmin()]
            geom_smpl = _bmutils.slice_list_of_geoms_to_closest_to_ref(geom_smpl, geom_most_pop)
        else:
            geom_smpl = _bmutils.transpose_geom_list(geom_smpl)

    if not return_data:
        return pos, geom_smpl
    else:
        return pos, geom_smpl, idata

