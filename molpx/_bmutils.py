from __future__ import print_function
import numpy as _np
import mdtraj as _md

try:
    from sklearn.mixture import GaussianMixture as _GMM
except ImportError:
    from sklearn.mixture import GMM as _GMM

# From pyemma's coordinates
from pyemma.coordinates import \
    source as _source, \
    cluster_regspace as _cluster_regspace, \
    save_traj as _save_traj

# From coor.data
from pyemma.coordinates.data.fragmented_trajectory_reader import FragmentedTrajectoryReader as _FragmentedTrajectoryReader
from pyemma.coordinates.data.feature_reader               import FeatureReader as _FeatureReader
from pyemma.coordinates.data.featurization.featurizer     import MDFeaturizer as _MDFeaturizer
from pyemma.coordinates.data.featurization.distances import DistanceFeature as _DF, \
    ResidueMinDistanceFeature as _ResMinDF
from pyemma.coordinates.data.featurization.misc import SelectionFeature as _SF
from pyemma.coordinates.data.featurization.angles import DihedralFeature as _DihF
from pyemma.coordinates.data.featurization.angles import AngleFeature as _AF
# From coor.transform
from pyemma.coordinates.transform import TICA as _TICA, PCA as _PCA
# From coor.util
from pyemma.util.discrete_trajectories import index_states as _index_states
from pyemma.util.types import is_string as _is_string,  is_int as _is_int

def listify_if_int(inp):
    if _is_int(inp):
        inp = [inp]

    return inp

def listify_if_not_list(inp, except_for_these_types=None):
    r"""
    :param inp:
    :param except_for_these_types: tuple, default is None

    :return: [inp] iff inp was not a list or any of the types in except_for_these_types
    """
    types = (list,)
    if except_for_these_types is not None:
        # listify inside of listify...
        if not isinstance(except_for_these_types, (list, tuple)):
            except_for_these_types = [except_for_these_types]
        for itype in except_for_these_types:
            types += (itype,)

    if not isinstance(inp, types):
        inp = [inp]

    return inp

def moldata_from_input(inp, MD_top=None):
    r"""

    Parameters
    ----------

    inp : str, md.Trajectory, list of strs, list of md.Trajectories or even a pyemma FeatureReader
        Where the molecular trajectory data is coming from

    MD_top : filename or md.Topology object
        If :py:obj:`inp` is needed to construct a :py:obj:`pyemma.coordinates.source` type of object,
        you have to parse it here

    Returns
    -------

    moldata: pyemma FeatureReader or list of md.Trajectories, depending on the input
    """

    if isinstance(inp, (_FeatureReader, _FragmentedTrajectoryReader)):
        moldata = inp

    # Everything else gets listified
    else:
        inp = listify_if_not_list(inp)
        # We have geometries so don't do anything
        if isinstance(inp[0], _md.Trajectory):
            moldata = inp
        elif isinstance(inp[0], str):
            moldata = _source(inp, top=MD_top)
        # TODO consider letting pyemma fail here instead of catching this
        else:
            raise TypeError("Please revise tyour input, it should be a str ",type(inp[0]))
    return moldata

def assert_moldata_belong_data(moldata, data, data_stride=1):
    r"""

    Parameters :
    ------------

        moldata : list or pyemma FeatureReader or FragmentedTrajectoryReader

        data : list of ndarrays

        data_stride : int, stride of the data vs. the moldata

    Returns :
    ---------

        boolean
    """
    try:
        n_traj = moldata.number_of_trajectories()
        traj_lengths = moldata.trajectory_lengths()
    except  AttributeError:
        n_traj = len(moldata)
        traj_lengths = [ii.n_frames for ii in moldata]

    # Stride:
    traj_lengths = [len(_np.arange(ii)[::data_stride]) for ii in traj_lengths]


    assert n_traj == len(data), ("Wrong number of molecular traj vs. data trajs: %u vs %u"%(n_traj, len(data)))
    assert _np.allclose(traj_lengths, [len(ii) for ii in data]), "Mismatch in the lengths of individual molecular trajs and data trajs"

def matplotlib_colors_no_blue():
    # Until we get the colorcyle thing working, this is a workaround:
    # http://stackoverflow.com/questions/13831549/get-matplotlib-color-cycle-state
    cc = ['green',
          'red',
          'cyan',
          'magenta',
          'yellow',
          'black',
          'white']
    for ii in range(3):
        cc += cc # this grows quickly
    return cc

def re_warp(array_in, lengths):
    """Return iterable ::py:obj:array_in as a list of arrays, each
     one with the length specified in lengths

    Parameters
    ----------

    array_in: any iterable
        Iterable to be re_warped

    lengths : int or iterable of integers
        Lengths of the individual elements of the returned array. If only one int is parsed, all lengths will
        be that int. Special cases:
            * more lengths than needed are parsed: the last elements of the returned value are empty
            until all lengths have been used
            * less lengths than array_in could take: only the lenghts specified are returned in the
            warped list, the rest is unreturned
    Returns
    -------
    warped: list
    """

    if _is_int(lengths):
        lengths = [lengths] * int(_np.ceil(len(array_in) / lengths))

    warped = []
    idxi = 0
    for ii, ll in enumerate(lengths):
        warped.append(array_in[idxi:idxi+ll])
        idxi += ll
    return warped

def regspace_cluster_to_target(data, n_clusters_target,
                               n_try_max=5, delta=5.,
                               verbose=False):
    r"""
    Clusters a dataset to a target n_clusters using regspace clustering by iteratively. "
    Work best with 1D data

    data: ndarray or list thereof
    n_clusters_target: int, number of clusters.
    n_try_max: int, default is 5. Maximum number of iterations in the heuristic.
    delta: float, defalut is 5. Percentage of n_clusters_target to consider converged.
             Eg. n_clusters_target=100 and delta = 5 will consider any clustering between 95 and 100 clustercenters as
             valid. Note. Note: An off-by-one in n_target_clusters is sometimes unavoidable

    returns: pyemma clustering object

    tested:True
    """
    delta = delta/100
    ndim = _np.vstack(data).shape[0]
    assert ndim >= n_clusters_target, "Cannot cluster " \
                                                      "%u datapoints on %u clustercenters. Reduce the number of target " \
                                                      "clustercenters."%(_np.vstack(data).shape[0], n_clusters_target)
    # Works well for connected, 1D-clustering,
    # otherwise it's bad starting guess for dmin
    cmax = _np.vstack(data).max()
    cmin = _np.vstack(data).min()
    dmin = (cmax-cmin)/(n_clusters_target+1)

    err = _np.ceil(n_clusters_target*delta)
    cl = _cluster_regspace(data, dmin=dmin, max_centers=5000)
    for cc in range(n_try_max):
        n_cl_now = cl.n_clusters
        delta_cl_now = _np.abs(n_cl_now - n_clusters_target)
        if not n_clusters_target-err <= cl.n_clusters <= n_clusters_target+err:
            # Cheap (VERY BAD IN HIGH DIM) heuristic to get relatively close relatively quick
            dmin = cl.dmin*cl.n_clusters/   n_clusters_target
            cl = _cluster_regspace(data, dmin=dmin, max_centers=5000)# max_centers is given so that we never reach it (dangerous)
        else:
            break
        if verbose:
            print('cl iter %u %u -> %u (Delta to target (%u +- %u): %u'%(cc, n_cl_now, cl.n_clusters,
                                                                         n_clusters_target, err, delta_cl_now))
    return cl

def min_disp_path(start, path_of_candidates,
                  exclude_coords=None, history_aware=False):
    r""""Returns a list of indices [i,j,k...] s.t. the distances
     d_i = |path_of_candidates[0][i] - path_of_candidates[1][j]|,
     d_j = |path_of_candidates[1][j] - path_of_candidates[2][k]|,...
     are minimized

    n is the dimension of the space in which the path exist
    Parameters:
    start : ndarray of shape (1,n)
       starting point. path_of_candidates[ii].shape[1] == start.shape[1] == n,
       for all ii

    path_of_candidates: iterable of 2D np.arrays each of shape (m, n)
       The list of candidates to pick from to minimize the distances

    exclude_coords: None, int, or iterable of ints, default is None
       The default behaviour is to consider all n-dimensions of the points when
       computing distances. However, it might be useful to exclude the i-th (and jth etc) coordinate(s)
       if the path is supossed to be advancing along the i-th coordinate (exclude=i)

    history_aware: boolean, default is false
       The default behaviour is to minimize distances stepwise, i.e. the j-th step will be chosen
       to minimise distance with i. However, if i-happened to be an outlier, the entire path can get
       derailed, hence it is sometimes better to minimize with the running_average of the path.

    Returns:
    path: list of integers
       One index per array of candidates, with len(path==len(path_of_candidates), s.t. the traveled
       distance is minimized

    tested = True
    """

    exclude_coords = listify_if_int([exclude_coords])

    path_out = []
    now = _np.copy(start)
    if _np.ndim(now) == 1:
       now = _np.array(now, ndmin=2)

    include = _np.arange(now.shape[1])
    if exclude_coords is not None:
       include = [ii for ii in include if ii not in exclude_coords]

    # For the list of candidates, extract the closest one
    history = [now]
    for ii, cands in enumerate(path_of_candidates):
        closest_to_now = _np.argmin(_np.sqrt(_np.sum((now[:,include]-cands[:,include])**2,1)))
        path_out.append(closest_to_now)
        # Debugging stuff
        #print("At moment %u we're at point %s and have chosen the point %s which has the index %u"%(ii, now, cands[path_out[-1]], path_out[-1]))
        #print("choose frame %u from %u cands"%(path_out[-1], len(cands)))
        now = _np.array(cands[closest_to_now], ndmin=2)
        history.append(now)
        if history_aware:
           now = _np.array(_np.vstack(history).mean(0), ndmin=2)
    return(path_out)

def min_rmsd_path(start, path_of_candidates, selection=None, history_aware=False):
    r""""Returns a list of indices [i,j,k...] s.t. the distances
     d_i = |path_of_candidates[0][i] - path_of_candidates[1][j]|,
     d_j = |path_of_candidates[1][j] - path_of_candidates[2][k]|,...
     are minimized

    Parameters
    ----------

    start : md.Trajectory object
       starting point, has to be have one frame only

    path_of_candidates: iterable of md.Trajectory objects
       The list of candidates to pick from to minimize the the rmsd to

    selection: None or iterable of ints, default is None
       The default behaviour is to consider all atoms, but this can be reduced to
       overlap only some selection of them

    history_aware: boolean, default is false
       The default behaviour is to minimize distances stepwise, i.e. the j-th step will be chosen
       to minimise distance with i. However, if i-happened to be an outlier, the entire path can get
       derailed, hence it is sometimes better to minimize with the running_average of the path.

    Returns:
    path: list of integers
       One index per array of candidates, with len(path==len(path_of_candidates), s.t. the traveled
       distance is minimized

    """
    path_out = []
    now = start
    assert now.n_frames == 1

    if selection is None:
       atom_indices = _np.arange(now.n_atoms)
    else:
       atom_indices = selection

    # For the list of candidates, extract the closest one
    history = now
    for ii, cands in enumerate(path_of_candidates):
        closest_to_now = _np.argmin(_md.rmsd(cands, now, atom_indices=atom_indices))
        path_out.append(closest_to_now)
        #print("choose frame %u from %u cands"%(path_out[-1], len(cands)))
        now = cands[closest_to_now]
        history = history.join(now)
        if history_aware:
           history.superpose(history, atom_indices=atom_indices)
           xyz = history.xyz.mean(0)
           now = _md.Trajectory(xyz, history.top)
    return path_out

def catalogues(cl, data=None, sort_by=None):
    r""" Returns the frames in catalogues form by cluster index:
     one as list (len Ncl) of ndarrays each of shape (Ni, 2) containing pairs of (traj_idx, frame_idx) values
     and one as lists of ndarrays of the actual (continous) data values at the (traj_idx, frame_idx)

    Parameters
    ----------

    cl : :obj:`pyemma.coordinates.cluster_regspace` object

    data : None or list, default is None
       The :obj:`cl` has its own  :obj:`cl.dataproducer.data` attribute from which it can
       retrieve the necessary information for  the :obj:`cat_data` (default behaviour)
       However, any other any data can be given here, **as long as the user is sure that it represents EXACTLY
       the data that was used to parametrize the :obj:`cl` object.
       Internally, the only checks that are carried out are:

           len(data) == len(cl.dataproducer.data)

           [len(idata) == len(jdata) for idata, jdata in zip(data, cl.dataproducer.data)]

       (Note that by construction the same relations should hold for :obj:`cl.dtrajs`)

    sort_by : None or int, default is None
       Default behaviour is to return the catalogues in the same order of clustercenters as the input,
       but it is sometimes useful have them sorted by ascending order of the n-th coordinate
       of the input space

    Returns
    --------

    cat_idxs : list of 2D np.arrays
        The discrete catalogue. It is a list of len = :obj:`cl.n_clustercenters` containing a 2D vector
        with all the (file, frame)-pairs in which each clustercenter appears

    cat_data : list of ndarrays
        The actual value (assumed continuous) of the data at the (file-frame)-pairs of the :obj:`cat_idxs` list

    tested: True
    """

    idata = cl.data_producer.data
    if data is not None:
       assert len(data) == len(idata)
       assert _np.all([len(jdata)==len(ddata) for jdata, ddata in zip(idata, data)])
       idata = data

    cat_idxs = _index_states(cl.dtrajs)
    cat_cont = []
    for __, icat in enumerate(cat_idxs):
        cat_cont.append(_np.vstack([idata[ii][jj] for ii,jj in icat]))

    if sort_by is not None:
       assert _is_int(sort_by)
       assert sort_by <= cl.clustercenters.shape[1], "Want to sort by %u-th coord, but centers have only %u dims"%(sort_by, cl.clustercenters.shape[1])
       sorts_coordinate = _np.argsort(cl.clustercenters[:,sort_by])
       cat_idxs = [cat_idxs[ii] for ii in sorts_coordinate]
       cat_cont = [cat_cont[ii] for ii in sorts_coordinate]

    return cat_idxs, cat_cont

def visual_path(cat_idxs, cat_data, path_type='min_disp', start_pos='maxpop', start_frame=None, **path_kwargs):
    r""" Create a path that advances in the coordinate of interest
    # while minimizing distance in the other coordinates (minimal displacement path)

    cat_idxs : list or np.ndarray of len(cat_data)
        Each element of this iterable is an ndarray (N,2) whith (traj_idx, frame_idx)
        pairs pointing towards the trajectory frames. It usually has been generated
        using cl.sample_indexes_by_cluster.

    cat_data:  iterable of length len(cat_idxs)
        Each element of this iterable contains the data correspoding to the frames contained
        in :py:obj:cat_idxs. At the moment, this data can be either an nd.array or an
        :py:obj:mdtraj.Trajectory

    start_pos: str or int, default is 'maxpop', alternatives are 'left', 'right'
       Where to start the path. It refers to an index of :py:obj:cat_idxs and :py:obj:cat_data
       Since the path is constructed to be visually appealing, it makes sense to start the path close to the most visited value of the coordinate. Options are
       'maxpop': does exactly that: Starting from the most populated value of the coordinate,
                 it creates two projection_paths, one moving forward and one moving backward.
                 These are the n and backward ('left') create a coordinate-increasing, diffusion-minimizing path from
       'left':   starts at the "left end" of the coordinate, i.e. at its minimum value, and moves forward
       'right'   starts at the "right end" of the coordinate, i.e. at its maximum value, and moves backward
        int:    path from cat_idxs[start_pop] and cat_data[start_pop]
    path_type = 'min_disp' or 'minRMSD'

    start_frame = if the user already knows, of the start_pos index, the frame that's best

    tested = False

    *path_kwargs: keyword arguments for the path-choosing algorithm. See min_disp_path or min_rmsd_path for details, but
     in the meantime, these are history_aware=True or False and exclude_coords=None or [0], or [0,1] etc...
    """
    #First sanity check
    assert len(cat_data) == len(cat_idxs)
    # Second sanity check
    assert _np.all([len(icd)==len(ici) for icd, ici in zip(cat_data, cat_idxs)])


    if start_pos == 'maxpop':
       start_idx = _np.argmax([len(icat) for icat in cat_idxs])
    elif _is_int(start_pos):
       start_idx = start_pos
    else:
       raise NotImplementedError(start_pos)

    if start_frame is None:
        # Draw a random frame from the starting point's catalgue
        start_frame = _np.random.randint(0, high=len(cat_idxs[start_idx]))

    start_fwd = cat_data[start_idx][start_frame]
    start_bwd = cat_data[start_idx][start_frame]
    if path_type == 'min_disp':
       path_fwd = [start_frame]+min_disp_path(start_fwd, cat_data[start_idx + 1:], **path_kwargs)
       path_bwd = [start_frame]+min_disp_path(start_bwd, cat_data[:start_idx][::-1], **path_kwargs)
    elif path_type == 'min_rmsd':
       path_fwd = [start_frame]+min_rmsd_path(start_fwd, cat_data[start_idx + 1:], **path_kwargs)
       path_bwd = [start_frame]+min_rmsd_path(start_bwd, cat_data[:start_idx][::-1], **path_kwargs)
    else:
         raise NotImplementedError(path_type)
    path_fwd = _np.vstack([cat_idxs[start_idx:][ii][idx] for ii, idx in enumerate(path_fwd)])
    # Take the catalogue entries until :start_idx and invert them
    # Slice up to including start_idx, need a plus one
    path_bwd = _np.vstack([cat_idxs[:start_idx+1][::-1][ii][idx] for ii, idx in enumerate(path_bwd)])
    # Invert path_bwd it and exclude last frame (otherwise the most visited appears twice)
    path = _np.vstack((path_bwd[::-1][:-1], path_fwd))

    # Sanity cheks
    #assert _np.all(_np.diff([cl.clustercenters[cl.dtrajs[ii][jj],0] for ii,jj in path])>0)
    assert len(path) == len(cat_idxs)
    return path, start_idx

def get_good_starting_point(cl, geom_samples, cl_order=None, strategy='smallest_Rgyr'):
    r""" provided a pyemma-cl object and a list of geometries, return the index of
    the clustercenter that's most suited to start a minimally diffusing path.

    Parameters
    ----------
    cl : :obj:`pyemma.coordinates` clustering object

    geom_samples : list of :obj:`mdtraj.Trajectory` objects corresponding to each clustercenter in :obj:`cl`

    cl_order : None or iterable of integers
        The order of the list :obj:`geom_samples` may or may not correspond to the order of :obj:`cl`.
        Very often, :obj:`geom_samples` is sorted in ascending order of a given coordinate while the
        clustercenters in :obj:`cl` are not. :obj:`cl_order` represents this reordering,
        so that :obj:`geom_samples[cl_order]` reproduces the order of the clusterscenters, so that finally:
        :obj:`geom_samples[cl_order][i]` contains geometries sampled for the :obj:`i`-th clustercenter

    strategy : str, default is 'smallest_Rgyr'
         Which property gets optimized
            * *smallest_Rgyr*:
              look for the geometries with smallest radius of gyration(:obj:`mdtraj.compute_rg`),
              regardless of the population

            * *most_pop*:
              look for the clustercenter that's most populated, regardless of the associated geometries

            * *most_pop_x_smallest_Rgyr*:
              Mix both criteria. Weight Rgyr values with population to avoid highly compact but
              rarely populated structures

            * *bimodal_compact*:
              assume the distribution of clustercenters is bimodal, then locate its
              centers and choose the one with smaller Rgyr

            * *bimodal_open*:
              assume the distribution of clustercenters is bimodal, then locate its
              centers and choose the one with larger Rgyr

    Returns
    -------
    start_idx : int, ndex of list :obj:`geom_samples`
        The :obj:`mdtraj.Trajectory` in :obj:`geom_samples[start_idx]` satisfies best the :obj:`strategy`
        criterion

    """
    if cl_order is None:
        cl_order = _np.arange(cl.n_clusters)
    if strategy == 'smallest_Rgyr':
        start_idx = _np.argmin([_md.compute_rg(igeoms).mean() for igeoms in geom_samples])
    elif strategy == 'most_pop':
        start_idx = (_np.bincount(_np.hstack(cl.dtrajs))[cl_order]).argmax()
    elif strategy == 'most_pop_x_smallest_Rgyr':
        rgyr = _np.array([_md.compute_rg(igeoms).mean() for igeoms in geom_samples])
        pop = (_np.bincount(_np.hstack(cl.dtrajs))[cl_order]).astype('float')
        # Normalize
        rgyr -= rgyr.min()
        rgyr = -rgyr + rgyr.max()
        rgyr /= rgyr.sum()
        pop /= pop.sum()
        start_idx = _np.argmax(rgyr*pop)

    elif strategy in ['bimodal_compact', 'bimodal_open']:
        #  assume bimodality in the coordinate of interest (usually the case at least for TIC_0)
        (left_idx, right_idx), igmm = find_centers_gmm(_np.vstack(cl.data_producer.data).reshape(-1,1),
                                                       cl.clustercenters[cl_order].squeeze(), n_components=2
                                                     )
        #  bias towards starting points with compact structures (low small radius of gyration)
        left_value, right_value = _md.compute_rg(geom_samples[left_idx]).mean(), \
                                  _md.compute_rg(geom_samples[right_idx]).mean()

        if strategy == 'bimodal_compact':
            start_idx = [left_idx, right_idx][_np.argmin([left_value, right_value])]
        else:
            start_idx = [left_idx, right_idx][_np.argmax([left_value, right_value])]
    else:
        raise NotImplementedError("This starting point strategy is unkown %s"%strategy)

    return start_idx


def find_centers_gmm(data, gridpoints, n_components=2):
    r""" Provided 1D data and a grid of points, return the indices of the points closest to
    the centers of an assumed n-modal distribution behind "data"

    data : 1D data ndarray, of shape (N, 1)
    gridpoints: 1D gridpoints

    returns: idxs, igmm
    idxs: 1D ndarray
            INDICES of gridpoints that are closest to the centers of the gaussians

    igmm: gaussian mixture model (sklearn type)

    tested = true
    """
    igmm = _GMM(n_components=n_components)
    igmm.fit(data)
    return _np.abs(gridpoints-_np.sort(igmm.means_, 0)).argmin(1), igmm

def data_from_input(projected_data):
    r""" Returns properly formatted data (list of ndarrays as data) from different types of inputs

    Parameters
    -----------

    projected data: string or list of strings containing filenames [.npy or any ascii format] with data to be read
                    or nd.array or list of ndarrays with data

    Returns
    --------

    data : list of ndarrays

    tested : True

    """

    # Create a list if ONE str or ONE ndarray are input
    if _is_string(projected_data) or isinstance(projected_data, _np.ndarray):
        projected_data = [projected_data]
    elif not isinstance(projected_data, list):
        raise ValueError("Data type not understood %s" % type(projected_data))

    if _is_string(projected_data[0]):
        if projected_data[0].endswith('npy'):
            idata = [_np.load(f) for f in projected_data]
        else:
            idata = [_np.loadtxt(f) for f in projected_data]
    else:
        idata = projected_data

    return idata

def save_traj_wrapper(traj_inp, indexes, outfile, top=None, stride=1, chunksize=1000, image_molecules=False, verbose=True):
    r"""wrapper for :pyemma:`save_traj` so that it works seamlessly with lists of :mdtraj:`Trajectories`

    Parameters
    -----------

    traj_inp : :pyemma:`FeatureReader` object or :mdtraj:`Trajectory` object or list of :mdtraj:`Trajectory` objects

    returns: see the return values of :pyemma:`save_traj`
    """

     # Listify the input in case its needed
    traj_inp = listify_if_not_list(traj_inp, except_for_these_types=(_FeatureReader, _FragmentedTrajectoryReader))

    # Do the normal thing in case of Feature_reader or list of strings
    if isinstance(traj_inp, (_FeatureReader, _FragmentedTrajectoryReader)) or _is_string(traj_inp[0]):
        geom_smpl = _save_traj(traj_inp, indexes, None, top=top, stride=stride,
                               chunksize=chunksize, image_molecules=image_molecules, verbose=verbose)
    elif isinstance(traj_inp[0], _md.Trajectory):
        file_idx, frame_idx = indexes[0]
        geom_smpl = traj_inp[file_idx][frame_idx]
        for file_idx, frame_idx in indexes[1:]:
            geom_smpl = geom_smpl.join(traj_inp[file_idx][frame_idx])
    else:
        raise TypeError("Cant handle input of type %s now"%(type(traj_inp[0])))

    return geom_smpl

def slice_list_of_geoms_to_closest_to_ref(geom_list, ref):
    r"""
    For a list of md.Trajectory objects, reduce md.Trajectory in the list
    to the frame closest to a reference
    :param geom_list: list of md.Trajectories
    :param ref: md.Trajectory
    :return: md.Trajectory of n_frames = len(geom_list), oriented wrt to ref
    """
    out_geoms = None
    for cand_geoms in geom_list:
        igeom = cand_geoms[_np.argmin(_md.rmsd(cand_geoms, ref))]
        if out_geoms is None:
            out_geoms = igeom
        else:
            out_geoms = out_geoms.join(igeom)

    return out_geoms.superpose(ref)


def get_ascending_coord_idx(pos, fail_if_empty=False, fail_if_more_than_one=False):
    r"""
    return the indices of the columns of :obj:`pos` that's sorted in ascending order

    Parameters
    ----------

    pos : 2D ndarray of shape(N,M)
        the array for which the ascending column is wanted

    fail_if_empty : bool, default is False
        If no column is found, fail

    fail_if_more_than_one : bool, default is False,
        If more than one column is found, fail. Otherwise return the first index of the ascending columns

    Returns
    -------

    idxs : 1D ndarray
        indices of the columns the values of which are sorted in ascending order
    """

    idxs = _np.argwhere(_np.all(_np.diff(pos,axis=0)>0, axis=0)).squeeze()
    if isinstance(idxs, _np.ndarray) and idxs.ndim==0:
        idxs = idxs[()]
    elif idxs == [] and fail_if_empty:
            raise ValueError('No column was found in ascending order')

    if _np.size(idxs) > 1:
        print('Found that more than one column in ascending order %s' % idxs)
        if fail_if_more_than_one:
            raise Exception
        else:
            print("Restricting to the first ascending coordinate."
                  "Band visuals might be wrong.")
            idxs = idxs[0]
    return idxs


def running_avg_idxs(l, n, symmetric=True, debug=False):
    r"""
    return the indices necessary for a running average of size 2n+1 of an array of length l

    Parameters
    ----------

    l : int
        lenght of input array

    n : int
        the averaging window will be of size 2n + 1

    symmetric : bool, default is True
        If False, the running average will be done with frame i and the n frames following it
        # TODO implement
    Returns
    -------

    frame_idx : 1D ndarray of length l-2n

    frame_window : list of length l-2n 1D ndarrays each of size 2n+1
        This list contains the frames idxs belonging to window [-n....i...n] for each i in :obj:`frame_idxs`
    """

    if not symmetric:
        # TODO
        raise NotImplementedError("Sorry symmetric=False is not implemented yet")

    assert n*2 < l, "Can't ask for a symmetric running average of 2*%u+1 with only %u frames. " \
                    "Choose a smaller parameter n."%(n, l)

    idxs = _np.arange(l)
    frame_idx= []
    frame_window = []
    for ii in idxs[n - 1 + 1 : l - n]:
        frame_idx.append(ii)
        frame_window.append(idxs[ii - n:ii + n + 1])
        if debug:
            print(frame_idx[-1],frame_window[-1])
    return _np.hstack(frame_idx), frame_window

def smooth_geom(geom, n, geom_data=None, superpose=True, symmetric=True):
    r"""
    return a smooth version of the input geometry by averaging over contiguous 2n+1 frames

    Note: Averaging **will only result in smoothing** if *contiguous* actually means something, like in a
        path-sampling, where the geometries in :obj:`geoms` are ordered in ascending order of a given projected
        coordinate) or in a trajectory, where the geometries ocurred in sequence. Otherwise, smoothing will
        *work* but will produce no meaningful results

    Parameters
    ----------
    geom: :any:`mdtraj.Trajectory' object

    n : int
        Number of frames that will be averaged over

    geom_data : nd.array of shape (geom.n_frames, N)
        If there is data associated with the geometry, smooth also the data and return it

    superpose : bool, default is True
        superpose geometries in the same window before averaging

    symmetric : bool, default is True
        An average is made so that the geometry at frame i is the average of frames
        :math: [i-n, i-n-1,.., i-1, i, i+1, .., i+n-1, i+n]

    Returns
    -------

    geom_out : :any:`mdtraj.Trajectory` object
        A geometry with 2*n less frames containing the smoothed out positions of the atoms.
        Note: you might need to re-orient this averaged geometry again
    """

    # Input checking here, otherwise we're seriously in trouble



    # Get the indices necessary for the running average
    frame_idxs, frame_windows = running_avg_idxs(geom.n_frames, n, symmetric=symmetric)

    if geom_data is not None:
        assert isinstance(geom_data, _np.ndarray), "Parameter geom_data has to be " \
                                                   "either None or a 2D ndarray. %s"%type(geom_data)
        assert geom_data.ndim == 2,                "Parameter geom_data has to be either None or a 2D ndarray, " \
                                                   "instead geom_data.ndim=%u"%geom_data.ndim
        assert geom_data.shape[0] == geom.n_frames, "Mismatch between data length (%u) and geometry length (%u)"\
                                                    %(geom_data.shape[0], geom.n_frames)
        data_out = _np.zeros((len(frame_idxs), geom_data.shape[1]))


    xyz = _np.zeros((len(frame_idxs), geom.n_atoms, 3))
    for ii, idx in enumerate(frame_idxs):
        #print(ii, idx, frame_windows[ii][n])
        if superpose:
            xyz[ii,:,:] = geom[frame_windows[ii]].superpose(geom, frame=frame_windows[ii][n]).xyz.mean(0)
        else:
            xyz[ii, :, :] = geom[frame_windows[ii]].xyz.mean(0)

        if geom_data is not None:
            data_out[ii,:] = geom_data[frame_windows[ii]].mean(0)

    geom_out = _md.Trajectory(xyz, geom.top)

    if geom_data is None:
        return geom_out
    else:
        return geom_out, data_out

def most_corr(correlation_input, geoms=None, proj_idxs=None, feat_name=None, n_args=1, proj_names='proj', featurizer=None):
    r"""
    return information about the most correlated features from a `:obj:pyemma.coodrinates.transformer` object

    Parameters
    ---------

    correlation_input : anything
        Something that could, in principle, be a :obj:`pyemma.coordinates.transformer,
        like a TICA, PCA or featurizer object or directly a correlation matrix, with a row for each feature and a column
        for each projection, very much like the :obj:`feature_TIC_correlation` of the TICA object of pyemma.

    geoms : None or obj:`md.Trajectory`, default is None
        The values of the most correlated features will be returned for the geometires in this object

    proj_idxs: None, or int, or iterable of integers, default is None
        The indices of the projections for which the most correlated feture will be returned
        If none it will default to the dimension of the correlation_input object

    feat_name : None or str, default is None
        The prefix with which to prepend the labels of the most correlated features. If left to None, the feature
        description found in :obj:`correlation_input` will be used (if available)

    n_args : int, default is 1
        Number of argmax correlation to return for each feature.

    featurizer : optional featurizer, default is None
        If :obj:`correlation_input` is not an :obj:`_MDFeautrizer` itself or doesn't have a
        data_producer.featurizer attribute, the user can input one here. If both an _MDfeaturizer *and* an :obj:`featurizer`
         are provided, the latter will be ignored.

    Returns
    -------

    a dictionary and the nglview widget
    The dictionary has these keys:
    idxs : list of lists of integers
        List of lists with the first :obj:`n_args` indices of the features that most correlate with the projected coordinates, for each
        coordinate specified in :obj:`proj_idxs`

    vals : list of lists of floats
        List with lists of correlation values (e [-1,1]) belonging to the feature indices in :obj:`most_corr_idxs'

    labels :  list of lists of strings
        The labels of the most correlated features. If a string was parsed as prefix in :obj:`feat_name`, these
        labels will be ['feat_name_%u'%i for i in most_corr_idxs']. Otherwise it will be the full feature
        description found in :obj:`pyemma.coordinates.

    feats : list of ndarrays
        If :obj:`geom` was given, this will contain the most correlated feature evaluated for every frame for
        every projection in :obj:`proj_idxs`. Otherwise this will just be an empty list

    atom_idxs : list of lists of integers
        In many cases, the most correlated feature can be represented visually by the atoms involved in defining it, e.g
            * two atoms for a distance
            * three for an angle
            * four for a dihedral
            * etc.
        If possible, most_corr will try to return these indices to be used later for visualization

    info : a list of dictionaries containing information as strings (for stdout printing use)

    """
    #TODO: extend to other inputs
    #todo:document proj_names
    # TODO: CONSIDER PURE STRING WITH \N INSTEAD for output "lines"
    # TODO: write a class instead of dictionary (easier refactoring)

    most_corr_idxs = []
    most_corr_vals = []
    most_corr_feats =  []
    most_corr_labels = []
    most_corr_atom_idxs = []
    info = []

    proj_idxs = listify_if_int(proj_idxs)

    if isinstance(correlation_input, _TICA):
        corr = correlation_input.feature_TIC_correlation
    elif isinstance(correlation_input, _PCA):
        corr = correlation_input.feature_PC_correlation
    elif isinstance(correlation_input, _np.ndarray):
        corr = correlation_input
    elif isinstance(correlation_input, _MDFeaturizer):
        corr = _np.eye(correlation_input.dimension())
        featurizer = correlation_input
    else:
        raise TypeError('correlation_input has to be either %s, not %s'%([_TICA, _PCA, _np.ndarray, _MDFeaturizer], type(correlation_input)))

    if featurizer is None:
        try:
            featurizer=correlation_input.data_producer.featurizer
            avail_FT = True
        except(AttributeError):
            avail_FT = False
    else:
        avail_FT = True

    dim = corr.shape[1]

    if avail_FT:
        assert featurizer.dimension()==corr.shape[0], "The provided featurizer and the number of rows of the " \
                                                      "correlation matrix differ in size %u vs %u"%(featurizer.dimension(), corr.shape[0])
    if proj_idxs is None:
        proj_idxs = _np.arange(dim)

    if _is_string(proj_names):
        proj_names = ['%s_%u' % (proj_names, ii) for ii in proj_idxs]

    if _np.max(proj_idxs) > dim:
        raise ValueError("Cannot ask for projection index %u if the "
                         "transformation only has %u projections"%(_np.max(proj_idxs), dim))

    for ii in proj_idxs:
        icorr = corr[:, ii]
        most_corr_idxs.append(_np.abs(icorr).argsort()[::-1][:n_args])
        most_corr_vals.append([icorr[jj] for jj in most_corr_idxs[-1]])
        if geoms is not None and avail_FT:
            most_corr_feats.append(featurizer.transform(geoms)[:, most_corr_idxs[-1]])

        if _is_string(feat_name):
            most_corr_labels.append('$\mathregular{%s_{%u}}$'%(feat_name, most_corr_idxs[-1]))
        elif feat_name is None and avail_FT:
            most_corr_labels.append([featurizer.describe()[jj] for jj in most_corr_idxs[-1]])

        if avail_FT:
            if len(featurizer.active_features) > 1:
                pass
                # TODO write a warning
            else:
                ifeat = featurizer.active_features[0]
                most_corr_atom_idxs.append(atom_idxs_from_feature(ifeat)[most_corr_idxs[-1]])

    for ii, iproj in enumerate(proj_names):
        info.append({"lines":[], "name":iproj})
        for jj, jidx in enumerate(most_corr_idxs[ii]):
            if avail_FT:
                istr = 'Corr[%s|feat] = %2.1f for %-30s (feat nr. %u, atom idxs %s' % \
                       (iproj, most_corr_vals[ii][jj], most_corr_labels[ii][jj], jidx, most_corr_atom_idxs[ii][jj])
            else:
                istr = 'Corr[%s|feat] = %2.1f (feat nr. %u)' % \
                       (iproj, most_corr_vals[ii][jj],jidx)

            info[-1]["lines"].append(istr)

    corr_dict = {'idxs': most_corr_idxs,
                 'vals': most_corr_vals,
                 'labels': most_corr_labels,
                 'feats':  most_corr_feats,
                 'atom_idxs': most_corr_atom_idxs,
                 'info':info}
    return corr_dict

def atom_idxs_from_feature(ifeat):
    r"""
    Return the atom_indices that best represent this input feature

    Parameters
    ----------

    ifeat : input feature, can be of two types:
        a :any:`pyemma.coordinates.featurizer` (Distancefeaturizer, AngleFeaturizer etc) or
        a :any:`pyemma.coordinates.data.featurization.featurizer.MDFeaturizer` itself, in which case the first of the
        obj:`ifeat.active_features` will be used

    Returns
    -------

    atom_indices : list with the atoms indices representative of this feature, whatever the feature
    """

    try:
        ifeat = ifeat.active_features[0]
    except AttributeError:
        pass

    if isinstance(ifeat, _DF) and not isinstance(ifeat, _ResMinDF):
        return ifeat.distance_indexes
    elif isinstance(ifeat, _SF):
        return _np.repeat(ifeat.indexes, 3)
    elif isinstance(ifeat, _ResMinDF):
        # Comprehend all the lists!!!!
        return _np.vstack([[list(ifeat.top.residue(pj).atoms_by_name('CA'))[0].index for pj in pair] for pair in ifeat.contacts])
    if isinstance(ifeat, (_DihF, _AF)):
        ai = ifeat.angle_indexes
        if ifeat.cossin:
            ai = _np.tile(ai, 2).reshape(-1, ai.shape[1])
        return ai
    else:
        raise NotImplementedError('bmutils.atom_idxs_from_feature cannot interpret the atoms behind %s yet'%ifeat)

def add_atom_idxs_widget(atom_idxs, ngl_wdg, color_list=None, radius=1):
    r"""
    provided a list of atom_idxs and a ngl_wdg, try to represent them as well as possible in the ngl_wdg
    It is assumed that this method is called once per feature, ie. the number of atoms defines the
    feature. This way, the method decides how to best represent them
    best to represent them. Currently, that means:
     * single atoms:   assume cartesian feature, represent with spacefill
     * pairs of atoms: assume distance feature, represent with distance
     * everything else is ignored

    Parameters
    ----------

    atom_idxs : list of iterables of integers. If [], the method won't do anything

    ngl_wdg : nglview ngl_wdg on which to represent stuff

    color_list: list, default is None
        list of colors to provide the representations with. The default None yields blue.
        In principle, len(atom_idxs) should be == len(color_list),
        but if your list is short it will just default to the last color. This way, color_list=['black'] will paint
        all black regardless len(atom_idxs)

    radius : float, default is 1
        radius of the spacefill representation

    Returns
    -------
    ngl_wdg : Input ngl_wdg with the representations added

    """

    if color_list in [None, [None]]:
        color_list = ['blue']*len(atom_idxs)
    elif isinstance(color_list, list) and len(color_list)<len(atom_idxs):
        color_list += [color_list[-1]]*(len(atom_idxs)-len(color_list))

    if atom_idxs is not []:
        for cc in range(len(ngl_wdg._ngl_component_ids)):
            for iidxs, color in zip(atom_idxs, color_list):
                if _is_int(iidxs):
                    ngl_wdg.add_spacefill(selection=[iidxs], radius=radius, color=color, component=cc)
                elif _np.ndim(iidxs)>0 and len(iidxs)==2:
                    ngl_wdg.add_distance(atom_pair=[[ii for ii in iidxs]],  # yes it has to be this way for now
                     color=color,
                                         #label_color='black',
                     label_size=0,
                                         component=cc)
                    # TODO add line thickness as **kwarg
                elif _np.ndim(iidxs) > 0 and len(iidxs) in [3,4]:
                    ngl_wdg.add_spacefill(selection=iidxs, radius=radius, color=color, component=cc)
                else:
                    print("Cannot represent features involving more than 5 atoms per single feature")

    return ngl_wdg

def transpose_geom_list(geom_list):
    r"""
    Transpose a list of md.Trajectory objects, so that an input having the frames input
    geom_list[0] = md.Trajectory with frames f00, f01, f02,..., f0M
    geom_list[1] = md.Trajectory with frames f10, f11, f12,..., f1M
    geom_list[2] = md.Trajectory with frames f20, f21, f22,..., f2M
    ...
    geom_list[N] = md.Trajectory with frames fN0, fN1, fN2,..., fNM

    gets transposed to

    geom_list[0] = md.Trajectory with frames f00, f10, f20,..., fM0
    geom_list[1] = md.Trajectory with frames f01, f11, f21,..., fM1
    geom_list[2] = md.Trajectory with frames f02, f12, f22,..., fM2
    ...
    geom_list[M] = md.Trajectory with frames f0N, f1N, f2N,..., fMN

    Parameters
    ----------

    geom_list : list of md.Trajectory objects, all must have the same geom.n_frames


    Returns
    -------

    geom_list_T : list of md.Trajectory objects (transposed)
    """

    assert isinstance(geom_list, list)
    n_frames_per_element = geom_list[0].n_frames
    assert _np.all([igeom.n_frames==n_frames_per_element for igeom in geom_list]), \
        "All geometries in the list have to have the same length"


    list_out = [[igeom[ii] for igeom in geom_list] for ii in range(n_frames_per_element)]
    geom_list_T = []
    for ilist in list_out:
        igeom = ilist[0]
        #TODO: avoid joining via copy_not_join
        for jgeom in ilist[1:]:
            igeom = igeom.join(jgeom)
        geom_list_T.append(igeom)

    return(geom_list_T)

def geom_list_2_geom(geom_list):
    r"""
    Join a list of md.Trajectory objects to one single trajectory


    Parameters
    ----------

    geom_list : list of md.Trajectory objects, each can have the arbitrary geom.n_frames


    Returns
    -------

    geom : of md.Trajectory object
    """

    assert isinstance(geom_list, list)

    geom = geom_list[0]
    #TODO: avoid joining via copy_not_join
    for jgeom in geom_list[1:]:
        geom = geom.join(jgeom)


    return(geom)

def labelize(proj_labels, proj_idxs):
    r"""
    Returns a list of strings of axis labels constructed from proj_labels and proj_idxs

    Parameters
    ----------

     proj_labels: string, list of strings, or a :obj:`pyemma.MDFeaturizer` object with a .describe method

     proj_idxs: list of integers with the projections


    Returns:
     proj_labels : list of strings of length len(proj_idxs)

    """
    # TODO TEST
    try:
        proj_labels.describe()
        proj_labels = [proj_labels.describe()[ii] for ii in proj_idxs]
    except AttributeError:
        pass

    if isinstance(proj_labels, str):
       proj_labels = ['$\mathregular{%s_{%u}}$'%(proj_labels, ii) for ii in proj_idxs]
    elif isinstance(proj_labels, list):
       pass
    else:
       raise TypeError("Parameter proj_labels has to be of type str or list, not %s"%type(proj_labels))

    return proj_labels

def superpose_to_most_compact_in_list(superpose_info, geom_list):
    r"""
    Provided a list of `mdtraj.Trajectory` objects, orient them to the most compact possible
    structure according to :obj:`superpose_info`

    Parameters
    ----------

    superpose_info : boolean, str, or iterable of integers
        boolean : "True" orients with all atoms or "False" won't do anything
        str  : superpose according to anything :obj:`mdtraj.Topology.select` can understand (http://mdtraj.org/latest/atom_selection.html)
        iterable of integers : superpose according to these atom idxs

    geom_list : list of :obj:`mdtraj.Trajectory` objects


    Returns
    -------

    geom_list : list of :obj:`mdtraj.Trajectory` objects
    """
    # Superpose if wanted
    sel = parse_atom_sel(superpose_info, geom_list[0].top)

    if sel is not None:
        ref = geom_list_2_geom(geom_list)
        ref = ref[_md.compute_rg(ref).argmin()]
        geom_list = [igeom.superpose(ref, atom_indices=sel) for igeom in geom_list]

    return geom_list

def parse_atom_sel(atom_selection, top):
    r"""
    Provided an `mdtraj.Topology` and :obj:`superpose_info` get the atoms that are needed
    to a subsequent superposition operation

    Parameters
    ----------

    atom_selection : boolean, str, or iterable of integers
        boolean : "True" orients with all atoms or "False" won't do anything
        str  : superpose according to anything :obj:`mdtraj.Topology.select` can understand (http://mdtraj.org/latest/atom_selection.html)
        iterable of integers : superpose according to these atom idxs

    top : :obj:`mdtraj.Topology` object


    Returns
    -------

    sel : iterable of integers or None
    """
    # Superpose if wanted
    sel = None
    if atom_selection is True:
        sel = _np.arange(top.n_atoms)
    elif atom_selection is False:
        pass
    elif isinstance(atom_selection, str):
        sel = top.select(atom_selection)
    elif isinstance(atom_selection, (list, _np.ndarray)):
        assert _np.all([_is_int(ii) for ii in atom_selection])
        sel = atom_selection
    return sel
