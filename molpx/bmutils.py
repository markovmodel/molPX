from __future__ import print_function
import numpy as _np
import mdtraj as _md
from matplotlib import pyplot as _plt
from matplotlib.widgets import AxesWidget as _AxesWidget
from glob import glob
import os
import tempfile

try:
    from sklearn.mixture import GaussianMixture as _GMM
except ImportError:
    from sklearn.mixture import GMM as _GMM

from pyemma.util.linalg import eig_corr
from pyemma.coordinates import source as _source, \
    cluster_regspace as _cluster_regspace, \
    save_traj as _save_traj
from pyemma.coordinates.data.feature_reader import  FeatureReader as _FeatureReader
from pyemma.util.discrete_trajectories import index_states as _index_states
from scipy.spatial import cKDTree as _cKDTree
#from myMDvisuals import customvmd

def re_warp(array_in, lengths):
    """Return iterable ::py:obj:array_in as a list of arrays, each
     one with the length specified in lengths

    Parameters
    ----------

    array_in: any iterable
        Iterable to be re_warped

    lenghts : iterable of integers
        Lenghts of the individual elements of the returned array


    Returns
    -------
    warped: list
    """

    warped = []
    idxi = 0
    for ii, ll in enumerate(lengths):
        warped.append(array_in[idxi:idxi+ll])
        idxi += ll
    return warped

def dictionarize_list(list, input_dict, output_dict = None):
    if output_dict is None:
        output_dict = {}

    for item in list:
        output_dict[item] = input_dict[item]
    return output_dict

def correlations2CA_pairs(icorr,  geom_sample, corr_cutoff_after_max=.95, feat_type='md.contacts'):
    max_corr = _np.abs(icorr).argsort()[::-1]
    max_corr = [max_corr[0]]+[mc for mc in max_corr[1:] if icorr[mc]>=corr_cutoff_after_max]
    top = geom_sample[0].top
    if feat_type == 'md.contacts':
        res_pairs = _md.compute_contacts(geom_sample[0][0])[1]
    else:
        raise NotImplementedError('%s feat type is not implemented (here) yet'%feat_type)

    CA_pairs = []
    for ii,jj in res_pairs[max_corr]:
        CA_pairs.append([top.residue(ii).atom('CA').index,
                         top.residue(jj).atom('CA').index])
    CA_pairs = _np.vstack(CA_pairs)

    return CA_pairs, max_corr

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

    assert _np.vstack(data).shape[0] >= n_clusters_target, "Cannot cluster " \
                                                      "%u datapoints on %u clustercenters. Reduce the number of target " \
                                                      "clustercenters."%(_np.vstack(data).shape[0], n_clusters_target)
    # Works well for connected, 1D-clustering,
    # otherwise it's bad starting guess for dmin
    cmax = _np.vstack(data).max()
    cmin = _np.vstack(data).min()
    dmin = (cmax-cmin)/(n_clusters_target+1)

    err = _np.ceil(n_clusters_target*delta)
    cl = _cluster_regspace(data, dmin=dmin)
    for cc in range(n_try_max):
        n_cl_now = cl.n_clusters
        delta_cl_now = _np.abs(n_cl_now - n_clusters_target)
        if not n_clusters_target-err <= cl.n_clusters <= n_clusters_target+err:
            # Cheap (and sometimes bad) heuristic to get relatively close relatively quick
            dmin = cl.dmin*cl.n_clusters/   n_clusters_target
            cl = _cluster_regspace(data, dmin=dmin, max_centers=5000)# max_centers is given so that we never reach it (dangerous)
        else:
            break
        if verbose:
            print('cl iter %u %u -> %u (Delta to target (%u +- %u): %u'%(cc, n_cl_now, cl.n_clusters,
                                                                         n_clusters_target, err, delta_cl_now))
    return cl

def fake_md_iterator(traj, chunk=None, stride=1):
    r"""Returns a list of (strided) trajectories of length chunk from 
    one single trajecty object
    """ 
    
    idxs = _np.arange(traj.n_frames)[::stride]
    if chunk is None:
       chunk = [len(idxs)]
    elif isinstance(chunk, (int, _np.int32, _np.int64)):       
       chunk = [chunk]*_np.ceil(len(idxs)/chunk)
    return [traj[idxs] for idxs in re_warp(idxs, chunk)]

def geom2tic(geom, tica_mean, U, cutoff=.4, kinetic_map_scaling=None, chunk=None):
    r"""Equivalent to pyemma.coordinates.tica.transform for Simon's feature
    """
    if chunk is None:
       n = geom.n_frames

    Y = []
    append = Y.append
    for ctcs in [_md.compute_contacts(igeom)[0]<cutoff for igeom in fake_md_iterator(geom, chunk=chunk)]:
        ctcs_meanfree = _np.array(ctcs-tica_mean, ndmin=2)
        if kinetic_map_scaling is None:
           l = _np.ones(ctcs.shape[1])[:U.shape[1]]
        else:
           l = kinetic_map_scaling[:U.shape[1]]
        append(_np.dot(ctcs_meanfree, U)*l[_np.newaxis,:])

    return _np.vstack(Y)

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
    if isinstance(exclude_coords, int):
        exclude_coords = [exclude_coords]
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

    Parameters:
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
    r""" Returns a catalogue of frames from a :obj:`pyemma.coordinates.cluster_regspace` object

    Parameters
    ----------

    cl : :obj:`pyemma.coordinates.cluster_regspace` object
    
    data : None or list, default is None
       The :obj:`cl` has its own  :obj:`cl.dataproducer.data` attribute from which it can
       retrieve the necessary information for  the :obj:`cat_cont` (default behaviour)
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

    cat_cont : list of ndarrays
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
       assert isinstance(sort_by, int)
       assert sort_by <= cl.clustercenters.shape[1], "Want to sort by %u-th coord, but centers have only %u dims"%(sort_by, cl.clustercenters.shape[1])
       sorts_coordinate = _np.argsort(cl.clustercenters[:,sort_by])
       cat_idxs = [cat_idxs[ii] for ii in sorts_coordinate]
       cat_cont = [cat_cont[ii] for ii in sorts_coordinate]

    return cat_idxs, cat_cont

def visual_path(cat_idxs, cat_cont, path_type='min_disp', start_pos='maxpop', start_frame=None, **path_kwargs):
    r""" Create a path that advances in the coordinate of interest
    # while minimizing distance in the other coordinates (minimal displacement path)

    start_pos: str, defualt is 'maxpop', alternatives are 'left', 'right'
       Where to start the path. Since the path is constructed to be visually appealing, 
       it makes sense to start the path close to the most visited value of the coordinatet. Options are
       'maxpop': does exactly that: Starting from the most populated value of the coordinate, 
                 it creates two projection_paths, one moving forward and one moving backward.
                 These are the n and backward ('left') create a coordinate-increasing, diffusion-minimizing path from 
       'left':   starts at the "left end" of the coordinate, i.e. at its minimum value, and moves forward
       'right'   starts at the "right end" of the coordinate, i.e. at its maximum value, and moves backward
     
    path_type = 'min_disp' or 'minRMSD'

    start_frame = if the user already knows, of the start_pos index, the frame that's best

    tested = False

    *path_kwargs: keyword arguments for the path-choosing algorithm. See min_disp_path or min_rmsd_path for details, but
     in the meantime, these are history_aware=True or False and exclude_coords=None or [0], or [0,1] etc...
    """
    if start_pos == 'maxpop':
       start_idx = _np.argmax([len(icat) for icat in cat_idxs])
    elif isinstance(start_pos, (int, _np.int32, _np.int64)):
       start_idx = start_pos
    else:
       raise NotImplementedError(start_pos) 

    if start_frame is None:
        # Draw a random frame from the starting point's catalgue
        start_frame = _np.random.randint(0, high=len(cat_idxs[start_idx]))

    start_fwd = cat_cont[start_idx][start_frame]
    if path_type == 'min_disp':
       path_fwd = [start_frame]+min_disp_path(start_fwd, cat_cont[start_idx+1:], **path_kwargs)
    elif path_type == 'min_rmsd':
       path_fwd = [start_frame]+min_rmsd_path(start_fwd, cat_cont[start_idx+1:], **path_kwargs)
    else:
         raise NotImplementedError(path_type)
    path_fwd = _np.vstack([cat_idxs[start_idx:][ii][idx] for ii, idx in enumerate(path_fwd)])
    # Path backward, 
    # Take the catalogue entries until :start_idx and invert them
    start_bwd = cat_cont[start_idx][start_frame]
    if path_type == 'min_disp':
       path_bwd = [start_frame]+min_disp_path(start_bwd, cat_cont[:start_idx][::-1], **path_kwargs)
    elif path_type == 'min_rmsd':
         path_bwd = [start_frame]+min_rmsd_path(start_bwd, cat_cont[:start_idx][::-1], **path_kwargs)
    else:
         raise NotImplementedError(path_type)
    # Slice up to including start_idx, need a plus one
    path_bwd = _np.vstack([cat_idxs[:start_idx+1][::-1][ii][idx] for ii, idx in enumerate(path_bwd)]) 
    # Invert path_bwd it and exclude last frame (otherwise the most visited appears twice)
    path = _np.vstack((path_bwd[::-1][:-1], path_fwd))

    # Sanity cheks
    #assert _np.all(_np.diff([cl.clustercenters[cl.dtrajs[ii][jj],0] for ii,jj in path])>0)
    assert len(path) == len(cat_idxs)
    return path, start_idx

def input2output_corr(icov, U):
    r""" Equivalent to feature_TIC_correlation of a pyemma-TICA object
    """
    feature_sigma = _np.sqrt(_np.diag(icov))
    return _np.dot(icov, U) / feature_sigma[:, _np.newaxis]

def sequential_rmsd_fit(geomin, start_frame=0):
    r"""Provided an md.Trajectory object and a starting frame, sequentially (forward and backwars) orient the frames
    to maximize the overlap between neihgboring-structures in geomin and return them
    """

    fwd_fit =  geomin[start_frame]
    for geom in geomin[start_frame+1:]:
        fwd_fit = fwd_fit.join(geom.superpose(fwd_fit[-1]))
    bwd_fit = geomin[start_frame]
    for geom in geomin[:start_frame][::-1]:
        bwd_fit = bwd_fit.join(geom.superpose(bwd_fit[-1]))

    visual_fit = bwd_fit[::-1][:-1].join(fwd_fit)
    assert visual_fit.n_frames == geomin.n_frames

    return visual_fit


def opentica_npz(ticanpzfile):
    r"""Open a simon-type of ticafile.npz and return some variables
    """
    lag_str = os.path.basename(ticanpzfile).replace('tica_','').replace('.npz','')
    trajdata = _np.load(ticanpzfile, encoding='latin1')
    icov, icovtau = trajdata['tica_cov'], trajdata['tica_cov_tau']
    l, U = eig_corr(icov, icovtau)
    tica_mean = trajdata['tica_mean']
    data = trajdata['projdat']
    corr = input2output_corr(icov, U)

    return lag_str, data, corr, tica_mean, l, U

def get_good_starting_point(cl, geom_samples, cl_order=None, strategy='smallest_Rgyr'):
    r""" provided a pyemma-cl object and a list of geometries, return the index of
    the clustercenter that's most suited to start a minimally diffusing path.

    cl: pyemma clustering object
    geom_samples: list of md.Trajectory objects corresponding to each clustercenter
    cl_order: None or iterable of integers
        Typically, the ordering of the list  "geom_samples" has meaning, i.e., the sampled-geometries are listed
        in ascending order of a given coordinate. MOST OF THE TIME, THIS WILL NOT BE THE CASE of the centers
        in the cl object, which is arbitrary. cl_order represents this reordering, such that geom_samples[cl_order] will
         reorder the list to represent the order of the clusterscenters:
         geom_samples[cl_order][ii] contains geometries sampled for the ii-th clustercenter

    returns:
    start_idx: int
        Index referring to the list of md.trajectories in geom_samples that best satisfies the "strategy" criterion

    Tested: true
    #TODO DOCUMENT
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


def find_centers_bimodal(distro, space, barrier=0):
    #, geoms, mdfunction, pop=None, barrier=0, fitness='min')
    r"""
    pop :  function assumed to be bimodal
    space: space in which the function is bimodal 


    ##given a space, divide it in two halves and find the max_pop on each side of the barrier
    """
    assert _np.ndim(distro) == _np.ndim(space) == 1
    assert len(space) == len(distro)
    n_points = len(space)

    # Points to the left of the barrier
    left = [ii for ii in range(n_points) if space[ii] < barrier] 
    # Take the most populated
    left = left[_np.argmax([distro[ii] for ii in left])] 

    # Points to the right of the barrier 
    right = [ii for ii in range(n_points) if space[ii] > barrier] 
    # Take the most populated 
    right = right[_np.argmax([distro[ii] for ii in right])] 

    return left, right
    """
    rg_pos = function(geoms[start_idx_pos]).mean()
    rg_neg = function(geoms[start_idx_neg]).mean()
    start_idx = [start_idx_neg, start_idx_pos][np.argmin([rg_neg, rg_pos])]
    path_smpl, start_idx_2 = visual_path(cat_smpl, geom_smpl, path_type='min_rmsd', start_pos=start_idx, history_aware=True) 
    Y_path_smpl = np.vstack([idata[ii][jj] for ii,jj in path_smpl])
    trajs_path_smpl = pyemma.coordinates.save_traj(src.filenames, path_smpl, None, stride=traj_stride, top=struct) 
    assert start_idx_2 == start_idx
    """

def elements_of_list1_in_list1(list1, list2):
    tokeep = []
    for el1 in list1:#args.target_featurizations:
        for el2 in list2:
            if el1 in el2:
                tokeep.append(el2)
                break
    return tokeep

def targets_in_candidates(candidates, targets, verbose=True ):
    if verbose:
        print('Available:\n', '\n'.join(candidates))

    if isinstance(targets, str):
        if targets == 'all':
            if verbose:
                print('All will be kept')
            out_list = candidates
        else:
            targets = list(targets)
    if isinstance(targets, list):
        out_list = elements_of_list1_in_list1(targets, candidates)
        if out_list == []:
            raise  ValueError('Your target features do not match any of the available tica_featurizations:\n'
                              '%s'%('\n'.join(targets)))
        else:
            if verbose:
                print('Kept:\n', '\n'.join(out_list))

    return out_list

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
    if isinstance(projected_data, str) or isinstance(projected_data, _np.ndarray):
        projected_data = [projected_data]
    elif not isinstance(projected_data, list):
        raise ValueError("Data type not understood %"%type(projected_data))

    if isinstance(projected_data[0],str):
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
    if not isinstance(traj_inp, list) and not isinstance(traj_inp, _FeatureReader):
        traj_inp = [traj_inp]

    # Do the normal thing in case of Feature_reader or list of strings
    if isinstance(traj_inp, _FeatureReader) or isinstance(traj_inp[0], str):
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

def minimize_rmsd2ref_in_sample(sample, ref):
    # Candidate selection

    out_geoms = None
    for cand_geoms in sample:
        igeom = cand_geoms[(_np.argmin(_md.rmsd(cand_geoms, ref)))]
        if out_geoms is None:
            out_geoms = igeom
        else:
            out_geoms = out_geoms.join(igeom)

    return out_geoms.superpose(ref)

def src_in_this_proj(proj, mdtraj_dir,
                      dirstartswith='DESRES-Trajectory_',
                      strfile_fmt = '%s-%u-protein',
                      ext='dcd',
                     starting_idx = 0, ## to deal with some dir-structure, unclean solution by now,
                     struct = None,
                     ):
    xtcs = []
    ii = starting_idx

    tocheck = os.path.join(mdtraj_dir, dirstartswith+proj+'*')
    assert len(glob(tocheck)) != 0,("globbing for %s yields an empty list"%tocheck)
    tocheck = sorted(glob(tocheck))
    if isinstance(tocheck, str):
        tocheck = [tocheck]
    for __, idir in enumerate(tocheck):
        if not idir.endswith('tar.gz'):
            subdir = os.path.join(idir,strfile_fmt%(proj, ii))
            these_trajs = os.path.join(subdir,'*'+ext)
            assert len(glob(these_trajs)) != 0,("globbing for %s yields an empty list"%these_trajs)
            these_trajs = sorted(glob(these_trajs))
            xtcs.append(these_trajs)
            if struct is None:
                struct = '%s-%u-protein.pdb'%(proj,ii)
            elif isinstance(struct, str):
                struct = os.path.join(subdir,struct)
                struct = sorted(glob(struct))
            if isinstance(struct,list):
                struct=struct[0]
            ii += 1

    src = _source(xtcs, top=struct)

    return src, xtcs

def plot_histo_reaction_coord(data, reaction_path, reaction_coord_idx, start_idx=None,
                              log=False, extra_paths=[], smooth_path = True, type_of_coord='TIC',
                              contour_alpha = 1.0, n_sigma=1,
                              **plot_path_kwargs):

    # 2D plot with the coordinate of choice as x-axis and the most varying as y-axis
    # "visual" coordinates (coords to visualize the 2D plots in)
    v_crd_1 = reaction_coord_idx # easy
    # Appart from  the reaction coordinate, which other coordinates vary the most?
    ivar = reaction_path.var(0)
    ivar[reaction_coord_idx] = 0
    v_crd_2 = _np.argmax(ivar)
    h, (x, y) = _np.histogramdd(_np.vstack(data)[:,[v_crd_1, v_crd_2]], bins=100)
    if log:
       h = -_np.log(h)
    _plt.figure()
    _plt.contourf(h.T, extent=[x[0], x[-1], y[0], y[-1]], alpha=contour_alpha)

    if start_idx is not None:
       refs = [reaction_path[start_idx, [v_crd_1, v_crd_2]]]
    else:
       refs = []

    path_list = [reaction_path[:,[v_crd_1, v_crd_2]]]
    if extra_paths != []:
       path_list += [ipath[:,[v_crd_1, v_crd_2]] for ipath in extra_paths]
    plot_path_kwargs['xlabel'] = '%s_%u'%(type_of_coord,v_crd_1)
    plot_path_kwargs['ylabel'] = '%s %u'%(type_of_coord,v_crd_2)
    plot_paths(path_list, refs=refs, ax=_plt.gca(), **plot_path_kwargs)

    if smooth_path:
       # "Some" heuristc to arrive at nice looking trajectory
       #dp = np.sqrt(np.sum(np.diff(Y_path, 0)**2, 1)) # displacements along the path
       #d_path = pyemma.coordinates.cluster_regspace(Y_path, dmin=2*dp.mean()).dtrajs[0]
       #largest_con_set = np.bincount(d_path).argmax()
       #compact_path = np.argwhere(d_path==largest_con_set).squeeze()
       compact_paths = [_np.argwhere(_np.abs(path[:,1]-path[:, 1].mean())<n_sigma*path[:,1].std()).squeeze() for path in path_list]
       path_list = [path[cp] for cp, path in zip(compact_paths, path_list)]
       path_labels = [lbl+' (%u sigma)'%n_sigma for lbl in plot_path_kwargs['path_labels']]

    iax = _plt.gca()
    iax.set_prop_cycle(None)
    #iax.set_color_cycle(None)
    for pp, ll in zip(path_list, path_labels):
        iax.plot(pp[:,0], pp[:,1], ' o', markeredgecolor='none', label=ll)
    iax.legend(loc='best')

    #plot_paths(path_list, refs=refs, ax=_plt.gca(), **plot_path_kwargs)

    # Prepare a dictionary with plot-specific stuff

    return compact_paths, dictionarize_list(['h', 'x', 'y', 'v_crd_1', 'v_crd_2', 'log'], locals())

def plot_paths(path_list, path_labels=[],
    ax=None, refs=[], backdrop=True, legend=True, markers=None, linestyles=None, alpha=1,
    xlabel='x', ylabel='y'):
    if ax is None:
      _plt.figure()
    else:
      _plt.sca(ax)

    if path_labels == []:
       path_labels = ['path_%u'%u for u in range(len(path_list))]

    if markers is None:
       markers = ' ' 

    if isinstance(markers, str) and len(markers)==1:
       markers = [markers] * len(path_list)

    if linestyles is None:
       linestyles = ' '

    if isinstance(linestyles, str):
       linestyles = [linestyles] *len(path_list)

    
    for ipath, ilabel, imark, ils in zip(path_list, path_labels, markers, linestyles):
       _plt.plot(ipath[:,0], ipath[:,1], marker=imark, linestyle = ils, label=ilabel, alpha=alpha)

    for iref in refs:
       _plt.plot(iref[0], iref[1], 'ok', zorder=10) 
    
    if backdrop:
      _plt.hlines(0, *_plt.gca().get_xlim(), color='k', linestyle='--')
      _plt.vlines(0, *_plt.gca().get_ylim(), color='k', linestyle='--')

    if legend:
      _plt.legend(loc='best')
    
    _plt.xlabel(xlabel)
    _plt.ylabel(ylabel)

def extract_visual_fnamez(fnamez, path_type, keys=['x','y','h',
                                            'v_crd_1', 'v_crd_2'
                                           ]
                  ):
    r"""
    Once the project has been visualized, extract key visual parameters

    """
    # Load stuff to the namespace
    a = _np.load(fnamez)
    if path_type == 'min_rmsd':
        data = a['Y_path_smpl']
        selection = a['compact_path_sample']
    elif path_type == 'min_disp':
        data = a['Y_path']
        selection = a['compact_path']
    else:
        raise ValueError("What type of path is %"%path_type)

    return [data, selection]+[a[key] for key in keys]

def link_ax_w_pos_2_nglwidget(ax, pos, nglwidget, link_with_lines=True, radius=0):
    r"""
    Initial idea for this function comes from @arose, the rest is @gph82
    """


    kdtree = _cKDTree(pos)
    assert nglwidget.trajectory_0.n_frames == pos.shape[0], \
        ("Mismatching frame numbers %u vs %u"%( nglwidget.trajectory_0.n_frames, pos.shape[0]))
    x, y = pos.T

    if link_with_lines:
        lineh = ax.axhline(ax.get_ybound()[0], c="black", ls='--')
        setattr(lineh, 'whatisthis', 'lineh')
        linev = ax.axvline(ax.get_xbound()[0], c="black", ls='--')
        setattr(linev, 'whatisthis', 'linev')
        showclick_objs=[lineh, linev]

    dot = ax.plot(pos[0,0],pos[0,1], 'o', c='red', ms=7)[0]
    setattr(dot,'whatisthis','dot')
    closest_to_click_obj = [dot]

    if radius > 0:
        rad = ax.plot(pos[0,0],pos[0,1], 'o',
                      ms=(1+radius**2)*7,
                      c='red', alpha=.25, markeredgecolor=None)[0]
        setattr(rad, 'whatisthis', 'dot')
        closest_to_click_obj.append(rad)
        rlinev = ax.axvline(ax.get_xbound()[0],
                            lw=2*radius*7,
                            c="red", ls='-',
                            alpha=.25)
        setattr(rlinev, 'whatisthis','linev')
        closest_to_click_obj.append(rlinev)

    nglwidget.isClick = False
    def onclick(event):
        if link_with_lines:
            for iline in showclick_objs:
                update2Dlines(iline,event.xdata, event.ydata)

        data = [event.xdata, event.ydata]
        _, index = kdtree.query(x=data, k=1)
        for idot in closest_to_click_obj:
            update2Dlines(idot,x[index],y[index])

        nglwidget.isClick = True
        nglwidget.frame = index

    def my_observer(change):
        r"""Here comes the code that you want to execute
        """
        #for c in change:
        #    print("%s -> %s" % (c, change[c]))
        nglwidget.isClick = False
        _idx = change["new"]
        try:
            for idot in closest_to_click_obj:
                update2Dlines(idot, x[_idx], y[_idx])
            #print("caught index error with index %s (new=%s, old=%s)" % (_idx, change["new"], change["old"]))
        except IndexError as e:
            for idot in closest_to_click_obj:
                update2Dlines(idot, x[0], y[0])
            print("caught index error with index %s (new=%s, old=%s)" % (_idx, change["new"], change["old"]))
            pass
        #print("set xy = (%s, %s)" % (x[_idx], y[_idx]))

    # Connect axes to widget
    axes_widget = _AxesWidget(ax)
    axes_widget.connect_event('button_release_event', onclick)

    # Connect widget to axes
    nglwidget.observe(my_observer, "frame", "change")

    return axes_widget

def update2Dlines(iline, x, y):
    r"""
    provide a common interface to update objects on the plot
    :param iline:
    :param x:
    :param y:
    :return:
    """
    # TODO FIND OUT A CLEANER WAY TO DO THIS (dict or class)

    if not hasattr(iline,'whatisthis'):
        raise AttributeError("This method will only work if iline has the attribute 'whatsthis'")
    else:
        # TODO find cleaner way of distinguishing these 2Dlines
        if iline.whatisthis in ['dot']:
            iline.set_xdata((x))
            iline.set_ydata((y))
        elif iline.whatisthis in ['lineh']:
            iline.set_ydata((y,y))
        elif iline.whatisthis in ['linev']:
            iline.set_xdata((x,x))
        else:
            # TODO: FIND OUT WNY EXCEPTIONS ARE NOT BEING RAISED
            raise TypeError("what is this type of 2Dline?")

def myflush(pipe, istr='#', size=1e4):
    pipe.write(''.join([istr+'\n\n' for ii in range(int(size))]))

def link_ax_w_pos_2_vmd(ax, pos, geoms, **customVMD_kwargs):
    r"""
    Initial idea and key VMD-interface for this function comes from @fabian-paul
    #TODO: CLEAN THE TEMPFILE
    """

    # Prepare tempdir
    tmpdir = tempfile.mkdtemp('vmd_interface')
    print("please remember to: rm -r %s"%tmpdir)

    # Prepare files
    topfile = os.path.join(tmpdir,'top.pdb')
    trjfile = os.path.join(tmpdir,'trj.xtc')
    geoms[0].save(topfile)
    geoms[1:].superpose(geoms[0]).save(trjfile)

    # Create pipe
    pipefile = os.path.join(tmpdir,'vmd_cmds.tmp.vmd')
    os.mkfifo(pipefile)
    os.system("vmd < %s & "%pipefile)

    vmdpipe = open(pipefile,'w')
    [vmdpipe.write(l) for l in customvmd(topfile, trajfile=trjfile, vmdout=None,
                                        **customVMD_kwargs)]
    myflush(vmdpipe)
    kdtree = _cKDTree(pos)
    x, y = pos.T

    lineh = ax.axhline(ax.get_ybound()[0], c="black", ls='--')
    linev = ax.axvline(ax.get_xbound()[0], c="black", ls='--')
    dot, = ax.plot(pos[0,0],pos[0,1], 'o', c='red', ms=7)

    def onclick(event):
        linev.set_xdata((event.xdata, event.xdata))
        lineh.set_ydata((event.ydata, event.ydata))
        data = [event.xdata, event.ydata]
        _, index = kdtree.query(x=data, k=1)
        dot.set_xdata((x[index]))
        dot.set_ydata((y[index]))
        vmdpipe.write(" animate goto %u;\nlist;\n\n"%index)
        #myflush(vmdpipe,
                #size=1e4
        #        )

    # Connect axes to widget
    axes_widget = _AxesWidget(ax)
    axes_widget.connect_event('button_release_event', onclick)

    return vmdpipe

def vmd_stage(background='white',
              axes=False,
              ):

    mystr = 'color Display Background %s\n'%background
    if not axes:
        mystr += 'axes location off\n'

    mystr += '\n'

    return mystr

def customvmd(structure,
              vmdout='custom.vmd',
              rep = 'lines',
              frames = None,
              smooth = None,
              distcolor = 'red',
              molcolor = 'name',
              atoms = None,
              atomrep = 'VDW 1. 12.',
              atompairs=None,
              residues = None,
              rescolor = 'name',
              trajfile=None,
              strfilextra = None,
              trajfilextra = None,
              atompairsextra = None,
              white_stage = True,
              freetext = None,
              **mdtraj_save_kwargs):

    lines = []
    lappend = lines.append

    if white_stage:
       [lappend(l+'\n') for l in vmd_stage().split('\n')] #bad! code
    if isinstance(structure, _md.Trajectory):
        raise NotImplementedError
        """
        strfile = 'temp'+_path.splitext(vmdout)[0]+'.%u.pdb'%_np.random.randint(1e8)
        lappend('set outfile [open "%s" w]\n'%strfile)
        for line in str.splitlines(structure.save_pdb(_StringIO(), **mdtraj_save_kwargs)):
            lappend('puts $outfile "%s"\n'%line)
        lappend("close $outfile\n")
        lappend('mol new %s waitfor all\n'%strfile)
        lappend('rm %s\n'%strfile)
        """
    elif isinstance(structure, str):
        strfile = structure
        lappend('mol new %s waitfor all\n'%strfile)
    else:
        raise Exception

    lappend('mol delrep 0 top\n')
    lappend('mol representation %s \n'%rep)
    lappend('mol color %s\n'%molcolor)
    lappend('mol addrep top\n')
    if smooth is not None:
       lappend('mol smoothrep top 0 %s\n'%smooth)
    if frames is not None:
       frames = '{%s}'%', '.join(['%u'%ii for ii in frames])
       lappend('mol drawframes top 0 %s\n'%frames)
    lappend('label textsize 0.000001\n')
    lappend('color Labels {Bonds} %s\n'%distcolor)

    #Cycle through the atoms
    if atoms is not None:
        atoms = _np.unique(atoms)
        if _np.size(atoms)==1:
           atoms = [atoms]
        atomsel='%u '*len(atoms)%(tuple(atoms))
        myline='mol representation %s\n'%atomrep
        lappend(myline)
        myline='mol selection {index %s}\n'%atomsel
        lappend(myline)
        myline='mol addrep top\n'
        lappend(myline)

    #Cycle through residues
    if residues is not None:
        assert _is_iterable_of_int(residues)
        residsel=' '.join([str(rr) for rr in residues])
        myline='mol representation Licorice .3 10. 10.\n'
        lappend(myline)
        myline='mol color %s\n'%rescolor
        lappend(myline)
        myline='mol selection {residue %s}\n'%residsel
        lappend(myline)
        myline='mol addrep top\n'
        lappend(myline)


    # Cycle through the pairs
    if atompairs is not None:
        atompairs = atompairs.flatten().reshape(-1,2)
        atompairs = _unique_redundant_pairlist(atompairs)
        for jj in atompairs:
            myline='label add Bonds 0/%u 0/%u \n'%(jj[0],jj[1])
            lappend(myline)

    # Add a trajectory
    if trajfile is not None:
        myline='mol addfile %s waitfor all\n'%trajfile
        lappend(myline)

    # Add second structre
    if strfilextra is not None:
        myline='mol new %s waitfor all\n'%strfilextra
        lappend(myline)
        lappend('mol delrep 0 top\n')
        lappend('mol representation %s \n'%rep)
        lappend('mol color %s\n'%molcolor)
        lappend('mol addrep top\n')

    # Add second trajectory
    if trajfilextra is not None:
        myline='mol addfile %s waitfor all\n'%trajfilextra
        lappend(myline)

    # Add second labels
    if atompairsextra is not None:
        for jj in atompairsextra:
            myline='label add Bonds 1/%u 1/%u \n'%(jj[0],jj[1])
            lappend(myline)

    # Add free text at the end if wanted
    if freetext is not None:
        lappend(freetext)

    if vmdout is None:
        return lines
    else:
        f = open(vmdout,'w')
        for line in lines:
            f.write(line)
        f.close()

from re import split
def sort_nicely( l ):
    """ Sort the given list in the way that humans expect.
    https://blog.codinghorror.com/sorting-for-humans-natural-sort-order/
    """
    convert = lambda text: int(text) if text.isdigit() else text
    alphanum_key = lambda key: [ convert(c) for c in split('([0-9]+)', key) ]
    l.sort( key=alphanum_key )
    return l

def fnamez2dict(fnamez, add_geometries=True):
    import mdtraj as _md
    project_dict = {}
    for key, value in _np.load(fnamez).items():
        project_dict[key] = value
    if add_geometries:
        # Load geometry
        for path_type in ['min_rmsd', 'min_disp']:
            project_dict['geom_'+path_type]= _md.load(fnamez.replace('.npz','.%s.pdb'%path_type))
    return project_dict

def runing_avg_idxs(l, n, symmetric=True, debug=False):
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

    assert n*2 <= l, "Can't ask for a running average of 2*%u+1 with only %u frames. " \
                     "Choose a smaller parameter n."%(n, l)

    if not symmetric:
        raise NotImplementedError("Sorry symmetric=False is not implemented yet")

    idxs = _np.arange(l)
    frame_idx= []
    frame_window = []
    for ii in idxs[n - 1 + 1 : l - n + 1]:
        frame_idx.append(ii)
        frame_window.append(idxs[ii - n:ii + n + 1])
        if debug:
            print(frame_idx[-1],frame_window[-1])

    return _np.hstack(frame_idx), frame_window

def smooth_geom(geom, n, geom_data=None, symmetric=True,):
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

    symmetric : bool, default is True
        An average is made so that the geometry at frame i is the average of frames
        :math: [i-n, i-n-1,.., i-1, i, i+1, .., i+n-1, i+n]

    Returns
    -------

    geomout : :any:`mdtraj.Trajectory` object
        A geometry with 2*n less frames containing the smoothed out positions of the atoms.

    """

    # Input checking here, otherwise we're seriously in trouble



    # Get the indices necessary for the running average
    frame_idxs, frame_windows = runing_avg_idxs(geom.n_frames, n, symmetric=symmetric)

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
        xyz[ii,:,:] = geom[frame_windows[ii]].superpose(geom, frame=frame_windows[ii][n]).xyz.mean(0)
        if geom_data is not None:
            data_out[ii,:] = geom_data[frame_windows[ii]].mean(0)

    geom_out = _md.Trajectory(xyz, geom.top)

    if geom_data is None:
        return geom_out.superpose(geom_out)
    else:
        return geom_out, data_out
