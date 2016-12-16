import numpy as _np
import mdtraj as _md
from matplotlib import pyplot as _plt
from matplotlib.widgets import AxesWidget as _AxesWidget
from glob import glob
import os
import tempfile
#from sklearn.mixture import GMM as _GMM
from sklearn.mixture import GaussianMixture as _GMM

from pyemma.util.linalg import eig_corr
from pyemma.coordinates import source as _source, cluster_regspace as _cluster_regspace
from pyemma.util.discrete_trajectories import index_states as _index_states
from scipy.spatial import cKDTree as _cKDTree
from myMDvisuals import customvmd

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

def cluster_to_target(data, n_clusters_target, n_try_max=5,
                      verbose=False):
    r"""
    Naive heuristic to try to get to the right n_clusters using 1D regspace cl in n_try_max tries"
    """

    # Works well for connected, 1D-clustering,
    # otherwise bad starting guess for dmin
    cmax = _np.hstack(data).max()
    cmin = _np.hstack(data).min()
    dmin = (cmax-cmin)/(n_clusters_target+1)

    err = _np.ceil(n_clusters_target*.05)
    cl = _cluster_regspace(data, dmin=dmin)
    for cc in range(n_try_max):
        n_cl_now = cl.n_clusters
        delta_cl_now = _np.abs(n_cl_now - n_clusters_target)
        if not n_clusters_target-err <= cl.n_clusters <= n_clusters_target+err:
            # Cheap (and sometimes bad) heuristic to get relatively close relatively quick
            dmin = cl.dmin*cl.n_clusters/n_clusters_target
            cl = _cluster_regspace(data, dmin=dmin)
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

def min_disp_path(start, path_of_candidates, exclude_coords=None, history_aware=False):
    r""""Returns a list of indices [i,j,k...] s.t. the distances 
     d_i = |path_of_candidates[0][i] - path_of_candidates[1][j]|, 
     d_j = |path_of_candidates[1][j] - path_of_candidates[2][k]|,...
     are minimized

    Parameters:
    start : 2D np.array of shape (1,n) 
       starting point, for the path. path_of_candidates[ii].shape[1] == start.shape[1] == n
         
    path_of_candidates: iterable of 2D np.arrays each of shape (m, n)
       The list of candidates to pick from to minimize the distances

    exclude_coords: None or iterable of ints, default is None
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
       
    """
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
    r""" Returns a catalogue of frames from a pyemma.coor.cl object
    Parameters:
    ----------
    cl: pyemma clustering object
    
    data : None or list, default is None
       The default behaviour is to return the catalogue of continuous values (cat_cont, see below)
       using the cl-object's own cl.dataproducer.data, but other any data can be given
       here, as long as it satisfies:
           len(data) == len(cl.dataproducer.data)
           [len(idata) == len(jdata) for idata, jdata in zip(data, cl.dataproducer.data)]
       
       (Note that by construction the same relations should hold for cl.dtrajs)
 
    sort_by : None or int, default is None
       Default behaviour is to return the catalogues in the same order of clustercenters as the input, 
       but it is sometimes useful have them sorted by ascending order of the n-th coordinate 
       of the input space, where sort_by = n
       
    Returns:
    --------
    cat_idxs : list of 2D np.arrays
        The discrete catalogue. It is a list of len = cl.n_clustercenters containing a 2D vector
        with all the (file, frame)-pairs in which this clustercenter appears

    cat_cont : list of ndarrays
        The actual value (asumed continuous) of the data at the (file-frame)-pairs of the cat_idxslist
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
                 it creates two paths, one moving forward and one moving backward.  
                 These are the n and backward ('left') create a coordinate-increasing, diffusion-minimizing path from 
       'left':   starts at the "left end" of the coordinate, i.e. at its minimum value, and moves forward
       'right'   starts at the "right end" of the coordinate, i.e. at its maximum value, and moves backward
     
    path_type = 'min_disp' or 'minRMSD'

    start_frame = if the user already knows, of the start_pos index, the frame that's best

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
    cl_order: iterable of integers
        It can be

    """

    Y = _np.sort(cl.clustercenters.squeeze())
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
    elif strategy == 'bimodal_compact':
        #  assume bimodality in the coordinate of interest (usually the case at least for TIC_0)
        (left_idx, right_idx), __ = find_centers_gmm(_np.vstack(cl.data_producer.data).reshape(-1,1),
                                                     Y, n_components=2
                                                     )
        #  bias towards starting points with compact structures (low small radius of gyration)
        left_value, right_value = _md.compute_rg(geom_samples[left_idx]).mean(), \
                                  _md.compute_rg(geom_samples[right_idx]).mean()
        start_idx = [left_idx, right_idx][_np.argmin([left_value, right_value])]
    else:
        raise NotImplementedError("This starting point strategy is unkown %s"%strategy)

    return start_idx


def find_centers_gmm(data, gridpoints, n_components=2):
    r""" Provided 1D data and a grid of points, return the indices of the points closest to
    the centers of an assumed n-modal distribution behind "data"
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

    if isinstance(projected_data, str) or isinstance(projected_data, _np.ndarray):
        projected_data = [projected_data]
    elif not isinstance(projected_data, list):
        raise ValueError("Data type not understood %"%type(projected_data))

    if isinstance(projected_data[0],str):
        idata = [_np.load(f) for f in projected_data]
    else:
        idata = projected_data

    return idata

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
                      ext='dcd'):
    xtcs = []
    ii = 0
    struct = None
    for __, idir in enumerate(sorted(glob(os.path.join(mdtraj_dir, dirstartswith+proj+'*')))):
        if not idir.endswith('tar.gz'):
            subdir = os.path.join(idir,strfile_fmt%(proj, ii))
            these_trajs = sorted(glob(os.path.join(subdir,'*'+ext)))
            xtcs.append(these_trajs)
            if struct is None:
                struct = os.path.join(subdir,'%s-%u-protein.pdb'%(proj,ii))
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

    return compact_paths, dictionarize_list(['h', 'x', 'y', 'v_crd_1', 'v_crd_2'], locals())

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

def link_ax_w_pos_2_nglwidget(ax, pos, nglwidget):
    r"""
    Initial idea for this function comes from @arose, the rest is @gph82
    """

    kdtree = _cKDTree(pos)
    assert nglwidget.trajectory_0.n_frames == pos.shape[0], \
        ("Mismatching frame numbers %u vs %u"%( nglwidget.trajectory_0.n_frames, pos.shape[0]))
    x, y = pos.T

    lineh = ax.axhline(ax.get_ybound()[0], c="black", ls='--')
    linev = ax.axvline(ax.get_xbound()[0], c="black", ls='--')
    dot, = ax.plot(pos[0,0],pos[0,1], 'o', c='red', ms=7)

    nglwidget.isClick = False

    def onclick(event):
        linev.set_xdata((event.xdata, event.xdata))
        lineh.set_ydata((event.ydata, event.ydata))
        data = [event.xdata, event.ydata]
        _, index = kdtree.query(x=data, k=1)
        dot.set_xdata((x[index]))
        dot.set_ydata((y[index]))
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
            dot.set_xdata((x[_idx]))
            dot.set_ydata((y[_idx]))
            #print("caught index error with index %s (new=%s, old=%s)" % (_idx, change["new"], change["old"]))
        except IndexError as e:
            dot.set_xdata((x[0]))
            dot.set_ydata((y[0]))
            print("caught index error with index %s (new=%s, old=%s)" % (_idx, change["new"], change["old"]))
            pass
        #print("set xy = (%s, %s)" % (x[_idx], y[_idx]))

    # Connect axes to widget
    axes_widget = _AxesWidget(ax)
    axes_widget.connect_event('button_release_event', onclick)

    # Connect widget to axes
    nglwidget.observe(my_observer, "frame", "change")

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
        vmdpipe.write(" animate goto %u\nlist\n\n"%index)
        myflush(vmdpipe,
                #size=1e4
                )

    # Connect axes to widget
    axes_widget = _AxesWidget(ax)
    axes_widget.connect_event('button_release_event', onclick)

    return vmdpipe