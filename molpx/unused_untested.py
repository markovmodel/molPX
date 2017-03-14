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

def _dictionarize_list(list, input_dict, output_dict = None):
    if output_dict is None:
        output_dict = {}

    for item in list:
        output_dict[item] = input_dict[item]
    return output_dict

def _correlations2CA_pairs(icorr,  geom_sample, corr_cutoff_after_max=.95, feat_type='md.contacts'):
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

def _correlations2CA_pairs(icorr,  geom_sample, corr_cutoff_after_max=.95, feat_type='md.contacts'):
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

def _input2output_corr(icov, U):
    r""" Equivalent to feature_TIC_correlation of a pyemma-TICA object
    """
    feature_sigma = _np.sqrt(_np.diag(icov))
    return _np.dot(icov, U) / feature_sigma[:, _np.newaxis]

def _sequential_rmsd_fit(geomin, start_frame=0):
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

def _opentica_npz(ticanpzfile):
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

def _find_centers_bimodal(distro, space, barrier=0):
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

def _elements_of_list1_in_list1(list1, list2):
    tokeep = []
    for el1 in list1:#args.target_featurizations:
        for el2 in list2:
            if el1 in el2:
                tokeep.append(el2)
                break
    return tokeep

def _targets_in_candidates(candidates, targets, verbose=True ):
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

def _plot_histo_reaction_coord(data, reaction_path, reaction_coord_idx, start_idx=None,
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

def _plot_paths(path_list, path_labels=[],
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

def _extract_visual_fnamez(fnamez, path_type, keys=['x','y','h',
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

def _customvmd(structure,
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
def _sort_nicely( l ):
    """ Sort the given list in the way that humans expect.
    https://blog.codinghorror.com/sorting-for-humans-natural-sort-order/
    """
    convert = lambda text: int(text) if text.isdigit() else text
    alphanum_key = lambda key: [ convert(c) for c in split('([0-9]+)', key) ]
    l.sort( key=alphanum_key )
    return l

def _fnamez2dict(fnamez, add_geometries=True):
    import mdtraj as _md
    project_dict = {}
    for key, value in _np.load(fnamez).items():
        project_dict[key] = value
    if add_geometries:
        # Load geometry
        for path_type in ['min_rmsd', 'min_disp']:
            project_dict['geom_'+path_type]= _md.load(fnamez.replace('.npz','.%s.pdb'%path_type))
    return project_dict

def _vmd_stage(background='white',
              axes=False,
              ):

    mystr = 'color Display Background %s\n'%background
    if not axes:
        mystr += 'axes location off\n'

    mystr += '\n'

    return mystr

def _src_in_this_proj(proj, mdtraj_dir,
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

def _myflush(pipe, istr='#', size=1e4):
    pipe.write(''.join([istr+'\n\n' for ii in range(int(size))]))

def _link_ax_w_pos_2_vmd(ax, pos, geoms, **customVMD_kwargs):
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

    def _onclick(event):
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

# Do I really need this? Don't think so
def _fnamez(fname,
           path_type='min_rmsd',
           proj_type='TIC',
           only_selection=True,
           selection_label='within 1 sigma',
           figsize=(10,10)
                   ):

    proj_name = _basename(fname)[:_basename(fname).find(proj_type)].strip('.')

    data, selection, x, y, h, v_crd_1, v_crd_2 = _extract_visual_fnamez(fname, path_type)
    data = data[:,[v_crd_1,v_crd_2]]

    # Load geometry
    geom = _md.load(fname.replace('.npz','.%s.pdb'%path_type))

    # Create contourplot
    _plt.figure(figsize=figsize)
    _plt.contourf(x[:-1],y[:-1], h.T, alpha=.50)
    #_plt.contourf(project_dict["h"].T, alpha=.50)
    # This can be take care of in "visualize sample"
    _plt.plot(data[:, v_crd_1],
              data[:, v_crd_2],
              alpha=.25, label=path_type)

    if only_selection:
        geom = geom[selection]
        data = data[selection]
        _plt.plot(data[:,0],data[:,1],'o', label=path_type+' '+selection_label)

    _plt.legend(loc='best')
    _plt.xlabel('%s %u'%(proj_type, v_crd_1))
    _plt.ylabel('%s %u'%(proj_type, v_crd_2))
    _plt.xlim(x[[0,-1]])
    _plt.ylim(y[[0,-1]])
    _plt.title('%s\n-np.log(counts)'%proj_name)

    iwd = sample(data, geom, _plt.gca(), plot_path=False, clear_lines=False)

    project_dict = {}
    for key, value in _np.load(fname).items():
        project_dict[key] = value

    return iwd, project_dict

# Do I really need this? Don't think so
def _project_dict(project_dict,
                 path_type='min_rmsd',
                 proj_type='TIC',
                 only_compact_path=True,
                 selection_label='within 1 sigma',
                 figsize=(10,10),
                 project_name = None
                   ):



    # From dict to variables
    if path_type == 'min_rmsd':
        path = project_dict["Y_path_smpl"]
        compact_path = project_dict["compact_path_sample"]
    elif path_type == 'min_disp':
        path = project_dict["Y_path"]
        compact_path = project_dict["compact_path"]
    v_crd_1 = project_dict["v_crd_1"]
    v_crd_2 = project_dict["v_crd_2"]
    path = path[:, [v_crd_1, v_crd_2]]
    geom = project_dict["geom_"+path_type]
    x = project_dict["x"]
    y = project_dict["y"]
    h = project_dict["h"]

    # Create contourplot
    _plt.figure(figsize=figsize)
    _plt.contourf(x[:-1],y[:-1], h.T, alpha=.50) #ATTN h is already log(PDF)

    # This can be taken care of in "visualize sample", but i'm doing it here
    _plt.plot(path[:, 0],
              path[:, 1],
              alpha=.25, label=path_type)

    if only_compact_path:
        geom = geom[compact_path]
        path = path[compact_path]
        _plt.plot(path[:,0],path[:,1],'o', label=path_type+' '+selection_label)

    _plt.legend(loc='best')
    _plt.xlabel('%s %u'%(proj_type, v_crd_1))
    _plt.ylabel('%s %u'%(proj_type, v_crd_2))
    _plt.xlim(x[[0,-1]])
    _plt.ylim(y[[0,-1]])
    if project_name is not None:
        _plt.title('-np.log(counts)\n%s'%project_name)

    iwd = sample(path, geom, _plt.gca(), plot_path=False, clear_lines=False)

    return iwd
