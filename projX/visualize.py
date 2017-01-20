from __future__ import print_function

__author__ = 'gph82'


from pyemma.plots import plot_free_energy
import numpy as _np
from .bmutils import link_ax_w_pos_2_nglwidget as _link_ax_w_pos_2_nglwidget, \
    data_from_input as _data_from_input, \
    extract_visual_fnamez as _extract_visual_fnamez
from . import generate

from matplotlib import pylab as _plt
import nglview as _nglview
import mdtraj as _md
from os.path import basename as _basename


def FES(MD_trajfile, MD_top, projected_trajectory,
        nbins=100,
        xlabel='proj_0', ylabel='proj_1'):
    r"""
    TODO: document everything, parse options to generate_sample. Right now everything is
    taking its default values (which work well)

    returns: ax, iwd, data_sample, geoms
        ax : matplotlib axis object
        iwd : nglwidget
        data_sample: the position of the red dots in the plot
        geoms: the geometries incorporated into the nglwidget
    """
    data_sample, geoms, data = generate.sample(MD_trajfile, MD_top,projected_trajectory,
                                        return_data=True
                                         )
    data = _np.vstack(data)
    #h, (x, y) = _np.histogramdd(data, bins=nbins)

    #irange = _np.hstack((x[[0,-1]], y[[0,-1]]))
    _plt.figure()
    plot_free_energy(data[:,0], data[:,1], nbins=nbins)
    #_plt.contourf(-_np.log(h).T, extent=irange)
    ax = _plt.gca()
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    iwd = sample(data_sample, geoms.superpose(geoms[0]), ax)

    return _plt.gca(), _plt.gcf(), iwd, data_sample, geoms

def traj(trajectory,
         MD_top, projected_trajectory, max_frames=1000,
         stride=1, proj_idxs=[0,1], dt=1, plot_FES=False):
    r"""
    TODO: document everything, parse options to generate_sample. Right now everything is
    taking its default values (which work well)

    returns: ax, iwd, data_sample, geoms
        ax : matplotlib axis object
        iwd : nglwidget
        data_sample: the position of the red dots in the plot
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
        data_sample =_np.vstack((time, idata)).T
        iax.plot(time, idata)
        widget = sample(data_sample, geoms.superpose(geoms[0]), iax, clear_lines=False, widget=widget)

    if plot_FES:
        h, (x, y) = _np.histogramdd(data, bins=50)
        irange = _np.hstack((x[[0,-1]], y[[0,-1]]))
        _plt.figure()
        _plt.contourf(-_np.log(h).T, extent=irange)
        ax = _plt.gca()
        widget = sample(data, geoms.superpose(geoms[0]), ax, widget=widget)

    return _plt.gca(), _plt.gcf(), widget, geoms


def sample(data_sample, geom,  ax,
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
        ax.plot(data_sample[:,0], data_sample[:,1], '-g', lw=3)

    # Link the axes widget with the ngl widget
    _link_ax_w_pos_2_nglwidget(ax,
                               data_sample,
                               iwd,
                               **link_ax2wdg_kwargs
                               )
    return iwd

def fnamez(fname,
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

def project_dict(project_dict,
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
