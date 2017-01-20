import argparse
from FAHutils import sort_nicely
from os import path as ospath
from pyemma.coordinates import save_traj, source
import numpy as np
from glob import glob
import matplotlib.pylab as plt
from projX.bmutils import *
#%matplotlib qt4

parser = argparse.ArgumentParser(description="Provided the project name(s) of previously simon-featurized project(s), do stuff",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('--proj', type=str, help="Give your project a name", default='proj')
parser.add_argument('-proj_top')
parser.add_argument('--md_ext', type=str, help='extensions of the md trajectory files', default='dcd')
parser.add_argument('--projection_dir', type=str, default='.',
                    help='directory where the simon-tica-files lie')

parser.add_argument('--projection_startswith', type=str, default='tica_',
                    help='How the simon-projection filenames start')

parser.add_argument('--target_projections', type=str, nargs='+', default='all',
                    help='If the user already knows which subset of the projections need to be visualized, '
                         'he can select them here. Typically, if a given project has been projected to '
                         'TICs with different lagtimes, or same lagtime with different input features, '
                         'this is the option to select them. Otherwise, the entire project will be analysed.')

parser.add_argument('--mdtraj_dir', type=str, default = '.',
                    help='directory where the original md-trajectories lie')
parser.add_argument('--out_dir', type=str, default=None, help='Directory to write results to. The default is '
                                                              'to write them in the projection_dir')

parser.add_argument('--proj_stride', type=int, default=10,
                    help='Strided value of the projected trajectory')
parser.add_argument('--n_projs', type=int, default = 1,
                    help='Number of projected coordinates (typically TICs) to be expressed as geometry snapshots')
parser.add_argument('--proj_dim', type=int, default=2, help='Distances between candidate snapshots will be computed in'
                                                            ' a proj_dim dimensional space.')
parser.add_argument('--proj_idxs', type=int, nargs='+', default=None,
                    help='Selection of projection idxs (zero-idxd) for which the samples will be generated. The default'
                         'behaviour is that proj_idxs = range(n_projs), if proj_idxs != None, then '
                         'n_projs is ignored and'
                         'proj_dim is set automatically ')

parser.add_argument('--n_points', type=int, default=100,
                    help='Number of points in which each projected coordinate will be discretized')
parser.add_argument('--n_try_max_cl', type=int, default=1, help='Try regspatial clustring this number of times'
                                                                'in trying to get the target n_points')

parser.add_argument('--n_geom_samples', type=int, default = 50,
                    help='Appart from a "minimally diffusing path" along each projection, a second path '
                         'will be computed also by sampling geometries directly. For every one of the '
                         'n_points, n_geom_samples geometries will also be sampled, providing sets to choose '
                         'the path from so that the rmsd along the path is minimized.')
parser.add_argument('--history_aware', type=bool, default = True,
                    help="In choosing the next point along the path, whether to minimize (diffusion or minRMSD) "
                         "wrt to the mean of the preceding points or just to the last point.")
parser.add_argument('--pdbs', type=bool, default=True,
                    help="Generate .pdb files matching the plots.")

parser.add_argument('--vmds', type=bool, default=True,
                    help="Generate .vmd files matching the plots.")

parser.add_argument('--type_of_visual_fit', type=str, default='sequential',
                    help='If a .pdb file is produced, the type of orientation that the structures will have.\n'
                         '"sequential" will maximize overlap of each frame with its neihgboring frame (forwards and backwards).\n'
                         '"min_to_start_idx" will maximize overlap with the frame that was chosen as reference'
                    )

args = parser.parse_args()

# Some default-interpreting...
if args.out_dir is None:
    args.out_dir = args.projection_dir

if args.proj_idxs is None:
    args.proj_idxs = np.arange(args.n_projs)
else:
    args.proj_dim = np.max((args.proj_dim, np.max(args.proj_idxs)+1))

# Quickndirty way of describing what needs saving
vars2dict = ['ticfile',
             'path',        'compact_path',         'Y_path',
             'path_smpl',   'compact_path_sample',  'Y_path_smpl',
             'CA_pairs',    'max_corr',
             'start_idx', 'start_frame']

# Create a src object for this project (this is lagtime independent)
# VERY STRONG ASSUMPTION, this is one single trajectory (may be fragmented, but must be only one!)
xtcs = [sorted(glob(args.mdtraj_dir+'*'+args.md_ext))]

src = source(xtcs, top=args.proj_top)

topology = src.data_producer._readers[0][0].featurizer.topology

# Create a base name for everything related to this project

fname_base =  ospath.join(args.out_dir,'%s.%s'%(args.proj, args.projection_startswith))


# Prune to target featurizations
avail_tica_trajfiles= sort_nicely(glob(ospath.join(args.projection_dir,args.projection_startswith)+'*.npz'))
avail_tica_trajfiles = targets_in_candidates(avail_tica_trajfiles,
                                             args.target_projections )

for ticfile in avail_tica_trajfiles:

    lag_str, idata, tcorr, tica_mean, l, U = opentica_npz(ticfile)
    # Prune to the first args.proj_dim
    idata = [jidata[:,:args.proj_dim] for jidata in idata]

    for coord in args.proj_idxs:
        print('lag %s, coord %u'%(lag_str, coord), flush=True)
        # Cluster in regspace along the dimension you want to advance, to approximately n_points
        cl = cluster_to_target([jdata[:,coord] for jdata in idata],
                               args.n_points, n_try_max=args.n_try_max_cl,
                                 #verbose=True
                                 )

        # Create full catalogues (discrete and continuous) in ascending order of the coordinate of interest
        cat_idxs, cat_cont = catalogues(cl, data=idata,
                                        sort_by=0, #here we always have to sort by the 1st coord
                                        )

        # Create sampled catalogues in ascending order of the cordinate of interest
        sorts_coord = np.argsort(cl.clustercenters[:,0]) # again, here we're ALWAYS USING "1" because cl. is 1D
        cat_smpl = cl.sample_indexes_by_cluster(sorts_coord, args.n_geom_samples)
        geom_smpl = save_traj(src, np.vstack(cat_smpl), None, topology, stride=args.proj_stride)
        geom_smpl = re_warp(geom_smpl, [args.n_geom_samples]*cl.n_clusters)

        path_sample = {}
        path_rmsd = {}
        start_idx = {}
        start_frame = {}
        # Different path-samples using different initiailzations
        for strategy in [
            'smallest_Rgyr',
            'most_pop',
            'most_pop_x_smallest_Rgyr',
            'bimodal_compact'
            ]:
            # Choose the starting point for the fwd and bwd paths, see the options of the method for more info
            istart_idx = get_good_starting_point(cl, geom_smpl, cl_order=sorts_coord,
                                                strategy=strategy,
                                                )
            # Of all the sampled geometries that share this starting point of "coord"
            # pick the one that's closest to zero
            istart_Y = np.vstack([idata[ii][jj] for ii,jj in cat_smpl[istart_idx]])
            istart_Y[:,coord] = 0
            istart_frame = np.sum(istart_Y**2,1).argmin()

            selection = geom_smpl[0].top.select('backbone')
            # Create a path minimising minRMSD between frames
            path_smpl, __ = visual_path(cat_smpl, geom_smpl,
                                        start_frame=istart_frame,
                                        path_type='min_rmsd',
                                        start_pos=istart_idx,
                                        history_aware=args.history_aware,
                                        selection=selection)

            y = np.vstack([idata[ii][jj] for ii,jj in path_smpl])
            y[:,coord] = 0
            path_diffusion = np.sqrt(np.sum(np.diff(y.T).T**2,1).mean())
            print('Strategy %s starting at %g diffuses %g in %uD proj-space'%(strategy,
                                                                              cl.clustercenters[sorts_coord][istart_idx],
                                                                              path_diffusion,
                                                                              args.proj_dim),
                  flush=True)

            path_sample[strategy] = path_smpl
            path_rmsd[strategy] = path_diffusion
            start_idx[strategy] = istart_idx
            start_frame[strategy] = istart_frame

        # Stick the rmsd-path that diffuses the least
        min_key = sorted(path_rmsd.keys())[np.argmin([path_rmsd[key] for key in sorted(path_rmsd.keys())])]
        print("Sticking with %s"%min_key)
        path_smpl = path_sample[min_key]
        istart_idx = start_idx[min_key]
        istart_frame = start_frame[min_key]

        Y_path_smpl = np.vstack([idata[ii][jj] for ii,jj in path_smpl])
        trajs_path_smpl = save_traj(src.filenames, path_smpl, None,
                                    stride=args.proj_stride, top=topology
                                    )

        # With the starting point the creates the minimally diffusive path,
        # create a path minimising displacemnt in the projected space (minimally diffusing path)
        istart_Y = cat_cont[istart_idx]
        istart_Y[:,coord] = 0
        istart_frame = np.sum(istart_Y**2,1).argmin()
        path, __ = visual_path(cat_idxs, cat_cont,
                               start_pos=istart_idx,
                               start_frame=istart_frame,
                               history_aware=args.history_aware,
                               exclude_coords=[coord])
        trajs_path = save_traj(src.filenames, path, None, stride=args.proj_stride, top=topology)
        Y_path = np.vstack([idata[ii][jj] for ii,jj in path])

        # Plot all and everything
        (compact_path, compact_path_sample), contourplot_dict = plot_histo_reaction_coord(
            np.vstack(idata),
            Y_path, coord,
            extra_paths=[Y_path_smpl],
            log=True,
            contour_alpha=.25,
            path_labels=['path TIC', 'path_mrmsd'],
            markers=' ', start_idx=istart_idx, alpha=.25, linestyles='-')

        fname =  fname_base+'%s.TIC_%u_path.min_step'%(lag_str, coord)
        if args.history_aware:
            fname = fname.replace('min_step', 'history_aware')
        plt.savefig(fname+'.png')
        plt.close()

        # Feature correlations (see options of correlations2CA_pairs for more info)
        CA_pairs, max_corr = correlations2CA_pairs(tcorr[:,coord], geom_smpl)

        # Save the paths in form of .pdbs and .vmds
        if args.pdbs:
            # Do a fit just to have them in approximately the same orientation
            trajs_path_smpl.superpose(trajs_path[istart_idx])
            trajs_path.superpose(trajs_path[istart_idx])
            if args.type_of_visual_fit == 'sequential':
                trajs_path = sequential_rmsd_fit(trajs_path, istart_idx)
                trajs_path_smpl = sequential_rmsd_fit(trajs_path_smpl, istart_idx)
            elif args.type_of_visual_fit == 'min_to_start_idx':
                ref = trajs_path_smpl[istart_idx]
                trajs_path.superpose     (ref, atom_indices=selection)
                trajs_path_smpl.superpose(ref, atom_indices=selection)
            else:
                raise NotImplementedError("What type of fit is %?"%args.type_of_visual_fit)
            trajs_path.save(fname+'.min_disp.pdb')
            trajs_path_smpl.save(fname+'.min_rmsd.pdb')

            if args.vmds:
                from myMDvisuals import customvmd, change_color_order_in_VMD
                # Some vmd playing-around...
                excluded_path =  np.argwhere(np.in1d(np.arange(cl.n_clusters), compact_path,        invert=True)).squeeze()
                if excluded_path.ndim != 0:
                    excluded_path = excluded_path[::-1]
                else:
                    excluded_path = [excluded_path]
                excluded_path_sample = np.argwhere(np.in1d(np.arange(cl.n_clusters), compact_path_sample, invert=True))
                if excluded_path_sample.ndim != 0:
                    excluded_path_sample = excluded_path_sample.squeeze()[::-1]
                else:
                    excluded_path_sample = [excluded_path_sample]
                extra_text  = 'mol top 0\n'+''.join(['animate delete beg %u end %u\n'%(ee,ee) for ee in excluded_path])
                extra_text += 'start_sscache\n'
                extra_text += 'mol top 1\n'+''.join(['animate delete beg %u end %u\n'%(ee,ee) for ee in excluded_path_sample])
                extra_text += 'start_sscache\n'
                extra_text += ''.join(change_color_order_in_VMD(['blue', 'green']))
                vmdname = fname+'.vmd'
                customvmd(fname+'.min_disp.pdb', vmdout=vmdname, rep='NewCartoon', molcolor='Molecule',
                          freetext=extra_text,
                          atompairs=CA_pairs,
                          strfilextra=fname+'.min_rmsd.pdb', atompairsextra=CA_pairs
                          )
                #vars2dict.append(vmdname)

        # Save all data except the geometries
        savedicc = dictionarize_list(vars2dict, globals(), output_dict=contourplot_dict)
        np.savez(fname+'.npz', **savedicc)