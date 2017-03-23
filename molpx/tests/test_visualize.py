__author__ = 'gph82'
# Since the complicated stuff and the potential for undetectable errors is in bmutils, heavy testing should be done there. This is an API test for checking
# input parsing and decision making (if then) inside the visualize API
import unittest
import pyemma
import os
import tempfile
import numpy as np
import shutil
import molpx
from glob import glob
from molpx import visualize
import mdtraj as md
import matplotlib.pyplot as plt
plt.switch_backend('Agg') # allow tests

class TestTrajInputs(unittest.TestCase):

    def setUp(self):
        self.MD_trajectory_files = glob(molpx._molpxdir(join='notebooks/data/c-alpha_centered.stride.1000*xtc'))
        self.MD_topology_file = glob(molpx._molpxdir(join='notebooks/data/*pdb'))[0]
        self.MD_geoms = [md.load(ff, top=self.MD_topology_file) for ff in self.MD_trajectory_files]
        self.MD_top = self.MD_geoms[0].topology
        self.tempdir = tempfile.mkdtemp('test_molpx')
        self.projected_files = [os.path.join(self.tempdir,'Y.%u.npy'%ii) for ii in range(len(self.MD_trajectory_files))]
        self.feat = pyemma.coordinates.featurizer(self.MD_topology_file)
        self.feat.add_all()
        self.source = pyemma.coordinates.source(self.MD_trajectory_files, features=self.feat)
        self.Y = self.source.get_output(dimensions=np.arange(10))
        [np.save(ifile,iY) for ifile, iY in zip(self.projected_files, self.Y)]
        [np.savetxt(ifile.replace('.npy','.dat'),iY) for ifile, iY in zip(self.projected_files, self.Y)]

    def tearDown(self):
        shutil.rmtree(self.tempdir)

    def test_simplest_inputs_memory(self):
        visualize.traj(self.MD_geoms, self.MD_top, self.Y)

    def test_simplest_inputs_disk(self):
        visualize.traj(self.MD_trajectory_files, self.MD_topology_file, self.projected_files)
        visualize.traj(self.MD_trajectory_files, self.MD_topology_file, [ifile.replace('.npy', '.dat') for ifile in self.projected_files])

    def test_fail_without_top(self):
        # This should fail
        try:
            visualize.traj(self.MD_trajectory_files, "unexisting_file", self.Y)
        except (OSError, IOError):
            pass
        # This should pass
        visualize.traj(self.MD_geoms, "unexisting_file", self.Y)

    def test_listify_inputs(self):
        visualize.traj(self.MD_geoms[0], self.MD_top, self.Y[0])

    def test_listify_params(self):
        visualize.traj(self.MD_geoms, self.MD_top, self.Y, proj_idxs=1)
        visualize.traj(self.MD_geoms, self.MD_top, self.Y, active_traj=1)

    def test_plotting_params(self):
        visualize.traj(self.MD_geoms, self.MD_top, self.Y, sharey_traj=True)

    def test_fail_on_FES(self):
        try:
            visualize.traj(self.MD_geoms, self.MD_top, self.Y, sharey_traj=True, plot_FES=True, proj_idxs=[0,1,2])
        except NotImplementedError:
            pass

class TestCorrelationsInput(unittest.TestCase):

    def setUp(self):
        self.MD_trajectory_files = glob(molpx._molpxdir(join='notebooks/data/c-alpha_centered.stride.1000*xtc'))[:1]
        self.MD_topology_file = glob(molpx._molpxdir(join='notebooks/data/*pdb'))[0]
        self.MD_geoms = [md.load(ff, top=self.MD_topology_file) for ff in self.MD_trajectory_files]
        self.MD_top = self.MD_geoms[0].topology
        self.tempdir = tempfile.mkdtemp('test_molpx')
        self.projected_files = [os.path.join(self.tempdir, 'Y.%u.npy' % ii) for ii in
                                range(len(self.MD_trajectory_files))]
        self.feat = pyemma.coordinates.featurizer(self.MD_topology_file)
        self.feat.add_all()
        source = pyemma.coordinates.source(self.MD_trajectory_files, features=self.feat)
        self.tica = pyemma.coordinates.tica(source, lag=1, dim=10)
        self.Y = self.tica.get_output()
        self.F = source.get_output()
        [np.save(ifile, iY) for ifile, iY in zip(self.projected_files, self.Y)]
        [np.savetxt(ifile.replace('.npy', '.dat'), iY) for ifile, iY in zip(self.projected_files, self.Y)]

    def tearDown(self):
        shutil.rmtree(self.tempdir)

    def test_correlations_inputs(self):
        visualize.correlations(self.tica)

    def test_correlations_inputs_verbose(self):
        visualize.correlations(self.tica, verbose=True)

    def test_correlations_inputs_color_list_parsing(self):
        visualize.correlations(self.tica, proj_color_list=['green'])

    def test_correlations_inputs_FAIL_color_list_parsing(self):
        try:
            visualize.correlations(self.tica, proj_color_list='green')
        except TypeError:
            pass
if __name__ == '__main__':
    unittest.main()
