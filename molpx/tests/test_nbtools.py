__author__ = 'gph82'

import unittest
import os
import numpy as np
import molpx

class TestNbtools(unittest.TestCase):

    def test_example_notebooks(self):
        molpx.example_notebooks(dry_run=False)
        molpx.example_notebooks(dry_run=False, extra_flags_as_one_string="--no-browser")


    def test_example_notebooks_dry(self):
        molpx.example_notebooks(dry_run=True)

class TestMolpxDir(unittest.TestCase):

    def test_just_runs(self):
        molpx._nbtools._molpxdir(None)
        molpx._nbtools._molpxdir("test")

class TestTemporaryDirectory(unittest.TestCase):

    def test_just_runs(self):
        with molpx._nbtools.TemporaryDirectory(suffix='_test_molpx_notebook') as tmpdir:
            assert os.path.exists(tmpdir)

    def test_just_runs_with_copy(self):
        with molpx._nbtools.TemporaryDirectory(suffix='_test_molpx_notebook') as tmpdir:
            ifile = os.path.join(tmpdir, "ndarray.npy")
            np.save(ifile, np.random.randn(100))
            assert os.path.exists(ifile)

    def test_its_not_there_anymore(self):
        with molpx._nbtools.TemporaryDirectory(suffix='_test_molpx_notebook') as tmpdir:
            ifile = os.path.join(tmpdir, "ndarray.npy")
            np.save(ifile, np.random.randn(100))
            assert os.path.exists(ifile)
        assert not os.path.exists(ifile)

if __name__ == '__main__':
    unittest.main()
