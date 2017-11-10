from __future__ import print_function

import warnings as _warnings
import os as _os
import sys as _sys
from glob import glob
from inspect import getfile
import shutil
import subprocess

from tempfile import mkdtemp

def _molpxdir(join=None):
    r"""
    return the directory where molpx is installed

    :param join: str, default is None
        _datadir(join='myfile.dat') will return os.path.join(_datadir(),'myfile.dat')

    :return: directory or filename where the data for the notebook lies
    """

    import molpx

    if join is None:
        return _os.path.dirname(getfile(molpx))
    else:
        assert isinstance(join,str), ("parameter join can only be a string", type(join))
        return _os.path.join(_os.path.dirname(getfile(molpx)), join)



# For python 2.7 compatibility if we don't want to depend also on backports
# http://stackoverflow.com/questions/19296146/tempfile-temporarydirectory-context-manager-in-python-2-7
class TemporaryDirectory(object):
    """Create and return a temporary directory.  This has the same
    behavior as mkdtemp but can be used as a context manager.  For
    example:

        with TemporaryDirectory() as tmpdir:
            ...

    Upon exiting the context, the directory and everything contained
    in it are removed.
    """

    def __init__(self, suffix="", prefix="tmp", dir=None):
        self._closed = False
        self.name = None # Handle mkdtemp raising an exception
        self.name = mkdtemp(suffix, prefix, dir)

    def __repr__(self):
        return "<{} {!r}>".format(self.__class__.__name__, self.name)

    def __enter__(self):
        return self.name

    def cleanup(self, _warn=False):
        if self.name and not self._closed:
            try:
                self._rmtree(self.name)
            except (TypeError, AttributeError) as ex:
                # Issue #10188: Emit a warning on stderr
                # if the directory could not be cleaned
                # up due to missing globals
                if "None" not in str(ex):
                    raise
                print("ERROR: {!r} while cleaning up {!r}".format(ex, self,),
                      file=_sys.stderr)
                return
            self._closed = True
            if _warn:
                self._warn("Implicitly cleaning up {!r}".format(self),
                           ResourceWarning)

    def __exit__(self, exc, value, tb):
        self.cleanup()

    def __del__(self):
        # Issue a ResourceWarning if implicit cleanup needed
        self.cleanup(_warn=True)

    # XXX (ncoghlan): The following code attempts to make
    # this class tolerant of the module nulling out process
    # that happens during CPython interpreter shutdown
    # Alas, it doesn't actually manage it. See issue #10188
    _listdir = staticmethod(_os.listdir)
    _path_join = staticmethod(_os.path.join)
    _isdir = staticmethod(_os.path.isdir)
    _islink = staticmethod(_os.path.islink)
    _remove = staticmethod(_os.remove)
    _rmdir = staticmethod(_os.rmdir)
    _warn = _warnings.warn

    def _rmtree(self, path):
        # Essentially a stripped down version of shutil.rmtree.  We can't
        # use globals because they may be None'ed out at shutdown.
        for name in self._listdir(path):
            fullname = self._path_join(path, name)
            try:
                isdir = self._isdir(fullname) and not self._islink(fullname)
            except OSError:
                isdir = False
            if isdir:
                self._rmtree(fullname)
            else:
                try:
                    self._remove(fullname)
                except OSError:
                    pass
        try:
            self._rmdir(path)
        except OSError:
            pass

def example_notebooks(dry_run=False, extra_flags_as_one_string=None, **kwargs_subprocess):
    r"""
    Open the list of available example notebooks in the default browser.
    The ipython terminal stays active while ipython kernel still active.
    Ctr+C in the ipython terminal will close the ipython kernel

    Note: The displayed notebooks are a working copy of the original notebooks. Feel free to mess around with them

    Parameters
    ----------

    dry_run : bool, default is False
        Show a list of available notebooks and exit.

    extra_flags_as_one_string : str
        Any flags you would parse along to the "jupyter notebook" command, like --no-browser etc

    **kwargs_subprocess: named arguments for the subprocess call.
        You can ignore this safely, this makes testing possible
    """
    avail_nbs = glob(_molpxdir(join='notebooks/*.ipynb'))
    if dry_run:
        print("List of available notebooks found in molpx's notebook directory %s"%_molpxdir(join='notebooks/'))
        print("You can use any of them as 'nb_file' to open them in a safe environment")
        for ff in avail_nbs:
            print('* %s'%_os.path.basename(ff))
        return

    with TemporaryDirectory(suffix='_test_molpx_notebook') as tmpdir:
        for nb_file in avail_nbs:
            tmpfile = _os.path.join(tmpdir, _os.path.basename(nb_file))
            shutil.copy(nb_file, tmpfile)

            nbstring = open(tmpfile).read()
            f = open(tmpfile,'w')
            f.write(nbstring.replace("# ", "<font color='red', size=1>"
                                           "This is a temporary copy of the original notebook found in `%s`. "
                                           "This temporary copy is located in `%s`. "
                                           "Feel free to play around, modify or even break this notebook. "
                                           "It wil be deleted on exit it and a new one created next time you issue "
                                           "`molpx.example_notebooks()`</font>\\n\\n"
                                           "# "
                                     %(nb_file, tmpfile),1))
            f.close()

        cmd = 'jupyter notebook --notebook-dir %s'%tmpdir
        if isinstance(extra_flags_as_one_string,str):
            cmd ='%s %s'%(cmd,extra_flags_as_one_string)

        try:
            subprocess.run(cmd.split(), **kwargs_subprocess)
        except AssertionError:
            _warnings.warn("molpx.example_notebooks could not open an in interactive shell. "
                           "Nothing happened.")

