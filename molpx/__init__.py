r"""
=====================================
molPX - Molecular Projection Explorer
=====================================
"""
from __future__ import print_function as _

__author__ = 'gph82'

from . import generate
from . import visualize
from . import _bmutils

from ._nbtools import example_notebooks, _molpxdir
def _report_status():
    r"""
    returns a boolen whether molpx is allowed to send user metadata
    """
    import os,sys
    conf_module = os.path.expanduser('~/.molpx')
    sys.path.append(conf_module)
    res = True
    try:
        import conf_molpx
        try:
            res =  conf_molpx.report_status
            if not isinstance(res, bool):
                pass
                #TODO write warning about malformed conf_molpx.py file
        except AttributeError:
            pass
            #TODO write warning about corrupted conf_molpx.py file
    except ImportError:
         #TODO write warning about corrupted conf_molpx.py file
        pass
    except NameError:
        #TODO write warning about corrupted conf_molpx.py file
        pass

    sys.path = sys.path[:-1]

    return res

def _version_check(current, testing=False):
    """ checks latest version online from the server and logs some user metadata.
    Can be disabled by setting config.check_version = False.

    """

    import platform
    import os
    from six.moves.urllib.request import urlopen, Request
    from contextlib import closing
    import threading
    import uuid

    import sys
    if 'pytest' in sys.modules or os.getenv('CI', False):
        testing = True

    def _impl():
        try:
            r = Request('http://emma-project.org/versions.json',
                        headers={'User-Agent': 'molpx-{molpx_version}-Py-{python_version}-{platform}-{addr}'
                        .format(molpx_version=current, python_version=platform.python_version(),
                                platform=platform.platform(terse=True), addr=uuid.getnode())} if not testing else {})
            with closing(urlopen(r, timeout=30)):
                pass
            """
            versions = json.loads(payload)
            latest_json = tuple(filter(lambda x: x['latest'], versions))[0]['version']
            latest = parse(latest_json)
            if parse(current) < latest:
                import warnings
                warnings.warn("You are not using the latest release of PyEMMA."
                              " Latest is {latest}, you have {current}."
                              .format(latest=latest, current=current), category=UserWarning)
            """
            # TODO add warnings about versions
        except Exception:
            pass
            """
            import logging
            logging.getLogger('pyemma').debug("error during version check", exc_info=True)
            """
            # TODO add loggers

    return threading.Thread(target=_impl if _report_status() and not testing else lambda: '')

from ._version import get_versions
__version__ = get_versions()['version']
del get_versions

# start check in background
_version_check(__version__).start()

#
import subprocess
import warnings as _warnings


_ext_mapping = {"nglview-js-widgets": "nglview",
                "jupyter-matplotlib":"ipympl",
                "jupyter-js-widgets":"widgetsnbextension"}

def _get_extension_status(ext_list=_ext_mapping.keys()):
    r"""
    Guess the status of the extensions in ext_list. the correct way of doing this
    would be using notebook.nbextensions, but you need paths to the extensions
    and I dont want to go there
    :return: dictionary with extensions in ext_list as keys and True or False as values
    """
    enabled_exts = {key:None for key in ext_list}

    lines = subprocess.check_output(("jupyter-nbextension", "list"), stderr=subprocess.DEVNULL)
    for ext in enabled_exts.keys():
        for iline in lines.decode().splitlines():
            if ext in iline and "disabled" in iline.lower():
                enabled_exts[ext] = False
            elif ext in iline and "enabled" in iline.lower():
                if enabled_exts[ext] is None:
                    enabled_exts[ext] = True
                enabled_exts[ext] = enabled_exts[ext] and True

    for key in enabled_exts.keys():
        if enabled_exts[key] is None:
            enabled_exts[key] = False

    return enabled_exts

def _enable_extensions(this_ext_path):
    r""" If some of the needed extensions are not enabled,
    try to install/enable them or prompt the user to do so"""

    try:
        subprocess.check_call([
            'jupyter', 'nbextension', 'install', '--py',
            '--sys-prefix', this_ext_path
        ], stderr=subprocess.DEVNULL)

        subprocess.check_call([
            'jupyter', 'nbextension', 'enable', '--py',
            '--sys-prefix', this_ext_path
        ], stderr=subprocess.DEVNULL)
    except:
        _warnings.warn("\nWe could not automatically enable the extention %s.\n"
                       "From the command line type:\n jupyter-nbextension enable %s --py --sys-prefix" % (this_ext_path, this_ext_path))
        return False
    return True

# Try to help the user getting molpx working out of the box
if not all(_get_extension_status().values()):

    status = True
    for ext, enabled in _get_extension_status().items():
        if not enabled:
            if not _enable_extensions(_ext_mapping[ext]):
                raise ModuleNotFoundError("Could not initialize molpx")

