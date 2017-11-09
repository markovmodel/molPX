r"""
=====================================
molPX - Molecular Projection Explorer
=====================================
"""
from __future__ import print_function as _

__author__ = 'gph82'

# To be able to run with a development version of nglview
# TODO PIN TO NGLVIEW 1.0 once it's released
pre_release = '1.0'
from distutils.version import LooseVersion
try:
    import nglview
    if LooseVersion(nglview.__version__) < LooseVersion(pre_release):
        raise ImportError
except ImportError:
    import os
    os.system('pip install nglview==%s'%pre_release)


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
