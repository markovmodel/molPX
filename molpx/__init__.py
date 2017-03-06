r"""
=====================================
molPX - Molecular Projection Explorer
=====================================
"""
from __future__ import print_function

__author__ = 'gph82'


from . import generate
from . import visualize

def _molpxdir(join=None):
    r"""
    return the directory molpx is installed

    :param join: str, default is None
        _datadir(join='myfile.dat') will return os.path.join(_datadir(),'myfile.dat')

    :return: directory or filename where the data for the notebook lies
    """

    from os.path import join as pjoin, dirname
    from inspect import getfile
    import molpx

    if join is None:
        return dirname(getfile(molpx))
    else:
        assert isinstance(join,str), ("parameter join can only be a string", type(join))
        return pjoin(dirname(getfile(molpx)), join)

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
    import six
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
                        headers={'User-Agent': 'conf_molpx-{conf_molpx_version}-Py-{python_version}-{platform}-{addr}'
                        .format(conf_molpx=current, python_version=platform.python_version(),
                                platform=platform.platform(terse=True), addr=uuid.getnode())} if not testing else {})
            encoding_args = {} if six.PY2 else {'encoding': 'ascii'}
            with closing(urlopen(r, timeout=30)) as response:
                pass
                #payload = str(response.read(), **encoding_args)
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

    return threading.Thread(target=_impl if _report_status() else lambda: '')
from ._version import get_versions
__version__ = get_versions()['version']
del get_versions


# start check in background
_version_check(__version__).start()

