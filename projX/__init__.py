from __future__ import print_function

__author__ = 'gph82'


from . import generate
from . import visualize

def _report_status():
    r"""
    returns a boolen whether projX is allowed to send user metadata
    """
    import os,sys
    conf_module = os.path.expanduser('~/.projX')
    sys.path.append(conf_module)
    res = True
    try:
        import conf_projX
        try:
            res =  conf_projX.report_status
            if not isinstance(res, bool):
                pass
                #TODO write warning about malformed conf_projX.py file
        except AttributeError:
            pass
            #TODO write warning about corrupted conf_projX.py file
    except ImportError:
         #TODO write warning about corrupted conf_projX.py file
        pass
    except NameError:
        #TODO write warning about corrupted conf_projX.py file
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
                        headers={'User-Agent': 'projX-{projX_version}-Py-{python_version}-{platform}-{addr}'
                        .format(projX_version=current, python_version=platform.python_version(),
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


# http://stackoverflow.com/questions/17583443/what-is-the-correct-way-to-share-package-version-with-setup-py-and-the-package
from pkg_resources import get_distribution, DistributionNotFound
import os.path
try:
    _dist = get_distribution('projX')
    # Normalize case for Windows systems
    dist_loc = os.path.normcase(_dist.location)
    here = os.path.normcase(__file__)
    if not here.startswith(os.path.join(dist_loc, 'projX')):
        # not installed, but there is another version that *is*
        raise DistributionNotFound
except DistributionNotFound:
    __version__ = 'Please install this project with setup.py'
else:
    __version__ = _dist.version


# start check in background
_version_check(__version__).start()



