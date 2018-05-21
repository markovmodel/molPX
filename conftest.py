import pytest


@pytest.hookimpl(trylast=True)
def pytest_collection_modifyitems(session, config, items):
    """ execute notebook tests last (nbval) """
    from operator import attrgetter
    from _pytest.unittest import TestCaseFunction
    from nbval.plugin import IPyNbCell

    normal_tests = filter(lambda x: isinstance(x, TestCaseFunction), items)
    nbval_tests = sorted(filter(lambda x: isinstance(x, IPyNbCell), items), key=attrgetter('location'))
    items = list(normal_tests) + list(nbval_tests)
