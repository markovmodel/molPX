.. Projection Explorer documentation master file, created by
   sphinx-quickstart on Mon Feb 20 13:34:27 2017.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

projX.visualize
===============

The core functionality is to interactively link two figures inside an ipython notebook, so that action in one of them (e.g.a click of the mouse or a slide of a slidebar) will trigger an event in the other one (e.g. a frame update or point moved) in the other one and vice versa. Usually, these two representations are:

* **molecules**:  a widget showing the molecular structure that a particular value of X1,X2 is associated with and
* **projections**: a matplotlib figure showing the projected coordinate(s) (X1, X2), either as a histogram (or a free energy surface) or a trajectory view (X1(t) vs. t)


You are **strongly encouraged** to check nglview' `documentation <https://github.com/arose/nglview>`_, since its functionalities extend beyond the scope of this package and the molecular visualization universe is rich and complex (unlike this module).

.. automodule:: projX.visualize
   :members:

.. toctree::
   :maxdepth: 4


