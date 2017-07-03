.. Projection Explorer documentation master file, created by
   sphinx-quickstart on Mon Feb 20 13:34:27 2017.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

molpx.visualize
===============

The core functionality is to link two interative figures, *fig1* and *fig2*, inside an IPython/Jupyter notebook,
so that an action in *fig1* (e.g.a click of the mouse or a slide of a slidebar) will trigger an event in *fig2*
(e.g. a frame update or point moved) and vice versa. Usually, these two figures contain representations from:

* **molecules**:  an `nglviewer <https://github.com/arose/nglview>`_ widget showing one (or more) molecular structure(s) that a particular value of the coordinate(s) is associated with and
* **projected coordinates**: a matplotlib figure showing the projected coordinates (e.g. TICs or PCs or any other), :math:`{Y_0, ..., Y_N}`, either as a 2D histogram, :math:`PDF(Y_i, Y_j)` or as trajectory views :math:`{Y_0(t), ...Y_N(t)}`

You are **strongly encouraged** to check nglview's `documentation <https://github.com/arose/nglview>`_, since its functionalities extend beyond the scope of this package and the molecular visualization universe is rich and complex (unlike this module).

The methods offered by this module are:

.. autosummary::

   molpx.visualize.FES
   molpx.visualize.sample
   molpx.visualize.traj
   molpx.visualize.correlations
   molpx.visualize.feature

.. automodule:: molpx.visualize
   :members:


