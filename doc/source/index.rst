.. Projection Explorer documentation master file, created by
   sphinx-quickstart on Mon Feb 20 13:34:27 2017.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to Projection Explorer's Documentation!
===============================================
Projection Explorer (projX) is a python module that provides **interactive visualization of projected coordinates
of molecular dynamics (MD) trajectories** inside an ipython notebook. At the moment the API consists of two subpackages:

 * ``projX.visualize``
 * ``projX.generate``

Projection Explorer uses the incredibly useful  ``nglview`` `IPython/Jupyter <https://github.com/arose/nglview>`_ widget. Other libraries heavily used are are `mdtraj <http://mdtraj.org/>`_ and `PyEMMA <http://www.emma-project.org/latest/>`_, a library into which projX will utimately be merged into.

Download and Install
=====================
At the moment, cloning or downloading the `source from github <https://github.com/gph82/projection_explorer>`_ is the only option to get projX. After that, just cd to the directory ``projection explorer`` and issue

    >>> python setup.py install
    >>>

You can build html documentation alongside the installation by issuing

    >>> python setup.py install build_sphinx
    >>>

See the warning and known issues for more info.

WARNINGS:
=========
 * The important methods (stored in bmutils) have been tested thoroughly. The higher level API-functions are not yet fully tested.
 * This is currently under heavy development and the API might change rapidly.

KNOWN ISSUES:
=============
The installation of nglview might give a "SandboxViolation" error. Until this is fixed, the recommended install is
to externally issue

    >>> conda install nglview -c bioconda
    >>>


Module's Documentation
======================
.. toctree::
   :maxdepth: 0

   index_visualize
   index_generate

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

