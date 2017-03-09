.. Projection Explorer documentation master file, created by
   sphinx-quickstart on Mon Feb 20 13:34:27 2017.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

###################################################
Welcome to molPX: The Molecular Projection Explorer
###################################################

The Molecular Projection Explorer, molPX, is a python module that provides **interactive visualization of
projected coordinates of molecular dynamics (MD) trajectories** inside a jupyter notebook.

molPX is based on the incredibly useful  `nglview IPython/Jupyter widget <https://github.com/arose/nglview>`_.
Other libraries heavily used are  `mdtraj <http://mdtraj.org/>`_ and `PyEMMA <http://www.emma-project.org/latest/>`_.
At the moment, there is also an `sklearn <http://scikit-learn.org/stable/index.html>`_  dependency that might disappear in the future.

.. image:: ../images/output.gif
   :align: center

At the moment the API consists of two subpackages:

.. toctree::
   :maxdepth: 1

   index_visualize
   index_generate

Most of the heavy lifting is done by the methods inside:

.. toctree::
    :maxdepth: 1

    index_bmutils

**TL;DR**: see molPX in action through the

.. toctree::
   :maxdepth: 1

   index_notebooks

Find more about the people behind molPX here:

.. toctree::

   About & YouTube Introduction <about>

.. .. contents::
   What you'll find on this page

Download and Install
=====================

At the moment, cloning or downloading the `source from github <https://github.com/markovmodel/molPX>`_
is the only option to get molPX. After that, just cd to the directory `projection explorer` and issue

    >>> python setup.py install

Quick Start
=============

    >>> cd molpx/notebooks
    >>> jupyter notebook Projection_Explorer.ipynb

should put you in front of a jupyter notebook explaining the basic functionality.

Documentation
==============
You can build html documentation by issuing

    >>> cd docs
    >>> make html 

This will generate ``projection_explorer/docs/build/html/index.html`` with the html
documentation.

Warnings
=========

 * The important methods (bmutils) have been tested, the API  has only been tested superficially. Expect some instability.

 * This is currently under heavy development and the API might change rapidly.

Data Privacy Statement
======================

When you import this Python package, some of your metadata is sent to our servers. These are:

 * molPX version
 * Python version
 * Operating System
 * Hostname/ mac address of the accessing computer
 * Time of retrieval

It is very easy to disable this feature, even before you use install `molpx` for the first time. Here's how
-----------------------------------------------------------------------------------------------------------

 1. Create a hidden folder `.molpx` in your home folder
 2. Create a file `conf_molpx.py` inside of `.molpx` with the following line:
    `report_status = False`
 3. Restart your ipython/jupyter sessions

Hints:

* You can check your report status anytime by typing this line in a (i)python terminal

        >>> import molpx
        >>> molpx._report_status()

* If you don't know where your home folder is (for whatever reason), you can find it out by typing in a (i)python terminal

        >>> import os
        >>> os.path.expanduser('~/.molpx')


Known Issues
=============
 * The installation of nglview might give a ``SandboxViolation`` error. Until this is fixed, the recommended install is
   to externally issue

    >>> conda install nglview -c bioconda

    or, alternatively

    >>> pip install nglview

 * Projection Explorer only works with ``nglview`` versions >=0.6.2.1. 

 * **The interplay between nglview, nbextensions, ipywidgets might limit you to use python3.X on some platforms.
   Sorry about that.**

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

