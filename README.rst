###################################################
Welcome to molPX: The Molecular Projection Explorer
###################################################
.. image::
   https://travis-ci.org/markovmodel/molPX.svg?branch=master
   :height: 10
   :target: https://travis-ci.org/markovmodel/molPX
   :alt: Travis build status

.. image::
   https://ci.appveyor.com/api/projects/status/396ej39s3fewhwy9/branch/master?svg=true
   :height: 10
   :target: https://ci.appveyor.com/project/gph82/molpx
   :alt: Appveyor build status

The Molecular Projection Explorer, molPX, is a python module that provides **interactive visualization of
projected coordinates of molecular dynamics (MD) trajectories** inside a `Jupyter notebook <http://jupyter.org/>`_.

molPX is based on the incredibly useful  `nglview IPython/Jupyter widget <https://github.com/arose/nglview>`_.
Other libraries heavily used are  `mdtraj <http://mdtraj.org/>`_ and `PyEMMA <http://www.emma-project.org/latest/>`_.
At the moment, there is also an `sklearn <http://scikit-learn.org/stable/index.html>`_ dependency that might disappear in the future.

.. image:: ../images/output.gif
   :align: center

At the moment the API consists of two subpackages:

* :doc:`molpx.visualize </index_visualize>`
* :doc:`molpx.generate  <index_generate>`

**TL;DR**: see molPX in action through the

* :doc:`Example Jupyter Notebook </index_notebooks>`

Find more about the people behind molPX here:

* :doc:`About & YouTube Introduction </about>`

Download and Install
=====================

At the moment, the easiest way is to get molPX is from `PyPI - the Python Package Index
<https://pypi.python.org/pypi/molPX/>`_ using `pip <https://packaging.python.org/installing/>`_ by typing this command
from the terminal:

    >>> pip install molpx

You can also clone or download the `source from github <https://github.com/markovmodel/molPX>`_.
After that, just cd to the download directory (and untar/unzip if necessary) and:

    >>> cd molPX
    >>> python setup.py install

See the "Known Issues" below if you get a ``SandboxViolation`` error.

Quick Start
=============

Start an ``IPython`` console

    >>> ipython

Import ``molpx`` and let the example notebook guide you

    >>> import molpx
    >>> molpx.example_notebook()

These commands should put you in front of a jupyter notebook explaining the basic functionality of molPX

Documentation
==============

You can find the latest documentation online `here <https://readthedocs.org/projects/molpx/>`_.
You can build a local copy of the html documentation by issuing

    >>> cd docs
    >>> make html

This will generate `molPX/docs/build/html/index.html` with the html documentation.

Warnings
=========

 * molPX is currently under heavy development and the API might change rapidly.

Data Privacy Statement
======================

When you import this Python package, some of your metadata is sent to our servers. These are:

 * molPX version
 * Python version
 * Operating System
 * Hostname/ mac address of the accessing computer
 * Time of retrieval

How to disable this feature easily:
-----------------------------------
Even before you use molPX for the first time:

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
 * The installation of nglview might give a ``SandboxViolation`` error. Until we figure this out,
 try to install ``nglview`` externally issuing:


    >>> conda install nglview -c bioconda

    or, alternatively

    >>> pip install nglview

 * Note that molPX only works with ``nglview`` versions >=0.6.2.1.

 * The interplay between some modules (nglview, nbextensions, ipywidgets) might limit you to use python3.X on some platforms. Sorry about that.