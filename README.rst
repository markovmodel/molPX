###################################################
Welcome to molPX: The Molecular Projection Explorer
###################################################
|DOI| |travis_build| |appveyor_build| |coverage| |docs_build|

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

* :doc:`Example Jupyter Notebooks </index_notebooks>`

Find more about the people behind molPX here:

* :doc:`About & YouTube Introduction </about>`

Download and Install
=====================

If you can't wait to play around with molPX, and you have the `Anaconda scientifc python distribution
<https://www.continuum.io/downloads>`_ (which we strongly recommend), the easiest way to get molPX is to issue
the `conda command <https://conda.io/docs/intro.html>`_:

   >>> conda install molpx -c omnia

and jump to the Quick Start section of this document. Otherwise, check out our more exhaustive

* :doc:`Installation Guide </INSTALL>`



Quick Start
=============

Start an ``IPython`` console

    >>> ipython

Import ``molpx`` and let the example notebook guide you

    >>> import molpx
    >>> molpx.example_notebooks()

Voilà: you should be looking at a list of jupyter notebooks explaining the basic functionality of molPX

Documentation
==============

You can find the latest documentation online `here <https://molpx.readthedocs.io/>`_
You can build a local copy of the html documentation by navigating to the molPX installation
directory and issuing:

    >>> cd doc
    >>> make html

This will generate `molPX/docs/build/html/index.html` with the html documentation. If you are missing some of
the requirements for the documentation , issue:

    >>> pip install -r ./source/doc_requirements.txt

If you don't know where molPX is installed, you can find out this way:

    >>> ipython
    >>> import molpx
    >>> molpx._molpxdir()

The output of the last command is one subdirectory of molPX's installation directory, so just copy it and issue:

    >>> cd the-output-of-the-molpx._molpxdir-command
    >>> cd ..

and you are there !

Warnings
=========

molPX is currently under heavy development and the API might change rapidly. Stay tuned.

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

* This is most easily realized from terminal by issuing:

        >>> mkdir ~/.molpx
        >>> echo "report_status = False" >> ~/.molpx/conf_molpx.py

* You can check your report status anytime by typing this line in a (i)python terminal

        >>> import molpx
        >>> molpx._report_status()

* If you don't know where your home folder is (for whatever reason), you can find it out by typing in a (i)python terminal

        >>> import os
        >>> os.path.expanduser('~/.molpx')

.. |DOI| image::
   https://zenodo.org/badge/76460348.svg
   :target: https://zenodo.org/badge/latestdoi/76460348
   :height: 20
   :alt: DOI

.. |travis_build| image::
   https://travis-ci.org/markovmodel/molPX.svg?branch=master
   :height: 10
   :target: https://travis-ci.org/markovmodel/molPX
   :alt: Travis build status

.. |appveyor_build| image::
   https://ci.appveyor.com/api/projects/status/396ej39s3fewhwy9/branch/master?svg=true
   :height: 10
   :target: https://ci.appveyor.com/project/gph82/molpx
   :alt: Appveyor build status

.. |coverage| image::
   https://codecov.io/gh/markovmodel/molPX/branch/master/graph/badge.svg
   :height: 20
   :target: https://codecov.io/gh/markovmodel/molPX
   :alt: Codecov

.. |docs_build| image::
   https://readthedocs.org/projects/molpx/badge/?version=latest
   :alt: Documentation Status
   :height: 20
   :target: http://molpx.readthedocs.io/en/latest/?badge=latest
   
