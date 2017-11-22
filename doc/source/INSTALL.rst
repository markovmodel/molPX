============
Installation
============

To install molPX , you need a few Python package dependencies. If these dependencies are not
available in their required versions, the installation will fail. We recommend one particular way for the installation
that is relatively safe, but you are welcome to try another approaches if you know what you are doing.


Anaconda Install (recommended)
==============================

We strongly recommend to use the Anaconda scientific python distribution in order to install
python-based software. Python-based software is not trivial to distribute
and this approach saves you many headaches and problems that frequently arise in other installation
methods. You are free to use a different approach (see below) if you know how to sort out problems,
but play at your own risk.

If you already have a conda installation, directly go to step 3:

1. Download and install miniconda for Python 3+, 32 or 64 bit depending on your system.

   http://conda.pydata.org/miniconda.html


   For Windows users, who do not know what to choose for 32 or 64 bit, it is strongly
   recommended to read the second question of this FAQ first:

   http://windows.microsoft.com/en-us/windows/32-bit-and-64-bit-windows


   Run the installer and select **yes** to add conda to the **PATH** variable.

2. If you have installed from a Linux shell, either open a new shell to have an updated PATH,
   or update your PATH variable by ``source ~/.bashrc`` (or .tcsh, .csh - whichever shell you are using).

3. Install molPX using the conda-forge channel:

   ::

      conda install molpx -c conda-forge

   if the command ``conda`` is unknown, the PATH variable is probably not set correctly (see 1. and 2.)

4. Check installation:

   ::

      conda list

   shows you the installed python packages. You should find a molpx 0.1.2 (or later).
   molPX requires all the following packages, s. t. they will be installed (and their dependencies) if they
   are not installed already.

   ::

      nglview>=1
      ipywidgets>=7
      pyemma
      scikit-learn
      notebook
      mdtraj
      ipympl

Python Package Index (PyPI)
===========================

If you do not like Anaconda for some reason you should use the Python package
manager **pip** to install. This is not recommended, because in the past,
various problems have arisen with pip in compiling the packages that molPX depends upon, see `this issue
<https://github.com/markovmodel/molPX/issues/16>`_ for more information.

1. If you do not have pip, please read the install guide:
   `install guide <http://pip.readthedocs.org/en/latest/installing.html>`_.

2. Make sure pip is enabled to install so called
   `wheel <http://wheel.readthedocs.org/en/latest/>`_ packages:

   ::

      pip install wheel

   Now you are able to install binaries if you use MacOSX or Windows.

3. Install molPX using

   ::

      pip install molPX

4. Check your installation

   ::

      python
      >>> import molpx
      >>> molpx.__version__

   should print 0.1.2 or later

   ::

      >>> import IPython
      >>> IPython.__version__

   should print 3.1 or later. If ipython is not up to date, update it by ``pip install ipython``

Building from Source
====================
Building all dependencies from molPX from source is sometimes (if not usually) tricky, takes a
long time and is error prone. **It is not recommended nor supported by us.**
If unsure, use the Anaconda installation.

What you can do is clone or download the `source from github <https://github.com/markovmodel/molPX>`_.
After that, just cd to the download directory (and untar/unzip if necessary) and:

    >>> cd molPX
    >>> python setup.py install

but be aware that success is not guaranteed. See the "Known Issues" below.

Known Issues
=============
 * After a successfull installation and execution of ``molpx.example_notebooks()``...no widgets are shown! Most probably, this has to do with `nglview <https://github.com/arose/nglview/#released-version>`_ and its needed Jupyter notebook extensions.
 **This is a known, frustrating behaviour:**

    * `deja vu: nglview widget does not appear <https://github.com/arose/nglview/issues/599>`_
    * `nglview widget does not appear <https://github.com/arose/nglview/issues/718>`_
    * `Troubleshooting: nglviewer fresh install tips <https://github.com/SBRG/ssbio/wiki/Troubleshooting#nglviewer-fresh-install-tips>`_

  We'are aware of this and molPX automatically checks for the needed extensions every time it gets imported, s.t. it will refuse
  to start-up if it finds any problems. If somehow it manages to start-up but no widget is shown, try issuing
    ::

        molpx._auto_enable_extensions()

    restarting everything and trying again. Otherwise, check the above links.

 * A ``SandboxViolation`` error might appear when installing from source. This is because the ``nglview`` dependency
 is trying to enable the needed extensions. Until we figure this out,
 try to install ``nglview`` externally issuing:


    >>> conda install nglview -c bioconda

    or, alternatively

    >>> pip install nglview

 * Note that molPX only works with ``nglview`` versions >=0.6.2.1.
