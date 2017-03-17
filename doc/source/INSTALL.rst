============
Installation
============

To install molPX , you need a few Python package dependencies. If these dependencies are not
available in their required versions, the installation will fail. We recommend one particular way for the installation
that is relatively safe, but you are welcome to try another approaches if you know what you are doing.


Anaconda install (Recommended)
==============================

We strongly recommend to use the Anaconda scientific python distribution in order to install
python-based software. Python-based software is not trivial to distribute
and this approach saves you many headaches and problems that frequently arise in other installation
methods. You are free to use a different approach (see below) if you know how to sort out problems,
but play at your own risk.

If you already have a conda installation, directly go to step 3:

1. Download and install miniconda for Python 2.7 or 3+, 32 or 64 bit depending on your system. Note that
   you can still use Python 2.7, however we recommend to use Python3:

   http://conda.pydata.org/miniconda.html


   For Windows users, who do not know what to choose for 32 or 64 bit, it is strongly
   recommended to read the second question of this FAQ first:

   http://windows.microsoft.com/en-us/windows/32-bit-and-64-bit-windows


   Run the installer and select **yes** to add conda to the **PATH** variable.

2. If you have installed from a Linux shell, either open a new shell to have an updated PATH,
   or update your PATH variable by ``source ~/.bashrc`` (or .tcsh, .csh - whichever shell you are using).

3. Add the omnia-md software channel, and install (or update) molPX:

   .. code::

      conda config --add channels omnia
      conda install pyemma

   if the command conda is unknown, the PATH variable is probably not set correctly (see 1. and 2.)

4. Check installation:

   .. code::

      conda list

   shows you the installed python packages. You should find a molpx 0.1.2 (or later)
   and ipython, ipython-notebook 3.1 (or later). If ipython is not up to date, you canot use molPX. Please update it by

   .. code::

      conda install ipython-notebook

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
 * A ``SandboxViolation`` error might appear when installing from source. Until we figure this out,
 try to install ``nglview`` externally issuing:


    >>> conda install nglview -c bioconda

    or, alternatively

    >>> pip install nglview

 * Note that molPX only works with ``nglview`` versions >=0.6.2.1.

 * The interplay between some modules (nglview, nbextensions, ipywidgets) might limit you to use python3.X on some platforms. Sorry about that.