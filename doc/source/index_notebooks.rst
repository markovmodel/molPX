.. Projection Explorer documentation master file, created by
   sphinx-quickstart on Mon Feb 20 13:34:27 2017.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Example Jupyter Notebooks
=========================

There are several ways to see the example notebooks, which you can find in the ``molpx/notebooks/``
installation directory.

1. Play:
    ``molpx`` has a method that will launch a working, temporary copy of the example notebooks.
    From an IPython console, just type::

    >>>> import molpx
    >>>> molpx.example_notebooks()

    A List of Juypter notebooks should automagically appear in front of you after a few seconds. This is the most interactive and
    usefull way to see ``molpx`` in action, but you'll only have access to it after downloading and installing ``molpx``

2. Read:
    The links you see below are an html-rendered version of the notebook.
    Click on them    to navigate the notebook. **Unfortunately** for this html documentation, ``nglview``‘s output,
    i.e. the pictures of molecular structures, `cannot be stored currently in the notebook file
    <https://github.com/ipython/ipywidgets/issues/754#issuecomment-267814374>`_. In short: the html-notebook is
    lacking the most visually appealing part of ``molpx``.

.. toctree::
   :maxdepth: 1

   Intro with BPTI <notebook_molpx_intro.rst>
   Intro with Di-Alanine <notebook_molpx_intro_DiAla.rst>
   Metadynamics <notebook_molpx_meta.rst>
   PyEMMA features <notebook_molpx_features.rst>

3. Watch:
    Our :doc:`Youtube video </about>`  or the :doc:`gif animation </index>` show ``molpx`` in action.



