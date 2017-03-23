=========
Changelog
=========

0.1.5 (tba)

**New features**:

- ``molpx.visualize``:
   - new method ``visualize.correlations()`` to visualize the feature_TIC_correlation attribute of PyEMMA TICA-objects
     in the widget
   - method ``visualize.traj()`` has new optional parameter ``projection`` to accept a PyEMMA TICA object to
     visualize the linear correlations in the widget AND in as trajectories f(t)

- ``molpx.bmutils``:
   - new method ``bmutils.most_corr_info()`` to support the new methods in ``molpx.visualize``

- notebooks:
   - new section added to showcase the better PyEMMA integration, in particular with TICA

**Fixes**:

- ``molpx.visualize``:
   - all calls to ``nglview.show_mdtraj()`` has been wrapped in ``_initialize_nglwidget_if_safe`` to avoid
     erroring in tests