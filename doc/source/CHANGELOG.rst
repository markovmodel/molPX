=========
Changelog
=========

0.1.6

**New features**:

- ``molpx.notebooks()``
    - interface changed to open the list of available notebooks
    - molpx.notebook() is tagged as deprecated

- ``molpx.visualize``:
   - new methods
        - ``correlations()`` to visualize the feature_TIC_correlation attribute of PyEMMA TICA-objects in the widget
        - ``feature()`` accepts a PyEMMA MDFeaturizer object directly

   - ``traj()``
    - has new optional parameter ``projection`` to accept a PyEMMA TICA object to
    visualize the linear correlations in the widget AND in as trajectories f(t)

    - ``FES``:
        - can show also 1D FESs
        - accepts "weights" as optional argument for showing re-weighted FESs (as in metadynamics)
        - argument "proj_labels" can be a PyEMMA TICA object to directly use the tica.describe() strings

- ``molpx.bmutils``:
    - new methods:
        - ``most_corr_info()`` to support the new methods in ``molpx.visualize``
        - ``labelize()`` to deal with proj_label inputs
        - ``superpose_to_most_compact_in_list()`` to superpose geoms with some intelligence in the input parsing

**New Notebooks**:
   - new notebooks added to the documentation

**Fixes**:

    - ``molpx.visualize``:
       - all calls to ``nglview.show_mdtraj()`` has been wrapped in ``_initialize_nglwidget_if_safe`` to avoid
            erroring in tests
    -
