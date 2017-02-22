## projection_explorer (projX) by gph82

projX (projection explorer) is an ipython API to visualize MD-trajectories interactively on any projection space inside an ipython notebook. 

It provides access to `projX.generate` and `projX.visualize`. The notebook `projX/notebooks/Projection_Explorer.ipynb` explains the general cases, but you can combine the methods freely.

Projection Explorer uses the incredibly useful  [``nglview``] (https://github.com/arose/nglview) `IPython/Jupyter widget`.
Other libraries heavily used are are [`mdtraj`] (http://mdtraj.org/) and [`PyEMMA`] (http://www.emma-project.org/latest/), a library into which projX will utimately be merged into. At the moment, there is also an [`sklearn`] (http://scikit-learn.org/stable/index.html)  dependency that might disappear in the future.

### WARNINGS:

* The important methods (bmutils) have been tested, the level API
 has only been tested superficially. Expect some instability
* This is currently under heavy development and the API might change rapidly

### INSTALLATION:
    
    >>> python setup.py install
    
### DOCUMENTATION:

    >>> python setup.py install build_sphinx
    
This will generate `projection_explorer/docs/build/html/index.html` with the html 
documentation.

There is also a static copy of the latest html pages [here] (http://page.mi.fu-berlin.de/gph82/projX/) (this will change soon)

### KNOWN ISSUES:
 
### The interplay between nglview, nbextensions, ipywidgets might limit you to use python3.X on some platforms. Sorry about that.

Installation of `nglview` might give a "SandboxViolation" error. IDK how to 
 fix this for now. Recommended install is then to externally use 
    
    >>> conda install nglview -c bioconda

or, alternatively
  
    >>> pip install nglview
    
On MAC-OS, an `unknown locale: UTF-8` error might show up when building the docs. [According 
to some guy on the internet] (https://coderwall.com/p/-k_93g/mac-os-x-valueerror-unknown-locale-utf-8-in-python), adding this line will help with that, although it seems pretty aggresive
to change all locale just for this.

    export LC_ALL=en_US.UTF-8
    export LANG=en_US.UTF-8
 
Have FUN!
