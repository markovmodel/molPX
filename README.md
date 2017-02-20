## projection_explorer (projX) by gph82
ipython API to visualize MD-trajectories interactively on any projection space inside an ipython notebook. The API is `projX`, which provides access to `projX.generate` and `projX.visualize`. The notebook `projX/notebooks/Projection_Explorer.ipynb` explains the general cases, but you can combine the methods freely.


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

### KNOWN ISSUES:
 
Installation of `nglview` might give a "SandboxViolation" error. IDK how to 
 fix this for now. Recommended install is then to externally use 
    
    >>> conda install nglview -c bioconda

or, alternatively
  
    >>> pip install nglview
    
### The interplay between nglview, nbextensions, ipywidgets might limit you to use python3.X on some platforms. Sorry about that.
 
Have FUN!
