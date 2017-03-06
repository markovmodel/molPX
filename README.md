## Molecular Projection Explorer (molPX)

molPX (Molecular Projection Explorer) is a jupyter API to visualize MD-trajectories interactively on any projection space inside a jupyter notebook.

It provides access to the methods `molpx.generate` and `molpx.visualize`. The notebook `molpx/notebooks/Projection_Explorer.ipynb` explains the general cases, but you can combine the methods freely.

Projection Explorer uses the incredibly useful  [``nglview``] (https://github.com/arose/nglview) `IPython/Jupyter widget`.
Other libraries heavily used are [`mdtraj`] (http://mdtraj.org/) and [`PyEMMA`] (http://www.emma-project.org/latest/). At the moment, there is also an [`sklearn`] (http://scikit-learn.org/stable/index.html)  dependency that might disappear in the future.

## Installation:
    
    >>> python setup.py install
    
## Documentation:
A lot of effort has been made to document this project properly. Appart from the docstring documentation that will show on
your ipython terminal, there are also html pages and ipython tutorial notebooks online. 
You can find everything [here] (http://projection-explorer.readthedocs.io/).

You can also build it locally issuing the following command:

    >>> python setup.py install build_sphinx --build-dir doc/build/
    
or alternatively:

    >>> cd docs
    >>> make html

This will generate `projection_explorer/docs/build/html/index.html` with the html documentation, which you can then access locally through 
your browser, e.g:

    >>> firefox doc/build/html/index.html


## Data Privacy Statement 

When you import this Python package, some of your metadata is sent to our servers. These are:

* molpx version
* Python version
* Operating System
* Hostname/ mac address of the accessing computer
* Time of retrieval

### It is very easy to disable this feature, even before you use install `molpx` for the first time. Here's how:

 1. Create a hidden folder `.molpx` in your home folder 
 2. Create a file `conf_molpx.py` inside of `.molpx` with the following line:
    `report_status = False`        
 3. Restart your ipython sessions
 
Hints:

* You can check your report status anytime by typing this line in a (i)python terminal

        >>> import molpx
        >>> molpx._report_status()
    
* If you don't know where your home folder is (for whatever reason), you can find it out by typing in a (i)python terminal
    
        >>> import os
        >>> os.path.expanduser('~/.molpx')

## Known Issues:
 
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
