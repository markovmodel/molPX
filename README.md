## projection_explorer (projX)

`projX` (projection explorer) is a jupyter API to visualize MD-trajectories interactively on any projection space inside a jupyter notebook.

It provides access to `projX.generate` and `projX.visualize`. The notebook `projX/notebooks/Projection_Explorer.ipynb` explains the general cases, but you can combine the methods freely.

Projection Explorer uses the incredibly useful  [``nglview``] (https://github.com/arose/nglview) `IPython/Jupyter widget`.
Other libraries heavily used are are [`mdtraj`] (http://mdtraj.org/) and [`PyEMMA`] (http://www.emma-project.org/latest/), a library into which `projX` will utimately be merged into. At the moment, there is also an [`sklearn`] (http://scikit-learn.org/stable/index.html)  dependency that might disappear in the future.

## Warnings:
* This is currently under **heavy development** and the API might change rapidly, to the point 
of even **changing its name** in its near future, so please stay tuned.

* Until there is a proper release, consider this just a ipython repository that changes
  rapidly.

* The important methods (`bmutils`) have been tested for correctness, 
the API methods has only been tested less thouroughly. Expect some instability
 

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

* projX version
* Python version
* Operating System
* Hostname/ mac address of the accessing computer
* Time of retrieval

### It is very easy to disable this feature, even before you use install `projX` for the first time. Here's how:#

 1. Create a hidden folder `.projX` in your home folder 
 2. Create a file `projX_conf.py` inside of `.projX` with the following line:
    `report_status = False`        
 3. Restart your ipython sessions
 
Hints:

* You can check your report status anytime by typing this line in a (i)python terminal

        >>> import projX
        >>> projX._report_status()
    
* If you don't know where your home folder is (for whatever reason), you can find it out by typing in a (i)python terminal
    
        >>> import os x    
        >>> os.path.expanduser('~/.projX')

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
