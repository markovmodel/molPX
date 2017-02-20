# projection_explorer by gph82.
ipython API to visualize MD-trajectories interactively on any projection space.

The API is projX, which provides access to projX.generate and projX.visualize

The notebook Projection_Explorer explains the general cases, 
but you can combine the methods freely.


WARNINGS:
- The important methods of bmutils have been tested, but higher level API
 have only been tested superfically.
- This is currently under heavy development and the API might change rapidly

INSTALL:
python setup.py install
(conda installation will follow)

KNOWN ISSUES:
- installation of nglview might give a "SandboxViolation" error. IDK how to 
    fix this for now. Recommended install is then to externally use "conda
    install nglview -c bioconda"

NOTE:
This package is based upon the nglview package by Alexander Rose, Ngyen

Have FUN!
