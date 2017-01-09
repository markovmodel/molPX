
# projection_explorer by gph82.
ipython API to visualize MD-trajectories interactively on any projection space.

The API is projX. bmutils is mostly untested (this will change shortly).
The notebook Projection_Explorer explains the general cases, 
but you can combine the methods in bmutils freely.

This is currently under heavy development and the API might change rapidly

There is a known issue with nglview's version specification:
```creating /home/mi/blah/miniconda/lib/python2.7/site-packages/nglview-0+unknown-py2.7.egg
Extracting nglview-0+unknown-py2.7.egg to /home/mi/blah/miniconda/lib/python2.7/site-packages
Adding nglview 0+unknown to easy-install.pth file
Installing nglview script to /home/mi/blah/miniconda/bin

Installed /home/mi/blah/miniconda/lib/python2.7/site-packages/nglview-0+unknown-py2.7.egg
error: The 'nglview>=0.6.2.1' distribution was not found and is required by projX
```
This is why we recommend a conda install


Have FUN!