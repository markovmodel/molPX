__author__ = 'gph82'

#Minimal setup script
from setuptools import setup
import shutil

# This is an ugly UGLY hack that I'll get rid of sometime soon
shutil.copy('projX/notebooks/Projection_Explorer.ipynb', 'doc/source/Projection_Explorer_Copy.ipynb')

setup(name='projX',
      version='0.1.0',
      packages=[
          'projX',
          'projX.tests'
      ],
      install_requires=[
          'nglview>=0.6.2.1',
          'pyemma',
          'scikit-learn',
          'notebook',
          'mdtraj',
          'sphinx',
          'sphinx_rtd_theme',
          'nbsphinx'
      ],
          package_data = {
              'projX': ['notebooks/*', 'notebooks/data/*']
          }
)



