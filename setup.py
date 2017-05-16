__author__ = 'gph82'

#Minimal setup script
from setuptools import setup
import versioneer

setup(name='molPX',
      author='Guillermo Perez-Hernandez',
      author_email='guille.perez@fu-berlin.de',
      maintainer='Martin K. Scherer',
      maintainer_email='m.scherer@fu-berlin.de',
      version=versioneer.get_version(),
      cmdclass=versioneer.get_cmdclass(),
      packages=[
          'molpx',
          'molpx.tests'
      ],
      install_requires=[
          'nglview>=0.6.2.2',
          'ipywidgets>=5.2.2',
          'pyemma',
          # fake requiring these here, see issue #16
          'numpy',     
          'cython>=0.19',
          'ipywidgets',
          # end fake require
          'scikit-learn',
          'notebook',
          'mdtraj',
      ],
      package_data = {
          'molpx': ['notebooks/*', 'notebooks/data/*'],
      }
)



