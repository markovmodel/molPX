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
          'nglview>=1',
          'ipywidgets>=7',
          'pyemma',
          'scikit-learn',
          'notebook',
          'mdtraj',
          'ipympl',
      ],
      package_data = {
          'molpx': ['notebooks/*', 'notebooks/data/*'],
      }
)



