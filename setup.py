__author__ = 'gph82'

#Minimal setup script
from setuptools import setup
import versioneer

setup(name='projX',
      version=versioneer.get_version(),
      cmdclass=versioneer.get_cmdclass(),
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



