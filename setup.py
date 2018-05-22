__author__ = 'gph82'

#Minimal setup script
from setuptools import setup
import versioneer

meta_data = dict(name='molPX',
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
          'ipympl',
          'ipywidgets>=7',
          'matplotlib',
          'mdtraj',
          'nglview>=1',
          'notebook',
          'pyemma',
          'scikit-learn',
      ],
      package_data={
          'molpx': ['notebooks/*',
                    'notebooks/data/*'],
      },
      include_package_data=True,
)

if __name__ == '__main__':
    setup(**meta_data)
