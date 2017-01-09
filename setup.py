__author__ = 'gph82'

#Minimal setup script
from setuptools import setup

setup(name='projX',
      version='0.0',
      packages=[
          'projX',
          'projX.tests'
      ],
      install_requires=[
          'nglview>=0.6.2.1',
          'pyemma',
          'scikit-learn',
          'notebook',
      ],
          package_data = {
              'projX': ['notebooks/*', 'notebooks/data/*']
          }
)



