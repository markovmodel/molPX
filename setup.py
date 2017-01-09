__author__ = 'gph82'

#Minimal setup script
from setuptools import setup

setup(name='projX',
      version='0.0',
      py_modules=['projX'],
      packages=[
                'projX',
          'projX.tests'
      ],
#      requires=[
                #'pyemma',
#                'nglview',
#                ],
      install_requires=[
          'nglview',
          'pyemma',
          'scikit-learn',
          #'jupyter',
          #'widgetsnbextension=1.2.6',
          #'ipywidgets=5.2.2',
      ]
)

