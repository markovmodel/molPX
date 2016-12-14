__author__ = 'gph82'

#Minimal setup script
from setuptools import setup

setup(name='projX',
      version='0.0',
      py_modules=['projX'],
      requires=['numpy (>=1.7.0)',  # These are the versions that PyEMMA should be providing already
#                'joblib' # not needed at the moment
                ])

