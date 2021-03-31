#!/usr/bin/env python

from setuptools import setup

setup(name='paradigm',
      version='0.0',
      description='Framework for designing and solving optimization problems',
      author='Giacomo Marangoni',
      author_email='jackjackk@users.noreply.github.com',
      license="GNU GPL version 3",
      url='https://github.com/jackjackk/paradigm',
      packages=['paradigm'],
      install_requires=['Platypus-Opt', 'dill', 'xarray'],
      classifiers=[],
     )
