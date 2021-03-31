#!/usr/bin/env python

from setuptools import setup

setup(name='dicedps',
      version='1.0',
      description='Framework for experimenting with DICE with DPS',
      author='Giacomo Marangoni',
      author_email='jackjackk@users.noreply.github.com',
      license="GNU GPL version 3",
      url='https://github.com/jackjackk/dicedps',
      packages=['dicedps'],
      install_requires=['sympy', 'pathos', 'tqdm', 'Platypus-Opt', 'dill', 'xarray'],
      classifiers=[],
     )
