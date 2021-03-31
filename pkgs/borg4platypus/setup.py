#!/usr/bin/env python

from setuptools import setup

setup(name='borg4platypus',
      version='0.1',
      description='Platypus-compatible wrapper of the Python wrapper of BORG',
      author='Giacomo Marangoni',
      author_email='jackjackk@users.noreply.github.com',
      license="GNU GPL version 3",
      url='https://github.com/jackjackk/borg4platypus',
      packages=['borg4platypus'],
      install_requires=['platypus', 'dill', 'tqdm', 'numpy', 'pandas'],
      classifiers=[],
     )
