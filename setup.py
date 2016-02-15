#!/usr/bin/env python
# -*- coding: utf-8 -*-

from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy

with open('README.rst') as f:
    long_description = f.read()

# TODO only cythonize for dev builds
ext_modules = cythonize([
    Extension(
        'frzout.sampler',
        ['frzout/sampler.pyx'],
        include_dirs=[numpy.get_include()]
    ),
    Extension(
        'frzout.test._test_fourvec',
        ['frzout/test/_test_fourvec.pyx'],
    ),
])

setup(
    name='frzout',
    version='0.1.0',  # TODO single source version
    description='Cooper-Frye hypersurface sampler',
    long_description=long_description,
    author='Jonah Bernhard',
    author_email='jonah.bernhard@gmail.com',
    url='https://github.com/Duke-QCD/frzout',
    license='MIT',
    packages=['frzout', 'frzout.test'],
    package_data={'frzout': ['mass_width_2015.mcd']},
    ext_modules=ext_modules,
    install_requires=['numpy', 'scipy'],
    classifiers=[
        'Development Status :: 1 - Planning',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Cython',
        'Programming Language :: Python',
        'Programming Language :: Python :: 2',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
        'Topic :: Scientific/Engineering :: Physics'
    ]
)
