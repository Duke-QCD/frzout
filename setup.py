#!/usr/bin/env python
# -*- coding: utf-8 -*-

from setuptools import setup, Extension
from Cython.Build import cythonize


def version():
    with open('frzout/__init__.py', 'r') as f:
        for l in f:
            if l.startswith('__version__ = '):
                return l.split("'")[1]

    raise RuntimeError('unable to determine version')


def long_description():
    with open('README.rst') as f:
        return f.read()


# TODO only cythonize for dev builds
ext_modules = cythonize([
    Extension(
        'frzout._frzout',
        ['frzout/_frzout.pyx'],
    ),
    Extension(
        'frzout.test._test_fourvec',
        ['frzout/test/_test_fourvec.pyx'],
    )],
    compiler_directives=dict(cdivision=True, language_level=3)
)

setup(
    name='frzout',
    version=version(),
    description='Particlization model for relativistic heavy-ion collisions',
    long_description=long_description(),
    author='Jonah Bernhard',
    author_email='jonah.bernhard@gmail.com',
    url='https://github.com/Duke-QCD/frzout',
    license='MIT',
    packages=['frzout', 'frzout.test'],
    package_data={'frzout': ['mass_width_2017.mcd']},
    ext_modules=ext_modules,
    install_requires=['numpy', 'scipy >= 0.18.0'],
    classifiers=[
        'Development Status :: 5 - Production/Stable',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Cython',
        'Programming Language :: Python',
        'Programming Language :: Python :: 2',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Topic :: Scientific/Engineering :: Physics'
    ]
)
