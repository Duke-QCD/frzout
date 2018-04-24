frzout
======
*Particlization model (Cooper-Frye sampler) for relativistic heavy-ion collisions*

:Author: Jonah Bernhard
:Language: Python / `Cython <http://cython.org>`_
:Source code: `github:duke-qcd/frzout <https://github.com/Duke-QCD/frzout>`_

Features
--------
- 2D (boost-invariant) and 3D sampling
- Resonance mass distributions
- Shear and bulk viscous corrections
- Fast: fraction of a second per sample

Quick start
-----------
Install::

   pip install frzout

Basic usage:

.. code-block:: python

   import frzout

   # create surface object from data arrays
   surface = frzout.Surface(x, sigma, v, pi=pi, Pi=Pi)

   # create hadron resonance gas object at T = 0.150 GeV
   hrg = frzout.HRG(.150)

   # sample particles
   parts = frzout.sample(surface, hrg)

.. note::
   frzout is part of the `Duke heavy-ion collision event generator <https://github.com/Duke-QCD/hic-eventgen>`_.
   If your goal is to run complete events, you probably don't need to use frzout directly.

.. toctree::
   :caption: User guide
   :maxdepth: 2

   install
   usage

.. toctree::
   :caption: Technical info
   :maxdepth: 2

   physics
   tests
