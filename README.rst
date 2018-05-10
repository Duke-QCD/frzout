frzout
======
*Particlization model (Cooper-Frye sampler) for relativistic heavy-ion collisions*

Documentation
-------------
`qcd.phy.duke.edu/frzout <http://qcd.phy.duke.edu/frzout>`_

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
