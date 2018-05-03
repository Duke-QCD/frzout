Physics notes
=============
The following summarizes some notable aspects of frzout.  See section 3.4 of
`the author's dissertation <https://inspirehep.net/record/1669345>`_ for a more
detailed and formal description, and the `tests` page for code verification.

.. _`Pratt & Torrieri, PRC 82 (2010)`: https://inspirehep.net/record/847070

Sampling algorithm
------------------
All credit to Scott Pratt for originally devising this algorithm.
See `Pratt & Torrieri, PRC 82 (2010)`_ and the appendix of
`Pratt, PRC 89 (2014) <https://inspirehep.net/record/1275867>`_.
frzout contains a new implementation of his ideas with slight modifications.

The Cooper-Frye formula is

.. math::

   E \frac{dN}{d^3p} = \frac{g}{(2\pi)^3} \int_\sigma f(p) \, p^\mu \, d^3\sigma_\mu

where the left-hand side is the average Lorentz-invariant spectrum, and the
integral on the right runs over the switching hypersurface.  Rearranging terms,
the average number of particles emitted from a finite volume element
:math:`\Delta\sigma_\mu` is

.. math::

   \langle dN \rangle
      = \frac{p\cdot\Delta\sigma}{E} \frac{d^3p}{(2\pi)^3} \, g \, f(p)
      = \frac{p\cdot\Delta\sigma}{p \cdot u} \frac{d^3p'}{(2\pi)^3} \, g \, f(p')

where in the second form *p'* is the momentum in the rest frame of the volume element.
Now multiplying and dividing by a volume *V*, this becomes

.. math::

   \langle dN \rangle = w(p) \, V \frac{d^3p'}{(2\pi)^3} \, g \, f(p'), \quad
   w(p) = \frac{1}{V} \frac{p\cdot\Delta\sigma}{p \cdot u}

where *w(p)* is a particle emission probability.
The volume *V* ensures *w(p)* ≤ 1; its optimal value is

.. math::

   V = \max\biggl( \frac{p\cdot\Delta\sigma}{p \cdot u} \biggr)
     = u\cdot\Delta\sigma + \sqrt{(u\cdot\Delta\sigma)^2 - (\Delta\sigma)^2}

In view of these relations, the sampling algorithm is:

1. Sample a particle four-momentum from a stationary thermal source of volume *V*.
   If the particle is a resonance, sample its mass in addition to the three-momentum (see below).
2. Apply the viscous correction transformation (see below).
3. Boost the momentum from the rest frame of the volume element,
   i.e.\ an inverse boost by four-velocity *u*.
4. If :math:`p\cdot\Delta\sigma < 0`, reject the particle, otherwise accept the particle with probability *w(p)*.

Efficient algorithm for achieving Poissonian particle production:

1. Initialize a variable *S* with the negative of an exponential random number.
   Such a random number can be generated as :math:`S = \log(U)`,
   where :math:`U \in (0, 1]` is a uniform random number.

2. For each particle species in the hadron gas:

   a. Add *V n* to *S*, where *n* is the density of the species,
      so *V n* is the average number emitted from the volume.

   b. If *S* < 0, continue to the next species, otherwise perform the above
      sampling algorithm and then subtract an exponential random number from *S*.
      Continue sampling particles and subtracting from *S* until it again goes
      negative, then continue to the next species.

3. Repeat for each volume element.

This works because the time between Poisson events has an exponential distribution.

.. _res-mass-dist:

Resonance mass distributions
----------------------------
When the `HRG` class option *res_width* is enabled, the widths of resonances are
taken into account.  The distribution function of a resonance becomes

.. math:: f(p) = \int_{m_\text{min}}^{m_\text{max}} dm \, P(m) \, f(m, p)

where *f(m,p)* is the usual Bose-Einstein or Fermi-Dirac distribution for a
particle of mass *m* and *P(m)* is the mass probability distribution, assumed to
be a Breit-Wigner distribution

.. math:: P(m) \propto \frac{\Gamma(m)}{(m - m_0)^2 + \Gamma(m)^2/4}

with mass-dependent width

.. math:: \Gamma(m) = \Gamma_0 \sqrt{\frac{m - m_\text{min}}{m_0 - m_\text{min}}}

where

.. math::

   m_0 &= \text{PDG mass} \\
   \Gamma_0 &= \text{PDG width} \\
   m_\text{min} &= \text{total mass of lightest decay products} \\
   m_\text{max} &= m_0 + 4\Gamma_0

This mass-dependent width is designed to be physically reasonable and satisfy
the constraints that :math:`\Gamma(m_\text{min}) = 0` and :math:`\Gamma(m_0) = \Gamma_0`.

When *res_width* is enabled, the masses of resonances are randomly sampled from
*P(m)* during particle sampling.

.. _viscous-corrections:

Viscous corrections
-------------------
When the shear pressure tensor *pi* and/or bulk pressure *Pi* are given to the
`Surface` class, viscous corrections are applied to particle sampling.  The
correction method, based on `Pratt & Torrieri, PRC 82 (2010)`_, is to sample
momentum vectors in the local rest frame of the volume element and then apply a
linear transformation

.. math::

   p_i \rightarrow p_i + \sum_j \lambda_{ij} \, p_j, \quad
   \lambda_{ij} = (\lambda_\text{shear})_{ij} + \lambda_\text{bulk}\delta_{ij}

where λ is a transformation matrix chosen to reproduce the given viscous
pressures.

As shown in the reference, the shear part of the transformation is

.. math:: (\lambda_\text{shear})_{ij} = \frac{\tau}{2\eta} \pi_{ij}

where π\ :sub:`ij` is the spatial part of the shear tensor in the local rest
frame, and the shear viscosity over relaxation time η/τ can be calculated in the
hadron gas model (see function `HRG.eta_over_tau`).
This transformation is an approximation in the limit of small shear pressure but
is sufficiently precise.

The bulk part of the transformation amounts to an overall scaling of the
momentum by the factor λ\ :sub:`bulk`.  Rather than use an approximation (as
with shear), the scale factor is determined purely numerically to reproduce the
target bulk pressure.  The distribution function is modified as

.. math:: f(p) \rightarrow z_\text{bulk} f(p + \lambda_\text{bulk} p)

where z\ :sub:`bulk` is an effective "fugacity" that scales the overall particle
density.  The two bulk parameters are determined by the requirements that the
bulk pressure is reproduced without changing the equilibrium energy density.

In the code, the bulk parameters are called ``nscale`` and ``pscale`` (for
"density" and "momentum" scale).  The function `HRG.bulk_scale_factors` provides
an interface to calculate them.

This bulk correction method works all the way down to bulk pressure equal to
negative the equilibrium pressure, i.e. zero total pressure, in which case all
particles have zero momentum in the rest frame and all energy is rest mass.
However, the method fails for large positive bulk pressure because the momentum
scale factor diverges, so it is truncated at a reasonable value.  In realistic
heavy-ion collision events, very few volume elements have such large positive
bulk pressure, so practically this doesn't matter.  The function `HRG.Pi_lim`
returns the minimum and maximum values of bulk pressure.
