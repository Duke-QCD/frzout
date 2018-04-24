Usage
=====

Sampling particles
------------------

Computing hadron gas quantities
-------------------------------

Reference
---------

Main classes
^^^^^^^^^^^^
.. autoclass:: frzout.Surface

   **Class attributes**

   .. autoattribute:: boost_invariant
   .. autoattribute:: volume

.. autoclass:: frzout.HRG

   **Thermodynamic quantities**

   The following methods compute thermodynamic quantities of the HRG, given its
   temperature, composition, and if resonance width is enabled.

   .. automethod:: density
   .. automethod:: energy_density
   .. automethod:: pressure
   .. automethod:: entropy_density
   .. automethod:: mean_momentum
   .. automethod:: cs2
   .. automethod:: eta_over_tau
   .. automethod:: zeta_over_tau

   **Bulk viscous corrections**

   The following methods relate to the parametric bulk correction method (see
   :ref:`viscous-corrections`).

   .. automethod:: bulk_scale_factors
   .. automethod:: Pi_lim

Sampling function
^^^^^^^^^^^^^^^^^
.. autofunction:: frzout.sample

Species information
^^^^^^^^^^^^^^^^^^^
Species information is read from the table
http://pdg.lbl.gov/2017/mcdata/mass_width_2017.mcd, which is updated annually by
the Particle Data Group (PDG).  The file is included with frzout.

Particle IDs follow the PDG numbering scheme
(http://pdg.lbl.gov/2017/reviews/rpp2017-rev-monte-carlo-numbering.pdf).

.. data:: frzout.species_dict

   Dictionary of species data.

   The keys are particle IDs; values are subdicts containing the following keys:

   - ``'name'`` -- name of the species
   - ``'mass'`` -- mass in GeV
   - ``'width'`` -- width in GeV
   - ``'degen'`` -- spin degeneracy
   - ``'boson'`` -- whether the species is a boson or fermion
   - ``'charge'`` -- electric charge
   - ``'has_anti'`` -- whether or not the species has a corresponding antiparticle

   For example, ``frzout.species_dict[211]['mass']`` is the mass of a charged
   pion (ID = 211).

   Use ``list(frzout.species_dict)`` to obtain the list of all particle IDs.

.. currentmodule:: frzout.species

.. data:: identified

   List of IDs of the standard identified particles (pions, kaons, protons).

.. data:: urqmd

   List of particle IDs in UrQMD.
