Usage
=====
frzout is a particlization model (Cooper-Frye sampler) for relativistic
heavy-ion collisions.  Its primary purpose is to sample particles in a
multistage collision model, after hydrodynamics and before a hadronic
afterburner.  It can also be used to compute hadron resonance gas thermodynamic
quantities (e.g. to calculate an equation of state).

Physics references:

- `The original paper <https://journals.aps.org/prd/abstract/10.1103/PhysRevD.10.186>`_ by Cooper & Frye
- `"Particlization in hybrid models" <https://inspirehep.net/record/1118440>`_ by Huovinen & Petersen
- `The author's dissertation <https://inspirehep.net/record/1669345>`_, section 3.4

Sampling particles
------------------
In typical heavy-ion collision models, the hydro code outputs a spacetime
hypersurface as a collection of finite volume elements.  The surface is usually
defined by a switching temperature.

The basic procedure for sampling particles:

1. Create a `Surface` object from the data output by the hydro code.
2. Create an `HRG` object, representing the hadron resonance gas from which particles will be sampled.
3. Call the `sample` function.

1. Create a Surface
^^^^^^^^^^^^^^^^^^^
Suppose a 3D ideal hydro code outputs surface data as a text table with the following columns::

   Column index:  0  1  2  3  4        5        6        7        8   9   10
   Quantity:      t  x  y  z  sigma_t  sigma_x  sigma_y  sigma_z  vx  vy  vz

The first four columns (0--3) are the components of the spacetime position
vector, the next four (4--7) are the covariant normal vector σ\ :sub:`μ`, and
the last three (8--10) are the velocity.

Each row of the table represents a volume element.

We will now load the data into arrays and create a `Surface` object:

.. code-block:: python

   import numpy as np
   import frzout

   # read surface data from text file
   surface_data = np.loadtxt('surface.dat')

   # separate into sub-arrays
   # (there are of course other ways to do this)
   x, sigma, v = np.hsplit(surface_data, [4, 8])

   # create Surface object
   surface = frzout.Surface(x, sigma, v)

.. note::

   The components of the vectors must be in standard order (t, x, y, z).  If the
   surface data columns are in a different order, they must be rearranged in the
   arrays *x*, *sigma*, *v*.

   The units of the spacetime position must be fm.

   The units of the normal vector must be fm\ :sup:`3`.  It must be covariant,
   i.e. σ\ :sub:`μ`, not σ\ :sup:`μ`.  Most surface finding algorithms output
   the covariant vector.  If for some reason it is contravariant, simply negate
   the spatial components.

   The velocity must be dimensionless (relative to the speed of light).  It is
   the three-vector **v**, *not* the four-velocity u\ :sup:`μ`.

Boost invariance
""""""""""""""""
Suppose a 2D (boost-invariant) ideal hydro code outputs a similar format, but
without the z components::

   Column index:  0  1  2  3        4        5        6   7
   Quantity:      t  x  y  sigma_t  sigma_x  sigma_y  vx  vy

Analogously, we can load the data as:

.. code-block:: python

   x, sigma, v = np.hsplit(np.loadtxt('surface.dat'), [3, 6])
   surface = frzout.Surface(x, sigma, v, ymax=1)

The `Surface` class infers whether the surface is 2D (boost-invariant) or 3D
from the shapes of the input arrays.  After creating the object, attribute
``surface.boost_invariant`` is a boolean indicating whether the surface is
boost-invariant or not (2D or 3D).

Parameter *ymax* sets the maximum momentum rapidity: The *dN/dy* of sampled
particles will be flat from −\ *ymax* to +\ *ymax*.  The default value of *ymax*
is 0.5.  It has no effect for 3D surfaces.

.. note::

   The units of the normal vector are formally fm\ :sup:`2` for boost-invariant
   surfaces.  This is what most surface finding algorithms output.  The full
   differential volume element is

   .. math:: d^3\sigma_\mu = \tau \, \Delta y \, d^2\sigma_\mu

   The `Surface` class internally scales 2D normal vectors by 2 × *ymax* × τ.

Viscous pressures
"""""""""""""""""
If the hydro code is viscous, we must also load the viscous pressures and input
them to the `Surface` class.

Suppose we are using
`the author's version of the OSU viscous hydro code <https://github.com/jbernhard/osu-hydro>`_,
which outputs surface data in the format::

   Column index:  0  1  2  3        4        5        6   7
   Quantity:      t  x  y  sigma_t  sigma_x  sigma_y  vx  vy

   Column index:  8      9      10     11     12     13     14     15
   Quantity:      pi^tt  pi^tx  pi^ty  pi^xx  pi^xy  pi^yy  pi^zz  Pi

The first 8 columns are the same as above, columns 8--14 are the components of
the shear pressure tensor π\ :sup:`μν`, and column 15 is the bulk pressure Π.

We can load the data as follows:

.. code-block:: python

   # data file is binary, not text
   surface_data = np.fromfile('surface.dat', dtype='float64').reshape(-1, 16)

   # extract usual sub-arrays
   x, sigma, v, _ = np.hsplit(surface_data, [3, 6, 8])

   # create mapping of pi components
   pi = dict(
      xx=surface_data[:, 11],
      xy=surface_data[:, 12],
      yy=surface_data[:, 13]
   )

   # extract bulk pressure
   Pi = surface_data[:, 15]

   # create Surface object
   surface = frzout.Surface(x, sigma, v, pi=pi, Pi=Pi)

As shown in the example, the shear tensor *pi* must be provided as a mapping
(dict-like object).

- For 2D surfaces, only components (xx, xy, yy) are required.
- For 3D surfaces, components (xz, yz) are additionally required.

The remaining components are computed internally by using that the shear tensor
is traceless and orthogonal to the velocity.

When *pi* and/or *Pi* are provided, viscous corrections are applied to
particle sampling.  See `viscous-corrections` for more information.

.. note::

   The units of the viscous pressures must be GeV/fm\ :sup:`3`.

.. warning::

   Surface data files can be very large, especially for central events and/or 3D
   hydro.  The `Surface` class stores a copy of the data internally, effectively
   doubling memory usage.  Loading data and creating a `Surface` may exceed
   system memory.  Currently, the only way around this is to break the surface
   into smaller chunks and load them one at a time.

   One way to ensure that the extra memory is freed after creating a `Surface`
   is to write a function that loads the data and returns the object:

   .. code-block:: python

      def load_surface():
         x, sigma, v = ...
         return frzout.Surface(x, sigma, v, ...)

   The local variables will go out of scope when the function returns and their
   memory will be released.

   Text files exacerbate the problem, since typical methods of reading text
   files in Python (such as *np.loadtxt*) create a temporary list which uses
   even more memory than the final array.  Further, reading and writing text
   files is slow.  I recommend using a binary format if possible.  The example
   above shows how to read the binary file output by
   `osu-hydro <https://github.com/jbernhard/osu-hydro>`_.


2. Create an HRG
^^^^^^^^^^^^^^^^
The `HRG` class represents a hadron resonance gas.  The only required parameter
is the temperature of the gas, which should be set to the switching temperature
of the hydro surface.  For example, to create an `HRG` with temperature 150 MeV:

.. code-block:: python

   hrg = frzout.HRG(.150)  # temperature units are GeV

Optional parameters control the composition of the gas and the behavior of
resonances.  See the reference section (below) for details, and `res-mass-dist`
for information on the physical treatment of resonances.

Examples of creating `HRG` with more options:

.. code-block:: python

   # for use with UrQMD
   hrg = frzout.HRG(.150, species='urqmd', res_width=True)

   # a pion gas
   hrg = frzout.HRG(.150, species=[111, 211])

.. tip::

   It is possible, and in fact advisable, to reuse an `HRG` object for multiple
   `Surface` (for example if running several events in a row, all with the same
   particlization temperature).  There are some preliminary calculations in
   constructing an `HRG` that are not repeated when the object is reused.

3. Call the sample function
^^^^^^^^^^^^^^^^^^^^^^^^^^^
After creating a `Surface` and `HRG`, sample particles by:

.. code-block:: python

   parts = frzout.sample(surface, hrg)

The function returns a
`numpy structured array <https://docs.scipy.org/doc/numpy/user/basics.rec.html>`_
with fields ``'ID'``, ``'x'``, ``'p'``, containing the ID number, spacetime
position four-vector (t, x, y, z), and energy-momentum four-vector
(E, p\ :sub:`x`, p\ :sub:`y`, p\ :sub:`z`), respectively, for each sampled
particle.  These can be extracted with a dict-like interface:

.. code-block:: python

   ID = parts['ID']  # 1D array of integers, shape (nparts,)
   x = parts['x']  # 2D array of floats, shape (nparts, 4)
   p = parts['p']  # 2D array of floats, shape (nparts, 4)

The particle array can be converted to a
`record array <https://docs.scipy.org/doc/numpy/user/basics.rec.html#record-arrays>`_
so that the fields are accessible by attribute rather than dict key:

.. code-block:: python

   parts = frzout.sample(surface, hrg).view(np.recarray)
   x = parts.x
   # etc...

We can loop over particles:

.. code-block:: python

   for p in parts:
      x = p['x']  # a single particle's position vector
      # ...

   # or unpack every particle's attributes
   for ID, x, p in parts:
      # ...

Examples of oversampling a surface:

.. code-block:: python

   # use samples one at a time
   for _ in range(nsamples):
      parts = frzout.sample(surface, hrg)
      # do something with particles

   # alternatively, to keep all samples at once
   # (this could use a lot of memory!)
   samples = [frzout.sample(surface, hrg) for _ in range(nsamples)]

The number of sampled particles is Poisson distributed.  The average number may
be computed as ``surface.volume * hrg.density()``.  Note that if the surface has
any negative contributions (most physical surfaces do), the actual average will
be somewhat larger than the computed average.  The number of particles in a
particular sample is given by ``parts.size``.

Computing hadron gas quantities
-------------------------------
The `HRG` class can also be used to compute thermodynamic quantities of a hadron
resonance gas, e.g. to calculate the equation of state.  For example, to compute
the energy density of a hadron gas at 150 MeV:

.. code-block:: python

   frzout.HRG(.15).energy_density()

The calculations use the actual composition of the gas and take resonance width
into account:

.. code-block:: python

   # pressure of 150 MeV gas of UrQMD species
   frzout.HRG(.15, species='urqmd').pressure()

   # same, but neglecting resonance width (result will be slightly different)
   frzout.HRG(.15, species='urqmd', res_width=False).pressure()

See the reference section for all available quantities.

By calculating energy density, pressure, etc over a range of temperatures, we
can construct an HRG equation of state.  The script `eos.py in the osu-hydro
repository <https://github.com/jbernhard/osu-hydro/blob/master/eos/eos.py>`_
does this and connects the result to the HotQCD lattice equation of state at
high temperature.

Note, the `HRG` class emits a warning about inaccurate momentum sampling for
high temperatures.  This may be safely ignored when only computing thermodynamic
quantities.

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
   temperature, composition, and whether or not resonance width is enabled.

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
   `viscous-corrections`).

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
