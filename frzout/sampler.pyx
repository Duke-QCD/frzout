# cython: boundscheck = False, wraparound = False, initializedcheck = False

import numpy as np

from .species import species_dict, _normalize_species

cimport numpy as np
from libc cimport math
from cpython.mem cimport PyMem_Malloc, PyMem_Realloc, PyMem_Free

from . cimport fourvec
from .fourvec cimport FourVector
from . cimport random
from .random cimport rand

cdef extern from "quadrature.h":
    size_t NQUADPTS_M, NQUADPTS_P
    struct QuadPoint:
        double x, w
    QuadPoint* quadpts_m
    QuadPoint* quadpts_p


__all__ = ['Sampler']


random.seed()


cdef struct SurfaceElem:
    FourVector x, sigma, u
    double vmax

cdef class Surface:
    """
    Represents a freeze-out hypersurface.
    Manages memory for an array of `SurfaceElem`.

    """
    cdef:
        SurfaceElem* data
        size_t n
        double total_volume
        int boost_invariant

    def __cinit__(self, double[:, :] x, double[:, :] sigma, double[:, :] v):
        self.n = x.shape[0]
        self.boost_invariant = 1

        self.data = <SurfaceElem*> PyMem_Malloc(self.n * sizeof(SurfaceElem))
        if not self.data:
            raise MemoryError()

        cdef:
            size_t i
            SurfaceElem* elem
            double tau, gamma, vx, vy, vz, volume

        self.total_volume = 0

        for i in range(self.n):
            elem = self.data + i

            tau = x[i, 0]
            elem.x.t = tau
            elem.x.x = x[i, 1]
            elem.x.y = x[i, 2]
            elem.x.z = 0

            # TODO rapidity range
            elem.sigma.t = tau*sigma[i, 0]
            elem.sigma.x = tau*sigma[i, 1]
            elem.sigma.y = tau*sigma[i, 2]
            elem.sigma.z = 0

            vx = v[i, 0]
            vy = v[i, 1]
            vz = 0
            gamma = 1/math.sqrt(1 - vx*vx - vy*vy - vz*vz)
            elem.u.t = gamma
            elem.u.x = gamma*vx
            elem.u.y = gamma*vy
            elem.u.z = gamma*vz

            volume = fourvec.dot(&elem.sigma, &elem.u)
            self.total_volume += volume
            elem.vmax = (
                math.fabs(volume) +
                math.sqrt(math.fabs(
                    volume**2 - fourvec.square(&elem.sigma)
                ))
            )

    def __dealloc__(self):
        if self.data:
            PyMem_Free(self.data)


cdef struct SpeciesInfo:
    # PDG code
    int ID
    # spin degeneracy
    int degen
    # +1 fermions, -1 bosons
    int sign
    # whether this species is stable (zero width) or a resonance (finite width)
    int stable
    # Breit-Wigner params and mass thresholds [GeV]
    double m0, width, m_min, m_max
    # normalization constant: 1/(integral of BW distribution)
    double bw_norm
    # cached calculations for use in BW inverse CDF
    double atan_min, atan_max
    # number density [fm^-3]
    double density
    # scale factors for momentum sampling
    # see init_species() and sample_four_momentum()
    double xscale, Pscale


cdef void init_species(
    SpeciesInfo* s, int ID, dict info, bint res_width, double T
):
    """
    Initialize a SpeciesInfo object with the given ID and properties from the
    Python dict `info`.  Prepare for Breit-Wigner mass sampling if `res_width`
    is true.  Use temperature `T` for computing thermodynamic quantities.

    """
    s.ID    = ID
    s.degen = info['degen']
    s.sign  = -1 if info['boson'] else 1
    s.m0    = info['mass']

    if res_width and 'mass_range' in info:
        s.stable   = 0
        s.width    = info['width']
        s.m_min    = info['mass_range'][0]
        s.m_max    = info['mass_range'][1]
        s.atan_min = math.atan(2*(s.m0 - s.m_min)/s.width)
        s.atan_max = math.atan(2*(s.m_max - s.m0)/s.width)
        s.bw_norm = 1/integrate_bw(s)
    else:
        s.stable   = 1

    s.density = integrate_species(s, T, DENSITY)

    cdef:
        double xmaxsq
        double z = (s.m0 if s.stable else s.m_min)/T

    # Set scale factors for momentum sampling, see also sample_four_momentum().
    # Use rescaling for all but the smallest masses.
    if z > 1.3:
        # Compute argmax of the Boltzmann dist x^2*exp(-sqrt(z^2 + x^2)).
        # No closed form for the argmax of Bose/Fermi dists.
        xmaxsq = 2*(1 + math.sqrt(1 + z*z))
        # The argmax of the envelope x^2*exp(-x) is x = 2.
        # Rescale x so that it peaks at the desired xmax.
        s.xscale = math.sqrt(xmaxsq)/2
        # Now rescale P so that it equals one (or just below one) at xmax.
        # Since xmax is approximate and the Bose/Fermi dists have slightly
        # different shapes as Boltzmann, the envelope can fall slightly below
        # the target.  The extra factor (1 + .5*exp(-2*z)) ensures the envelope
        # remains above the target for z > 1.26.
        s.Pscale = (
            math.exp(-2) * (math.exp(math.sqrt(xmaxsq + z*z)) + s.sign) /
            (1 + .5*math.exp(-2*z))
        )
    elif z < .86 and s.sign == -1:
        # There is one more wrinkle for very light bosons (z < .855): the
        # envelope briefly falls below the target near x ~ 1.  This extra
        # factor extends the range to z ~ .7, which is the z of the lightest
        # species (pi0) at T ~ 192 MeV.  So this works for any reasonable
        # particlization temperature.
        s.xscale = 1
        s.Pscale = 1/(1 + 1.3*(.86 - z))
    else:
        s.xscale = 1
        s.Pscale = 1


cdef int equiv_species(const SpeciesInfo* a, const SpeciesInfo* b) nogil:
    """
    Determine if two species are equivalent for the purposes of phase-space
    integrals.  This is the case for multiplets and anti/particle pairs.

    """
    return (
        a.degen == b.degen and
        a.sign == b.sign and
        math.fabs(a.m0 - b.m0) < 1e-6 and (
            (a.stable and b.stable) or
            (
                not a.stable and not b.stable and
                math.fabs(a.width - b.width) < 1e-6 and
                math.fabs(a.m_min - b.m_min) < 1e-6 and
                math.fabs(a.m_max - b.m_max) < 1e-6
            )
        )
    )


cdef inline double _bw_dist(double m0, double w, double m) nogil:
    """
    Evaluate the *unnormalized* Breit-Wigner distribution at mass `m` with pole
    mass `m0` and width `w`.  For internal use by bw_dist() and bw_accept_prob().

    """
    return w/((m - m0)**2 + .25*w*w)


cdef inline double bw_dist(const SpeciesInfo* s, double m) nogil:
    """
    Evaluate the *unnormalized* Breit-Wigner distribution with mass-dependent
    width.  It is assumed (NOT checked) that m > m_min.

    """
    cdef double w = s.width * math.sqrt((m - s.m_min)/(s.m0 - s.m_min))
    return _bw_dist(s.m0, w, m)


cdef inline double bw_accept_prob(const SpeciesInfo* s, double m) nogil:
    """
    Compute an acceptance probability for sampling bw_dist().

    """
    # Use the constant-width distribution as an envelope.  The ratio is less
    # than one (or perhaps slightly above one) except in the high-mass tail.
    # Hence, divide by a constant factor to keep the ratio below one somewhat
    # further into the tail.
    cdef double ratio = bw_dist(s, m) / _bw_dist(s.m0, s.width, m)
    return math.fmin(ratio/1.2, 1)


cdef inline double bw_icdf(const SpeciesInfo* s, double q) nogil:
    """
    Evaluate the inverse CDF of a Breit-Wigner distribution with constant
    width.  Use bw_accept_prob() for sampling bw_dist().

    """
    return s.m0 + .5*s.width*math.tan((q-1)*s.atan_min + q*s.atan_max)


cdef double integrate_bw(const SpeciesInfo* s) nogil:
    """
    Integrate the Breit-Wigner distribution for the given species.

    """
    cdef:
        double total = 0
        double dm = s.m_max - s.m_min
        size_t i

    for i in range(NQUADPTS_M):
        total += quadpts_m[i].w * (
            bw_dist(s, s.m_min + dm*quadpts_m[i].x) +
            bw_dist(s, s.m_min + dm*(1 - quadpts_m[i].x))
        )

    return dm * total


cdef enum IntegralType:
    DENSITY
    ENERGY_DENSITY
    PRESSURE

cdef double integrate_species_stable(
    double m, int sign, double T, IntegralType integral
) nogil:
    """
    Integrate over momentum, for stable species with zero mass width:

        \int d^3p g(p) f(p)

    where f(p) is the distribution function (Bose-Einstein or Fermi-Dirac) and
    g(p) depends on the quantity to compute (energy density, pressure, etc).

    Some of these integrals may be written as infinite sums of Bessel functions
    and therefore computed by a truncated series.  That works fine, but while
    Bessel functions look simple on paper, they still require some effort to
    evaluate numerically.

    This method (Gauss-Laguerre quadrature -- see quadrature.h for details)
    requires only 64 evaluations of a simple integrand and is extremely precise
    (typical relative error ~ 10^-14).  I benchmarked this algorithm against
    sums of Bessel functions and it was several times faster for the same
    accuracy.  In addition, it's trivial to integrate any g(p), while in the
    Bessel function approach a series must be derived for each integrand.

    """
    cdef:
        double total = 0
        double zsq = (m/T)**2
        double xsq
        double E_over_T
        double g
        size_t i

    for i in range(NQUADPTS_P):
        # x = p/T from quadrature table
        xsq = quadpts_p[i].x**2
        E_over_T = math.sqrt(zsq + xsq)

        if integral == DENSITY:
            g = 1
        elif integral == ENERGY_DENSITY:
            g = T * E_over_T
        elif integral == PRESSURE:
            g = T * xsq/(3*E_over_T)
        else:
            return 0

        total += quadpts_p[i].w * g / (math.exp(E_over_T) + sign)

    # Multiply by T^3 to account for the change of variables d^3p -> d^3x.
    # The quadrature weights already include all other prefactors.
    return T*T*T * total


cdef double integrate_species_unstable(
    const SpeciesInfo* s, double T, IntegralType integral
) nogil:
    """
    Integrate over mass and momentum, for unstable resonances with finite mass
    width:

        \int_{m_min}^{m_max} dm bw(m) \int d^3p g(m, p) f(m, p)

    where (m_min, m_max) are the mass thresholds for the given species and the
    inner momentum integral is integrate_species_stable().

    """
    cdef:
        double total = 0
        double dm = s.m_max - s.m_min
        double m1, m2
        size_t i

    for i in range(NQUADPTS_M):
        m1 = s.m_min + dm*quadpts_m[i].x
        m2 = s.m_min + dm*(1 - quadpts_m[i].x)

        total += quadpts_m[i].w * (
            bw_dist(s, m1)*integrate_species_stable(m1, s.sign, T, integral) +
            bw_dist(s, m2)*integrate_species_stable(m2, s.sign, T, integral)
        )

    return s.bw_norm * dm * total


cdef double integrate_species(
    const SpeciesInfo* s, double T, IntegralType integral
) nogil:
    """
    Compute a phase-space integral for the given species and temperature.

    """
    cdef double I

    if s.stable:
        I = integrate_species_stable(s.m0, s.sign, T, integral)
    else:
        I = integrate_species_unstable(s, T, integral)

    return s.degen * I


cdef void sample_four_momentum(
    const SpeciesInfo* s, double T, FourVector* p
) nogil:
    """
    Choose a random four-momentum, with mass either constant (for stable
    species) or sampled from the Breit-Wigner distribution (resonances),
    momentum magnitude sampled from the the Bose-Einstein or Fermi-Dirac
    distribution:

        p ~ p^2 / (exp(sqrt(m^2 + p^2)/T) +/- 1),

    and momentum direction sampled isotropically.

    For momentum sampling, use a massless Boltzmann distribution x^2*exp(-x)
    (aka gamma or erlang dist with k = 3) as an envelope for the Bose/Fermi
    dists.  This is efficient for small mass (z = m/T ~ 1) but very
    innefficient for large mass.  In typical Cooper-Frye sampling all species
    except the pion have masses much larger than the temperature.

    To improve efficiency, rescale x = p/T by a constant factor (xscale) and
    the envelope dist by another factor (Pscale) so that the envelope peaks at
    the same point as the target dist.  This is commensurate with using a
    larger effective temperature so that the envelope has a longer tail than
    the target and hence remains above it for all x (provided Pscale is
    well-chosen).  These scale factors are pre-computed in init_species().

    Efficiency is ~90% for small masses (z ~ 1) decreasing to ~70% for large
    masses (z ~ 10).

    """
    cdef double m, z, r, x, P

    while True:
        if s.stable:
            m = s.m0
            P = 1
        else:
            # sample proposal mass from Breit-Wigner
            m = bw_icdf(s, rand())
            P = bw_accept_prob(s, m)

        # sample proposal x from envelope x^2*exp(-x/xscale)
        r = (1 - rand())*(1 - rand())*(1 - rand())
        x = -s.xscale*math.log(r)

        # acceptance probability
        z = m/T
        P *= s.Pscale / r / (math.exp(math.sqrt(x*x + z*z)) + s.sign)

        if rand() < P:
            break

    cdef:
        # change variables, x = p/T
        double pmag = x*T
        # sample 3D direction
        double cos_theta = 2*rand() - 1
        double sin_theta = math.sqrt(1 - cos_theta*cos_theta)
        double phi = 6.28318530717958648*rand()  # 2*pi

    p.t = T*math.sqrt(x*x + z*z)
    p.x = pmag*sin_theta*math.cos(phi)
    p.y = pmag*sin_theta*math.sin(phi)
    p.z = pmag*cos_theta


cdef class HRG:
    """
    A hadron resonance gas, i.e. a set of particle species.
    Manages memory for an array of `SpeciesInfo`.

    """
    cdef:
        SpeciesInfo* data
        size_t n
        double T
        double total_density

    def __cinit__(self, double T, object species='all', bint res_width=True):
        self.T = T

        cdef list species_items = _normalize_species(species)
        self.n = len(species_items)

        self.data = <SpeciesInfo*> PyMem_Malloc(self.n * sizeof(SpeciesInfo))
        if not self.data:
            raise MemoryError()

        self.total_density = 0

        cdef:
            size_t i
            int ID
            dict info

        for i in range(self.n):
            ID, info = species_items[i]
            init_species(self.data + i, ID, info, res_width, T)
            self.total_density += self.data[i].density

    def __dealloc__(self):
        if self.data:
            PyMem_Free(self.data)

    cdef double _sum_integrals(self, IntegralType integral):
        """
        Compute the sum of the given phase-space integral over all the species
        in this HRG.

        """
        cdef:
            double total, last
            size_t i

        with nogil:
            total = last = integrate_species(self.data, self.T, integral)
            for i in range(1, self.n):
                # reuse previous calculation if possible
                if not equiv_species(self.data + i, self.data + i - 1):
                    last = integrate_species(self.data + i, self.T, integral)
                total += last

        return total

    def density(self):
        """
        Particle density [fm^-3].

        """
        return self._sum_integrals(DENSITY)

    def energy_density(self):
        """
        Energy density [GeV/fm^-3].

        """
        return self._sum_integrals(ENERGY_DENSITY)

    def pressure(self):
        """
        Pressure [GeV/fm^-3].

        """
        return self._sum_integrals(PRESSURE)


cdef struct Particle:
    int ID
    FourVector x, p

cdef class ParticleList:
    """
    Represents an ensemble of produced particles.
    Manages memory for an array of `Particle`.

    Unlike `Surface` and `HRG`, a `ParticleList` may change size as more
    particles are produced.  The following functions `extend_particle_list` and
    `add_particle` are non-members to allow inlining (Cython calls member
    functions through a virtual table, so inlining them is usually impossible).

    """
    cdef:
        Particle* data
        size_t n, capacity

    def __cinit__(self, double navg):
        self.n = 0
        # Guess initial capacity based on Poissonian particle production:
        # average number of particles plus three standard deviations.
        self.capacity = <size_t> max(navg + 3*math.sqrt(navg), 10)

        self.data = <Particle*> PyMem_Malloc(self.capacity * sizeof(Particle))
        if not self.data:
            raise MemoryError()

    def __dealloc__(self):
        if self.data:
            PyMem_Free(self.data)


cdef inline void extend_particle_list(ParticleList particles) with gil:
    """
    Increase the capacity of a `ParticleList`.

    """
    cdef:
        # Again, resize based on the Poisson distribution:
        # add (roughly) another standard deviation to the capacity.
        size_t new_capacity = (
            particles.capacity + <size_t>(math.sqrt(particles.capacity))
        )
        Particle* new_data = <Particle*> PyMem_Realloc(
            particles.data, new_capacity * sizeof(Particle)
        )

    if not new_data:
        raise MemoryError()

    particles.data = new_data
    particles.capacity = new_capacity


cdef inline void add_particle(
    const SpeciesInfo* species,
    const FourVector* x,
    const FourVector* p,
    ParticleList particles
) nogil:
    """
    Append a new particle to a `ParticleList`, extending capacity if necessary.

    """
    if particles.n == particles.capacity:
        extend_particle_list(particles)

    cdef Particle* part = particles.data + particles.n
    part.ID = species.ID
    part.x = x[0]
    part.p = p[0]

    particles.n += 1


cdef class Sampler:
    """
    Main sampler class.

    """
    cdef:
        Surface surface
        HRG hrg
        ParticleList particles

    def __cinit__(self, x, sigma, v, T):
        self.surface = Surface(np.asarray(x), np.asarray(sigma), np.asarray(v))
        self.hrg = HRG(T)
        self.particles = ParticleList(
            self.surface.total_volume * self.hrg.total_density
        )

    def sample(self):
        """
        Sample the surface once and return an array of particle data.

        """
        with nogil:
            self._sample()

        return np.asarray(<Particle[:self.particles.n]> self.particles.data)

    cdef void _sample(self) nogil:
        """
        Perform sampling.

        """
        cdef:
            double N, new_N
            double p_dot_sigma, p_dot_sigma_max
            double y, y_minus_eta_s
            double eta_s, cosh_eta_s, sinh_eta_s
            double pt_prime
            size_t ielem, ispecies
            SurfaceElem* elem
            SpeciesInfo* species
            FourVector x, p

        # reset particle list
        self.particles.n = 0

        N = math.log(1 - rand())

        for ielem in range(self.surface.n):
            elem = self.surface.data + ielem

            new_N = N + elem.vmax*self.hrg.total_density
            if new_N < 0:
                N = new_N
                continue

            for ispecies in range(self.hrg.n):
                species = self.hrg.data + ispecies
                N += elem.vmax*species.density
                while N > 0:
                    # adding a negative number
                    N += math.log(1 - rand())

                    sample_four_momentum(species, self.hrg.T, &p)
                    fourvec.boost_inverse(&p, &elem.u)

                    p_dot_sigma = fourvec.dot(&p, &elem.sigma)
                    if p_dot_sigma < 0:
                        continue

                    p_dot_sigma_max = elem.vmax*fourvec.dot(&p, &elem.u)

                    if p_dot_sigma > p_dot_sigma_max*rand():
                        if self.surface.boost_invariant:
                            y_minus_eta_s = .5*math.log((p.t + p.z)/(p.t - p.z))
                            y = rand() - .5
                            eta_s = y - y_minus_eta_s
                            cosh_eta_s = math.cosh(eta_s)
                            sinh_eta_s = math.sinh(eta_s)

                            x.t = elem.x.t*cosh_eta_s
                            x.x = elem.x.x
                            x.y = elem.x.y
                            x.z = elem.x.t*sinh_eta_s

                            pt_prime = p.t
                            p.t = pt_prime*cosh_eta_s + p.z*sinh_eta_s
                            p.z = pt_prime*sinh_eta_s + p.z*cosh_eta_s
                        else:
                            x = elem.x

                        add_particle(species, &x, &p, self.particles)
