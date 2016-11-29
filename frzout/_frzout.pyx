# cython: boundscheck = False, wraparound = False, initializedcheck = False

import warnings

import numpy as np

from .species import species_dict, _normalize_species

from libc cimport math
from cpython.mem cimport PyMem_Malloc, PyMem_Realloc, PyMem_Free
from cpython.buffer cimport Py_buffer, PyBUF_FORMAT

from . cimport fourvec
from .fourvec cimport FourVector
from .random cimport seed_rand, rand

cdef extern from "quadrature.h":
    size_t NQUADPTS_M, NQUADPTS_P
    struct QuadPoint:
        double x, w
    QuadPoint* quadpts_m
    QuadPoint* quadpts_p


__all__ = ['Surface', 'HRG', 'sample']


seed_rand()

DEF TWO_PI = 6.28318530717958648


cdef struct ShearTensor:
    double xx, yy, zz, xy, xz, yz

cdef struct SurfaceElem:
    FourVector x, sigma, u
    double vmax
    ShearTensor pi
    double Pi

cdef class Surface:
    """
    Represents a freeze-out hypersurface.
    Manages memory for an array of `SurfaceElem`.

    """
    cdef:
        SurfaceElem* data
        size_t n
        double total_volume
        double ymax
        readonly bint boost_invariant
        int shear, bulk

    def __cinit__(
            self,
            double[:, :] x not None,
            double[:, :] sigma not None,
            double[:, :] v not None,
            object pi=None,
            double[:] Pi=None,
            double ymax=.5
    ):
        self.n = x.shape[0]

        # TODO more informative error messages below

        if x.shape[1] == 4 and sigma.shape[1] == 4 and v.shape[1] == 3:
            self.boost_invariant = 0
            if ymax != .5:
                warnings.warn('ymax has no effect for 3D surfaces')
        elif x.shape[1] == 3 and sigma.shape[1] == 3 and v.shape[1] == 2:
            self.boost_invariant = 1
        else:
            raise ValueError('invalid shape')

        if sigma.shape[0] != self.n or v.shape[0] != self.n:
            raise ValueError('invalid shape')

        cdef double[:] pixx, piyy, pixy, pixz, piyz

        self.shear = (pi is not None)
        if self.shear:
            pixx = pi['xx']
            piyy = pi['yy']
            pixy = pi['xy']
            if not (pixx.shape[0] == piyy.shape[0] == pixy.shape[0] == self.n):
                raise ValueError('invalid shape')
            if self.boost_invariant:
                if 'xz' in pi or 'yz' in pi:
                    warnings.warn(
                        '(xz, yz) components of pi '
                        'have no effect for 2D surfaces'
                    )
            else:
                pixz = pi['xz']
                piyz = pi['yz']
                if not (pixz.shape[0] == piyz.shape[0] == self.n):
                    raise ValueError('invalid shape')

        self.bulk = (Pi is not None)
        if self.bulk:
            if Pi.shape[0] != self.n:
                raise ValueError('invalid shape')

        self.data = <SurfaceElem*> PyMem_Malloc(self.n * sizeof(SurfaceElem))
        if not self.data:
            raise MemoryError()

        cdef:
            size_t i
            SurfaceElem* elem
            double gamma, vx, vy, vz, volume, sigma_scale

        self.total_volume = 0
        self.ymax = ymax

        for i in range(self.n):
            elem = self.data + i

            # read in transverse data first
            elem.x.t = x[i, 0]
            elem.x.x = x[i, 1]
            elem.x.y = x[i, 2]

            elem.sigma.t = sigma[i, 0]
            elem.sigma.x = sigma[i, 1]
            elem.sigma.y = sigma[i, 2]

            vx = v[i, 0]
            vy = v[i, 1]

            # handle longitudinal direction depending on boost invariance
            if self.boost_invariant:
                elem.x.z = 0
                elem.sigma.z = 0
                vz = 0
                # scale dsigma by (2*ymax*tau)
                sigma_scale = 2*ymax*x[i, 0]
                elem.sigma.t *= sigma_scale
                elem.sigma.x *= sigma_scale
                elem.sigma.y *= sigma_scale
            else:
                elem.x.z = x[i, 3]
                elem.sigma.z = sigma[i, 3]
                vz = v[i, 2]

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

            if self.shear:
                elem.pi.xx = pixx[i]
                elem.pi.yy = piyy[i]
                elem.pi.xy = pixy[i]
                if self.boost_invariant:
                    elem.pi.xz = 0
                    elem.pi.yz = 0
                else:
                    elem.pi.xz = pixz[i]
                    elem.pi.yz = piyz[i]
                boost_pi_lrf(&elem.pi, &elem.u)
            else:
                elem.pi.xx = 0
                elem.pi.yy = 0
                elem.pi.zz = 0
                elem.pi.xy = 0
                elem.pi.xz = 0
                elem.pi.yz = 0

            elem.Pi = Pi[i] if self.bulk else 0

    def __dealloc__(self):
        PyMem_Free(self.data)

    def __len__(self):
        return self.n

    property volume:
        """
        Total freeze-out volume.

        """
        def __get__(self):
            return self.total_volume


cdef void boost_pi_lrf(ShearTensor* pi, const FourVector* u) nogil:
    """
    Boost shear tensor `pi` with flow velocity `u` to its local rest frame.
    `pi` must have its (xx, yy, xy, xz, yz) components set; the remaining
    components are determined because pi must is traceless and orthogonal to
    the flow velocity.

    """
    cdef:
        double pi_array[4][4]
        double L[4][4]

    # Construct the full pi array from the given components.
    # pi is traceless
    #
    #   \pi^\mu_\mu = 0
    #
    # and orthogonal to the flow velocity
    #
    #   \pi^{\mu\nu} u_\mu = 0
    #
    # These five equations may be solved for components (tt, tx, ty, tz, zz) in
    # terms of the five given components and the flow velocity.
    pi_array[0][0] = (
        u.x*u.x*pi.xx + u.y*u.y*pi.yy - u.z*u.z*(pi.xx + pi.yy)
        + 2*u.x*u.y*pi.xy + 2*u.x*u.z*pi.xz + 2*u.y*u.z*pi.yz
    )/(u.t*u.t - u.z*u.z)

    pi.zz = pi_array[0][0] - pi.xx - pi.yy

    pi_array[1][1] = pi.xx
    pi_array[2][2] = pi.yy
    pi_array[3][3] = pi.zz

    pi_array[0][1] = pi_array[1][0] = (u.x*pi.xx + u.y*pi.xy + u.z*pi.xz)/u.t
    pi_array[0][2] = pi_array[2][0] = (u.x*pi.xy + u.y*pi.yy + u.z*pi.yz)/u.t
    pi_array[0][3] = pi_array[3][0] = (u.x*pi.xz + u.y*pi.yz + u.z*pi.zz)/u.t
    pi_array[1][2] = pi_array[2][1] = pi.xy
    pi_array[1][3] = pi_array[3][1] = pi.xz
    pi_array[2][3] = pi_array[3][2] = pi.yz

    # now construct boost matrix from flow velocity
    L[0][0] = u.t
    L[1][1] = 1 + u.x*u.x/(1 + u.t)
    L[2][2] = 1 + u.y*u.y/(1 + u.t)
    L[3][3] = 1 + u.z*u.z/(1 + u.t)

    L[0][1] = L[1][0] = -u.x
    L[0][2] = L[2][0] = -u.y
    L[0][3] = L[3][0] = -u.z
    L[1][2] = L[2][1] = u.x*u.y/(1 + u.t)
    L[1][3] = L[3][1] = u.x*u.z/(1 + u.t)
    L[2][3] = L[3][2] = u.y*u.z/(1 + u.t)

    # calculate boosted pi elements
    pi.xx = boost_tensor_elem(pi_array, L, 1, 1)
    pi.yy = boost_tensor_elem(pi_array, L, 2, 2)
    pi.zz = -pi.xx - pi.yy  # traceless
    pi.xy = boost_tensor_elem(pi_array, L, 1, 2)
    pi.xz = boost_tensor_elem(pi_array, L, 1, 3)
    pi.yz = boost_tensor_elem(pi_array, L, 2, 3)


cdef double boost_tensor_elem(
    const double[4][4] A, const double[4][4] L, int a, int b
) nogil:
    """
    Compute the (a, b) element of tensor A after boosting by L.

    """
    cdef:
        double total = 0
        int c, d

    for c in range(4):
        for d in range(4):
            total += L[a][c] * L[b][d] * A[c][d]

    return total


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
    # change in density and momentum density due to bulk viscosity
    # see HRG._prepare_bulk()
    double delta_density_Pi, rel_p_density_Pi
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
        s.bw_norm  = 1/integrate_bw(s)
    else:
        s.stable   = 1

    s.density = integrate_species(s, T, 0, DENSITY)
    s.delta_density_Pi = 0
    s.rel_p_density_Pi = 0

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
    MOMENTUM_DENSITY
    ENERGY_DENSITY
    PRESSURE
    CS2_NUMER
    CS2_DENOM
    ETA_OVER_TAU
    ZETA_OVER_TAU
    DELTA_DENSITY
    DELTA_MOMENTUM_DENSITY

cdef double integrate_species_stable(
    double m, int sign, double T, double cs2, IntegralType integral
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
        double p, psq
        double E, Esq
        double f0, g
        size_t i

    for i in range(NQUADPTS_P):
        # x = p/T from quadrature table
        p = quadpts_p[i].x * T
        psq = p*p
        Esq = m*m + psq
        E = math.sqrt(Esq)
        f0 = 1 / (math.exp(E/T) + sign)

        # compute inner function
        if integral == DENSITY:
            g = 1
        elif integral == MOMENTUM_DENSITY:
            g = p
        elif integral == ENERGY_DENSITY:
            g = E
        elif integral == PRESSURE:
            g = psq/(3*E)
        # remaining integrals have f0*(1 -/+ f0)
        elif integral == CS2_NUMER:
            g = psq/3 * (1 - sign*f0)
        elif integral == CS2_DENOM:
            g = Esq * (1 - sign*f0)
        elif integral == ETA_OVER_TAU:
            g = psq*psq/Esq / (15*T) * (1 - sign*f0)
        elif integral == ZETA_OVER_TAU:
            g = m*m / (3*T) * (cs2 - psq/(3*Esq)) * (1 - sign*f0)
        # These are proportional to the change in density and momentum density
        # due to the bulk delta-f.  They must be multiplied by Pi/(zeta/tau).
        elif integral == DELTA_DENSITY:
            g = (psq/(3*E) - cs2*E)/T * (1 - sign*f0)
        elif integral == DELTA_MOMENTUM_DENSITY:
            g = (psq/(3*E) - cs2*E)/T * (1 - sign*f0) * p
        else:
            return 0

        total += quadpts_p[i].w * g * f0

    # Multiply by T^3 to account for the change of variables d^3p -> d^3x.
    # The quadrature weights already include all other prefactors.
    return T*T*T * total


cdef double integrate_species_unstable(
    const SpeciesInfo* s, double T, double cs2, IntegralType integral
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
            bw_dist(s, m1)*integrate_species_stable(m1, s.sign, T, cs2, integral) +
            bw_dist(s, m2)*integrate_species_stable(m2, s.sign, T, cs2, integral)
        )

    return s.bw_norm * dm * total


cdef double integrate_species(
    const SpeciesInfo* s, double T, double cs2, IntegralType integral
) nogil:
    """
    Compute a phase-space integral for the given species and temperature.

    The speed of sound squared `cs2` is required for anything related to bulk
    viscosity, otherwise it is not used.

    """
    cdef double I

    if s.stable:
        I = integrate_species_stable(s.m0, s.sign, T, cs2, integral)
    else:
        I = integrate_species_unstable(s, T, cs2, integral)

    return s.degen * I


cdef void sample_four_momentum(
    const SpeciesInfo* s, double T,
    double shear_pscale, const ShearTensor* pi, double bulk_pscale,
    FourVector* p
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
    # note: lowercase p is momentum, capital P is probability
    cdef double m, r, pmag, P

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
        pmag = -T*s.xscale*math.log(r)

        # acceptance probability
        P *= s.Pscale / r / (math.exp(math.sqrt(m*m + pmag*pmag)/T) + s.sign)

        if rand() < P:
            break

    cdef:
        # sample 3D direction
        double rz = 2*rand() - 1  # cos(theta)
        double sin_theta = math.sqrt(1 - rz*rz)
        double phi = TWO_PI*rand()
        double rx = sin_theta*math.cos(phi)
        double ry = sin_theta*math.sin(phi)

    # compute 3D momentum vector with viscous corrections
    p.x = pmag*(bulk_pscale*rx + shear_pscale*(rx*pi.xx + ry*pi.xy + rz*pi.xz))
    p.y = pmag*(bulk_pscale*ry + shear_pscale*(rx*pi.xy + ry*pi.yy + rz*pi.yz))
    p.z = pmag*(bulk_pscale*rz + shear_pscale*(rx*pi.xz + ry*pi.yz + rz*pi.zz))

    # compute energy
    p.t = math.sqrt(m*m + p.x*p.x + p.y*p.y + p.z*p.z)


cdef class HRG:
    """
    A hadron resonance gas, i.e. a set of particle species.
    Manages memory for an array of `SpeciesInfo`.

    """
    cdef:
        SpeciesInfo* data
        size_t n
        readonly double T
        double total_density
        double delta_density_Pi
        double shear_pscale
        double Pi_min, Pi_max
        int decay_f500
        int shear_prepared, bulk_prepared

    def __cinit__(
            self, double T,
            object species='all', bint res_width=True, bint decay_f500=False
    ):
        self.T = T

        cdef list species_items = _normalize_species(species)
        self.n = len(species_items)

        self.data = <SpeciesInfo*> PyMem_Malloc(self.n * sizeof(SpeciesInfo))
        if not self.data:
            raise MemoryError()

        self.total_density = 0
        self.delta_density_Pi = 0
        self.shear_pscale = 0
        self.Pi_min = 0
        self.Pi_max = 0
        self.decay_f500 = (decay_f500 or species == 'urqmd')
        self.shear_prepared = 0
        self.bulk_prepared = 0

        cdef:
            size_t i
            int ID
            dict info

        for i in range(self.n):
            ID, info = species_items[i]
            init_species(self.data + i, ID, info, res_width, T)
            self.total_density += self.data[i].density

    def __dealloc__(self):
        PyMem_Free(self.data)

    def __len__(self):
        return self.n

    def _data(self):
        """
        Return a view of the internal species data.  Mainly for testing.

        """
        return np.asarray(<SpeciesInfo[:self.n]> self.data)

    cdef double _sum_integrals(self, IntegralType integral, double cs2=0):
        """
        Compute the sum of the given phase-space integral over all the species
        in this HRG.

        """
        cdef:
            double total, last
            size_t i

        with nogil:
            total = last = integrate_species(self.data, self.T, cs2, integral)
            for i in range(1, self.n):
                # reuse previous calculation if possible
                if not equiv_species(self.data + i, self.data + i - 1):
                    last = integrate_species(self.data + i, self.T, cs2, integral)
                total += last

        return total

    def density(self):
        """
        Particle density [fm^-3].

        """
        # return self._sum_integrals(DENSITY)
        return self.total_density

    cpdef double energy_density(self):
        """
        Energy density [GeV/fm^3].

        """
        return self._sum_integrals(ENERGY_DENSITY)

    cpdef double pressure(self):
        """
        Pressure [GeV/fm^3].

        """
        return self._sum_integrals(PRESSURE)

    def entropy_density(self):
        """
        Entropy density [fm^-3].

        """
        return (self.energy_density() + self.pressure()) / self.T

    def mean_momentum(self):
        """
        Average magnitude of momentum [GeV].

        """
        return self._sum_integrals(MOMENTUM_DENSITY) / self.total_density

    cpdef double cs2(self):
        """
        Speed of sound squared.

        """
        return self._sum_integrals(CS2_NUMER) / self._sum_integrals(CS2_DENOM)

    cpdef double eta_over_tau(self):
        """
        Shear viscosity over relaxation time [GeV/fm^3].

        """
        return self._sum_integrals(ETA_OVER_TAU)

    cpdef double zeta_over_tau(self):
        """
        Bulk viscosity over relaxation time [GeV/fm^3].

        """
        return self._sum_integrals(ZETA_OVER_TAU, cs2=self.cs2())

    cdef void _prepare_shear(self):
        """
        Prepare for sampling with shear viscous corrections by precalculating
        the shear momentum scale factor = tau/(2*eta).  Shear corrections are
        then applied by transforming sampled momentum vectors p_i by

            p_i  ->  p_i + shear_pscale * pi_ij p_j

        with both p_i and the shear tensor pi_ij in the local rest frame.

        """
        self.shear_pscale = .5/self.eta_over_tau()

    cdef void _prepare_bulk(self):
        """
        Prepare for sampling with bulk viscous corrections by pre-calculating
        the changes in density and momentum density for each species as well as
        the allowed range for bulk pressure.

        delta_density_Pi is the change in particle density over bulk pressure:

            density  ->  density + delta_density_Pi * Pi

        This is saved for each species and the sum is saved for the HRG.

        rel_p_density_Pi is the relative change in momentum density over bulk
        pressure:

            p_density  ->  p_density * (1 + rel_p_density_Pi * Pi)

        """
        cdef:
            double cs2 = self.cs2()
            double zeta_over_tau = self._sum_integrals(ZETA_OVER_TAU, cs2=cs2)
            double rel_min = 0, rel_max = 0
            size_t i
            SpeciesInfo* s

        self.delta_density_Pi = 0

        for i in range(self.n):
            s = self.data + i

            s.delta_density_Pi = (
                integrate_species(s, self.T, cs2, DELTA_DENSITY) /
                zeta_over_tau
            )

            self.delta_density_Pi += s.delta_density_Pi

            s.rel_p_density_Pi = (
                integrate_species(s, self.T, cs2, DELTA_MOMENTUM_DENSITY) /
                zeta_over_tau /
                integrate_species(s, self.T, cs2, MOMENTUM_DENSITY)
            )

            rel_min = math.fmin(
                rel_min,
                math.fmin(s.delta_density_Pi/s.density, s.rel_p_density_Pi)
            )
            rel_max = math.fmax(
                rel_max,
                math.fmax(s.delta_density_Pi/s.density, s.rel_p_density_Pi)
            )

        # Set Pi min and max so that no species' density or momentum density
        # ever goes negative.  In fact, give them 10% leeway.
        self.Pi_min = -.9/rel_max
        self.Pi_max = -.9/rel_min

        self.bulk_prepared = 1

    def Pi_lim(self):
        """
        Minimum and maximum allowable bulk pressure [GeV/fm^3].

        """
        if not self.bulk_prepared:
            self._prepare_bulk()

        return self.Pi_min, self.Pi_max


# represents a sampled particle
cdef struct Particle:
    int ID
    FourVector x, p

cdef class ParticleArray:
    """
    Dynamic array of sampled particles.

    Non-member function `add_particle` appends particles to the array and calls
    `increase_capacity` if the array is full.  These functions are non-members
    to allow inlining (Cython calls member functions through a virtual table,
    so inlining them is usually impossible).  `add_particle` can easily be
    called millions of times per second.

    Implements the PEP 3118 buffer protocol to expose a contiguous array of
    structs with format: 'ID' (int), 'x' (4 doubles), 'p' (4 doubles).

    Why not use cython.view.array?  It throws errors for zero-sized arrays
    (which certainly can happen), it's unnecessarily complex, and anyway it's
    barely any easier than this.

    """
    cdef:
        Particle* data
        Py_ssize_t n, capacity

    def __cinit__(self, double navg):
        self.n = 0
        # Guess initial capacity based on Poissonian particle production:
        # average number of particles plus three standard deviations.
        self.capacity = <Py_ssize_t> max(navg + 3*math.sqrt(navg), 10)

        self.data = <Particle*> PyMem_Malloc(self.capacity * sizeof(Particle))
        if not self.data:
            raise MemoryError()

    def __dealloc__(self):
        PyMem_Free(self.data)

    def __getbuffer__(self, Py_buffer* view, int flags):
        view.buf = self.data
        view.obj = self
        view.len = self.n * sizeof(Particle)
        view.itemsize = sizeof(Particle)
        view.strides = &view.itemsize
        view.ndim = 1
        view.shape = &self.n
        view.readonly = 0
        view.suboffsets = NULL

        if flags & PyBUF_FORMAT:
            view.format = 'i:ID: 4d:x: 4d:p:'
        else:
            view.format = NULL


cdef inline void increase_capacity(ParticleArray particles) with gil:
    """
    Increase the capacity of a `ParticleArray`.

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
    ParticleArray particles,
    const int ID,
    const FourVector* x,
    const FourVector* p
) nogil:
    """
    Append a new particle to a `ParticleArray`; increase capacity if necessary.

    """
    if particles.n == particles.capacity:
        increase_capacity(particles)

    cdef Particle* part = particles.data + particles.n
    part.ID = ID
    part.x = x[0]
    part.p = p[0]

    particles.n += 1


cdef void decay_f500(ParticleArray particles) nogil:
    """
    Decay all f(0)(500) resonances into pion pairs.

    Workaround for afterburners e.g. UrQMD that aren't aware of the f(0)(500).

    """
    # For each f(0)(500) in the particle array, do the following:
    #   - Replace with a pion and also create a new pion.
    #   - Compute the momenta of the daughter pions in the parent's rest frame
    #     (basic kinematics).
    #   - Assign daughter pions equal and opposite momenta (with a random
    #     isotropic angle) in the parent's rest frame.
    #   - Boost each daughter pion from the parent's rest frame.
    #   - Freestream each pion by a short time so they don't exactly overlap.

    cdef:
        # loop index
        size_t i
        # original number of particles (will increase as pions are produced)
        size_t nparts = particles.n
        # parent and daughter particles mass and momentum
        double M, P, m, p
        # spherical angles
        double cos_theta, sin_theta, phi
        # to save some typing
        Particle* part
        # ID number of second daughter particle
        int ID2
        # four-velocity and position of parent particle
        FourVector u, x
        # four-momentum of daughters in parent's rest frame
        FourVector p_prime

    for i in range(nparts):
        part = particles.data + i

        # skip all but f(0)(500)
        if part.ID != 9000221:
            continue

        # choose decay channel:
        #   pi0 pi0  (1/3)
        #   pi+ pi-  (2/3)
        if rand() < .333333333:
            part.ID = 111
            ID2 = 111
            m = .1349766
        else:
            part.ID = 211
            ID2 = -211
            m = .13957018

        # momentum and mass of parent
        P = math.sqrt(part.p.x**2 + part.p.y**2 + part.p.z**2)
        M = math.sqrt(part.p.t**2 - P*P)

        # four-velocity of parent
        u.t = part.p.t/M
        u.x = part.p.x/M
        u.y = part.p.y/M
        u.z = part.p.z/M

        # save parent particle position
        x = part.x

        # momentum magnitude of daughters in parent's rest frame
        p = math.sqrt(M*M - 4*m*m)/2

        # sample isotropic direction
        cos_theta = 2*rand() - 1
        sin_theta = math.sqrt(1 - cos_theta*cos_theta)
        phi = TWO_PI*rand()

        # first daughter's four-momentum in parent's rest frame
        p_prime.t = M/2
        p_prime.x = p*sin_theta*math.cos(phi)
        p_prime.y = p*sin_theta*math.sin(phi)
        p_prime.z = p*cos_theta

        # boost and freestream first daughter
        part.p = p_prime
        fourvec.boost_inverse(&part.p, &u)
        freestream(part, .1)

        # create second daughter pion at parent's spacetime position
        # and momentum opposite to first daughter in parent's rest frame
        p_prime.x *= -1
        p_prime.y *= -1
        p_prime.z *= -1
        add_particle(particles, ID2, &x, &p_prime)

        # boost and freestream second daughter
        part = particles.data + particles.n - 1
        fourvec.boost_inverse(&part.p, &u)
        freestream(part, .1)


cdef void freestream(Particle* part, double t) nogil:
    """
    Freestream a particle for the given time:

        x -> x + v*t, v = p/E

    """
    cdef double t_over_E = t / part.p.t
    part.x.t += t
    part.x.x += part.p.x * t_over_E
    part.x.y += part.p.y * t_over_E
    part.x.z += part.p.z * t_over_E


cdef void _sample(Surface surface, HRG hrg, ParticleArray particles) nogil:
    """
    Sample an ensemble of particles from freeze-out hypersurface `surface`,
    using thermodynamic quantities (temperature, etc) and species information
    from `hrg`, and write data to `particles`.

    """
    cdef:
        double N, new_N
        double p_dot_sigma, p_dot_sigma_max
        double y, y_minus_eta_s
        double eta_s, cosh_eta_s, sinh_eta_s
        double pt_prime
        double Pi
        double bulk_pscale
        size_t ielem, ispecies
        SurfaceElem* elem
        SpeciesInfo* species
        FourVector x, p

    N = math.log(1 - rand())

    for ielem in range(surface.n):
        elem = surface.data + ielem

        # clip bulk pressure to allowed range
        Pi = math.fmin(math.fmax(elem.Pi, hrg.Pi_min), hrg.Pi_max)

        new_N = N + elem.vmax * (hrg.total_density + hrg.delta_density_Pi * Pi)
        if new_N < 0:
            N = new_N
            continue

        for ispecies in range(hrg.n):
            species = hrg.data + ispecies
            N += elem.vmax * (species.density + species.delta_density_Pi * Pi)

            bulk_pscale = (
                (1 + Pi * species.rel_p_density_Pi) /
                (1 + Pi * species.delta_density_Pi / species.density)
            )

            while N > 0:
                # adding a negative number
                N += math.log(1 - rand())

                sample_four_momentum(
                    species, hrg.T,
                    hrg.shear_pscale, &elem.pi, bulk_pscale,
                    &p
                )
                fourvec.boost_inverse(&p, &elem.u)

                p_dot_sigma = fourvec.dot(&p, &elem.sigma)
                if p_dot_sigma < 0:
                    continue

                p_dot_sigma_max = elem.vmax*fourvec.dot(&p, &elem.u)

                if p_dot_sigma > p_dot_sigma_max*rand():
                    if surface.boost_invariant:
                        y_minus_eta_s = .5*math.log((p.t + p.z)/(p.t - p.z))
                        y = surface.ymax*(2*rand() - 1)
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

                    add_particle(particles, species.ID, &x, &p)


def sample(Surface surface not None, HRG hrg not None):
    """
    Sample an ensemble of particles from freeze-out hypersurface `surface`
    using thermodynamic quantities (temperature, etc) and species information
    from `hrg`.

    Return a numpy structured array with fields

        - 'ID' particle ID number
        - 'x' position four-vector
        - 'p' momentum four-vector

    """
    particles = ParticleArray(abs(surface.total_volume) * hrg.total_density)

    if surface.shear and not hrg.shear_prepared:
        hrg._prepare_shear()

    if surface.bulk and not hrg.bulk_prepared:
        hrg._prepare_bulk()

    with nogil:
        _sample(surface, hrg, particles)
        if hrg.decay_f500:
            decay_f500(particles)

    return np.asarray(particles)
