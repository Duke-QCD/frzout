from libc cimport math

from .fourvec cimport FourVector
from .species cimport SpeciesInfo, bw_icdf, bw_accept_prob

# TODO better RNG?
cdef extern from "stdlib.h" nogil:
    void set_seed "srand48"(long int seed)
    double random "drand48"()

cdef extern from "unistd.h" nogil:
    long int syscall(long int sysno, ...)

cdef extern from "sys/syscall.h" nogil:
    long int SYS_getrandom


cdef inline void seed() nogil:
    """
    Seed from the system's entropy source.

    """
    cdef long int seed
    # TODO fallback when getrandom() is not available
    syscall(SYS_getrandom, &seed, sizeof(seed), 0)
    set_seed(seed)


cdef inline void four_momentum(
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
    the envelope dist by another factor (Pxscale) so that the envelope peaks at
    the same point as the target dist.  This is commensurate with using a
    larger effective temperature so that the envelope has a longer tail than
    the target and hence remains above it for all x (provided Pxscale is
    well-chosen).

    Efficiency is ~90% for small masses (z ~ 1) decreasing to ~70% for large
    masses (z ~ 10).

    """
    cdef double z, Pm

    if s.stable:
        z = s.m0/T
        Pm = 1
    else:
        # Use the minimum threshold mass for the momentum envelope.
        z = s.m_min/T

    cdef double xmaxsq, xscale, Pxscale

    # Use rescaling for all but the smallest masses.
    if z > 1.3:
        # Compute argmax of the Boltzmann dist x^2*exp(-sqrt(z^2 + x^2)).
        # No closed form for the argmax of Bose/Fermi dists.
        xmaxsq = 2*(1 + math.sqrt(1 + z*z))
        # The argmax of the envelope x^2*exp(-x) is x = 2.
        # Rescale x so that it peaks at the desired xmax.
        xscale = math.sqrt(xmaxsq)/2
        # Now rescale P so that it equals one (or just below one) at xmax.
        # Since xmax is approximate and the Bose/Fermi dists have slightly
        # different shapes as Boltzmann, the envelope can fall slightly below
        # the target.  The extra factor (1 + .5*exp(-2*z)) ensures the envelope
        # remains above the target for z > 1.26.
        Pxscale = (
            math.exp(-2) * (math.exp(math.sqrt(xmaxsq + z*z)) + s.sign) /
            (1 + .5*math.exp(-2*z))
        )
    elif z < .86 and s.sign == -1:
        # There is one more wrinkle for very light bosons (z < .855): the
        # envelope briefly falls below the target near x ~ 1.  This extra
        # factor extends the range to z ~ .7, which is the z of the lightest
        # species (pi0) at T ~ 192 MeV.  So this works for any reasonable
        # particlization temperature.
        xscale = 1
        Pxscale = 1/(1 + 1.3*(.86 - z))
    else:
        xscale = 1
        Pxscale = 1

    cdef double m, r, x, Px

    # main sampling loop
    while True:
        if not s.stable:
            # sample proposal mass
            m = bw_icdf(s, random())
            Pm = bw_accept_prob(s, m)
            z = m/T

        # sample proposal x from envelope x^2*exp(-x/xscale)
        r = (1 - random())*(1 - random())*(1 - random())
        x = -xscale*math.log(r)

        # acceptance probability
        Px = Pxscale / r / (math.exp(math.sqrt(x*x + z*z)) + s.sign)

        if random() < Px*Pm:
            break

    cdef:
        # change variables, x = p/T
        double pmag = x*T
        # sample 3D direction
        double cos_theta = 2*random() - 1
        double sin_theta = math.sqrt(1 - cos_theta*cos_theta)
        double phi = 6.28318530717958648*random()  # 2*pi

    p.t = T*math.sqrt(x*x + z*z)
    p.x = pmag*sin_theta*math.cos(phi)
    p.y = pmag*sin_theta*math.sin(phi)
    p.z = pmag*cos_theta
