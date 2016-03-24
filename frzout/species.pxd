# This file contains the definition of the SpeciesInfo object and small inline
# functions related to the Breit-Wigner distribution which are used for both
# hadron gas calculations and random sampling.

from libc cimport math


# These objects are managed and initialized by the HRG class.
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
