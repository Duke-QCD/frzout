cdef struct FourVector:
    double t, x, y, z


cdef inline double dot(const FourVector* a, const FourVector* b) nogil:
    """
    Compute the dot product of two four vectors in Minkowski space.

    """
    return a.t*b.t - a.x*b.x - a.y*b.y - a.z*b.z


cdef inline double square(const FourVector* a) nogil:
    """
    Compute the square (dot product with itself) of a four vector.

    """
    return dot(a, a)


cdef inline void boost_inverse(FourVector* a, const FourVector* u) nogil:
    """
    Boost four-vector `a` from the frame specified by four-velocity `u` to the
    local rest frame.

    This method was derived by expanding out a full Lorentz transformation,
    collecting common terms, and invoking the normalization

        ut^2 - ux^2 - uy^2 - uz^2 = 1

    """
    # dot product of the space components
    cdef double a3u3 = a.x*u.x + a.y*u.y + a.z*u.z

    # scale factor for boosting space components
    cdef double w = a.t + a3u3/(1 + u.t)

    # boost time component
    a.t = a.t*u.t + a3u3

    # boost space components
    a.x += u.x*w
    a.y += u.y*w
    a.z += u.z*w
