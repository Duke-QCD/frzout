# cython: cdivision = True

from libc cimport math

from fourvec cimport FourVector

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


cdef inline void boltzmann(double m, double T, FourVector* p) nogil:
    """
    Sample a momentum from the Boltzmann distribution:

        p ~ p^2 exp(-sqrt(m^2 + p^2)/T)

    """
    cdef double pmag

    # TODO alternate algorithm for m >> T
    while True:
        pmag = -T*math.log(random()*random()*random())
        p.t = math.sqrt(m*m + pmag*pmag)
        if random() > math.exp((pmag - p.t)/T):
            break

    # TODO reuse random numbers?
    cdef:
        double cos_theta = 2*random() - 1
        double sin_theta = math.sqrt(1 - cos_theta*cos_theta)
        double phi = 2*math.M_PI*random()

    p.x = pmag*sin_theta*math.cos(phi)
    p.y = pmag*sin_theta*math.sin(phi)
    p.z = pmag*cos_theta
