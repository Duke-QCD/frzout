# TODO better RNG?
cdef extern from "stdlib.h" nogil:
    void set_seed "srand48"(long int seed)
    double rand "drand48"()

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
