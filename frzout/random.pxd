# TODO better RNG?
cdef extern from "random.h" nogil:
    void seed_rand()
    double rand "drand48"()
