cdef extern from "random.h" nogil:
    ctypedef struct RNG:
        pass

    void init "random_init"(RNG*)
    double rand "random_rand"(RNG*)
    double rand_c "random_rand_c"(RNG*)
    double direction "random_direction"(RNG*, double* x, double* y, double* z)
