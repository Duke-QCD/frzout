# cython: cdivision = True
# cython: boundscheck = False, wraparound = False, initializedcheck = False

import numpy as np
from scipy import special

from .species import species_dict

cimport numpy as np
from libc cimport math
from cpython.mem cimport PyMem_Malloc, PyMem_Realloc, PyMem_Free

cimport fourvec
from fourvec cimport FourVector
cimport random

__all__ = ['Sampler']

DEF hbarc = 0.1973269788  # GeV fm

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
    int ID
    double mass, width, density

cdef class SpeciesList:
    """
    Represents a set of particle species, i.e. a hadron gas composition.
    Manages memory for an array of `SpeciesInfo`.

    """
    cdef:
        SpeciesInfo* data
        size_t n
        double total_density

    def __cinit__(self, double T):
        def generate_IDs():
            for ID, info in species_dict.items():
                if info['mass'] < 1:
                    yield ID
                    if info['has_anti']:
                        yield -ID

        IDs = sorted(
            generate_IDs(),
            key=lambda ID: species_dict[abs(ID)]['mass']
        )

        self.n = len(IDs)

        self.data = <SpeciesInfo*> PyMem_Malloc(self.n * sizeof(SpeciesInfo))
        if not self.data:
            raise MemoryError()

        cdef:
            double prefactor = 1/(2*math.M_PI**2*hbarc**3)
            double density = 0, m, last_m = -100
            int g, ID
            size_t i
            SpeciesInfo* species

        self.total_density = 0

        for i, ID in enumerate(IDs):
            info = species_dict[abs(ID)]
            m = info['mass']
            g = info['degen']
            # reuse last calculation if masses are equal
            if abs(m - last_m) > 1e-6:
                density = prefactor * g * m*m*T * special.kn(2, m/T)
            last_m = m

            species = self.data + i
            species.ID      = ID
            species.mass    = m
            species.width   = info['width']
            species.density = density

            self.total_density += density

    def __dealloc__(self):
        if self.data:
            PyMem_Free(self.data)


cdef struct Particle:
    int ID
    FourVector x, p

cdef class ParticleList:
    """
    Represents an ensemble of produced particles.
    Manages memory for an array of `Particle`.

    Unlike `Surface` and `SpeciesList`, a `ParticleList` may change size as
    more particles are produced.  The following functions
    `extend_particle_list` and `add_particle` are non-members to allow inlining
    (Cython calls member functions through a virtual table, so inlining them is
    usually impossible).

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


cdef inline void extend_particle_list(ParticleList particles):
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
        with gil:
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
        SpeciesList species
        ParticleList particles
        double T

    def __cinit__(self, x, sigma, v, T):
        self.surface = Surface(np.asarray(x), np.asarray(sigma), np.asarray(v))
        self.species = SpeciesList(T)
        self.particles = ParticleList(
            self.surface.total_volume * self.species.total_density
        )
        self.T = T

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

        N = math.log(random.random())

        for ielem in range(self.surface.n):
            elem = self.surface.data + ielem

            new_N = N + elem.vmax*self.species.total_density
            if new_N < 0:
                N = new_N
                continue

            for ispecies in range(self.species.n):
                species = self.species.data + ispecies
                N += elem.vmax*species.density
                while N > 0:
                    # adding a negative number
                    N += math.log(random.random())

                    random.boltzmann(species.mass, self.T, &p)
                    fourvec.boost_inverse(&p, &elem.u)

                    p_dot_sigma = fourvec.dot(&p, &elem.sigma)
                    if p_dot_sigma < 0:
                        continue

                    p_dot_sigma_max = elem.vmax*fourvec.dot(&p, &elem.u)

                    if p_dot_sigma > p_dot_sigma_max*random.random():
                        if self.surface.boost_invariant:
                            y_minus_eta_s = .5*math.log((p.t + p.z)/(p.t - p.z))
                            y = random.random() - .5
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
