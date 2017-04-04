#ifndef RANDOM_H
#define RANDOM_H

#include <inttypes.h>
#include <math.h>
#include <time.h>
#include <unistd.h>
#include <sys/syscall.h>

/*
 * PCG random number generator
 * Copyright 2014 Melissa O'Neill
 * pcg-random.org
 * Apache License 2.0
 *
 * Slightly modified from the minimal C implementation.
 */

// Internals are *private*.
struct pcg_state_setseq_64 {
  // RNG state.  All values are possible.
  uint64_t state;
  // Controls which RNG sequence (stream) is selected.  Must *always* be odd.
  uint64_t inc;
};

typedef struct pcg_state_setseq_64 pcg32_random_t;

// Generate a uniformly distributed 32-bit random number
static inline uint32_t pcg32_random_r(pcg32_random_t* rng) {
  uint64_t oldstate = rng->state;
  rng->state = oldstate * 6364136223846793005ULL + rng->inc;
  uint32_t xorshifted = ((oldstate >> 18u) ^ oldstate) >> 27u;
  uint32_t rot = oldstate >> 59u;
  return (xorshifted >> rot) | (xorshifted << ((-rot) & 31));
}

// Seed the rng.  Specified in two parts, state initializer and a sequence
// selection constant (a.k.a. stream id).
static inline void pcg32_srandom_r(
    pcg32_random_t* rng, uint64_t initstate, uint64_t initseq
) {
  rng->state = 0U;
  rng->inc = (initseq << 1u) | 1u;
  pcg32_random_r(rng);
  rng->state += initstate;
  pcg32_random_r(rng);
}

/* End PCG random code. */

// Helper function: read system entropy into `s` and return status.
// Use getrandom syscall if available, otherwise read /dev/urandom directly.
static inline int get_entropy(uint64_t* s, size_t n) {
#ifdef SYS_getrandom
  long nbytes = n * sizeof *s;
  return syscall(SYS_getrandom, s, nbytes, 0) == nbytes;
#else
  FILE* fd = fopen("/dev/urandom", "r");
  if (!fd)
    return 0;
  size_t count = fread(s, sizeof *s, n, fd);
  fclose(fd);
  return count == n;
#endif
}

// This typedef and the following functions are the public interface.
typedef pcg32_random_t RNG;

// Initialize the rng from system entropy.
static inline void random_init(RNG* rng) {
  uint64_t s[2];

  if (!get_entropy(s, sizeof s / sizeof *s)) {
    // Fallback method: use system clock and memory addresses as (inferior)
    // sources of entropy.  The rng stream is set from its memory address; this
    // ensures a different stream for each rng, even if they are initialized by
    // the same program at the same time.
    s[0] = time(NULL) ^ (intptr_t)&get_entropy;
    s[1] = (intptr_t)rng;
  }

  pcg32_srandom_r(rng, s[0], s[1]);
}

/*
 * Generate a random float in [0, 1) with 53-bit resolution.
 *
 * This method is originally from the Mersenne Twister library -- function
 * genrand_res53 by Isaku Wadu, 2002/01/09.  It is used in both the Python
 * standard library and numpy.random.
 *
 * `a` provides 27 random bits; `b` provides 26.  `a` is then effectively
 * shifted left by 26 bits (67108864 == 2^26) and added to `b` to produce a
 * 53-bit numerator, which is then divided by 2^53 == 9007199254740992.
 */
static inline double random_rand(RNG* rng) {
  uint32_t a = pcg32_random_r(rng) >> 5, b = pcg32_random_r(rng) >> 6;
  return (a * 67108864. + b) / 9007199254740992.;
}

// float in (0, 1], i.e. the "complement" of rand()
static inline double random_rand_c(RNG* rng) {
  return 1 - random_rand(rng);
}

// isotropic 3D direction
static inline void random_direction(RNG* rng, double* x, double* y, double* z) {
  // unit sphere:
  //   z = cos(theta) in [-1, 1)
  //   r_perp = sin(theta)
  //   r_perp^2 + z^2 = sin^2 + cos^2 = 1
  *z = 2*random_rand(rng) - 1;
  double r_perp = sqrt(1 - (*z)*(*z));

  // azimuthal angle in [0, 2*pi)
  double phi = 6.28318530717958647693*random_rand(rng);

  *x = r_perp * cos(phi);
  *y = r_perp * sin(phi);
}

#endif  // RANDOM_H
