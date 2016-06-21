#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <sys/syscall.h>
#include <sys/time.h>

/*
 * Read system entropy into `s` and return status.
 * Use getrandom syscall if available, otherwise read /dev/urandom directly.
 */
int get_entropy(long int* s) {
#ifdef SYS_getrandom
  return syscall(SYS_getrandom, s, sizeof *s, 0) == sizeof *s;
#else
  FILE* fd = fopen("/dev/urandom", "r");
  if (!fd)
    return 0;
  size_t ret = fread(s, sizeof *s, 1, fd);
  fclose(fd);
  return ret == 1;
#endif
}

/*
 * Set random seed from system entropy, fall back to system clock.
 */
void seed_rand(void) {
  long int s;

  if (!get_entropy(&s)) {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    s = 1000000*tv.tv_sec + tv.tv_usec;
  }

  srand48(s);
}
