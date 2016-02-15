# cython: cdivision = True

import random

from libc cimport math

from .. cimport fourvec


def _test_fourvec():
    cdef:
        double vmax = .99/math.sqrt(3.)
        double vx = random.uniform(-vmax, vmax)
        double vy = random.uniform(-vmax, vmax)
        double vz = random.uniform(-vmax, vmax)
        double gamma = 1./math.sqrt(1. - vx*vx - vy*vy - vz*vz)

    cdef fourvec.FourVector u
    u.t = gamma
    u.x = gamma*vx
    u.y = gamma*vy
    u.z = gamma*vz

    cdef double uu = fourvec.dot(&u, &u)

    assert math.fabs(uu - 1.) < 1e-15, \
        'Four-velocity is not normalized: {} != 1'.format(uu)

    cdef double m = random.uniform(.1, 1)

    cdef fourvec.FourVector p
    p.x = random.uniform(-1, 1)
    p.y = random.uniform(-1, 1)
    p.z = random.uniform(-1, 1)
    p.t = math.sqrt(m*m + p.x*p.x + p.y*p.y + p.z*p.z)

    cdef double pp = fourvec.dot(&p, &p)
    assert math.fabs(pp/(m*m) - 1.) < 1e-13, \
        'Four-momentum is not normalized: {} != {}'.format(pp, m*m)

    fourvec.boost_inverse(&p, &u)
    pp = fourvec.dot(&p, &p)

    assert math.fabs(pp/(m*m) - 1.) < 1e-13, \
        'Boosted four-momentum is not normalized: {} != {}'.format(pp, m*m)

    cdef fourvec.FourVector neg_u = u
    neg_u.x *= -1
    neg_u.y *= -1
    neg_u.z *= -1

    fourvec.boost_inverse(&neg_u, &u)
    assert all([
        math.fabs(neg_u.t - 1.) < 1e-15,
        math.fabs(neg_u.x) < 1e-15,
        math.fabs(neg_u.y) < 1e-15,
        math.fabs(neg_u.z) < 1e-15,
    ]), 'Boosted four-velocity is not zero: {}'.format(u)

    cdef:
        double vx1 = random.uniform(-.99, .99)
        double vx2 = random.uniform(-.99, .99)
        double gamma1 = 1./math.sqrt(1. - vx1*vx1)
        double gamma2 = 1./math.sqrt(1. - vx2*vx2)

    cdef fourvec.FourVector u1, u2
    u1.t = gamma1
    u1.x = gamma1*vx1
    u1.y = 0.
    u1.z = 0.
    u2.t = gamma2
    u2.x = gamma2*vx2
    u2.y = 0.
    u2.z = 0.

    fourvec.boost_inverse(&u1, &u2)
    cdef:
        double vnew = u1.x/u1.t
        double vcorrect = (vx1 + vx2)/(1. + vx1*vx2)

    assert math.fabs(vnew/vcorrect - 1) < 1e-14, (
        'Boosted four-velocity does not agree with 1D velocity addition: '
        '{} != {}'.format(vnew, vcorrect)
    )
