from libc cimport math

from .. cimport fourvec
from ..fourvec cimport FourVector
from .. cimport random


def _test_fourvec():
    cdef random.RNG rng
    random.init(&rng)

    cdef:
        double vmag = random.rand(&rng)
        double gamma = 1/math.sqrt(1 - vmag*vmag)
        double vx, vy, vz

    random.direction(&rng, &vx, &vy, &vz)
    vx *= vmag
    vy *= vmag
    vz *= vmag

    cdef FourVector u
    u.t = gamma
    u.x = gamma*vx
    u.y = gamma*vy
    u.z = gamma*vz

    cdef double uu = fourvec.square(&u)

    assert math.fabs(uu - 1) < 1e-14, \
        'Four-velocity is not normalized: {} != 1'.format(uu)

    cdef double m = random.rand(&rng) + .1

    cdef FourVector p
    p.x = 2*random.rand(&rng) - 1
    p.y = 2*random.rand(&rng) - 1
    p.z = 2*random.rand(&rng) - 1
    p.t = math.sqrt(m*m + p.x*p.x + p.y*p.y + p.z*p.z)

    cdef double pp = fourvec.square(&p)
    assert math.fabs(pp/(m*m) - 1) < 1e-13, \
        'Four-momentum is not normalized: {} != {}'.format(pp, m*m)

    fourvec.boost_inverse(&p, &u)
    pp = fourvec.square(&p)

    assert math.fabs(pp/(m*m) - 1) < 1e-12, \
        'Boosted four-momentum is not normalized: {} != {}'.format(pp, m*m)

    cdef FourVector a
    a.t = random.rand(&rng)
    a.x = random.rand(&rng)
    a.y = random.rand(&rng)
    a.z = random.rand(&rng)

    cdef:
        double vsq = vx*vx + vy*vy + vz*vz
        double L[4][4]

    L[0][0] = gamma
    L[1][1] = 1 + (gamma - 1)*vx*vx/vsq
    L[2][2] = 1 + (gamma - 1)*vy*vy/vsq
    L[3][3] = 1 + (gamma - 1)*vz*vz/vsq
    L[0][1] = gamma*vx
    L[0][2] = gamma*vy
    L[0][3] = gamma*vz
    L[1][2] = (gamma - 1)*vx*vy/vsq
    L[1][3] = (gamma - 1)*vx*vz/vsq
    L[2][3] = (gamma - 1)*vy*vz/vsq
    L[1][0] = L[0][1]
    L[2][0] = L[0][2]
    L[3][0] = L[0][3]
    L[2][1] = L[1][2]
    L[3][1] = L[1][3]
    L[3][2] = L[2][3]

    cdef:
        double* aptr = &a.t
        double a_boosted[4]
        int i, j

    for i in range(4):
        a_boosted[i] = 0.
        for j in range(4):
            a_boosted[i] += aptr[j]*L[i][j]

    fourvec.boost_inverse(&a, &u)
    assert all([
        math.fabs(a.t/a_boosted[0] - 1) < 1e-14,
        math.fabs(a.x/a_boosted[1] - 1) < 1e-14,
        math.fabs(a.y/a_boosted[2] - 1) < 1e-14,
        math.fabs(a.z/a_boosted[3] - 1) < 1e-14,
    ]), 'Boost does not agree with full boost matrix.'

    cdef FourVector neg_u = u
    neg_u.x *= -1
    neg_u.y *= -1
    neg_u.z *= -1

    fourvec.boost_inverse(&neg_u, &u)
    assert all([
        math.fabs(neg_u.t - 1) < 1e-14,
        math.fabs(neg_u.x) < 1e-14,
        math.fabs(neg_u.y) < 1e-14,
        math.fabs(neg_u.z) < 1e-14,
    ]), 'Boosted four-velocity is not zero: {}'.format(u)

    cdef:
        double vx1 = .99*(2*random.rand(&rng) - 1)
        double vx2 = .99*(2*random.rand(&rng) - 1)
        double gamma1 = 1/math.sqrt(1 - vx1*vx1)
        double gamma2 = 1/math.sqrt(1 - vx2*vx2)

    cdef FourVector u1, u2
    u1.t = gamma1
    u1.x = gamma1*vx1
    u1.y = 0
    u1.z = 0
    u2.t = gamma2
    u2.x = gamma2*vx2
    u2.y = 0
    u2.z = 0

    fourvec.boost_inverse(&u1, &u2)
    cdef:
        double vnew = u1.x/u1.t
        double vcorrect = (vx1 + vx2)/(1 + vx1*vx2)

    assert math.fabs(vnew/vcorrect - 1) < 1e-13, (
        'Boosted four-velocity does not agree with 1D velocity addition: '
        '{} != {}'.format(vnew, vcorrect)
    )
