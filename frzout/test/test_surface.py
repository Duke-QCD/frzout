# -*- coding: utf-8 -*-

import numpy as np

from nose.tools import (
    assert_almost_equal,
    assert_warns_regex,
    assert_raises_regex,
)

from .. import Surface


def test_surface():
    volume = np.random.uniform(10, 100)
    tau = np.random.uniform(.5, 5.)

    x = [tau, 0, 0]
    sigma = [volume/tau, 0, 0]
    v = [0, 0]

    surf = Surface(x, sigma, v)

    assert surf.boost_invariant, 'Surface should be boost-invariant.'

    assert_almost_equal(
        surf.volume, volume, delta=1e-12,
        msg='incorrect volume'
    )

    ymax = np.random.uniform(.5, 2.)
    surf = Surface(x, sigma, v, ymax=ymax)

    assert_almost_equal(
        surf.volume, 2*ymax*volume, delta=1e-12,
        msg='incorrect volume'
    )

    x = np.random.uniform(0, 10, size=(1, 4))
    # ensure sigma is timelike (sigma^2 > 0) so that the volume is positive
    sigma = np.random.uniform([3, -1, -1, -1], [4, 1, 1, 1], size=(1, 4))
    v = np.random.uniform(-.5, .5, size=(1, 3))

    surf = Surface(x, sigma, v)

    assert not surf.boost_invariant, 'Surface should not be boost-invariant.'

    u = np.insert(v, 0, 1) / np.sqrt(1 - (v*v).sum())
    volume = np.inner(sigma, u)

    assert_almost_equal(
        surf.volume, volume, delta=1e-12,
        msg='incorrect volume'
    )

    with assert_warns_regex(Warning, 'ymax has no effect for a 3D surface'):
        Surface(x, sigma, v, ymax=np.random.rand())

    with assert_warns_regex(Warning, 'total freeze-out volume is negative'):
        Surface(x, np.concatenate([[[0]], -v], axis=1), v)

    with assert_raises_regex(
            ValueError,
            'number of spacetime dimensions of x, sigma, and/or v'
    ):
        Surface([1, 0], [1, 0], 0)

    with assert_raises_regex(
            ValueError,
            'number of spacetime dimensions of x, sigma, and/or v'
    ):
        Surface(
            np.ones((1, 4)),
            np.ones((1, 3)),
            np.ones((1, 2)),
        )

    with assert_raises_regex(
            ValueError,
            'number of elements of x, sigma, and/or v do not match'
    ):
        Surface(
            np.ones((2, 4)),
            np.ones((2, 4)),
            np.ones((3, 3)),
        )

    with assert_raises_regex(
            ValueError,
            'number of elements of pi components do not match'
    ):
        Surface(
            np.ones((3, 4)),
            np.ones((3, 4)),
            np.ones((3, 3)),
            pi=dict(
                xx=np.ones(3),
                yy=np.ones(3),
                xy=np.ones(3),
                xz=np.ones(3),
                yz=np.ones(4),
            )
        )

    with assert_raises_regex(
            ValueError,
            'number of elements of Pi do not match'
    ):
        Surface(
            np.ones((3, 4)),
            np.ones((3, 4)),
            np.ones((3, 3)),
            Pi=np.ones(4)
        )
