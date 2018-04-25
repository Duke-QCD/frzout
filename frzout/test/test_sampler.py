# -*- coding: utf-8 -*-

import numpy as np

from .. import Surface, HRG, sample


def test_dtype():
    surface = Surface([1, 0, 0], [1, 0, 0], [0, 0])
    hrg = HRG(.15)

    parts = sample(surface, hrg)

    dt = parts.dtype

    assert dt.names == ('ID', 'x', 'p'), 'Incorrect dtype fields.'

    assert dt['ID'].shape == (), 'Incorrect ID field shape.'
    assert dt['x'].shape == (4,), 'Incorrect x field shape.'
    assert dt['p'].shape == (4,), 'Incorrect x field shape.'


def test_ymax():
    hrg = HRG(.15)

    ymax = np.random.uniform(.1, 1.)

    surface = Surface(
        np.array([[1., 0., 0.]]),
        np.array([[1e3/hrg.density(), 0., 0.]]),
        np.array([[0., 0.]]),
        ymax=ymax
    )

    parts = sample(surface, hrg)
    p = parts['p']

    E, px, py, pz = p.T
    y = .5*np.log((E + pz)/(E - pz))

    assert np.all(np.fabs(y) < ymax), 'Rapidity outside ymax.'


def test_decay_f500():
    hrg = HRG(.15, species=[9000221], decay_f500=True)

    surface = Surface(
        np.array([[1., 0., 0.]]),
        np.array([[1e2/hrg.density(), 0., 0.]]),
        np.array([[0., 0.]]),
    )

    parts = sample(surface, hrg)
    i = parts['ID']

    assert np.all((i == 111) | (i == 211) | (i == -211)), \
        'f500 not decayed to pions.'
