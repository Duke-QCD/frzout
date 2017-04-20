# -*- coding: utf-8 -*-

import random

import numpy as np
from scipy import integrate
from scipy import special

from nose.tools import assert_almost_equal, assert_warns_regex

from .. import HRG, species_dict


hbarc = 0.1973269788


def test_hrg():
    ID, info = random.choice(list(species_dict.items()))
    m = info['mass']
    g = info['degen']
    sign = -1 if info['boson'] else 1

    prefactor = g / (2*np.pi**2*hbarc**3)
    if info['has_anti']:
        prefactor *= 2

    T = random.uniform(.13, .15)
    hrg = HRG(T, species=[ID], res_width=False)

    assert_almost_equal(
        hrg.T, T, delta=1e-15,
        msg='incorrect temperature'
    )

    n = np.arange(1, 50)
    density = prefactor * m*m*T * (
        (-sign)**(n-1)/n * special.kn(2, n*m/T)
    ).sum()

    assert_almost_equal(
        hrg.density(), density, delta=1e-12,
        msg='incorrect density'
    )

    def integrand(p):
        E = np.sqrt(m*m + p*p)
        return p*p * E / (np.exp(E/T) + sign)

    energy_density = prefactor * integrate.quad(integrand, 0, 10)[0]

    assert_almost_equal(
        hrg.energy_density(), energy_density, delta=1e-12,
        msg='incorrect energy density'
    )

    def integrand(p):
        E = np.sqrt(m*m + p*p)
        return p**4 / (3*E) / (np.exp(E/T) + sign)

    pressure = prefactor * integrate.quad(integrand, 0, 10)[0]

    assert_almost_equal(
        hrg.pressure(), pressure, delta=1e-12,
        msg='incorrect pressure'
    )

    with assert_warns_regex(Warning, 'high particlization temperature'):
        HRG(.193)


def test_bulk_corrections():
    hrg = HRG(.15, res_width=False)
    p0 = hrg.pressure()

    Pi_min, Pi_max = hrg.Pi_lim()

    assert Pi_min == -p0, \
        'The minimum bulk pressure should equal negative the ideal pressure.'

    assert 0 < Pi_max <= p0, \
        'The maximum bulk pressure should be <= the ideal pressure.'

    assert hrg.bulk_scale_factors(0) == (1, 1), \
        'The scale factors should equal one at zero bulk pressure.'

    nscale, pscale = hrg.bulk_scale_factors(random.uniform(.1, .9)*Pi_max)
    assert nscale < 1 and pscale > 1, \
        'Incorrect scale factors for positive bulk pressure.'

    nscale, pscale = hrg.bulk_scale_factors(random.uniform(.1, .9)*Pi_min)
    assert nscale > 1 and pscale < 1, \
        'Incorrect scale factors for negative bulk pressure.'

    nscale, pscale = hrg.bulk_scale_factors(-p0)
    assert nscale > 1 and pscale == 0, \
        'Incorrect scale factors for zero total pressure.'

    assert hrg.bulk_scale_factors(-1.1*p0) == (nscale, pscale), \
        'Scale factors should not change outside the Pi range.'

    assert \
        hrg.bulk_scale_factors(Pi_max) == hrg.bulk_scale_factors(1.5*Pi_max), \
        'Scale factors should not change outside the Pi range.'
