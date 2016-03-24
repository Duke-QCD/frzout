# -*- coding: utf-8 -*-

import pkgutil

__all__ = ['species_dict']


def _nth_digit(i, n):
    """
    Determine the nth digit of an integer, starting at the right and counting
    from zero.

    >>> [_nth_digit(123, n) for n in range(5)]
    [3, 2, 1, 0, 0]

    """
    return (i // 10**n) % 10


def _mass_range(name, mass, width, degen, boson):
    """
    Determine the mass thresholds of a resonance for truncating its
    Breit-Wigner distribution.

    """
    # Masses of decay products, rounded up so that the minimum threshold is
    # slightly larger than the actual total mass of the decay products.
    PION = .14
    KAON = .50
    ETA = .55
    RHO = .78
    OMEGA = .79
    KSTAR = .90
    NUCLEON = .94
    LAMBDA = 1.12
    SIGMA = 1.20
    XI = 1.33

    if boson:  # meson
        m_min = {
            'f(0)(500)': 2*PION,
            'rho(770)': 2*PION,
            'omega(782)': 2*PION,
            "eta'(958)": 2*PION + ETA,
            'f(0)(980)': 2*PION,
            'a(0)(980)': ETA + PION,
            'phi(1020)': 2*KAON,
            'h(1)(1170)': RHO + PION,
            'b(1)(1235)': OMEGA + PION,  # 4*PION
            'a(1)(1260)': RHO + PION,
            'f(2)(1270)': 2*PION,
            'f(1)(1285)': 4*PION,
            'eta(1295)': ETA + 2*PION,
            'pi(1300)': RHO + PION,  # 3*PION
            'a(2)(1320)': 3*PION,
            'f(0)(1370)': 2*PION,
            'pi(1)(1400)': ETA + PION,
            'eta(1405)': 4*PION,  # ETA + 2*PION
            'f(1)(1420)': 2*KAON + PION,
            'omega(1420)': RHO + PION,
            'a(0)(1450)': ETA + PION,
            'rho(1450)': 2*PION,
            'eta(1475)': 2*KAON + PION,
            'f(0)(1500)': 2*PION,
            "f(2)'(1525)": 2*KAON,  # 2*PION
            'pi(1)(1600)': .958 + PION,  # 3*PION
            'eta(2)(1645)': ETA + 2*PION,
            'omega(1650)': RHO + PION,
            'omega(3)(1670)': RHO + PION,
            'pi(2)(1670)': 3*PION,
            'phi(1680)': 2*KAON,
            'rho(3)(1690)': 2*PION,
            'rho(1700)': RHO + 2*PION,  # 2*PION
            'f(0)(1710)': 2*PION,
            'pi(1800)': 3*PION,
            'phi(3)(1850)': 2*KAON,
            'f(2)(1950)': 4*PION,  # 2*PION
            'f(2)(2010)': 2*KAON,
            'a(4)(2040)': 3*PION,
            'f(4)(2050)': 2*PION,
            'f(2)(2300)': 2*KAON,
            'f(2)(2340)': 2*ETA,
            'K*(892)': KAON + PION,
            'K(1)(1270)': KSTAR + PION,
            'K(1)(1400)': KSTAR + PION,
            'K*(1410)': KAON + PION,
            'K(0)*(1430)': KAON + PION,
            'K(2)*(1430)': KAON + PION,
            'K*(1680)': KAON + PION,
            'K(2)(1770)': KAON + 2*PION,  # KSTAR + PION
            'K(3)*(1780)': KAON + PION,
            'K(2)(1820)': KSTAR + PION,
            'K(4)*(2045)': KAON + PION,
        }.get(name)

    else:  # baryon
        m_min = {
            'N': NUCLEON + PION,
            'Delta': NUCLEON + PION,
            'Lambda': SIGMA + PION,
            'Sigma': LAMBDA + PION,
            'Xi': XI + PION,
            'Omega': XI + KAON + PION,
        }.get(name.split('(')[0])

    assert m_min is not None, \
        'unknown mass threshold for {}'.format(name)

    assert mass > m_min, \
        'minimum mass larger than pole mass for {}'.format(name)

    # Truncate the distribution at several widths.  The Breit-Wigner shape has
    # a long, slowly-decaying tail so we must cut it somewhere.
    m_min = max(m_min, mass - 4*width)
    m_max = mass + 4*width

    return m_min, m_max


def _read_particle_data():
    """
    Parse particle data from the PDG table.

    Yields pairs (ID, data) where ID is the Monte Carlo ID number and data is a
    dict of properties.

    """
    charge_codes = {
        b'-' : -1,
        b'0' :  0,
        b'+' :  1,
        b'++':  2,
    }

    for l in pkgutil.get_data('frzout', 'mass_width_2015.mcd').splitlines():
        # skip comments
        if l.startswith(b'*'):
            continue

        # extract particle IDs (possibly multiple on a line)
        IDs = [int(i) for i in l[:32].split()]

        # skip elementary particles
        if IDs[0] < 100:
            continue

        # digits 1-3 of the ID denote the quark content
        q1, q2, q3 = quarks = [_nth_digit(IDs[0], n) for n in [1, 2, 3]]

        # skip charm and bottom particles
        if any(q > 3 for q in quarks):
            continue

        # the last digit (ones place) of the ID is the degeneracy
        degen = _nth_digit(IDs[0], 0)

        # skip K0S (ID = 310) and K0L (130)
        if degen == 0:
            continue

        # extract rest of data
        mass = float(l[33:50])
        try:
            width = float(l[70:87])
        except ValueError:
            width = 0.
        name, charges = l[107:].split()
        name = name.strip().decode()
        charges = [charge_codes[i] for i in charges.split(b',')]

        assert len(IDs) == len(charges)

        base_data = dict(
            name=name,
            mass=mass,
            width=width,
            degen=degen,
            boson=bool(degen % 2),
        )

        # determine mass thresholds for resonances
        if width > 1e-3:
            base_data.update(mass_range=_mass_range(**base_data))

        # yield an entry for each (ID, charge) pair
        for ID, charge in zip(IDs, charges):
            data = base_data.copy()
            # The PDG does not explicitly list antiparticles.  A particle has a
            # corresponding antiparticle if any of the following are true:
            #   - it is a baryon (i.e. has three quarks)
            #   - it is charged
            #   - it is a neutral meson with two different quarks
            data.update(
                charge=charge,
                has_anti=(q3 != 0 or charge != 0 or q1 != q2),
            )
            yield ID, data


species_dict = dict(_read_particle_data())
