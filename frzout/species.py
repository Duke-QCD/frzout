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
